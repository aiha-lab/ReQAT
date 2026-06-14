import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .auto_scale import auto_scale_pre_rope, apply_pre_rope_scale

__all__ = ["run_pre_rope_scaling"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        layers = model.model.layers
    elif isinstance(model, Qwen2ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, MistralForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, Qwen2ForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_pre_rope_scaling(
    model, enc,
    x_bit, w_bit, q_config,
    n_samples=512, seqlen=512,
    calib_data="pileval", model_path=None,
):
    """
    Run Pre-RoPE scaling calibration to minimize quantization error
    after RoPE for key_states and value_states.
    """
    from .calib_data import get_calib_dataset
    from ..utils.data_utils import get_loaders
    from .module import append_str_prefix, get_op_name
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "bigcode" in str(model.__class__).lower():
        model.transformer.bias = model.transformer.bias.to("cuda")
    layers = get_blocks(model)

    if calib_data == "pileval":
        samples = get_calib_dataset(
            data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
        samples = torch.cat(samples, dim=0)
    else:
        trainloader = get_loaders(calib_data, nsamples=n_samples, seed=0, seqlen=seqlen, model=model_path, eval_mode=False)
        samples = torch.cat([data[0] for data in trainloader], dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cuda()
    
    # Get input and kwargs to layer 0
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
        
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    # Patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Position embeddings setup
    seq_len = inps.shape[1]
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)
    
    # Don't compute position_embeddings upfront - compute per layer with correct shape
    if 'position_ids' not in layer_kwargs or layer_kwargs['position_ids'] is None:
        layer_kwargs['position_ids'] = position_ids

    pre_rope_results = {
        "scale": [],
    }

    historys = []

    # Solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running Pre-RoPE Scaling..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # Capture q_proj, k_proj, v_proj outputs
        def cache_output_hook(m, x, y, name, feat_dict):
            y = y.detach().cpu()
            feat_dict[name].append(y)

        output_feat = defaultdict(list)
        handles = []
        
        # Get attention layer info
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            q_proj = attn.q_proj
            k_proj = attn.k_proj
            v_proj = attn.v_proj
            
            # Get num_heads and head_dim from config or weight shapes
            if hasattr(attn, 'num_heads'):
                num_heads = attn.num_heads
                num_kv_heads = getattr(attn, 'num_key_value_heads', num_heads)
                head_dim = getattr(attn, 'head_dim', None)
                if head_dim is None:
                    head_dim = q_proj.weight.shape[0] // num_heads
            elif hasattr(model.config, 'num_attention_heads'):
                num_heads = model.config.num_attention_heads
                num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
                head_dim = getattr(model.config, 'head_dim', None)
                if head_dim is None:
                    head_dim = q_proj.weight.shape[0] // num_heads
            else:
                # Infer from weight shapes (assume standard head_dim like 128)
                head_dim = 128  # Common default
                num_heads = q_proj.weight.shape[0] // head_dim
                num_kv_heads = k_proj.weight.shape[0] // head_dim
            
            # Get actual head_dim from rotary_emb if available (most accurate)
            if hasattr(attn, 'rotary_emb') and hasattr(attn.rotary_emb, 'inv_freq'):
                # rotary_emb's head_dim is determined by inv_freq shape
                # inv_freq shape is [head_dim // 2]
                rotary_head_dim = attn.rotary_emb.inv_freq.shape[0] * 2
                if rotary_head_dim > 0:
                    head_dim = rotary_head_dim
            elif hasattr(model.model, 'rotary_emb') and hasattr(model.model.rotary_emb, 'inv_freq'):
                rotary_head_dim = model.model.rotary_emb.inv_freq.shape[0] * 2
                if rotary_head_dim > 0:
                    head_dim = rotary_head_dim
            
            # Register hooks for q, k, v projections
            handles.append(q_proj.register_forward_hook(
                functools.partial(cache_output_hook, name='q_proj', feat_dict=output_feat)))
            handles.append(k_proj.register_forward_hook(
                functools.partial(cache_output_hook, name='k_proj', feat_dict=output_feat)))
            handles.append(v_proj.register_forward_hook(
                functools.partial(cache_output_hook, name='v_proj', feat_dict=output_feat)))
        else:
            # Skip if no attention layer
            inps = inps.to(next(layer.parameters()).device)
            if 'position_ids' in layer_kwargs and layer_kwargs['position_ids'] is not None:
                layer_kwargs['position_ids'] = layer_kwargs['position_ids'].to(inps.device)
            # No need to compute position_embeddings for non-attention layers
            # Just remove it if present
            if 'position_embeddings' in layer_kwargs:
                del layer_kwargs['position_embeddings']
            inps = layer(inps, **layer_kwargs)[0]
            layer = layer.cpu()
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        inps = inps.to(next(layer.parameters()).device)
        if 'position_ids' in layer_kwargs and layer_kwargs['position_ids'] is not None:
            layer_kwargs['position_ids'] = layer_kwargs['position_ids'].to(inps.device)
        
        # Compute position_embeddings for this layer using the layer's own rotary_emb
        # This ensures correct head_dim matching
        if hasattr(layer.self_attn, 'rotary_emb'):
            rotary_emb = layer.self_attn.rotary_emb
            rotary_emb = rotary_emb.cuda()
            position_ids_local = layer_kwargs['position_ids'].to(inps.device)
            # Create dummy tensor with correct shape: [batch, num_kv_heads, seq_len, head_dim]
            batch_size, seq_len_local = inps.shape[:2]
            dummy_value_states = torch.zeros(
                batch_size, num_kv_heads, seq_len_local, head_dim,
                device=inps.device, dtype=inps.dtype
            )
            layer_kwargs['position_embeddings'] = rotary_emb(dummy_value_states, position_ids_local)
            rotary_emb = rotary_emb.cpu()
        
        # Forward pass to capture q, k, v outputs and get next layer's input
        layer_output = layer(inps, **layer_kwargs)
        inps = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        
        for h in handles:
            h.remove()
        
        # Concatenate captured features
        query_feat = torch.cat(output_feat['q_proj'], dim=0).cuda()
        key_feat = torch.cat(output_feat['k_proj'], dim=0).cuda()
        value_feat = torch.cat(output_feat['v_proj'], dim=0).cuda()
        
        # Compute RoPE embeddings for optimization
        # Use value_states shape to match transformers' rotary_emb behavior
        if hasattr(layer.self_attn, 'rotary_emb'):
            rotary_emb = layer.self_attn.rotary_emb
            rotary_emb = rotary_emb.cuda()
            position_ids = layer_kwargs.get('position_ids', torch.arange(seq_len, device='cuda').unsqueeze(0))
            # Use value_feat shape to compute RoPE (transformers uses value_states)
            # Reshape to [batch, seq_len, num_kv_heads, head_dim] for rotary_emb
            batch_size, seq_len_feat = value_feat.shape[:2]
            value_feat_reshaped = value_feat.view(batch_size, seq_len_feat, num_kv_heads, head_dim)
            value_states_for_rope = value_feat_reshaped[:, :1, :, :].transpose(1, 2)  # [batch, num_kv_heads, 1, head_dim]
            cos, sin = rotary_emb(value_states_for_rope, position_ids)
            rotary_emb = rotary_emb.cpu()
        elif hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.cuda()
            position_ids = layer_kwargs.get('position_ids', torch.arange(seq_len, device='cuda').unsqueeze(0))
            # Use value_feat shape
            batch_size, seq_len_feat = value_feat.shape[:2]
            value_feat_reshaped = value_feat.view(batch_size, seq_len_feat, num_kv_heads, head_dim)
            value_states_for_rope = value_feat_reshaped[:, :1, :, :].transpose(1, 2)
            cos, sin = model.model.rotary_emb(value_states_for_rope, position_ids)
            model.model.rotary_emb = model.model.rotary_emb.cpu()
        else:
            raise ValueError("Cannot get RoPE embeddings")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Optimize Pre-RoPE scaling
        scales_tuple, history = auto_scale_pre_rope(
            layer, layer_kwargs,
            x_bit=x_bit, w_bit=w_bit, q_config=q_config,
            query_feat=query_feat, key_feat=key_feat, value_feat=value_feat,
            cos=cos, sin=sin,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        historys.append(history)

        
        # Apply scaling to weights
        apply_pre_rope_scale(layer, [scales_tuple])
        
        # Append prefix to make names global
        pre_rope_results["scale"].append(
            append_str_prefix(scales_tuple, get_op_name(model, layer) + ".")
        )

        layer = layer.cpu()
        del layer
        del query_feat, key_feat, value_feat
        gc.collect()
        torch.cuda.empty_cache()

    #torch.save(historys, f'Jupyters/scale_shift_history.pt')
        
    model.config.use_cache = use_cache
    return pre_rope_results


def apply_pre_rope_scaling(model, pre_rope_results):
    """Apply Pre-RoPE scaling results to model."""
    apply_pre_rope_scale(model, pre_rope_results["scale"])
