import torch
import torch.nn as nn
import numpy as np
import logging

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from .rope_utils import scaled_rotary_embed, apply_rotary_pos_emb
from .module import get_op_by_name, get_op_name, set_op_by_name
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

__all__ = ["auto_scale_pre_rope", "apply_pre_rope_scale"]

logger = logging.getLogger(__name__)

@torch.no_grad()
def calculate_sqnr(original: torch.Tensor, quantized: torch.Tensor, axis=None):
    noise = original - quantized
    if axis is None:
        signal_power = torch.mean(original.pow(2))
        noise_power = torch.mean(noise.pow(2))
    else:
        signal_power = torch.mean(original.pow(2), dim=axis)
        noise_power = torch.mean(noise.pow(2), dim=axis)
    
    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
    
    return sqnr

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

@torch.no_grad()
def attention(q, k, v, n_rep, head_dim, cos, sin, quantizer=None, k_shift=None):
    input_shape = q.shape[:-1] # B,T
    hidden_shape = (*input_shape, -1, head_dim)
    q = q.view(hidden_shape).transpose(1, 2)
    k = k.view(hidden_shape).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if k_shift is not None:
        k_shift = repeat_kv(k_shift, n_rep)
        k = k-k_shift
    if quantizer is not None:
        k = quantizer(k)
        v = quantizer(v)
    attn_weights = torch.matmul(q, k.transpose(2, 3))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_score = torch.matmul(attn_weights, v)
    return attn_score

@torch.no_grad()
def optimize_pre_rope_scaling(
    query_states, key_states, value_states, cos, sin,
    quantizer, num_heads, num_kv_heads, head_dim,
):
    key_states = key_states.view(-1,key_states.shape[-1]) # B*T, D
    num_tokens = key_states.shape[0]
    n_rep = num_heads//num_kv_heads
    half_dim = head_dim // 2
    k_reshaped = key_states.view(num_tokens, num_kv_heads, head_dim)
    k_max = k_reshaped.abs().max(dim=0).values
    k_max_half1 = k_max[:, :half_dim] # [num_kv_heads, half_dim]
    k_max_half2 = k_max[:, half_dim:] # [num_kv_heads, half_dim]
    head_half_scale = torch.max(k_max_half1, k_max_half2) # [num_kv_heads, half_dim]
    head_full_scale = torch.cat([head_half_scale, head_half_scale], dim=1)
    # Init Scale
    k_scale = head_full_scale.view(1, -1).clone()
    q_scale = head_full_scale.unsqueeze(1).repeat(1, n_rep, 1)
    q_scale = q_scale.view(1, -1).clone()
    k_scale = k_scale.clamp(min=1e-5)
    q_scale = q_scale.clamp(min=1e-5)
    # Init Shift
    k_shift = key_states.mean(0).view(1, -1, 1, head_dim) # [1, head, 1, head_dim]

    attn_score_ref = attention(query_states, key_states, value_states, n_rep, head_dim, cos, sin)

    best_error = float('inf')
    history = torch.zeros([20, 20])
    for scale_ratio in range(20):
    #for scale_ratio in [0]:
        for shift_ratio in range(20):
        #for shift_ratio in [0]:
            new_k_scale = k_scale.pow(scale_ratio/20)
            new_q_scale = q_scale.pow(scale_ratio/20)
            new_k_shift = k_shift*shift_ratio/20
            new_key_states = key_states.clone()/new_k_scale
            new_query_states = query_states.clone()*new_q_scale.unsqueeze(0)
            attn_score_new = attention(new_query_states, new_key_states, value_states, n_rep, head_dim, cos, sin, quantizer, new_k_shift)
            err = (attn_score_ref-attn_score_new).pow(2).mean().item()
            history[scale_ratio, shift_ratio] = err

            if err < best_error:
                best_k_scale = new_k_scale
                best_q_scale = new_q_scale
                best_k_shift = new_k_shift
                best_error = err
                print(f'[Updated] Best error in (scale/shift)=({scale_ratio}/{shift_ratio}): Error {err*1e3:.4f}e-3')

    #print(best_k_scale)
    #print(best_k_shift)
    return best_k_scale, best_q_scale, best_k_shift, history


@torch.no_grad()
def merge_scales_to_weights(q_proj, k_proj, k_scale, q_scale, num_heads, num_kv_heads):
    device = q_proj.weight.device
    head_dim = k_scale.shape[-1]
    n_rep = num_heads // num_kv_heads
    k_scale_flat = k_scale.view(-1)  # [num_kv_heads * head_dim]
    q_scale_flat = q_scale.view(-1)

    @torch.no_grad()
    def rescale_qk_proj_in_fp32(
        q_proj, k_proj,
        q_scale_flat: torch.Tensor,  # [q_out]
        k_scale_flat: torch.Tensor,  # [k_out]
    ):
        device = q_proj.weight.device
    
        # ---- K ----
        Wk = k_proj.weight
        wk_dtype = Wk.dtype
        k_scale_col = k_scale_flat.to(device=device, dtype=torch.float32).view(-1, 1)
    
        Wk_fp32 = Wk.detach().to(torch.float32)
        Wk_fp32.div_(k_scale_col)
        Wk.copy_(Wk_fp32.to(dtype=wk_dtype))
    
        if k_proj.bias is not None:
            bk = k_proj.bias
            bk_dtype = bk.dtype
            bk_fp32 = bk.detach().to(torch.float32)
            bk_fp32.div_(k_scale_flat.to(device=device, dtype=torch.float32))
            bk.copy_(bk_fp32.to(dtype=bk_dtype))
    
        # ---- Q ----
        Wq = q_proj.weight
        wq_dtype = Wq.dtype
        q_scale_col = q_scale_flat.to(device=device, dtype=torch.float32).view(-1, 1)
    
        Wq_fp32 = Wq.detach().to(torch.float32)
        Wq_fp32.mul_(q_scale_col)
        Wq.copy_(Wq_fp32.to(dtype=wq_dtype))
    
        if q_proj.bias is not None:
            bq = q_proj.bias
            bq_dtype = bq.dtype
            bq_fp32 = bq.detach().to(torch.float32)
            bq_fp32.mul_(q_scale_flat.to(device=device, dtype=torch.float32))
            bq.copy_(bq_fp32.to(dtype=bq_dtype))

    rescale_qk_proj_in_fp32(q_proj, k_proj, q_scale_flat, k_scale_flat)

@torch.no_grad()
def auto_scale_pre_rope(
    module, module_kwargs,
    x_bit, w_bit, q_config,
    query_feat, key_feat, value_feat, cos, sin,
    num_heads, num_kv_heads, head_dim
):
    quantizer = TensorQuantizer()
    cfg_nvfp4 = {'num_bits': (1, 2), 'block_sizes': {-1: 16, 'type': 'dynamic', 'scale_bits': (4, 3)}, 'axis': None, 'enable': True, 'pass_through_bwd': True}
    quantizer.set_from_attribute_config(cfg_nvfp4)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    batch_size, seq_len = query_feat.shape[:2]
    query_states = query_feat
    key_states = key_feat
    value_states = value_feat
    
    # Optimize scaling
    k_scale, q_scale, k_shift, history = optimize_pre_rope_scaling(
        query_states, key_states, value_states, cos, sin, quantizer,
        num_heads, num_kv_heads, head_dim,
    )
    
    # Get layer names
    if isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer)):
        q_proj_name = get_op_name(module, module.self_attn.q_proj)
        k_proj_name = get_op_name(module, module.self_attn.k_proj)
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")
    
    return (q_proj_name, (q_proj_name, k_proj_name), (k_scale, q_scale, k_shift)), history


@torch.no_grad()
def apply_pre_rope_scale(module, scales_list):
    """
    Apply Pre-RoPE scaling factors to weights.
    
    Args:
        module: module to apply scaling to
        scales_list: list of (prev_op_name, layer_names, (k_scale, v_scale)) tuples
    """
    for prev_op_name, layer_names, (k_scale, q_scale, k_shift) in scales_list:
        q_proj = get_op_by_name(module, layer_names[0])
        k_proj = get_op_by_name(module, layer_names[1])
        attn = get_op_by_name(module, layer_names[1][:-7])
        
        # Get num_heads and num_kv_heads from weight shapes
        num_heads = q_proj.weight.shape[0] // k_scale.shape[-1]
        num_kv_heads = k_proj.weight.shape[0] // k_scale.shape[-1]
        head_dim = k_scale.shape[-1]
        
        merge_scales_to_weights(q_proj, k_proj, k_scale, q_scale, 
                               num_heads, num_kv_heads)
        attn.register_buffer("k_shift", k_shift)

