import torch
from tqdm import tqdm
from .qmodule import ScaledActivation
from .module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'bigcode' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'neox' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    

import torch
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs

class MXQuantizer(torch.nn.Module):
    def __init__(self, bits, data_format, block_size, scale_mode, rounding):
        super(MXQuantizer, self).__init__()
        self.data_format = data_format
        self.bits = bits
        self.block_size = block_size
        self.scale_bits = 8
        self.scale_mode = scale_mode
        self.rounding = rounding
        self.mx_specs = MxSpecs(
            scale_bits=self.scale_bits,
            a_elem_format=data_format,
            block_size=block_size,
            custom_cuda=True,
            per_tensor=False,
        )
        self.tensor_max = -1

    def quant(self, x):
        if self.bits < 16:
            dtype = x.dtype
            qx = quantize_mx_op(
                x.float(),
                self.mx_specs,
                elem_format=self.data_format,
                scale_mode=self.scale_mode,
                axes=[-1],
                round=self.rounding,
                tensor_max=self.tensor_max,
            ).to(dtype)
        else:
            qx = x
        return qx

    def __repr__(self):
        return f"MXQuantizer(mx_format={self.data_format}, group_size={self.block_size}, scale_mode={self.scale_mode}, rounding={self.rounding})"

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_config, quantizer
):    
    from .pre_quant import get_blocks, get_named_linears
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            dev = m.weight.device
            if dev.type == "cpu":
                m.cuda()
            print(f'Quantizing {n}')
            m.weight.data = quantizer(m.weight.data)
            if dev.type == "cpu":
                m.cpu()

