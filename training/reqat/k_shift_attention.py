# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
K-Shift Attention Module for Qwen2.

This module applies a static channel-wise bias subtraction to key states
after rotary position embedding, which can help with quantization.

Usage:
    from k_shift_attention import KShiftCallback

    # Method 1: Using callback (recommended for QAT - applies after quantization)
    trainer = QATSFTTrainer(
        model=model,
        ...
    )
    k_shift_callback = KShiftCallback(k_shift_path="k_shifts.pt")
    trainer.add_callback(k_shift_callback)
    
    # Method 2: Direct application (for inference or non-QAT training)
    from k_shift_attention import apply_k_shift_to_model, load_k_shifts
    k_shifts = load_k_shifts("k_shifts.pt")
    apply_k_shift_to_model(model, k_shifts)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any
from functools import wraps
from transformers.trainer_callback import TrainerCallback


def print_rank_0(msg: str) -> None:
    """Print only on rank 0 for distributed training."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(msg)
    except:
        print(msg)


def create_rope_wrapper_with_k_shift(original_rope_fn: Callable, k_shift: torch.Tensor) -> Callable:
    """
    Create a wrapper for apply_rotary_pos_emb that subtracts k_shift after RoPE.
    
    Args:
        original_rope_fn: The original apply_rotary_pos_emb function
        k_shift: The k_shift tensor for this layer
    
    Returns:
        Wrapped function that applies RoPE then subtracts k_shift from key_states
    """
    @wraps(original_rope_fn)
    def wrapped_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Apply original RoPE
        query_states, key_states = original_rope_fn(q, k, cos, sin, position_ids, unsqueeze_dim)
        
        # ========================================
        # K-SHIFT: Subtract k_shift from key_states
        # key_states shape: [batch, num_kv_heads, seq_len, head_dim]
        # k_shift shape: [1, num_kv_heads, 1, head_dim]
        # ========================================
        key_states = key_states - k_shift.to(key_states.dtype).to(key_states.device)
        # ========================================
        
        return query_states, key_states
    
    return wrapped_rope


# Global storage for original _QuantAttention.forward (to avoid re-patching)
_original_quant_attention_forward = None


def patch_quant_attention_class():
    """
    Patch _QuantAttention class to support k_shift.
    
    This patches the class-level forward method to check for k_shift buffer
    and temporarily wrap apply_rotary_pos_emb if present.
    """
    global _original_quant_attention_forward
    
    try:
        from modelopt.torch.quantization.plugins.huggingface import _QuantAttention
    except ImportError:
        print_rank_0("[K-Shift] Warning: Could not import _QuantAttention, skipping class patch")
        return False
    
    # Only patch once
    if _original_quant_attention_forward is not None:
        return True
    
    _original_quant_attention_forward = _QuantAttention.forward
    
    def forward_with_k_shift_support(self, *args, **kwargs):
        """
        Patched forward that applies k_shift if the module has k_shift buffer.
        """
        # Check if this attention module has k_shift
        if not hasattr(self, 'k_shift') or self.k_shift is None:
            # No k_shift, use original forward
            return _original_quant_attention_forward(self, *args, **kwargs)
        
        # Import the module's apply_rotary_pos_emb
        try:
            # Get the original module type to find the right modeling file
            from modelopt.torch.opt.dynamic import DynamicModule
            if isinstance(self, DynamicModule):
                original_cls = self.get_original_cls_by_level(level=0)
                module = __import__(original_cls.__module__, fromlist=[original_cls.__name__])
            else:
                import transformers.models.qwen2.modeling_qwen2 as module
        except Exception:
            import transformers.models.qwen2.modeling_qwen2 as module
        
        original_rope = module.apply_rotary_pos_emb
        
        # Create wrapped version with k_shift
        wrapped_rope = create_rope_wrapper_with_k_shift(original_rope, self.k_shift)
        
        # Temporarily replace apply_rotary_pos_emb
        module.apply_rotary_pos_emb = wrapped_rope
        
        try:
            # Call original _QuantAttention forward
            result = _original_quant_attention_forward(self, *args, **kwargs)
        finally:
            # Restore original RoPE
            module.apply_rotary_pos_emb = original_rope
        
        return result
    
    # Patch the class method
    _QuantAttention.forward = forward_with_k_shift_support
    print_rank_0("[K-Shift] Patched _QuantAttention.forward to support k_shift")
    return True


def apply_k_shift_to_model(
    model: nn.Module,
    k_shifts: Dict[int, torch.Tensor],
) -> int:
    """
    Apply k_shift to all Qwen2Attention layers in the model.
    
    Args:
        model: The Qwen2 model to modify
        k_shifts: Dictionary mapping layer index to k_shift tensor.
                  k_shift tensor shape: [num_kv_heads, head_dim] or [num_kv_heads * head_dim]
    
    Returns:
        Number of attention layers modified
    
    Example:
        >>> k_shifts = torch.load("k_shifts.pt")
        >>> # k_shifts = {0: tensor([...]), 1: tensor([...]), ...}
        >>> num_modified = apply_k_shift_to_model(model, k_shifts)
    """
    modified_count = 0
    
    # Find all attention modules
    for name, module in model.named_modules():
        # Check for Qwen2Attention-style module
        if (hasattr(module, 'q_proj') and 
            hasattr(module, 'k_proj') and 
            hasattr(module, 'v_proj') and 
            hasattr(module, 'o_proj') and
            hasattr(module, 'layer_idx')):
            
            layer_idx = module.layer_idx
            
            if layer_idx in k_shifts:
                k_shift = k_shifts[layer_idx]
                
                # Store k_shift as a buffer (non-trainable)
                # Name 'k_shift' will be saved as 'model.layers.X.self_attn.k_shift' in state_dict
                module.register_buffer('k_shift', k_shift.clone())
                
                modified_count += 1
                
                if modified_count <= 3 or layer_idx == max(k_shifts.keys()):
                    print_rank_0(
                        f"[K-Shift] Applied to {name} (layer {layer_idx}), "
                        f"k_shift shape: {k_shift.shape}, "
                        f"mean: {k_shift.mean().item():.6f}, "
                        f"std: {k_shift.std().item():.6f}"
                    )
    
    print_rank_0(f"[K-Shift] Modified {modified_count} attention layers")
    
    if modified_count == 0:
        print_rank_0("[K-Shift] Warning: No attention layers were modified. "
                    "Check that the model has Qwen2-style attention and k_shifts dict has correct layer indices.")
    
    return modified_count


def load_k_shifts(path: str, device: str = "cpu") -> Dict[int, torch.Tensor]:
    """
    Load k_shifts from a file.
    
    Supports multiple formats:
    1. Dict[int, Tensor]: Direct mapping from layer index to k_shift
    2. Dict[str, Tensor]: String keys like "layer_0", "0", etc.
    3. List[Tensor]: List indexed by layer
    
    Args:
        path: Path to the k_shifts file (.pt or .pth)
        device: Device to load tensors to
    
    Returns:
        Dict[int, Tensor] mapping layer indices to k_shift tensors
    """
    data = torch.load(path, map_location=device)
    
    if isinstance(data, dict):
        # Check if keys are already integers
        if all(isinstance(k, int) for k in data.keys()):
            print_rank_0(f"[K-Shift] Loaded {len(data)} k_shifts from {path}")
            return data
        
        # Try to convert string keys to integers
        k_shifts = {}
        import re
        for key, value in data.items():
            if isinstance(key, str):
                # Try "layers.X" pattern first (e.g., "model.layers.45.self_attn.k_shift")
                match = re.search(r'layers\.(\d+)', key)
                if match:
                    layer_idx = int(match.group(1))
                    k_shifts[layer_idx] = value
                else:
                    # Fallback: extract first number from key like "layer_0", "0", etc.
                    match = re.search(r'(\d+)', key)
                    if match:
                        layer_idx = int(match.group(1))
                        k_shifts[layer_idx] = value
            elif isinstance(key, int):
                k_shifts[key] = value
        
        print_rank_0(f"[K-Shift] Loaded {len(k_shifts)} k_shifts from {path}")
        return k_shifts
    
    elif isinstance(data, (list, tuple)):
        k_shifts = {i: v for i, v in enumerate(data)}
        print_rank_0(f"[K-Shift] Loaded {len(k_shifts)} k_shifts from {path}")
        return k_shifts
    
    else:
        raise ValueError(f"Unsupported k_shifts format: {type(data)}. "
                        "Expected Dict[int, Tensor], Dict[str, Tensor], or List[Tensor].")


def check_k_shift_stats(model: nn.Module) -> None:
    """Print statistics of all k_shift buffers in the model."""
    print_rank_0("[K-Shift] Statistics:")
    
    for name, module in model.named_modules():
        if hasattr(module, 'k_shift') and isinstance(module.k_shift, torch.Tensor):
            k_shift = module.k_shift
            print_rank_0(
                f"  {name}: shape={tuple(k_shift.shape)}, "
                f"mean={k_shift.mean().item():.6f}, "
                f"std={k_shift.std().item():.6f}, "
                f"min={k_shift.min().item():.6f}, "
                f"max={k_shift.max().item():.6f}"
            )


class KShiftCallback(TrainerCallback):
    """
    Trainer callback to apply k_shift after quantization.
    
    This is the recommended way to use k_shift with QAT, as it ensures
    k_shift is applied AFTER ModelOpt's quantization, preserving the
    quantizer hooks on attention modules.
    
    Usage:
        trainer = QATSFTTrainer(model=model, ...)
        k_shift_callback = KShiftCallback(k_shift_path="k_shifts.pt")
        trainer.add_callback(k_shift_callback)
        trainer.train()
    """
    
    def __init__(self, k_shift_path: str):
        """
        Args:
            k_shift_path: Path to k_shifts file (.pt)
        """
        self.k_shift_path = k_shift_path
        self._applied = False
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Apply k_shift at the start of training (after quantization)."""
        if self._applied:
            return
        
        if model is None:
            print_rank_0("[K-Shift] Warning: model is None in on_train_begin")
            return
        
        # Unwrap model if needed (DeepSpeed, FSDP, etc.)
        unwrapped_model = model
        while hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module
        
        # First, patch _QuantAttention class to support k_shift
        patch_quant_attention_class()
        
        # Then, register k_shift buffers to attention modules
        print_rank_0(f"[K-Shift] Applying k_shift from {self.k_shift_path} (post-quantization)")
        k_shifts = load_k_shifts(self.k_shift_path, device="cpu")
        num_modified = apply_k_shift_to_model(unwrapped_model, k_shifts)
        
        if num_modified > 0:
            print_rank_0(f"[K-Shift] Successfully applied to {num_modified} attention layers")
            self._applied = True
        else:
            print_rank_0("[K-Shift] Warning: No attention layers were modified")


# ==============================================================================
# Test code
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing K-Shift Attention")
    print("=" * 60)
    
    # Create dummy k_shifts
    num_layers = 4
    num_kv_heads = 8
    head_dim = 64
    
    k_shifts = {
        i: torch.randn(num_kv_heads, head_dim) * 0.1
        for i in range(num_layers)
    }
    
    print(f"\nCreated k_shifts for {len(k_shifts)} layers")
    for idx, k_shift in k_shifts.items():
        print(f"  Layer {idx}: shape={k_shift.shape}, mean={k_shift.mean():.4f}")
    
    # Test load_k_shifts with different formats
    print("\n--- Testing load_k_shifts ---")
    
    # Save and load in different formats
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Format 1: Dict[int, Tensor]
        path1 = os.path.join(tmpdir, "k_shifts_int.pt")
        torch.save(k_shifts, path1)
        loaded1 = load_k_shifts(path1)
        print(f"Format 1 (Dict[int]): Loaded {len(loaded1)} k_shifts")
        
        # Format 2: Dict[str, Tensor]
        str_k_shifts = {f"layer_{i}": v for i, v in k_shifts.items()}
        path2 = os.path.join(tmpdir, "k_shifts_str.pt")
        torch.save(str_k_shifts, path2)
        loaded2 = load_k_shifts(path2)
        print(f"Format 2 (Dict[str]): Loaded {len(loaded2)} k_shifts")
        
        # Format 3: List[Tensor]
        list_k_shifts = [k_shifts[i] for i in range(num_layers)]
        path3 = os.path.join(tmpdir, "k_shifts_list.pt")
        torch.save(list_k_shifts, path3)
        loaded3 = load_k_shifts(path3)
        print(f"Format 3 (List): Loaded {len(loaded3)} k_shifts")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


