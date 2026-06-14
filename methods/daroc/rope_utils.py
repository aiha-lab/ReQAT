import torch
import torch.nn as nn


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


def scaled_rotary_embed(query_states, key_states, cos, sin, k_scale=None, q_scale=None):
    """
    Apply channel-wise scaling before RoPE.
    
    Args:
        query_states: [batch, num_heads, seq_len, head_dim]
        key_states: [batch, num_kv_heads, seq_len, head_dim]
        cos, sin: RoPE embeddings
        k_scale: [num_kv_heads, head_dim] scaling factors for key_states
        q_scale: [num_heads, head_dim] scaling factors for query_states (if None, uses k_scale repeated)
    
    Returns:
        q, k: scaled and rotated query and key states
    """
    input_shape = query_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 128)
    half_dim = 64
    
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    
    if k_scale is None:
        # Compute scaling factors from key_states statistics
        k_max = key_states.abs().max(dim=2).values  # [batch, num_kv_heads, head_dim]
        k_max = k_max.mean(dim=0)  # [num_kv_heads, head_dim]
        k_max_half1 = k_max[:, :half_dim]  # [num_kv_heads, half_dim]
        k_max_half2 = k_max[:, half_dim:]  # [num_kv_heads, half_dim]
        # Geometric mean for stability
        head_half_scale = (k_max_half1 * k_max_half2).pow(1/2)  # [num_kv_heads, half_dim]
        head_full_scale = torch.cat([head_half_scale, head_half_scale], dim=-1)  # [num_kv_heads, head_dim]
        k_scale = head_full_scale.unsqueeze(-1).transpose(-1, -2)  # [num_kv_heads, 1, head_dim]
    
    # Apply scaling
    if k_scale is not None:
        key_states = key_states / k_scale
    
    if q_scale is None:
        # Repeat k_scale for query (assuming GQA with n_rep)
        num_heads = query_states.shape[1]
        num_kv_heads = key_states.shape[1]
        n_rep = num_heads // num_kv_heads
        q_scale = k_scale.repeat(1, n_rep, 1, 1)
    
    if q_scale is not None:
        query_states = query_states / q_scale
    
    # Apply RoPE
    q, k = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return q, k

