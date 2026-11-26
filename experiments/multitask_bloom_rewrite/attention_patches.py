"""Attention dispatch fixes for training on builds without flash attention.

Qwen2.5 uses grouped-query attention (12 query heads, 2 KV heads for 1.5B).
When no attention mask is passed, transformers hands the 2 KV heads to
``scaled_dot_product_attention(enable_gqa=True)``. PyTorch wheels built without
flash attention (e.g. Windows) have no fused kernel for that case and fall back
to the math kernel, which materializes the full L x L attention matrix:
measured on an RTX 4080, one 7,217-token LoRA forward/backward needed 10.3 GB of
activations and 7.3 s. Repeating KV heads first (the same ``repeat_kv`` path
transformers already uses whenever a mask is present) lets the memory-efficient
kernel run: 2.2 GB and 2.4 s for the same row, linear in length.
"""
from __future__ import annotations


def force_sdpa_repeat_kv() -> bool:
    """Make transformers' SDPA path repeat KV heads instead of using enable_gqa."""
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention
    except ImportError:
        return False
    if not hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        return False
    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    return True
