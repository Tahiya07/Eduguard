"""
Multi-task EduGuard experiment package (Qwen 0.5B vs 1.5B).

Isolated under experiments/multitask_bloom_rewrite/.
Does not modify production application code or models/qwen.gguf.
"""
from __future__ import annotations

__all__ = [
    "paths",
    "prompts",
    "bloom_validation",
    "loss_masking",
]
