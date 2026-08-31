"""Bloom taxonomy federated task metadata."""

from __future__ import annotations

from training.federated.config import BLOOM_LABELS, BLOOM_LABEL_ORDER, FederatedLoraConfig

TASK_NAME = "bloom_seq_cls"
NUM_LABELS = len(BLOOM_LABELS)


def task_summary(config: FederatedLoraConfig) -> dict:
    return {
        "task": TASK_NAME,
        "num_labels": NUM_LABELS,
        "label_order": list(BLOOM_LABEL_ORDER),
        "base_model": config.base_model,
        "lora": config.lora_config_dict(),
    }
