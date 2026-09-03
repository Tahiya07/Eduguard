"""Unit tests for federated DP training helpers."""

from __future__ import annotations

from training.federated.dp_training import (
    _canonical_trainable_state,
    apply_locked_training_rules,
    compose_federated_privacy_report,
    dp_config_from_lock,
    normalize_dp_mode,
)
from training.federated.config import FederatedLoraConfig, effective_prox_mu


def test_normalize_dp_mode():
    assert normalize_dp_mode("full") == "full"
    assert normalize_dp_mode("score-head-only") == "score-head-only"
    assert normalize_dp_mode("score_head") == "score-head-only"


def test_apply_locked_training_rules_zeros_smoothing_and_weights():
    cfg = FederatedLoraConfig(label_smoothing=0.05, use_class_weights=True, prox_mu=0.01)
    dp = dp_config_from_lock(
        {"dp_mode": "full", "locked_procedure": {"validation_lora_dropout": 0.0}},
        noise_multiplier=1.0,
        target_delta=1e-5,
    )
    locked = apply_locked_training_rules(cfg, dp)
    assert locked.label_smoothing == 0.0
    assert locked.use_class_weights is False
    assert locked.lora_dropout == 0.0
    assert locked.algorithm == cfg.algorithm
    assert locked.prox_mu == cfg.prox_mu


def test_compose_federated_privacy_report_naive_bound():
    report = compose_federated_privacy_report(
        client_epsilons=[1.2, 1.5, 1.1],
        rounds=5,
        delta=1e-5,
        noise_multiplier=1.0,
        dp_mode="full",
    )
    assert report["local_epsilon_max"] == 1.5
    assert report["naive_composition_upper_bound"] == 7.5


def test_dp_config_from_lock_reads_max_grad_norm():
    lock = {
        "dp_mode": "full",
        "federated_config": {"max_grad_norm": 0.75},
        "locked_procedure": {"validation_lora_dropout": 0.0},
    }
    dp = dp_config_from_lock(lock, noise_multiplier=0.8, target_delta=1e-5, lock_path="/tmp/lock.json")
    assert dp.max_grad_norm == 0.75
    assert dp.noise_multiplier == 0.8
    assert dp.lock_path == "/tmp/lock.json"


def test_effective_prox_mu_ignores_default_on_fedavg():
    assert effective_prox_mu("fedavg", 0.01) == 0.0
    assert effective_prox_mu("fedprox", 0.01) == 0.01


def test_canonical_trainable_state_strips_opacus_prefix():
    import torch

    state = _canonical_trainable_state(
        {"_module.base_model.model.layers.0.lora_A": torch.ones(1)}
    )
    assert "base_model.model.layers.0.lora_A" in state
    assert "_module.base_model.model.layers.0.lora_A" not in state
