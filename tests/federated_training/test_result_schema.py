"""Verify federated result JSON schema completeness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.federated.config import FederatedLoraConfig
from training.federated.result_report import (
    REQUIRED_AGGREGATION_STATE_KEYS,
    REQUIRED_COMMUNICATION_KEYS,
    REQUIRED_METRICS_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    REQUIRED_TRAINING_ACTUAL_KEYS,
    REQUIRED_TRAINING_CONFIGURED_KEYS,
    REQUIRED_TRAINING_KEYS,
    RESULT_FORMAT,
    build_federated_result_report,
    estimate_optimizer_steps_per_client,
    validate_result_report,
    write_schema_sample,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "artifacts" / "federated" / "results" / "schema_smoke_sample.json"


def test_required_field_sets_are_documented():
    assert "optimizer" in REQUIRED_TRAINING_KEYS
    assert "configured" in REQUIRED_TRAINING_KEYS
    assert "actual" in REQUIRED_TRAINING_KEYS
    assert "total_optimizer_steps_estimate" in REQUIRED_TRAINING_CONFIGURED_KEYS
    assert "total_optimizer_steps_completed" in REQUIRED_TRAINING_ACTUAL_KEYS
    assert "per_class_f1" in REQUIRED_METRICS_KEYS
    assert "upload_bytes" in REQUIRED_COMMUNICATION_KEYS
    assert "trainable_aggregation_state" in REQUIRED_TOP_LEVEL_KEYS


def test_build_report_contains_all_required_fields():
    cfg = FederatedLoraConfig(num_clients=2, rounds=1, local_epochs=0.05)
    report = build_federated_result_report(
        cfg=cfg,
        run_id="test",
        experiment_id="test_exp",
        git_revision="abc",
        config_hash="def",
        dataset_hashes={"data/figshare_bloom_v1_train.csv": "hash"},
        client_sample_counts={"client_0": 10, "client_1": 12},
        partition_strategy="iid",
        dirichlet_alpha=None,
        setting_tag="fedavg_iid",
        global_adapter="/tmp/adapter",
        results_json_path="/tmp/out.json",
        round_checkpoint_path="/tmp/ckpt.json",
        communication={
            "total_upload_bytes": 100,
            "total_download_bytes": 200,
            "trainable_parameters": 999,
        },
        final_test_metrics={
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "per_class": {
                "Remember": {"f1": 0.3, "precision": 0.3, "recall": 0.3, "support": 1},
            },
            "n_eval": 5,
        },
        runtime_seconds=1.23,
        start_time="2026-01-01T00:00:00+00:00",
    )
    errors = validate_result_report(report)
    assert errors == [], errors
    assert report["format"] == RESULT_FORMAT
    assert report["metrics"]["per_class_f1"]["Remember"] == 0.3
    assert report["training"]["optimizer"] == "AdamW"
    assert report["training"]["configured"]["total_optimizer_steps_estimate"] > 0
    assert report["training"]["actual"]["execution_status"] == "NOT_EXECUTED"
    assert report["trainable_aggregation_state"]["trainable_parameter_count"] == 999
    assert report["communication"]["total_bytes"] == 300


def test_optimizer_step_estimate():
    cfg = FederatedLoraConfig(batch_size=2, grad_accum=8, local_epochs=1.0)
    steps = estimate_optimizer_steps_per_client(32, cfg)
    assert steps >= 1


def test_schema_smoke_sample_file_written():
    report = write_schema_sample(SAMPLE_PATH, clients=2, rounds=1)
    assert SAMPLE_PATH.is_file()
    loaded = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    errors = validate_result_report(loaded)
    assert errors == [], errors
    assert loaded["metrics"]["evaluation_status"] == "NOT_EVALUATED"
    assert loaded["metrics"]["accuracy"] is None
    assert loaded["metrics"]["macro_f1"] is None
    assert loaded["training"]["federated_rounds"] == 1
    assert loaded["training"]["configured"]["federated_rounds"] == 1
    assert loaded["training"]["actual"]["federated_rounds_configured"] == 1
    assert loaded["training"]["client_count"] == 2
    assert len(loaded["client_sample_counts"]) == 2
    assert loaded["partition"]["strategy"] == "iid"
    assert "lora" in loaded
    assert loaded["lora"]["r"] == 32
    assert report["config_hash"] == loaded["config_hash"]
