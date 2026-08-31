"""Optimizer step reporting: configured estimates vs actual execution counters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.federated.config import FederatedLoraConfig
from training.federated.execution_stats import (
    execution_from_bundle,
    get_privacy_accounting_steps,
    read_trainer_execution_stats,
    summarize_execution_actual,
    summarize_round_bundles,
)
from training.federated.result_report import (
    REQUIRED_TRAINING_ACTUAL_KEYS,
    REQUIRED_TRAINING_CONFIGURED_KEYS,
    build_federated_result_report,
    validate_result_report,
)

ROOT = Path(__file__).resolve().parents[2]


def _fake_bundle(client_id: str, round_idx: int, steps: int, n_samples: int = 10) -> dict:
    return {
        "client_id": client_id,
        "round": round_idx,
        "n_samples": n_samples,
        "execution": {
            "optimizer_steps_completed": steps,
            "epochs_completed": 0.05,
            "source": "trainer.state.global_step",
        },
    }


def _base_report_kwargs(cfg: FederatedLoraConfig, history: list | None = None) -> dict:
    return dict(
        cfg=cfg,
        run_id="step_test",
        experiment_id="step_test",
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
        communication={"total_upload_bytes": 0, "total_download_bytes": 0},
        final_test_metrics=None,
        runtime_seconds=0.1,
        start_time="2026-01-01T00:00:00+00:00",
        history=history or [],
        status="EXECUTED",
    )


def test_configured_and_actual_fields_are_distinguishable():
    cfg = FederatedLoraConfig(num_clients=2, rounds=5, local_epochs=3.0)
    report = build_federated_result_report(**_base_report_kwargs(cfg))

    configured = report["training"]["configured"]
    actual = report["training"]["actual"]

    assert set(configured.keys()) >= REQUIRED_TRAINING_CONFIGURED_KEYS
    assert set(actual.keys()) >= REQUIRED_TRAINING_ACTUAL_KEYS
    assert configured["total_optimizer_steps_estimate"] > 0
    assert actual["total_optimizer_steps_completed"] == 0
    assert actual["execution_status"] == "NOT_EXECUTED"
    assert configured["federated_rounds"] == 5
    assert actual["federated_rounds_completed"] == 0
    assert actual["federated_rounds_configured"] == 5
    assert report["training"]["total_optimizer_steps_estimate"] == configured[
        "total_optimizer_steps_estimate"
    ]
    assert "total_optimizer_steps_estimate" not in actual
    assert "total_optimizer_steps_completed" not in configured


def test_actual_counters_increment_in_smoke_aggregation():
    bundles_r1 = [_fake_bundle("client_0", 1, 3), _fake_bundle("client_1", 1, 4)]
    bundles_r2 = [_fake_bundle("client_0", 2, 2), _fake_bundle("client_1", 2, 5)]

    round1 = summarize_round_bundles(bundles_r1, 1)
    round2 = summarize_round_bundles(bundles_r2, 2)

    assert round1["optimizer_steps_round_total"] == 7
    assert round2["optimizer_steps_round_total"] == 7
    assert round1["optimizer_steps_per_client"]["client_0"] == 3
    assert round2["optimizer_steps_per_client"]["client_1"] == 5

    history = [{"round": 1, "execution": round1}, {"round": 2, "execution": round2}]
    actual = summarize_execution_actual(history, configured_rounds=5)

    assert actual["federated_rounds_completed"] == 2
    assert actual["total_optimizer_steps_completed"] == 14
    assert actual["optimizer_steps_per_round_all_clients"] == [7, 7]
    assert actual["optimizer_steps_per_client_per_round"]["round_01"]["client_0"] == 3
    assert actual["execution_status"] == "PARTIAL"
    assert actual["privacy_accounting_steps"] == 14
    assert actual["privacy_accounting_steps_source"] == "actual_global_step"


def test_interrupted_training_does_not_report_unexecuted_steps():
    cfg = FederatedLoraConfig(num_clients=2, rounds=5, local_epochs=3.0)
    round1 = summarize_round_bundles(
        [_fake_bundle("client_0", 1, 2), _fake_bundle("client_1", 1, 3)],
        1,
    )
    # Simulates checkpoint after round 1 only; rounds 2-5 never ran.
    history = [{"round": 1, "execution": round1}]
    report = build_federated_result_report(**_base_report_kwargs(cfg, history=history))
    actual = report["training"]["actual"]
    configured = report["training"]["configured"]

    assert actual["federated_rounds_completed"] == 1
    assert actual["federated_rounds_configured"] == 5
    assert actual["total_optimizer_steps_completed"] == 5
    assert actual["execution_status"] == "PARTIAL"
    assert len(actual["optimizer_steps_per_round_all_clients"]) == 1
    assert configured["total_optimizer_steps_estimate"] > actual["total_optimizer_steps_completed"]
    assert get_privacy_accounting_steps(report) == 5


def test_result_json_records_actual_completed_round_and_step_counts(tmp_path):
    cfg = FederatedLoraConfig(num_clients=2, rounds=2, local_epochs=1.0)
    round1 = summarize_round_bundles(
        [_fake_bundle("client_0", 1, 1), _fake_bundle("client_1", 1, 2)],
        1,
    )
    round2 = summarize_round_bundles(
        [_fake_bundle("client_0", 2, 3), _fake_bundle("client_1", 2, 4)],
        2,
    )
    history = [{"round": 1, "execution": round1}, {"round": 2, "execution": round2}]
    out = tmp_path / "result.json"
    kwargs = _base_report_kwargs(cfg, history=history)
    kwargs["results_json_path"] = str(out)
    report = build_federated_result_report(**kwargs)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    errors = validate_result_report(loaded)
    assert errors == [], errors
    assert loaded["training"]["actual"]["federated_rounds_completed"] == 2
    assert loaded["training"]["actual"]["total_optimizer_steps_completed"] == 10
    assert loaded["training"]["actual"]["execution_status"] == "COMPLETE"
    assert loaded["training"]["actual"]["optimizer_steps_per_round_all_clients"] == [3, 7]


def test_read_trainer_execution_stats_from_mock_trainer():
    trainer = SimpleNamespace(
        state=SimpleNamespace(
            global_step=11,
            epoch=0.25,
            max_steps=100,
            logging_steps=10,
        )
    )
    stats = read_trainer_execution_stats(trainer)
    assert stats["optimizer_steps_completed"] == 11
    assert stats["epochs_completed"] == 0.25
    assert stats["max_steps"] == 100


def test_execution_from_bundle_requires_execution_block():
    bundle = _fake_bundle("client_0", 1, 8)
    ex = execution_from_bundle(bundle)
    assert ex["optimizer_steps_completed"] == 8
    assert ex["client_id"] == "client_0"

    missing = execution_from_bundle({"client_id": "client_0", "round": 1})
    assert missing["optimizer_steps_completed"] is None
