"""Tests for best FL checkpoint selection and paper eval artifact layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_pick_best_history_round_by_accuracy():
    from training.federated.best_checkpoint import pick_best_history_round

    history = [
        {"round": 1, "accuracy": 0.5, "macro_f1": 0.4, "quadratic_weighted_kappa": 0.3},
        {"round": 2, "accuracy": 0.8, "macro_f1": 0.7, "quadratic_weighted_kappa": 0.6},
        {"round": 3, "accuracy": 0.75, "macro_f1": 0.9, "quadratic_weighted_kappa": 0.9},
    ]
    best = pick_best_history_round(history)
    assert best is not None
    assert best["round"] == 2


def test_pick_best_history_round_tie_break_macro_f1():
    from training.federated.best_checkpoint import pick_best_history_round

    history = [
        {"round": 5, "accuracy": 0.82, "macro_f1": 0.70, "quadratic_weighted_kappa": 0.80},
        {"round": 6, "accuracy": 0.82, "macro_f1": 0.78, "quadratic_weighted_kappa": 0.70},
    ]
    best = pick_best_history_round(history)
    assert best is not None
    assert best["round"] == 6


def test_maybe_save_best_checkpoint_copies(tmp_path):
    from training.federated.best_checkpoint import maybe_save_best_checkpoint

    src = tmp_path / "adapter"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "adapter_model.safetensors").write_bytes(b"x")

    saved, record = maybe_save_best_checkpoint(
        global_dir=src,
        round_idx=3,
        metrics={"accuracy": 0.9, "macro_f1": 0.8, "quadratic_weighted_kappa": 0.7},
        current_best=None,
    )
    assert saved is True
    assert record["best_round"] == 3
    dest = Path(record["best_adapter_path"])
    assert (dest / "adapter_config.json").is_file()

    saved2, record2 = maybe_save_best_checkpoint(
        global_dir=src,
        round_idx=4,
        metrics={"accuracy": 0.85, "macro_f1": 0.9, "quadratic_weighted_kappa": 0.9},
        current_best=record,
    )
    assert saved2 is False
    assert record2["best_round"] == 3


def test_select_best_across_runs_prefers_fedprox_peak():
    from training.federated.best_checkpoint import select_best_across_runs

    winner = select_best_across_runs(
        [
            {
                "experiment_id": "fedavg_iid_r20",
                "best_round": 17,
                "best_val_metrics": {"accuracy": 0.8223, "macro_f1": 0.8, "quadratic_weighted_kappa": 0.8},
            },
            {
                "experiment_id": "fedprox_iid_r20",
                "best_round": 17,
                "best_val_metrics": {"accuracy": 0.8481, "macro_f1": 0.83, "quadratic_weighted_kappa": 0.84},
            },
        ]
    )
    assert winner is not None
    assert winner["experiment_id"] == "fedprox_iid_r20"


def test_paper_eval_writes_expected_artifacts(tmp_path):
    from experiments.federated.scripts import evaluate_deployable_fl_model as pe

    # Use real history if present; otherwise skip curve-dependent asserts beyond file names
    out = tmp_path / "paper"
    code = pe.main.__wrapped__ if hasattr(pe.main, "__wrapped__") else None  # noqa: F841
    # Invoke via argv
    old = list(__import__("sys").argv)
    try:
        __import__("sys").argv = [
            "evaluate_deployable_fl_model.py",
            "--out-dir",
            str(out),
            "--skip-live-eval",
        ]
        rc = pe.main()
    finally:
        __import__("sys").argv = old
    assert rc == 0
    assert (out / "paper_main_results.json").is_file()
    assert (out / "PAPER_RESULTS.md").is_file()
    assert (out / "table1_main_metrics.md").is_file()
    assert (out / "table2_algorithm_comparison.md").is_file()
    assert (out / "table3_per_class.md").is_file()
    # Learning curves require result JSONs + matplotlib
    fedavg = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
    try:
        import matplotlib  # noqa: F401

        has_mpl = True
    except ImportError:
        has_mpl = False
    if fedavg.is_file() and has_mpl:
        assert (out / "fig_learning_curves.png").is_file()
        assert (out / "fig_best_vs_final.png").is_file()


def test_existing_r20_histories_peak_before_final():
    fedavg = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
    fedprox = ROOT / "artifacts/federated/results/federated_lora_fedprox_iid_r20.json"
    if not fedavg.is_file() or not fedprox.is_file():
        pytest.skip("r20 result artifacts missing")
    from training.federated.best_checkpoint import pick_best_history_round

    a = json.loads(fedavg.read_text(encoding="utf-8"))
    p = json.loads(fedprox.read_text(encoding="utf-8"))
    ba = pick_best_history_round(a["history"])
    bp = pick_best_history_round(p["history"])
    assert ba is not None and bp is not None
    assert ba["accuracy"] >= a["history"][-1]["accuracy"]
    assert bp["accuracy"] >= p["history"][-1]["accuracy"]
    assert bp["accuracy"] >= ba["accuracy"]
