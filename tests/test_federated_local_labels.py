"""Local label store used by deployment-side federated rounds (no ML deps)."""
from __future__ import annotations

import json

import pytest

from backend.label_store import LocalLabelStore, question_key


def test_latest_confirmation_wins_and_whitespace_is_normalized(tmp_path):
    store = LocalLabelStore(tmp_path / "labels.jsonl", "site_a")
    store.add("What is  osmosis?", "Understand", predicted_level="Understand")
    store.add("  What is osmosis? ", "Remember", predicted_level="Understand")
    rows = store.rows()
    assert len(rows) == 1
    assert rows[0]["bloom_level"] == "Remember"
    assert rows[0]["teacher_corrected"] is True
    assert rows[0]["key"] == question_key("what is osmosis?")


def test_stats_count_levels_and_corrections(tmp_path):
    store = LocalLabelStore(tmp_path / "labels.jsonl", "site_a")
    store.add("Define entropy.", "Remember", predicted_level="Remember")
    store.add("Design an experiment to test drag.", "Create", predicted_level="Apply")
    stats = store.stats()
    assert stats["n_labels"] == 2
    assert stats["n_teacher_corrected"] == 1
    assert stats["per_level"]["Create"] == 1 and stats["per_level"]["Analyze"] == 0


def test_rejects_unknown_level_and_empty_question(tmp_path):
    store = LocalLabelStore(tmp_path / "labels.jsonl", "site_a")
    with pytest.raises(ValueError):
        store.add("Explain inertia.", "Synthesis")
    with pytest.raises(ValueError):
        store.add("   ", "Remember")


def test_rows_only_hold_question_and_label_fields(tmp_path):
    """Notes and session ids are never written to the training label file."""
    path = tmp_path / "labels.jsonl"
    LocalLabelStore(path, "site_a").add("State Ohm's law.", "Remember", decision="approved")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "notes" not in row and "sid" not in row
    assert row["site_id"] == "site_a"


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = LocalLabelStore(path, "site_a")
    store.add("State Ohm's law.", "Remember")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    assert store.stats()["n_labels"] == 1


def test_site_export_requires_minimum_labels(tmp_path):
    pytest.importorskip("pandas")
    from training.federated.local_round import export_site_csv

    store = LocalLabelStore(tmp_path / "labels.jsonl", "site_a")
    store.add("State Ohm's law.", "Remember")
    with pytest.raises(SystemExit):
        export_site_csv(store, tmp_path / "out.csv", min_labels=30)
    for i in range(30):
        store.add(f"Question number {i}?", "Remember" if i % 2 else "Analyze")
    stats = export_site_csv(store, tmp_path / "out.csv", min_labels=30)
    assert stats["n_labels"] == 31
    header = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "question,bloom_level"
