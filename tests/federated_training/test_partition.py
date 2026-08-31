"""Tests for federated partitioning."""

from __future__ import annotations

import pandas as pd

from training.federated.config import BLOOM_LABELS
from training.federated.partition import partition_csv


def test_iid_partition_covers_all_rows(tmp_path):
    rows = []
    for label in BLOOM_LABELS:
        for i in range(10):
            rows.append({"question": f"{label} q{i}", "bloom_level": label})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    parts = partition_csv(
        csv_path,
        num_clients=4,
        strategy="iid",
        seed=42,
    )
    assert len(parts) == 4
    total = sum(len(df) for df in parts.values())
    assert total == len(rows)


def test_dirichlet_partition_non_empty_clients(tmp_path):
    rows = []
    for label in BLOOM_LABELS:
        for i in range(20):
            rows.append({"question": f"{label} q{i}", "bloom_level": label})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    parts = partition_csv(
        csv_path,
        num_clients=4,
        strategy="non_iid_label",
        dirichlet_alpha=0.5,
        seed=42,
    )
    assert all(len(df) >= 1 for df in parts.values())
