"""Repository paths for training and research code (not used at inference runtime)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_FEDERATED = ARTIFACTS_DIR / "federated"
ARTIFACTS_PRIVACY = ARTIFACTS_DIR / "privacy"
ARTIFACTS_EVALUATION = ARTIFACTS_DIR / "evaluation"

BUNDLES_DIR = ARTIFACTS_FEDERATED / "bundles"
UPDATES_DIR = ARTIFACTS_FEDERATED / "updates"
RUNS_DIR = ARTIFACTS_FEDERATED / "runs"
EXPERIMENTS_FEDERATED = ROOT / "experiments" / "federated"
