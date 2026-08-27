"""Repository-relative paths for the Bloom rewrite experiment.

No machine-specific absolute paths are stored here.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
REWRITE_DATA_DIR = DATA_DIR / "bloom_rewrite"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
HUMAN_EVAL_DIR = EXPERIMENT_DIR / "human_eval"
DOCS_DIR = REPO_ROOT / "docs"

FIGSHARE_V1 = DATA_DIR / "figshare_bloom_v1.csv"
FIGSHARE_COMBINED = DATA_DIR / "figshare_combined_dataset.csv"

DATASET_VERSION = "bloom_rewrite_synth_v2"
SEED = 42
REWRITE_ARCHIVE_DIR = DATA_DIR / "bloom_rewrite_versions"
