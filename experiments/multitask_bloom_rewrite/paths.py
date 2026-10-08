"""Repository-relative paths for the multi-task Bloom rewrite experiment.

The 1.5B generator is trained on three tasks only:
Bloom transformation (BloomShift), QA, and summarization.
The 0.5B Bloom classifier dataset is intentionally excluded.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"

# Canonical Bloom transformation corpus used by the final generator pipeline.
BLOOMSHIFT_DIR = DATA_DIR / "rewrite dataset" / "bloomshift_final_candidate"

# Historical corpora remain available for reproducibility; they are not final inputs.
LEGACY_BLOOM_REWRITE_DIR = DATA_DIR / "bloom_rewrite"
MULTITASK_DATA_DIR = DATA_DIR / "multitask_bloom_rewrite"
MULTITASK_FINAL_DATA_DIR = DATA_DIR / "multitask_bloom_rewrite_final"
FIGSHARE_V1 = DATA_DIR / "figshare_bloom_v1.csv"

CONFIG_DIR = EXPERIMENT_DIR / "configs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
HUMAN_EVAL_DIR = EXPERIMENT_DIR / "human_eval"
MODELS_DIR = EXPERIMENT_DIR / "models"
CACHE_DIR = EXPERIMENT_DIR / ".cache"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
TESTS_DIR = EXPERIMENT_DIR / "tests"
REPRODUCE_DIR = EXPERIMENT_DIR / "reproduce"

BLOOM_DATASET_VERSION = "bloomshift_final_candidate"
SEED = 42
TOPIC_SIMILARITY_THRESHOLD = 0.20

TASK_BLOOM = "bloom_rewrite"
TASK_QA = "qa"
TASK_SUMMARIZATION = "summarization"
TASKS = (TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION)

DEFAULT_TRAIN_MIX = {
    TASK_BLOOM: 0.40,
    TASK_QA: 0.30,
    TASK_SUMMARIZATION: 0.30,
}
