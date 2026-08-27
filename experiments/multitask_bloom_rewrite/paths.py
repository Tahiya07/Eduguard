"""Repository-relative paths for the multi-task Bloom rewrite experiment.

Production code and production GGUF are never written here.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
BLOOM_REWRITE_DIR = DATA_DIR / "bloom_rewrite"
MULTITASK_DATA_DIR = DATA_DIR / "multitask_bloom_rewrite"
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

BLOOM_DATASET_VERSION = "bloom_rewrite_synth_v2"
BLOOM_DATASET_HASH = "b3725b77862868dcd3d7ad07f1d2e15ae41d6d9887e8510d5396de8c4e790bae"
SEED = 42
TOPIC_SIMILARITY_THRESHOLD = 0.20

TASK_BLOOM = "bloom_rewrite"
TASK_QA = "qa"
TASK_SUMMARIZATION = "summarization"
TASKS = (TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION)

# Initial Mix A (pre-registered). Mix B/C are sensitivity options only.
DEFAULT_TRAIN_MIX = {
    TASK_BLOOM: 0.40,
    TASK_QA: 0.30,
    TASK_SUMMARIZATION: 0.30,
}

QA_TRAIN_SUBSAMPLE = 9000
SUM_TRAIN_SUBSAMPLE = 9000
