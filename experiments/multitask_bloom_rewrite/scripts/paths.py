"""Shared paths/constants for the EduGuard multitask experiment scripts."""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = EXPERIMENT_DIR / "configs"
HUMAN_EVAL_DIR = EXPERIMENT_DIR / "human_eval"
MULTITASK_DATA_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"
MULTITASK_FINAL_DATA_DIR = MULTITASK_DATA_DIR
SEED = 42

TASK_BLOOM = "bloom_rewrite"
TASK_QA = "qa"
TASK_SUMMARIZATION = "summarization"
TASKS = (TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION)
