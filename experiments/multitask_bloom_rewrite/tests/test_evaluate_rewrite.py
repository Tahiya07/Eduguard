"""Unit tests for multitask evaluation pipeline (no model weights required)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import (  # noqa: E402
    is_valid_imperative_exam,
    validate_bloom_example,
)
from eval_dataset import load_evaluation_split  # noqa: E402
from eval_metrics import (  # noqa: E402
    aggregate_qa_metrics,
    bloom_classification_metrics,
    qa_exact_match,
    qa_token_f1,
    source_target_matrix,
)
from eval_classifier import resolve_classifier_dir  # noqa: E402
from eval_model import resolve_checkpoint, _sanitize_lora_adapter_config  # noqa: E402
from paths import REPO_ROOT, TASK_SEPARATED_DATA_DIR  # noqa: E402
from prompts import assert_no_source_level_in_prompt, build_generation_prompt  # noqa: E402


class TestDatasetAssertions(unittest.TestCase):
    def test_rejects_non_evaluation_split(self) -> None:
        with self.assertRaises(ValueError):
            load_evaluation_split(TASK_SEPARATED_DATA_DIR, split="train")

    def test_test_split_matches_manifest_task_counts(self) -> None:
        rows, meta = load_evaluation_split(TASK_SEPARATED_DATA_DIR, split="test")
        manifest = json.loads(
            (TASK_SEPARATED_DATA_DIR / "dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_by_task = manifest["counts"]["test"]
        self.assertEqual(meta["split"], "test")
        self.assertEqual(meta["counts"]["by_task"], expected_by_task)
        self.assertEqual(meta["counts"]["test_count"], sum(expected_by_task.values()))
        self.assertEqual(len(rows), sum(expected_by_task.values()))


class TestPromptContract(unittest.TestCase):
    def test_no_source_bloom_in_bloom_prompt(self) -> None:
        row = {
            "source_question": "What is mitosis?",
            "target_bloom_level": "Analyze",
            "source_bloom_level": "Remember",
        }
        prompt = build_generation_prompt("bloom_rewrite", row)
        assert_no_source_level_in_prompt(prompt)
        self.assertIn("Target Bloom level:\nAnalyze", prompt)
        # General instructions enumerate every Bloom level. Only a source-level
        # field is forbidden; the bare word can legitimately appear in guidance.
        self.assertNotIn("Source Bloom level:\nRemember", prompt)
        self.assertNotIn("Original Bloom level:\nRemember", prompt)


class TestMetrics(unittest.TestCase):
    def test_qa_em_f1(self) -> None:
        self.assertEqual(qa_exact_match("Paris", "paris"), 1.0)
        self.assertGreater(qa_token_f1("the capital Paris", "Paris"), 0.0)
        agg = aggregate_qa_metrics([("Paris", "paris"), ("no", "yes")])
        self.assertEqual(agg["n"], 2)

    def test_bloom_f1_shape(self) -> None:
        m = bloom_classification_metrics(
            ["Remember", "Understand"], ["Remember", "Apply"]
        )
        self.assertIn("macro_f1", m)
        self.assertIn("confusion_matrix", m)

    def test_source_target_matrix_cells(self) -> None:
        recs = [
            {
                "source_bloom_level": "Remember",
                "target_bloom_level": "Understand",
                "target_match": True,
                "fully_validated": True,
            }
        ]
        mat = source_target_matrix(recs)
        self.assertEqual(len(mat["cells"]), 30)


class TestCheckpointResolution(unittest.TestCase):
    def test_resolve_classifier_prefers_merged(self) -> None:
        merged = REPO_ROOT / "models" / "qwen_bloom_merged0.5B"
        if not (merged / "config.json").is_file():
            self.skipTest("merged Bloom classifier not present")
        resolved = resolve_classifier_dir(None, repo_root=REPO_ROOT)
        self.assertTrue(resolved.exists())
        self.assertIn("qwen_bloom", resolved.name)

    def test_sanitize_strips_unknown_peft_keys(self) -> None:
        try:
            import peft  # noqa: F401
        except ImportError:
            self.skipTest("peft not installed")
        raw = {
            "r": 16,
            "lora_alpha": 32,
            "peft_type": "LORA",
            "alora_invocation_tokens": None,
        }
        sanitized, stripped = _sanitize_lora_adapter_config(raw)
        self.assertIn("alora_invocation_tokens", stripped)
        self.assertNotIn("alora_invocation_tokens", sanitized)
        self.assertEqual(sanitized["r"], 16)

    def test_base_condition(self) -> None:
        cfg = {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": str(EXPERIMENT_DIR / "models" / "qwen05b_multitask_lora"),
        }
        info = resolve_checkpoint(cfg, "base")
        self.assertEqual(info.condition, "base")
        self.assertIsNone(info.adapter_path)

    def test_missing_lora_raises(self) -> None:
        cfg = {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": str(EXPERIMENT_DIR / "models" / "qwen05b_multitask_lora"),
        }
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(cfg, "lora")


class TestBloomValidation(unittest.TestCase):
    def test_validate_fields(self) -> None:
        v = validate_bloom_example(
            "Define photosynthesis.",
            "Understand",
            "Explain how photosynthesis converts light energy into chemical energy?",
        )
        self.assertTrue(v.format_valid)

    def test_recognizes_diverse_action_verb_imperatives(self) -> None:
        prompts = (
            "Relate the stages of mitosis to their roles.",
            "Judge which mitosis stage best explains the observed event.",
            "Clarify how mitosis produces two daughter cells.",
            "Verify the mitosis sequence against the described stages.",
            "Compose a diagram of the stages of mitosis.",
            "Trace the sequence of stages in mitosis.",
            "Check whether the cells completed mitosis.",
            "Draw a labeled diagram of mitosis.",
            "Validate the proposed account of mitosis.",
            "Write a brief account of mitosis.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(is_valid_imperative_exam(prompt))


if __name__ == "__main__":
    unittest.main()
