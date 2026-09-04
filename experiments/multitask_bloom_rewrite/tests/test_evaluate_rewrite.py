"""Unit tests for multitask evaluation pipeline (no model weights required)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import validate_bloom_example  # noqa: E402
from eval_dataset import (  # noqa: E402
    EXPECTED_TEST_TOTAL,
    assert_split_is_test,
    assert_test_counts,
    load_test_split,
)
from eval_metrics import (  # noqa: E402
    aggregate_qa_metrics,
    bloom_classification_metrics,
    qa_exact_match,
    qa_token_f1,
    source_target_matrix,
)
from eval_classifier import resolve_classifier_dir  # noqa: E402
from eval_model import resolve_checkpoint, _sanitize_lora_adapter_config  # noqa: E402
from paths import MULTITASK_DATA_DIR, REPO_ROOT  # noqa: E402
from prompts import assert_no_source_level_in_prompt, build_generation_prompt  # noqa: E402


class TestDatasetAssertions(unittest.TestCase):
    def test_rejects_validation_file_name(self) -> None:
        rows = [{"task": "qa", "split": "validation", "example_id": "x"}]
        with self.assertRaises(AssertionError):
            assert_split_is_test(rows, path=MULTITASK_DATA_DIR / "validation.jsonl")

    def test_test_split_loads_if_present(self) -> None:
        if not (MULTITASK_DATA_DIR / "test.jsonl").exists():
            self.skipTest("test.jsonl not prepared")
        rows, meta = load_test_split(MULTITASK_DATA_DIR)
        self.assertEqual(meta["counts"]["test_count"], EXPECTED_TEST_TOTAL)
        self.assertEqual(len(rows), EXPECTED_TEST_TOTAL)

    def test_wrong_count_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_test_counts([{"task": "qa"}] * 10)


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
        self.assertNotIn("Remember", prompt.split("Target Bloom level:")[0])


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


if __name__ == "__main__":
    unittest.main()
