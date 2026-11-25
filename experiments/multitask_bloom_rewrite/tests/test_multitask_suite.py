"""Tests for multi-task Bloom rewrite experiment (stdlib unittest)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import validate_bloom_example  # noqa: E402
from loss_masking import assert_assistant_only_loss, mask_prompt_labels  # noqa: E402
from paths import (  # noqa: E402
    ACTIVE_TRAINING_CONFIG, CONFIG_DIR, REPO_ROOT as PATHS_REPO_ROOT, TASK_SEPARATED_DATA_DIR,
    TOPIC_SIMILARITY_THRESHOLD,
)
from prompts import (  # noqa: E402
    FORBIDDEN_SOURCE_LEVEL_MARKERS,
    assert_no_source_level_in_prompt,
    bloom_messages,
    build_generation_prompt,
    build_sft_text,
    render_chatml,
)


class TestPromptContract(unittest.TestCase):
    def test_no_source_bloom_in_prompt(self) -> None:
        text = build_sft_text(
            "bloom_rewrite",
            {
                "source_question": "Define mitosis.",
                "target_bloom_level": "Analyze",
                "target_rewrite": "Compare the stages of mitosis and meiosis.",
            },
        )
        assert_no_source_level_in_prompt(text)
        for marker in FORBIDDEN_SOURCE_LEVEL_MARKERS:
            self.assertNotIn(marker.lower(), text.lower())
        self.assertIn("Target Bloom level:", text)
        self.assertIn("Analyze", text)

    def test_target_present_generation(self) -> None:
        prompt = build_generation_prompt(
            "bloom_rewrite",
            {"source_question": "What is osmosis?", "target_bloom_level": "Evaluate"},
        )
        self.assertIn("Target Bloom level:\nEvaluate", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))


    def test_each_target_has_explicit_action_verb_guidance(self) -> None:
        expected = {
            "Remember": "Identify, Define, List",
            "Understand": "Explain, Describe, Summarize",
            "Apply": "Apply, Use, Calculate",
            "Analyze": "Analyze, Compare, Contrast",
            "Evaluate": "Evaluate, Assess, Justify",
            "Create": "Design, Develop, Create",
        }
        for level, verbs in expected.items():
            with self.subTest(level=level):
                prompt = build_generation_prompt(
                    "bloom_rewrite",
                    {"source_question": "Explain virtual memory.", "target_bloom_level": level},
                )
                self.assertIn(level, prompt)
                self.assertIn(verbs, prompt)
                # The current contract states this as "the task the student must perform";
                # retain the semantic check without coupling it to the retired wording.
                self.assertIn("task the student must perform", prompt)
                self.assertIn("OUTPUT ONLY ONE", prompt)

    def test_unknown_target_level_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_generation_prompt(
                "bloom_rewrite",
                {"source_question": "Explain virtual memory.", "target_bloom_level": "Invent"},
            )

class TestBloomTransformations(unittest.TestCase):
    def test_c1_c6_and_non_self(self) -> None:
        levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        self.assertEqual(len(levels), 6)
        pairs = [(a, b) for a in levels for b in levels if a != b]
        self.assertEqual(len(pairs), 30)

    def test_validate_bloom_example_fields(self) -> None:
        result = validate_bloom_example(
            "Define photosynthesis.",
            "Understand",
            "Explain how photosynthesis converts light energy into chemical energy.",
        )
        for field in (
            "format_valid",
            "semantic_valid",
            "cognitive_valid",
            "trivial_transform",
            "topic_preserved",
            "accepted",
            "rejection_reason",
        ):
            self.assertTrue(hasattr(result, field))
        self.assertEqual(TOPIC_SIMILARITY_THRESHOLD, 0.20)


class TestLossMasking(unittest.TestCase):
    def test_assistant_only_loss(self) -> None:
        ids = list(range(10))
        labels = mask_prompt_labels(ids, 6)
        self.assertEqual(labels[:6], [-100] * 6)
        self.assertEqual(labels[6:], [6, 7, 8, 9])
        assert_assistant_only_loss(labels, 6)

    def test_message_ordering(self) -> None:
        messages = bloom_messages("Q?", "Apply", "Apply X in scenario Y?")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        text = render_chatml(messages)
        self.assertLess(text.find("system"), text.find("user"))
        self.assertLess(text.find("user"), text.find("assistant"))


class TestTrainingConfig(unittest.TestCase):
    def test_active_separate_config_uses_prepared_dataset(self) -> None:
        config = json.loads(
            (CONFIG_DIR / ACTIVE_TRAINING_CONFIG).read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (TASK_SEPARATED_DATA_DIR / "dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(config["dataset_dir"], TASK_SEPARATED_DATA_DIR.relative_to(PATHS_REPO_ROOT).as_posix())
        self.assertEqual(set(config["train_tasks"]), set(manifest["tasks"]))
        for split in ("validation", "test"):
            self.assertEqual(
                config[f"{split}_task_limits"],
                manifest["counts"][split],
            )

    def test_decision_rule_loads(self) -> None:
        rule = json.loads((CONFIG_DIR / "decision_rule.json").read_text(encoding="utf-8"))
        self.assertTrue(rule["frozen_before_test_evaluation"])
        self.assertIn("INCONCLUSIVE", rule["allowed_recommendations"])


class TestLeakageAndDataset(unittest.TestCase):
    def test_splits_disjoint_if_prepared(self) -> None:
        paths = {
            (task, split): TASK_SEPARATED_DATA_DIR / task / f"{split}.jsonl"
            for task in ("bloom_rewrite", "qa", "summarization")
            for split in ("train", "validation", "test")
        }
        if not all(path.exists() for path in paths.values()):
            self.skipTest("multitask dataset not prepared yet")
        ids = {}
        for split in ("train", "validation", "test"):
            s = set()
            for task in ("bloom_rewrite", "qa", "summarization"):
                with paths[(task, split)].open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row["task"] == "bloom_rewrite":
                            key = str(
                                row.get("group_id")
                                or row.get("source_id")
                                or row["source_question"]
                            ).lower()
                        else:
                            key = str(row.get("source_id") or row["id"])
                        s.add((row["task"], key))
            ids[split] = s
        self.assertEqual(len(ids["train"] & ids["validation"]), 0)
        self.assertEqual(len(ids["train"] & ids["test"]), 0)
        self.assertEqual(len(ids["validation"] & ids["test"]), 0)

    def test_task_labels(self) -> None:
        paths = [
            TASK_SEPARATED_DATA_DIR / task / "train.jsonl"
            for task in ("bloom_rewrite", "qa", "summarization")
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("multitask dataset not prepared yet")
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if i > 200:
                        break
                    row = json.loads(line)
                    self.assertEqual(row["task"], path.parent.name)


class TestEvaluationMetrics(unittest.TestCase):
    def test_em_f1_helpers(self) -> None:
        from collections import Counter

        def normalize(s: str) -> str:
            return " ".join((s or "").lower().split())

        def exact_match(pred: str, gold: str) -> float:
            return float(normalize(pred) == normalize(gold))

        def f1(pred: str, gold: str) -> float:
            pt, gt = normalize(pred).split(), normalize(gold).split()
            if not pt and not gt:
                return 1.0
            if not pt or not gt:
                return 0.0
            common = Counter(pt) & Counter(gt)
            num = sum(common.values())
            if num == 0:
                return 0.0
            precision = num / len(pt)
            recall = num / len(gt)
            return 2 * precision * recall / (precision + recall)

        self.assertEqual(exact_match("Paris", "paris"), 1.0)
        self.assertGreater(f1("the french capital paris", "paris"), 0.0)


class TestGgufPipeline(unittest.TestCase):
    def test_production_path_not_experiment_default(self) -> None:
        prod = (REPO_ROOT / "models" / "qwen.gguf").resolve()
        exp05 = (EXPERIMENT_DIR / "models" / "qwen05b_multitask.gguf").resolve()
        exp15 = (EXPERIMENT_DIR / "models" / "qwen15b_multitask.gguf").resolve()
        self.assertNotEqual(prod, exp05)
        self.assertNotEqual(prod, exp15)


if __name__ == "__main__":
    unittest.main()
