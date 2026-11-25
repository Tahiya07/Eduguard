"""Training-pipeline tests for closed-book academic QA (data/multitask_bloom_rewrite_separate_eduguard_v3_2).

No training, no model weights, no GPU: prompts, loader validation and config checks only.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
for item in (EXPERIMENT_DIR, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import dataset_io  # noqa: E402
import prompts  # noqa: E402
import validate_multitask_dataset as qc  # noqa: E402

DATA = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_2"
V31 = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_1"
QA_SRC = REPO_ROOT / "data" / "qa_eduguard_v2"
CONFIG = EXPERIMENT_DIR / "configs" / "qwen15b_multitask_eduguard_v3_2_20261126.json"
SPLITS = dataset_io.SPLITS
ROW = {"task": "qa", "qa_format": "closed_book", "question": "Why is the sky blue? ", "answer": " Rayleigh scattering. "}


class TestClosedBookPrompts(unittest.TestCase):
    def test_closed_book_rendering(self) -> None:
        sft = prompts.build_sft_text("qa", ROW)
        prompt = prompts.build_prompt_only_text("qa", ROW)
        gen = prompts.build_generation_prompt("qa", ROW)
        self.assertTrue(sft.startswith(prompt))
        self.assertTrue(sft.endswith("<|im_start|>assistant\nRayleigh scattering.<|im_end|>"))
        self.assertTrue(gen.startswith(prompt) and gen.endswith("<|im_start|>assistant\n"))
        self.assertIn("<|im_start|>user\nQuestion:\nWhy is the sky blue?<|im_end|>", prompt)
        self.assertIn(prompts.QA_CLOSED_BOOK_SYSTEM, prompt)
        self.assertNotIn("Context:", sft)
        self.assertNotIn("supplied context", sft)

    def test_unknown_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prompts.build_sft_text("qa", {**ROW, "qa_format": "open"})

    def test_extractive_rows_without_context_still_fail(self) -> None:
        with self.assertRaises(KeyError):
            prompts.build_sft_text("qa", {"question": "q", "answer": "a"})

    @unittest.skipUnless(V31.is_dir(), "v3.1 dataset not built")
    def test_extractive_prompts_byte_identical(self) -> None:
        for split in SPLITS:
            for row in dataset_io.read_jsonl(V31 / "qa" / f"{split}.jsonl"):
                self.assertEqual(prompts.build_sft_text("qa", row), row["sft_text"], row["id"])
                self.assertEqual(prompts.build_prompt_only_text("qa", row), row["prompt_text"], row["id"])
                self.assertEqual(prompts.build_generation_prompt("qa", row), row["generation_prompt"], row["id"])

    def test_qc_required_fields_by_format(self) -> None:
        self.assertNotIn("context", qc.required_fields("qa", ROW))
        self.assertIn("reference_answer", qc.required_fields("qa", ROW))
        self.assertIn("context", qc.required_fields("qa", {"task": "qa"}))


@unittest.skipUnless(DATA.is_dir(), "v3.2 dataset not built")
class TestEduGuardV32Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.integrity = dataset_io.validate_separate_dataset(DATA)
        cls.manifest = dataset_io.load_manifest(DATA)

    def test_counts_and_checks(self) -> None:
        expected_qa = {"train": 1172, "validation": 146, "test": 147}
        v31 = dataset_io.load_manifest(V31)["counts"]
        for split in SPLITS:
            self.assertEqual(self.integrity["counts"][split]["qa"], expected_qa[split])
            for task in ("bloom_rewrite", "summarization"):
                self.assertEqual(self.integrity["counts"][split][task], v31[split][task])
        self.assertEqual(self.integrity["qa_source_check"]["rows_checked"], 1465)
        self.assertEqual(self.integrity["bloom_source_check"]["contested_rows_excluded"], 10)

    def test_bloom_and_summarization_unchanged_from_v31(self) -> None:
        v31 = dataset_io.load_manifest(V31)["generated_files_sha256"]
        for name, digest in self.manifest["generated_files_sha256"].items():
            if not name.startswith("qa/"):
                self.assertEqual(digest, v31[name], name)

    def test_qa_rows_match_source(self) -> None:
        for split in SPLITS:
            src = {r["id"]: r for r in dataset_io.read_jsonl(QA_SRC / f"{split}.jsonl")}
            rows = dataset_io.load_task_split(DATA, "qa", split, validate=False)
            self.assertEqual({r["id"] for r in rows}, set(src))
            for r in rows:
                self.assertEqual(r["qa_format"], "closed_book")
                self.assertEqual(r["answer"], src[r["id"]]["reference_answer"])
                self.assertEqual(r["sft_text"], prompts.build_sft_text("qa", r))

    def test_training_loader_reads_all_tasks(self) -> None:
        rows, meta = dataset_io.load_split_rows(DATA, "train", validate=False)
        self.assertEqual(meta["split_counts_loaded"], self.manifest["counts"]["train"])
        self.assertTrue(all(r["sft_text"].startswith(r["prompt_text"]) for r in rows))

    def test_config_matches_dataset(self) -> None:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(cfg["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(cfg["dataset_dir"], "data/multitask_bloom_rewrite_separate_eduguard_v3_2")
        self.assertEqual(cfg["train_tasks"], list(dataset_io.TASKS))
        self.assertEqual(cfg["max_seq_length"], self.manifest["tokenizer"]["max_seq_length"])
        for split in ("validation", "test"):
            self.assertEqual(cfg[f"{split}_task_limits"], self.manifest["counts"][split])
        longest = max(v["max_answer_tokens"] for v in self.manifest["tokenizer"]["qa_check"]["qa"].values())
        self.assertGreaterEqual(cfg["generation"]["max_new_tokens_by_task"]["qa"], longest)


@unittest.skipUnless(DATA.is_dir(), "v3.2 dataset not built")
class TestEduGuardV32FailsClosed(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "ds"
        shutil.copytree(DATA, self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _rewrite(self, split: str, mutate, manifest_mutate=None) -> None:
        path = self.dir / "qa" / f"{split}.jsonl"
        rows = dataset_io.read_jsonl(path)
        mutate(rows)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest = dataset_io.load_manifest(self.dir)
        manifest["generated_files_sha256"][f"qa/{split}.jsonl"] = dataset_io.sha256_file(path)
        manifest["counts"][split]["qa"] = len(rows)
        if manifest_mutate:
            manifest_mutate(manifest)
        (self.dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_changed_answer_rejected(self) -> None:
        def edit(rows):
            rows[0]["answer"] = rows[0]["answer"] + " Extra."
        self._rewrite("train", edit)
        with self.assertRaisesRegex(ValueError, "answer differs"):
            dataset_io.validate_separate_dataset(self.dir)

    def test_reserve_row_rejected(self) -> None:
        reserve = dataset_io.read_jsonl(QA_SRC / "reserve.jsonl")[0]
        self._rewrite("train", lambda rows: rows.append({**rows[0], "id": reserve["id"]}))
        with self.assertRaisesRegex(ValueError, "reserve"):
            dataset_io.validate_separate_dataset(self.dir)

    def test_moved_row_rejected(self) -> None:
        moved = dataset_io.read_jsonl(self.dir / "qa" / "test.jsonl")[0]
        self._rewrite("validation", lambda rows: rows.append({**moved, "split": "validation"}))
        with self.assertRaises(ValueError):
            dataset_io.validate_separate_dataset(self.dir)

    def test_undeclared_closed_book_rows_rejected(self) -> None:
        self._rewrite("train", lambda rows: None, lambda m: m.pop("qa_source"))
        with self.assertRaisesRegex(ValueError, "closed-book qa rows"):
            dataset_io.validate_separate_dataset(self.dir)


if __name__ == "__main__":
    unittest.main()
