"""Regression tests for the physically task-separated SFT dataset pipeline.

These tests use a tiny temporary fixture and do not download datasets or load model
weights. They verify storage layout and loader behavior independently of training.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
for item in (EXPERIMENT_DIR, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import dataset_io  # noqa: E402
from paths import CONFIG_DIR, TASK_SEPARATED_DATA_DIR  # noqa: E402


TASKS = ("bloom_rewrite", "qa", "summarization")
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_row(task: str, split: str) -> dict:
    row = {
        "task": task,
        "split": split,
        "id": f"{task}_{split}_001",
        "sft_text": f"prompt and answer for {task} {split}",
        "prompt_text": f"prompt for {task} {split}",
        "generation_prompt": f"generation prompt for {task} {split}",
    }
    if task == "bloom_rewrite":
        row.update({
            "source_id": f"source_{split}",
            "group_id": f"group_{split}",
            "source_question": f"Define the concept for {split}.",
            "target_bloom_level": "Understand",
            "target_rewrite": f"Explain the concept using the {split} example.",
        })
    elif task == "qa":
        row.update({
            "context": f"Unique context for {split}. The answer is blue.",
            "question": f"What is the answer in {split}?",
            "answer": "blue",
            "answer_start": len(f"Unique context for {split}. The answer is "),
        })
    else:
        row.update({
            "article": f"Unique article text for {split}.",
            "abstract": f"Summary for {split}.",
        })
    return row


class TestTaskSeparatedDatasetIO(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.data_dir = self.root / "data" / "multitask_bloom_rewrite_separate"
        self.source_dir = self.root / "synthetic_bloomshift"
        self.source_dir.mkdir(parents=True)
        for name in ("train.json", "validation.json", "test.json"):
            (self.source_dir / name).write_text("[]\n", encoding="utf-8")
        (self.source_dir / "batch_review_attestation.json").write_text(
            "{}\n", encoding="utf-8"
        )

        file_hashes = {}
        counts = {}
        for split in SPLITS:
            counts[split] = {}
            for task in TASKS:
                path = self.data_dir / task / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(_make_row(task, split), ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                file_hashes[path.relative_to(self.data_dir).as_posix()] = _sha256(path)
                counts[split][task] = 1

        source_hashes = {
            name: _sha256(self.source_dir / name)
            for name in (
                "train.json", "validation.json", "test.json",
                "batch_review_attestation.json",
            )
        }
        manifest = {
            "version": dataset_io.MANIFEST_VERSION,
            "storage_layout": "one_directory_per_task",
            "tasks": list(TASKS),
            "counts": counts,
            "generated_files_sha256": file_hashes,
            "bloom_source": {
                "path": "synthetic_bloomshift",
                "files_sha256": source_hashes,
                "heldout_annotation_provenance": {
                    "inclusion_basis": "synthetic_test_fixture"
                },
            },
        }
        (self.data_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_loader_selects_only_requested_task_and_writes_no_merged_split(self) -> None:
        with patch.object(dataset_io, "REPO_ROOT", self.root):
            integrity = dataset_io.validate_separate_dataset(self.data_dir, self.root)
            rows, meta = dataset_io.load_split_rows(
                self.data_dir,
                "train",
                tasks=("bloom_rewrite",),
                validate=True,
            )

        self.assertEqual(integrity["layout"], "task-separated")
        self.assertEqual([row["task"] for row in rows], ["bloom_rewrite"])
        self.assertEqual(meta["loaded_tasks"], ["bloom_rewrite"])
        self.assertTrue(meta["combined_in_memory_only"])
        for split in SPLITS:
            self.assertFalse((self.data_dir / f"{split}.jsonl").exists())

    def test_loader_can_read_all_tasks_in_memory_without_persisting_them(self) -> None:
        with patch.object(dataset_io, "REPO_ROOT", self.root):
            rows, meta = dataset_io.load_split_rows(
                self.data_dir, "validation", validate=True
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["task"] for row in rows}, set(TASKS))
        self.assertTrue(meta["combined_in_memory_only"])
        self.assertEqual(meta["split_counts_loaded"], {task: 1 for task in TASKS})
        self.assertFalse((self.data_dir / "validation.jsonl").exists())

    def test_rejects_legacy_merged_root_level_files(self) -> None:
        (self.data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
        with patch.object(dataset_io, "REPO_ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "root-level merged split files"):
                dataset_io.validate_separate_dataset(self.data_dir, self.root)

    def test_rejects_hash_mismatch_in_a_task_file(self) -> None:
        path = self.data_dir / "qa" / "train.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        with patch.object(dataset_io, "REPO_ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                dataset_io.validate_separate_dataset(self.data_dir, self.root)


class TestTaskSeparatedExperimentConfigs(unittest.TestCase):
    def test_a_b_and_c_use_the_separate_data_root(self) -> None:
        expected_dir = str(TASK_SEPARATED_DATA_DIR.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        )
        for filename in (
            "experiment_A_base_prompt_only.json",
            "experiment_B_bloom_only.json",
            "qwen15b_multitask_final.json",
        ):
            config = json.loads(
                (CONFIG_DIR / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(config["dataset_dir"].replace("\\", "/"), expected_dir)

    def test_b_reads_only_bloom_and_c_reads_all_three_tasks(self) -> None:
        bloom_only = json.loads(
            (CONFIG_DIR / "experiment_B_bloom_only.json").read_text(encoding="utf-8")
        )
        multitask = json.loads(
            (CONFIG_DIR / "qwen15b_multitask_final.json").read_text(encoding="utf-8")
        )
        self.assertEqual(bloom_only["train_tasks"], ["bloom_rewrite"])
        self.assertEqual(set(multitask["train_tasks"]), set(TASKS))


if __name__ == "__main__":
    unittest.main()
