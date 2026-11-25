"""Loader tests for the EduGuard v3.1 Bloom splits integrated into the task-separated SFT layout.

No training, no model weights, no GPU: these tests only read JSONL/manifest files through dataset_io.
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

DATA = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_1"
SOURCE = REPO_ROOT / "data" / "bloom_rewrite_eduguard_v3_1"
BASE = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_sumclean_v1"
REFERENCE_SPLITS = REPO_ROOT / "final_corrected_dataset" / "training_ready" / "tests" / "reference_split_ids.json"
SPLITS = dataset_io.SPLITS
EXPECTED_BLOOM = {"train": 2345, "validation": 295, "test": 280}
CONTESTED_EXAMPLES = {"213", "570"}


@unittest.skipUnless(DATA.is_dir(), "EduGuard v3.1 multitask dataset not built")
class TestEduGuardV31Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.integrity = dataset_io.validate_separate_dataset(DATA)
        cls.manifest = dataset_io.load_manifest(DATA)
        cls.rows = {s: dataset_io.load_task_split(DATA, "bloom_rewrite", s, validate=False) for s in SPLITS}
        cls.source = {s: dataset_io.read_jsonl(SOURCE / f"{s}.jsonl") for s in SPLITS}

    def test_counts(self) -> None:
        base = dataset_io.load_manifest(BASE)["counts"]
        for s in SPLITS:
            self.assertEqual(self.integrity["counts"][s]["bloom_rewrite"], EXPECTED_BLOOM[s])
            self.assertEqual(len(self.rows[s]), EXPECTED_BLOOM[s])
            for task in ("qa", "summarization"):
                self.assertEqual(self.integrity["counts"][s][task], base[s][task])
        self.assertEqual(self.integrity["bloom_source_check"]["rows_checked"], sum(EXPECTED_BLOOM.values()))

    def test_hashes_match_manifest(self) -> None:
        for name, digest in self.manifest["generated_files_sha256"].items():
            self.assertEqual(dataset_io.sha256_file(DATA / name), digest, name)
        for name, digest in self.manifest["bloom_source"]["files_sha256"].items():
            self.assertEqual(dataset_io.sha256_file(SOURCE / name), digest, name)

    def test_qa_and_summarization_identical_to_base(self) -> None:
        base = dataset_io.load_manifest(BASE)["generated_files_sha256"]
        for task in ("qa", "summarization"):
            for s in SPLITS:
                name = f"{task}/{s}.jsonl"
                self.assertEqual(self.manifest["generated_files_sha256"][name], base[name])
        self.assertEqual(dataset_io.sha256_file(DATA / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE),
                         dataset_io.sha256_file(BASE / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE))

    def test_originals_unchanged(self) -> None:
        base_integrity = dataset_io.validate_separate_dataset(BASE)
        self.assertEqual(base_integrity["manifest_sha256"], self.manifest["derived_from"]["manifest_sha256"])
        src_manifest = json.loads((SOURCE / "dataset_manifest.json").read_text(encoding="utf-8"))
        for name, digest in src_manifest["files_sha256"].items():
            self.assertEqual(dataset_io.sha256_file(SOURCE / name), digest)

    def test_all_contested_rows_excluded(self) -> None:
        contested = {r["id"] for s in SPLITS for r in self.source[s] if r["gold_label_contested"]}
        self.assertEqual(len(contested), 10)
        self.assertEqual({r["source_id"] for s in SPLITS for r in self.source[s] if r["gold_label_contested"]}, CONTESTED_EXAMPLES)
        self.assertEqual({e["id"] for e in self.manifest["bloom_exclusions"]["excluded_rows"]}, contested)
        for s in SPLITS:
            self.assertFalse({r["id"] for r in self.rows[s]} & contested)
            self.assertFalse({r["source_id"] for r in self.rows[s]} & CONTESTED_EXAMPLES)
            rows, _ = dataset_io.load_split_rows(DATA, s, tasks=("bloom_rewrite",), validate=False)
            self.assertFalse(any(r.get("gold_label_contested") for r in rows))

    def test_ids_labels_and_splits_preserved(self) -> None:
        for s in SPLITS:
            src = {r["id"]: r for r in self.source[s] if not r["gold_label_contested"]}
            self.assertEqual({r["id"] for r in self.rows[s]}, set(src))
            for r in self.rows[s]:
                for field in ("source_id", "group_id", "split", "source_bloom_level", "target_bloom_level",
                              "source_question", "target_rewrite"):
                    self.assertEqual(r[field], src[r["id"]][field], (r["id"], field))
        ref = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))["example_ids"]
        for s in SPLITS:
            self.assertEqual({r["source_id"] for r in self.rows[s]}, set(ref[s]) - CONTESTED_EXAMPLES)

    def test_split_isolation(self) -> None:
        for key in ("id", "source_id", "group_id", "source_question"):
            sets = {s: {str(r[key]) for r in self.rows[s]} for s in SPLITS}
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(sets[a] & sets[b], (key, a, b))

    def test_sft_format(self) -> None:
        for s in SPLITS:
            for r in self.rows[s]:
                self.assertTrue(r["sft_text"].startswith(r["prompt_text"]), r["id"])
                self.assertTrue(r["sft_text"].endswith(r["target_rewrite"].strip() + "<|im_end|>"), r["id"])
                self.assertTrue(r["generation_prompt"].endswith("<|im_start|>assistant\n"), r["id"])
                self.assertIn(f"Target Bloom level:\n{r['target_bloom_level']}", r["prompt_text"])
                self.assertNotIn("source bloom level", r["prompt_text"].lower())

    def test_tokenizer_check_used_1_5b_model(self) -> None:
        self.assertEqual(self.manifest["tokenizer"]["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        check = self.manifest["tokenizer"]["bloom_rewrite_check"]
        self.assertTrue(check["performed"])
        self.assertTrue(all(check["bloom_rewrite"][s]["overlength"] == 0 for s in SPLITS))


@unittest.skipUnless(DATA.is_dir(), "EduGuard v3.1 multitask dataset not built")
class TestEduGuardV31FailsClosed(unittest.TestCase):
    """Tampering with a temporary copy must be rejected by the loader."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "ds"
        shutil.copytree(DATA, self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _rewrite(self, split: str, mutate) -> None:
        path = self.dir / "bloom_rewrite" / f"{split}.jsonl"
        rows = dataset_io.read_jsonl(path)
        mutate(rows)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest = dataset_io.load_manifest(self.dir)
        manifest["generated_files_sha256"][f"bloom_rewrite/{split}.jsonl"] = dataset_io.sha256_file(path)
        manifest["counts"][split]["bloom_rewrite"] = len(rows)
        (self.dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_injected_contested_row_rejected(self) -> None:
        contested = next(r for r in dataset_io.read_jsonl(SOURCE / "test.jsonl") if r["gold_label_contested"])
        self._rewrite("test", lambda rows: rows.append(dict(rows[0], id=contested["id"], gold_label_contested=True)))
        with self.assertRaises(dataset_io.ContestedBloomLabelError):
            dataset_io.validate_separate_dataset(self.dir)
        with self.assertRaises(dataset_io.ContestedBloomLabelError):
            dataset_io.load_split_rows(self.dir, "test", tasks=("bloom_rewrite",), validate=False)
        with self.assertRaises(dataset_io.ContestedBloomLabelError):
            dataset_io.load_task_split(self.dir, "bloom_rewrite", "test", validate=False)

    def test_changed_target_label_rejected(self) -> None:
        def flip(rows):
            rows[0]["target_bloom_level"] = "Create" if rows[0]["target_bloom_level"] != "Create" else "Remember"
        self._rewrite("train", flip)
        with self.assertRaisesRegex(ValueError, "target_bloom_level differs"):
            dataset_io.validate_separate_dataset(self.dir)

    def test_moved_row_rejected(self) -> None:
        moved = dataset_io.read_jsonl(self.dir / "bloom_rewrite" / "validation.jsonl")[0]
        self._rewrite("train", lambda rows: rows.append(dict(moved, split="train")))
        with self.assertRaises(ValueError):
            dataset_io.validate_separate_dataset(self.dir)

    def test_dropped_exclusion_record_rejected(self) -> None:
        manifest = dataset_io.load_manifest(self.dir)
        manifest["bloom_exclusions"]["excluded_rows"] = manifest["bloom_exclusions"]["excluded_rows"][1:]
        (self.dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bloom_exclusions"):
            dataset_io.validate_separate_dataset(self.dir)

    def test_edited_qa_file_rejected(self) -> None:
        (self.dir / "qa" / "train.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            dataset_io.validate_separate_dataset(self.dir)


if __name__ == "__main__":
    unittest.main()
