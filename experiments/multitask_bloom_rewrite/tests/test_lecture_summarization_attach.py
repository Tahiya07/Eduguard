"""Tests for attaching an owner-built lecture summarization dataset (attach_summarization_dataset.py).

No training, no model weights, no GPU. The end-to-end build copies the active dataset's bloom_rewrite/qa files into a
temporary folder under data/ and removes it afterwards.
"""
from __future__ import annotations

import json
import shutil
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

import attach_summarization_dataset as attach  # noqa: E402
import dataset_io  # noqa: E402
import paths  # noqa: E402
import prompts  # noqa: E402

SPLITS = dataset_io.SPLITS
TOPICS = {
    "train": ["normalization", "indexing", "transactions", "recursion"],
    "validation": ["photosynthesis", "elasticity"],
    "test": ["ohms law", "plate tectonics"],
}


def lecture_row(split: str, i: int, topic: str) -> dict:
    words = f"{topic} lecture section {i} explains core definitions methods worked examples and limitations " * 6
    return {"id": f"LEC-{split}-{i}", "department": "CSE", "course": "Course", "topic": topic,
            "source_type": "lecture_pdf", "review_status": "verified",
            "source_text": words.strip(), "summary": f"{topic} summary {i}: definitions, methods and limitations."}


def write_source(root: Path, rows: dict[str, list[dict]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split_rows), encoding="utf-8")


def good_rows() -> dict[str, list[dict]]:
    return {s: [lecture_row(s, i, t) for i, t in enumerate(ts)] for s, ts in TOPICS.items()}


class TestLecturePrompt(unittest.TestCase):
    def test_lecture_rendering(self) -> None:
        rec = {"summarization_format": "lecture", "article": " Lecture text. ", "abstract": " Notes. "}
        sft = prompts.build_sft_text("summarization", rec)
        prompt = prompts.build_prompt_only_text("summarization", rec)
        self.assertTrue(sft.startswith(prompt))
        self.assertIn(prompts.LECTURE_SUMMARIZATION_SYSTEM, prompt)
        self.assertIn("<|im_start|>user\nLecture material:\nLecture text.<|im_end|>", prompt)
        self.assertTrue(sft.endswith("<|im_start|>assistant\nNotes.<|im_end|>"))
        self.assertTrue(prompts.build_generation_prompt("summarization", rec).endswith("<|im_start|>assistant\n"))

    def test_article_format_unchanged(self) -> None:
        self.assertEqual(
            prompts.build_sft_text("summarization", {"article": " A. ", "abstract": " B. "}),
            f"<|im_start|>system\n{prompts.SUMMARIZATION_SYSTEM}<|im_end|>\n"
            "<|im_start|>user\nText:\nA.<|im_end|>\n<|im_start|>assistant\nB.<|im_end|>",
        )

    def test_unknown_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prompts.build_sft_text("summarization", {"summarization_format": "slides", "article": "a", "abstract": "b"})


class TestSourceChecks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assert_rejected(self, mutate, pattern: str, require: str = "verified") -> None:
        rows = good_rows()
        mutate(rows)
        write_source(self.root, rows)
        with self.assertRaises(SystemExit) as ctx:
            attach.check_source(self.root, require, 0.8)
        self.assertRegex(str(ctx.exception), pattern)

    def test_good_source_accepted(self) -> None:
        write_source(self.root, good_rows())
        rows, near = attach.check_source(self.root, "verified", 0.8)
        self.assertEqual({s: len(r) for s, r in rows.items()}, {s: len(t) for s, t in TOPICS.items()})

    def test_unverified_row(self) -> None:
        self.assert_rejected(lambda r: r["test"][0].update(review_status="draft"), "review_status")

    def test_unverified_allowed_when_disabled(self) -> None:
        rows = good_rows()
        rows["test"][0]["review_status"] = "draft"
        write_source(self.root, rows)
        attach.check_source(self.root, "", 0.8)

    def test_empty_summary(self) -> None:
        self.assert_rejected(lambda r: r["train"][0].update(summary="  "), "empty 'summary'")

    def test_duplicate_id_across_splits(self) -> None:
        self.assert_rejected(lambda r: r["test"][0].update(id=r["train"][0]["id"]), "already used")

    def test_duplicate_source_across_splits(self) -> None:
        self.assert_rejected(lambda r: r["test"][0].update(source_text=r["train"][0]["source_text"]),
                             "duplicate source_text")

    def test_summary_not_shorter(self) -> None:
        self.assert_rejected(lambda r: r["train"][1].update(summary=r["train"][1]["source_text"] + " more"),
                             "not shorter")

    def test_reserved_field(self) -> None:
        self.assert_rejected(lambda r: r["train"][0].update(abstract="x"), "reserved")

    def test_wrong_split_field(self) -> None:
        self.assert_rejected(lambda r: r["validation"][0].update(split="train"), "does not match")


@unittest.skipUnless(paths.TASK_SEPARATED_DATA_DIR.is_dir(), "active dataset not built")
class TestAttachEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(dir=REPO_ROOT / "data", prefix="_test_attach_")
        cls.tmp = Path(cls._tmp.name)
        cls.src, cls.out = cls.tmp / "lecture_src", cls.tmp / "dataset"
        write_source(cls.src, good_rows())
        argv = ["attach", "--source", str(cls.src), "--out", str(cls.out)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            attach.main()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def copy(self) -> Path:
        dst = self.tmp / f"copy_{self._testMethodName}"
        shutil.copytree(self.out, dst)
        return dst

    def test_dataset_validates_with_only_lecture_summaries(self) -> None:
        integrity = dataset_io.validate_separate_dataset(self.out)
        self.assertEqual(integrity["summarization_source_check"]["rows_checked"], 8)
        base = dataset_io.load_manifest(paths.TASK_SEPARATED_DATA_DIR)
        for split in SPLITS:
            self.assertEqual(integrity["counts"][split]["summarization"], len(TOPICS[split]))
            for task in ("bloom_rewrite", "qa"):
                self.assertEqual(integrity["counts"][split][task], base["counts"][split][task])
            rows, _ = dataset_io.load_split_rows(self.out, split, tasks=("summarization",), validate=False)
            self.assertTrue(all(r["summarization_format"] == "lecture" for r in rows))
        manifest = dataset_io.load_manifest(self.out)
        self.assertNotIn("summarization_quality_filter", manifest)
        self.assertFalse((self.out / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE).exists())

    def test_edited_row_rejected_even_without_validation(self) -> None:
        d = self.copy()
        path = d / "summarization" / "train.jsonl"
        rows = dataset_io.read_jsonl(path)
        rows[0]["abstract"] = rows[0]["summary"] = "Different summary."
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifest = dataset_io.load_manifest(d)
        manifest["generated_files_sha256"]["summarization/train.jsonl"] = dataset_io.sha256_file(path)
        (d / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(dataset_io.SummarizationEligibilityError):
            dataset_io.load_split_rows(d, "train", tasks=("summarization",), validate=False)
        with self.assertRaises(ValueError):
            dataset_io.validate_separate_dataset(d)

    def test_undeclared_lecture_rows_rejected(self) -> None:
        d = self.copy()
        manifest = dataset_io.load_manifest(d)
        manifest.pop("summarization_source")
        (d / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(dataset_io.SummarizationEligibilityError):
            dataset_io.load_split_rows(d, "validation", tasks=("summarization",), validate=False)

    def test_changed_source_file_rejected(self) -> None:
        d = self.copy()
        manifest = dataset_io.load_manifest(d)
        src_copy = self.tmp / "src_copy"
        shutil.copytree(self.src, src_copy)
        (src_copy / "test.jsonl").write_text((src_copy / "test.jsonl").read_text(encoding="utf-8") + "\n",
                                            encoding="utf-8")
        manifest["summarization_source"]["path"] = src_copy.relative_to(REPO_ROOT).as_posix()
        (d / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            dataset_io.validate_separate_dataset(d)

    def test_refuses_source_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            argv = ["attach", "--source", outside, "--out", str(self.tmp / "never")]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "inside the repository"):
                attach.main()


if __name__ == "__main__":
    unittest.main()
