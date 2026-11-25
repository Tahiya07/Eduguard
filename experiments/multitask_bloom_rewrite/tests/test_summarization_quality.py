"""Tests for the arXiv summarization quality filter, its export, and loader enforcement (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, EXPERIMENT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dataset_io  # noqa: E402
import summarization_quality as Q  # noqa: E402
from build_clean_summarization_dataset import build  # noqa: E402

ORIGINAL = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"
CLEAN = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_sumclean_v1"
SPLITS = ("train", "validation", "test")
BLOOM_MASTER = REPO_ROOT / "final_corrected_dataset"
# SHA-256 of the original summarization files, recorded before the 2026-11-25 cleaning work.
ORIGINAL_SUMMARIZATION_SHA256 = {
    "train": "a26e40e60809476d",
    "validation": "eb42c05685b7fdd8",
    "test": "9337e14d5df20a61",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


CLEAN_ABSTRACT = ("we study the stability of a driven oscillator and show that the response saturates at large "
                  "amplitude , in agreement with numerical simulations .")
ARTICLE = " ".join(["the oscillator is driven periodically and its response is measured ."] * 40)


def make_row(abstract=CLEAN_ABSTRACT, article=ARTICLE, rid="x1", split="train"):
    prompt = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nText:\n" + article + "<|im_end|>"
    return {"task": "summarization", "split": split, "id": rid, "article": article, "abstract": abstract,
            "prompt_text": prompt, "sft_text": prompt + "\n<|im_start|>assistant\n" + abstract + "<|im_end|>"}


class TestDetection(unittest.TestCase):
    def test_clean_row_is_eligible(self):
        self.assertEqual(Q.detect_issues(make_row()), [])

    def test_math_placeholder_detected(self):
        row = make_row(abstract=CLEAN_ABSTRACT.replace("large", "@xmath0 large").replace("show", "show that @xmath12"))
        self.assertIn("abstract_math_placeholder", Q.detect_issues(row))
        self.assertEqual(Q.math_placeholders(row["abstract"]), ["@xmath12", "@xmath0"])

    def test_placeholder_in_supervised_target_only_is_detected(self):
        row = make_row()
        row["sft_text"] = row["sft_text"].replace("large", "@xmath3")
        self.assertIn("abstract_math_placeholder", Q.detect_issues(row))

    def test_other_artefacts_detected(self):
        cases = {
            "abstract_other_conversion_placeholder": CLEAN_ABSTRACT.replace("simulations", "simulations @xcite"),
            "abstract_latex_or_markup_residue": CLEAN_ABSTRACT.replace("large", "\\rm large"),
            "abstract_metadata_appended": CLEAN_ABSTRACT + " pacs numbers : 05.45.-a .",
            "abstract_bibliography_reference": CLEAN_ABSTRACT.replace("simulations", "simulations [ 8 ]"),
            "abstract_incomplete_ending": CLEAN_ABSTRACT + " \n chaotic scattering , billiards , hill s problem",
            "abstract_too_short": "we study oscillators .",
        }
        for rule, abstract in cases.items():
            with self.subTest(rule=rule):
                self.assertIn(rule, Q.detect_issues(make_row(abstract=abstract)))

    def test_input_side_rules(self):
        self.assertIn("article_not_paper_body", Q.detect_issues(make_row(article="we thank the referee .")))
        dense = " ".join(["the value @xmath1 equals @xmath2 here ."] * 80)
        self.assertIn("article_math_placeholder_density", Q.detect_issues(make_row(article=dense)))
        self.assertIn("abstract_copied_in_article", Q.detect_issues(make_row(article=ARTICLE + " " + CLEAN_ABSTRACT)))

    def test_sparse_article_placeholders_do_not_exclude(self):
        article = ARTICLE + " the frequency is @xmath0 ."
        self.assertEqual(Q.detect_issues(make_row(article=article)), [])


class TestVerifiedRecovery(unittest.TestCase):
    def setUp(self):
        self.row = make_row(abstract="the energy scales as @xmath1 and the rate as @xmath10 , as shown numerically here .")

    def entry(self, **kw):
        e = {"id": "x1", "split": "train", "field": "abstract", "field_sha256": Q.sha256_text(self.row["abstract"]),
             "placeholder": "@xmath1", "expression": "$E \\propto L^{2}$", "provenance": "test fixture"}
        e.update(kw)
        return e

    def test_verified_entry_replaces_whole_token_only(self):
        new, applied = Q.apply_verified_recoveries(self.row, [self.entry()])
        self.assertIn("$E \\propto L^{2}$ and", new["abstract"])
        self.assertIn("@xmath10", new["abstract"])            # @xmath1 must not match inside @xmath10
        self.assertEqual(len(applied), 1)
        self.assertEqual(Q.recovery_status(new, applied), "unrecoverable_no_source_expression")  # @xmath10 remains
        both, applied2 = Q.apply_verified_recoveries(self.row, [self.entry(), self.entry(placeholder="@xmath10", expression="$\\Gamma$")])
        self.assertEqual(Q.recovery_status(both, applied2), "recovered_verified")
        self.assertEqual(self.row["abstract"].count("@xmath"), 2)  # input row untouched

    def test_mismatched_hash_is_rejected(self):
        with self.assertRaises(Q.RecoveryError):
            Q.apply_verified_recoveries(self.row, [self.entry(field_sha256="0" * 64)])

    def test_absent_placeholder_or_missing_provenance_is_rejected(self):
        with self.assertRaises(Q.RecoveryError):
            Q.apply_verified_recoveries(self.row, [self.entry(placeholder="@xmath7")])
        with self.assertRaises(Q.RecoveryError):
            Q.apply_verified_recoveries(self.row, [self.entry(provenance="")])

    def test_entries_for_other_rows_are_ignored(self):
        new, applied = Q.apply_verified_recoveries(self.row, [self.entry(id="other")])
        self.assertIs(new, self.row)
        self.assertEqual(applied, [])


@unittest.skipUnless(CLEAN.is_dir() and ORIGINAL.is_dir(), "datasets not present")
class TestCleanDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elig = json.loads((CLEAN / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE).read_text(encoding="utf-8"))
        cls.orig = {s: read_jsonl(ORIGINAL / "summarization" / f"{s}.jsonl") for s in SPLITS}
        cls.clean = {s: read_jsonl(CLEAN / "summarization" / f"{s}.jsonl") for s in SPLITS}

    def test_original_files_unchanged(self):
        for s in SPLITS:
            h = sha256_file(ORIGINAL / "summarization" / f"{s}.jsonl")
            self.assertTrue(h.startswith(ORIGINAL_SUMMARIZATION_SHA256[s]), s)
            self.assertEqual(h, self.elig["original_summarization_files_sha256"][f"summarization/{s}.jsonl"])
        manifest = json.loads((ORIGINAL / "dataset_manifest.json").read_text(encoding="utf-8"))
        for rel, h in manifest["generated_files_sha256"].items():
            self.assertEqual(sha256_file(ORIGINAL / rel), h, rel)

    def test_every_row_accounted_for_with_recovery_status(self):
        for s in SPLITS:
            rows = [r for r in self.elig["rows"] if r["split"] == s]
            self.assertEqual([r["id"] for r in rows], [r["id"] for r in self.orig[s]])
            for r, o in zip(rows, self.orig[s]):
                has_ph = bool(Q.math_placeholders(o["abstract"]))
                self.assertEqual(r["abstract_math_placeholders"], len(Q.math_placeholders(o["abstract"])))
                self.assertEqual(r["recovery_status"], "unrecoverable_no_source_expression" if has_ph else "not_needed")
                if has_ph:
                    self.assertEqual(r["status"], "excluded")
                    self.assertIn("abstract_math_placeholder", r["reasons"])
                self.assertEqual(r["status"] == "eligible", not r["reasons"])
                if r["status"] == "eligible" and r["article_math_placeholders"]:
                    self.assertTrue(r["retention_note"])

    def test_verified_counts(self):
        stats = self.elig["placeholder_statistics"]
        self.assertEqual((stats["train"]["abstracts_with_math_placeholders"], stats["train"]["abstract_math_placeholder_occurrences"]), (680, 10375))
        self.assertEqual((stats["validation"]["abstracts_with_math_placeholders"], stats["validation"]["abstract_math_placeholder_occurrences"]), (105, 397))
        self.assertEqual((stats["test"]["abstracts_with_math_placeholders"], stats["test"]["abstract_math_placeholder_occurrences"]), (78, 262))
        self.assertEqual({s: self.elig["counts"][s]["eligible"] for s in SPLITS}, {s: len(self.clean[s]) for s in SPLITS})
        self.assertFalse(self.elig["recovery"]["verified_sources_available"])

    def test_clean_files_hold_only_eligible_unchanged_rows_in_same_split(self):
        for s in SPLITS:
            eligible = {r["id"]: r for r in self.elig["rows"] if r["split"] == s and r["status"] == "eligible"}
            orig_by_id = {r["id"]: r for r in self.orig[s]}
            self.assertEqual([r["id"] for r in self.clean[s]], [r["id"] for r in self.orig[s] if r["id"] in eligible])
            for row in self.clean[s]:
                self.assertEqual(row, orig_by_id[row["id"]])     # content and split unchanged
                self.assertEqual(row["split"], s)
                self.assertEqual(Q.detect_issues(row), [])
                self.assertFalse(Q.math_placeholders(Q.sft_target(row)))
            other = {r["id"] for t in SPLITS if t != s for r in self.orig[t]}
            self.assertFalse({r["id"] for r in self.clean[s]} & other)

    def test_exclusions_csv_matches(self):
        with (CLEAN / "summarization" / "summarization_exclusions.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({(r["id"], r["split"]) for r in rows},
                         {(r["id"], r["split"]) for r in self.elig["rows"] if r["status"] == "excluded"})

    def test_bloom_and_qa_files_byte_identical(self):
        for task in ("bloom_rewrite", "qa"):
            for s in SPLITS:
                self.assertEqual(sha256_file(CLEAN / task / f"{s}.jsonl"), sha256_file(ORIGINAL / task / f"{s}.jsonl"), f"{task}/{s}")

    def test_repository_validation_passes(self):
        meta = dataset_io.validate_separate_dataset(CLEAN, REPO_ROOT)
        self.assertEqual({s: meta["summarization_eligibility"][s]["status"] for s in SPLITS}, {s: "enforced" for s in SPLITS})
        self.assertEqual(meta["counts"]["train"]["bloom_rewrite"], 1075)
        self.assertEqual(meta["counts"]["train"]["qa"], 1075)

    def test_reproducible_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rebuild"
            build(ORIGINAL, out)
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(out)
                    self.assertEqual(sha256_file(p), sha256_file(CLEAN / rel), str(rel))


@unittest.skipUnless(CLEAN.is_dir() and ORIGINAL.is_dir(), "datasets not present")
class TestLoaderEnforcement(unittest.TestCase):
    def test_clean_loader_returns_only_eligible_rows(self):
        elig = json.loads((CLEAN / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE).read_text(encoding="utf-8"))
        for s in SPLITS:
            rows, meta = dataset_io.load_split_rows(CLEAN, s, tasks=["summarization"], validate=False)
            self.assertEqual({r["id"] for r in rows}, set(elig["eligible_ids"][s]))
            self.assertEqual(meta["summarization_eligibility_loaded"]["status"], "enforced")

    def test_bloom_and_qa_loads_unaffected(self):
        for task in ("bloom_rewrite", "qa"):
            a, _ = dataset_io.load_split_rows(CLEAN, "train", tasks=[task], validate=False)
            b, _ = dataset_io.load_split_rows(ORIGINAL, "train", tasks=[task], validate=False)
            self.assertEqual(a, b)

    def test_legacy_train_load_is_refused(self):
        with self.assertRaises(dataset_io.SummarizationEligibilityError):
            dataset_io.load_split_rows(ORIGINAL, "train", validate=False)
        with self.assertRaises(dataset_io.SummarizationEligibilityError):
            dataset_io.load_task_split(ORIGINAL, "summarization", "train", validate=False)
        rows, meta = dataset_io.load_split_rows(ORIGINAL, "validation", tasks=["summarization"], validate=False)
        self.assertEqual(meta["summarization_eligibility_loaded"]["status"], "unaudited_legacy")
        rows, _ = dataset_io.load_split_rows(ORIGINAL, "train", tasks=["summarization"], validate=False,
                                            allow_unaudited_summarization=True)
        self.assertEqual(len(rows), 1075)

    def _tampered_copy(self, tmp: Path) -> Path:
        dst = tmp / "copy"
        shutil.copytree(CLEAN, dst)
        return dst

    def test_injected_excluded_row_is_rejected_even_with_refreshed_manifest(self):
        excluded = next(r for r in read_jsonl(ORIGINAL / "summarization" / "train.jsonl") if Q.math_placeholders(r["abstract"]))
        with tempfile.TemporaryDirectory() as tmp:
            d = self._tampered_copy(Path(tmp))
            path = d / "summarization" / "train.jsonl"
            with path.open("a", encoding="utf-8", newline="") as f:
                f.write(json.dumps(excluded, ensure_ascii=False) + "\n")
            with self.assertRaises(ValueError):          # SHA-256 mismatch against the manifest
                dataset_io.validate_separate_dataset(d, REPO_ROOT)
            manifest = json.loads((d / "dataset_manifest.json").read_text(encoding="utf-8"))
            manifest["generated_files_sha256"]["summarization/train.jsonl"] = sha256_file(path)
            manifest["counts"]["train"]["summarization"] += 1
            (d / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(dataset_io.SummarizationEligibilityError):
                dataset_io.validate_separate_dataset(d, REPO_ROOT)
            with self.assertRaises(dataset_io.SummarizationEligibilityError):
                dataset_io.load_split_rows(d, "train", tasks=["summarization"], validate=False)

    def test_edited_eligible_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._tampered_copy(Path(tmp))
            path = d / "summarization" / "validation.jsonl"
            rows = read_jsonl(path)
            rows[0]["abstract"] += " extra"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaises(dataset_io.SummarizationEligibilityError):
                dataset_io.load_split_rows(d, "validation", tasks=["summarization"], validate=False)

    def test_tampered_eligibility_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._tampered_copy(Path(tmp))
            path = d / dataset_io.SUMMARIZATION_ELIGIBILITY_FILE
            elig = json.loads(path.read_text(encoding="utf-8"))
            next(r for r in elig["rows"] if r["status"] == "excluded")["status"] = "eligible"
            path.write_text(json.dumps(elig), encoding="utf-8")
            with self.assertRaises(dataset_io.SummarizationEligibilityError):
                dataset_io.load_split_rows(d, "train", tasks=["summarization"], validate=False)


@unittest.skipUnless((BLOOM_MASTER / "training_ready").is_dir(), "corrected Bloom dataset not present")
class TestCorrectedBloomMasterUntouched(unittest.TestCase):
    def test_master_and_frozen_splits_unchanged(self):
        tr = BLOOM_MASTER / "training_ready"
        report = json.loads((tr / "integrity_report.json").read_text(encoding="utf-8"))
        for f, h in report["master_files_sha256"].items():
            self.assertEqual(sha256_file(BLOOM_MASTER / f), h, f)
        ref = json.loads((tr / "tests" / "reference_split_ids.json").read_text(encoding="utf-8"))["example_ids"]
        man = json.loads((tr / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
        for s in SPLITS:
            self.assertEqual(man["splits"][s]["example_ids"], ref[s])
        ag = json.loads((tr / "answer_generation" / "answer_generation_manifest.json").read_text(encoding="utf-8"))
        for s, e in ag["splits"].items():
            self.assertEqual(sha256_file(tr / e["file"]), e["sha256"], s)


if __name__ == "__main__":
    unittest.main()
