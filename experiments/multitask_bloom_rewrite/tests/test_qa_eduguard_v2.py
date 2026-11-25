"""Integrity tests for data/qa_eduguard_v2 (academic closed-book QA, 1,172 / 146 / 147).

Read-only: no training, no model weights, no GPU. The similarity test needs scikit-learn (project venv) and is
skipped without it.
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
V2 = REPO_ROOT / "data" / "qa_eduguard_v2"
V1 = REPO_ROOT / "data" / "qa_eduguard_v1"
SRC = REPO_ROOT / "data" / "train.jsonl"
REVIEW = REPO_ROOT / "experiments" / "multitask_bloom_rewrite" / "qa_review" / "answer_review_v2.json"
SPLITS = ("train", "validation", "test")
EXPECTED = {"train": 1172, "validation": 146, "test": 147, "reserve": 135}
ADDED_FIELDS = {"discipline_canonical", "discipline_group", "qa_type_canonical", "difficulty_canonical",
                "answer_review", "answer_review_note", "reserve_reason"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@unittest.skipUnless(V2.is_dir(), "qa_eduguard_v2 not built")
class TestQAEduGuardV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((V2 / "dataset_manifest.json").read_text(encoding="utf-8"))
        cls.rows = {s: read(V2 / f"{s}.jsonl") for s in (*SPLITS, "reserve")}
        cls.source = {r["id"]: r for r in read(SRC)}
        cls.review = json.loads(REVIEW.read_text(encoding="utf-8"))["reviews"]

    def test_exact_counts(self) -> None:
        for split, n in EXPECTED.items():
            self.assertEqual(len(self.rows[split]), n, split)
        self.assertEqual(sum(len(self.rows[s]) for s in SPLITS), 1465)

    def test_hashes_match_manifest(self) -> None:
        for name, digest in self.manifest["files_sha256"].items():
            self.assertEqual(sha256(V2 / name), digest, name)

    def test_source_and_v1_unchanged(self) -> None:
        self.assertEqual(sha256(SRC), self.manifest["source"]["sha256"])
        for name, digest in self.manifest["audit_baseline"]["files_sha256"].items():
            self.assertEqual(sha256(V1 / name), digest, name)

    def test_partition_of_source_with_fields_preserved(self) -> None:
        ids = [r["id"] for rows in self.rows.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.source))
        for split, rows in self.rows.items():
            for r in rows:
                self.assertEqual(r["split"], split)
                src = self.source[r["id"]]
                for key, value in src.items():
                    if key != "split":
                        self.assertEqual(r[key], value, (r["id"], key))
                self.assertLessEqual(set(r) - set(src), ADDED_FIELDS, r["id"])

    def test_no_empty_question_or_answer(self) -> None:
        for split in SPLITS:
            for r in self.rows[split]:
                self.assertTrue(r["question"].strip() and r["reference_answer"].strip(), r["id"])

    def test_no_exact_overlap_across_splits(self) -> None:
        for key in ("id", "question", "reference_answer", "topic"):
            sets = {s: {norm(r[key]) for r in self.rows[s]} for s in SPLITS}
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(sets[a] & sets[b], (key, a, b))

    def test_no_duplicate_questions_in_used_splits(self) -> None:
        q = [norm(r["question"]) for s in SPLITS for r in self.rows[s]]
        self.assertEqual(len(q), len(set(q)))

    def test_heldout_answers_all_reviewed_correct(self) -> None:
        for split in ("validation", "test"):
            for r in self.rows[split]:
                self.assertEqual(self.review.get(r["id"], {}).get("verdict"), "correct", r["id"])
                self.assertEqual(r["answer_review"], "correct", r["id"])

    def test_flagged_answers_excluded_from_all_splits(self) -> None:
        flagged = {i for i, v in self.review.items() if v["verdict"] != "correct"}
        self.assertTrue(flagged)
        used = {r["id"] for s in SPLITS for r in self.rows[s]}
        self.assertFalse(flagged & used)

    def test_discipline_groups_stratified(self) -> None:
        used = Counter(r["discipline_group"] for s in SPLITS for r in self.rows[s])
        total = sum(used.values())
        for split in ("validation", "test"):
            got = Counter(r["discipline_group"] for r in self.rows[split])
            n = len(self.rows[split])
            for group, count in used.items():
                self.assertLessEqual(abs(got[group] - count * n / total), 1.0, (split, group))

    def test_similarity_leakage(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            self.skipTest("scikit-learn not installed")
        import numpy as np
        rows = [r for s in SPLITS for r in self.rows[s]]
        q = [r["question"] for r in rows]
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(q)
        word = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(q)
        sim = np.maximum(cosine_similarity(char), cosine_similarity(word))
        split = np.array([r["split"] for r in rows])
        threshold = self.manifest["thresholds"]["leakage_group_question_similarity"]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            block = sim[np.ix_(np.where(split == a)[0], np.where(split == b)[0])]
            self.assertLess(float(block.max()), threshold, (a, b))

    def test_v1_known_leaks_resolved(self) -> None:
        where = {r["id"]: s for s, rows in self.rows.items() for r in rows}
        for a, b in (("QA-0883", "QA-1274"), ("QA-0803", "QA-0660"), ("QA-0827", "QA-0340"),
                     ("QA-0137", "QA-0763"), ("QA-1218", "QA-1568")):
            if "reserve" not in (where[a], where[b]):
                self.assertEqual(where[a], where[b], (a, b))


if __name__ == "__main__":
    unittest.main()
