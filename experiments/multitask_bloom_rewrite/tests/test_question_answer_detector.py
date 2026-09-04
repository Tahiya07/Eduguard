"""Unit tests for question-vs-answer validator (do not execute in this Cursor task).

Representative cases for interrogative / imperative / answer / meta forms.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import (  # noqa: E402
    is_valid_imperative_exam,
    is_valid_interrogative,
    validate_bloom_example,
)


class TestQuestionAnswerDetector(unittest.TestCase):
    def test_valid_interrogative(self) -> None:
        q = "How would virtual memory be used when an application requires more memory than available RAM?"
        self.assertTrue(is_valid_interrogative(q))
        v = validate_bloom_example("Explain virtual memory.", "Apply", q)
        self.assertTrue(v.format_valid)

    def test_valid_imperative_calculate(self) -> None:
        q = "Calculate the page-fault rate for the following access pattern."
        self.assertTrue(is_valid_imperative_exam(q))
        v = validate_bloom_example("Compute the page-fault rate.", "Apply", q)
        self.assertTrue(v.format_valid)
        self.assertFalse(v.rejection_reason and "answer_or_declarative" in v.rejection_reason)

    def test_valid_imperative_given(self) -> None:
        q = "Given a concrete case involving virtual memory, how would you apply the relevant procedure?"
        v = validate_bloom_example("Explain virtual memory.", "Apply", q)
        self.assertTrue(v.format_valid)

    def test_answer_declarative_bad(self) -> None:
        bad = "Virtual memory allows disk space to act as additional memory."
        self.assertFalse(is_valid_interrogative(bad))
        self.assertFalse(is_valid_imperative_exam(bad))
        v = validate_bloom_example("Explain virtual memory.", "Understand", bad)
        self.assertFalse(v.format_valid)

    def test_explanation_bad(self) -> None:
        bad = "In conclusion, virtual memory improves multitasking by extending RAM."
        v = validate_bloom_example("Explain virtual memory.", "Understand", bad)
        self.assertFalse(v.format_valid)

    def test_definition_as_response_bad(self) -> None:
        bad = "It is a memory management technique that uses disk as RAM."
        v = validate_bloom_example("Define virtual memory.", "Remember", bad)
        self.assertFalse(v.format_valid)

    def test_meta_response_bad(self) -> None:
        bad = "Here is the rewritten question at Bloom level Understand."
        v = validate_bloom_example("Explain virtual memory.", "Understand", bad)
        self.assertFalse(v.format_valid)

    def test_topic_drift_rejected(self) -> None:
        bad = "How does CPU pipelining improve performance?"
        v = validate_bloom_example("Explain virtual memory.", "Understand", bad)
        self.assertFalse(v.semantic_valid)

    def test_empty_bad(self) -> None:
        v = validate_bloom_example("Explain virtual memory.", "Remember", "")
        self.assertFalse(v.format_valid)


if __name__ == "__main__":
    unittest.main()
