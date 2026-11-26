"""Tests for token-budget batching (token_budget_batching.py). No model, no GPU."""
from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from token_budget_batching import TokenBudgetBatchSampler, pack_indices, padded_length  # noqa: E402


def skewed_lengths(n: int = 2000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.choice([rng.randint(150, 1500)] * 9 + [rng.randint(2000, 7200)]) for _ in range(n)]


class TokenBudgetBatchingTests(unittest.TestCase):
    def test_budget_and_row_cap_respected(self) -> None:
        lengths = skewed_lengths()
        budget = max(lengths)
        for batch in pack_indices(range(len(lengths)), lengths, budget, 16):
            self.assertLessEqual(len(batch), 16)
            if len(batch) > 1:
                self.assertLessEqual(padded_length(max(lengths[i] for i in batch), 8) * len(batch), budget)

    def test_over_budget_example_runs_alone_not_dropped(self) -> None:
        lengths = [100, 100, 9000, 100]
        batches = pack_indices(range(4), lengths, 1000, 16)
        self.assertIn([2], batches)
        self.assertEqual(sorted(i for b in batches for i in b), [0, 1, 2, 3])

    def test_fixed_batch_count_every_epoch(self) -> None:
        lengths = skewed_lengths()
        sampler = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), seed=1)
        counts = {len(list(iter(sampler))) for _ in range(4)}
        self.assertEqual(counts, {len(sampler)})

    def test_epochs_differ_and_are_reproducible(self) -> None:
        lengths = skewed_lengths()
        a = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), seed=7)
        b = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), seed=7)
        self.assertEqual(a.epoch_plan(0), b.epoch_plan(0))
        self.assertNotEqual(a.epoch_plan(0), a.epoch_plan(1))
        a.set_epoch(2)
        self.assertEqual(list(iter(a)), b.epoch_plan(2))

    def test_epoch_covers_about_n_examples(self) -> None:
        lengths = skewed_lengths()
        sampler = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), seed=3)
        sizes = [sum(map(len, sampler.epoch_plan(e))) for e in range(4)]
        self.assertLess(abs(sum(sizes) / len(sizes) - len(lengths)) / len(lengths), 0.05)

    def test_task_weights_respected(self) -> None:
        lengths = skewed_lengths(3000)
        tasks = ["a"] * 2000 + ["b"] * 1000
        target = {"a": 0.25, "b": 0.75}
        counts = Counter(tasks)
        weights = [target[t] / counts[t] for t in tasks]
        sampler = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), weights=weights, seed=5)
        seen = Counter(tasks[i] for e in range(3) for b in sampler.epoch_plan(e) for i in b)
        self.assertAlmostEqual(seen["b"] / sum(seen.values()), 0.75, delta=0.03)

    def test_unweighted_epoch_has_no_duplicates_within_first_pass(self) -> None:
        lengths = skewed_lengths(512)
        sampler = TokenBudgetBatchSampler(lengths, max_tokens=max(lengths), megabatch_size=512, seed=9)
        first = [i for b in sampler.epoch_plan(0) for i in b][:500]
        self.assertEqual(len(first), len(set(first)))


if __name__ == "__main__":
    unittest.main()
