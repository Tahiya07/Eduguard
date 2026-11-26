"""Length-grouped, token-budget batching for causal-LM SFT.

Why: with ``per_device_train_batch_size=1`` every example (median ~1k tokens)
runs alone, leaving the GPU mostly idle. This sampler packs examples of similar
length into one padded batch whose ``rows * padded_len`` never exceeds a token
budget. Nothing is truncated: an example longer than the budget runs alone.

Memory: activation and logits memory scale with padded tokens per batch, so a
budget equal to the longest training example keeps the peak at what the
max-length GPU preflight already measured.

Steps: the Trainer fixes its step/LR schedule from ``len(dataloader)``, so every
epoch yields exactly ``num_batches`` batches. Examples are drawn as a stream
(task-weighted with replacement, or shuffled permutations) and packed in
megabatches until that count is reached.
"""
from __future__ import annotations

import math
import random
from typing import Iterator, Sequence


def padded_length(length: int, pad_multiple: int) -> int:
    return int(math.ceil(length / pad_multiple) * pad_multiple) if pad_multiple > 1 else length


def pack_indices(
    indices: Sequence[int],
    lengths: Sequence[int],
    max_tokens: int,
    max_rows: int,
    pad_multiple: int = 8,
) -> list[list[int]]:
    """Greedy pack, longest first, under ``rows * padded_max_len <= max_tokens``."""
    ordered = sorted(indices, key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_max = 0
    for idx in ordered:
        new_max = max(batch_max, lengths[idx])
        cost = padded_length(new_max, pad_multiple) * (len(batch) + 1)
        if batch and (cost > max_tokens or len(batch) >= max_rows):
            batches.append(batch)
            batch, new_max = [], lengths[idx]
        batch.append(idx)
        batch_max = new_max
    if batch:
        batches.append(batch)
    return batches


class TokenBudgetBatchSampler:
    """Batch sampler yielding a fixed number of token-budget batches per epoch."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_tokens: int,
        max_rows: int = 16,
        weights: Sequence[float] | None = None,
        seed: int = 42,
        megabatch_size: int = 256,
        pad_multiple: int = 8,
        num_batches: int | None = None,
    ) -> None:
        if not lengths:
            raise ValueError("TokenBudgetBatchSampler needs at least one example.")
        if max_tokens <= 0 or max_rows <= 0 or megabatch_size <= 0:
            raise ValueError("max_tokens, max_rows and megabatch_size must be positive.")
        if weights is not None and len(weights) != len(lengths):
            raise ValueError("weights must align with lengths.")
        self.lengths = list(lengths)
        self.max_tokens = int(max_tokens)
        self.max_rows = int(max_rows)
        self.weights = list(weights) if weights is not None else None
        self.seed = int(seed)
        self.megabatch_size = int(megabatch_size)
        self.pad_multiple = int(pad_multiple)
        self.epoch = 0
        n = len(self.lengths)
        if num_batches is None:
            # Calibrate mean rows/batch on a long stream (8 epochs of draws) so an
            # epoch covers ~n examples, as with the previous one-row batches.
            rng = self._rng(-1)
            perm_buffer: list[int] = []
            rows = batches = 0
            for _ in range(max(1, (8 * n) // self.megabatch_size)):
                packed = self._pack(self._draw(rng, self.megabatch_size, perm_buffer))
                rows += sum(len(b) for b in packed)
                batches += len(packed)
            num_batches = max(1, round(n * batches / rows))
        self.num_batches = int(num_batches)

    def _rng(self, epoch: int) -> random.Random:
        return random.Random(self.seed * 1_000_003 + epoch)

    def _draw(self, rng: random.Random, k: int, perm_buffer: list[int]) -> list[int]:
        n = len(self.lengths)
        if self.weights is not None:
            return rng.choices(range(n), weights=self.weights, k=k)
        out: list[int] = []
        while len(out) < k:
            if not perm_buffer:
                perm_buffer.extend(range(n))
                rng.shuffle(perm_buffer)
            take = min(k - len(out), len(perm_buffer))
            out.extend(perm_buffer[:take])
            del perm_buffer[:take]
        return out

    def _pack(self, indices: Sequence[int]) -> list[list[int]]:
        return pack_indices(indices, self.lengths, self.max_tokens, self.max_rows, self.pad_multiple)

    def epoch_plan(self, epoch: int) -> list[list[int]]:
        rng = self._rng(epoch)
        perm_buffer: list[int] = []
        plan: list[list[int]] = []
        while len(plan) < self.num_batches:
            packed = self._pack(self._draw(rng, self.megabatch_size, perm_buffer))
            rng.shuffle(packed)
            plan.extend(packed)
        return plan[: self.num_batches]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        plan = self.epoch_plan(self.epoch)
        self.epoch += 1
        yield from plan

    def describe(self) -> dict:
        plan = self.epoch_plan(0)
        rows = [len(b) for b in plan]
        padded = [padded_length(max(self.lengths[i] for i in b), self.pad_multiple) * len(b) for b in plan]
        real = [sum(self.lengths[i] for i in b) for b in plan]
        return {
            "batches_per_epoch": self.num_batches,
            "examples_per_epoch": sum(rows),
            "mean_rows_per_batch": round(sum(rows) / len(rows), 2),
            "max_rows_per_batch_seen": max(rows),
            "max_padded_tokens_per_batch": max(padded),
            "padding_fraction": round(1 - sum(real) / sum(padded), 4),
            "max_tokens_per_batch": self.max_tokens,
            "max_rows_per_batch": self.max_rows,
        }
