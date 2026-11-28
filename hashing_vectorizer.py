"""Dependency-free copy of the scikit-learn HashingVectorizer configuration used by the encoder fallback.

encoder_backends.StableTextEncoder falls back to a hashing encoder when the BGE model cannot load. It
used sklearn's HashingVectorizer(n_features, alternate_sign=False, norm=None, ngram_range=(1, 2),
lowercase=True, strip_accents="unicode"), which pulled scikit-learn and SciPy (~100 MB) into the
packaged app for this one fallback. This module reproduces that configuration exactly (same
tokenizer, same signed MurmurHash3-32 with seed 0, same index rule), so vectors - and any index
built with the fallback - are identical; tests/test_hashing_vectorizer.py checks it against sklearn.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def murmurhash3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 x86 32-bit, returned as a signed int (as sklearn's murmurhash3_32)."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    length = len(data)
    h1 = seed & 0xFFFFFFFF
    end = length & ~3
    for i in range(0, end, 4):
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
    k1 = 0
    tail = length & 3
    if tail == 3:
        k1 ^= data[end + 2] << 16
    if tail >= 2:
        k1 ^= data[end + 1] << 8
    if tail >= 1:
        k1 ^= data[end]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1 - 0x100000000 if h1 & 0x80000000 else h1


def _strip_accents_unicode(s: str) -> str:
    try:
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join(c for c in normalized if not unicodedata.combining(c))


def _analyze(doc: str) -> list[str]:
    tokens = _TOKEN.findall(_strip_accents_unicode(doc.lower()))
    grams = list(tokens)
    grams.extend(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    return grams


class HashingVectorizer:
    """Same output as sklearn HashingVectorizer(n_features, alternate_sign=False, norm=None,
    ngram_range=(1, 2), lowercase=True, strip_accents="unicode"); returns a dense float32 array."""

    def __init__(self, n_features: int = 384) -> None:
        self.n_features = int(n_features)

    def _index(self, feature: str) -> int:
        h = murmurhash3_32(feature.encode("utf-8"), 0)
        return (2 ** 31) % self.n_features if h == -(2 ** 31) else abs(h) % self.n_features

    def transform(self, docs: Iterable[str]) -> np.ndarray:
        docs = list(docs)
        out = np.zeros((len(docs), self.n_features), dtype=np.float32)
        for row, doc in enumerate(docs):
            for feature in _analyze(doc):
                out[row, self._index(feature)] += 1.0
        return out
