"""The built-in hashing vectorizer must stay identical to the sklearn configuration it replaced."""
import numpy as np
import pytest

from hashing_vectorizer import HashingVectorizer, murmurhash3_32

sklearn = pytest.importorskip("sklearn")  # dev environment only; the packaged app ships without it

TEXTS = ["Process scheduling decides which ready process runs next on the CPU.",
         "Café naïve résumé — Ångström façade!", "", "a", "TCP vs UDP: 2NF, IPv6 & O(n log n)",
         "日本語 テキスト mixed with English words", "Deadlock deadlock DEADLOCK repeated repeated."]


def test_murmurhash_matches_sklearn_for_every_tail_length():
    from sklearn.utils import murmurhash3_32 as reference
    for n in range(0, 24):
        data = ("abé中" * 10)[:n].encode("utf-8")
        assert murmurhash3_32(data) == reference(data, seed=0)


def test_vectors_match_sklearn_exactly():
    from sklearn.feature_extraction.text import HashingVectorizer as Reference
    ref = Reference(n_features=384, alternate_sign=False, norm=None, ngram_range=(1, 2), lowercase=True,
                    strip_accents="unicode").transform(TEXTS).toarray().astype(np.float32)
    assert np.array_equal(HashingVectorizer(384).transform(TEXTS), ref)
