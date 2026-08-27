"""Load official SQuAD 1.1 JSON (train/dev) reproducibly.

HF `rajpurkar/squad` currently fails on some datasets library versions with:
  TypeError: must be called with a dataclass type or instance

Official sources (SQuAD 1.1):
  https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

These are the canonical SQuAD 1.1 splits (train + validation/dev).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterator

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_squad_json(cache_dir: Path, split: str) -> Path:
    if split not in SQUAD_URLS:
        raise ValueError(f"Unknown SQuAD split: {split}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"squad_v1.1_{split}.json"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = SQUAD_URLS[split]
    tmp = dest.with_suffix(".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def iter_squad_examples(path: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    for article in payload.get("data") or []:
        title = article.get("title")
        for para in article.get("paragraphs") or []:
            context = para.get("context") or ""
            for qa in para.get("qas") or []:
                answers = qa.get("answers") or []
                texts = [a.get("text") for a in answers if a.get("text")]
                answer_starts = [a.get("answer_start") for a in answers if "answer_start" in a]
                yield {
                    "id": qa.get("id"),
                    "title": title,
                    "context": context,
                    "question": qa.get("question") or "",
                    "answers": {"text": texts, "answer_start": answer_starts},
                    "dataset_version": version,
                }


def load_squad_split(cache_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = ensure_squad_json(cache_dir, split)
    rows = list(iter_squad_examples(path))
    meta = {
        "source": "official_squad_v1.1_json",
        "url": SQUAD_URLS[split],
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "n_examples": len(rows),
        "hf_fallback_note": (
            "Loaded from official SQuAD Explorer JSON because HF rajpurkar/squad "
            "failed under the local datasets version."
        ),
    }
    return rows, meta
