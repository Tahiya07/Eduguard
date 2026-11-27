"""Local store of teacher-confirmed Bloom labels for federated training rounds.

Nothing here leaves the machine. The file is read only by the local federated
client (``training/federated/local_round.py``), which exports adapter weights,
never these rows. The backend must not import ``training`` (runtime isolation),
so this module has no ML dependencies.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
FORMAT = "eduguard_confirmed_bloom_label_v1"


def normalize_question(text: str) -> str:
    return " ".join((text or "").split()).strip()


def question_key(text: str) -> str:
    return hashlib.sha256(normalize_question(text).lower().encode("utf-8")).hexdigest()


class LocalLabelStore:
    """Append-only JSONL; the latest label for a question wins when read."""

    def __init__(self, path: Path, site_id: str) -> None:
        self.path = Path(path)
        self.site_id = site_id
        self._lock = threading.Lock()

    def add(
        self,
        question: str,
        confirmed_level: str,
        *,
        predicted_level: str | None = None,
        predicted_confidence: float | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        if confirmed_level not in BLOOM_LEVELS:
            raise ValueError(f"Unknown Bloom level: {confirmed_level!r}")
        text = normalize_question(question)
        if not text:
            raise ValueError("Question is empty.")
        row = {
            "format": FORMAT,
            "key": question_key(text),
            "question": text,
            "bloom_level": confirmed_level,
            "predicted_level": predicted_level if predicted_level in BLOOM_LEVELS else None,
            "predicted_confidence": None if predicted_confidence is None else round(float(predicted_confidence), 4),
            "teacher_corrected": bool(predicted_level and predicted_level != confirmed_level),
            "decision": decision,
            "site_id": self.site_id,
            "confirmed_at": int(time.time()),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def rows(self) -> list[dict[str, Any]]:
        """Deduplicated rows (latest confirmation per question), in first-seen order."""
        if not self.path.is_file():
            return []
        latest: dict[str, dict[str, Any]] = {}
        with self._lock, self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("bloom_level") in BLOOM_LEVELS and row.get("question"):
                    latest[row.get("key") or question_key(row["question"])] = row
        return list(latest.values())

    def stats(self) -> dict[str, Any]:
        rows = self.rows()
        per_level = Counter(r["bloom_level"] for r in rows)
        return {
            "site_id": self.site_id,
            "path": str(self.path),
            "n_labels": len(rows),
            "n_teacher_corrected": sum(1 for r in rows if r.get("teacher_corrected")),
            "per_level": {lvl: per_level.get(lvl, 0) for lvl in BLOOM_LEVELS},
        }
