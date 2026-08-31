"""Lightweight production regression — no model download."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_backend_imports_without_training():
    backend_file = ROOT / "backend" / "service.py"
    src = backend_file.read_text(encoding="utf-8")
    assert "training.federated" not in src
    assert "opacus" not in src


def test_predict_bloom_prompt_builder():
    from predict_bloom import build_prompt

    p = build_prompt("What is photosynthesis?")
    assert "photosynthesis" in p


def test_training_not_imported_by_backend():
    src = (ROOT / "backend" / "service.py").read_text(encoding="utf-8")
    assert "training.federated" not in src
    assert "opacus" not in src
    assert "peft" not in src
