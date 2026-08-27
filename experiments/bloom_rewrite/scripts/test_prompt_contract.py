#!/usr/bin/env python
"""Verify production-aligned generator prompt: question + target only."""
from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from prompt_format import (  # noqa: E402
    build_generation_prompt,
    build_sft_text,
    describe_training_task,
)


def main() -> None:
    q = "Explain what virtual memory is."
    for level in ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"):
        prompt = build_generation_prompt(q, level)
        sft = build_sft_text(q, level, f"Placeholder rewrite for {level}.")
        assert "Target Bloom level:" in prompt
        assert level in prompt
        assert "Original Bloom level:" not in prompt
        assert "Source Bloom level:" not in prompt
        assert "Original Bloom level:" not in sft
        assert "Source Bloom level:" not in sft
        assert q in prompt
    print(json_dumps := __import__("json").dumps(describe_training_task(), indent=2))
    print("PASS: all six targets accepted; source Bloom absent from generator prompts.")


if __name__ == "__main__":
    main()
