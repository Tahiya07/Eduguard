"""Summary requests: instruction/source separation, exact formatting, and the trained prompt."""
import importlib.util
from pathlib import Path

import pytest

from summary_format import SummaryFormat, apply_format, split_request

SRC = ("Combinational logic circuits depend only on current inputs and do not store states. They use Boolean "
       "algebra, logic expressions and truth tables. Basic gates include AND, OR, NOT, NAND, NOR and XOR. "
       "Karnaugh maps reduce complexity, and propagation delay guides implementation.")
SUMMARY = ("Combinational circuits depend on current inputs and use Boolean algebra. Basic gates include AND, OR, NOT, "
           "NAND, NOR and XOR, and NAND and NOR are universal. Karnaugh maps reduce gate count and delay. "
           "Propagation delay and timing analysis guide design.")


@pytest.mark.parametrize("message, kind, n", [
    (f"Summarize this in 5 bullet points:\n{SRC}", "bullets", 5),
    (f"{SRC}\n\nsummarize in 3 sentences", "sentences", 3),
    (f"summarise in five points: {SRC}", "bullets", 5),
    (f"Give me a brief summary. {SRC}", "sentences", 3),
    (f"Summarize in 100 words\n\n{SRC}", "words", 100),
    (SRC, None, None),
])
def test_instruction_is_separated_from_the_text(message, kind, n):
    source, fmt = split_request(message)
    assert (fmt.kind, fmt.n) == (kind, n)
    assert source.startswith("Combinational") and "bullet" not in source.lower() and "summar" not in source.lower()


def test_request_without_text_has_no_source():
    source, fmt = split_request("Summarize my notes in 4 bullet points")
    assert source == "" and (fmt.kind, fmt.n) == ("bullets", 4)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_exact_bullet_count_and_no_idea_dropped(n):
    out = apply_format(SUMMARY, SummaryFormat("bullets", n))
    lines = out.splitlines()
    assert len(lines) == n and all(line.startswith("- ") for line in lines)
    for term in ("Boolean", "XOR", "Karnaugh", "timing"):
        assert term in out


def test_sentence_and_word_limits():
    assert apply_format(SUMMARY, SummaryFormat("sentences", 2)).count(".") == 2
    assert len(apply_format(SUMMARY, SummaryFormat("words", 20)).split()) <= 20


def test_paragraph_is_unchanged():
    assert apply_format(SUMMARY, SummaryFormat()) == SUMMARY


def test_app_summary_prompt_matches_training_prompt():
    from models import TRAINED_SUMMARY_SYSTEM_PROMPT

    path = Path(__file__).resolve().parents[1] / "experiments" / "multitask_bloom_rewrite" / "prompts.py"
    spec = importlib.util.spec_from_file_location("v3_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert TRAINED_SUMMARY_SYSTEM_PROMPT == module.LECTURE_SUMMARIZATION_SYSTEM
