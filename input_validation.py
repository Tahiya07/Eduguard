"""Reject inputs that are not real questions before the Bloom classifier or the rewriter sees them.

Both models always produce an answer: the classifier picks the most likely of six levels even for
"hdoiwdnc" (usually Remember), and the rewriter invents a question about "the supplied input".
Neither can say "this is not a question", so the app checks the text first.

A word counts as real if it is in input_vocab.txt (built from the project's own academic
datasets by scripts/build_input_vocab.py), is a short acronym or contains digits (TCP, IPv6),
or its letter patterns look like English (a character-trigram model trained on the same
vocabulary), which accepts new technical terms such as "kubernetes" but not keyboard mashing.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

VOCAB_PATH = Path(__file__).with_name("input_vocab.txt")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+")
# Average per-trigram log-probability below this = letter pattern not seen in English words.
PLAUSIBLE_MIN_LOGP = -3.4
MIN_WORDS = 2
MIN_REAL_SHARE = 0.6

NOT_A_QUESTION = (
    "This doesn't look like a question. Enter a complete exam question in English "
    "(for example: \"Explain how a stack differs from a queue.\")."
)
NO_TASK = (
    "This doesn't read as an exam question or task. Include what the student should do, e.g. "
    "\"Explain ...\", \"Compare ...\", \"What is ...?\" (a name or topic on its own can't be classified or rewritten)."
)
TOO_SHORT = (
    "Enter a complete question, not a single word "
    "(for example: \"Define recursion.\" or \"Explain how a stack works.\")."
)


@lru_cache(maxsize=1)
def _model() -> tuple[frozenset[str], dict[str, float], dict[str, float], float]:
    words = [w.strip() for w in VOCAB_PATH.read_text(encoding="utf-8").splitlines() if w.strip()] if VOCAB_PATH.exists() else []
    tri: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    for w in words:
        padded = f"^{w}$"
        for i in range(len(padded) - 2):
            tri[padded[i:i + 3]] += 1
            bi[padded[i:i + 2]] += 1
    alphabet = 28  # a-z plus the two boundary markers
    tri_logp = {t: math.log((c + 1) / (bi[t[:2]] + alphabet)) for t, c in tri.items()}
    bi_total = {b: float(c) for b, c in bi.items()}
    return frozenset(words), tri_logp, bi_total, float(alphabet)


def _plausibility(word: str) -> float:
    """Mean log P(char | previous two chars) with add-one smoothing; higher = more English-like."""
    _, tri_logp, bi_total, alphabet = _model()
    padded = f"^{word}$"
    scores = []
    for i in range(len(padded) - 2):
        t = padded[i:i + 3]
        scores.append(tri_logp.get(t, math.log(1.0 / (bi_total.get(t[:2], 0.0) + alphabet))))
    return sum(scores) / max(1, len(scores))


def _known(word: str, vocab: frozenset[str]) -> bool:
    if word in vocab:
        return True
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("'s", "")):
        if word.endswith(suffix) and word[: -len(suffix)] + repl in vocab:
            return True
    return False


def _is_real(token: str, vocab: frozenset[str]) -> bool:
    if any(ch.isdigit() for ch in token):
        return True  # IPv6, O(n), 2NF, years
    if token.isupper() and 2 <= len(token) <= 5:
        return True  # TCP, SQL, CPU, MiCA-style acronyms
    word = token.lower().strip("'_-")
    if len(word) < 2:
        return word in {"a", "i"}
    if re.search(r"(.)\1\1", word):
        return False  # "aaaa", "zzzz"
    parts = [p for p in re.split(r"[-_']", word) if p]
    if parts and all(_known(p, vocab) for p in parts):
        return True
    return all(len(p) < 3 or _plausibility(p) >= PLAUSIBLE_MIN_LOGP for p in parts)


def question_problem(text: str) -> str | None:
    """Return a user-facing reason the text cannot be classified or rewritten, or None if it is fine."""
    tokens = _TOKEN.findall(text or "")
    # Single letters are variables or option labels (V={A,B,C}, "part a"), not evidence either way.
    alpha = [t for t in tokens if re.search(r"[A-Za-z]", t) and len(t) > 1]
    if not alpha:
        return NOT_A_QUESTION
    vocab = _model()[0]
    if not vocab:
        return None  # word list missing: do not block anyone
    real = [t for t in alpha if _is_real(t, vocab)]
    known = [t for t in alpha if _known(t.lower(), vocab)]
    if len(alpha) < MIN_WORDS:
        return TOO_SHORT if real else NOT_A_QUESTION
    if not known or len(real) / len(alpha) < MIN_REAL_SHARE:
        return NOT_A_QUESTION
    if not _has_task(text, alpha):
        return NO_TASK
    return None


# An exam question asks something or tells the student what to do. A name ("John Smith") or a bare
# topic ("Operating systems") does neither, so it has no Bloom level and nothing to rewrite.
_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose", "whether", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "should", "would", "will", "may", "might", "shall", "has", "have",
}
_TASK_VERBS = {
    # Remember / Understand
    "define", "list", "name", "state", "identify", "recall", "recognize", "recognise", "label", "outline", "mention",
    "write", "enumerate", "match", "select", "locate", "memorize", "repeat", "tell", "give", "find", "indicate",
    "explain", "describe", "discuss", "summarize", "summarise", "interpret", "classify", "illustrate", "paraphrase",
    "clarify", "express", "report", "review", "translate", "infer", "distinguish", "differentiate", "elaborate",
    "highlight", "show", "note", "comment", "sketch", "draw", "represent", "depict", "exemplify", "relate",
    # Apply / Analyze
    "apply", "calculate", "compute", "solve", "use", "demonstrate", "implement", "execute", "determine", "estimate",
    "derive", "prove", "convert", "simulate", "trace", "perform", "construct", "measure", "predict", "plot", "find",
    "analyze", "analyse", "compare", "contrast", "examine", "investigate", "categorize", "categorise", "organize",
    "organise", "deconstruct", "diagnose", "break", "test", "question", "inspect", "correlate", "attribute",
    "explore", "debug", "optimize", "optimise", "verify", "check", "simplify", "reduce", "evaluate", "assess",
    # Evaluate / Create
    "judge", "justify", "critique", "criticize", "criticise", "argue", "defend", "appraise", "recommend", "rate",
    "rank", "prioritize", "prioritise", "conclude", "decide", "support", "validate", "weigh", "choose", "measure",
    "design", "create", "develop", "formulate", "propose", "plan", "compose", "construct", "invent", "devise",
    "build", "produce", "generate", "modify", "improve", "combine", "integrate", "adapt", "write", "draft",
    "prepare", "suggest", "hypothesize", "hypothesise", "model", "program", "code", "reorganize", "imagine",
    "consider", "suppose", "assume", "complete", "fill", "answer", "mark", "provide", "make", "set", "draw",
    "specify", "recite", "restate", "retell", "rewrite", "reproduce", "cite", "quote", "discriminate", "synthesize",
    "synthesise", "map", "divide", "place", "arrange", "order", "sort", "group", "sequence", "participate",
    "conduct", "survey", "interview", "dramatize", "dramatise", "act", "role-play", "discover", "point", "pick",
    "split", "rearrange", "reorder", "substitute", "transform", "calculate", "graph", "chart", "tabulate",
}


def _has_task(text: str, words: list[str]) -> bool:
    if "?" in text:
        return True
    for w in words:
        w = w.lower()
        stems = {w, w.rstrip("s"), w[:-2] if w.endswith("es") else w, w[:-1] if w.endswith("d") else w,
                 w[:-2] if w.endswith("ed") else w, w[:-3] if w.endswith("ing") else w, w[:-3] + "e" if w.endswith("ing") else w}
        if stems & (_QUESTION_WORDS | _TASK_VERBS):
            return True
    return False


def require_question(text: str) -> None:
    """Raise ValueError (shown to the user as the error modal) when the text is not a question."""
    problem = question_problem(text)
    if problem:
        raise ValueError(problem)
