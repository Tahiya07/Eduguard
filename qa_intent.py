"""Understand a notes question before answering it: requested format, comparisons and follow-ups.

The v3.9 model answers in about two short sentences (it was trained on ~31-word answers), so
"differentiate side chains and sharding" got two lines about one of them, and "write it in points"
or "as a comparison table" was treated as a brand-new question. These helpers let the app
  * resolve follow-ups against the previous question (conversation),
  * split a comparison into short per-topic questions the model answers well, and
  * lay the result out as points or a table.
"""
from __future__ import annotations

import re
from typing import Optional

_TABLE = re.compile(r"\b(table|tabular|tabulate|tabulated|grid)\b", re.I)
_POINTS = re.compile(r"\b(points?|bullets?|bullet[- ]points?|point[- ]wise|pointwise|list (?:them|it|out)|in a list|key points|step[- ]by[- ]step|steps)\b", re.I)

# "difference(s) between X and Y", "differentiate X and Y", "compare X with Y", "X vs Y", ...
_COMPARE = [
    re.compile(r"\b(?:differen\w*|distinguish\w*|compar\w*|contrast\w*)\b\s+(?:between\s+|of\s+)?(?P<a>.+?)\s+(?:and|vs\.?|versus|with|from|to)\s+(?P<b>.+)", re.I),
    re.compile(r"\bhow\s+(?:is|are|does|do)\s+(?P<a>.+?)\s+differ\w*\s+(?:from|to|with)\s+(?P<b>.+)", re.I),
    re.compile(r"^(?:what(?:'s| is| are)?\s+)?(?P<a>[\w\s-]+?)\s+(?:vs\.?|versus)\s+(?P<b>[\w\s-]+)", re.I),
]
_TRAILING_FORMAT = re.compile(
    r"\s+(?:in|as|into|using|with)\s+(?:a\s+|an\s+|the\s+)?(?:comparison\s+|short\s+|simple\s+|detailed\s+)?"
    r"(?:table|tabular\s+form|points?|bullet\s+points?|bullets?|list|detail|words).*$", re.I)
_LEADING = re.compile(r"^(?:the|a|an|between)\s+", re.I)

# A message that only asks to re-present the previous answer.
_FORMAT_ONLY_WORDS = {
    "write", "give", "show", "make", "put", "present", "rewrite", "format", "convert", "explain", "it", "this", "that",
    "them", "me", "in", "as", "into", "a", "an", "the", "comparison", "compare", "table", "tabular", "form", "points",
    "point", "bullet", "bullets", "list", "please", "can", "you", "could", "now", "same", "answer", "again", "of",
    "with", "and", "or", "to", "do", "short", "simple", "detailed", "detail", "pointwise", "wise", "differences",
    "difference", "key", "steps", "step", "by", "tabulate", "using", "for", "above", "previous", "your",
}
# Pronouns / ellipsis that only make sense with the previous question.
_REFERENCE = re.compile(
    r"^\s*(?:and\b|what about\b|how about\b|also\b|then\b|why\b(?=.{0,25}$))|\b(?:it|its|it's|they|them|their|this|that|these|those|the latter|the former|same)\b",
    re.I)


def answer_format(question: str) -> Optional[str]:
    """'table', 'points' or None (plain answer)."""
    q = question or ""
    if _TABLE.search(q):
        return "table"
    if _POINTS.search(q):
        return "points"
    return None


def _clean_entity(text: str) -> str:
    t = _TRAILING_FORMAT.sub("", text.strip())
    t = re.split(r"[?.!;]|\s+(?:in terms of|with respect to|based on|according to)\s+", t)[0]
    t = _LEADING.sub("", t.strip())
    t = re.sub(r"^(?:the\s+)", "", t, flags=re.I)
    return t.strip(" ,:-").strip()


def comparison_entities(question: str) -> Optional[tuple[str, str]]:
    """('side chains', 'sharding') for "differentiate between side chains and sharding in points"."""
    q = (question or "").strip()
    for pat in _COMPARE:
        m = pat.search(q)
        if m:
            a, b = _clean_entity(m.group("a")), _clean_entity(m.group("b"))
            if a and b and a.lower() != b.lower() and len(a.split()) <= 6 and len(b.split()) <= 6:
                return a, b
    return None


def is_format_followup(question: str) -> bool:
    """'write it in points', 'make a comparison table', 'in points please' -> True."""
    q = (question or "").lower()
    if not answer_format(q):
        return False
    words = re.findall(r"[a-z]+", q)
    return bool(words) and all(w in _FORMAT_ONLY_WORDS for w in words)


def is_reference_followup(question: str) -> bool:
    """'what about its security?', 'and the risks?', 'how does it work' -> True (needs the previous question)."""
    q = (question or "").strip()
    return bool(q) and len(q.split()) <= 10 and bool(_REFERENCE.search(q))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
_GLYPHS = "•·▪●‣◦�"  # �: a bullet the PDF font could not encode


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    # Slide bullets (•, ·, ▪ …) separate points just like sentence ends do.
    for piece in re.split(f"[{_GLYPHS}]", text or ""):
        flat = re.sub(r"\s*\n\s*", " ", piece.strip())
        flat = re.sub(r"^\s*[-*]\s*", "", flat)
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", flat) if s.strip()]
    return out


def first_sentences(text: str, n: int = 2) -> str:
    return " ".join(_sentences(text)[:n])


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3}


def as_points(answer: str, extra_sentences: list[str], max_points: int = 7) -> str:
    """The answer as '- ' points: its own sentences first, then sentences from the notes that add
    something new (less than 60% word overlap with every point already listed)."""
    points: list[str] = []
    extra = [x for e in (extra_sentences or []) for x in _sentences(e)]  # also strips slide bullet glyphs
    for s in _sentences(answer) + extra:
        s = s.strip()
        if not s or len(points) >= max_points:
            continue
        w = _words(s)
        # skip a sentence that mostly repeats a listed point (overlap measured on the shorter one)
        if w and any(len(w & _words(p)) / max(1, min(len(w), len(_words(p)))) >= 0.6 for p in points):
            continue
        points.append(s if s.endswith((".", "!", "?")) else s + ".")
    return "\n".join(f"- {p}" for p in points) if points else answer


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("|", "/")).strip() or "Not covered in your notes"


def _title(name: str) -> str:
    return name[:1].upper() + name[1:]


def layout_comparison(a: str, b: str, rows: list[tuple[str, str, str]], fmt: str) -> str:
    """rows: (aspect, text for a, text for b). 'table' -> Markdown table; otherwise points per topic."""
    A, B = _title(a), _title(b)
    if fmt == "table":
        lines = [f"| Aspect | {A} | {B} |", "|---|---|---|"]
        lines += [f"| {label} | {_cell(x)} | {_cell(y)} |" for label, x, y in rows]
        return "\n".join(lines)
    out = [f"{A}:"] + [f"- {label}: {x}" for label, x, _ in rows if x]
    out += ["", f"{B}:"] + [f"- {label}: {y}" for label, _, y in rows if y]
    return "\n".join(out)


_PRONOUN = re.compile(r"\b(its|it's|it|they|them|their|this|that|these|those)\b", re.I)
_STOP_TOPIC = {
    "what", "which", "when", "where", "why", "how", "who", "is", "are", "was", "were", "does", "do", "did", "the", "a",
    "an", "of", "in", "on", "for", "to", "and", "or", "explain", "define", "describe", "tell", "me", "about", "give",
    "list", "mention", "write", "please", "can", "you", "its", "it", "main", "key", "difference", "between",
    "differentiate", "compare", "points", "point", "table", "work", "works", "mean", "means", "meant", "by", "with",
    "into", "detail", "details", "briefly", "short", "notes", "pdf",
}


def _topic_of(question: str) -> str:
    """The subject of a question: its content words in order ('what is a side chain' -> 'side chain')."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", question or "")
    return " ".join([w for w in words if w.lower() not in _STOP_TOPIC][:4])


def resolve_reference(question: str, previous: str) -> str:
    """'what about its security?' after 'what is a side chain' -> "what about side chain's security?".
    Without a pronoun ('and the risks?') the previous topic is appended ('the risks of side chain?')."""
    topic = _topic_of(previous)
    if not topic:
        return f"{previous} {question}"
    q = (question or "").strip()

    def sub(m: re.Match) -> str:
        if m.group(1).lower() in ("its", "it's", "their"):
            return f"{topic}'" if topic.endswith("s") else f"{topic}'s"
        return topic

    if _PRONOUN.search(q):
        return _PRONOUN.sub(sub, q, count=1)
    q = re.sub(r"^\s*(?:and|also|then)\s+", "", q, flags=re.I).rstrip("?. ")
    return f"{q} of {topic}?"
