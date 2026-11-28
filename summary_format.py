"""Summary requests: separate the user's format instruction from the text, and apply the format.

The v3.9 generator was trained only on paragraph summaries, so it ignores instructions such as
"in 5 bullet points" or "in 3 sentences" (measured: 0 bullets, wrong sentence counts). The app
therefore (1) strips the instruction from the pasted text, (2) lets the model summarize with
its training prompt, and (3) formats the result here, deterministically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "a single": 1, "single": 1,
}
_NUM = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|a single|single)"
# A short line or sentence that talks about the summary itself rather than being source text.
_INSTRUCTION_HINT = re.compile(
    r"\b(summar(?:y|ise|ize|ised|ized|ising|izing)|tl;?dr|bullet|bulleted|points?|sentences?|words?|"
    r"key (?:ideas|points|takeaways)|brief(?:ly)?|short(?:er)?|concise(?:ly)?|overview|in (?:simple|plain) (?:terms|words))\b",
    re.I,
)


@dataclass
class SummaryFormat:
    kind: str | None = None  # "bullets" | "sentences" | "words" | None (paragraph, model default)
    n: int | None = None

    def describe(self) -> str:
        if self.kind is None:
            return "paragraph"
        return f"{self.n} {self.kind}" if self.n else self.kind


def _to_int(token: str) -> int | None:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def parse_format(instruction: str) -> SummaryFormat:
    text = instruction.lower()
    m = re.search(_NUM + r"\s*(?:-\s*)?(bullet(?:\s*points?)?|bulleted points?|points?|key points?|sentences?|lines?|words?)\b", text)
    if m:
        n = _to_int(m.group(1))
        unit = m.group(2)
        if "sentence" in unit or "line" in unit:
            return SummaryFormat("sentences", n)
        if "word" in unit:
            return SummaryFormat("words", n)
        return SummaryFormat("bullets", n)
    if re.search(r"\bbullet|\bkey points\b|\bpoint form\b", text):
        return SummaryFormat("bullets", None)
    if re.search(r"\b(one|single)[- ]sentence\b", text):
        return SummaryFormat("sentences", 1)
    if re.search(r"\b(brief(?:ly)?|short|concise|tl;?dr)\b", text):
        return SummaryFormat("sentences", 3)
    return SummaryFormat()


def split_request(message: str) -> tuple[str, SummaryFormat]:
    """Return (source_text, format) from a summarize message such as
    "Summarize this in 5 bullet points:\\n<text>" or "<text>\\n\\nGive me 3 sentences."
    """
    msg = (message or "").strip()
    if not msg:
        return "", SummaryFormat()
    instruction_parts: list[str] = []

    def is_instruction(fragment: str) -> bool:
        f = fragment.strip()
        return 0 < len(f) <= 220 and bool(_INSTRUCTION_HINT.search(f)) and len(f.split()) <= 30

    lines = msg.splitlines()
    # Leading instruction line(s), e.g. "Summarize in 5 bullet points:" (possibly followed by the text on that line).
    while lines:
        first = lines[0].strip()
        head, sep, rest = first.partition(":")
        if sep and is_instruction(head) and rest.strip():
            instruction_parts.append(head)
            lines[0] = rest
            break
        if first and is_instruction(first):
            instruction_parts.append(first)
            lines.pop(0)
            continue
        if not first:
            lines.pop(0)
            continue
        break
    # Trailing instruction line(s), e.g. "... text ...\n\nsummarize this in 3 sentences".
    while lines:
        last = lines[-1].strip()
        if last and is_instruction(last):
            instruction_parts.append(last)
            lines.pop()
            continue
        if not last:
            lines.pop()
            continue
        break
    source = "\n".join(lines).strip()
    # Single paragraph with the instruction as its first or last sentence.
    if not instruction_parts and source:
        sentences = re.split(r"(?<=[.!?:])\s+", source)
        if len(sentences) > 1 and is_instruction(sentences[0]) and re.search(r"summar|bullet|point", sentences[0], re.I):
            instruction_parts.append(sentences[0])
            source = " ".join(sentences[1:]).strip()
        elif len(sentences) > 1 and is_instruction(sentences[-1]) and re.search(r"summar|bullet|point", sentences[-1], re.I):
            instruction_parts.append(sentences[-1])
            source = " ".join(sentences[:-1]).strip()
    fmt = parse_format(" ".join(instruction_parts)) if instruction_parts else SummaryFormat()
    return source, fmt


def _sentences(text: str) -> list[str]:
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", re.sub(r"\s+", " ", text))
    return [p.strip() for p in parts if p.strip()]


def _split_clause(sentence: str) -> list[str]:
    """Split one long sentence at a clause boundary (used to reach the requested bullet count)."""
    for sep in ("; ", ", while ", ", and ", ", whereas ", ", but ", " and "):
        i = sentence.find(sep, len(sentence) // 4)
        if i != -1 and i < len(sentence) * 3 // 4:
            left = sentence[:i].rstrip(" ,;") + "."
            right = sentence[i + len(sep):].strip()
            right = right[:1].upper() + right[1:]
            return [left, right if right.endswith((".", "!", "?")) else right + "."]
    return [sentence]


def apply_format(summary: str, fmt: SummaryFormat) -> str:
    text = (summary or "").strip()
    if not text or fmt.kind is None:
        return text
    sents = _sentences(text)
    if not sents:
        return text
    if fmt.kind == "sentences":
        n = max(1, fmt.n or len(sents))
        return " ".join(sents[:n])
    if fmt.kind == "words":
        n = max(5, fmt.n or 60)
        out: list[str] = []
        for s in sents:
            if out and len(" ".join(out + [s]).split()) > n:
                break
            out.append(s)
        if len(" ".join(out).split()) > n:  # a single very long first sentence
            out = [" ".join(" ".join(out).split()[:n]).rstrip(",;:") + "…"]
        return " ".join(out)
    # bullets
    n = fmt.n or len(sents)
    items = list(sents)
    # Too few sentences: split the longest at a clause boundary until we reach n (or cannot).
    while len(items) < n:
        longest = max(range(len(items)), key=lambda i: len(items[i]))
        pieces = _split_clause(items[longest])
        if len(pieces) == 1:
            break
        items[longest:longest + 1] = pieces
    # Too many: merge neighbours evenly so every idea is kept.
    if len(items) > n:
        size, extra = divmod(len(items), n)
        merged, i = [], 0
        for k in range(n):
            take = size + (1 if k < extra else 0)
            merged.append(" ".join(items[i:i + take]))
            i += take
        items = merged
    return "\n".join(f"- {item}" for item in items)


_DOC_WORDS = r"(?:pdf|document|doc|file|notes?|material|slides?|handout|upload(?:ed)?|text|chapter|lecture)"
_DOCUMENT_QUESTION = re.compile(
    r"^\s*(?:what(?:'s| is| are)?\s+(?:this|the|my|these|that)?\s*(?:uploaded\s+)?" + _DOC_WORDS + r"\s+(?:about|on|covering)"
    r"|(?:give|provide|tell)(?: me)?(?: an?)? (?:overview|summary|gist) of (?:the |this |my |these )?(?:uploaded )?" + _DOC_WORDS +
    r"|(?:summari[sz]e|overview of|gist of|main (?:topic|idea|points?) of) (?:the |this |my |these )?(?:uploaded )?" + _DOC_WORDS +
    r"|what(?:'s| is) (?:the )?main (?:topic|idea|point) of (?:the |this |my |these )?(?:uploaded )?" + _DOC_WORDS + r")",
    re.I,
)


def is_document_question(question: str) -> bool:
    """Questions about the uploaded material as a whole ("what is the pdf about?"). No single passage
    matches these well, so they are answered with a summary of the document instead of passage QA."""
    return bool(_DOCUMENT_QUESTION.search(question or ""))
