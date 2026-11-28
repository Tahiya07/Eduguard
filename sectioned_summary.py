"""Section-by-section summaries for documents longer than one model call can read.

A single summary call sees at most ``generator_summary_source_chars`` (8,000 characters, about
1,300 words), so anything after that in a long lecture PDF was silently left out. Long documents
are instead split into consecutive sections (using the headings the chunker detected), each
section is summarized with the trained summary prompt, and the section summaries are joined in
document order. The whole document is covered, and bullet requests get one idea per section
instead of a paragraph cut into pieces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

# Each call yields roughly 80-100 words whatever its input length (the length the model learned), so
# the summary's length is set by how many sections there are. ~600-word sections give summaries of
# about 12-15% of the document; the former ~1,300-word sections gave 7-9%, which read as too thin.
SECTION_MAX_CHARS = 3600
SECTION_MIN_CHARS = 1200   # smaller neighbouring sections are merged so each call has enough to summarize
MAX_SECTIONS = 12          # beyond this, sections grow (up to HARD_MAX_CHARS) rather than multiply
HARD_MAX_CHARS = 8000      # one call's source budget (generator_summary_source_chars)
SHORT_SECTION_SUMMARY_WORDS = 25  # a section summary shorter than this (or <5% of its section) is retried


@dataclass
class Section:
    heading: Optional[str]
    text: str


def _ordered(chunks: Sequence[Any]) -> list[Any]:
    return sorted(chunks, key=lambda c: (str(getattr(c, "source", "")), int(getattr(c, "page", 0) or 0),
                                         int(getattr(c, "chunk_id", 0) or 0)))


def _strip_overlap(previous: str, current: str, min_words: int = 8) -> str:
    """Chunks overlap by whole sentences; drop the repeated start of ``current``."""
    prev_words, cur = previous.split(), current.split()
    for size in range(min(160, len(prev_words), len(cur)), min_words - 1, -1):
        if [w.lower() for w in prev_words[-size:]] == [w.lower() for w in cur[:size]]:
            return " ".join(cur[size:])
    return current


def sections_from_chunks(chunks: Sequence[Any]) -> list[Section]:
    """Consecutive chunks under the same heading become one section, overlap removed."""
    sections: list[Section] = []
    last_text = ""
    for c in _ordered(chunks):
        text = str(getattr(c, "text", "") or "").strip()
        if not text:
            continue
        text = _strip_overlap(last_text, text) if last_text else text
        last_text = str(getattr(c, "text", "") or "")
        heading = (getattr(c, "topic", None) or None)
        if sections and sections[-1].heading == heading:
            sections[-1].text += "\n" + text
        else:
            sections.append(Section(heading, text))
    return [s for s in sections if s.text.strip()]


def sections_from_text(text: str) -> list[Section]:
    """Pasted text has no chunk headings: use paragraphs as the units."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) <= 1:
        sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        paras = [" ".join(sents[i:i + 8]) for i in range(0, len(sents), 8)]
    return [Section(None, p) for p in paras]


def _units(text: str) -> list[str]:
    """Sentences (or lines); a single unit longer than one call's budget is cut at word boundaries."""
    out = []
    for u in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        u = u.strip()
        while len(u) > HARD_MAX_CHARS:
            cut = u.rfind(" ", 0, HARD_MAX_CHARS)
            cut = cut if cut > 0 else HARD_MAX_CHARS
            out.append(u[:cut]); u = u[cut:].strip()
        if u:
            out.append(u)
    return out


def _pack(sections: list[Section], target: int, min_chars: int = SECTION_MIN_CHARS) -> list[Section]:
    groups: list[Section] = []
    limit = min(int(HARD_MAX_CHARS * 1.15), int(target * 1.15))
    for s in sections:
        first = True
        for u in _units(s.text):
            cur = groups[-1] if groups else None
            size = len(cur.text) if cur else 0
            new_group = cur is None or size + len(u) + 1 > limit or (first and size >= 0.6 * target)
            if new_group and (cur is None or size >= min_chars):
                groups.append(Section(s.heading, u))
            else:
                cur.text += ("\n" if first else " ") + u
                if first and s.heading and s.heading not in (cur.heading or ""):
                    cur.heading = f"{cur.heading} / {s.heading}" if cur.heading else s.heading
            first = False
    return groups


def pack_sections(sections: list[Section], max_chars: int = SECTION_MAX_CHARS,
                  max_sections: int = MAX_SECTIONS, min_chars: int = SECTION_MIN_CHARS) -> tuple[list[Section], bool]:
    """Group the document's sentences, in order, into parts of about ``target`` characters; return
    (groups, truncated).

    target = max(max_chars, total / max_sections), capped at one call's budget. A new part starts at a
    section heading once the current part is reasonably full, so parts follow the document's topics;
    parts below SECTION_MIN_CHARS are merged with the next. If the result still has more than
    max_sections parts, the target grows and the document is packed again.
    """
    total = sum(len(s.text) for s in sections)
    target = min(HARD_MAX_CHARS, max(max_chars, -(-total // max_sections)))
    groups = _pack(sections, target, min_chars)
    while len(groups) > max_sections and target < HARD_MAX_CHARS:
        target = min(HARD_MAX_CHARS, int(target * 1.15) + 1)
        groups = _pack(sections, target, min_chars)
    return groups[:max_sections], len(groups) > max_sections


def _short_heading(heading: Optional[str], index: int) -> str:
    """The part's first section title (a part may span several slides; listing every title made
    labels like "A / B / …" that read as noise)."""
    if not heading:
        return f"Part {index}"
    first = next((p.strip() for p in heading.split(" / ") if p.strip()), "")
    first = first.rstrip(":").strip()
    return (first[:80] + "…") if len(first) > 80 else (first or f"Part {index}")


def summarize_sections(groups: list[Section], summarize: Callable[[str], str],
                       on_progress: Optional[Callable[[int, int, str], None]] = None) -> list[tuple[str, str]]:
    """Summarize each group; returns [(heading label, summary)] in document order."""
    out: list[tuple[str, str]] = []
    for i, g in enumerate(groups, 1):
        if on_progress:
            on_progress(i, len(groups), _short_heading(g.heading, i))
        summary = (summarize(g.text) or "").strip()
        if summary:
            out.append((_short_heading(g.heading, i), summary))
    return out


def trim_to_last_sentence(text: str) -> str:
    """A section summary cut by the token limit keeps its complete sentences."""
    text = (text or "").strip()
    end = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
    if text.endswith((".", "!", "?")):
        return text
    return text[: end + 1].strip() if end > 0 else ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", (text or "").strip()) if s.strip()]


def _quotas(n: int, weights: list[int]) -> list[int]:
    """Split n items over sections in proportion to weights, at least one each (n >= len(weights))."""
    k = len(weights)
    q = [1] * k
    for _ in range(n - k):
        i = max(range(k), key=lambda j: weights[j] / (q[j] + 1))
        q[i] += 1
    return q


def format_sections(parts: list[tuple[str, str]], kind: Optional[str], n: Optional[int]) -> str:
    """Apply a requested format along section boundaries, so no bullet or sentence mixes topics.

    bullets n: n >= sections -> each section gets a share of the bullets (split at its sentences);
               n <  sections -> neighbouring sections share a bullet.
    sentences n: the first sentences of each section, shared out the same way.
    words n: the first sentence of each section in order, until the word budget is used.
    """
    if not parts:
        return ""
    if kind is None:
        return combine(parts)
    sents = [_sentences(s) or [s] for _, s in parts]
    if kind == "words":
        budget, out = max(5, n or 60), []
        for i in range(max(len(x) for x in sents)):
            for ss in sents:
                if i < len(ss):
                    if out and len(" ".join(out + [ss[i]]).split()) > budget:
                        return " ".join(out)
                    out.append(ss[i])
        return " ".join(out)
    n = max(1, n or len(parts))
    if n >= len(parts):
        quota = _quotas(n, [len(x) for x in sents])
        picked = []
        for (h, _), ss, q in zip(parts, sents, quota):
            if kind == "sentences":
                picked.extend(ss[:q])
                continue
            # bullets: q roughly equal runs of this section's sentences
            size, extra = divmod(len(ss), q) if len(ss) >= q else (1, 0)
            i = 0
            for b in range(min(q, len(ss))):
                take = size + (1 if b < extra else 0)
                text = " ".join(ss[i:i + take]); i += take
                picked.append(f"{h}: {text}" if b == 0 else text)
        return " ".join(picked) if kind == "sentences" else "\n".join(f"- {p}" for p in picked)
    # fewer items than sections: consecutive sections share one item (their first sentences)
    size, extra = divmod(len(parts), n)
    items, i = [], 0
    for b in range(n):
        take = size + (1 if b < extra else 0)
        group = list(zip(parts[i:i + take], sents[i:i + take])); i += take
        heads = " / ".join(h for (h, _), _ in group)
        body = " ".join(ss[0] for _, ss in group)
        items.append(body if kind == "sentences" else f"{heads}: {body}")
    return " ".join(items) if kind == "sentences" else "\n".join(f"- {x}" for x in items)


def combine(section_summaries: list[tuple[str, str]], with_headings: bool = True) -> str:
    """Default output, structured: each section's title on its own line ending in ":", followed by
    one "- " bullet per point (sentence) of its summary; a blank line between sections. Consecutive
    parts of the same section share one title. Untitled parts (pasted text) get bullets only.
    The UI renders title lines in bold and "- " lines as a list."""
    if not with_headings:
        return " ".join(s for _, s in section_summaries)
    lines: list[str] = []
    last = None
    for h, s in section_summaries:
        titled = not re.fullmatch(r"Part \d+", h or "")
        if h != last:
            if lines:
                lines.append("")
            if titled:
                lines.append(f"{h}:")
            last = h
        points = _sentences(s) or [s]
        lines.extend(f"- {p}" for p in points)
    return "\n".join(lines)


def per_heading_groups(sections: list[Section], max_groups: int = 8, min_chars: int = 150) -> list[Section]:
    """Short slide decks: one part per slide title (titles stay attached to their own content).
    Very small slides join the next one; beyond max_groups, neighbours are paired."""
    groups: list[Section] = []
    for s in sections:
        if groups and len(groups[-1].text) < min_chars:
            g = groups[-1]
            g.text += "\n" + s.text
            if s.heading and s.heading not in (g.heading or ""):
                g.heading = f"{g.heading} / {s.heading}" if g.heading else s.heading
        else:
            groups.append(Section(s.heading, s.text))
    while len(groups) > max_groups:
        merged = []
        for i in range(0, len(groups), 2):
            pair = groups[i:i + 2]
            heading = " / ".join(h for h in (p.heading for p in pair) if h) or None
            merged.append(Section(heading, "\n".join(p.text for p in pair)))
        groups = merged
    return groups
