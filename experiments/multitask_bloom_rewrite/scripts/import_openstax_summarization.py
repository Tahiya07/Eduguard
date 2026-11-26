"""Extract (section text -> author-written section summary) pairs from OpenStax textbooks (CNXML sources).

Source: the official OpenStax book repositories (https://github.com/openstax/osbooks-*), shallow-cloned with only
collections/ and modules/ checked out. Each section module of these books ends with an author-written summary
(<section class="summary|section-summary">, titled e.g. "Key Concepts and Summary", "Section Summary", "Summary").
Some books put the per-section summaries in a chapter-review module instead; those are matched to the section
by title.

Usage (repository root):
  python -I experiments/multitask_bloom_rewrite/scripts/import_openstax_summarization.py RAW_DIR OUT.jsonl

RAW_DIR holds one clone per repository (any folder names). Output: one JSON object per candidate pair with
book, licence, chapter, section, repo commit, module id, source_text and summary. Nothing is filtered by length
here except obviously unusable pairs; selection happens in build_summarization_v3.py.

Text rendering (plain text, no Markdown): section/sub-section titles become their own lines; paragraphs and list
items are kept; figures, media, tables, exercises, glossaries, learning objectives, footnotes, link-up / interactive
notes (all boxed features: link-ups, visual-connection questions, references, career boxes...), paragraphs that
carry a URL (citations), review/critical-thinking/self-check sections and the summary itself are removed from the
source. Cross-references to removed figures/tables ("(see Figure 3)", "as shown in Figure 3") are removed; any other
empty cross-reference becomes "the figure" / "the table". Headings with no text under them are dropped.
Modules with more than MAX_MATH MathML expressions are skipped (math does not survive plain-text rendering).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

NS = {"c": "http://cnx.rice.edu/cnxml", "col": "http://cnx.rice.edu/collxml", "md": "http://cnx.rice.edu/mdml"}
C = "{http://cnx.rice.edu/cnxml}"
MATH = "{http://www.w3.org/1998/Math/MathML}math"
SUMMARY_CLASSES = {"summary", "section-summary"}
DROP_TAGS = {"figure", "media", "table", "exercise", "glossary", "footnote", "equation", "code", "preformat"}
DROP_SECTION_HINTS = ("summary", "review", "critical-thinking", "self-check", "problems", "exercise", "question",
                      "reference", "glossary", "key-terms", "key-equations", "further", "assessment", "check-your",
                      "conceptual", "writing", "application", "lab", "case-study-questions", "bibliography",
                      "objectives", "outcomes", "ap-connection", "science-practice", "ap-test-prep",
                      "suggested-reading", "bibliograph")
INLINE_SKIP = {C + "footnote", C + "note", MATH}  # inline citation notes (<cite><note class="reference">) too
OBJECTIVES = re.compile(r"^(learning (objectives|outcomes)|by the end of this section)", re.I)
DROP_NOTE_HINTS = ("linkup", "link-to-learning", "interactive", "everyday-connection-video", "media-2", "watch",
                   "explore", "simulation", "ost-assignable", "socratic", "web")
MAX_MATH = 3


def classes(el) -> set[str]:
    return set((el.get("class") or "").split())


TARGET_KIND: dict[str, str] = {}
_REF = r"(?:the )?(?:figure|table|example|equation)s?"
_REF_PATTERNS = [
    re.compile(rf"\s*\((?:see |refer to |as shown in |shown in |in )?{_REF}(?:(?:,| and| or) {_REF})*\)", re.I),
    re.compile(rf",?\s+(?:as )?(?:shown|illustrated|seen|depicted|summarized|listed|presented) (?:in|by) {_REF}(?=[\s.,;])",
               re.I),
    re.compile(r"\s*\(\s*\)"),
]


def clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t.replace(" ", " ")).strip()
    for pat in _REF_PATTERNS:
        t = pat.sub("", t)
    return re.sub(r"\s+([.,;:])", r"\1", t).strip()


def text_of(el) -> str:
    """Inline text of an element, skipping footnotes and math."""
    parts = []

    def walk(e):
        if e.tag in INLINE_SKIP:
            if e.tail:
                parts.append(e.tail)
            return
        if e.tag == C + "link" and not "".join(e.itertext()).strip():
            kind = TARGET_KIND.get(e.get("target-id", ""), "")
            parts.append({"figure": " the figure", "table": " the table", "example": " the example",
                          "equation": " the equation"}.get(kind, " the related section" if e.get("document") else ""))
            return
        if e.text:
            parts.append(e.text)
        for ch in e:
            walk(ch)
            if ch.tag not in INLINE_SKIP and ch.tail:
                parts.append(ch.tail)

    if el.text:
        parts.append(el.text)
    for ch in el:
        walk(ch)
        if ch.tag not in INLINE_SKIP and ch.tail:
            parts.append(ch.tail)
    return clean("".join(parts))


def render(el, out: list[str], drop_sections: bool = True) -> None:
    skip_list = False
    for ch in el:
        tag = ch.tag.replace(C, "")
        cl = classes(ch)
        if tag in DROP_TAGS:
            continue
        if skip_list and tag == "list":  # the objectives list after "By the end of this section, you will be able to:"
            skip_list = False
            continue
        skip_list = False
        if drop_sections and tag == "para" and OBJECTIVES.match(text_of(ch)):
            skip_list = True
            continue
        if tag == "section":
            if drop_sections and any(h in c for c in cl for h in DROP_SECTION_HINTS):
                continue
            t = ch.find("c:title", NS)
            if drop_sections and t is not None and OBJECTIVES.match(text_of(t)):
                continue
            render(ch, out, drop_sections)
        elif tag == "note":
            continue
        elif tag == "title":
            t = text_of(ch)
            if t:
                out.append(t)
        elif tag == "para":
            if ch.find("c:list", NS) is not None or ch.find("c:figure", NS) is not None:
                t = clean((ch.text or ""))
                if t:
                    out.append(t)
                render(ch, out, drop_sections)
            else:
                t = text_of(ch)
                if t:
                    out.append(t)
        elif tag == "list":
            for it in ch.findall("c:item", NS):
                t = text_of(it)
                if t:
                    out.append(t)
        elif tag in ("div", "content", "example"):
            render(ch, out, drop_sections)


def as_prose(lines: list[str]) -> str:
    out = []
    for t in lines:
        t = t.strip()
        if t and not re.search(r"[.!?:;)\"']$", t):
            t += "."
        out.append(t)
    return " ".join(out)


def summary_sections(root):
    return [s for s in root.iter(C + "section") if classes(s) & SUMMARY_CLASSES]


def summary_text(sec) -> str:
    lines: list[str] = []
    render(sec, lines, drop_sections=False)
    title = sec.find("c:title", NS)
    if title is not None and lines and lines[0] == text_of(title):
        lines = lines[1:]
    return as_prose(lines)


def collection_entries(path: Path):
    """(book title, licence url, chapter title, module id) in reading order."""
    root = ET.parse(path).getroot()
    book = root.find("col:metadata/md:title", NS).text
    lic = root.find("col:metadata/md:license", NS).get("url")
    entries = []

    def walk(node, chapter):
        for ch in node:
            if ch.tag.endswith("subcollection"):
                title = ch.find("md:title", NS).text
                walk(ch.find("col:content", NS), title if chapter is None else chapter)
            elif ch.tag.endswith("module"):
                entries.append((book, lic, chapter, ch.get("document")))

    walk(root.find("col:content", NS), None)
    return entries


def module_parts(path: Path):
    root = ET.parse(path).getroot()
    for el in root.iter():
        if el.get("id"):
            TARGET_KIND[el.get("id")] = el.tag.replace(C, "")
    title = clean(root.find("c:title", NS).text or "")
    content = root.find("c:content", NS)
    return root, title, content


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    raw, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows, stats = [], Counter()
    seen_modules: set[tuple[str, str]] = set()  # module ids are repository-local
    for repo in sorted(p for p in raw.iterdir() if (p / "collections").is_dir()):
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        origin = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"], capture_output=True,
                                text=True).stdout.strip()
        for coll in sorted((repo / "collections").glob("*.xml")):
            entries = collection_entries(coll)
            parsed = {}
            for book, lic, chapter, mid in entries:
                f = repo / "modules" / mid / "index.cnxml"
                if f.is_file():
                    parsed[mid] = module_parts(f)
            # chapter-review modules: per-section summaries titled by section name
            review_map: dict[tuple[str, str], str] = {}
            for book, lic, chapter, mid in entries:
                if mid not in parsed:
                    continue
                root, title, content = parsed[mid]
                for s in summary_sections(root):
                    for sub in s.findall("c:section", NS):
                        t = sub.find("c:title", NS)
                        if t is not None:
                            review_map[(chapter, re.sub(r"^[\d.]+\s*", "", text_of(t)).lower())] = summary_text(sub)
            for idx, (book, lic, chapter, mid) in enumerate(entries):
                if mid not in parsed:
                    continue
                stats["modules"] += 1
                root, title, content = parsed[mid]
                if content is None:
                    continue
                own = [s for s in summary_sections(root) if not s.findall("c:section", NS)]
                summ = summary_text(own[0]) if own else review_map.get((chapter, title.lower()), "")
                if not summ:
                    stats["no_summary"] += 1
                    continue
                if len(list(root.iter(MATH))) > MAX_MATH:
                    stats["math_heavy"] += 1
                    continue
                lines: list[str] = [title]
                render(content, lines)
                lines = [t for t in lines if not re.search(r"https?://|www\.", t)]
                titles = {text_of(t) for t in root.iter(C + "title")}
                # drop headings that have no body text before the next heading or the end
                lines = [t for i, t in enumerate(lines)
                         if not (t in titles and (i + 1 == len(lines) or lines[i + 1] in titles))]
                source = "\n\n".join(lines)
                if (repo.name, mid) in seen_modules:
                    stats["module_in_several_books"] += 1
                    continue
                seen_modules.add((repo.name, mid))
                rows.append({"book": book, "license": lic, "chapter": chapter, "section": title,
                             "module": mid, "repo": origin, "commit": commit, "order": idx,
                             "summary_location": "section" if own else "chapter_review",
                             "source_text": source, "summary": summ})
                stats["pairs"] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(stats))
    words = lambda t: len(t.split())
    by = Counter()
    for r in rows:
        n = words(r["source_text"])
        cat = "short" if n <= 600 else "medium" if n <= 1200 else "long" if n <= 2500 else "very_long" if n <= 4000 else "over"
        by[(r["book"], cat)] += 1
    for b in sorted({r["book"] for r in rows}):
        print(f"{b[:45]:45s}", {c: by[(b, c)] for c in ("short", "medium", "long", "very_long", "over")})
    tot = Counter(c for (_, c), v in by.items() for _ in range(v))
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()
