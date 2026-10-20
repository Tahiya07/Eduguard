"""Shared helpers for student/teacher RAG quality control."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Sequence

from retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Chunks below this cosine similarity are dropped before generation.
DEFAULT_MIN_COSINE = 0.22
DOCUMENT_SUMMARY_MIN_COSINE = 0.05

_REFUSAL = "I don't know based on the provided context."

_DOCUMENT_SUMMARY_CUES = (
    "summar",
    "overview",
    "main point",
    "key point",
    "this pdf",
    "this document",
    "the pdf",
    "the document",
    "uploaded",
    "whole document",
    "entire document",
    "what is in",
    "what does this",
    "tell me about this",
)


def is_document_summary_query(query: str) -> bool:
    q = _normalize(query)
    if not q:
        return False
    return any(cue in q for cue in _DOCUMENT_SUMMARY_CUES)


def boost_summary_retrieval_query(
    query: str,
    corpus_chunks: Sequence[str],
) -> str:
    """Anchor retrieval in document content when the user asks to summarize the upload."""
    sample = " ".join((t or "").strip() for t in corpus_chunks[:4] if (t or "").strip())
    sample = re.sub(r"\s+", " ", sample)[:500]
    if sample:
        return f"central topic main claims key findings important details methods limitations conclusions {sample}"
    return f"document main topics key claims findings important details {query.strip()}"


def merge_corpus_for_summary(chunks: Sequence[Any], *, max_chars: int = 7200) -> str:
    """Merge document chunks in source order while removing ingestion overlap."""
    ordered = sorted(
        chunks,
        key=lambda c: (str(getattr(c, "source", "")), int(getattr(c, "page", 0) or 0), int(getattr(c, "chunk_id", 0))),
    )
    seen: set[str] = set()
    parts: List[str] = []
    total = 0

    def token_key(token: str) -> str:
        return re.sub(r"\W+", "", token.lower())

    for chunk in ordered:
        text = str(getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        key = _normalize(text[:280])
        if key in seen:
            continue
        seen.add(key)

        # Ingestion uses overlapping word windows. Remove the longest suffix
        # of the previous chunk that repeats at the start of this chunk, while
        # retaining the source's original punctuation and line breaks.
        if parts:
            previous_words = list(re.finditer(r"\S+", parts[-1]))
            current_words = list(re.finditer(r"\S+", text))
            max_overlap = min(160, len(previous_words), len(current_words))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                left = [token_key(m.group()) for m in previous_words[-size:]]
                right = [token_key(m.group()) for m in current_words[:size]]
                if all(left) and left == right:
                    overlap = size
                    break
            if overlap >= 8:
                text = text[current_words[overlap - 1].end():].lstrip()

        if not text:
            continue
        separator = "\n\n" if parts else ""
        room = max_chars - total - len(separator)
        if room <= 0:
            break
        if len(text) > room:
            cut = text[:room]
            # Prefer a complete sentence/line at the budget boundary.
            boundary = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("\n"))
            if boundary >= room // 2:
                cut = cut[:boundary + (1 if cut[boundary] == "." else 0)]
            else:
                cut = cut.rsplit(" ", 1)[0]
            if cut.strip():
                parts.append(cut.strip())
            break
        parts.append(text)
        total += len(separator) + len(text)
    return "\n\n".join(parts)

def merged_document_retrieval_result(chunks: Sequence[Any], *, max_chars: int = 7200) -> List[RetrievalResult]:
    body = merge_corpus_for_summary(chunks, max_chars=max_chars)
    if not body.strip():
        return []
    return [
        RetrievalResult(
            rank=1,
            doc_id=0,
            text=body,
            cosine=1.0,
        )
    ]


def build_document_summary_task(source_text: str = "") -> str:
    """Build a document-aware, source-faithful summary task for varied material."""
    text = source_text or ""
    low = text.lower()
    # Avoid classifying ordinary academic/project documents as a CV from generic
    # words such as "education", "projects", or "skills" alone.
    explicit_cv = any(marker in low for marker in (
        "curriculum vitae", "resume", "résumé", "professional summary",
        "work experience", "employment history",
    ))
    cv_heading_count = sum(
        1 for marker in ("education", "technical skills", "projects", "experience", "certifications")
        if re.search(rf"(?im)^\s*(?:\d+[.)]\s*)?{re.escape(marker)}\s*:?[ \t]*$", text)
    )
    looks_like_cv = explicit_cv or (
        cv_heading_count >= 3 and (
            "@" in text or "linkedin.com" in low or "github.com" in low
        )
    )
    if looks_like_cv:
        return (
            "DOCUMENT TYPE: CV/resume. Summarize it as a concise professional/academic profile in one "
            "coherent paragraph of 4-5 complete sentences. Lead with the person's identity or current "
            "academic/professional focus when stated; then cover the most relevant skills, experience, "
            "and strongest projects or achievements supported by the source. Preserve concrete technologies "
            "and evidence, but do not invent seniority, proficiency, metrics, or achievements. Write grammatical "
            "sentences that synthesize education, technical focus, and selected projects into a coherent narrative; "
            "explain what a project does only when the source states it. Exclude contact details, URLs, raw headings, "
            "and bullet-by-bullet transcription. Never join project titles, stack labels, dates, and URLs with separators. "
            "Ignore extraction/layout artifacts. Output only the profile."
        )
    return (
        "Summarize the supplied source itself; infer its document type from its content and structure, "
        "but do not announce or guess a type when uncertain. Adapt the summary to the material:\\n"
        "- For research papers, capture the research problem or aim, approach/method, principal findings, "
        "and stated limitations or implications when available.\\n"
        "- For lecture notes or textbooks, explain the central concepts and how the important ideas relate; "
        "retain key definitions, formulas, examples, and distinctions when they carry meaning.\\n"
        "- For reports or case studies, preserve the purpose, evidence, results, and conclusions.\\n"
        "- For technical documentation or procedures, retain what the system/process does, important steps, "
        "constraints, and dependencies; do not turn steps into vague generalities.\\n"
        "- For essays or other prose, preserve the thesis, main supporting arguments, and qualifications.\\n"
        "Use only categories that are actually present; do not force a paper-style methods/findings structure "
        "onto other material. Prioritize central information over minor examples, but keep essential names, "
        "technical terms, figures, equations, dates, and caveats accurate. Explain relationships rather than "
        "listing disconnected keywords. Remove repeated overlap and extraction noise without losing meaning. "
        "Do not invent facts, causal links, results, or conclusions. Write a concise, coherent summary "
        "(usually 4-6 sentences, shorter for a short source); use brief labeled sections or bullets only when "
        "that structure materially improves a complex or structured source. Start with the substantive main "
        "point and finish with a complete sentence. Output only the summary."
    )


_META_SUMMARY_MARKERS = (
    "the topic of the query",
    "the query \"",
    "the query '",
    "is asking for",
    "the context provides",
    "relevant to summarizing",
    "which are relevant to",
)


def looks_like_meta_summary(text: str) -> bool:
    low = _normalize(text)
    return sum(1 for m in _META_SUMMARY_MARKERS if m in low) >= 2


def extractive_document_summary(chunk_texts: Sequence[str]) -> str:
    """Return a domain-neutral, deterministic extractive summary as a safe fallback."""
    merged = merge_corpus_for_summary(
        [type("_C", (), {"text": t, "source": "", "page": 0, "chunk_id": i})() for i, t in enumerate(chunk_texts)],
        max_chars=9000,
    )
    if not merged.strip():
        return _REFUSAL

    # Keep actual source sentences; rank them by informative-term coverage and
    # a small lead bias, then restore their original order for readability.
    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", merged)
        if len(re.findall(r"\w+", part)) >= 5
    ]
    if not sentences:
        return re.sub(r"\s+", " ", merged).strip()[:900]

    stop_words = {
        "about", "after", "again", "also", "among", "because", "been", "being",
        "between", "both", "could", "does", "each", "from", "have", "into",
        "more", "most", "other", "over", "same", "some", "such", "than", "that",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "under", "very", "were", "what", "when", "where", "which",
        "while", "with", "would", "your", "the", "and", "for", "are", "was",
        "but", "not", "you", "its", "our", "can", "may", "has", "had", "all",
        "any", "one", "use", "used", "using", "will", "within", "from", "per",
    }
    tokenized = [
        [w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", sentence)]
        for sentence in sentences
    ]
    freq: dict[str, int] = {}
    for words in tokenized:
        for word in set(words):
            if len(word) > 2 and word not in stop_words:
                freq[word] = freq.get(word, 0) + 1

    scored = []
    for i, (sentence, words) in enumerate(zip(sentences, tokenized)):
        informative = [w for w in words if len(w) > 2 and w not in stop_words]
        lexical = sum(freq.get(w, 0) for w in set(informative)) / max(1, len(set(informative)))
        lead_bonus = 1.0 / (1.0 + i * 0.35)
        length_penalty = min(1.0, 24 / max(24, len(words)))
        scored.append(((lexical * 0.8 + lead_bonus * 0.2) * length_penalty, i, sentence))

    target_count = min(5, max(2, round(len(sentences) * 0.22)))
    selected = sorted(scored, reverse=True)[:target_count]
    selected = sorted(selected, key=lambda item: item[1])
    summary = " ".join(item[2] for item in selected).strip()
    return summary or re.sub(r"\s+", " ", merged).strip()[:900]


def filter_relevant_chunks(
    chunks: Sequence[RetrievalResult],
    *,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> List[RetrievalResult]:
    """Keep only chunks that are semantically close to the query."""
    if not chunks:
        return []
    kept = [c for c in chunks if float(c.cosine) >= float(min_cosine)]
    return list(kept) if kept else [max(chunks, key=lambda c: float(c.cosine))]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize(text)))


def chunk_copy_ratio(answer: str, chunk_texts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in the best-matching source chunk."""
    ans = _token_set(answer)
    if not ans:
        return 0.0
    best = 0.0
    for chunk in chunk_texts:
        cset = _token_set(chunk)
        if not cset:
            continue
        overlap = len(ans & cset) / len(ans)
        best = max(best, overlap)
    return best


def sanitize_rag_answer(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    max_copy_ratio: float = 0.88,
) -> str:
    """Strip verbatim chunk dumps; refuse when the model only pasted context."""
    text = (answer or "").strip()
    if not text:
        return _REFUSAL

    # Remove numbered context echoes like "[1] ..." or "Context:" blocks.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\[\d+\]\s", stripped):
            continue
        if stripped.lower().startswith(("context:", "bounded context", "[bounded context]")):
            continue
        lines.append(stripped)
    if lines:
        text = " ".join(lines).strip()
    else:
        text = re.sub(r"^\[\d+\]\s*", "", text).strip()
    text = re.sub(r"\[\d+\]\s*", "", text).strip()

    ratio = chunk_copy_ratio(text, chunk_texts)
    # Refuse only when the model dumped a long passage with almost no rewriting.
    if ratio >= max_copy_ratio and len(_token_set(text)) >= 20:
        return _REFUSAL
    return text


def trim_chunks_for_context(
    chunks: Sequence[RetrievalResult],
    *,
    max_chars_per_chunk: int = 700,
    max_total_chars: int = 2800,
) -> List[RetrievalResult]:
    """Trim long chunks so the LLM prompt fits and stays focused."""
    out: List[RetrievalResult] = []
    total = 0
    for chunk in chunks:
        body = (chunk.text or "").strip()
        if len(body) > max_chars_per_chunk:
            cut = body[:max_chars_per_chunk]
            for sep in (". ", ".\n", "\n"):
                j = cut.rfind(sep)
                if j > max_chars_per_chunk // 3:
                    body = cut[: j + len(sep)].strip()
                    break
            else:
                body = cut.strip()
        if total + len(body) > max_total_chars and out:
            break
        out.append(
            RetrievalResult(
                rank=chunk.rank,
                doc_id=chunk.doc_id,
                text=body,
                score=chunk.score,
                cosine=chunk.cosine,
                infonce_risk=chunk.infonce_risk,
                privacy_score=chunk.privacy_score,
                l2_distance=chunk.l2_distance,
                bloom_level=chunk.bloom_level,
                topic=chunk.topic,
                difficulty=chunk.difficulty,
                final_score=chunk.final_score,
            )
        )
        total += len(body)
    return out


def sanitize_summary_answer(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    finish_reason: str | None = None,
) -> str:
    if finish_reason == "length":
        logger.warning("Summary generation hit max_tokens; using a source-derived fallback rather than returning a possibly incomplete answer")
        return extractive_document_summary(chunk_texts)
    text = sanitize_rag_answer(answer, chunk_texts, max_copy_ratio=0.92)
    if text == _REFUSAL or looks_like_meta_summary(text):
        return extractive_document_summary(chunk_texts)

    # A summary ending without sentence-final punctuation is often a completion
    # cut (e.g. "... links PDF/DOCX files into a"). Never display that as a
    # successful summary; fall back to source-derived sentences instead.
    stripped = text.rstrip().strip('"\u201c\u201d\u2018\u2019* ')
    if stripped and stripped[-1] not in ".!?":
        logger.warning("Summary output appears truncated; using extractive fallback")
        return extractive_document_summary(chunk_texts)
    if re.search(r"\b(?:a|an|the|and|or|but|to|of|for|with|into|from|by|as|is|are|was|were|that|which|who|such|including)[.!?]?$", stripped, flags=re.I):
        logger.warning("Summary output ends with a dangling phrase; using extractive fallback")
        return extractive_document_summary(chunk_texts)
    return text
