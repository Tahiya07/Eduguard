"""Shared helpers for student/teacher RAG quality control."""

from __future__ import annotations

import logging
import re

import numpy as np
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

    # PDF text breaks lines wherever the page wrapped: a single newline is a wrap, not a sentence end.
    # Keep actual source sentences; rank them by informative-term coverage and
    # a small lead bias, then restore their original order for readability.
    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", re.sub(r"(?<!\n)\n(?!\n)", " ", merged))
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


_GROUNDING_STOPWORDS = frozenset(
    "the a an and or but of to in on at by for with from into as is are was were be been being this that these those "
    "it its they them their there here which who whom whose what when where why how than then so such can could may "
    "might must should would will shall do does did not no also very more most other some any each both all only own "
    "same just about over under between because while during before after above below up down out off again further "
    "once if into your you we our us i me my he she his her "
    # Generic connective words a correct paraphrase adds; they say nothing about where the content came from.
    "often usually typically generally commonly mainly mostly through via using use used uses allow allows allowed "
    "allowing enable enables enabled make makes made making set sets get gets help helps involve involves involving "
    "known called refers refer referred example examples include includes including such like way ways thing things "
    "one two three four five several many much well even still already within without across among per".split()
)


def _content_stems(text: str) -> set[str]:
    """Content words cut to a 5-letter stem, so 'organizes'/'organizing' and 'tables'/'table' match."""
    words = re.findall(r"[a-z0-9]+", _normalize(text))
    # Drop a plural "s" first: otherwise short words differ ("call" -> "call", "calls" -> "calls").
    return {(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)[:5]
            for w in words if len(w) > 2 and w not in _GROUNDING_STOPWORDS}


def keep_supported_sentences(answer: str, chunk_texts: Sequence[str], *, min_support: float = 0.5) -> str:
    """Keep only sentences whose content words mostly appear in the notes; refuse if none remain.

    Used when the user has uploaded notes: an answer must come from them, so a sentence built
    from outside knowledge (e.g. "The capital of France is Paris." against OS notes) is dropped.
    """
    text = (answer or "").strip()
    if not text or text == _REFUSAL:
        return _REFUSAL
    source = set()
    for chunk in chunk_texts:
        source |= _content_stems(chunk)
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        stems = _content_stems(sentence)
        if len(stems) < 3:  # too short to judge (e.g. "Yes."): keep with its neighbours
            kept.append(sentence)
            continue
        if len(stems & source) / len(stems) >= min_support:
            kept.append(sentence)
    result = " ".join(s for s in kept if s).strip()
    # A short answer such as "96%" or "243 days" has no long content words but is still an answer.
    if not result or not re.search(r"\w", result):
        return _REFUSAL
    return result


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
    on_copy: str = "refuse",
    max_sentences: int = 3,
) -> str:
    """Strip context echoes ([n] labels, "Context:" lines) and handle near-verbatim copies.

    on_copy="keep" (summaries): returned as is; summaries legitimately reuse source wording.
    on_copy="refuse": a near-verbatim dump becomes the refusal (the caller falls back).
    on_copy="trim" (question answering): a short answer that reuses the notes' wording is the
    correct, grounded answer and is kept; a long verbatim dump is cut to its first
    ``max_sentences`` sentences instead of being thrown away. Protected-content screening is
    done separately by PrivacyGuard, not here.
    """
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

    if on_copy == "keep":
        # Summaries: reusing the source's wording is expected, so overlap is never a reason to discard one.
        return text or _REFUSAL
    ratio = chunk_copy_ratio(text, chunk_texts)
    if on_copy == "trim":
        # Faithful answers from short notes naturally overlap them; only shorten real passage dumps.
        if ratio >= max_copy_ratio and len(text.split()) > 60:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sentences[:max_sentences]).strip()
        return text or _REFUSAL
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
    # A faithful summary of short notes naturally reuses their wording; the former 92% overlap rule
    # replaced good summaries with the extractive fallback. Keep the model's summary.
    text = sanitize_rag_answer(answer, chunk_texts, on_copy="keep")
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


# ---------------------------------------------------------------------------
# Notes QA helpers: list-type questions and key-term coverage
# ---------------------------------------------------------------------------
_ENUM_QUESTION = re.compile(
    r"\b(all|every|each|types?|kinds?|categor(?:y|ies)|classes|forms|list|enumerate|mention|name the|"
    r"what are the|which are the|outline|steps|stages|phases|components|features|advantages|disadvantages|"
    r"benefits|limitations|examples|differences)\b",
    re.I,
)
_GENERIC_QUESTION_WORDS = {
    "what", "which", "when", "where", "why", "how", "who", "whom", "whose", "does", "do", "did", "is", "are", "was",
    "were", "the", "this", "that", "these", "those", "there", "their", "them", "they", "with", "from", "into", "about",
    "for", "and", "or", "of", "in", "on", "to", "a", "an", "all", "every", "each", "any", "some", "type", "types", "kind",
    "kinds", "list", "mention", "name", "names", "explain", "describe", "define", "give", "tell", "show", "state",
    "included", "include", "includes", "mentioned", "discussed", "covered", "given", "listed", "notes", "note", "pdf",
    "document", "lecture", "slides", "slide", "material", "uploaded", "please", "main", "different", "various",
    "according", "between", "also", "other", "more", "most", "many", "much", "can", "could", "should", "would", "will",
    "have", "has", "had", "been", "being", "its", "it's", "your", "my", "our", "you", "use", "used", "using", "example",
    "examples", "briefly", "detail", "details", "summary", "summarize", "category", "categories", "classes", "forms",
}


def is_enumerative_question(question: str) -> bool:
    """"Mention all types of threats", "list the measures": answers that need every relevant section."""
    return bool(_ENUM_QUESTION.search(question or ""))


def question_key_terms(question: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9-]{2,}", (question or "").lower())
    return [w for w in words if w not in _GENERIC_QUESTION_WORDS and len(w) >= 4]


def notes_contain_terms(question: str, notes_text: str) -> bool:
    """True when every key term of the question occurs in the notes (plural/singular tolerant)."""
    terms = question_key_terms(question)
    if not terms:
        return False
    text = (notes_text or "").lower()

    def found(term: str) -> bool:
        stems = {term, term.rstrip("s"), term[:-2] if term.endswith("es") else term,
                 term[:-3] + "y" if term.endswith("ies") else term}
        return any(re.search(r"\b" + re.escape(s), text) for s in stems if len(s) >= 3)

    return all(found(t) for t in terms)


# ---------------------------------------------------------------------------
# Hybrid retrieval and question-focused context
# ---------------------------------------------------------------------------
# Dense (BGE) similarity cannot tell "section 3" from "section 33", or find a rare code or term the
# embedding barely encodes. Measured: the answer chunk for "What is the value for section 3?" ranked
# 11th of 75 by BGE alone, so it never reached the model. Exact-term overlap fixes that ordering.
HYBRID_LEXICAL_WEIGHT = 0.12


def _lexical_terms(question: str) -> list[str]:
    q = (question or "").lower()
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?", q)
    words = [w for w in re.findall(r"[a-z][a-z0-9-]{3,}", q) if w not in _GENERIC_QUESTION_WORDS]
    return list(dict.fromkeys(numbers + words))


def _term_in(term: str, text: str) -> bool:
    if term[0].isdigit():
        return re.search(r"(?<![\d.])" + re.escape(term) + r"(?![\d])", text) is not None
    return re.search(r"\b" + re.escape(term[:-1] if term.endswith("s") and len(term) > 4 else term), text) is not None


def hybrid_rerank(question: str, results: Sequence[RetrievalResult], keep: int = 8,
                  weight: float = HYBRID_LEXICAL_WEIGHT) -> List[RetrievalResult]:
    """Re-order dense results by cosine + weight * IDF-weighted share of the question's exact terms
    (numbers and content words) found in each chunk. Cosine values are left unchanged."""
    import math

    results = list(results)
    terms = _lexical_terms(question)
    if not results or not terms:
        return results[:keep]
    texts = [str(r.text or "").lower() for r in results]
    n = len(texts)
    idf = {t: math.log(1 + n / (1 + sum(_term_in(t, x) for x in texts))) for t in terms}
    total = sum(idf.values()) or 1.0
    lex = [sum(idf[t] for t in terms if _term_in(t, x)) / total for x in texts]
    order = sorted(range(n), key=lambda i: float(results[i].cosine) + weight * lex[i], reverse=True)
    return [results[i] for i in order[:keep]]


def key_sentences(question: str, chunk_texts: Sequence[str], encoder: Any, n: int = 4) -> List[str]:
    """The n sentences of the given chunks most relevant to the question (dense similarity plus exact
    terms), in their original order. Shown to the model right before the question: a 1.5B model
    reading ~4,000 characters otherwise often picks a nearby wrong number or answers from memory."""
    sents: List[str] = []
    for t in chunk_texts:
        flat = re.sub(r"\s*\n\s*", " ", str(t or ""))
        for s in re.split(r"(?<=[.!?])\s+", flat):
            s = s.strip()
            if len(s.split()) >= 4 and s not in sents:
                sents.append(s)
    if len(sents) <= n:
        return sents
    try:
        q = np.asarray(encoder.encode_queries([question]))[0]
        e = np.asarray(encoder.encode_passages(sents))
        dense = e @ q
    except Exception:  # no encoder: lexical only
        dense = np.zeros(len(sents))
    terms = _lexical_terms(question)
    lex = np.array([sum(_term_in(t, s.lower()) for t in terms) / max(1, len(terms)) for s in sents])
    score = dense + 0.25 * lex
    top = sorted(np.argsort(-score)[:n].tolist())
    return [sents[i] for i in top]


def _split_sentences(text: str) -> List[str]:
    flat = re.sub(r"\s*\n\s*", " ", str(text or ""))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.split()) >= 4]


def sentence_rescore(question: str, results: Sequence[RetrievalResult], encoder: Any, cache: dict,
                     keep: int = 8, first_pass: int = 12, max_candidates: int = 30,
                     weight: float = HYBRID_LEXICAL_WEIGHT) -> List[RetrievalResult]:
    """Rank chunks by their best-matching *sentence* as well as the whole chunk.

    One vector per chunk mostly reflects the chunk's main topic, so a single fact inside an
    unrelated chunk is missed (measured: the chunk holding "the value for section 3 is 21" inside
    calculus text scored 0.51 and ranked 29th). Candidates are the best chunks by meaning + exact
    terms, plus chunks containing a rare exact term of the question; each candidate is scored by
    max(chunk score, best sentence score) + the exact-term bonus. Sentence vectors are cached.
    """
    import math

    results = list(results)
    if not results:
        return results
    terms = _lexical_terms(question)
    texts = [str(r.text or "").lower() for r in results]
    n = len(texts)
    df = {t: sum(_term_in(t, x) for x in texts) for t in terms}
    idf = {t: math.log(1 + n / (1 + df[t])) for t in terms}
    total = sum(idf.values()) or 1.0
    lex = [sum(idf[t] for t in terms if _term_in(t, x)) / total for x in texts]
    base = [float(r.cosine) + weight * lex[i] for i, r in enumerate(results)]
    order = sorted(range(n), key=lambda i: base[i], reverse=True)
    rare = [t for t in terms if 0 < df[t] <= max(3, 0.1 * n)]
    cand = list(dict.fromkeys(order[:first_pass] + [i for i in order if any(_term_in(t, texts[i]) for t in rare)]))[:max_candidates]
    try:
        q = np.asarray(encoder.encode_queries([question]))[0]
        sents = {i: _split_sentences(results[i].text) for i in cand}
        missing = list(dict.fromkeys(s for ss in sents.values() for s in ss if s not in cache))
        if missing:
            for s, v in zip(missing, np.asarray(encoder.encode_passages(missing))):
                cache[s] = v
        best_sent = {i: max((float(cache[s] @ q) for s in sents[i]), default=0.0) for i in cand}
    except Exception:  # no encoder available: keep the chunk-level order
        return [results[i] for i in order[:keep]]
    from dataclasses import replace

    score = {i: max(float(results[i].cosine), best_sent[i]) + weight * lex[i] for i in cand}
    # Returned chunks carry max(chunk, best sentence) as their cosine, so later relevance filters keep a
    # chunk whose best sentence answers the question. (The "notes cover this" gate keeps using the
    # whole-chunk scores; see backend/service.py.)
    return [replace(results[i], cosine=max(float(results[i].cosine), best_sent[i]))
            for i in sorted(cand, key=lambda i: score[i], reverse=True)[:keep]]
