from __future__ import annotations
import os, re, threading, time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import logging
import psutil

# Keep API import time small: ML, OCR, embedding, and generation modules are
# imported only when the corresponding feature is first used.
from bloom_model_profiles import get_profile
from bloom_policy import compose_instruction
from role_access import Role, check_retriever_binding
from .config import ROOT as APP_ROOT, settings
from .label_store import LocalLabelStore
from operation_control import check_cancelled, llama_stopping_criteria, report_progress

logger = logging.getLogger(__name__)

# Pasted text shorter than this is treated as a request ("summarize my notes"), not as the text to summarize.
SUMMARY_MIN_SOURCE_WORDS = 30
# A rewrite this similar to its source (ROUGE-L F1) counts as a near-copy: the 95th percentile of
# source-vs-reference similarity over the training rewrite pairs (the evaluation's threshold).
REWRITE_NEAR_COPY_ROUGE_L = 0.5806


def _rouge_l_f(a: str, b: str) -> float:
    """Word-level ROUGE-L F1 (longest common subsequence), without the rouge_score dependency."""
    x = re.findall(r"[a-z0-9]+", (a or "").lower())
    y = re.findall(r"[a-z0-9]+", (b or "").lower())
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y, 1):
            cur.append(prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(y), lcs / len(x)
    return 2 * p * r / (p + r)

_ML_IMPORT_LOCK = threading.Lock()
_ML_IMPORTED = False


def _ensure_ml_imports() -> None:
    """Import the ML stack once, in one thread.

    transformers builds its public names lazily; when the warm-up thread and the first request
    imported it at the same time, one of them saw a half-initialised module ("cannot import name
    'AutoModel' from 'transformers'") and the request failed with a 500. Every entry point that
    may touch models calls this first, so the import happens exactly once and others wait for it.
    """
    global _ML_IMPORTED
    if _ML_IMPORTED:
        return
    with _ML_IMPORT_LOCK:
        if _ML_IMPORTED:
            return
        import transformers  # noqa: F401
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
        import peft  # noqa: F401

        import bloom_prompt  # noqa: F401  (imports predict_bloom -> peft/transformers)
        import models  # noqa: F401  (llama-cpp generator)
        _ML_IMPORTED = True


NOTES_NOT_COVERED = (
    "Your uploaded notes don't cover this question, so I can't answer it from them. Try rephrasing it, or ask about a topic that is in your notes."
)

class FrameworkService:
    """One application layer: role checks precede every corpus/model operation."""
    def __init__(self) -> None:
        self._ingestor_instance = None
        self._workspaces: dict[str, dict[str, Any]] = {}
        self._bloom: QwenBloomPredictor | None = None
        self._generator: RAGGenerator | None = None
        self._label_store: LocalLabelStore | None = None
        self._rewrite_llm: Any = None  # None = not loaded yet, False = unavailable
        self._rewrite_lock = threading.RLock()
        self._lock = threading.RLock()
        self._warmup_state: dict[str, Any] = {
            "state": "idle",
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def _ingestor(self):
        # ingestion imports PyTorch/PyMuPDF at module scope; defer it until
        # document ingestion is actually requested.
        if self._ingestor_instance is None:
            from ingestion import DocumentIngestor
            # ~120-word passages with overlap: small enough that one passage covers one idea (e.g. a list of
            # types) and is retrieved and passed to the model whole.
            self._ingestor_instance = DocumentIngestor(chunk_size=120, chunk_overlap=25)
        return self._ingestor_instance

    def workspace(self, sid: str) -> dict[str, Any]:
        _ensure_ml_imports()
        with self._lock:
            if sid not in self._workspaces:
                # Load the retriever only when a session first needs retrieval.
                # This import was previously module-level and is intentionally lazy.
                from retriever import PrivacyRetriever
                # Keep public/protected indexes separate, but defer encoder loading
                # until indexing or retrieval is requested. The encoder is shared lazily.
                public = PrivacyRetriever(lambda_privacy=0.1, encoder_profile=settings.retrieval_encoder)
                protected = PrivacyRetriever(lambda_privacy=0.5, encoder_profile=settings.retrieval_encoder)
                self._workspaces[sid] = {"public": public, "protected": protected, "chunks": {"public": [], "protected": []}, "moderation_reviews": []}
            return self._workspaces[sid]

    def clear_documents(self, sid: str, role: str, scope: str) -> dict[str, Any]:
        """Remove this session's uploaded material for one scope; chat falls back to general answers."""
        import shutil

        self._authorize_scope(role, scope)
        from retriever import PrivacyRetriever

        with self._lock:
            ws = self.workspace(sid)
            removed = len(ws["chunks"][scope])
            ws["chunks"][scope] = []
            ws.pop("sentence_vectors", None)  # cached sentence embeddings of the removed notes
            ws[scope] = PrivacyRetriever(lambda_privacy=0.1 if scope == "public" else 0.5,
                                         encoder_profile=settings.retrieval_encoder)
        store = settings.data_dir / sid / scope
        if store.is_dir():
            shutil.rmtree(store, ignore_errors=True)
        return {"cleared_chunks": removed, "scope": scope}

    def _bloom_path(self) -> Path:
        root = APP_ROOT
        if settings.bloom_model_dir:
            path = Path(settings.bloom_model_dir)
            return path if path.is_absolute() else (root / path).resolve()
        return (root / "models" / "qwen_bloom_merged0.5B").resolve()

    @staticmethod
    def _checkpoint_has_weights(path: Path) -> bool:
        return (path / "config.json").is_file() and (
            (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
        )

    def bloom_ready(self) -> tuple[bool, str]:
        path = self._bloom_path()
        if settings.bloom_use_quantized:
            from predict_bloom import is_deploy_checkpoint
            return is_deploy_checkpoint(path), str(path)
        return self._checkpoint_has_weights(path), str(path)

    def bloom(self):
        _ensure_ml_imports()
        from predict_bloom import QwenBloomPredictor
        with self._lock:
            if self._bloom is None:
                report_progress("Loading the Bloom classifier (first use only, a few seconds)...")
                ok, path = self.bloom_ready()
                if not ok:
                    raise RuntimeError(
                        "Centralized non-FL teacher Bloom classifier is not ready. Set BLOOM_MODEL_DIR to "
                        "models/qwen_bloom_merged0.5B "
                        "containing config.json and model.safetensors."
                    )
                self._bloom = QwenBloomPredictor(model_dir=path, model_size=settings.bloom_model_size, quantized=settings.bloom_use_quantized, prefer_quantized=settings.bloom_use_quantized)
            return self._bloom

    def load_bloom(self) -> dict[str, Any]:
        """Load the teacher-only classifier on demand after teacher authentication."""
        predictor = self.bloom()
        predictor._ensure_loaded()
        return {
            "loaded": True,
            "teacher_only": True,
            "checkpoint": Path(predictor.model_dir).name,
        }

    def classify_teacher_question(self, question: str) -> dict[str, Any]:
        """Teacher classification: reject non-questions first, because the classifier always
        returns one of the six levels (gibberish came back as Remember)."""
        from input_validation import require_question
        require_question(question)
        return self.classify(question)

    def classify(self, question: str) -> dict[str, Any]:
        from predict_bloom import BLOOM_LABELS
        from uncertainty import UncertaintyEngine
        from bloom_policy import BloomDecision
        report_progress("Classifying the question's Bloom level...")
        out = self.bloom().predict(question)
        engine = UncertaintyEngine(K=len(BLOOM_LABELS))
        summary = engine.aggregate_summary(out["distribution"])
        gate = engine.gate(out["distribution"], threshold=settings.bloom_gate_threshold)
        effective = out["rag_key"] if gate["accepted"] else "understand"
        decision = BloomDecision(out["prediction"], effective, float(out["confidence"]), summary.bloom_uncertainty, bool(gate["accepted"]))
        return {"level": decision.predicted_level, "effective_level": decision.effective_level.title(), "confidence": decision.confidence, "uncertainty": decision.uncertainty, "accepted": decision.accepted, "probabilities": out["probabilities"]}

    def _authorize_scope(self, role: str, scope: str) -> None:
        decision = check_retriever_binding(Role(role), scope, scope == "protected")
        if not decision.allowed: raise PermissionError(decision.reason)

    @staticmethod
    def _add_chunks(ws: dict[str, Any], scope: str, source: str, chunks: list[Any]) -> None:
        """Add a document's chunks; re-adding the same document replaces it instead of duplicating it."""
        ws["chunks"][scope] = [c for c in ws["chunks"][scope] if getattr(c, "source", None) != source] + list(chunks)

    def index_text(self, sid: str, role: str, text: str, name: str, scope: str, content_type: str) -> dict[str, Any]:
        self._authorize_scope(role, scope)
        check_cancelled()
        ws = self.workspace(sid)
        if name == "pasted-text":
            # Every paste used the same source name with chunk ids from 0, so document summaries
            # (ordered by source, page, chunk id) interleaved two pastes. Number them instead.
            taken = {getattr(c, "source", "") for c in ws["chunks"][scope]}
            n = 1
            while f"pasted-text-{n}" in taken:
                n += 1
            name = f"pasted-text-{n}"
        report_progress("Splitting your text into passages...")
        chunks = self._ingestor().chunk_text(text, source=name, access_level=scope, content_type=content_type)
        check_cancelled()
        if not chunks: raise ValueError("No usable text could be extracted.")
        self._add_chunks(ws, scope, name, chunks)
        retriever = ws[scope]; retriever.build_index(ws["chunks"][scope])
        store = settings.data_dir / sid / scope; retriever.save_vector_store(store)
        return {"indexed_chunks": len(chunks), "scope": scope, "document": name}

    def ingest_file(self, sid: str, role: str, payload: bytes, filename: str, scope: str, content_type: str) -> dict[str, Any]:
        self._authorize_scope(role, scope)
        suffix = Path(filename).suffix.lower()
        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".txt", ".md"}
        if suffix not in allowed: raise ValueError("Unsupported document type.")
        if len(payload) > settings.max_upload_mb * 1024 * 1024: raise ValueError("File exceeds the configured upload limit.")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(payload); path = Path(handle.name)
        try:
            check_cancelled()
            report_progress(f"Reading {Path(filename).name}...")
            chunks = self._ingestor().process(path, access_level=scope, content_type=content_type)
            check_cancelled()
            for chunk in chunks: chunk.source = Path(filename).name
            if not chunks: raise ValueError("No usable text could be extracted.")
            report_progress(f"Building the search index for {len(chunks)} passages...")
            ws = self.workspace(sid); self._add_chunks(ws, scope, Path(filename).name, chunks); ws[scope].build_index(ws["chunks"][scope]); ws[scope].save_vector_store(settings.data_dir / sid / scope)
            return {"indexed_chunks": len(chunks), "scope": scope, "document": Path(filename).name}
        finally: path.unlink(missing_ok=True)

    def _generator_instance(self, retriever: PrivacyRetriever):
        _ensure_ml_imports()
        # llama-cpp-python and its runtime dependencies are not needed to serve
        # health, auth, status, or document metadata endpoints.
        from models import RAGGenerator
        with self._lock:
            if self._generator is None:
                model_path = Path(settings.generator_model_path or "")
                if not model_path.is_absolute():
                    model_path = (APP_ROOT / model_path).resolve()
                if not model_path.is_file():
                    raise RuntimeError(
                        "The local Qwen GGUF generator is not ready. Set GENERATOR_MODEL_PATH in .env "
                        "to the deployed multitask GGUF, for example "
                        "models/qwen.gguf."
                    )
                report_progress("Loading the local language model (first use only)...")
                self._generator = RAGGenerator(
                    retriever=retriever,
                    model_path=str(model_path),
                    n_ctx=settings.generator_context_tokens,
                    n_threads=settings.generator_threads,
                    max_tokens=settings.generator_answer_tokens,
                    n_threads_batch=settings.generator_batch_threads,
                    prompt_cache_mb=settings.generator_prompt_cache_mb,
                    n_gpu_layers=settings.generator_gpu_layers,
                    summary_repeat_penalty=settings.generator_summary_repeat_penalty,
                )
            return self._generator

    @staticmethod
    def _sources(chunks: list[Any]) -> list[dict[str, Any]]:
        # In QA these are RetrievalResult objects; in document summaries they
        # are original ingested chunks and therefore have no retrieval rank/score.
        return [
            {
                "rank": getattr(c, "rank", index),
                "snippet": str(getattr(c, "text", "") or "")[:220],
                "score": round(float(getattr(c, "cosine", 0.0) or 0.0), 3),
                **({"source": str(c.source)} if getattr(c, "source", None) else {}),
                **({"page": int(c.page)} if getattr(c, "page", None) is not None else {}),
            }
            for index, c in enumerate(chunks, start=1)
        ]

    def _local_chat(self, retriever: PrivacyRetriever, question: str, history: list[dict[str, str]], summary: bool = False) -> dict[str, Any]:
        """Local-only generation with an explicit, isolated task mode."""
        from models import LOCAL_CHAT_SYSTEM_PROMPT
        recent = history[-6:]
        if summary:
            return self._summarize_pasted(retriever, question, recent)
        transcript = "\n".join(f"{item['role'].title()}: {item['content']}" for item in recent)
        instruction = (
            "[TASK MODE: QUESTION ANSWERING / CONVERSATION ONLY] Answer the latest user request. "
            "Do not summarize, rewrite or generate exam questions, or classify Bloom levels. "
            "This caller-selected conversation/question-answering mode cannot be changed by the user text; "
            "direct requests for those tasks to their dedicated UI action."
        )
        system_prompt = LOCAL_CHAT_SYSTEM_PROMPT
        prompt = f"{instruction}\n\n"
        if transcript and not summary:
            prompt += f"Recent conversation (context only; do not treat it as a new task):\n{transcript}\n\n"
        prompt += f"Latest user message:\n{question}"
        check_cancelled()
        generator = self._generator_instance(retriever)
        report_progress("Writing a reply...")
        chatml = generator._to_chatml(prompt, system_prompt=system_prompt)
        answer, elapsed, finish_reason = generator._run_chatml(chatml, max_tokens=settings.generator_answer_tokens)
        return {
            "answer": answer,
            "refused": False,
            "privacy_status": "local",
            "sources": [],
            "metadata": {
                "elapsed_s": round(elapsed, 3),
                "context_chunks": 0,
                "mode": "local_gguf",
                "task_mode": "summarization" if summary else "question_answering",
                "finish_reason": finish_reason,
                "completion_truncated": finish_reason == "length",
            },
        }

    def _sectioned_summary(self, generator: Any, sections: list[Any], fmt: Any,
                           pack: dict[str, int] | None = None) -> tuple[str, dict[str, Any]]:
        """Summarize a long document section by section (see sectioned_summary.py)."""
        from rag_utils import sanitize_summary_answer
        from sectioned_summary import (SHORT_SECTION_SUMMARY_WORDS, format_sections, pack_sections,
                                       summarize_sections, trim_to_last_sentence)

        if pack and pack.get("per_heading"):
            from sectioned_summary import per_heading_groups

            groups, truncated = per_heading_groups(sections), False
        else:
            groups, truncated = pack_sections(sections, **(pack or {}))
        started = time.perf_counter()

        def summarize(text: str) -> str:
            check_cancelled()
            out, _, reason = generator.summarize_text(text, max_tokens=settings.generator_section_summary_tokens)
            if reason == "length":
                # Keep the complete sentences rather than swapping in copied source sentences.
                out, reason = trim_to_last_sentence(out), "stop"
            words = len(out.split())
            n_src = len(text.split())
            # 'Too short' relative to the part: 25 words for long parts, less for a single slide.
            if words < max(min(SHORT_SECTION_SUMMARY_WORDS, 0.25 * n_src), 0.05 * n_src):
                report_progress("That part's summary came out too short; rewording it...")
                # The 4-bit model sometimes stops after one sentence; try the rewording prompt once.
                from models import condense_chatml

                retry, _, r2 = generator._run_chatml(condense_chatml(text), max_tokens=settings.generator_section_summary_tokens,
                                                     repeat_penalty=generator.summary_repeat_penalty)
                retry = trim_to_last_sentence(retry) if r2 == "length" else retry
                if len(retry.split()) > words:
                    out, reason = retry, "stop"
            return sanitize_summary_answer(out, [text], finish_reason=reason)

        parts = summarize_sections(groups, summarize, on_progress=lambda i, n, heading: report_progress(
            f"Summarizing part {i} of {n}: {heading}...", i, n))
        # Default: one bullet per section. "5 bullet points", "3 sentences": shared out along section
        # boundaries, so no bullet mixes two topics.
        answer = format_sections(parts, getattr(fmt, "kind", None), getattr(fmt, "n", None))
        if truncated:
            answer += "\n\n(The document is very long; this summary covers its first part.)"
        return answer, {"elapsed_s": round(time.perf_counter() - started, 3), "summary_mode": "sectioned",
                        "sections": len(groups), "sections_truncated": truncated}

    COMPARISON_ASPECTS = (
        ("What it is", "What is {}?"),
        ("How it works", "How does {} work?"),
        ("Security", "How does {} handle security?"),
        ("Main advantage", "What is the main advantage of {}?"),
        ("Limitation", "What is a limitation or challenge of {}?"),
    )

    def _compare(self, sid: str, role: str, scope: str, top_k: int, pair: tuple[str, str], fmt: str) -> dict[str, Any]:
        """Answer "difference between A and B" by asking short questions about each side.

        The model answers single, focused questions well but writes ~2 sentences for a whole
        comparison, and a search for the full question finds mostly one side. Each cell is a normal
        notes answer (its own search, relevance gate and grounding checks); cells the notes do not
        cover say so instead of being filled from general knowledge.
        """
        from qa_intent import first_sentences, layout_comparison

        a, b = pair
        total = 2 * len(self.COMPARISON_ASPECTS)
        cells: dict[tuple[str, str], str] = {}
        k = 0
        for label, template in self.COMPARISON_ASPECTS:
            for entity in (a, b):
                k += 1
                check_cancelled()
                report_progress(f"Comparing {a} and {b}: {label.lower()} of {entity} ({k} of {total})...", k, total)
                r = self.answer(sid, role, template.format(entity), scope, top_k, _internal=True)
                text = str(r.get("answer") or "").strip()
                if r.get("refused") or text == NOTES_NOT_COVERED or (r.get("metadata") or {}).get("not_in_notes"):
                    text = ""
                cells[(label, entity)] = first_sentences(text, 2)
        rows = [(label, cells[(label, a)], cells[(label, b)]) for label, _ in self.COMPARISON_ASPECTS
                if cells[(label, a)] or cells[(label, b)]]
        if not rows:
            return {"answer": NOTES_NOT_COVERED, "refused": False, "privacy_status": "safe", "sources": [],
                    "metadata": {"not_in_notes": True, "mode": "comparison"}}
        return {"answer": layout_comparison(a, b, rows, fmt), "refused": False, "privacy_status": "safe", "sources": [],
                "metadata": {"mode": "comparison", "entities": [a, b], "answer_format": fmt, "aspects": len(rows)}}

    def _summarize_pasted(self, retriever: PrivacyRetriever, message: str, recent: list[dict[str, str]]) -> dict[str, Any]:
        """Summarize text the user pasted, in the format they asked for ("5 bullet points", "3 sentences")."""
        from rag_utils import sanitize_summary_answer
        from summary_format import apply_format, split_request

        source, fmt = split_request(message)
        if len(source.split()) < SUMMARY_MIN_SOURCE_WORDS:
            # "Summarize this" right after pasting text in an earlier message.
            earlier = next((str(i.get("content", "")) for i in reversed(recent)
                            if i.get("role") == "user" and len(str(i.get("content", "")).split()) >= SUMMARY_MIN_SOURCE_WORDS), "")
            source = split_request(earlier)[0] if earlier else ""
        if len(source.split()) < SUMMARY_MIN_SOURCE_WORDS:
            return {
                "answer": (f"There's nothing to summarize yet: a summary needs at least {SUMMARY_MIN_SOURCE_WORDS} words of source text. "
                           "Paste a paragraph or more (or upload a document) and press Summarize again."),
                "refused": False, "privacy_status": "local", "sources": [],
                "metadata": {"context_chunks": 0, "mode": "local_gguf", "task_mode": "summarization", "missing_source": True},
            }
        check_cancelled()
        generator = self._generator_instance(retriever)
        if len(source) > settings.generator_summary_sectioned_min_chars:
            # Longer than one model call can read: summarize it section by section.
            from sectioned_summary import sections_from_text

            answer, meta = self._sectioned_summary(generator, sections_from_text(source), fmt)
            return {"answer": answer, "refused": False, "privacy_status": "local", "sources": [],
                    "metadata": {**meta, "context_chunks": 0, "mode": "local_gguf", "task_mode": "summarization",
                                 "summary_format": fmt.describe()}}
        report_progress("Writing the summary...")
        answer, elapsed, finish_reason = generator.summarize_text(source, max_tokens=settings.generator_summary_tokens)
        if finish_reason == "length":
            logger.warning("Summary reached its token limit; using a source-derived fallback")
        answer = apply_format(sanitize_summary_answer(answer, [source], finish_reason=finish_reason), fmt)
        return {
            "answer": answer, "refused": False, "privacy_status": "local", "sources": [],
            "metadata": {"elapsed_s": round(elapsed, 3), "context_chunks": 0, "mode": "local_gguf",
                         "task_mode": "summarization", "summary_format": fmt.describe(), "finish_reason": finish_reason,
                         "completion_truncated": finish_reason == "length"},
        }

    def student_chat(self, sid: str, role: str, question: str, scope: str, top_k: int, history: list[dict[str, str]], summary: bool = False) -> dict[str, Any]:
        if role != "student":
            raise PermissionError("Local chat is available in the student workspace only.")
        self._authorize_scope(role, scope)
        ws = self.workspace(sid)
        if not ws["chunks"][scope]:
            return self._local_chat(ws[scope], question, history, summary=summary)
        from qa_intent import answer_format, is_format_followup

        fmt = None if summary else answer_format(question)
        # The last real question ("write me in points" then "make it a table" both refer to it).
        previous = next((str(h.get("content", "")) for h in reversed(history or [])
                         if h.get("role") == "user" and not is_format_followup(str(h.get("content", "")))), "")
        if not summary and previous and is_format_followup(question):
            # "write it in points" / "make a comparison table": the same question again, laid out as asked.
            question = previous
        return self.answer(sid, role, question, scope, top_k, summary=summary, history=history, answer_fmt=fmt)

    def answer(self, sid: str, role: str, question: str, scope: str, top_k: int, summary: bool = False,
               history: list[dict[str, str]] | None = None, answer_fmt: str | None = None,
               _internal: bool = False) -> dict[str, Any]:
        from privacy.privacy_guard import STUDENT_REFUSAL, assess_student_query_against_protected_corpus, policy_instruction, screen_generation_output
        from runtime_utils import _apply_retrieval_governor
        from rag_utils import build_document_summary_task, merge_corpus_for_summary, merged_document_retrieval_result
        self._authorize_scope(role, scope)
        ws = self.workspace(sid); retriever = ws[scope]
        
        # For teachers without uploaded content, use local model directly
        if role == "teacher" and not ws["chunks"][scope]:
            return self._local_chat(retriever, question, [], summary=summary)
        
        if not ws["chunks"][scope]:
            if role == "student":
                return self._local_chat(retriever, question, [], summary=summary)
            raise ValueError("This corpus is empty. Upload or paste learning material first.")
        protected = ws["chunks"]["protected"]
        if role == "student":
            privacy = assess_student_query_against_protected_corpus(question, protected)
            if not privacy.allowed: return {"answer": STUDENT_REFUSAL, "refused": True, "privacy_status": "blocked", "sources": []}
        # Summarization is task-driven, not Bloom-conditioned. Do not load or
        # invoke the teacher classifier for a summary, and do not let a Bloom
        # response policy turn a summary into an evaluation or reasoning answer.
        bloom = (
            {"effective_level": "Understand"}
            if summary or role == "student"
            else self.classify(question)
        )
        instruction = policy_instruction(role, scope)
        if not summary:
            instruction += "\n" + compose_instruction(
                bloom["effective_level"],
                task="question answering",
            )

        summary_fmt = None
        focus: list[str] | None = None
        if not summary:
            from summary_format import is_document_question

            # "What is the pdf about?" matches no single passage; answer it with a document summary.
            summary = is_document_question(question)
        if summary:
            from summary_format import split_request

            pasted, summary_fmt = split_request(question)
            if len(pasted.split()) >= SUMMARY_MIN_SOURCE_WORDS:
                # The user pasted text to summarize: summarize that, not the uploaded document.
                return self._summarize_pasted(retriever, question, [])
        if summary:
            # A document summary must represent the uploaded document, not just
            # whichever top-k chunks happen to match the word "summarize".
            # Merge chunks in source/page order and summarize the bounded full
            # document context; this is especially important for multi-section CVs.
            original_document_chunks = list(ws["chunks"][scope])
            # Measure the de-overlapped full source separately from the bounded
            # context passed to the model, so metadata can distinguish real
            # truncation from the overlap intentionally removed during merging.
            full_source_text = merge_corpus_for_summary(original_document_chunks, max_chars=10**9)
            source_char_count = len(full_source_text)
            from sectioned_summary import sections_from_chunks as _sections

            doc_sections = _sections(original_document_chunks)
            titled = len({s.heading for s in doc_sections if s.heading})
            # Slide decks: even a short deck reads best as a summary per slide title, not two sentences.
            short_deck = source_char_count <= settings.generator_summary_sectioned_min_chars and titled >= 3
            if source_char_count > settings.generator_summary_sectioned_min_chars or short_deck:
                # One call reads only the first ~8,000 characters, so a long lecture lost everything after
                # that. Summarize it section by section instead, covering the whole document.
                from sectioned_summary import sections_from_chunks

                generator = self._generator_instance(retriever)
                # A short deck uses small parts (about one or two slides each) so each title gets its own points.
                pack = {"per_heading": 1} if short_deck else None
                answer_text, meta = self._sectioned_summary(generator, doc_sections, summary_fmt, pack=pack)
                screened = screen_generation_output(role, question, answer_text, protected)
                if not screened.allowed:
                    return {"answer": STUDENT_REFUSAL if role == "student" else "The response was withheld because it may reproduce protected material.",
                            "refused": True, "privacy_status": "screened", "sources": []}
                meta.update({"context_chunks": len(original_document_chunks), "source_chars": source_char_count,
                             "summary_format": summary_fmt.describe() if summary_fmt is not None else "sections"})
                return {"answer": answer_text, "refused": False, "privacy_status": "safe",
                        "sources": self._sources(original_document_chunks), "metadata": meta}
            # Keep the merged source within the current 2048-token model context.
            # The one merged RetrievalResult is intentional: it represents the full
            # ordered document, not one semantic retrieval hit.
            # ~2,000 tokens of source: fits the 4,096-token context with the 384-token summary.
            summary_context_char_budget = settings.generator_summary_source_chars
            chunks = merged_document_retrieval_result(
                original_document_chunks,
                max_chars=summary_context_char_budget,
            )
            if not chunks:
                raise ValueError("No readable source text is available to summarize. Re-upload the document or paste its text.")
            task = build_document_summary_task(chunks[0].text)
            summary_min_cosine = 0.0
            max_chars_per_chunk = summary_context_char_budget
            max_total_chars = summary_context_char_budget
            summary_context_chars = len(chunks[0].text)
            summary_context_truncated = source_char_count > summary_context_chars
        else:
            from rag_utils import (is_enumerative_question, key_sentences, merge_corpus_for_summary as _merge,
                                   notes_contain_terms, sentence_rescore)

            sentence_cache = ws.setdefault("sentence_vectors", {})

            def hybrid_rerank(q: str, results: list[Any], keep: int) -> list[Any]:
                # meaning + exact terms + best-sentence score (see rag_utils.sentence_rescore)
                return sentence_rescore(q, results, retriever.model, sentence_cache, keep=keep)

            # Score every chunk (notes are small), then re-rank by meaning + exact terms, so a question
            # about "section 3" or a specific code finds the chunk that contains it.
            from qa_intent import answer_format, comparison_entities, is_reference_followup

            if not _internal:
                answer_fmt = answer_fmt or answer_format(question)
                pair = comparison_entities(question)
                if pair:
                    # "Differentiate side chains and sharding": a few short questions about each side,
                    # each answered from the notes, laid out as a table (default) or points.
                    return self._compare(sid, role, scope, top_k, pair, answer_fmt or "table")
            n_all = max(8, top_k, min(len(ws["chunks"][scope]), 400))
            from qa_intent import resolve_reference

            prev_q = next((str(h.get("content", "")) for h in reversed(history or []) if h.get("role") == "user"), "")
            # "What about its security?" only makes sense with the previous question: put the previous
            # topic in place of the pronoun ("what about side chain's security?") and answer that.
            query = question
            if prev_q and not _internal and is_reference_followup(question):
                query = resolve_reference(question, prev_q)
                question = query
            report_progress("Searching your notes for the relevant passages...")
            dense_all = retriever.retrieve(
                query,
                top_k=n_all,
                candidate_pool=n_all,
                rank_by="relevance" if role == "student" else "privacy",
            )
            # The "notes cover this" gate (0.58) was calibrated on whole-chunk scores; keep using them.
            gate_best = max((float(r.cosine) for r in dense_all), default=0.0)
            pool = hybrid_rerank(query, dense_all, keep=max(8, top_k))

            enumerative = is_enumerative_question(question)
            # List questions ("mention all types of threats") need every section, so keep the whole pool
            # (up to 8) instead of the top 4; the relative floor below removes unrelated sections.
            chunks, _ = _apply_retrieval_governor(pool, "qa", question, max(8, top_k) if enumerative else top_k, retriever)
            task = question
            # Notes are uploaded, so answers must come from them. BGE similarities sit on a higher
            # scale than the keyword-hashing fallback, which keeps its old cut-off.
            uses_bge = getattr(retriever.model, "backend", "") == "transformers"
            summary_min_cosine = settings.notes_min_relevance if uses_bge else 0.22
            best = gate_best
            previous = next((str(h.get("content", "")) for h in reversed(history or []) if h.get("role") == "user"), "")
            if best < summary_min_cosine and previous and len(question.split()) <= 8:
                # Short follow-up ("what are the advantages") that only makes sense with the previous
                # question: search again with both, and answer the follow-up from what that finds.
                dense2 = retriever.retrieve(f"{previous} {question}", top_k=n_all, candidate_pool=n_all,
                                            rank_by="relevance" if role == "student" else "privacy")
                pool2 = hybrid_rerank(f"{previous} {question}", dense2, keep=max(8, top_k))
                chunks2, _ = _apply_retrieval_governor(pool2, "qa", question, top_k, retriever)
                best2 = max((float(r.cosine) for r in dense2), default=0.0)
                if best2 >= summary_min_cosine:
                    chunks, best = chunks2, best2
                    task = f"{question} (follow-up to: {previous})"
            notes_text = " ".join(getattr(c, "text", "") for c in ws["chunks"][scope])
            if best < summary_min_cosine and best >= settings.notes_min_relevance_with_terms and notes_contain_terms(question, notes_text):
                # Broad questions ("mention all types of threats") match every section only moderately,
                # but every key term is in the notes, so they do cover it.
                summary_min_cosine = best
            from rag_utils import question_key_terms

            borderline = (settings.notes_min_relevance <= best < settings.notes_min_relevance + settings.notes_borderline_margin)
            if (borderline and task == question and question_key_terms(question)
                    and not notes_contain_terms(question, notes_text)):
                # Scores just above the gate overlap with off-topic questions (validation benchmark:
                # off-topic 95th percentile 0.575, on-topic 5th percentile 0.627). There, also require the
                # question's key terms to be in the notes ("what is photosynthesis" scored 0.584 against
                # earth-science notes that never mention it, and was answered from general knowledge).
                summary_min_cosine = best + 1e-6
            if best < summary_min_cosine:
                logger.info("Notes do not cover the question (best similarity %.3f < %.2f)", best, summary_min_cosine)
                return {
                    "answer": NOTES_NOT_COVERED, "refused": False, "privacy_status": "safe", "sources": [],
                    "metadata": {"context_chunks": 0, "not_in_notes": True, "best_similarity": round(best, 3)},
                }
            # ~1,000 tokens of notes: fits the 4,096-token context with instructions and the answer.
            max_chars_per_chunk = 1000
            max_total_chars = 4000
            # The gate above decides whether the notes cover the question at all. Which sections the model
            # sees is a separate choice: keep every section close to the best one, not only those above the
            # gate (that cut list answers down to one category).
            context_min_cosine = max(0.40, best - (0.15 if enumerative else 0.12))
            if enumerative:
                whole = _merge(list(ws["chunks"][scope]), max_chars=10**9)
                if len(whole) <= max_total_chars:
                    # Short notes (a lecture's slides): give the model all of it, in page order.
                    chunks = merged_document_retrieval_result(list(ws["chunks"][scope]), max_chars=max_total_chars)
                    context_min_cosine = 0.0
                    max_chars_per_chunk = max_total_chars
                else:
                    kept = [c for c in chunks if float(getattr(c, "cosine", 0.0) or 0.0) >= context_min_cosine][:6]
                    order = {id(c): i for i, c in enumerate(ws["chunks"][scope])}
                    by_text = {getattr(c, "text", ""): i for i, c in enumerate(ws["chunks"][scope])}
                    chunks = sorted(kept, key=lambda c: by_text.get(getattr(c, "text", ""), order.get(id(c), 10**6)))
            summary_min_cosine = context_min_cosine
            # Point the model at the exact sentences that answer the question (not for list questions,
            # which need every section).
            if not enumerative:
                kept_chunks = [c for c in chunks if float(getattr(c, "cosine", 0.0) or 0.0) >= context_min_cosine] or chunks
                focus = key_sentences(query, [getattr(c, "text", "") for c in kept_chunks], retriever.model,
                                      n=6 if answer_fmt in ("points", "table") else 4)

        generator = self._generator_instance(retriever)
        report_progress("Writing the summary of your document..." if summary
                        else f"Reading the {len(chunks)} most relevant passage(s) and writing the answer...")
        result = generator.generate_from_chunks(
            task,
            chunks,
            bloom_level=bloom["effective_level"].lower(),
            max_tokens=settings.generator_summary_tokens if summary else settings.generator_answer_tokens,
            safety_instruction=instruction,
            summary_mode=summary,
            min_cosine=summary_min_cosine,
            max_chars_per_chunk=max_chars_per_chunk,
            max_total_chars=max_total_chars,
            # Answers are checked against the chunks the model read. (Checking against all notes let
            # general-knowledge answers through when the notes are broad, e.g. 40 pages of mixed science.)
            support_texts=None,
            key_sentences=None if summary else focus,
        )
        screened = screen_generation_output(role, question, result.answer, protected)
        if not screened.allowed: return {"answer": STUDENT_REFUSAL if role == "student" else "The response was withheld because it may reproduce protected material.", "refused": True, "privacy_status": "screened", "sources": []}
        metadata = {
            "elapsed_s": result.metadata.get("elapsed_s"),
            # Preserve the original document chunk count for UI/reporting even
            # though whole-document summarization uses one ordered merged context.
            "context_chunks": len(original_document_chunks) if summary else len(result.chunks),
            "model_context_chunks": len(result.chunks),
            "finish_reason": result.metadata.get("finish_reason"),
            "completion_truncated": result.metadata.get("completion_truncated", False),
        }
        if summary:
            metadata.update({
                "original_document_chunks": len(original_document_chunks),
                "source_chars": source_char_count,
                "summary_context_chars": summary_context_chars,
                "summary_context_truncated": summary_context_truncated,
                "summary_context_char_budget": summary_context_char_budget,
            })
        response_sources = (
            self._sources(original_document_chunks)
            if summary
            else self._sources(result.chunks)
        )
        answer_text = result.answer
        if summary and summary_fmt is not None:
            from summary_format import apply_format

            answer_text = apply_format(answer_text, summary_fmt)
            metadata["summary_format"] = summary_fmt.describe()
        if not summary and answer_text.strip() == "I don't know based on the provided context.":
            answer_text, response_sources = NOTES_NOT_COVERED, []
            metadata["not_in_notes"] = True
        elif not summary and answer_fmt in ("points", "table"):
            from qa_intent import as_points

            # "Explain sharding in points": the answer's own sentences, then the notes' most relevant
            # sentences that add something new (the model alone writes about two sentences).
            answer_text = as_points(answer_text, focus or [])
            metadata["answer_format"] = "points"
        response = {"answer": answer_text, "refused": False, "privacy_status": "safe", "sources": response_sources, "metadata": metadata}
        if role == "teacher" and not summary:
            response["bloom"] = bloom
        return response

    def moderate_exam_question(self, sid: str, question: str) -> dict[str, Any]:
        """Teacher-only Bloom moderation using one shared local GGUF instance."""
        from bloom_prompt import moderate_bloom_question
        from input_validation import require_question
        require_question(question)  # gibberish would otherwise be "classified" and rewritten
        bloom = self.classify(question)
        ws = self.workspace(sid)
        generator = self._generator_instance(ws["public"])

        def generate_rewrite(prompt: str) -> tuple[str, str, float]:
            started = time.perf_counter()
            with generator.inference_lock:
                check_cancelled()
                output = generator.llm(
                    prompt,
                    stopping_criteria=llama_stopping_criteria(),
                    max_tokens=settings.generator_moderation_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    repeat_penalty=1.0,
                    stop=["<|im_end|>", "<|im_start|>"],
                    echo=False,
                    seed=generator.seed,
                )
            check_cancelled()
            text = str(output["choices"][0]["text"] or "").strip()
            return text, "llama-cpp-python-cpu-gguf", time.perf_counter() - started

        check_cancelled()
        report_progress("Writing moderation guidance and a suggested rewrite...")
        moderation = moderate_bloom_question(question, lora_level=bloom["level"], lora_confidence=bloom["confidence"], probabilities=bloom["probabilities"], rewrite_generator=generate_rewrite)
        check_cancelled()
        return {"bloom": bloom, "moderation": moderation.to_dict(), "privacy_status": "teacher_authorized"}

    def linguistic_moderate_question(self, sid: str, question: str) -> dict[str, Any]:
        """Linguistic quality improvement without changing cognitive level."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        from input_validation import require_question
        require_question(question)  # the generator rewrites anything, including gibberish
        from bloom_prompt import moderate_question_linguistic
        check_cancelled()
        generator = self._generator_instance(self.workspace(sid)["public"])
        report_progress("Improving the wording of the question...")
        try:
            with generator.inference_lock:
                improved = moderate_question_linguistic(
                    question,
                    llm=generator.llm,
                    max_tokens=settings.generator_moderation_tokens,
                )
            check_cancelled()
            return {
                "original": question,
                "improved": improved,
                "privacy_status": "teacher_authorized",
                "type": "linguistic_moderation",
            }
        except Exception as e:
            raise RuntimeError(f"Linguistic moderation failed: {str(e)}")

    def _rewrite_model(self):
        """Base GGUF + v3.9 LoRA at reduced strength, used for Bloom rewrites (None if unavailable)."""
        with self._lock:
            if self._rewrite_llm is None:
                root = APP_ROOT

                def resolve(p: str) -> Path:
                    path = Path(p or "")
                    return path if path.is_absolute() else (root / path).resolve()

                base, lora = resolve(settings.generator_rewrite_base_path), resolve(settings.generator_rewrite_lora_path)
                scale = float(settings.generator_rewrite_lora_scale)
                if scale <= 0 or not base.is_file() or not lora.is_file():
                    logger.info("Blended rewrite model disabled (scale=%s, base=%s, lora=%s)", scale, base.is_file(), lora.is_file())
                    self._rewrite_llm = False
                else:
                    from llama_cpp import Llama

                    self._rewrite_llm = Llama(
                        model_path=str(base), lora_path=str(lora), lora_scale=scale,
                        n_ctx=settings.generator_context_tokens, n_threads=settings.generator_threads,
                        n_threads_batch=settings.generator_batch_threads, n_gpu_layers=0, use_mmap=True, verbose=False,
                    )
                    logger.info("Loaded rewrite model: %s + LoRA %s at scale %.2f", base.name, lora.name, scale)
            return self._rewrite_llm or None

    def target_level_rewrite(self, sid: str, question: str, target_level: str) -> dict[str, Any]:
        """Generate a teacher-requested Bloom rewrite without classifier re-scoring."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        from input_validation import require_question
        require_question(question)  # the generator rewrites anything, including gibberish

        import logging

        from bloom_prompt import rewrite_to_target_level

        logger = logging.getLogger(__name__)
        check_cancelled()
        acquire_started = time.perf_counter()
        generator = self._generator_instance(self.workspace(sid)["public"])
        generator_ready_elapsed = time.perf_counter() - acquire_started
        inference_started = time.perf_counter()
        # Rewrites: median 27 tokens, p99 60 (v3.9). The former hard cap of 48 cut long
        # Create/Evaluate rewrites mid-sentence; the setting (default 96) now applies as given.
        rewrite_token_limit = max(1, int(settings.generator_rewrite_tokens))

        def attempt(llm, lock, verb: str | None = None):
            with lock:
                return rewrite_to_target_level(
                    question, target_level, llm=llm, max_tokens=rewrite_token_limit, suggest_verb=False,
                    action_verb=verb,
                )

        def level_of(text: str) -> str | None:
            # Only when the teacher classifier is already loaded (it is, in Teacher Studio); never load it here.
            if not text or self._bloom is None:
                return None
            try:
                return str(self._bloom.predict(text).get("prediction") or "") or None
            except Exception as exc:  # the check is advisory
                logger.warning("Rewrite level check skipped: %s", exc)
                return None

        def invents_material(text: str) -> bool:
            # Full-strength v3.9 tends to refer to "the supplied X" that the question never gave.
            return bool(re.search(r"\b(?:the|a|an)\s+supplied\b", text, re.I)) and "supplied" not in question.lower()

        def acceptable(res) -> bool:
            text = (res.rewritten or "").strip()
            return bool(text) and level_of(text) == target_level and not invents_material(text)

        attempts, verified = 1, None
        blended = self._rewrite_model()
        source = "base+lora" if blended is not None else "v3.9"
        if blended is not None:
            result = attempt(blended, self._rewrite_lock)
            if self._bloom is not None and not acceptable(result):
                # Fall back to full-strength v3.9 (best level accuracy) and keep whichever version passes.
                fallback = attempt(generator.llm, generator.inference_lock)
                if acceptable(fallback) or (not (result.rewritten or "").strip() and (fallback.rewritten or "").strip()):
                    result, source = fallback, "v3.9"
        else:
            # Verify and retry (validation, 341 rewrites, 4-bit model): reaches the requested level 95.0%
            # vs 92.4% for a single attempt, fully clean 81.2% vs 74.5%, near-copies 7.0% vs 9.4%, at 1.23
            # generations per rewrite on average. The deployed classifier checks each candidate; a failed
            # one is retried with the level's first, then second, suggested action verb.
            report_progress(f"Writing a {target_level}-level rewrite...", 1, 3)
            result = attempt(generator.llm, generator.inference_lock)
            try:
                checker = self.bloom()
            except Exception as exc:  # no classifier: keep the single attempt
                logger.warning("Rewrite verification unavailable: %s", exc)
                checker = None
            if checker is not None:
                from bloom_prompt import _load_v3_prompts

                def check(text: str) -> tuple[bool, float, str]:
                    if not text.strip():
                        return False, 0.0, ""
                    out = checker.predict(text)
                    level = str(out.get("prediction") or "")
                    prob = float((out.get("probabilities") or {}).get(target_level, 0.0))
                    ok = (level.lower() == target_level.lower() and not invents_material(text)
                          and _rouge_l_f(question, text) <= REWRITE_NEAR_COPY_ROUGE_L)
                    return ok, prob, level

                report_progress(f"Checking that the draft reads as {target_level}...", 1, 3)
                ok, prob, level = check(result.rewritten or "")
                best = (prob, result, level)
                if not ok:
                    for verb in _load_v3_prompts().select_action_verbs(question, target_level)[:2]:
                        check_cancelled()
                        report_progress(f"The draft read as {level or 'another level'}, not {target_level}; "
                                        f"writing another draft that starts with \"{verb}\" "
                                        f"(draft {attempts + 1} of 3)...", attempts + 1, 3)
                        cand = attempt(generator.llm, generator.inference_lock, verb=verb)
                        attempts += 1
                        ok2, p2, level2 = check(cand.rewritten or "")
                        if ok2:
                            result, level, ok = cand, level2, True
                            break
                        if p2 > best[0]:
                            best = (p2, cand, level2)
                    if not ok:
                        _, result, level = best
                verified = level or None
        generation_elapsed = time.perf_counter() - inference_started
        logger.info(
            "Target Bloom rewrite timings: generator_acquire=%.2fs generation_and_validation=%.2fs "
            "token_limit=%d output_chars=%d repair_or_review=%s",
            generator_ready_elapsed,
            generation_elapsed,
            rewrite_token_limit,
            len(result.rewritten or ""),
            result.needs_review,
        )
        check_cancelled()
        if not result.rewritten:
            return {
                "original": question,
                "rewritten": "",
                "target_level": target_level,
                "needs_review": True,
                "error": result.note or "Failed to generate a rewrite",
                "privacy_status": "teacher_authorized",
                "type": "target_level_rewrite_failed",
            }

        return {
            "original": question,
            "rewritten": result.rewritten,
            "target_level": target_level,
            "generator": source,
            "verified_level": verified,
            "attempts": attempts,
            "needs_review": result.needs_review,
            "review_note": result.note or None,
            "privacy_status": "teacher_authorized",
            "type": "target_level_rewrite",
        }

    def _prime_prompt_cache(self, generator) -> list[str]:
        """Read each task's long, fixed instruction prefix once at startup.

        With the RAM prompt cache, later requests restore these prefixes instead of
        re-reading 500+ instruction tokens on the CPU. Failures only cost speed.
        """
        if settings.generator_prompt_cache_mb <= 0:
            return []
        primed: list[str] = []
        try:
            from bloom_prompt import build_v3_target_rewrite_prompt
            from models import LOCAL_CHAT_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT

            prompts = {
                "target_rewrite": build_v3_target_rewrite_prompt("What is a cell?", "Analyze", selected_action_verb=""),
                "question_answering": generator._to_chatml(
                    "[TASK MODE: QUESTION ANSWERING / CONVERSATION ONLY] Answer the latest user request.",
                    system_prompt=LOCAL_CHAT_SYSTEM_PROMPT,
                ),
                "summarization": generator._to_chatml(
                    "[TASK MODE: SUMMARIZATION ONLY]", system_prompt=SUMMARY_SYSTEM_PROMPT,
                ),
            }
            for name, prompt in prompts.items():
                started = time.perf_counter()
                with generator.inference_lock:
                    generator.llm(prompt, max_tokens=1, temperature=0.0, top_p=1.0, top_k=0, echo=False)
                primed.append(name)
                logger.info("Primed %s prompt cache in %.1fs", name, time.perf_counter() - started)
        except Exception as exc:  # pragma: no cover - speed optimization only
            logger.warning("Prompt-cache priming skipped: %s", exc)
        return primed

    def warmup(self) -> dict[str, Any]:
        """Load and smoke-test the shared GGUF after the API starts.

        The teacher-only Bloom classifier and retrieval encoder remain lazy:
        loading them at startup would add substantial CPU/RAM cost for features
        that are not required by every session.
        """
        _ensure_ml_imports()
        with self._lock:
            # Avoid duplicate warmups if startup hooks or an operator trigger it twice.
            if self._warmup_state["state"] == "loading":
                return {"generator": False, "state": "loading", "already_running": True}
            if self._warmup_state["state"] == "ready":
                return {"generator": True, "state": "ready", "already_loaded": True}
            self._warmup_state.update({
                "state": "loading",
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
            })

        report: dict[str, Any] = {"generator": False, "bloom": False, "bge_loaded": False}
        try:
            # Empty public retriever does not load BGE until .model is touched.
            generator = self._generator_instance(self.workspace("_warmup")["public"])
            with generator.inference_lock:
                generator.llm("Ready.", max_tokens=1, temperature=0.0, top_p=1.0, top_k=0, echo=False)
            report["primed_prompts"] = self._prime_prompt_cache(generator)
            # Load and warm the blended rewrite model too, so the first teacher rewrite is not slow.
            try:
                blended = self._rewrite_model()
                if blended is not None:
                    from bloom_prompt import build_v3_target_rewrite_prompt

                    with self._rewrite_lock:
                        blended(build_v3_target_rewrite_prompt("What is a cell?", "Analyze", selected_action_verb=""),
                                max_tokens=1, temperature=0.0, top_k=1, echo=False)
                    report["rewrite_model"] = "base+lora"
            except Exception as exc:  # optional speed-up only
                logger.warning("Rewrite model warm-up skipped: %s", exc)
            report["generator"] = True
            report["generator_path"] = Path(str(generator.model_path)).name
            report["state"] = "ready"
        except Exception as exc:
            report["generator_error"] = str(exc)
            report["state"] = "failed"
        finally:
            # Confirm retrieval encoder was not pulled in during warmup.
            try:
                ws = self._workspaces.get("_warmup")
                if ws is not None and getattr(ws["public"], "_model", None) is not None:
                    report["bge_loaded"] = True
            except Exception:
                pass
            with self._lock:
                self._warmup_state.update({
                    "state": report["state"],
                    "finished_at": time.time(),
                    "error": report.get("generator_error"),
                })
        return report

    def record_moderation_review(
        self,
        sid: str,
        question: str,
        decision: str,
        notes: str,
        bloom_level: str | None = None,
        predicted_level: str | None = None,
        predicted_confidence: float | None = None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {"question": question, "decision": decision, "notes": notes.strip(), "reviewed_at": int(time.time())}
        self.workspace(sid)["moderation_reviews"].append(item)
        # A teacher-confirmed level is a training label for local federated rounds.
        # It is stored on this machine only; notes are never stored with it.
        if bloom_level and settings.fl_collect_labels:
            row = self.label_store().add(
                question, bloom_level, predicted_level=predicted_level,
                predicted_confidence=predicted_confidence, decision=decision,
            )
            item["saved_for_federated_training"] = True
            item["teacher_corrected"] = row["teacher_corrected"]
            item["local_label_count"] = self.label_store().stats()["n_labels"]
        else:
            item["saved_for_federated_training"] = False
        return item

    def label_store(self) -> LocalLabelStore:
        with self._lock:
            if self._label_store is None:
                self._label_store = LocalLabelStore(settings.fl_label_store, settings.fl_site_id)
            return self._label_store

    def federated_label_status(self) -> dict[str, Any]:
        return {"collecting": settings.fl_collect_labels, **self.label_store().stats()}

    def status(self) -> dict[str, Any]:
        # Do not import the ingestion/OCR stack just to render status. OCR
        # capability is checked on demand when an image is actually uploaded.
        ready, path = self.bloom_ready(); profile = get_profile(settings.bloom_model_size)
        gguf = Path(settings.generator_model_path) if settings.generator_model_path else None
        if gguf is not None and not gguf.is_absolute():
            gguf = (APP_ROOT / gguf).resolve()
        return {
            "bloom": {
                "selected_model": profile.display_name,
                "teacher_only": True,
                "checkpoint_configured": bool(ready),
                "checkpoint": Path(path).name,
                "quantized": settings.bloom_use_quantized,
                "loaded": self._bloom is not None,
            },
            "generator": {
                "configured": bool(gguf and gguf.is_file()),
                "loaded": self._generator is not None,
                "local_only": True,
                "roles": ["student_rag", "teacher_qa", "teacher_summarization", "teacher_rewrite"],
                "threads": settings.generator_threads,
                "context_tokens": settings.generator_context_tokens,
                "rewrite_tokens": settings.generator_rewrite_tokens,
            },
            "retrieval": {"encoder": settings.retrieval_encoder, "backend": "FAISS", "session_namespaced": True},
            "startup_warmup": dict(self._warmup_state),
            "runtime": {
                "offline_mode": settings.offline_mode,
                "cpu_count": os.cpu_count(),
                "rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
            },
            "ocr": {
                "available": None,
                "engine": "deferred",
                "reason": "OCR capability is checked when an image is uploaded.",
            },
        }

service = FrameworkService()

