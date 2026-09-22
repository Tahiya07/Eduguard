from __future__ import annotations
import os, threading, time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import psutil

from bloom_model_profiles import get_profile
from bloom_policy import BloomDecision, compose_instruction
from bloom_prompt import moderate_question_linguistic, rewrite_to_target_level
from ingestion import DocumentChunk, DocumentIngestor, ocr_backend_status
from models import RAGGenerator
from predict_bloom import BLOOM_LABELS, QwenBloomPredictor, is_deploy_checkpoint
from privacy.privacy_guard import STUDENT_REFUSAL, assess_student_query_against_protected_corpus, policy_instruction, screen_generation_output
from retriever import PrivacyRetriever, RetrievalResult
from role_access import Role, check_retriever_binding
from runtime_utils import _apply_retrieval_governor
from uncertainty import UncertaintyEngine
from bloom_prompt import moderate_bloom_question
from .config import settings

class FrameworkService:
    """One application layer: role checks precede every corpus/model operation."""
    def __init__(self) -> None:
        self.ingestor = DocumentIngestor(chunk_size=220, chunk_overlap=32)
        self._workspaces: dict[str, dict[str, Any]] = {}
        self._bloom: QwenBloomPredictor | None = None
        self._generator: RAGGenerator | None = None
        self._lock = threading.RLock()

    def workspace(self, sid: str) -> dict[str, Any]:
        with self._lock:
            if sid not in self._workspaces:
                public = PrivacyRetriever(lambda_privacy=0.1, encoder_profile=settings.retrieval_encoder)
                protected = PrivacyRetriever(lambda_privacy=0.5, model=public.model, encoder_profile=settings.retrieval_encoder)
                self._workspaces[sid] = {"public": public, "protected": protected, "chunks": {"public": [], "protected": []}, "moderation_reviews": []}
            return self._workspaces[sid]

    def _bloom_path(self) -> Path:
        root = Path(__file__).resolve().parents[1]
        if settings.bloom_model_dir:
            path = Path(settings.bloom_model_dir)
            preferred = path if path.is_absolute() else (root / path).resolve()
        else:
            preferred = (root / "models" / "qwen_bloom_fedprox_r20").resolve()
            artifact = (
                root
                / "artifacts"
                / "federated"
                / "global"
                / "qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
            ).resolve()
            if self._checkpoint_has_weights(preferred):
                return preferred
            if self._checkpoint_has_weights(artifact):
                return artifact
            return preferred
        if settings.bloom_use_quantized:
            return preferred
        # Final deploy is FedProx r20 only. Do not silently fall back to the
        # centralized merge — that is a different checkpoint.
        if self._checkpoint_has_weights(preferred):
            return preferred
        packaged = (root / "models" / "qwen_bloom_fedprox_r20").resolve()
        artifact = (
            root
            / "artifacts"
            / "federated"
            / "global"
            / "qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
        ).resolve()
        for candidate in (packaged, artifact):
            if candidate != preferred and self._checkpoint_has_weights(candidate):
                return candidate
        return preferred

    @staticmethod
    def _checkpoint_has_weights(path: Path) -> bool:
        return (path / "config.json").is_file() and (
            (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
        )

    def bloom_ready(self) -> tuple[bool, str]:
        path = self._bloom_path()
        if settings.bloom_use_quantized:
            return is_deploy_checkpoint(path), str(path)
        return self._checkpoint_has_weights(path), str(path)

    def bloom(self) -> QwenBloomPredictor:
        with self._lock:
            if self._bloom is None:
                ok, path = self.bloom_ready()
                if not ok:
                    raise RuntimeError(
                        "Final FedProx r20 Bloom checkpoint is not ready. "
                        "Set BLOOM_MODEL_DIR to the merged FedProx folder containing "
                        "model.safetensors (artifacts/.../qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged "
                        "or models/qwen_bloom_fedprox_r20)."
                    )
                self._bloom = QwenBloomPredictor(model_dir=path, model_size=settings.bloom_model_size, quantized=settings.bloom_use_quantized, prefer_quantized=settings.bloom_use_quantized)
            return self._bloom

    def classify(self, question: str) -> dict[str, Any]:
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

    def index_text(self, sid: str, role: str, text: str, name: str, scope: str, content_type: str) -> dict[str, Any]:
        self._authorize_scope(role, scope)
        chunks = self.ingestor.chunk_text(text, source=name, access_level=scope, content_type=content_type)
        if not chunks: raise ValueError("No usable text could be extracted.")
        ws = self.workspace(sid); ws["chunks"][scope].extend(chunks)
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
            chunks = self.ingestor.process(path, access_level=scope, content_type=content_type)
            for chunk in chunks: chunk.source = Path(filename).name
            if not chunks: raise ValueError("No usable text could be extracted.")
            ws = self.workspace(sid); ws["chunks"][scope].extend(chunks); ws[scope].build_index(ws["chunks"][scope]); ws[scope].save_vector_store(settings.data_dir / sid / scope)
            return {"indexed_chunks": len(chunks), "scope": scope, "document": Path(filename).name}
        finally: path.unlink(missing_ok=True)

    def _generator_instance(self, retriever: PrivacyRetriever) -> RAGGenerator:
        with self._lock:
            if self._generator is None:
                model_path = Path(settings.generator_model_path or "")
                if not model_path.is_absolute():
                    model_path = (Path(__file__).resolve().parents[1] / model_path).resolve()
                if not model_path.is_file():
                    raise RuntimeError(
                        "The local Qwen GGUF generator is not ready. Set GENERATOR_MODEL_PATH in .env "
                        "to an existing GGUF file, for example models/qwen.gguf."
                    )
                self._generator = RAGGenerator(
                    retriever=retriever,
                    model_path=str(model_path),
                    n_ctx=settings.generator_context_tokens,
                    n_threads=settings.generator_threads,
                    max_tokens=settings.generator_answer_tokens,
                )
            return self._generator

    @staticmethod
    def _sources(chunks: list[RetrievalResult]) -> list[dict[str, Any]]:
        return [{"rank": c.rank, "snippet": c.text[:220], "score": round(c.cosine, 3)} for c in chunks]

    def _local_chat(self, retriever: PrivacyRetriever, question: str, history: list[dict[str, str]], summary: bool = False) -> dict[str, Any]:
        """Small, local-only GGUF conversation path used by students without a corpus."""
        recent = history[-6:]
        transcript = "\n".join(f"{item['role'].title()}: {item['content']}" for item in recent)
        instruction = "Summarize the user's text clearly and concisely." if summary else "Answer helpfully and accurately. If you are uncertain, say so."
        prompt = f"{instruction}\n\n"
        if transcript:
            prompt += f"Conversation:\n{transcript}\n\n"
        prompt += f"User: {question}"
        generator = self._generator_instance(retriever)
        chatml = generator._to_chatml(prompt)
        answer, elapsed = generator._run_chatml(chatml, max_tokens=settings.generator_summary_tokens if summary else settings.generator_answer_tokens)
        return {"answer": answer, "refused": False, "privacy_status": "local", "sources": [], "metadata": {"elapsed_s": round(elapsed, 3), "context_chunks": 0, "mode": "local_gguf"}}

    def student_chat(self, sid: str, role: str, question: str, scope: str, top_k: int, history: list[dict[str, str]], summary: bool = False) -> dict[str, Any]:
        if role != "student":
            raise PermissionError("Local chat is available in the student workspace only.")
        self._authorize_scope(role, scope)
        ws = self.workspace(sid)
        if not ws["chunks"][scope]:
            return self._local_chat(ws[scope], question, history, summary=summary)
        return self.answer(sid, role, question, scope, top_k, summary=summary)

    def answer(self, sid: str, role: str, question: str, scope: str, top_k: int, summary: bool = False) -> dict[str, Any]:
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
        bloom = self.classify(question)
        pool = retriever.retrieve(question, top_k=max(8, top_k), candidate_pool=max(8, top_k), rank_by="relevance" if role == "student" else "privacy")
        chunks, _ = _apply_retrieval_governor(pool, "summary" if summary else "qa", question, top_k, retriever)
        instruction = policy_instruction(role, scope) + "\n" + compose_instruction(bloom["effective_level"], task="summarization" if summary else "question answering")
        generator = self._generator_instance(retriever)
        task = "Summarize the supplied academic material." if summary else question
        result = generator.generate_from_chunks(
            task,
            chunks,
            bloom_level=bloom["effective_level"].lower(),
            max_tokens=settings.generator_summary_tokens if summary else settings.generator_answer_tokens,
            safety_instruction=instruction,
            summary_mode=summary,
            min_cosine=0.0 if summary else 0.22,
        )
        screened = screen_generation_output(role, question, result.answer, protected)
        if not screened.allowed: return {"answer": STUDENT_REFUSAL if role == "student" else "The response was withheld because it may reproduce protected material.", "refused": True, "privacy_status": "screened", "sources": []}
        return {"answer": result.answer, "refused": False, "privacy_status": "safe", "bloom": bloom, "sources": self._sources(result.chunks), "metadata": {"elapsed_s": result.metadata.get("elapsed_s"), "context_chunks": len(result.chunks)}}

    def moderate_exam_question(self, sid: str, question: str) -> dict[str, Any]:
        """Teacher-only Bloom moderation using one shared local GGUF instance."""
        bloom = self.classify(question)
        ws = self.workspace(sid)
        generator = self._generator_instance(ws["public"])

        def generate_rewrite(prompt: str) -> tuple[str, str, float]:
            reset = getattr(generator.llm, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
            started = time.perf_counter()
            output = generator.llm(
                prompt,
                max_tokens=settings.generator_moderation_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repeat_penalty=1.0,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
                seed=generator.seed,
            )
            text = str(output["choices"][0]["text"] or "").strip()
            return text, "llama-cpp-python-cpu-gguf", time.perf_counter() - started

        from bloom_prompt import moderate_bloom_question
        moderation = moderate_bloom_question(question, lora_level=bloom["level"], lora_confidence=bloom["confidence"], probabilities=bloom["probabilities"], rewrite_generator=generate_rewrite)
        return {"bloom": bloom, "moderation": moderation.to_dict(), "privacy_status": "teacher_authorized"}

    def linguistic_moderate_question(self, sid: str, question: str) -> dict[str, Any]:
        """Linguistic quality improvement without changing cognitive level."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        generator = self._generator_instance(self.workspace(sid)["public"])
        try:
            improved = moderate_question_linguistic(
                question,
                llm=generator.llm,
                max_tokens=settings.generator_moderation_tokens,
            )
            return {
                "original": question,
                "improved": improved,
                "privacy_status": "teacher_authorized",
                "type": "linguistic_moderation",
            }
        except Exception as e:
            raise RuntimeError(f"Linguistic moderation failed: {str(e)}")

    def target_level_rewrite(self, sid: str, question: str, target_level: str) -> dict[str, Any]:
        """Rewrite with the finalized v3 GGUF prompt, then verify with the deployed classifier.

        The generator sees only the question and the selected target level. The classifier
        scores the rewrite and does not choose a different question.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        generator = self._generator_instance(self.workspace(sid)["public"])
        predictor = None
        predictor_error = ""
        try:
            predictor = self.bloom()
        except Exception as exc:
            predictor_error = str(exc)

        result = rewrite_to_target_level(
            question,
            target_level,
            llm=generator.llm,
            predictor=predictor,
            predictor_error=predictor_error,
            max_tokens=settings.generator_rewrite_tokens,
        )
        if not result.rewritten:
            return {
                "original": question,
                "rewritten": "",
                "target_level": target_level,
                "predicted_level": "",
                "validation_match": False,
                "validation_confidence": 0.0,
                "needs_review": True,
                "error": result.note or "Failed to generate a rewrite",
                "privacy_status": "teacher_authorized",
                "type": "target_level_rewrite_failed",
            }

        response = {
            "original": question,
            "rewritten": result.rewritten,
            "target_level": target_level,
            "predicted_level": result.predicted_level or "unknown",
            "validation_match": result.validation_match,
            "validation_confidence": result.confidence,
            "needs_review": result.needs_review,
            "privacy_status": "teacher_authorized",
        }
        if result.validation_error:
            response["type"] = "target_level_rewrite_validation_failed"
            response["error"] = result.note
            response["review_reason"] = result.note
        elif not result.validation_match:
            response["type"] = "target_level_rewrite_mismatch"
            response["error"] = result.note
            response["review_reason"] = result.note
        else:
            response["type"] = "target_level_rewrite"
        return response

    def warmup(self) -> dict[str, Any]:
        """Warm only the two inference models: 1.5B GGUF + 0.5B Bloom.

        BGE stays lazy until a corpus is indexed or searched.
        """
        report: dict[str, Any] = {"generator": False, "bloom": False, "bge_loaded": False}
        try:
            # Empty public retriever does not load BGE until .model is touched.
            generator = self._generator_instance(self.workspace("_warmup")["public"])
            reset = getattr(generator.llm, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
            generator.llm("Ready.", max_tokens=1, temperature=0.0, top_p=1.0, top_k=0, echo=False)
            report["generator"] = True
            report["generator_path"] = Path(str(generator.model_path)).name
        except Exception as exc:
            report["generator_error"] = str(exc)
        try:
            predictor = self.bloom()
            predictor.predict("Define photosynthesis.")
            report["bloom"] = True
            report["bloom_checkpoint"] = Path(predictor.model_dir).name
        except Exception as exc:
            report["bloom_error"] = str(exc)
        # Confirm retrieval encoder was not pulled in during warmup.
        try:
            ws = self._workspaces.get("_warmup")
            if ws is not None and getattr(ws["public"], "_model", None) is not None:
                report["bge_loaded"] = True
        except Exception:
            pass
        return report

    def record_moderation_review(self, sid: str, question: str, decision: str, notes: str) -> dict[str, Any]:
        item = {"question": question, "decision": decision, "notes": notes.strip(), "reviewed_at": int(time.time())}
        self.workspace(sid)["moderation_reviews"].append(item)
        return item

    def status(self) -> dict[str, Any]:
        ready, path = self.bloom_ready(); profile = get_profile(settings.bloom_model_size)
        gguf = Path(settings.generator_model_path) if settings.generator_model_path else None
        if gguf is not None and not gguf.is_absolute():
            gguf = (Path(__file__).resolve().parents[1] / gguf).resolve()
        return {
            "bloom": {
                "selected_model": profile.display_name,
                "checkpoint_configured": bool(ready),
                "checkpoint": Path(path).name,
                "quantized": settings.bloom_use_quantized,
                "loaded": self._bloom is not None,
            },
            "generator": {
                "configured": bool(gguf and gguf.is_file()),
                "loaded": self._generator is not None,
                "local_only": True,
                "threads": settings.generator_threads,
                "context_tokens": settings.generator_context_tokens,
                "rewrite_tokens": settings.generator_rewrite_tokens,
            },
            "retrieval": {"encoder": settings.retrieval_encoder, "backend": "FAISS", "session_namespaced": True},
            "runtime": {
                "offline_mode": settings.offline_mode,
                "cpu_count": os.cpu_count(),
                "rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
            },
            "ocr": ocr_backend_status(),
        }

service = FrameworkService()
