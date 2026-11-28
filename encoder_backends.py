from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Exact, dependency-free copy of the sklearn HashingVectorizer setup below (identical vectors), so the
# packaged app no longer needs scikit-learn + SciPy (~100 MB) for this fallback.
from hashing_vectorizer import HashingVectorizer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger("encoder_backends")


def resolve_local_transformers_path(model_name: str) -> Optional[str]:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)

    repo_candidates = [model_name]
    if "/" not in model_name:
        repo_candidates.append(f"sentence-transformers/{model_name}")

    cache_root = (
        Path(os.environ.get("HF_HUB_CACHE", ""))
        if os.environ.get("HF_HUB_CACHE")
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    for repo_id in repo_candidates:
        model_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
        refs_main = model_cache / "refs" / "main"
        snapshots = model_cache / "snapshots"
        if refs_main.is_file():
            snapshot = snapshots / refs_main.read_text(encoding="utf-8").strip()
            if snapshot.is_dir():
                return str(snapshot)
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir(), reverse=True):
                if snapshot.is_dir() and (snapshot / "config.json").is_file():
                    return str(snapshot)
    return None


class StableTextEncoder:
    """Offline-first encoder with a transformers backend and hashing fallback."""

    def __init__(
        self,
        model_name: str = "models/bge-small",
        device: str = "cpu",
        local_files_only: bool = True,
        n_features: int = 384,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_length: int = 384,
        encode_batch_size: int = 12,
        hf_fallback: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.hf_fallback = hf_fallback
        self.model_ref: Optional[str] = None
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.n_features = int(n_features)
        self.query_prefix = str(query_prefix or "")
        self.passage_prefix = str(passage_prefix or "")
        self.max_length = int(max_length)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.backend = "hashing"
        self._tokenizer = None
        self._model = None
        # Same as sklearn HashingVectorizer(n_features, alternate_sign=False, norm=None,
        # ngram_range=(1, 2), lowercase=True, strip_accents="unicode").
        self._hashing = HashingVectorizer(n_features=self.n_features)
        self._init_backend()

    def _init_backend(self) -> None:
        """Try the configured model, then the Hugging Face fallback, then the hashing encoder.

        Order: local folder (e.g. models/bge-small) -> HF cache copy of ``hf_fallback``
        -> download ``hf_fallback`` from Hugging Face (only when not in HF offline mode)
        -> hashing. The fallback has the same embedding size, so stored indexes stay compatible.
        """
        offline = os.environ.get("HF_HUB_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}
        candidates = [self.model_name]
        if self.hf_fallback and self.hf_fallback != self.model_name:
            candidates.append(self.hf_fallback)
        errors: list[str] = []
        for name in candidates:
            local_ref = resolve_local_transformers_path(name)
            may_download = name == self.hf_fallback and local_ref is None and not offline
            if local_ref is None and not may_download and name != self.model_name:
                errors.append(f"{name}: not in the Hugging Face cache (offline mode)")
                continue
            ref = local_ref or name
            local_only = self.local_files_only and not may_download
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=local_only)
                self._model = AutoModel.from_pretrained(ref, local_files_only=local_only)
                if torch is not None:
                    self._model.to(self.device)
                    self._model.eval()
                self.backend = "transformers"
                self.model_ref = ref
                if name != self.model_name:
                    logger.info("StableTextEncoder: %r not found; using Hugging Face fallback %s (%s)",
                                self.model_name, name, ref)
                else:
                    logger.info("StableTextEncoder backend=transformers model=%s", ref)
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                self._tokenizer = None
                self._model = None
        self.backend = "hashing"
        logger.warning(
            "StableTextEncoder falling back to hashing backend (tried %s). To use BGE offline, download it once: "
            "python -c \"from huggingface_hub import snapshot_download; snapshot_download('%s')\"",
            "; ".join(errors), self.hf_fallback or self.model_name,
        )

    def _apply_prefix(
        self,
        texts: Sequence[str],
        *,
        mode: str,
    ) -> list[str]:
        if mode == "query" and self.query_prefix:
            return [f"{self.query_prefix}{t}" for t in texts]
        if mode == "passage" and self.passage_prefix:
            return [f"{self.passage_prefix}{t}" for t in texts]
        return list(texts)

    def _encode_transformers(
        self,
        texts: Sequence[str],
        batch_size: int,
        normalize_embeddings: bool,
        *,
        mode: str = "passage",
    ) -> np.ndarray:
        assert self._tokenizer is not None and self._model is not None and torch is not None
        prepared = self._apply_prefix(texts, mode=mode)
        rows = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            toks = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                out = self._model(**toks)
                last_hidden = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                # Mean pooling, although bge-small's own config specifies CLS pooling: on EduGuard's
                # 146-document validation benchmark mean pooling retrieved as well or better (Recall@1
                # 0.911 vs 0.897, MRR 0.939 vs 0.934, identical on/off-topic AUC 0.996), and the 0.58
                # notes-relevance gate is calibrated on these scores. Changing it requires re-calibration.
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                arr = pooled.cpu().numpy().astype(np.float32, copy=False)
                rows.append(arr)
        emb = np.vstack(rows) if rows else np.zeros((0, self.n_features), dtype=np.float32)
        if normalize_embeddings and emb.size:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-12, None)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def _encode_hashing(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool,
    ) -> np.ndarray:
        emb = self._hashing.transform(list(texts))  # dense float32
        if normalize_embeddings and emb.size:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-12, None)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        *,
        mode: str = "passage",
    ) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        if len(texts) == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)
        bs = batch_size if batch_size > 0 else self.encode_batch_size
        if self.backend == "transformers" and torch is not None:
            return self._encode_transformers(
                texts,
                bs,
                normalize_embeddings,
                mode=mode,
            )
        return self._encode_hashing(texts, normalize_embeddings)

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        return self.encode(
            texts,
            batch_size=self.encode_batch_size,
            normalize_embeddings=normalize_embeddings,
            mode="query",
        )

    def encode_passages(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        return self.encode(
            texts,
            batch_size=self.encode_batch_size,
            normalize_embeddings=normalize_embeddings,
            mode="passage",
        )
