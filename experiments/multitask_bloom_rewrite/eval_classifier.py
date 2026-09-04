"""Bloom classifier wiring for multitask evaluation (fixed existing classifier only)."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from bloom_model_profiles import DEFAULT_MODEL_SIZE, get_profile
from eval_metrics import canonical_bloom

ClassifyFn = Callable[[str], dict[str, Any]]

# Candidate dirs checked in order when --classifier-dir is omitted.
CLASSIFIER_CANDIDATES_0_5B = (
    "models/qwen_bloom_merged0.5B",
    "models/qwen_bloom_trained0.5B",
    "models/qwen_bloom_federated0.5B",
    "models/qwen_bloom_quantized0.5B",
)


def _is_usable_classifier_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").is_file():
        return True
    if (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
        or any(path.glob("*.safetensors"))
    ):
        return True
    if (path / "quantization.json").is_file() and (path / "config.json").is_file():
        return True
    return False


def resolve_classifier_dir(
    classifier_dir: str | None, *, repo_root: Path
) -> Path:
    if classifier_dir:
        path = Path(classifier_dir)
        if not path.is_absolute():
            path = repo_root / path
        if not _is_usable_classifier_dir(path):
            raise FileNotFoundError(
                f"Classifier dir not usable: {path}. "
                "Expected merged weights (config.json + safetensors) or LoRA adapter_config.json."
            )
        return path.resolve()

    for rel in CLASSIFIER_CANDIDATES_0_5B:
        cand = (repo_root / rel).resolve()
        if _is_usable_classifier_dir(cand):
            return cand

    # Fall back to profile resolution (may still fail at load time).
    profile = get_profile(DEFAULT_MODEL_SIZE)
    for rel in (profile.merged_dir, profile.lora_dir, profile.federated_lora_dir):
        cand = (repo_root / rel).resolve()
        if _is_usable_classifier_dir(cand):
            return cand

    searched = ", ".join(CLASSIFIER_CANDIDATES_0_5B)
    raise FileNotFoundError(
        "No trained Bloom classifier found. Looked for: "
        f"{searched}. Pass --classifier-dir explicitly."
    )


def load_classifier(
    classifier_dir: str | None,
    *,
    repo_root: Path,
    require_smoke: bool = True,
) -> tuple[ClassifyFn, dict[str, Any]]:
    """Load fixed Bloom classifier and optionally smoke-test it.

    Returns (predict_fn, meta). Raises if the classifier cannot load/smoke-test
    when require_smoke=True.
    """
    from predict_bloom import QwenBloomPredictor

    resolved = resolve_classifier_dir(classifier_dir, repo_root=repo_root)
    predictor = QwenBloomPredictor(model_dir=str(resolved))
    meta: dict[str, Any] = {
        "classifier_dir": str(resolved),
        "smoke_ok": False,
        "smoke_error": None,
    }

    def predict(text: str) -> dict[str, Any]:
        result = predictor.predict(text)
        level = result.get("prediction") or result.get("label") or ""
        return {
            "predicted_level": canonical_bloom(level) or level,
            "confidence": float(result.get("confidence") or 0.0),
        }

    if require_smoke:
        try:
            # Force weight load + one prediction before evaluation starts.
            predictor._ensure_loaded()
            smoke = predict(
                "Identify the key facts about board games and justify why rules are important."
            )
            if not smoke.get("predicted_level"):
                raise RuntimeError(f"Classifier smoke returned empty level: {smoke}")
            meta["smoke_ok"] = True
            meta["smoke_prediction"] = smoke
        except Exception as exc:  # noqa: BLE001
            meta["smoke_error"] = f"{type(exc).__name__}: {exc}"
            meta["smoke_traceback"] = traceback.format_exc()
            raise RuntimeError(
                "Bloom classifier failed smoke test. "
                f"dir={resolved} error={meta['smoke_error']}"
            ) from exc

    return predict, meta


class ClassifierCallStats:
    def __init__(self) -> None:
        self.n_ok = 0
        self.n_fail = 0
        self.first_error: str | None = None
        self.first_traceback: str | None = None

    def record_ok(self) -> None:
        self.n_ok += 1

    def record_fail(self, exc: BaseException) -> None:
        self.n_fail += 1
        if self.first_error is None:
            self.first_error = f"{type(exc).__name__}: {exc}"
            self.first_traceback = traceback.format_exc()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "first_error": self.first_error,
            "first_traceback": self.first_traceback,
        }
