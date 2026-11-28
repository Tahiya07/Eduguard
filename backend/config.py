from __future__ import annotations
import os
import sys
import psutil
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

def _app_root() -> Path:
    """Folder that holds models/, data/ and .env.

    Source checkout: the repository root. Packaged EXE: the launcher passes EDUGUARD_ROOT (the folder
    with EduGuard.exe); otherwise the folder of the running executable. Code files inside a frozen
    bundle live in a private folder, so __file__ must not be used to find models or user data there.
    """
    if os.environ.get("EDUGUARD_ROOT"):
        return Path(os.environ["EDUGUARD_ROOT"]).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = _app_root()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    bloom_model_size: str = "0.5b"
    bloom_model_dir: str | None = None
    bloom_use_quantized: bool = False
    generator_model_path: str = "models/qwen.gguf"
    retrieval_encoder: str = "bge-small"
    offline_mode: bool = True
    data_dir: Path = ROOT / "data" / "web_sessions"
    max_upload_mb: int = 20
    max_question_chars: int = 4000
    bloom_gate_threshold: float = 0.40
    # Keep the shared CPU generator context moderate: larger contexts reserve more KV-cache
    # memory even for short QA prompts. Summaries are bounded to fit this budget.
    generator_threads: int = max(4, min(8, (os.cpu_count() or 4)))
    # GPU offload for the GGUF generator: -1 = all layers if the llama.cpp build supports a GPU
    # (CUDA/Vulkan), otherwise it silently runs on the CPU. 0 forces CPU.
    generator_gpu_layers: int = -1
    # Token budgets sized from measured v3.9 outputs (median / p99 tokens): rewrite 27 / 60, QA 45 / 107,
    # summary 106 / 431. A cap only bounds the worst case (generation stops at end-of-turn), so these do
    # not slow typical requests; caps below p99 cut answers mid-sentence and force summary re-generation.
    # 4096: a long-document summary prompt (~1,250 source tokens + instructions) plus a 384-token summary.
    generator_context_tokens: int = 4096
    # CPU: prompt reading (batch) scales with physical cores; token-by-token decoding peaks near 8 threads.
    generator_batch_threads: int = max(4, psutil.cpu_count(logical=False) or (os.cpu_count() or 8) // 2)
    # Keeps each task's long instruction prefix in RAM so it is not re-read on every request (0 = off).
    generator_prompt_cache_mb: int = 768
    generator_answer_tokens: int = 160
    # With uploaded notes, answer only when the best note passage is at least this similar (BGE cosine).
    # Calibrated on CSE/EEE/DB/Economics notes: on-topic >= 0.60, off-topic <= 0.556.
    # With every key term of the question present in the notes, a lower similarity still counts as covered.
    notes_min_relevance_with_terms: float = 0.45
    # Best scores within this margin above notes_min_relevance must also have the question's key terms in the notes.
    notes_borderline_margin: float = 0.05
    notes_min_relevance: float = 0.58
    generator_summary_tokens: int = 384
    # Summaries only (rewrites and QA stay at 1.0). Validated on 146 summaries with the 4-bit model:
    # repetition loops 27 -> 7, verbatim source copying 38% -> 25%, ROUGE-L 0.263 -> 0.245 (bf16: 0.245).
    generator_summary_repeat_penalty: float = 1.1
    # Characters of an uploaded document given to the summarizer (~2,000 tokens; fits the 4,096 context).
    generator_summary_source_chars: int = 8000
    # Documents longer than this (~600 words) are summarized section by section (see sectioned_summary.py);
    # a single call yields ~100 words whatever the length, so longer documents need more sections.
    generator_summary_sectioned_min_chars: int = 3500
    # Per-section summary length when a long document is summarized section by section.
    generator_section_summary_tokens: int = 224
    generator_moderation_tokens: int = 96
    generator_rewrite_tokens: int = 96
    # Bloom rewrites use the full v3.9 generator (the main GGUF). An optional blend of the base model
    # with the v3.9 LoRA at reduced strength exists for experiments: set a scale > 0 and both paths.
    generator_rewrite_base_path: str = ""
    generator_rewrite_lora_path: str = ""
    generator_rewrite_lora_scale: float = 0.0
    session_secret: str = "local1234"
    # Sign-in codes for the two workspaces. Can be overridden with STUDENT_ACCESS_CODE /
    # TEACHER_ACCESS_CODE in the environment or the package's .env file.
    student_access_code: str = "student!@#$"
    teacher_access_code: str = "teacher!1@2"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Federated learning: teacher-confirmed Bloom labels are kept on this machine only and
    # used by training/federated/local_round.py; only adapter weights ever leave the site.
    fl_collect_labels: bool = True
    fl_label_store: Path = ROOT / "data" / "federated_local" / "confirmed_labels.jsonl"
    fl_site_id: str = "site_local"
    def origins(self) -> list[str]: return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1" if settings.offline_mode else "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1" if settings.offline_mode else "0")
