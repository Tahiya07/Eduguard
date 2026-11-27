from __future__ import annotations
import os
import psutil
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

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
    notes_min_relevance: float = 0.58
    generator_summary_tokens: int = 384
    generator_moderation_tokens: int = 96
    generator_rewrite_tokens: int = 96
    # Bloom rewrites: base Qwen2.5-1.5B + the v3.9 LoRA at reduced strength keeps the base model's
    # knowledge and natural wording; full-strength v3.9 is the fallback when the level check fails.
    # Empty paths or scale <= 0 disable this and use the main generator only.
    generator_rewrite_base_path: str = "models/gguf/qwen25_1_5b_instruct_base_q4_k_m.gguf"
    generator_rewrite_lora_path: str = "models/gguf/qwen15b_multitask_eduguard_v3_9_lora_f16.gguf"
    generator_rewrite_lora_scale: float = 0.5
    session_secret: str = "local1234"
    student_access_code: str = "student-local"
    teacher_access_code: str = "teacher-local"
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
