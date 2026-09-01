"""HF generator loading and checkpoint validation for multitask evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class CheckpointInfo:
    condition: str
    base_model_id: str
    adapter_path: Path | None
    resolved_path: str
    checkpoint_selection: str


def _find_best_checkpoint_dir(output_dir: Path) -> Path | None:
    best = output_dir / "best_adapter"
    if (best / "adapter_config.json").is_file():
        return best
    candidates: list[tuple[float, Path]] = []
    for ckpt in sorted(output_dir.glob("checkpoint-*")):
        state = ckpt / "trainer_state.json"
        if not state.is_file():
            continue
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            best_metric = data.get("best_metric")
            if best_metric is not None:
                candidates.append((float(best_metric), ckpt))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    numbered = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.name)
    return numbered[-1] if numbered else None


def _load_tokenizer(base_model_id: str) -> Any:
    """Load tokenizer from the base HF model (LoRA does not change vocabulary)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)


def _lora_config_param_names() -> set[str]:
    from peft import LoraConfig
    import inspect

    return set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}


def _sanitize_lora_adapter_config(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop adapter_config keys that the installed PEFT cannot parse."""
    lora_params = _lora_config_param_names()
    meta_keys = {
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "revision",
        "peft_version",
        "transformers_version",
        "auto_mapping",
    }
    allowed = lora_params | meta_keys
    stripped = sorted(k for k in raw if k not in allowed)
    sanitized = {k: v for k, v in raw.items() if k in allowed}
    return sanitized, stripped


def _load_peft_adapter(base_model: Any, adapter_path: Path) -> Any:
    """Load LoRA weights, tolerating adapter_config fields from newer PEFT."""
    import shutil
    import tempfile

    from peft import PeftModel

    config_path = adapter_path / "adapter_config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    sanitized, stripped = _sanitize_lora_adapter_config(raw)
    if stripped:
        with tempfile.TemporaryDirectory(prefix="peft_adapter_") as tmp:
            tmp_path = Path(tmp)
            for item in adapter_path.iterdir():
                dest = tmp_path / item.name
                if item.name == "adapter_config.json":
                    dest.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
                elif item.is_file():
                    shutil.copy2(item, dest)
            return PeftModel.from_pretrained(base_model, str(tmp_path))
    return PeftModel.from_pretrained(base_model, str(adapter_path))


def resolve_checkpoint(
    cfg: dict[str, Any], condition: str, *, adapter_override: str | None = None
) -> CheckpointInfo:
    condition = condition.lower().strip()
    if condition not in {"base", "lora"}:
        raise ValueError(f"condition must be base or lora, got {condition!r}")
    base_id = cfg["model_id"]
    if condition == "base":
        return CheckpointInfo(
            condition="base",
            base_model_id=base_id,
            adapter_path=None,
            resolved_path=base_id,
            checkpoint_selection="n/a_base_model",
        )
    output_dir = Path(cfg["output_dir"])
    if adapter_override:
        adapter = Path(adapter_override)
    else:
        adapter = _find_best_checkpoint_dir(output_dir)
    if adapter is None or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter not found under {output_dir}. "
            "Expected best_adapter/adapter_config.json or checkpoint-*/"
        )
    return CheckpointInfo(
        condition="lora",
        base_model_id=base_id,
        adapter_path=adapter,
        resolved_path=str(adapter),
        checkpoint_selection="validation_loss_only",
    )


def validate_checkpoint(info: CheckpointInfo, max_seq_length: int, gen_cfg: dict) -> dict[str, Any]:
    """Load model/tokenizer and run one smoke generation. Raises on failure."""
    from transformers import AutoModelForCausalLM

    from prompts import bloom_messages, render_chatml

    load_start = time.perf_counter()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if info.condition == "base":
        tokenizer = _load_tokenizer(info.base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            info.base_model_id, trust_remote_code=True, torch_dtype=dtype
        )
    else:
        assert info.adapter_path is not None
        adapter_cfg = json.loads(
            (info.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        adapter_base = adapter_cfg.get("base_model_name_or_path", info.base_model_id)
        if adapter_base != info.base_model_id:
            raise ValueError(
                f"Adapter base {adapter_base!r} != config model_id {info.base_model_id!r}"
            )
        # Always use the base-model tokenizer: adapter copies can fail across
        # tokenizers versions (ModelWrapper parse errors on tokenizer.json).
        tokenizer = _load_tokenizer(adapter_base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            adapter_base, trust_remote_code=True, torch_dtype=dtype
        )
        model = _load_peft_adapter(base, info.adapter_path)
    model.eval()
    model.to(device)
    load_s = time.perf_counter() - load_start

    smoke_prompt = render_chatml(
        bloom_messages("Define osmosis.", "Understand"),
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        smoke_prompt, return_tensors="pt", truncation=True, max_length=max_seq_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    smoke_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not smoke_text:
        raise RuntimeError("Smoke generation returned empty output")
    return {
        "load_time_s": round(load_s, 4),
        "smoke_output_chars": len(smoke_text),
        "device": str(device),
        "dtype": str(dtype),
    }


class HFGenerator:
    def __init__(self, info: CheckpointInfo, max_seq_length: int) -> None:
        from transformers import AutoModelForCausalLM

        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if info.condition == "base":
            self.tokenizer = _load_tokenizer(info.base_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                info.base_model_id, trust_remote_code=True, torch_dtype=dtype
            )
        else:
            assert info.adapter_path is not None
            adapter_cfg = json.loads(
                (info.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
            base_id = adapter_cfg.get("base_model_name_or_path", info.base_model_id)
            self.tokenizer = _load_tokenizer(base_id)
            base = AutoModelForCausalLM.from_pretrained(
                base_id, trust_remote_code=True, torch_dtype=dtype
            )
            self.model = _load_peft_adapter(base, info.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.model.to(self.device)

    def generate(self, prompt: str, gen_cfg: dict[str, Any]) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 128)),
            "do_sample": bool(gen_cfg.get("do_sample", False)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.0))
            gen_kwargs["top_p"] = float(gen_cfg.get("top_p", 1.0))
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_resources() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    try:
        import psutil

        proc = psutil.Process()
        mi = proc.memory_info()
        info["rss_mb"] = round(mi.rss / 1024**2, 2)
        try:
            full = proc.memory_full_info()
            if hasattr(full, "uss"):
                info["uss_mb"] = round(full.uss / 1024**2, 2)
        except Exception:
            info["uss_mb"] = None
        info["cpu_percent"] = proc.cpu_percent(interval=0.1)
    except ImportError:
        info["psutil"] = False
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / 1024**2, 2
        )
        info["gpu_memory_reserved_mb"] = round(
            torch.cuda.memory_reserved(0) / 1024**2, 2
        )
    return info
