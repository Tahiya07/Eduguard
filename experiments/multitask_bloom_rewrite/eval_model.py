"""HF generator loading and checkpoint validation for multitask evaluation."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


# end of a complete sentence: terminal punctuation (plus closing quotes/brackets) followed by whitespace or the end
_SENT_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the complete sentences in `text`; a trailing fragment is not included."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    return spans


def trim_to_sentence(text: str) -> str:
    """Drop a trailing sentence fragment (keeps the text unchanged if it has no complete sentence)."""
    spans = sentence_spans(text)
    return text[: spans[-1][1]] if spans else text


def _opening(sentence: str, k: int) -> tuple[str, ...] | None:
    w = re.findall(r"\w+", sentence.lower())
    return tuple(w[:k]) if len(w) >= k else None


class LoopGuard:
    """Stopping criterion for degenerate greedy-decoding loops in the generated text (the prompt is not checked).

    * span: the last `ngram` generated tokens already occurred earlier in the output (verbatim repeat);
      the output is cut back to before the repeat, then to the last complete sentence.
    * template: the last `sentence_run` complete sentences all start with the same `prefix_words` words
      ("X is a way to make a product sanitary. X is a way to make a product hygienic. ..."); only the first
      sentence of the run is kept.
    """

    def __init__(self, tokenizer: Any, prompt_len: int, ngram: int, sentence_run: int = 0, prefix_words: int = 6) -> None:
        self.tokenizer, self.prompt_len = tokenizer, prompt_len
        self.ngram, self.sentence_run, self.prefix_words = ngram, sentence_run, prefix_words
        self.seen: set[tuple[int, ...]] = set()
        self.reason: str | None = None
        self.repeat_start: int | None = None
        self.keep_text: str | None = None

    def _template_cut(self, gen: list[int]) -> str | None:
        if not re.search(r"[.!?]", self.tokenizer.decode(gen[-1:], skip_special_tokens=True)):
            return None  # sentences can only have closed on a punctuation token
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        spans = sentence_spans(text)
        if len(spans) < self.sentence_run:
            return None
        run = spans[-self.sentence_run:]
        first = _opening(text[run[0][0]:run[0][1]], self.prefix_words)
        if first and all(_opening(text[a:b], self.prefix_words) == first for a, b in run[1:]):
            return text[: run[0][1]]
        return None

    def __call__(self, input_ids: torch.Tensor, scores: Any = None, **kwargs: Any) -> torch.Tensor:
        gen = input_ids[0, self.prompt_len:].tolist()
        stop = False
        if self.ngram > 0 and len(gen) >= self.ngram:
            tail = tuple(gen[-self.ngram:])
            if tail in self.seen:
                self.reason, self.repeat_start, stop = "span", len(gen) - self.ngram, True
            else:
                self.seen.add(tail)
        if not stop and self.sentence_run > 0 and gen:
            keep = self._template_cut(gen)
            if keep is not None:
                self.reason, self.keep_text, stop = "template", keep, True
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


@dataclass
class CheckpointInfo:
    condition: str
    base_model_id: str
    adapter_path: Path | None
    resolved_path: str
    checkpoint_selection: str


def eval_dtype() -> torch.dtype:
    """Match training precision: bf16 on GPUs that support it (Qwen2.5 can overflow in fp16)."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _enable_efficient_gqa_attention() -> None:
    """Same SDPA kernel fix as training (mathematically identical, linear memory)."""
    try:
        from attention_patches import force_sdpa_repeat_kv
    except ImportError:
        return
    force_sdpa_repeat_kv()


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
    if condition not in {"base", "lora", "gguf"}:
        raise ValueError(f"condition must be base, lora or gguf, got {condition!r}")
    base_id = cfg["model_id"]
    if condition == "gguf":
        gguf = Path(adapter_override or "")
        if not adapter_override or not gguf.is_file():
            raise FileNotFoundError(f"GGUF file not found: {gguf} (pass --gguf PATH)")
        return CheckpointInfo(
            condition="gguf",
            base_model_id=base_id,
            adapter_path=None,
            resolved_path=str(gguf.resolve()),
            checkpoint_selection="gguf_export_of_best_adapter",
        )
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
    dtype = eval_dtype()
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
    def __init__(self, info: CheckpointInfo, max_seq_length: int, load_in_4bit: bool = False) -> None:
        from transformers import AutoModelForCausalLM

        _enable_efficient_gqa_attention()

        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = eval_dtype()
        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
        if load_in_4bit:
            # same NF4 quantization as QLoRA training; base and LoRA are loaded identically
            if self.device.type != "cuda":
                raise SystemExit("eval_load_in_4bit needs a CUDA GPU.")
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
            load_kwargs["device_map"] = {"": 0}
        self.load_in_4bit = load_in_4bit
        if info.condition == "base":
            self.tokenizer = _load_tokenizer(info.base_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(info.base_model_id, **load_kwargs)
        else:
            assert info.adapter_path is not None
            adapter_cfg = json.loads(
                (info.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
            base_id = adapter_cfg.get("base_model_name_or_path", info.base_model_id)
            self.tokenizer = _load_tokenizer(base_id)
            base = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
            self.model = _load_peft_adapter(base, info.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        if not load_in_4bit:
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
        guard = None
        loop_n = int(gen_cfg.get("loop_guard_ngram") or 0)
        sentence_run = int(gen_cfg.get("loop_guard_sentence_run") or 0)
        if loop_n > 0 or sentence_run > 0:
            from transformers import StoppingCriteriaList

            guard = LoopGuard(self.tokenizer, inputs["input_ids"].shape[1], loop_n, sentence_run,
                              int(gen_cfg.get("loop_guard_prefix_words") or 6))
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([guard])
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        self.last_loop_reason = guard.reason if guard else None
        self.last_loop_stopped = self.last_loop_reason is not None
        # Output length of the last call; it hit the cap if it used every allowed token without emitting EOS.
        eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        self.last_hit_token_limit = bool(
            not self.last_loop_stopped
            and len(new_tokens) >= gen_kwargs["max_new_tokens"]
            and int(new_tokens[-1]) not in eos_ids
        )
        if self.last_loop_reason == "span":
            text = trim_to_sentence(self.tokenizer.decode(new_tokens[: guard.repeat_start], skip_special_tokens=True))
        elif self.last_loop_reason == "template":
            text = guard.keep_text
        else:
            self.last_output_tokens = int(len(new_tokens))
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.last_output_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return text


class GGUFGenerator:
    """llama.cpp backend with the same decoding contract as HFGenerator.

    Parity with the HF evaluation is deliberate so the only difference measured is quantization:
    * the prompt is tokenized with the same HF tokenizer and those exact token ids are fed to llama.cpp;
    * greedy decoding, the same per-task max_new_tokens, stop at EOS (<|im_end|> / <|endoftext|>);
    * the same LoopGuard object decides span/template stops, and the same trimming is applied.
    """

    def __init__(self, gguf_path: str | Path, base_model_id: str, max_seq_length: int,
                 max_new_tokens_cap: int, n_threads: int | None = None, n_gpu_layers: int = 0,
                 lora_path: str | None = None, lora_scale: float = 1.0) -> None:
        import os

        from llama_cpp import Llama

        self.max_seq_length = max_seq_length
        self.device = torch.device("cpu") if n_gpu_layers == 0 else torch.device("cuda")
        self.tokenizer = _load_tokenizer(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            import psutil

            physical = psutil.cpu_count(logical=False) or (os.cpu_count() or 8)
        except ImportError:
            physical = os.cpu_count() or 8
        decode_threads = int(n_threads or min(8, physical))
        # The prompt may use the full max_seq_length and still needs room for the longest task cap.
        self.n_ctx = int(max_seq_length + max_new_tokens_cap + 16)
        self.llm = Llama(
            model_path=str(gguf_path), n_ctx=self.n_ctx, n_threads=decode_threads,
            n_threads_batch=max(decode_threads, physical), n_gpu_layers=n_gpu_layers,
            use_mmap=True, verbose=False,
            # A LoRA GGUF applied at runtime at the given strength (the app's blended rewrite model).
            **({"lora_path": str(lora_path), "lora_scale": float(lora_scale)} if lora_path else {}),
        )
        stop_ids = {self.llm.token_eos(), self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(tid)
        self.stop_ids = {int(t) for t in stop_ids if t is not None}
        self.n_threads, self.n_threads_batch = decode_threads, max(decode_threads, physical)

    def tokenizer_parity(self, prompt: str) -> bool:
        """True when llama.cpp's own tokenizer agrees with the HF ids for this prompt."""
        hf_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return list(self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)) == list(hf_ids)

    def generate(self, prompt: str, gen_cfg: dict[str, Any]) -> str:
        ids = self.tokenizer(prompt, truncation=True, max_length=self.max_seq_length,
                             add_special_tokens=False)["input_ids"]
        max_new = int(gen_cfg.get("max_new_tokens", 128))
        guard = None
        loop_n = int(gen_cfg.get("loop_guard_ngram") or 0)
        sentence_run = int(gen_cfg.get("loop_guard_sentence_run") or 0)
        if loop_n > 0 or sentence_run > 0:
            guard = LoopGuard(self.tokenizer, len(ids), loop_n, sentence_run,
                              int(gen_cfg.get("loop_guard_prefix_words") or 6))
        gen: list[int] = []
        hit_eos = False
        for tok in self.llm.generate(ids, temp=0.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0):
            if int(tok) in self.stop_ids:
                hit_eos = True
                break
            gen.append(int(tok))
            if guard is not None and bool(guard(torch.tensor([ids + gen]))[0]):
                break
            if len(gen) >= max_new:
                break
        self.last_loop_reason = guard.reason if guard else None
        self.last_loop_stopped = self.last_loop_reason is not None
        self.last_hit_token_limit = bool(not self.last_loop_stopped and not hit_eos and len(gen) >= max_new)
        if self.last_loop_reason == "span":
            text = trim_to_sentence(self.tokenizer.decode(gen[: guard.repeat_start], skip_special_tokens=True))
        elif self.last_loop_reason == "template":
            text = guard.keep_text
        else:
            self.last_output_tokens = len(gen)
            return self.tokenizer.decode(gen, skip_special_tokens=True)
        self.last_output_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return text


def gguf_file_info(path: str | Path) -> dict[str, Any]:
    """Name, quantization type and size of a GGUF file, read from its header (no weights loaded)."""
    info: dict[str, Any] = {"path": str(path), "size_bytes": Path(path).stat().st_size}
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))

        def field(key: str) -> Any:
            f = reader.fields.get(key)
            if f is None:
                return None
            part = f.parts[f.data[0]]
            return bytes(part).decode("utf-8", "replace") if part.dtype.kind == "u" and part.dtype.itemsize == 1 \
                else part.tolist()[0]

        file_types = {0: "F32", 1: "F16", 7: "Q8_0", 2: "Q4_0", 15: "Q4_K_M", 14: "Q4_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}
        ftype = field("general.file_type")
        info.update({
            "name": field("general.name"),
            "architecture": field("general.architecture"),
            "file_type": file_types.get(ftype, ftype),
            "imatrix_dataset": field("quantize.imatrix.dataset"),
        })
    except Exception as exc:  # metadata is informative only
        info["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return info


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
