#!/usr/bin/env python
"""One-batch sanity check for the final 1.5B multitask LoRA SFT setup.

This is NOT a training run. It verifies that the exact training stack can:
1. load the final tokenizer/model,
2. inject the configured LoRA adapters,
3. tokenize one example from each task with assistant-only masking,
4. run a finite forward loss, and
5. optionally run backward and verify finite gradients.

Use a machine with a suitable GPU for --backward.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for search_path in (REPO_ROOT, EXPERIMENT_DIR, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from dataset_io import load_split_rows
from loss_masking import tokenize_with_assistant_only_loss


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--allow-cpu-backward", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    data_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    train_rows, _ = load_split_rows(data_dir, "train", validate=True)
    rows_by_task = {}
    for row in train_rows:
        rows_by_task.setdefault(row["task"], []).append(row)

    required_tasks = {"bloom_rewrite", "qa", "summarization"}
    missing = required_tasks - set(rows_by_task)
    if missing:
        raise SystemExit(f"Missing required training tasks: {sorted(missing)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backward and device == "cpu" and not args.allow_cpu_backward:
        raise SystemExit(
            "Refusing CPU backward for the 1.5B model. Use a suitable GPU or "
            "pass --allow-cpu-backward explicitly."
        )

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(cfg["max_seq_length"])
    by_task = {
        task: max(rows_by_task[task], key=lambda row: len(row.get("sft_text", "")))
        for task in required_tasks
    }
    batch = []
    for task in sorted(required_tasks):
        row = by_task[task]
        tokenized = tokenize_with_assistant_only_loss(
            tokenizer,
            row["sft_text"],
            row["prompt_text"],
            max_length,
        )
        labels = tokenized["labels"]
        supervised = sum(x != -100 for x in labels)
        if supervised <= 0:
            raise RuntimeError(f"{task}: no supervised assistant tokens")
        batch.append((task, tokenized, supervised))

    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if load_in_4bit:
        if device != "cuda":
            raise SystemExit("4-bit QLoRA is configured but CUDA is unavailable.")
        try:
            import bitsandbytes  # noqa: F401
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "This config requires 4-bit QLoRA. Install bitsandbytes in the active "
                "GPU environment with: python -m pip install bitsandbytes"
            ) from exc
        if cfg.get("bnb_4bit_compute_dtype", "float16") != "float16":
            raise SystemExit("This QLoRA config currently supports only float16 compute.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    if args.backward and device == "cuda" and use_gradient_checkpointing:
        # Disable KV caching during backward to reduce long-context memory.
        model.config.use_cache = False
    if args.backward and device == "cuda" and use_gradient_checkpointing and not load_in_4bit:
        model.gradient_checkpointing_enable()
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(args.backward and use_gradient_checkpointing),
        )
    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg["lora_target_modules"]),
    )
    model = get_peft_model(model, lora)
    if not load_in_4bit:
        model.to(device)
    model.train(args.backward)
    if args.backward and device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable <= 0:
        raise RuntimeError("No trainable LoRA parameters were created.")

    results = {
        "model_id": cfg["model_id"],
        "device": device,
        "max_seq_length": max_length,
        "tasks_checked": {},
        "trainable_parameters": trainable,
        "total_parameters": total,
        "backward_checked": bool(args.backward),
        "quantization": {
            "load_in_4bit": load_in_4bit,
            "quant_type": cfg.get("bnb_4bit_quant_type") if load_in_4bit else None,
            "compute_dtype": cfg.get("bnb_4bit_compute_dtype") if load_in_4bit else None,
        },
    }

    for task, tokenized, supervised in batch:
        if args.backward:
            model.zero_grad(set_to_none=True)
        inputs = {
            k: torch.tensor([v], dtype=torch.long, device=device)
            for k, v in tokenized.items()
            if k in {"input_ids", "attention_mask", "labels"}
        }
        with torch.set_grad_enabled(args.backward):
            out = model(**inputs)
            loss = out.loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{task}: non-finite loss: {loss.item()}")

        task_result = {
            "sequence_tokens": len(tokenized["input_ids"]),
            "supervised_tokens": supervised,
            "loss": float(loss.detach().cpu()),
        }

        if args.backward:
            loss.backward()
            finite_grads = 0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all().item():
                        raise RuntimeError(f"{task}: non-finite LoRA gradient")
                    finite_grads += 1
            if finite_grads == 0:
                raise RuntimeError(f"{task}: no LoRA gradients were produced")
            task_result["gradient_tensors"] = finite_grads

        results["tasks_checked"][task] = task_result
        # Do not retain the previous task's autograd graph while constructing
        # the next task's long-context forward pass.
        del out, loss, inputs

    if args.backward and device == "cuda":
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_bytes, _ = torch.cuda.mem_get_info(0)
        headroom_gb = free_bytes / 1024**3
        results["gpu_memory"] = {
            "peak_reserved_gb": round(peak_reserved_gb, 2),
            "total_gb": round(total_gb, 2),
            "free_after_backward_gb": round(headroom_gb, 2),
            "headroom_gb": round(headroom_gb, 2),
        }
        required_headroom_gb = float(cfg.get("preflight_min_free_vram_gb", 2.0))
        if headroom_gb < required_headroom_gb:
            raise SystemExit(
                f"GPU sanity backward left {headroom_gb:.2f} GB VRAM headroom; "
                f"{required_headroom_gb:.2f} GB is required."
            )

    print(json.dumps(results, indent=2))
    print("TRAINING SANITY CHECK: PASS")


if __name__ == "__main__":
    main()
