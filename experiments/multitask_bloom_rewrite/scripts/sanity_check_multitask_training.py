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
from pathlib import Path

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

    train_rows = read_jsonl(data_dir / "train.jsonl")
    by_task = {}
    for row in train_rows:
        by_task.setdefault(row["task"], row)

    required_tasks = {"bloom_rewrite", "qa", "summarization"}
    missing = required_tasks - set(by_task)
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

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        torch_dtype=dtype,
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
    model.to(device)
    model.eval()

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
    }

    for task, tokenized, supervised in batch:
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
            model.zero_grad(set_to_none=True)
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

    print(json.dumps(results, indent=2))
    print("TRAINING SANITY CHECK: PASS")


if __name__ == "__main__":
    main()
