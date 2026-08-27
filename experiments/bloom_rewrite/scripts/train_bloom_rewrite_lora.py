#!/usr/bin/env python
"""LoRA SFT for Bloom target-level question rewriting.

Trains one model per invocation. Use the 0.5B and 1.5B configs independently.
Does not train GGUF. Does not modify the production rewrite pipeline.

If resources are insufficient, this script prints:
    TRAINING NOT STARTED — insufficient resources
and exits without training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_training_resources import assess, collect_resources  # noqa: E402
from paths import REWRITE_DATA_DIR, SEED  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "numpy"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def print_banner(cfg: dict, sizes: dict, device: str, n_params: int | None) -> None:
    print("=" * 72)
    print("BLOOM TARGET-REWRITE LORA TRAINING")
    print("=" * 72)
    print("Model:", cfg["model_id"])
    print("Number of parameters:", "unknown until load" if n_params is None else f"{n_params:,}")
    print("Dataset sizes:", sizes)
    print("Train/validation/test counts:", sizes)
    print("Sequence length:", cfg["max_seq_length"])
    print("LoRA rank:", cfg["lora_r"])
    print("LoRA alpha:", cfg["lora_alpha"])
    print("LoRA dropout:", cfg["lora_dropout"])
    print("Learning rate:", cfg["learning_rate"])
    print("Epochs:", cfg["epochs"])
    print("Batch size:", cfg["per_device_train_batch_size"])
    print("Gradient accumulation:", cfg["gradient_accumulation_steps"])
    print("Seed:", cfg["seed"])
    print("Device:", device)
    print("=" * 72)


def tokenize_example(tokenizer, text: str, prompt_text: str, max_len: int) -> dict:
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    labels = tokenized["input_ids"][:]
    prompt_len = min(len(prompt_ids), len(labels))
    # Mask the instruction tokens so loss is computed on the rewrite only.
    labels = [-100] * prompt_len + labels[prompt_len:]
    tokenized["labels"] = labels
    return tokenized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to LoRA JSON config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the resource safety check (use only on a known-suitable machine).",
    )
    args = parser.parse_args()
    cfg = load_json(Path(args.config))
    seed = int(cfg.get("seed", SEED))
    set_seed(seed)

    resources = collect_resources()
    feasibility = assess(cfg["model_id"], resources)
    device = "cuda" if resources.get("cuda_available") else "cpu"
    print(json.dumps(feasibility, indent=2))
    if not feasibility["feasible"] and not args.force:
        print(feasibility["verdict"])
        output_dir = Path(cfg["output_dir"])
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resource_block.json").write_text(
            json.dumps(feasibility, indent=2), encoding="utf-8"
        )
        raise SystemExit(2)

    data_dir = Path(cfg.get("dataset_dir", REWRITE_DATA_DIR))
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    sizes = {
        "train": len(train_rows),
        "validation": len(val_rows),
        "test": len(test_rows),
        "dataset": len(train_rows) + len(val_rows) + len(test_rows),
    }
    print_banner(cfg, sizes, device, None)
    if args.force and not feasibility["feasible"]:
        print("WARNING: --force used despite insufficient-resource estimate.")

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print("TRAINING NOT STARTED — missing training dependencies:", exc)
        print("Install peft, transformers, accelerate, and torch in the training environment.")
        raise SystemExit(2) from exc

    from prompt_format import build_generation_prompt, build_sft_text, describe_training_task

    print(json.dumps(describe_training_task(), indent=2))

    class RewriteDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            # source_bloom_level is metadata only — never in the generator prompt.
            prompt = build_generation_prompt(
                row["source_question"],
                row["target_bloom_level"],
            )
            full = build_sft_text(
                row["source_question"],
                row["target_bloom_level"],
                row["target_rewrite"],
            )
            if "Original Bloom level:" in full or "Source Bloom level:" in full:
                raise RuntimeError("source Bloom level leaked into training text")
            return tokenize_example(
                tokenizer,
                full + tokenizer.eos_token,
                prompt,
                int(cfg["max_seq_length"]),
            )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    n_params = count_parameters(model)
    print_banner(cfg, sizes, device, n_params)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg.get("lora_target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    run_dir = output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    adapter_dir = run_dir / "final_adapter"
    best_dir = run_dir / "best_adapter"
    run_dir.mkdir(parents=True, exist_ok=True)

    training_config = {
        **cfg,
        "dataset_dir": str(data_dir),
        "run_dir": str(run_dir),
        "seed": seed,
        "device": device,
        "n_params": n_params,
        "dataset_sizes": sizes,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resource_check": feasibility,
    }
    (run_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    args_hf = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", cfg["per_device_train_batch_size"])),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(cfg["epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(cfg.get("logging_steps", 20)),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
        seed=seed,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=RewriteDataset(train_rows),
        eval_dataset=RewriteDataset(val_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience", 2)))],
    )
    trainer.train()
    metrics = trainer.evaluate()
    (run_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print("Training complete →", run_dir)


if __name__ == "__main__":
    main()
