#!/usr/bin/env python
"""Multi-task LoRA SFT for Qwen2.5 (QA + summarization + Bloom rewrite).

Loss is computed ONLY on assistant tokens (prompt masked with -100).
Does not train GGUF. Does not modify production models.

If the resource gate fails:
    TRAINING NOT STARTED â€” INSUFFICIENT RESOURCES
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

from check_resources import assess, collect_resources  # noqa: E402
from loss_masking import tokenize_with_assistant_only_loss  # noqa: E402
from paths import MULTITASK_DATA_DIR, SEED  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override resource gate (only on a known-suitable machine).",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    cfg = load_json(cfg_path)
    seed = int(cfg.get("seed", SEED))
    set_seed(seed)

    resources = collect_resources()
    feasibility = assess(cfg["model_id"], resources)
    print(json.dumps(feasibility, indent=2))
    if not feasibility["feasible"] and not args.force:
        print(feasibility["verdict"])
        print("TRAINING NOT STARTED â€” INSUFFICIENT RESOURCES")
        out = Path(cfg["output_dir"])
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        (out / "resource_block.json").write_text(
            json.dumps(feasibility, indent=2), encoding="utf-8"
        )
        raise SystemExit(2)

    data_dir = Path(cfg.get("dataset_dir", MULTITASK_DATA_DIR))
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    results_dir = Path(cfg.get("results_dir", output_dir / "results"))
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    print("DATASET DIR:", data_dir.resolve())
    print("TRAIN FILE:", (data_dir / "train.jsonl").resolve())
    print("TRAIN ROWS:", len(train_rows))
    print("TRAIN MISSING SFT:", [i for i,r in enumerate(train_rows) if "sft_text" not in r][:20])
    print("TRAIN MISSING PROMPT:", [i for i,r in enumerate(train_rows) if "prompt_text" not in r][:20])
    print("VAL MISSING SFT:", [i for i,r in enumerate(val_rows) if "sft_text" not in r][:20])
    device = "cuda" if resources.get("cuda_available") else "cpu"
    print("=" * 72)
    print("MULTI-TASK LORA TRAINING")
    print("model:", cfg["model_id"])
    print("train:", len(train_rows), "validation:", len(val_rows))
    print("device:", device)
    print("=" * 72)

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
        print("TRAINING NOT STARTED â€” missing training dependencies:", exc)
        raise SystemExit(2) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class JsonlChatDataset(Dataset):
        def __init__(self, rows: list[dict], max_len: int) -> None:
            self.rows = rows
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            return tokenize_with_assistant_only_loss(
                tokenizer,
                row["sft_text"],
                row["prompt_text"],
                self.max_len,
            )

    # Prefer official chat template when available to locate assistant boundary.
    # We still use stored prompt_text / sft_text which already follow ChatML.
    train_ds = JsonlChatDataset(train_rows, int(cfg["max_seq_length"]))
    val_ds = JsonlChatDataset(val_rows, int(cfg["max_seq_length"]))

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
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
    model.print_trainable_parameters()

    import math

    effective_batch_size = (
        int(cfg["per_device_train_batch_size"])
        * int(cfg["gradient_accumulation_steps"])
    )
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))

    args_tr = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        seed=seed,
        bf16=False,
        fp16=bool(device == "cuda"),
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.get("early_stopping_patience", 2))
            )
        ],
    )
    train_result = trainer.train(resume_from_checkpoint=True if (output_dir / "checkpoint-0").exists() else None)
    trainer.save_model(str(output_dir / "best_adapter"))
    tokenizer.save_pretrained(str(output_dir / "best_adapter"))
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "resources": resources,
        "feasibility": feasibility,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "metrics": train_result.metrics if train_result else None,
        "checkpoint_selection": "validation_loss_only",
        "assistant_only_loss": True,
    }
    (results_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Training complete. Best adapter:", output_dir / "best_adapter")


if __name__ == "__main__":
    main()




