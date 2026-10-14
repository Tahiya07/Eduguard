#!/usr/bin/env python
"""Multi-task LoRA SFT for Qwen2.5 (QA + summarization + Bloom rewrite).

Loss is computed ONLY on assistant tokens (prompt masked with -100).
Does not train GGUF. Does not modify production models.

If the resource gate fails:
    TRAINING NOT STARTED â€” INSUFFICIENT RESOURCES
"""
from __future__ import annotations

import argparse
import hashlib
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
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                raise ValueError(f"Blank JSONL line in {path} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prepared_dataset(data_dir: Path, repo_root: Path) -> dict:
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing dataset manifest: {manifest_path}. Regenerate the dataset before training."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read dataset manifest {manifest_path}: {exc}") from exc

    generated_hashes = manifest.get("generated_files_sha256")
    if not isinstance(generated_hashes, dict):
        raise SystemExit(
            "Dataset manifest has no generated_files_sha256 fingerprints. "
            "Rerun prepare_multitask_dataset_final.py before training."
        )

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing prepared split {path}. Regenerate the dataset.")
        expected_hash = generated_hashes.get(name)
        if not expected_hash or sha256_file(path) != expected_hash:
            raise SystemExit(
                f"Prepared split fingerprint mismatch: {path}. "
                "The corpus may be stale or edited; rerun preparation and QC."
            )

    bloom_source = manifest.get("bloom_source", {})
    source_dir_rel = bloom_source.get("path")
    source_hashes = bloom_source.get("files_sha256", {})
    if not source_dir_rel or not source_hashes:
        raise SystemExit("Dataset manifest lacks canonical BloomShift source fingerprints.")
    source_dir = (repo_root / source_dir_rel).resolve()
    for name in ("train.json", "validation.json", "test.json"):
        source_path = source_dir / name
        expected_hash = source_hashes.get(name)
        if not source_path.is_file():
            raise SystemExit(
                f"Canonical BloomShift source is missing: {source_path}. "
                "Cannot verify that the prepared dataset matches the current source."
            )
        if not expected_hash or sha256_file(source_path) != expected_hash:
            raise SystemExit(
                f"Canonical BloomShift source changed since preparation: {source_path}. "
                "Regenerate the multitask dataset before training."
            )

    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "bloom_source": str(source_dir),
        "generated_files_sha256": generated_hashes,
        "bloom_source_sha256": source_hashes,
    }


def set_seed(seed: int) -> None
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
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Explicit Hugging Face checkpoint directory to resume. Default is a fresh run.",
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

    dataset_fingerprint = validate_prepared_dataset(data_dir, REPO_ROOT)

    # Do not start optimization unless the same strict QC used for manual
    # validation passes against the exact training config and all three splits.
    qc_script = SCRIPT_DIR / "validate_multitask_dataset.py"
    qc_command = [
        sys.executable,
        str(qc_script),
        "--data-dir", str(data_dir.resolve()),
        "--max-length", str(int(cfg["max_seq_length"])),
        "--model-id", str(cfg["model_id"]),
    ]
    print("RUNNING DATASET QC:", " ".join(qc_command))
    try:
        subprocess.run(qc_command, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Dataset QC failed (exit={exc.returncode}); training has not started. "
            "Regenerate the final corpus and resolve every QC error."
        ) from exc

    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    if not train_rows or not val_rows:
        raise SystemExit("Prepared training and validation splits must both be non-empty.")
    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        for row_number, row in enumerate(rows, 1):
            for key in ("task", "sft_text", "prompt_text"):
                if not row.get(key):
                    raise SystemExit(
                        f"{split_name}.jsonl row {row_number} has a missing/empty {key!r}."
                    )
    print("DATASET FINGERPRINT CHECK: PASS")
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
        evaluation_strategy="epoch",
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
    # Never resume implicitly: a refreshed dataset must not silently continue
    # optimizer/scheduler state from an older run. Resumption is explicit.
    from transformers.trainer_utils import get_last_checkpoint
    existing_checkpoint = get_last_checkpoint(str(output_dir))
    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint).resolve()
        if not resume_path.is_dir() or not (resume_path / "trainer_state.json").is_file():
            raise SystemExit(
                f"Not a valid Hugging Face Trainer checkpoint: {resume_path}"
            )
        if existing_checkpoint and resume_path != Path(existing_checkpoint).resolve():
            print(
                "WARNING: explicit resume checkpoint differs from the latest checkpoint "
                "in the configured output directory."
            )
    else:
        if existing_checkpoint:
            raise SystemExit(
                f"Existing checkpoint found at {existing_checkpoint}. "
                "Use a new output_dir for a clean run, or pass "
                "--resume-from-checkpoint with the exact checkpoint to resume intentionally."
            )
        resume_path = None

    train_result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
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
        "dataset_fingerprint": dataset_fingerprint,
        "resume_from_checkpoint": str(resume_path) if resume_path else None,
    }
    (results_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Training complete. Best adapter:", output_dir / "best_adapter")


if __name__ == "__main__":
    main()




