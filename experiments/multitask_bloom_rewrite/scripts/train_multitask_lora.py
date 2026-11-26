#!/usr/bin/env python
"""Multi-task LoRA SFT for Qwen2.5 (QA + summarization + Bloom rewrite).

Loss is computed ONLY on assistant tokens (prompt masked with -100).
Does not train GGUF. Does not modify production models.

If the resource gate fails:
    TRAINING NOT STARTED - INSUFFICIENT RESOURCES
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from dataset_io import (  # noqa: E402
    TASKS, load_manifest, load_split_rows, validate_separate_dataset,
)
from loss_masking import slice_to_assistant_logits, tokenize_with_assistant_only_loss  # noqa: E402
from paths import SEED, TASK_SEPARATED_DATA_DIR  # noqa: E402


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
    """Validate task-separated files and their original BloomShift fingerprints."""
    try:
        return validate_separate_dataset(data_dir, repo_root)
    except Exception as exc:
        raise SystemExit(f"Dataset fingerprint/isolation check failed: {exc}") from exc


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
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "numpy"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def run_gpu_memory_preflight(model, tokenizer, train_rows, max_length, min_free_gb, torch, assistant_logits_only=False):
    """Backpropagate through the longest configured training example before Trainer starts.

    Uses the same forward as training (assistant-only logits when enabled), so it
    measures real training memory instead of a full-vocabulary 4+ GB logits spike.
    """
    if not torch.cuda.is_available():
        raise SystemExit("GPU memory preflight requires a CUDA device.")
    longest_row = None
    longest_tokens = -1
    for row in train_rows:
        token_count = len(tokenizer(
            row["sft_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"])
        if token_count > longest_tokens:
            longest_row, longest_tokens = row, token_count
    if longest_row is None:
        raise SystemExit("GPU memory preflight found no training examples.")
    encoded = tokenize_with_assistant_only_loss(
        tokenizer, longest_row["sft_text"], longest_row["prompt_text"], max_length
    )
    inputs = {
        key: torch.tensor([encoded[key]], dtype=torch.long, device="cuda")
        for key in ("input_ids", "attention_mask", "labels")
        if key in encoded
    }
    if assistant_logits_only:
        inputs = slice_to_assistant_logits(inputs)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    safety_reserve = None
    try:
        reserve_bytes = int(min_free_gb * 1024**3)
        safety_reserve = torch.empty(
            reserve_bytes, dtype=torch.uint8, device="cuda"
        )
        loss = model(**inputs).loss
        if not torch.isfinite(loss).item():
            raise SystemExit("Maximum-length GPU preflight produced a non-finite loss.")
        loss.backward()
        del loss
        torch.cuda.synchronize()
        result = {
            "example_id": longest_row.get("id"),
            "sequence_tokens": longest_tokens,
            "explicit_safety_reserve_gb": min_free_gb,
            "reserve_held_during_backward": safety_reserve is not None,
            "device": str(safety_reserve.device),
            "status": "PASS",
            "required_headroom_gb": min_free_gb,
            "assistant_logits_only": bool(assistant_logits_only),
            "peak_allocated_gb_incl_reserve": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "peak_reserved_gb_incl_reserve": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        }
        print("MAX-LENGTH GPU MEMORY PREFLIGHT:", json.dumps(result, sort_keys=True))
        return result
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            f"GPU memory preflight OOMed on the longest training example "
            f"({longest_tokens} tokens, id={longest_row.get('id')}) while holding "
            f"the {min_free_gb:.2f} GB safety reserve. "
            "Training has not started; use a GPU with more VRAM or a validated shorter-context recipe."
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        del inputs
        if safety_reserve is not None:
            del safety_reserve
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override dataset_dir from the config (supports task-separated datasets).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the config; use a fresh directory for a new run.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results_dir from the config.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model_id from the config, e.g. a different Hugging Face causal LM.",
    )
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
    # CLI overrides make it possible to reuse a validated training recipe with
    # a different task-separated dataset, base model, or fresh artifact paths.
    for argument, config_key in (
        (args.dataset_dir, "dataset_dir"),
        (args.output_dir, "output_dir"),
        (args.results_dir, "results_dir"),
        (args.model_id, "model_id"),
    ):
        if argument:
            cfg[config_key] = argument
    seed = int(cfg.get("seed", SEED))
    set_seed(seed)

    resources = collect_resources()
    feasibility = assess(
        cfg["model_id"],
        resources,
        max_seq_length=int(cfg.get("max_seq_length", 8192)),
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
    )
    print(json.dumps(feasibility, indent=2))
    if not feasibility["feasible"] and not args.force:
        print(feasibility["verdict"])
        print("TRAINING NOT STARTED - INSUFFICIENT RESOURCES")
        out = Path(cfg["output_dir"])
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        (out / "resource_block.json").write_text(
            json.dumps(feasibility, indent=2), encoding="utf-8"
        )
        raise SystemExit(2)

    data_dir = Path(cfg.get("dataset_dir", TASK_SEPARATED_DATA_DIR))
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

    # Load each configured task from its own JSONL files. All selected examples
    # are retained; rows are combined in memory only because Hugging Face Trainer
    # accepts one Dataset object. No merged split JSONL is written.
    train_tasks = list(cfg.get("train_tasks", list(TASKS)))
    if not train_tasks or set(train_tasks) - set(TASKS) or len(set(train_tasks)) != len(train_tasks):
        raise SystemExit(f"Invalid train_tasks={train_tasks!r}; expected a unique, non-empty subset of {TASKS}.")
    train_rows, train_load_meta = load_split_rows(
        data_dir, "train", tasks=train_tasks, validate=False
    )
    val_rows, val_load_meta = load_split_rows(
        data_dir, "validation", tasks=train_tasks, validate=False
    )
    if not train_rows or not val_rows:
        raise SystemExit(
            f"Active training tasks {train_tasks} produced an empty split: "
            f"train={len(train_rows)}, validation={len(val_rows)}."
        )

    # Each task is read from its own verified JSONL files. The Trainer needs a
    # single in-memory Dataset, so the full task rows are concatenated in memory
    # only; there is no ratio-based downsampling or merged JSONL artifact.

    # Validate/fingerprint the complete canonical splits above, then optionally
    # cap validation rows per task in memory. This keeps full source data and
    # test data untouched while avoiding thousands of expensive eval examples
    # for tasks that dominate the original validation split.
    validation_task_limits = cfg.get("validation_task_limits", {})
    if validation_task_limits:
        grouped_val: dict[str, list[dict]] = {}
        for row in val_rows:
            grouped_val.setdefault(str(row.get("task", "unknown")), []).append(row)
        limited_val: list[dict] = []
        before_counts = {task: len(rows) for task, rows in grouped_val.items()}
        for task in sorted(grouped_val):
            task_rows = grouped_val[task]
            limit = validation_task_limits.get(task)
            if limit is None or int(limit) <= 0 or len(task_rows) <= int(limit):
                selected = task_rows
            else:
                # Stable, reproducible sample; no model-performance-based selection.
                selected = random.Random(seed + sum(ord(ch) for ch in task)).sample(
                    task_rows, int(limit)
                )
            limited_val.extend(selected)
        val_rows = sorted(
            limited_val,
            key=lambda row: (str(row.get("task", "")), str(row.get("id", ""))),
        )
        after_counts: dict[str, int] = {}
        for row in val_rows:
            task = str(row.get("task", "unknown"))
            after_counts[task] = after_counts.get(task, 0) + 1
        print("VALIDATION TASK COUNTS BEFORE LIMIT:", json.dumps(before_counts, sort_keys=True))
        print("VALIDATION TASK COUNTS AFTER LIMIT:", json.dumps(after_counts, sort_keys=True))
        print("VALIDATION TASK LIMITS:", json.dumps(validation_task_limits, sort_keys=True))

    training_selection = {
        "layout": "task-separated",
        "combined_in_memory_only": True,
        "dataset_manifest_sha256": dataset_fingerprint.get("manifest_sha256"),
        "active_tasks": train_tasks,
        "train_counts": {task: sum(row.get("task") == task for row in train_rows) for task in TASKS},
        "validation_counts": {task: sum(row.get("task") == task for row in val_rows) for task in TASKS},
        "train_ids_by_task": {
            task: [str(row.get("id", "")) for row in train_rows if row.get("task") == task]
            for task in TASKS if task in train_tasks
        },
        "validation_ids_by_task": {
            task: [str(row.get("id", "")) for row in val_rows if row.get("task") == task]
            for task in TASKS if task in train_tasks
        },
        "validation_sampling": "stable per-task ID-sorted seed sample after loading separate validation files",
    }
    configured_task_weights = cfg.get("task_sampling_weights")
    active_counts = {
        task: sum(row.get("task") == task for row in train_rows)
        for task in train_tasks
    }
    if configured_task_weights is not None:
        if not isinstance(configured_task_weights, dict):
            raise SystemExit("task_sampling_weights must map active task names to positive weights.")
        if set(configured_task_weights) != set(train_tasks):
            raise SystemExit(
                "task_sampling_weights must specify exactly the active train_tasks: "
                f"active={sorted(train_tasks)}, weights={sorted(configured_task_weights)}"
            )
        try:
            raw_weights = {task: float(configured_task_weights[task]) for task in train_tasks}
        except (TypeError, ValueError) as exc:
            raise SystemExit("task_sampling_weights values must be positive numbers.") from exc
        if any(not math.isfinite(weight) or weight <= 0 for weight in raw_weights.values()):
            raise SystemExit("task_sampling_weights values must be finite and greater than zero.")
        weight_total = sum(raw_weights.values())
        task_probabilities = {
            task: raw_weights[task] / weight_total for task in train_tasks
        }
        training_selection["task_sampling"] = {
            "mode": "weighted_with_replacement",
            "probabilities_per_epoch": task_probabilities,
            "expected_examples_per_epoch": {
                task: round(len(train_rows) * task_probabilities[task], 2)
                for task in train_tasks
            },
            "seed": seed,
        }
    else:
        task_probabilities = None
        training_selection["task_sampling"] = {
            "mode": "uniform_over_rows",
            "probabilities_per_epoch": {
                task: count / len(train_rows)
                for task, count in active_counts.items()
            },
            "expected_examples_per_epoch": {
                task: count for task, count in active_counts.items()
            },
            "seed": seed,
        }
    selection_path = results_dir / "training_selection_manifest.json"
    selection_path.write_text(json.dumps(training_selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        for row_number, row in enumerate(rows, 1):
            for key in ("task", "sft_text", "prompt_text"):
                if not row.get(key):
                    raise SystemExit(
                        f"{split_name}.jsonl row {row_number} has a missing/empty {key!r}."
                    )
    print("DATASET FINGERPRINT CHECK: PASS")
    print("DATASET DIR:", data_dir.resolve())
    print("DATA LAYOUT: one directory per task; no merged split JSONL persisted")
    print("ACTIVE TASKS:", json.dumps(train_tasks))
    print("TRAIN FILES:", json.dumps([str((data_dir / task / "train.jsonl").resolve()) for task in train_tasks]))
    print("TRAIN ROWS:", len(train_rows))
    print("TRAIN MISSING SFT:", [i for i,r in enumerate(train_rows) if "sft_text" not in r][:20])
    print("TRAIN MISSING PROMPT:", [i for i,r in enumerate(train_rows) if "prompt_text" not in r][:20])
    print("VAL MISSING SFT:", [i for i,r in enumerate(val_rows) if "sft_text" not in r][:20])
    device = "cuda" if resources.get("cuda_available") else "cpu"
    print("=" * 72)
    print("TASK-SELECTED LORA TRAINING")
    print("model:", cfg["model_id"])
    print("train:", len(train_rows), "validation:", len(val_rows), "tasks:", train_tasks)
    print("device:", device)
    print("=" * 72)

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import Dataset, WeightedRandomSampler
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print("TRAINING NOT STARTED - missing training dependencies:", exc)
        raise SystemExit(2) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class JsonlChatDataset(Dataset):
        """Tokenized once up front (same masking as before, no per-step re-tokenizing)."""

        def __init__(self, rows: list[dict], max_len: int) -> None:
            self.rows = rows
            self.max_len = max_len
            self.encoded = [
                {
                    key: list(value)
                    for key, value in tokenize_with_assistant_only_loss(
                        tokenizer, row["sft_text"], row["prompt_text"], max_len
                    ).items()
                    if key in ("input_ids", "attention_mask", "labels")
                }
                for row in rows
            ]
            self.lengths = [len(item["input_ids"]) for item in self.encoded]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            return self.encoded[idx]

    def task_sample_weights(dataset: "JsonlChatDataset") -> list[float]:
        counts = {
            task: sum(row.get("task") == task for row in dataset.rows)
            for task in train_tasks
        }
        return [task_probabilities[row["task"]] / counts[row["task"]] for row in dataset.rows]

    assistant_logits_only = bool(cfg.get("assistant_logits_only", False))

    from transformers import TrainerCallback

    class GpuMemoryLogCallback(TrainerCallback):
        """Add peak GPU memory to each log line; reserved near the cap = spill risk."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and torch.cuda.is_available():
                logs["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                logs["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
                torch.cuda.reset_peak_memory_stats()

    class TaskWeightedTrainer(Trainer):
        """Optionally sample tasks at configured probabilities per epoch."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if assistant_logits_only:
                inputs = slice_to_assistant_logits(inputs)
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        def get_train_dataloader(self):
            if token_batch_sampler is None:
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader

            loader = DataLoader(
                self.train_dataset,
                batch_sampler=token_batch_sampler,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return self.accelerator.prepare(loader)

        def _get_train_sampler(self, train_dataset=None):
            if task_probabilities is None:
                return super()._get_train_sampler()
            dataset = self.train_dataset
            if not isinstance(dataset, JsonlChatDataset):
                raise TypeError("Weighted task sampling requires JsonlChatDataset.")
            sample_weights = task_sample_weights(dataset)
            generator = torch.Generator()
            generator.manual_seed(seed)
            return WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )

    # Prefer official chat template when available to locate assistant boundary.
    # We still use stored prompt_text / sft_text which already follow ChatML.
    train_ds = JsonlChatDataset(train_rows, int(cfg["max_seq_length"]))
    val_ds = JsonlChatDataset(val_rows, int(cfg["max_seq_length"]))

    target_samples_per_step = (
        int(cfg["per_device_train_batch_size"])
        * int(cfg["gradient_accumulation_steps"])
    )
    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    token_batching_cfg = dict(cfg.get("token_budget_batching") or {})
    token_batch_sampler = None
    token_batching_summary = None
    if token_batching_cfg.get("enabled", False):
        from token_budget_batching import TokenBudgetBatchSampler

        # Default budget = longest training example, i.e. exactly what the
        # max-length GPU preflight backpropagates, so peak memory is not raised.
        max_tokens = int(token_batching_cfg.get("max_tokens_per_batch") or max(train_ds.lengths))
        token_batch_sampler = TokenBudgetBatchSampler(
            train_ds.lengths,
            max_tokens=max_tokens,
            max_rows=int(token_batching_cfg.get("max_rows_per_batch", 16)),
            weights=task_sample_weights(train_ds) if task_probabilities is not None else None,
            seed=seed,
            megabatch_size=int(token_batching_cfg.get("megabatch_size", 256)),
            pad_multiple=8,
        )
        # Keep ~the same examples per optimizer step as rows=1 x accumulation.
        gradient_accumulation_steps = max(
            1,
            round(target_samples_per_step * len(token_batch_sampler) / len(train_ds)),
        )
        token_batching_summary = token_batch_sampler.describe()
        token_batching_summary["gradient_accumulation_steps"] = gradient_accumulation_steps
        token_batching_summary["approx_examples_per_optimizer_step"] = round(
            gradient_accumulation_steps * token_batching_summary["mean_rows_per_batch"], 1
        )
        print("TOKEN-BUDGET BATCHING:", json.dumps(token_batching_summary))

    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True)) and device == "cuda"
    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    if load_in_4bit and device != "cuda":
        raise SystemExit("4-bit QLoRA is configured but CUDA is unavailable.")
    precision = str(cfg.get("mixed_precision", "fp16")).lower()
    if precision not in ("fp16", "bf16"):
        raise SystemExit("mixed_precision must be 'fp16' or 'bf16'.")
    use_bf16 = (
        precision == "bf16"
        and device == "cuda"
        and not load_in_4bit
        and torch.cuda.is_bf16_supported()
    )
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # On Windows the driver silently spills VRAM into shared system RAM
        # (sysmem fallback) instead of raising OOM, so PyTorch's cache keeps
        # growing and training slows ~8x. Capping the allocator below physical
        # VRAM makes it free and reuse cached blocks instead of spilling.
        memory_fraction = cfg.get("cuda_memory_fraction")
        if memory_fraction:
            memory_fraction = float(memory_fraction)
            if not 0.1 <= memory_fraction <= 1.0:
                raise SystemExit("cuda_memory_fraction must be between 0.1 and 1.0.")
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA MEMORY CAP: {memory_fraction:.2f} x {total_gb:.2f} GB = {memory_fraction * total_gb:.2f} GB")
    print("PRECISION:", "bf16" if use_bf16 else ("fp16" if device == "cuda" else "fp32"))
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": (
            torch.float32 if device == "cpu"
            else torch.bfloat16 if use_bf16
            else torch.float16
        ),
    }
    if cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = str(cfg["attn_implementation"])
    sdpa_repeat_kv = bool(cfg.get("sdpa_repeat_kv_for_gqa", False)) and device == "cuda"
    if sdpa_repeat_kv:
        from attention_patches import force_sdpa_repeat_kv

        sdpa_repeat_kv = force_sdpa_repeat_kv()
        print("SDPA GQA: repeat KV heads so the memory-efficient kernel is used:", sdpa_repeat_kv)
    checkpointing_kwargs = dict(cfg.get("gradient_checkpointing_kwargs", {}))
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "This config requires 4-bit QLoRA. Install bitsandbytes in the active "
                "GPU environment with: python -m pip install bitsandbytes"
            ) from exc
        compute_dtype = torch.float16
        if cfg.get("bnb_4bit_compute_dtype", "float16") != "float16":
            raise SystemExit("This QLoRA config currently supports only float16 compute.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
    if use_gradient_checkpointing:
        # Trade compute for activation memory so long-context LoRA training can fit.
        model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs
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
    if device == "cuda" and not load_in_4bit:
        model.to("cuda")
    model.print_trainable_parameters()
    preflight_result = None
    if device == "cuda" and bool(cfg.get("max_length_memory_preflight", True)):
        preflight_result = run_gpu_memory_preflight(
            model,
            tokenizer,
            train_rows,
            int(cfg["max_seq_length"]),
            float(cfg.get("preflight_min_free_vram_gb", 2.0)),
            torch,
            assistant_logits_only=assistant_logits_only,
        )

    if token_batch_sampler is not None:
        steps_per_epoch = math.ceil(len(token_batch_sampler) / gradient_accumulation_steps)
    else:
        steps_per_epoch = math.ceil(len(train_ds) / target_samples_per_step)
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs=checkpointing_kwargs or None,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        seed=seed,
        bf16=use_bf16,
        fp16=bool(device == "cuda" and not use_bf16),
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if token_batch_sampler is not None else None,
    )
    trainer = TaskWeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.get("early_stopping_patience", 2))
            ),
            *([GpuMemoryLogCallback()] if device == "cuda" else []),
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
        "dataset_layout": "task-separated; merged rows exist only in memory",
        "active_training_tasks": train_tasks,
        "training_selection_manifest": str(selection_path.resolve()),
        "training_selection_manifest_sha256": sha256_file(selection_path),
        "train_rows_by_task": training_selection["train_counts"],
        "validation_rows_by_task": training_selection["validation_counts"],
        "resume_from_checkpoint": str(resume_path) if resume_path else None,
        "max_length_memory_preflight": preflight_result,
        "cuda_memory_fraction": cfg.get("cuda_memory_fraction"),
        "sdpa_repeat_kv_for_gqa": sdpa_repeat_kv,
        "mixed_precision": "bf16" if use_bf16 else ("fp16" if device == "cuda" else "fp32"),
        "token_budget_batching": token_batching_summary,
        "quantization": {
            "load_in_4bit": load_in_4bit,
            "quant_type": cfg.get("bnb_4bit_quant_type") if load_in_4bit else None,
            "double_quant": cfg.get("bnb_4bit_use_double_quant") if load_in_4bit else None,
            "compute_dtype": cfg.get("bnb_4bit_compute_dtype") if load_in_4bit else None,
        },
    }
    (results_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Training complete. Best adapter:", output_dir / "best_adapter")


if __name__ == "__main__":
    main()




