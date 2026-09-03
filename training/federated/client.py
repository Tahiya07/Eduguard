#!/usr/bin/env python
"""Local teacher-client Bloom LoRA training; exports integrity-protected update bundle.

Matches centralized train_qwen_bloom.py architecture.
Supports FedAvg and FedProx. Client DP-SGD may use the FedProx objective;
privacy is from Opacus DP-SGD, not from the proximal term.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from training.paths import ROOT
from training.federated.aggregation import extract_trainable_state, trainable_param_count
from training.federated.communication import (
    attach_client_communication_metadata,
    require_bundle_communication,
    trainable_param_breakdown,
)
from training.federated.class_weights import resolve_class_weights
from training.federated.config import (
    BLOOM_LABELS,
    FederatedLoraConfig,
    TEACHER_ROLE,
    effective_prox_mu,
    make_peft_lora_config,
)
from training.federated.execution_stats import read_trainer_execution_stats
from training.federated.transport import pack_update, save_bundle
from training.federated.dp import load_dp_lock, resolve_dp_lock_path
from training.federated.dp_training import dp_config_from_lock, train_local_adapter_dp

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_bloom import build_prompt  # noqa: E402


class _BloomDS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class _WeightedTrainer(Trainer):
    """Centralized BloomTrainer loss + optional FedProx proximal term."""

    def __init__(
        self,
        class_weights=None,
        label_smoothing: float = 0.0,
        prox_mu: float = 0.0,
        global_params: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.prox_mu = float(prox_mu)
        self._global_params = {k: v.detach().cpu().clone() for k, v in (global_params or {}).items()}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fn(logits, labels)

        if self.prox_mu > 0.0 and self._global_params:
            prox = loss.new_zeros(())
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                g_cpu = self._global_params.get(name)
                if g_cpu is None:
                    continue
                g = g_cpu.to(device=param.device, dtype=param.dtype)
                prox = prox + torch.sum((param - g) ** 2)
            loss = loss + 0.5 * self.prox_mu * prox

        return (loss, outputs) if return_outputs else loss


def _resolve_training_precision() -> tuple[torch.dtype, bool, bool]:
    """Return (model_dtype, trainer_fp16, trainer_bf16) with mutually compatible settings."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    # bf16 avoids GradScaler and is stable for LoRA on modern NVIDIA GPUs.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    # fp16 AMP requires fp32 master weights; loading the full model in fp16 breaks unscaling.
    return torch.float32, True, False


def _load_model_stack(config: FederatedLoraConfig, global_dir: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype, _, _ = _resolve_training_precision()
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if global_dir and (global_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
        print(f"[client] warm-started from global adapter {global_dir}")
    else:
        model = get_peft_model(base, make_peft_lora_config(config))
        print(
            f"[client] fresh LoRA r={config.lora_r} alpha={config.lora_alpha} "
            f"modules_to_save={config.modules_to_save}"
        )
    return tokenizer, model


def train_local_adapter(
    df: pd.DataFrame,
    config: FederatedLoraConfig,
    global_dir: Path | None,
) -> tuple[dict, int, dict]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    tokenizer, model = _load_model_stack(config, global_dir)
    global_named = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    texts = [build_prompt(str(q)) for q in df[config.text_col]]
    labels = [BLOOM_LABELS[str(l)] for l in df[config.label_col]]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=config.max_length)
    ds = _BloomDS(enc, labels)

    warm_started = bool(global_dir and (global_dir / "adapter_config.json").is_file())
    lr = config.finetune_learning_rate if warm_started else config.learning_rate
    if warm_started:
        print(f"[client] warm-start finetune lr={lr}")

    class_weights = resolve_class_weights(config, labels)
    if class_weights is not None:
        print(
            f"[client] class weights ({config.class_weight_source}):",
            [round(float(w), 3) for w in class_weights.tolist()],
        )

    prox_mu = float(config.prox_mu) if config.algorithm == "fedprox" else 0.0
    if prox_mu > 0:
        print(f"[client] FedProx mu={prox_mu}")

    cache_dir = ROOT / "artifacts" / "federated" / "_client_cache"
    _, use_fp16, use_bf16 = _resolve_training_precision()

    optimizer_steps_per_epoch = math.ceil(
        len(ds) / (config.batch_size * config.grad_accum)
    )
    total_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * config.local_epochs
    )
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)

    args = TrainingArguments(
        output_dir=str(cache_dir),
        learning_rate=lr,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.local_epochs,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=config.seed,
    )

    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights,
        label_smoothing=config.label_smoothing,
        prox_mu=prox_mu,
        global_params=global_named if prox_mu > 0 else None,
    )
    trainer.train()
    exec_stats = read_trainer_execution_stats(trainer)

    local_state = extract_trainable_state(model)
    param_breakdown = trainable_param_breakdown(local_state)
    stats = {
        "trainable_parameters": trainable_param_count(local_state),
        "trainable_param_breakdown": param_breakdown,
        "prox_mu": prox_mu,
        "algorithm": config.algorithm,
        **exec_stats,
    }
    return local_state, len(df), stats


def _configure_cpu_threads() -> None:
    import os

    n = int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "8")
    n = max(1, min(n, os.cpu_count() or 8))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(4, n)))
    except RuntimeError:
        pass


def main() -> int:
    _configure_cpu_threads()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Federated teacher client local LoRA training.")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--global-adapter", default=None)
    parser.add_argument("--out-bundle", required=True)
    parser.add_argument("--local-epochs", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--enable-dp", action="store_true")
    parser.add_argument("--dp-scope", default="auto")
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    args = parser.parse_args()

    cfg = FederatedLoraConfig()
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.base_model:
        cfg.base_model = args.base_model
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.algorithm = args.algorithm
    if args.prox_mu is not None:
        cfg.prox_mu = args.prox_mu
    cfg.prox_mu = effective_prox_mu(cfg.algorithm, cfg.prox_mu)

    df = pd.read_csv(args.csv).dropna()
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=cfg.seed)

    global_dir = Path(args.global_adapter) if args.global_adapter else None
    if args.enable_dp:
        lock_path = resolve_dp_lock_path(args.dp_scope)
        lock = load_dp_lock(args.dp_scope)
        dp_cfg = dp_config_from_lock(
            lock,
            noise_multiplier=args.dp_noise_multiplier,
            target_delta=args.dp_delta,
            lock_path=str(lock_path),
        )
        local_state, n, stats = train_local_adapter_dp(
            df, cfg, global_dir, dp_cfg, build_prompt=build_prompt
        )
    else:
        local_state, n, stats = train_local_adapter(df, cfg, global_dir)

    bundle = pack_update(
        client_id=args.client_id,
        round_idx=args.round,
        role=TEACHER_ROLE,
        n_samples=n,
        state=local_state,
    )
    bundle = attach_client_communication_metadata(bundle, local_state)
    bundle["prox_mu"] = stats["prox_mu"]
    bundle["algorithm"] = stats["algorithm"]
    bundle["execution"] = {
        "optimizer_steps_completed": stats.get("optimizer_steps_completed"),
        "epochs_completed": stats.get("epochs_completed"),
        "max_steps": stats.get("max_steps"),
        "source": "trainer.state.global_step",
    }
    if stats.get("dp_enabled"):
        bundle["differential_privacy"] = {
            "enabled": True,
            "dp_mode": stats.get("dp_mode"),
            "noise_multiplier": stats.get("dp_noise_multiplier"),
            "delta": stats.get("dp_delta"),
            "max_grad_norm": stats.get("dp_max_grad_norm"),
            "epsilon_local": stats.get("dp_epsilon_local"),
            "lock_path": stats.get("dp_lock_path"),
            "loss": stats.get("dp_loss"),
        }
    require_bundle_communication(bundle, context=f"client {args.client_id} round {args.round}")
    save_bundle(Path(args.out_bundle), bundle)
    comm = bundle["communication"]
    print(
        f"[client] saved bundle -> {args.out_bundle} "
        f"(n={n}, params={comm['trainable_parameter_count']}, bytes={comm['update_bytes']}, "
        f"optimizer_steps={stats.get('optimizer_steps_completed')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


