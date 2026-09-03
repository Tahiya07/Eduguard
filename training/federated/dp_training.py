"""Client-side DP-SGD for federated Bloom LoRA (Opacus, locked procedure)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.federated.aggregation import extract_trainable_state, trainable_param_count
from training.federated.class_weights import resolve_class_weights
from training.federated.communication import trainable_param_breakdown
from training.federated.config import BLOOM_LABELS, FederatedLoraConfig, effective_prox_mu, make_peft_lora_config

DP_MODE_FULL = "full"
DP_MODE_SCORE_HEAD = "score-head-only"


@dataclass
class DpRuntimeConfig:
    enabled: bool = True
    dp_mode: str = DP_MODE_FULL
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    lock_path: str = ""
    locked_procedure: Dict[str, Any] = field(default_factory=dict)


def normalize_dp_mode(raw: str | None) -> str:
    value = (raw or DP_MODE_FULL).strip().lower().replace("_", "-")
    if value in {DP_MODE_SCORE_HEAD, "score-head", "scorehead"}:
        return DP_MODE_SCORE_HEAD
    return DP_MODE_FULL


def dp_config_from_lock(
    lock: Dict[str, Any],
    *,
    noise_multiplier: float,
    target_delta: float,
    lock_path: str = "",
) -> DpRuntimeConfig:
    dp_mode = normalize_dp_mode(lock.get("dp_mode"))
    fed_cfg = lock.get("federated_config") or {}
    return DpRuntimeConfig(
        enabled=True,
        dp_mode=dp_mode,
        max_grad_norm=float(fed_cfg.get("max_grad_norm", 1.0)),
        noise_multiplier=float(noise_multiplier),
        target_delta=float(target_delta),
        lock_path=lock_path,
        locked_procedure=dict(lock.get("locked_procedure") or {}),
    )


def apply_locked_training_rules(cfg: FederatedLoraConfig, dp: DpRuntimeConfig) -> FederatedLoraConfig:
    """Enforce Phase-2A locked loss / dropout settings during DP training."""
    updates: Dict[str, Any] = {
        "label_smoothing": 0.0,
        "use_class_weights": False,
    }
    if dp.dp_mode == DP_MODE_FULL:
        updates["lora_dropout"] = float(
            (dp.locked_procedure or {}).get("validation_lora_dropout", 0.0)
        )
    return replace(cfg, **updates)


def _load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_dp_model_stack(
    cfg: FederatedLoraConfig,
    dp: DpRuntimeConfig,
    global_dir,
):
    tokenizer = _load_tokenizer(cfg.base_model)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if dp.dp_mode == DP_MODE_SCORE_HEAD:
        if global_dir and (global_dir / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=False)
        else:
            model = get_peft_model(base, make_peft_lora_config(cfg))
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = False
        for name, param in model.named_parameters():
            param.requires_grad = "score" in name
        return tokenizer, model

    if global_dir and (global_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
    else:
        model = get_peft_model(base, make_peft_lora_config(cfg))
    return tokenizer, model


def _collate_batch(batch, tokenizer, max_length: int):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(list(labels), dtype=torch.long)
    return enc


def _device_for_model(model: nn.Module) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        return device
    return torch.device("cpu")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def compose_federated_privacy_report(
    *,
    client_epsilons: List[float],
    rounds: int,
    delta: float,
    noise_multiplier: float,
    dp_mode: str,
) -> Dict[str, Any]:
  local_max = max(client_epsilons) if client_epsilons else None
  local_mean = sum(client_epsilons) / len(client_epsilons) if client_epsilons else None
  return {
      "mechanism": "client_dp_sgd_opacus",
      "dp_mode": dp_mode,
      "noise_multiplier": noise_multiplier,
      "delta": delta,
      "local_epsilon_max": local_max,
      "local_epsilon_mean": local_mean,
      "naive_composition_upper_bound": (rounds * local_max) if local_max is not None else None,
      "composition_note": (
          "Per-client epsilon is computed by Opacus for one local DP-SGD run. "
          "Federated composition uses a conservative naive upper bound (rounds * max local epsilon). "
          "Tighter FL accounting requires a dedicated accountant."
      ),
  }


def _strip_opacus_prefix(name: str) -> str:
    while name.startswith("_module."):
        name = name[len("_module.") :]
    return name


def _canonical_trainable_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_strip_opacus_prefix(k): v for k, v in state.items()}


def _proximal_penalty(model: nn.Module, global_params: Dict[str, torch.Tensor], prox_mu: float) -> torch.Tensor:
    prox = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        g_cpu = global_params.get(name) or global_params.get(_strip_opacus_prefix(name))
        if g_cpu is None:
            continue
        g = g_cpu.to(device=param.device, dtype=param.dtype)
        term = torch.sum((param - g) ** 2)
        prox = term if prox is None else prox + term
    if prox is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return 0.5 * float(prox_mu) * prox


def train_local_adapter_dp(
    df: pd.DataFrame,
    config: FederatedLoraConfig,
    global_dir,
    dp: DpRuntimeConfig,
    *,
    build_prompt,
) -> Tuple[dict, int, dict]:
    algorithm = (config.algorithm or "fedavg").lower().strip()
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported federated algorithm for DP-SGD: {algorithm}")
    prox_mu = effective_prox_mu(algorithm, config.prox_mu)

    cfg = apply_locked_training_rules(config, dp)
    tokenizer, model = _load_dp_model_stack(cfg, dp, global_dir)
    device = _device_for_model(model)

    global_named = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    rows = [
        (build_prompt(str(q)), BLOOM_LABELS[str(l)])
        for q, l in zip(df[cfg.text_col], df[cfg.label_col])
    ]

    warm_started = bool(global_dir and (global_dir / "adapter_config.json").is_file())
    lr = cfg.finetune_learning_rate if warm_started else cfg.learning_rate

    optimizer_steps_per_epoch = max(1, math.ceil(len(rows) / max(1, cfg.batch_size)))
    total_optimizer_steps = max(1, math.ceil(optimizer_steps_per_epoch * cfg.local_epochs))

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("DP training has no trainable parameters after scope setup.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=cfg.weight_decay,
    )

    loader = DataLoader(
        rows,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer, cfg.max_length),
    )

    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise RuntimeError("Opacus is required for DP training.") from exc

    model = ModuleValidator.fix(model)
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
    )

    model.train()
    steps_done = 0
    while steps_done < total_optimizer_steps:
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            if prox_mu > 0.0:
                loss = loss + _proximal_penalty(model, global_named, prox_mu)
            loss.backward()
            optimizer.step()
            steps_done += 1
            if steps_done >= total_optimizer_steps:
                break

    epsilon = float(privacy_engine.get_epsilon(dp.target_delta))
    local_state = _canonical_trainable_state(extract_trainable_state(model))
    param_breakdown = trainable_param_breakdown(local_state)
    stats = {
        "trainable_parameters": trainable_param_count(local_state),
        "trainable_param_breakdown": param_breakdown,
        "prox_mu": prox_mu,
        "algorithm": algorithm,
        "optimizer_steps_completed": steps_done,
        "epochs_completed": cfg.local_epochs,
        "max_steps": total_optimizer_steps,
        "dp_enabled": True,
        "dp_mode": dp.dp_mode,
        "dp_noise_multiplier": dp.noise_multiplier,
        "dp_delta": dp.target_delta,
        "dp_max_grad_norm": dp.max_grad_norm,
        "dp_epsilon_local": epsilon,
        "dp_lock_path": dp.lock_path,
        "dp_loss": "uniform_cross_entropy",
        "class_weights_skipped": True,
    }
    return local_state, len(df), stats
