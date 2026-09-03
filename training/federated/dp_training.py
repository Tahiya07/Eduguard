"""Client-side DP-SGD for federated Bloom LoRA (Opacus, locked procedure)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.federated.aggregation import extract_trainable_state, trainable_param_count
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


def _load_sequence_classifier(base_model: str, num_labels: int):
    kwargs = {
        "num_labels": num_labels,
        "trust_remote_code": True,
    }
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            base_model, dtype=torch.float32, **kwargs
        )
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(
            base_model, torch_dtype=torch.float32, **kwargs
        )


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model._module if hasattr(model, "_module") else model


def _make_inputs_require_grads(_module, _inputs, output):
    """Module-level hook so Opacus clone_module/pickle can succeed."""
    if torch.is_tensor(output):
        output.requires_grad_(True)
    return output


def _enable_input_require_grads(model: nn.Module) -> None:
    root = _unwrap_module(model)
    if getattr(root, "_eduguard_input_grad_hook", False):
        return
    embed = root.get_input_embeddings() if hasattr(root, "get_input_embeddings") else None
    if embed is None:
        return
    embed.register_forward_hook(_make_inputs_require_grads)
    root._eduguard_input_grad_hook = True


def _maybe_fix_opacus_modules(model: nn.Module) -> nn.Module:
    from opacus.validators import ModuleValidator

    # Do not register forward hooks before this: ModuleValidator.fix() pickles the module.
    if ModuleValidator.is_valid(model):
        return model
    print("[dp] ModuleValidator.fix: replacing Opacus-incompatible submodules")
    try:
        return ModuleValidator.fix(model)
    except Exception as exc:
        print(f"[dp] ModuleValidator.fix skipped ({type(exc).__name__}: {exc}); using original module")
        return model


def _load_dp_model_stack(
    cfg: FederatedLoraConfig,
    dp: DpRuntimeConfig,
    global_dir,
):
    tokenizer = _load_tokenizer(cfg.base_model)
    base = _load_sequence_classifier(cfg.base_model, len(BLOOM_LABELS))
    base.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base, "config", None) is not None:
        base.config.use_cache = False

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
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return tokenizer, model


def _collate_batch(batch, tokenizer, max_length: int):
    """Legacy text collate kept for unit tests; training uses tokenized tensors."""
    if not batch:
        return {
            "input_ids": torch.zeros((0, 1), dtype=torch.long),
            "attention_mask": torch.zeros((0, 1), dtype=torch.long),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
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


class _TokenizedBloomDataset(Dataset):
    """Map-style tensor dataset so Opacus empty-batch zeros use real torch dtypes.

    Opacus inspects ``dataset[0]`` to build empty Poisson batches. Text/label
    tuples make it pass Python ``str``/``int`` as dtypes and crash.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def _collate_tokenized(batch):
    if not batch:
        return {
            "input_ids": torch.zeros((0, 1), dtype=torch.long),
            "attention_mask": torch.zeros((0, 1), dtype=torch.long),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    input_ids, attention_mask, labels = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def _normalize_batch(batch) -> Dict[str, torch.Tensor]:
    """Accept dict batches or Opacus empty placeholders (list of zero tensors)."""
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and all(torch.is_tensor(t) for t in batch):
        return {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }
    raise TypeError(f"Unsupported DP batch type: {type(batch)!r}")


def _batch_size(batch) -> int:
    if not isinstance(batch, dict):
        try:
            batch = _normalize_batch(batch)
        except TypeError:
            return 0
    ids = batch.get("input_ids")
    if ids is not None and hasattr(ids, "shape") and ids.ndim >= 1:
        return int(ids.shape[0])
    labels = batch.get("labels")
    if labels is not None and hasattr(labels, "shape") and labels.ndim >= 1:
        return int(labels.shape[0])
    return 0


def _iter_nonempty_batches(loader, *, max_iters: int):
    """Yield (batch, loader_iters, empty_skipped) for non-empty DP batches."""
    iters = 0
    skipped = 0
    while iters < max_iters:
        data_iter = iter(loader)
        exhausted = True
        while iters < max_iters:
            try:
                raw = next(data_iter)
            except StopIteration:
                break
            except TypeError as exc:
                if "dtype" not in str(exc).lower():
                    raise
                skipped += 1
                iters += 1
                exhausted = False
                continue
            exhausted = False
            iters += 1
            try:
                batch = _normalize_batch(raw)
            except TypeError:
                skipped += 1
                continue
            if _batch_size(batch) == 0:
                skipped += 1
                continue
            yield batch, iters, skipped
        if exhausted:
            raise RuntimeError("DP DataLoader produced no batches")
    return


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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = apply_locked_training_rules(config, dp)
    tokenizer, model = _load_dp_model_stack(cfg, dp, global_dir)

    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise RuntimeError("Opacus is required for DP training.") from exc

    # Validator.fix pickles the module; input-grad hooks must be attached after that.
    model = _maybe_fix_opacus_modules(model)
    device = _device_for_model(model)
    _enable_input_require_grads(model)

    global_named = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    rows_text = [
        build_prompt(str(q))
        for q in df[cfg.text_col]
    ]
    rows_labels = [BLOOM_LABELS[str(l)] for l in df[cfg.label_col]]
    dataset = _TokenizedBloomDataset(rows_text, rows_labels, tokenizer, cfg.max_length)

    warm_started = bool(global_dir and (global_dir / "adapter_config.json").is_file())
    lr = cfg.finetune_learning_rate if warm_started else cfg.learning_rate

    optimizer_steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, cfg.batch_size)))
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
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate_tokenized,
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
        poisson_sampling=True,
        grad_sample_mode="hooks",
    )
    _enable_input_require_grads(model)

    model.train()
    steps_done = 0
    empty_batches_skipped = 0
    max_loader_iters = max(total_optimizer_steps * 20, total_optimizer_steps + 8)
    loader_iters = 0
    for batch, loader_iters, empty_batches_skipped in _iter_nonempty_batches(
        loader, max_iters=max_loader_iters
    ):
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
    if steps_done < total_optimizer_steps:
        raise RuntimeError(
            f"DP-SGD stopped after {steps_done}/{total_optimizer_steps} steps "
            f"(empty Poisson batches skipped={empty_batches_skipped})."
        )
    if empty_batches_skipped:
        print(f"[dp] skipped {empty_batches_skipped} empty Poisson-sampled batches")

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
        "dp_poisson_sampling": True,
        "dp_empty_batches_skipped": empty_batches_skipped,
        "dp_dataset": "tokenized_tensor",
    }
    return local_state, len(df), stats
