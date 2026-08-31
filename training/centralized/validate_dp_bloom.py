#!/usr/bin/env python
"""Phase 2A: Mandatory DP-SGD validation gate for Bloom LoRA training.

Does NOT claim differential privacy until all gates pass and
artifacts/privacy/dp_bloom_validated_v1.json is written.

Run from repository root:
  python -m training.centralized.validate_dp_bloom --output artifacts/privacy/dp_bloom_validated_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from training.paths import ARTIFACTS_PRIVACY, ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_bloom import build_prompt  # noqa: E402
from training.federated.config import BLOOM_LABELS, FederatedLoraConfig, make_peft_lora_config  # noqa: E402


DEFAULT_OUTPUT = ARTIFACTS_PRIVACY / "dp_bloom_validated_v1.json"


@dataclass
class GateResult:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_tiny_batch(n: int = 4, seed: int = 42):
    import pandas as pd
    from transformers import AutoTokenizer

    rng = torch.Generator().manual_seed(seed)
    labels = list(BLOOM_LABELS.keys())
    rows = []
    for i in range(n):
        label = labels[i % len(labels)]
        rows.append({"question": f"Sample question {i} about topic {label}", "bloom_level": label})
    df = pd.DataFrame(rows)
    cfg = FederatedLoraConfig()
    texts = [build_prompt(str(q)) for q in df["question"]]
    y = [BLOOM_LABELS[str(l)] for l in df["bloom_level"]]
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.long)
    return cfg, tokenizer, enc


def _load_peft_model(cfg: FederatedLoraConfig, tokenizer=None):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    return get_peft_model(base, make_peft_lora_config(cfg)), tokenizer


def gate_opacus_import() -> GateResult:
    try:
        import opacus  # noqa: F401
        from opacus import PrivacyEngine  # noqa: F401

        version = getattr(opacus, "__version__", "unknown")
        return GateResult("opacus_import", True, {"opacus_version": version})
    except ImportError as exc:
        return GateResult("opacus_import", False, failure_reason=str(exc))


def _manual_per_sample_grads(model, batch: dict, trainable_names: List[str]) -> Dict[str, torch.Tensor]:
    """Per-example backward on a fresh model (weights unchanged)."""
    manual: Dict[str, List[torch.Tensor]] = {}
    for i in range(batch["input_ids"].shape[0]):
        model.zero_grad(set_to_none=True)
        single = {k: v[i : i + 1] for k, v in batch.items() if k != "labels"}
        labels = batch["labels"][i : i + 1]
        out = model(**single)
        loss = nn.CrossEntropyLoss()(out.logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if name in trainable_names and param.grad is not None:
                manual.setdefault(name, []).append(param.grad.detach().clone())
    return {name: torch.stack(tensors, dim=0) for name, tensors in manual.items()}


def gate_per_sample_gradients(
    cfg: FederatedLoraConfig, batch: dict, tokenizer, rtol: float = 1e-3
) -> GateResult:
    """Compare Opacus per-sample grads to manual per-example backward."""
    try:
        from opacus.grad_sample import GradSampleModule
    except ImportError as exc:
        return GateResult("per_sample_gradients", False, failure_reason=str(exc))

    model_manual, _ = _load_peft_model(cfg, tokenizer)
    model_gsm, _ = _load_peft_model(cfg, tokenizer)
    model_gsm.load_state_dict(model_manual.state_dict())

    trainable_names = [n for n, p in model_manual.named_parameters() if p.requires_grad]
    score_names = [n for n in trainable_names if "score" in n]
    lora_names = [n for n in trainable_names if n not in score_names]

    model_manual.train()
    manual = _manual_per_sample_grads(model_manual, batch, trainable_names)

    gsm = GradSampleModule(model_gsm)
    gsm.train()
    gsm.zero_grad(set_to_none=True)
    out = gsm(**{k: v for k, v in batch.items() if k != "labels"})
    loss = nn.CrossEntropyLoss()(out.logits, batch["labels"])
    loss.backward()

    gsm_by_name = {n: p for n, p in gsm.named_parameters() if p.requires_grad}
    mismatches: List[str] = []
    checked = 0
    score_checked = 0
    lora_checked = 0
    internal_ok = 0

    for name in trainable_names:
        param = gsm_by_name.get(name)
        if param is None:
            mismatches.append(f"missing gsm param {name}")
            continue
        if not hasattr(param, "grad_sample") or param.grad_sample is None:
            mismatches.append(f"missing grad_sample for {name}")
            continue
        ref = manual.get(name)
        if ref is None:
            continue
        checked += 1
        if name in score_names:
            score_checked += 1
        else:
            lora_checked += 1
        gs = param.grad_sample.detach().cpu()
        if gs.shape != ref.shape:
            mismatches.append(f"shape {gs.shape} vs {ref.shape} for {name}")
            continue
        if param.grad is not None:
            mean_gs = gs.mean(dim=0)
            if torch.allclose(param.grad.detach().cpu().float(), mean_gs.float(), rtol=rtol, atol=1e-4):
                internal_ok += 1
        if not torch.allclose(gs.float(), ref.float(), rtol=rtol, atol=1e-4):
            max_err = (gs.float() - ref.float()).abs().max().item()
            mismatches.append(f"max_err={max_err:.6f} for {name}")

    details = {
        "checked_params": checked,
        "score_head_checked": score_checked,
        "lora_checked": lora_checked,
        "gsm_internal_mean_consistency": internal_ok,
        "mismatches": mismatches[:15],
    }

    if checked == 0:
        return GateResult(
            "per_sample_gradients",
            False,
            details,
            failure_reason="no overlapping trainable params between manual and GradSampleModule",
        )

    score_only_pass = score_checked > 0 and not any("score" in m for m in mismatches)
    if mismatches:
        details["score_head_only_pass"] = score_only_pass
        return GateResult(
            "per_sample_gradients",
            False,
            details,
            failure_reason="per-sample gradient mismatch (LoRA+PEFT may require Opacus-native loop or score-head-only DP)",
        )
    return GateResult("per_sample_gradients", True, details)


def gate_score_head_only_diagnostic(
    cfg: FederatedLoraConfig, batch: dict, tokenizer, rtol: float = 1e-3
) -> GateResult:
    """DIAGNOSTIC ONLY: frozen base + trainable score head per-sample gradients.

    Isolates whether Opacus per-sample grad mismatch is LoRA/PEFT-specific.
    NOT a replacement for full LoRA+score DP validation.
    """
    try:
        from opacus.grad_sample import GradSampleModule
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        return GateResult("score_head_only_diagnostic", False, failure_reason=str(exc))

    def _load_score_only():
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(BLOOM_LABELS),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        for p in base.parameters():
            p.requires_grad = False
        for name, p in base.named_parameters():
            if "score" in name:
                p.requires_grad = True
        return base

    model_manual = _load_score_only()
    model_gsm = _load_score_only()
    model_gsm.load_state_dict(model_manual.state_dict())

    trainable_names = [n for n, p in model_manual.named_parameters() if p.requires_grad]
    model_manual.train()
    manual = _manual_per_sample_grads(model_manual, batch, trainable_names)

    gsm = GradSampleModule(model_gsm)
    gsm.train()
    gsm.zero_grad(set_to_none=True)
    out = gsm(**{k: v for k, v in batch.items() if k != "labels"})
    loss = nn.CrossEntropyLoss()(out.logits, batch["labels"])
    loss.backward()

    mismatches = []
    checked = 0
    gsm_by_name = {n: p for n, p in gsm.named_parameters() if p.requires_grad}
    for name in trainable_names:
        param = gsm_by_name.get(name)
        if param is None or not hasattr(param, "grad_sample") or param.grad_sample is None:
            mismatches.append(f"missing grad_sample for {name}")
            continue
        ref = manual.get(name)
        if ref is None:
            continue
        checked += 1
        gs = param.grad_sample.detach().cpu()
        if not torch.allclose(gs.float(), ref.float(), rtol=rtol, atol=1e-4):
            max_err = (gs.float() - ref.float()).abs().max().item()
            mismatches.append(f"max_err={max_err:.6f} for {name}")

    details = {
        "diagnostic_only": True,
        "checked_params": checked,
        "trainable_param_names": trainable_names,
        "mismatches": mismatches[:10],
        "interpretation": (
            "If this passes but LoRA+score fails, mismatch is likely PEFT/LoRA integration. "
            "If this also fails, investigate loss/objective or Opacus setup."
        ),
    }
    passed = checked > 0 and not mismatches
    return GateResult(
        "score_head_only_diagnostic",
        passed,
        details,
        None if passed else "score-head-only per-sample gradient mismatch",
    )


def gate_clipping(
    cfg: FederatedLoraConfig, batch: dict, tokenizer, max_grad_norm: float = 1.0
) -> GateResult:
    from torch.optim import SGD
    from torch.utils.data import DataLoader, TensorDataset

    model, _ = _load_peft_model(cfg, tokenizer)
    dataset = TensorDataset(
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )

    def collate(samples):
        input_ids = torch.stack([s[0] for s in samples])
        attention_mask = torch.stack([s[1] for s in samples])
        labels = torch.stack([s[2] for s in samples])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)
    optimizer = SGD(model.parameters(), lr=0.01)
    try:
        from opacus import PrivacyEngine

        pe = PrivacyEngine()
        model, optimizer, loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
        )
    except Exception as exc:
        return GateResult("clipping", False, failure_reason=f"make_private failed: {exc}")

    model.train()
    for batch_in in loader:
        model.zero_grad()
        labels = batch_in.pop("labels")
        out = model(**batch_in)
        loss = nn.CrossEntropyLoss()(out.logits, labels)
        loss.backward()
        optimizer.step()
        break

    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(float(p.grad.norm().item()))
    post_max = max(norms) if norms else 0.0
    ok = post_max <= max_grad_norm + 1e-3 or max_grad_norm <= 0
    return GateResult(
        "clipping",
        ok,
        {"max_grad_norm": max_grad_norm, "post_step_max_grad_norm": post_max},
        None if ok else "gradient norm exceeds clip bound after step",
    )


def gate_accounting_monotonicity() -> GateResult:
    try:
        from opacus.accountants import RDPAccountant
    except ImportError as exc:
        return GateResult("accounting_monotonicity", False, failure_reason=str(exc))

    accountant = RDPAccountant()
    noise = 1.0
    sample_rate = 0.01
    steps_a = 100
    steps_b = 200
    for _ in range(steps_a):
        accountant.step(noise_multiplier=noise, sample_rate=sample_rate)
    eps_a = accountant.get_epsilon(delta=1e-5)
    for _ in range(steps_b - steps_a):
        accountant.step(noise_multiplier=noise, sample_rate=sample_rate)
    eps_b = accountant.get_epsilon(delta=1e-5)
    ok = eps_b >= eps_a and eps_a >= 0
    return GateResult(
        "accounting_monotonicity",
        ok,
        {"epsilon_after_100_steps": eps_a, "epsilon_after_200_steps": eps_b, "delta": 1e-5},
        None if ok else "epsilon did not increase monotonically with steps",
    )


def gate_loss_variants(cfg: FederatedLoraConfig, batch: dict, tokenizer) -> GateResult:
    """Test whether label smoothing / class weights break grad_sample."""
    results = {}
    for variant, smoothing, weights in [
        ("uniform_ce", 0.0, None),
        ("label_smoothing", 0.05, None),
        ("class_weights", 0.0, torch.ones(len(BLOOM_LABELS))),
    ]:
        try:
            from opacus.grad_sample import GradSampleModule

            model = _load_peft_model(cfg, tokenizer)[0]
            gsm = GradSampleModule(model)
            gsm.train()
            gsm.zero_grad()
            out = gsm(**{k: v for k, v in batch.items() if k != "labels"})
            loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            has_gs = any(
                hasattr(p, "grad_sample") and p.grad_sample is not None
                for p in gsm.parameters()
                if p.requires_grad
            )
            results[variant] = {"grad_sample_present": has_gs}
        except Exception as exc:
            results[variant] = {"error": str(exc)}

    uniform_ok = results.get("uniform_ce", {}).get("grad_sample_present", False)
    smoothing_ok = results.get("label_smoothing", {}).get("grad_sample_present", False)
    weights_ok = results.get("class_weights", {}).get("grad_sample_present", False)

    recommended = "uniform_ce"
    if not smoothing_ok or not weights_ok:
        recommended = "uniform_ce_no_smoothing_no_class_weights"

    return GateResult(
        "loss_variants",
        uniform_ok,
        {"variants": results, "recommended_dp_loss": recommended},
        None if uniform_ok else "uniform CE grad_sample failed",
    )


def run_validation(output_path: Path, *, skip_model_gates: bool = False) -> dict:
    gates: List[GateResult] = []
    diagnostics: List[GateResult] = []
    gates.append(gate_opacus_import())
    gates.append(gate_accounting_monotonicity())

    cfg, tokenizer, batch = _build_tiny_batch()
    if not skip_model_gates and gates[0].passed:
        gates.append(gate_per_sample_gradients(cfg, batch, tokenizer))
        diagnostics.append(gate_score_head_only_diagnostic(cfg, batch, tokenizer))
        gates.append(gate_clipping(cfg, batch, tokenizer))
        gates.append(gate_loss_variants(cfg, batch, tokenizer))

    all_passed = all(g.passed for g in gates)
    report = {
        "format": "dp_bloom_validation_report_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gate_passed": all_passed,
        "pytorch_version": torch.__version__,
        "train_script": str(ROOT / "train_qwen_bloom.py"),
        "train_script_sha256": _file_sha256(ROOT / "train_qwen_bloom.py"),
        "federated_config": FederatedLoraConfig().to_dict(),
        "gates": [
            {
                "name": g.name,
                "passed": g.passed,
                "details": g.details,
                "failure_reason": g.failure_reason,
            }
            for g in gates
        ],
        "diagnostics": [
            {
                "name": g.name,
                "passed": g.passed,
                "details": g.details,
                "failure_reason": g.failure_reason,
                "note": "Non-blocking diagnostic — does not affect validation_gate_passed",
            }
            for g in diagnostics
        ],
    }

    if all_passed:
        report["locked_procedure"] = {
            "base_model": cfg.base_model,
            "lora": FederatedLoraConfig().lora_config_dict(),
            "loss": "uniform_cross_entropy",
            "label_smoothing": 0.0,
            "class_weights": None,
            "accountant": "Opacus RDPAccountant",
            "note": "Use this locked config for any formal DP claim.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[dp-validation] PASSED -> {output_path}")
    else:
        failed = [g.name for g in gates if not g.passed]
        print(f"[dp-validation] FAILED gates: {failed}")
        fail_path = output_path.parent / "dp_validation_failed_latest.json"
        failure_path = output_path.parent / "dp_validation_failure.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        failure_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2A DP validation gate for Bloom LoRA.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-model-gates", action="store_true", help="Only run import/accounting gates.")
    args = parser.parse_args()
    report = run_validation(Path(args.output), skip_model_gates=args.skip_model_gates)
    return 0 if report["validation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
