#!/usr/bin/env python
"""Evaluate multi-task Qwen models on the locked held-out TEST split.

Evaluates Bloom rewrite, SQuAD-style QA, and PubMed summarization.
Does not modify production code or models/qwen.gguf.

Usage:
  python evaluate_rewrite.py --config configs/qwen05b_multitask.json --condition lora
  python evaluate_rewrite.py --config configs/qwen05b_multitask.json --condition base
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloom_validation import validate_bloom_example  # noqa: E402
from eval_classifier import ClassifierCallStats, load_classifier  # noqa: E402
from eval_dataset import load_test_split  # noqa: E402
from eval_metrics import (  # noqa: E402
    CLASSIFIER_CONFIDENCE_MIN,
    aggregate_qa_metrics,
    bloom_classification_metrics,
    bloom_rates,
    canonical_bloom,
    failure_analysis,
    latency_summary,
    optional_bertscore,
    rouge_scores,
    source_target_matrix,
)
from eval_model import (  # noqa: E402
    HFGenerator,
    resolve_checkpoint,
    sample_resources,
    set_seed,
    validate_checkpoint,
)
from paths import (  # noqa: E402
    CONFIG_DIR,
    HUMAN_EVAL_DIR,
    MULTITASK_DATA_DIR,
    SEED,
    TASK_BLOOM,
    TASK_QA,
    TASK_SUMMARIZATION,
)
from prompts import (  # noqa: E402
    FORBIDDEN_SOURCE_LEVEL_MARKERS,
    IM_END,
    IM_START,
    assert_no_source_level_in_prompt,
    build_generation_prompt,
)

FAILURE_CATEGORIES = (
    "WRONG_TARGET_LEVEL",
    "DECLARATIVE_OUTPUT",
    "TOPIC_DRIFT",
    "MEANING_DRIFT",
    "TRIVIAL_VERB_SWAP",
    "META_RESPONSE",
    "INVALID_QUESTION",
    "EMPTY_OUTPUT",
    "LOW_CONFIDENCE",
    "OTHER",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "numpy", "rouge_score", "bert_score"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "ok")
        except ImportError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def clean_generation(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace(IM_END, "").replace(IM_START, "")
    cleaned = re.sub(
        r"(?im)^(bloom level|reason|rewrite|question|answer|the rewritten question is)\s*:\s*",
        "",
        cleaned,
    )
    return re.sub(r"\s+", " ", cleaned.strip().strip('"').strip("'"))


def classify_failure(record: dict[str, Any]) -> str:
    if record.get("empty_output"):
        return "EMPTY_OUTPUT"
    if record.get("meta_response"):
        return "META_RESPONSE"
    if record.get("answer_output"):
        return "DECLARATIVE_OUTPUT"
    if record.get("trivial_transform"):
        return "TRIVIAL_VERB_SWAP"
    if not record.get("semantic_valid"):
        return "TOPIC_DRIFT"
    if not record.get("format_valid"):
        return "INVALID_QUESTION"
    if not record.get("cognitive_valid"):
        return "INVALID_QUESTION"
    if record.get("target_match") is False:
        return "WRONG_TARGET_LEVEL"
    if record.get("classifier_confidence", 0.0) < CLASSIFIER_CONFIDENCE_MIN:
        return "LOW_CONFIDENCE"
    return "OTHER"


def evaluate_bloom_row(
    row: dict[str, Any],
    generator: HFGenerator,
    gen_cfg: dict[str, Any],
    classify_fn,
    clf_stats: ClassifierCallStats | None = None,
) -> dict[str, Any]:
    prompt = build_generation_prompt(TASK_BLOOM, row)
    assert_no_source_level_in_prompt(prompt)
    for marker in FORBIDDEN_SOURCE_LEVEL_MARKERS:
        if marker.lower() in prompt.lower():
            raise AssertionError("source_bloom_level leaked into generation prompt")
    if row.get("source_bloom_level"):
        src = str(row["source_bloom_level"]).lower()
        if f"target bloom level:\n{src}" in prompt.lower().replace(" ", ""):
            pass  # target only
        # Ensure source level text doesn't appear as its own field
        if f"original bloom level:\n{src}" in prompt.lower():
            raise AssertionError("source_bloom_level in prompt")

    t0 = time.perf_counter()
    raw = generator.generate(prompt, gen_cfg)
    latency = time.perf_counter() - t0
    prediction = clean_generation(raw)

    validation = validate_bloom_example(
        row["source_question"], row["target_bloom_level"], prediction
    )
    meta_response = "meta_response" in (validation.rejection_reason or "")
    answer_output = "answer_or_declarative" in (validation.rejection_reason or "")
    empty_output = not prediction

    clf_pred = None
    clf_conf = 0.0
    if classify_fn and prediction:
        try:
            clf = classify_fn(prediction)
            clf_pred = clf["predicted_level"]
            clf_conf = clf["confidence"]
            if clf_stats is not None:
                clf_stats.record_ok()
        except Exception as exc:  # noqa: BLE001
            if clf_stats is not None:
                clf_stats.record_fail(exc)
                if clf_stats.n_fail == 1:
                    print(
                        "WARNING: Bloom classifier call failed "
                        f"(will count as MISSING): {type(exc).__name__}: {exc}"
                    )
            clf_pred = None
            clf_conf = 0.0

    target = canonical_bloom(row["target_bloom_level"])
    target_match = bool(clf_pred and canonical_bloom(clf_pred) == target)
    fully_validated = (
        validation.format_valid
        and validation.semantic_valid
        and validation.cognitive_valid
        and not validation.trivial_transform
        and target_match
        and clf_conf >= CLASSIFIER_CONFIDENCE_MIN
    )

    rec = {
        "id": row.get("example_id"),
        "task": TASK_BLOOM,
        "source_question": row["source_question"],
        "source_bloom_level": row.get("source_bloom_level"),
        "target_bloom_level": row["target_bloom_level"],
        "reference": row.get("target_rewrite"),
        "prediction": prediction,
        "format_valid": validation.format_valid,
        "semantic_valid": validation.semantic_valid,
        "cognitive_valid": validation.cognitive_valid,
        "topic_preserved": validation.topic_preserved,
        "trivial_transform": validation.trivial_transform,
        "semantic_similarity": validation.semantic_similarity,
        "meta_response": meta_response,
        "answer_output": answer_output,
        "empty_output": empty_output,
        "classifier_prediction": clf_pred,
        "classifier_confidence": clf_conf,
        "target_match": target_match,
        "fully_validated": fully_validated,
        "latency_s": round(latency, 6),
    }
    rec["failure_category"] = classify_failure(rec) if not fully_validated else ""
    return rec


def evaluate_qa_row(
    row: dict[str, Any], generator: HFGenerator, gen_cfg: dict[str, Any]
) -> dict[str, Any]:
    prompt = build_generation_prompt(TASK_QA, row)
    t0 = time.perf_counter()
    raw = generator.generate(prompt, gen_cfg)
    latency = time.perf_counter() - t0
    prediction = clean_generation(raw)
    return {
        "id": row.get("example_id"),
        "task": TASK_QA,
        "context": row.get("context"),
        "question": row.get("question"),
        "reference": row.get("answer"),
        "prediction": prediction,
        "latency_s": round(latency, 6),
    }


def evaluate_sum_row(
    row: dict[str, Any], generator: HFGenerator, gen_cfg: dict[str, Any]
) -> dict[str, Any]:
    prompt = build_generation_prompt(TASK_SUMMARIZATION, row)
    t0 = time.perf_counter()
    raw = generator.generate(prompt, gen_cfg)
    latency = time.perf_counter() - t0
    prediction = clean_generation(raw)
    return {
        "id": row.get("example_id"),
        "task": TASK_SUMMARIZATION,
        "article": row.get("article"),
        "reference": row.get("abstract"),
        "prediction": prediction,
        "latency_s": round(latency, 6),
    }


def results_dir_for(cfg: dict[str, Any], condition: str) -> Path:
    key = cfg.get("model_key", "model")
    if condition == "base":
        return EXPERIMENT_DIR / "results" / f"qwen{key}_base"
    return Path(cfg.get("results_dir", EXPERIMENT_DIR / "results" / f"qwen{key}_lora"))


def write_human_eval_export(bloom_records: list[dict[str, Any]], out_dir: Path) -> Path | None:
    export = HUMAN_EVAL_DIR / "predictions_for_rating.jsonl"
    if not bloom_records:
        return None
    lines = []
    for r in bloom_records:
        lines.append(
            json.dumps(
                {
                    "id": r["id"],
                    "original_question": r["source_question"],
                    "requested_target_level": r["target_bloom_level"],
                    "generated_rewrite": r["prediction"],
                    "model_label": "blinded_A",
                    "ratings": {
                        "bloom_target_correctness": None,
                        "cognitive_demand": None,
                        "topic_preservation": None,
                        "academic_quality": None,
                        "grammar": None,
                        "overall": None,
                    },
                },
                ensure_ascii=False,
            )
        )
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return export


def write_report(
    out_dir: Path,
    metrics: dict[str, Any],
    dataset_meta: dict[str, Any],
    checkpoint_info: dict[str, Any],
) -> None:
    b = metrics.get("bloom", {})
    q = metrics.get("qa", {}).get("squad", {})
    s = metrics.get("summarization", {})
    lat = metrics.get("latency", {})
    res = metrics.get("resources", {})
    lines = [
        "# Multi-task Evaluation Report",
        "",
        f"Generated (UTC): {metrics.get('evaluated_utc')}",
        f"Condition: **{metrics.get('condition')}**",
        f"Model: `{metrics.get('model_id')}`",
        "",
        "## Dataset",
        "",
        f"- Test count: **{dataset_meta['counts']['test_count']}**",
        f"- Corpus hash: `{dataset_meta.get('dataset_hash')}`",
        f"- Test file: `{dataset_meta.get('test_manifest_path')}`",
        "",
        "## Bloom rewrite (held-out test)",
        "",
        f"- N: {b.get('classification', {}).get('n')}",
        f"- Target accuracy: {b.get('classification', {}).get('accuracy')}",
        f"- Macro-F1: {b.get('classification', {}).get('macro_f1')}",
        f"- Weighted-F1: {b.get('classification', {}).get('weighted_f1')}",
        f"- Fully validated rate: {b.get('rates', {}).get('fully_validated_rewrite_rate')}",
        f"- Semantic preservation rate: {b.get('rates', {}).get('semantic_valid_rate')}",
        f"- Cognitive validity rate: {b.get('rates', {}).get('cognitive_valid_rate')}",
        f"- Trivial transform rate: {b.get('rates', {}).get('trivial_transform_rate')}",
        "",
        "## QA (SQuAD held-out test half)",
        "",
        f"- N: {q.get('n')}",
        f"- Exact Match: {q.get('exact_match')}",
        f"- Token F1: {q.get('f1')}",
        "",
        "## Summarization (PubMed test)",
        "",
        f"- N: {s.get('n')}",
        f"- ROUGE-1: {s.get('rouge1')}",
        f"- ROUGE-2: {s.get('rouge2')}",
        f"- ROUGE-L: {s.get('rougeL')}",
        "",
        "## Efficiency",
        "",
        f"- Mean latency (s): {lat.get('per_example', {}).get('mean')}",
        f"- P50: {lat.get('per_example', {}).get('p50')}",
        f"- P95: {lat.get('per_example', {}).get('p95')}",
        f"- Model load time (s): {lat.get('model_load_s')}",
        f"- RSS (MB): {res.get('rss_mb')}",
        f"- USS (MB): {res.get('uss_mb')}",
        f"- GPU memory allocated (MB): {res.get('gpu_memory_allocated_mb')}",
        "",
        "## Deployment recommendation",
        "",
        "**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be "
        "evaluated under the same protocol before model selection.",
        "",
        f"Checkpoint: `{checkpoint_info.get('resolved_path')}`",
        "",
    ]
    (out_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-task Qwen on locked TEST split (8321 examples)."
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "qwen05b_multitask.json"),
        help="Training JSON config (default: qwen05b_multitask.json)",
    )
    parser.add_argument(
        "--condition",
        choices=["base", "lora"],
        default="lora",
        help="Evaluate base HF model or trained LoRA adapter",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(MULTITASK_DATA_DIR),
        help="Multi-task dataset directory (must contain test.jsonl)",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Override LoRA adapter directory (default: best_adapter under output_dir)",
    )
    parser.add_argument(
        "--classifier-dir",
        default=None,
        help=(
            "Fixed Bloom classifier directory. Default auto-detects among "
            "models/qwen_bloom_merged0.5B, models/qwen_bloom_trained0.5B, "
            "models/qwen_bloom_federated0.5B"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Results directory (default: from config / condition)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug cap per task (0 = full test set)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION],
        default=None,
        help="Restrict to specific tasks",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only validate checkpoint and dataset assertions; no full eval",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_json(cfg_path)
    if not cfg_path.is_absolute():
        for key in ("output_dir", "results_dir", "dataset_dir"):
            if key in cfg and cfg[key]:
                p = Path(cfg[key])
                if not p.is_absolute():
                    cfg[key] = str((REPO_ROOT / p).resolve())

    gen_path = CONFIG_DIR / "generation.json"
    gen_frozen = load_json(gen_path) if gen_path.exists() else {}
    gen_cfg = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "max_new_tokens": int(cfg.get("generation", {}).get("max_new_tokens", 128)),
        "decoding": "greedy",
        "seed": SEED,
    }
    gen_cfg.update({k: v for k, v in gen_frozen.items() if k != "notes"})
    set_seed(int(gen_cfg.get("seed", SEED)))

    data_dir = Path(args.dataset_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    print("=" * 72)
    print("MULTI-TASK EVALUATION — LOCKED TEST SPLIT")
    print("=" * 72)

    try:
        test_rows, dataset_meta = load_test_split(data_dir)
    except AssertionError as exc:
        print("EVALUATION NOT STARTED —", exc)
        raise SystemExit(2) from exc

    print(json.dumps(dataset_meta, indent=2))

    try:
        ckpt = resolve_checkpoint(cfg, args.condition, adapter_override=args.adapter_path)
    except FileNotFoundError as exc:
        print("EVALUATION NOT STARTED —", exc)
        raise SystemExit(2) from exc

    ckpt_info = {
        "condition": ckpt.condition,
        "base_model_id": ckpt.base_model_id,
        "adapter_path": str(ckpt.adapter_path) if ckpt.adapter_path else None,
        "resolved_path": ckpt.resolved_path,
        "checkpoint_selection": ckpt.checkpoint_selection,
    }
    print("Checkpoint:", json.dumps(ckpt_info, indent=2))

    try:
        smoke = validate_checkpoint(
            ckpt, int(cfg.get("max_seq_length", 512)), gen_cfg
        )
    except Exception as exc:
        print("EVALUATION NOT STARTED — checkpoint validation failed:", exc)
        raise SystemExit(2) from exc
    print("Smoke test OK:", json.dumps(smoke, indent=2))

    if args.smoke_only:
        print("Smoke-only mode complete.")
        return

    out_dir = Path(args.output_dir) if args.output_dir else results_dir_for(cfg, args.condition)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    load_time = smoke["load_time_s"]
    task_filter = set(args.tasks) if args.tasks else {TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION}

    classify_fn = None
    clf_meta: dict[str, Any] = {"skipped": True}
    clf_stats = ClassifierCallStats()
    if TASK_BLOOM in task_filter:
        # Load classifier BEFORE the generative model so failures are loud and early.
        try:
            classify_fn, clf_meta = load_classifier(
                args.classifier_dir, repo_root=REPO_ROOT, require_smoke=True
            )
        except Exception as exc:
            print("EVALUATION NOT STARTED — Bloom classifier unavailable:", exc)
            raise SystemExit(2) from exc
        print(
            "Classifier OK:",
            json.dumps({k: v for k, v in clf_meta.items() if k != "smoke_traceback"}, indent=2),
        )

    generator = HFGenerator(ckpt, int(cfg.get("max_seq_length", 512)))

    bloom_rows = [r for r in test_rows if r["task"] == TASK_BLOOM and TASK_BLOOM in task_filter]
    qa_rows = [r for r in test_rows if r["task"] == TASK_QA and TASK_QA in task_filter]
    sum_rows = [
        r for r in test_rows if r["task"] == TASK_SUMMARIZATION and TASK_SUMMARIZATION in task_filter
    ]
    if args.limit:
        bloom_rows = bloom_rows[: args.limit]
        qa_rows = qa_rows[: args.limit]
        sum_rows = sum_rows[: args.limit]

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    print(f"Evaluating Bloom: {len(bloom_rows)} examples...")
    bloom_preds: list[dict[str, Any]] = []
    for i, row in enumerate(bloom_rows):
        rec = evaluate_bloom_row(row, generator, gen_cfg, classify_fn, clf_stats)
        bloom_preds.append(rec)
        predictions.append(rec)
        latencies.append(rec["latency_s"])
        if (i + 1) % 100 == 0:
            print(f"  Bloom {i + 1}/{len(bloom_rows)}")
        if clf_stats.n_fail >= 5 and clf_stats.n_ok == 0:
            print(
                "EVALUATION ABORTED — Bloom classifier failed on first 5 calls. "
                f"first_error={clf_stats.first_error}"
            )
            if clf_stats.first_traceback:
                print(clf_stats.first_traceback)
            raise SystemExit(2)

    print(f"Evaluating QA: {len(qa_rows)} examples...")
    qa_preds: list[dict[str, Any]] = []
    for i, row in enumerate(qa_rows):
        rec = evaluate_qa_row(row, generator, gen_cfg)
        qa_preds.append(rec)
        predictions.append(rec)
        latencies.append(rec["latency_s"])
        if (i + 1) % 500 == 0:
            print(f"  QA {i + 1}/{len(qa_rows)}")

    print(f"Evaluating Summarization: {len(sum_rows)} examples...")
    sum_preds: list[dict[str, Any]] = []
    for i, row in enumerate(sum_rows):
        rec = evaluate_sum_row(row, generator, gen_cfg)
        sum_preds.append(rec)
        predictions.append(rec)
        latencies.append(rec["latency_s"])
        if (i + 1) % 200 == 0:
            print(f"  Sum {i + 1}/{len(sum_rows)}")

    resources = sample_resources()

    y_true = [r["target_bloom_level"] for r in bloom_preds]
    y_pred = [r.get("classifier_prediction") or "MISSING" for r in bloom_preds]
    bloom_class = bloom_classification_metrics(y_true, y_pred)
    bloom_rate = bloom_rates(bloom_preds)
    st_matrix = source_target_matrix(bloom_preds)
    failures = failure_analysis(bloom_preds)

    qa_pairs = [(r["prediction"], r["reference"]) for r in qa_preds]
    squad_qa = aggregate_qa_metrics(qa_pairs)
    sum_pairs = [(r["prediction"], r["reference"]) for r in sum_preds]
    sum_metrics = rouge_scores(sum_pairs)
    bert = optional_bertscore(sum_pairs)

    metrics: dict[str, Any] = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "model_id": cfg["model_id"],
        "model_key": cfg.get("model_key"),
        "checkpoint": ckpt_info,
        "dataset": dataset_meta,
        "generation": gen_cfg,
        "classifier_error": None,
        "classifier_meta": clf_meta,
        "classifier_call_stats": clf_stats.as_dict(),
        "classifier_is_not_ground_truth": True,
        "bloom": {
            "classification": bloom_class,
            "rates": bloom_rate,
            "per_level": bloom_class.get("per_level"),
        },
        "qa": {
            "squad": squad_qa,
            "eduguard_rag_qa": {
                "available": False,
                "reason": "No EduGuard RAG QA benchmark file found in repository",
            },
        },
        "summarization": sum_metrics,
        "bertscore": bert,
        "latency": {
            "model_load_s": load_time,
            "per_example": latency_summary(latencies),
        },
        "resources": resources,
        "package_versions": package_versions(),
        "git_commit": git_commit(),
        "deployment_recommendation": "INCONCLUSIVE",
    }

    pred_path = out_dir / "predictions.jsonl"
    pred_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + ("\n" if predictions else ""),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps(bloom_class.get("confusion_matrix", {}), indent=2), encoding="utf-8"
    )
    (out_dir / "source_target_matrix.json").write_text(
        json.dumps(st_matrix, indent=2), encoding="utf-8"
    )
    (out_dir / "failure_analysis.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    (out_dir / "environment.json").write_text(
        json.dumps(
            {
                "package_versions": metrics["package_versions"],
                "git_commit": metrics["git_commit"],
                "resources": resources,
                "checkpoint": ckpt_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    human_export = write_human_eval_export(bloom_preds, out_dir)
    if human_export:
        metrics["human_eval_export"] = str(human_export)

    write_report(out_dir, metrics, dataset_meta, ckpt_info)
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("Results:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
