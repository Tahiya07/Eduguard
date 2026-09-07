#!/usr/bin/env python
"""
Evaluate a GGUF multitask Qwen model on the locked held-out TEST split.

This is a separate evaluator from evaluate_rewrite.py so that the HF
evaluation pipeline remains untouched.

Tasks:
  - Bloom rewrite
  - SQuAD-style QA
  - PubMed summarization

The same dataset, prompts, validation functions, classifier, and metric
functions used by the HF evaluator are reused wherever possible.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
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

from bloom_validation import (  # noqa: E402
    is_valid_imperative_exam,
    is_valid_interrogative,
    validate_bloom_example,
)

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

from paths import (  # noqa: E402
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


def clean_generation(text: str) -> str:
    cleaned = (text or "").strip()

    cleaned = cleaned.replace(IM_END, "").replace(IM_START, "")

    cleaned = re.sub(
        r"(?im)^(bloom level|reason|rewrite|question|answer|the rewritten question is)\s*:\s*",
        "",
        cleaned,
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned.strip().strip('"').strip("'"),
    )


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


def process_memory_mb() -> dict[str, float | None]:
    try:
        import psutil

        p = psutil.Process()
        mi = p.memory_info()

        result: dict[str, float | None] = {
            "rss_mb": round(mi.rss / 1024**2, 2),
            "uss_mb": None,
        }

        try:
            full = p.memory_full_info()
            if hasattr(full, "uss"):
                result["uss_mb"] = round(full.uss / 1024**2, 2)
        except Exception:
            pass

        return result

    except Exception:
        return {
            "rss_mb": None,
            "uss_mb": None,
        }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}

    for name in (
        "llama_cpp",
        "torch",
        "transformers",
        "peft",
        "numpy",
        "rouge_score",
        "bert_score",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "ok")
        except ImportError:
            versions[name] = None

    return versions


def git_commit() -> str | None:
    try:
        import subprocess

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


def evaluate_bloom_row(
    row: dict[str, Any],
    llm,
    max_tokens: int,
    classifier_fn,
    classifier_stats: ClassifierCallStats,
) -> dict[str, Any]:

    prompt = build_generation_prompt(TASK_BLOOM, row)

    assert_no_source_level_in_prompt(prompt)

    for marker in FORBIDDEN_SOURCE_LEVEL_MARKERS:
        if marker.lower() in prompt.lower():
            raise AssertionError(
                f"Forbidden source-level marker found in prompt: {marker}"
            )

    t0 = time.perf_counter()

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0,
    )

    latency = time.perf_counter() - t0

    raw = result["choices"][0]["text"]
    prediction = clean_generation(raw)

    validation = validate_bloom_example(
        row["source_question"],
        row["target_bloom_level"],
        prediction,
    )

    meta_response = (
        "meta_response" in (validation.rejection_reason or "")
    )

    answer_output = (
        "answer_or_declarative" in (validation.rejection_reason or "")
    )

    empty_output = not prediction

    interrogative_valid = is_valid_interrogative(prediction)
    imperative_valid = is_valid_imperative_exam(prediction)

    question_form_valid = (
        interrogative_valid or imperative_valid
    )

    clf_pred = None
    clf_conf = 0.0

    if classifier_fn and prediction:
        try:
            clf = classifier_fn(prediction)

            clf_pred = clf["predicted_level"]
            clf_conf = clf["confidence"]

            classifier_stats.record_ok()

        except Exception as exc:
            classifier_stats.record_fail(exc)

            if classifier_stats.n_fail == 1:
                print(
                    "WARNING: Bloom classifier call failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    target = canonical_bloom(
        row["target_bloom_level"]
    )

    target_match = bool(
        clf_pred
        and canonical_bloom(clf_pred) == target
    )

    fully_validated = (
        validation.format_valid
        and validation.semantic_valid
        and validation.cognitive_valid
        and not validation.trivial_transform
        and target_match
        and clf_conf >= CLASSIFIER_CONFIDENCE_MIN
    )

    record = {
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
        "interrogative_valid": interrogative_valid,
        "imperative_valid": imperative_valid,
        "question_form_valid": question_form_valid,
        "classifier_prediction": clf_pred,
        "classifier_confidence": clf_conf,
        "target_match": target_match,
        "fully_validated": fully_validated,
        "latency_s": round(latency, 6),
    }

    record["failure_category"] = (
        classify_failure(record)
        if not fully_validated
        else ""
    )

    return record


def evaluate_qa_row(
    row: dict[str, Any],
    llm,
    max_tokens: int,
) -> dict[str, Any]:

    prompt = build_generation_prompt(TASK_QA, row)

    t0 = time.perf_counter()

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0,
    )

    latency = time.perf_counter() - t0

    raw = result["choices"][0]["text"]
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
    row: dict[str, Any],
    llm,
    max_tokens: int,
) -> dict[str, Any]:

    prompt = build_generation_prompt(
        TASK_SUMMARIZATION,
        row,
    )

    t0 = time.perf_counter()

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0,
    )

    latency = time.perf_counter() - t0

    raw = result["choices"][0]["text"]
    prediction = clean_generation(raw)

    return {
        "id": row.get("example_id"),
        "task": TASK_SUMMARIZATION,
        "article": row.get("article"),
        "reference": row.get("abstract"),
        "prediction": prediction,
        "latency_s": round(latency, 6),
    }


def write_report(
    out_dir: Path,
    metrics: dict[str, Any],
) -> None:

    bloom = metrics.get("bloom", {})
    classification = bloom.get("classification", {})
    rates = bloom.get("rates", {})

    qa = metrics.get("qa", {}).get("squad", {})
    summ = metrics.get("summarization", {})

    latency = metrics.get("latency", {})
    resources = metrics.get("resources", {})

    lines = [
        "# GGUF Multi-task Evaluation Report",
        "",
        f"Generated (UTC): {metrics.get('evaluated_utc')}",
        f"GGUF: `{metrics.get('gguf')}`",
        "",
        "## Dataset",
        "",
        f"- Test count: **{metrics['dataset']['counts']['test_count']}**",
        f"- Corpus hash: `{metrics['dataset'].get('dataset_hash')}`",
        f"- Test file: `{metrics['dataset'].get('test_manifest_path')}`",
        "",
        "## Bloom rewrite",
        "",
        f"- N: {classification.get('n')}",
        f"- Target accuracy: {classification.get('accuracy')}",
        f"- Macro-F1: {classification.get('macro_f1')}",
        f"- Weighted-F1: {classification.get('weighted_f1')}",
        f"- Fully validated rate: {rates.get('fully_validated_rewrite_rate')}",
        f"- Semantic preservation rate: {rates.get('semantic_valid_rate')}",
        f"- Cognitive validity rate: {rates.get('cognitive_valid_rate')}",
        f"- Trivial transform rate: {rates.get('trivial_transform_rate')}",
        "",
        "## QA",
        "",
        f"- N: {qa.get('n')}",
        f"- Exact Match: {qa.get('exact_match')}",
        f"- Token F1: {qa.get('f1')}",
        "",
        "## Summarization",
        "",
        f"- N: {summ.get('n')}",
        f"- ROUGE-1: {summ.get('rouge1')}",
        f"- ROUGE-2: {summ.get('rouge2')}",
        f"- ROUGE-L: {summ.get('rougeL')}",
        "",
        "## Efficiency",
        "",
        f"- Model load time (s): {latency.get('model_load_s')}",
        f"- Mean latency (s): {latency.get('per_example', {}).get('mean')}",
        f"- P50 latency (s): {latency.get('per_example', {}).get('p50')}",
        f"- P95 latency (s): {latency.get('per_example', {}).get('p95')}",
        f"- RSS (MB): {resources.get('rss_mb')}",
        f"- USS (MB): {resources.get('uss_mb')}",
        "",
        "## Generation",
        "",
        f"- Temperature: {metrics['generation']['temperature']}",
        f"- Top-p: {metrics['generation']['top_p']}",
        f"- Top-k: {metrics['generation']['top_k']}",
        f"- Repeat penalty: {metrics['generation']['repeat_penalty']}",
        f"- Max tokens: {metrics['generation']['max_tokens']}",
        f"- Threads: {metrics['generation']['threads']}",
        f"- Context: {metrics['generation']['context']}",
        f"- GPU layers: {metrics['generation']['gpu_layers']}",
        "",
    ]

    (out_dir / "evaluation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GGUF multi-task Qwen on locked "
            "TEST split."
        )
    )

    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to GGUF model",
    )

    parser.add_argument(
        "--dataset-dir",
        default=str(MULTITASK_DATA_DIR),
        help="Directory containing test.jsonl",
    )

    parser.add_argument(
        "--classifier-dir",
        default=None,
        help="Fixed Bloom classifier directory",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug limit per task; 0 = full test set",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            TASK_BLOOM,
            TASK_QA,
            TASK_SUMMARIZATION,
        ],
        default=None,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads",
    )

    parser.add_argument(
        "--ctx",
        type=int,
        default=512,
        help="Context size",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens",
    )

    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help=(
            "Number of GGUF layers offloaded to GPU. "
            "0 = CPU-only."
        ),
    )

    args = parser.parse_args()

    gguf = Path(args.gguf)

    if not gguf.is_absolute():
        gguf = REPO_ROOT / gguf

    gguf = gguf.resolve()

    if not gguf.exists():
        print(f"STOP: GGUF missing: {gguf}")
        raise SystemExit(2)

    data_dir = Path(args.dataset_dir)

    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    data_dir = data_dir.resolve()

    if not (data_dir / "test.jsonl").exists():
        print(
            "STOP: test.jsonl missing: "
            f"{data_dir / 'test.jsonl'}"
        )
        raise SystemExit(2)

    print("=" * 72)
    print("GGUF MULTI-TASK EVALUATION — LOCKED TEST SPLIT")
    print("=" * 72)

    print(f"GGUF: {gguf}")
    print(
        f"GGUF size: "
        f"{gguf.stat().st_size / 1024**2:.2f} MiB"
    )

    print(f"Dataset: {data_dir}")

    try:
        test_rows, dataset_meta = load_test_split(data_dir)

    except AssertionError as exc:
        print(
            "EVALUATION NOT STARTED — dataset validation failed:",
            exc,
        )
        raise SystemExit(2) from exc

    print(
        json.dumps(
            dataset_meta,
            indent=2,
        )
    )

    task_filter = (
        set(args.tasks)
        if args.tasks
        else {
            TASK_BLOOM,
            TASK_QA,
            TASK_SUMMARIZATION,
        }
    )

    bloom_rows = [
        r
        for r in test_rows
        if r["task"] == TASK_BLOOM
        and TASK_BLOOM in task_filter
    ]

    qa_rows = [
        r
        for r in test_rows
        if r["task"] == TASK_QA
        and TASK_QA in task_filter
    ]

    sum_rows = [
        r
        for r in test_rows
        if r["task"] == TASK_SUMMARIZATION
        and TASK_SUMMARIZATION in task_filter
    ]

    if args.limit:
        bloom_rows = bloom_rows[:args.limit]
        qa_rows = qa_rows[:args.limit]
        sum_rows = sum_rows[:args.limit]

    print()
    print("Examples:")
    print(f"  Bloom: {len(bloom_rows)}")
    print(f"  QA: {len(qa_rows)}")
    print(f"  Summarization: {len(sum_rows)}")

    classifier_fn = None
    classifier_meta: dict[str, Any] = {
        "skipped": True
    }

    classifier_stats = ClassifierCallStats()

    if TASK_BLOOM in task_filter:

        try:
            classifier_fn, classifier_meta = load_classifier(
                args.classifier_dir,
                repo_root=REPO_ROOT,
                require_smoke=True,
            )

        except Exception as exc:
            print(
                "EVALUATION NOT STARTED — "
                "Bloom classifier unavailable:",
                exc,
            )
            raise SystemExit(2) from exc

        print(
            "Classifier OK:",
            json.dumps(
                {
                    k: v
                    for k, v in classifier_meta.items()
                    if k != "smoke_traceback"
                },
                indent=2,
            ),
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        print(
            "STOP: llama-cpp-python is not installed:",
            exc,
        )
        raise SystemExit(2) from exc

    print()
    print("Loading GGUF...")

    load_start = time.perf_counter()

    llm = Llama(
        model_path=str(gguf),
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )

    load_time = time.perf_counter() - load_start

    resources_after_load = process_memory_mb()

    print(
        f"Model loaded in {load_time:.4f} seconds"
    )

    print(
        "Memory after load:",
        json.dumps(
            resources_after_load,
            indent=2,
        ),
    )

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    bloom_preds: list[dict[str, Any]] = []
    qa_preds: list[dict[str, Any]] = []
    sum_preds: list[dict[str, Any]] = []

    if TASK_BLOOM in task_filter:

        print(
            f"\nEvaluating Bloom: "
            f"{len(bloom_rows)} examples..."
        )

        for i, row in enumerate(bloom_rows):

            rec = evaluate_bloom_row(
                row,
                llm,
                args.max_tokens,
                classifier_fn,
                classifier_stats,
            )

            bloom_preds.append(rec)
            predictions.append(rec)
            latencies.append(rec["latency_s"])

            if (i + 1) % 100 == 0:
                print(
                    f"  Bloom "
                    f"{i + 1}/{len(bloom_rows)}"
                )

            if (
                classifier_stats.n_fail >= 5
                and classifier_stats.n_ok == 0
            ):
                print(
                    "EVALUATION ABORTED — "
                    "Bloom classifier failed on "
                    "first 5 calls."
                )

                if classifier_stats.first_traceback:
                    print(
                        classifier_stats.first_traceback
                    )

                raise SystemExit(2)

    if TASK_QA in task_filter:

        print(
            f"\nEvaluating QA: "
            f"{len(qa_rows)} examples..."
        )

        for i, row in enumerate(qa_rows):

            rec = evaluate_qa_row(
                row,
                llm,
                args.max_tokens,
            )

            qa_preds.append(rec)
            predictions.append(rec)
            latencies.append(rec["latency_s"])

            if (i + 1) % 500 == 0:
                print(
                    f"  QA "
                    f"{i + 1}/{len(qa_rows)}"
                )

    if TASK_SUMMARIZATION in task_filter:

        print(
            f"\nEvaluating Summarization: "
            f"{len(sum_rows)} examples..."
        )

        for i, row in enumerate(sum_rows):

            rec = evaluate_sum_row(
                row,
                llm,
                args.max_tokens,
            )

            sum_preds.append(rec)
            predictions.append(rec)
            latencies.append(rec["latency_s"])

            if (i + 1) % 200 == 0:
                print(
                    f"  Sum "
                    f"{i + 1}/{len(sum_rows)}"
                )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    bloom_class: dict[str, Any] = {}
    bloom_rate: dict[str, Any] = {}
    source_matrix: dict[str, Any] = {}
    failures: dict[str, Any] = {}

    if bloom_preds:

        y_true = [
            r["target_bloom_level"]
            for r in bloom_preds
        ]

        y_pred = [
            r.get("classifier_prediction")
            or "MISSING"
            for r in bloom_preds
        ]

        bloom_class = bloom_classification_metrics(
            y_true,
            y_pred,
        )

        bloom_rate = bloom_rates(
            bloom_preds
        )

        source_matrix = source_target_matrix(
            bloom_preds
        )

        failures = failure_analysis(
            bloom_preds
        )

    qa_metrics: dict[str, Any] = {}

    if qa_preds:

        qa_pairs = [
            (
                r["prediction"],
                r["reference"],
            )
            for r in qa_preds
        ]

        qa_metrics = aggregate_qa_metrics(
            qa_pairs
        )

    sum_metrics: dict[str, Any] = {}

    bert: dict[str, Any] | None = None

    if sum_preds:

        sum_pairs = [
            (
                r["prediction"],
                r["reference"],
            )
            for r in sum_preds
        ]

        sum_metrics = rouge_scores(
            sum_pairs
        )

        bert = optional_bertscore(
            sum_pairs
        )

    metrics: dict[str, Any] = {

        "evaluated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "condition":
            "gguf",

        "gguf":
            str(gguf),

        "gguf_size_bytes":
            gguf.stat().st_size,

        "dataset":
            dataset_meta,

        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "max_tokens": args.max_tokens,
            "decoding": "greedy",
            "threads": args.threads,
            "context": args.ctx,
            "gpu_layers": args.gpu_layers,
        },

        "classifier_error":
            None,

        "classifier_meta":
            classifier_meta,

        "classifier_call_stats":
            classifier_stats.as_dict(),

        "classifier_is_not_ground_truth":
            True,

        "bloom": {
            "classification":
                bloom_class,

            "rates":
                bloom_rate,

            "per_level":
                bloom_class.get(
                    "per_level"
                ),
        },

        "qa": {
            "squad":
                qa_metrics,

            "eduguard_rag_qa": {
                "available": False,
                "reason":
                    "No EduGuard RAG QA benchmark "
                    "file found in repository",
            },
        },

        "summarization":
            sum_metrics,

        "bertscore":
            bert,

        "latency": {
            "model_load_s":
                round(load_time, 6),

            "per_example":
                latency_summary(
                    latencies
                ),
        },

        "resources":
            resources_after_load,

        "package_versions":
            package_versions(),

        "git_commit":
            git_commit(),

        "deployment_recommendation":
            "INCONCLUSIVE",
    }

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    if args.output_dir:

        out_dir = Path(
            args.output_dir
        )

    else:

        out_dir = (
            EXPERIMENT_DIR
            / "results"
            / "gguf"
            / gguf.stem
        )

    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_path = (
        out_dir / "predictions.jsonl"
    )

    pred_path.write_text(
        "\n".join(
            json.dumps(
                p,
                ensure_ascii=False,
            )
            for p in predictions
        )
        + (
            "\n"
            if predictions
            else ""
        ),
        encoding="utf-8",
    )

    (
        out_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir / "confusion_matrix.json"
    ).write_text(
        json.dumps(
            bloom_class.get(
                "confusion_matrix",
                {},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir / "source_target_matrix.json"
    ).write_text(
        json.dumps(
            source_matrix,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir / "failure_analysis.json"
    ).write_text(
        json.dumps(
            failures,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir / "environment.json"
    ).write_text(
        json.dumps(
            {
                "package_versions":
                    metrics[
                        "package_versions"
                    ],

                "git_commit":
                    metrics[
                        "git_commit"
                    ],

                "resources":
                    resources_after_load,

                "gguf":
                    str(gguf),

                "gguf_size_bytes":
                    gguf.stat().st_size,

                "generation":
                    metrics[
                        "generation"
                    ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        out_dir,
        metrics,
    )

    print()
    print("=" * 72)
    print("GGUF EVALUATION COMPLETE")
    print("=" * 72)

    print(
        "Results:",
        out_dir,
    )

    print()
    print(
        json.dumps(
            {
                "bloom_accuracy":
                    bloom_class.get(
                        "accuracy"
                    ),

                "bloom_macro_f1":
                    bloom_class.get(
                        "macro_f1"
                    ),

                "bloom_weighted_f1":
                    bloom_class.get(
                        "weighted_f1"
                    ),

                "fully_validated":
                    bloom_rate.get(
                        "fully_validated_rewrite_rate"
                    ),

                "qa_exact_match":
                    qa_metrics.get(
                        "exact_match"
                    ),

                "qa_token_f1":
                    qa_metrics.get(
                        "f1"
                    ),

                "rouge1":
                    sum_metrics.get(
                        "rouge1"
                    ),

                "rouge2":
                    sum_metrics.get(
                        "rouge2"
                    ),

                "rougeL":
                    sum_metrics.get(
                        "rougeL"
                    ),

                "mean_latency":
                    latency_summary(
                        latencies
                    ).get("mean"),

                "p50_latency":
                    latency_summary(
                        latencies
                    ).get("p50"),

                "p95_latency":
                    latency_summary(
                        latencies
                    ).get("p95"),

                "rss_mb":
                    resources_after_load.get(
                        "rss_mb"
                    ),

                "uss_mb":
                    resources_after_load.get(
                        "uss_mb"
                    ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()