#!/usr/bin/env python
"""Re-score Bloom classifier labels on an existing predictions.jsonl.

Does NOT regenerate model outputs. Use after fixing classifier wiring.

Example:
  python experiments/multitask_bloom_rewrite/scripts/rescore_bloom_classifier.py `
    --predictions experiments/multitask_bloom_rewrite/results/qwen05b_lora/predictions.jsonl `
    --classifier-dir models/qwen_bloom_merged0.5B
"""
from __future__ import annotations

import argparse
import json
import sys
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
from eval_metrics import (  # noqa: E402
    CLASSIFIER_CONFIDENCE_MIN,
    bloom_classification_metrics,
    bloom_rates,
    canonical_bloom,
    failure_analysis,
    source_target_matrix,
)
from paths import TASK_BLOOM  # noqa: E402


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
    if not record.get("format_valid") or not record.get("cognitive_valid"):
        return "INVALID_QUESTION"
    if record.get("target_match") is False:
        return "WRONG_TARGET_LEVEL"
    if record.get("classifier_confidence", 0.0) < CLASSIFIER_CONFIDENCE_MIN:
        return "LOW_CONFIDENCE"
    return "OTHER"


def rescore_record(rec: dict[str, Any], classify_fn, stats: ClassifierCallStats) -> dict[str, Any]:
    if rec.get("task") != TASK_BLOOM:
        return rec
    prediction = rec.get("prediction") or ""
    clf_pred = None
    clf_conf = 0.0
    if prediction:
        try:
            clf = classify_fn(prediction)
            clf_pred = clf["predicted_level"]
            clf_conf = clf["confidence"]
            stats.record_ok()
        except Exception as exc:  # noqa: BLE001
            stats.record_fail(exc)
            if stats.n_fail == 1:
                print(f"WARNING: classifier failed: {type(exc).__name__}: {exc}")

    target = canonical_bloom(rec.get("target_bloom_level"))
    target_match = bool(clf_pred and canonical_bloom(clf_pred) == target)

    # Recompute validation fields if missing (older files should already have them).
    if "format_valid" not in rec and prediction and rec.get("source_question"):
        validation = validate_bloom_example(
            rec["source_question"], rec["target_bloom_level"], prediction
        )
        rec["format_valid"] = validation.format_valid
        rec["semantic_valid"] = validation.semantic_valid
        rec["cognitive_valid"] = validation.cognitive_valid
        rec["trivial_transform"] = validation.trivial_transform

    fully_validated = (
        bool(rec.get("format_valid"))
        and bool(rec.get("semantic_valid"))
        and bool(rec.get("cognitive_valid"))
        and not bool(rec.get("trivial_transform"))
        and target_match
        and clf_conf >= CLASSIFIER_CONFIDENCE_MIN
    )
    out = dict(rec)
    out["classifier_prediction"] = clf_pred
    out["classifier_confidence"] = clf_conf
    out["target_match"] = target_match
    out["fully_validated"] = fully_validated
    out["failure_category"] = classify_failure(out) if not fully_validated else ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score Bloom classifier fields on existing predictions.jsonl"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl from evaluate_rewrite.py",
    )
    parser.add_argument(
        "--classifier-dir",
        default=None,
        help="Bloom classifier dir (default: auto-detect merged/trained/federated 0.5B)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write rescored outputs (default: same dir as predictions)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = REPO_ROOT / pred_path
    if not pred_path.is_file():
        print("REScore NOT STARTED — missing predictions:", pred_path)
        raise SystemExit(2)

    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    classify_fn, clf_meta = load_classifier(
        args.classifier_dir, repo_root=REPO_ROOT, require_smoke=True
    )
    print("Classifier OK:", json.dumps({k: v for k, v in clf_meta.items() if k != "smoke_traceback"}, indent=2))

    rows = [
        json.loads(line)
        for line in pred_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = ClassifierCallStats()
    rescored: list[dict[str, Any]] = []
    bloom_preds: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        new_row = rescore_record(row, classify_fn, stats)
        rescored.append(new_row)
        if new_row.get("task") == TASK_BLOOM:
            bloom_preds.append(new_row)
        if (i + 1) % 200 == 0:
            print(f"  Rescored {i + 1}/{len(rows)}")

    if stats.n_ok == 0 and bloom_preds:
        print("REScore FAILED — classifier produced 0 successful calls")
        print(stats.as_dict())
        raise SystemExit(2)

    y_true = [r["target_bloom_level"] for r in bloom_preds]
    y_pred = [r.get("classifier_prediction") or "MISSING" for r in bloom_preds]
    bloom_class = bloom_classification_metrics(y_true, y_pred)
    bloom_rate = bloom_rates(bloom_preds)
    st_matrix = source_target_matrix(bloom_preds)
    failures = failure_analysis(bloom_preds)

    metrics = {
        "rescored_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions": str(pred_path),
        "classifier_meta": {k: v for k, v in clf_meta.items() if k != "smoke_traceback"},
        "classifier_call_stats": stats.as_dict(),
        "bloom": {
            "classification": bloom_class,
            "rates": bloom_rate,
            "per_level": bloom_class.get("per_level"),
        },
        "note": "QA/summarization unchanged; only Bloom classifier fields were rescored.",
        "deployment_recommendation": "INCONCLUSIVE",
    }

    out_pred = out_dir / "predictions_rescored.jsonl"
    out_pred.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rescored) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics_bloom_rescored.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "confusion_matrix_rescored.json").write_text(
        json.dumps(bloom_class.get("confusion_matrix", {}), indent=2), encoding="utf-8"
    )
    (out_dir / "source_target_matrix_rescored.json").write_text(
        json.dumps(st_matrix, indent=2), encoding="utf-8"
    )
    (out_dir / "failure_analysis_rescored.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("BLOOM CLASSIFIER RESCORE COMPLETE")
    print(json.dumps(metrics["bloom"], indent=2))
    print("Wrote:", out_pred)


if __name__ == "__main__":
    main()
