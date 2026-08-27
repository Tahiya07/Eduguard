#!/usr/bin/env python
"""Evaluate one Bloom rewrite generator on the held-out test set.

The Bloom classifier is FIXED. It is never retrained for a generator.
Do not change this evaluator after seeing results for one model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloom_target_policy import (  # noqa: E402
    BLOOM_LEVELS,
    TOPIC_OVERLAP_THRESHOLD,
    canonical_level,
    validate_rewrite,
)
from grouping import normalize_question  # noqa: E402
from paths import REWRITE_DATA_DIR  # noqa: E402
from prompt_format import IM_END, IM_START, build_generation_prompt  # noqa: E402

META_RES = [
    re.compile(pat, re.I)
    for pat in (
        r"this question asks",
        r"the student should",
        r"the rewritten question",
        r"to understand",
        r"this requires",
    )
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_generation(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace(IM_END, "").replace(IM_START, "")
    cleaned = re.sub(
        r"(?im)^(bloom level|reason|rewrite|question|answer|the rewritten question is)\s*:\s*",
        "",
        cleaned,
    )
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def classify_failure(source: str, rewrite: str, target: str, predicted: str | None, validation) -> str:
    if not rewrite:
        return "INVALID_QUESTION"
    if validation.failure_category:
        return validation.failure_category
    if predicted and canonical_level(predicted) != canonical_level(target):
        return "WRONG_TARGET_LEVEL"
    if not rewrite.strip():
        return "INVALID_QUESTION"
    return "OTHER"


def load_hf_generator(model_path: str, max_seq_length: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(path if (path / "tokenizer_config.json").is_file() else model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if (path / "adapter_config.json").is_file():
        adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        base = adapter.get("base_model_name_or_path")
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, str(path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate(prompt: str, gen_cfg: dict) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = bool(gen_cfg.get("do_sample", False))
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 96)),
                do_sample=do_sample,
                temperature=float(gen_cfg.get("temperature", 0.0)) if do_sample else None,
                top_p=float(gen_cfg.get("top_p", 1.0)) if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def load_gguf_generator(model_path: str, gen_cfg: dict):
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=0,
        use_mmap=True,
        verbose=False,
    )

    def generate(prompt: str, cfg: dict) -> str:
        out = llm(
            prompt,
            max_tokens=int(cfg.get("max_new_tokens", 96)),
            temperature=0.0,
        )
        return out["choices"][0]["text"]

    return generate


def load_fixed_classifier(classifier_dir: str | None):
    if not classifier_dir:
        try:
            from predict_bloom import QwenBloomPredictor

            predictor = QwenBloomPredictor()
        except Exception as exc:  # noqa: BLE001
            return None, f"classifier_unavailable: {exc}"
    else:
        try:
            from predict_bloom import QwenBloomPredictor

            predictor = QwenBloomPredictor(model_dir=classifier_dir)
        except Exception as exc:  # noqa: BLE001
            return None, f"classifier_unavailable: {exc}"

    def predict(text: str) -> dict:
        result = predictor.predict(text)
        return {
            "predicted_level": canonical_level(result.get("prediction") or result.get("label") or ""),
            "confidence": float(result.get("confidence") or 0.0),
        }

    return predict, None


def f1_scores(y_true: list[str], y_pred: list[str]) -> dict:
    labels = BLOOM_LEVELS
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for gold, pred in zip(y_true, y_pred):
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    per = {}
    f1s = []
    supports = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        support = sum(1 for item in y_true if item == label)
        per[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        supports.append(support)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    total = sum(supports) or 1
    weighted = sum(f * s for f, s in zip(f1s, supports)) / total
    return {"per_level": per, "macro_f1": macro, "weighted_f1": weighted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, help="0.5b or 1.5b label for output folder")
    parser.add_argument("--generator-path", required=True, help="HF adapter, merged model, or GGUF path")
    parser.add_argument("--dataset-dir", default=str(REWRITE_DATA_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--classifier-dir", default=None, help="Fixed Bloom classifier directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--backend", default="hf", choices=["hf", "base", "gguf"], help="hf LoRA/merged, base HF, or GGUF")
    parser.add_argument("--role", default="lora", help="base|lora|gguf|production_gguf label for comparison")
    parser.add_argument("--limit", type=int, default=0, help="Optional debug cap; 0 = full split")
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    rows = read_jsonl(data_dir / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / "results" / args.model_key / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "temperature": 0.0,
        "top_p": 1.0,
        "decoding": "greedy" if not args.do_sample else "sampling",
    }
    generator_path = Path(args.generator_path)
    if args.backend == "gguf" or generator_path.suffix.lower() == ".gguf":
        generate_fn = load_gguf_generator(str(generator_path), gen_cfg)
    else:
        generate_fn = load_hf_generator(str(generator_path), args.max_seq_length)
    classify_fn, clf_error = load_fixed_classifier(args.classifier_dir)

    records = []
    y_true = []
    y_pred = []
    correct = []
    secondary = Counter()
    matrix = defaultdict(Counter)
    per_target = Counter()
    per_target_correct = Counter()
    start = time.time()
    for row in rows:
        prompt = build_generation_prompt(
            row["source_question"],
            row["target_bloom_level"],
        )
        raw = generate_fn(prompt, gen_cfg)
        rewrite = clean_generation(raw)
        validation = validate_rewrite(row["source_question"], rewrite, row["target_bloom_level"])
        predicted = None
        confidence = None
        if classify_fn is not None and rewrite:
            try:
                clf = classify_fn(rewrite)
                predicted = clf["predicted_level"]
                confidence = clf["confidence"]
            except Exception as exc:  # noqa: BLE001
                clf_error = str(exc)
        target = row["target_bloom_level"]
        is_correct = bool(predicted == target) if predicted else False
        failure = "" if is_correct and validation.ok else classify_failure(
            row["source_question"], rewrite, target, predicted, validation
        )
        if not rewrite:
            secondary["empty_invalid"] += 1
        if validation.meta:
            secondary["meta_language"] += 1
        if validation.trivial:
            secondary["trivial"] += 1
        if validation.invalid_question:
            secondary["invalid_question"] += 1
        if validation.forbidden:
            secondary["forbidden"] += 1
        if validation.topic_overlap >= TOPIC_OVERLAP_THRESHOLD:
            secondary["topic_preserved"] += 1
        if validation.ok:
            secondary["question_valid"] += 1
        if rewrite and normalize_question(rewrite) == normalize_question(row["source_question"]):
            secondary["exact_source_duplicate"] += 1
        secondary["output_words"] += len(rewrite.split()) if rewrite else 0
        secondary["n"] += 1
        per_target[target] += 1
        if is_correct:
            per_target_correct[target] += 1
        matrix[row["source_bloom_level"]][target] += int(is_correct)
        if "Original Bloom level:" in prompt or "Source Bloom level:" in prompt:
            raise RuntimeError("source Bloom level leaked into evaluation prompt")
        records.append(
            {
                "example_id": row["example_id"],
                "source_id": row["source_id"],
                "source_question": row["source_question"],
                "source_level": row["source_bloom_level"],  # metadata only
                "target_level": target,
                "generated_rewrite": rewrite,
                "predicted_level": predicted,
                "classifier_confidence": confidence,
                "target_match": is_correct,
                "validation": validation.__dict__,
                "failure_category": failure,
                "generator_inputs": ["source_question", "target_bloom_level"],
            }
        )
        y_true.append(target)
        y_pred.append(predicted or "MISSING")
        correct.append(is_correct)

    n = len(rows) or 1
    f1 = f1_scores(y_true, y_pred)
    confusion = {src: {tgt: 0 for tgt in BLOOM_LEVELS + ["MISSING"]} for src in BLOOM_LEVELS}
    cell_n = defaultdict(int)
    cell_correct = defaultdict(int)
    for item in records:
        src = item["source_level"]
        tgt = item["target_level"]
        pred = item["predicted_level"] or "MISSING"
        if src in confusion and pred in confusion[src]:
            confusion[src][pred] += 1
        cell_n[(src, tgt)] += 1
        cell_correct[(src, tgt)] += int(bool(item["target_match"]))
    source_target_matrix = {
        f"{src}->{tgt}": {
            "n": cell_n[(src, tgt)],
            "classifier_agreement": (cell_correct[(src, tgt)] / cell_n[(src, tgt)]) if cell_n[(src, tgt)] else None,
        }
        for src in BLOOM_LEVELS
        for tgt in BLOOM_LEVELS
        if src != tgt
    }
    lengths = [len((item["generated_rewrite"] or "").split()) for item in records]
    summary = {
        "model_key": args.model_key,
        "role": args.role,
        "backend": args.backend,
        "generator_path": str(generator_path),
        "split": args.split,
        "n": len(rows),
        "elapsed_s": round(time.time() - start, 3),
        "generation": gen_cfg,
        "classifier_error": clf_error,
        "classifier_is_not_ground_truth": True,
        "classifier_based_target_agreement": {
            "accuracy": sum(correct) / n,
            "macro_f1": f1["macro_f1"],
            "weighted_f1": f1["weighted_f1"],
            "per_level": f1["per_level"],
            "mean_confidence": (
                sum(item["classifier_confidence"] or 0.0 for item in records) / n
            ),
        },
        "overall_target_accuracy": sum(correct) / n,
        "macro_accuracy": sum(
            (per_target_correct[level] / per_target[level]) if per_target[level] else 0.0
            for level in BLOOM_LEVELS
        ) / len(BLOOM_LEVELS),
        "per_target_accuracy": {
            level: (per_target_correct[level] / per_target[level]) if per_target[level] else None
            for level in BLOOM_LEVELS
        },
        "per_target_counts": {
            level: {"correct": int(per_target_correct[level]), "n": int(per_target[level])}
            for level in BLOOM_LEVELS
        },
        "macro_f1": f1["macro_f1"],
        "weighted_f1": f1["weighted_f1"],
        "f1_per_level": f1["per_level"],
        "confusion_matrix_true_target_vs_classifier_pred": confusion,
        "source_target_matrix": source_target_matrix,
        "topic_preservation_threshold": TOPIC_OVERLAP_THRESHOLD,
        "secondary": {
            "question_validity_rate": secondary["question_valid"] / n,
            "format_validity_rate": secondary["question_valid"] / n,
            "topic_preservation_rate": secondary["topic_preserved"] / n,
            "cognitive_task_validity_rate": secondary["question_valid"] / n,
            "trivial_transformation_rate": secondary["trivial"] / n,
            "meta_language_rate": secondary["meta_language"] / n,
            "empty_invalid_rate": secondary["empty_invalid"] / n,
            "invalid_question_rate": secondary["invalid_question"] / n,
            "forbidden_operation_rate": secondary["forbidden"] / n,
            "exact_source_duplicate_rate": secondary["exact_source_duplicate"] / n,
            "average_output_length_words": (sum(lengths) / len(lengths)) if lengths else 0.0,
        },
        "failure_counts": dict(Counter(item["failure_category"] for item in records if item["failure_category"])),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if clf_error:
        print("WARNING: fixed classifier was unavailable. Target accuracy is incomplete.")
        print(clf_error)


if __name__ == "__main__":
    main()
