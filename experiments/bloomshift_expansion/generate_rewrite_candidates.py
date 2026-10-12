#!/usr/bin/env python3
"""Generate raw, unvalidated BloomShift candidates from a JSONL queue using a local HF model.

This script never certifies meaning preservation or Bloom alignment. It retains the raw
model output, parsed output, generation metadata, and human-review status. Use --resume
for long runs so already-written generation IDs are not overwritten or regenerated.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_existing_ids(path):
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("generation_id"):
                    ids.add(row["generation_id"])
            except json.JSONDecodeError:
                continue
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="JSONL from build_candidate_generation_queue.py or build_generation_queue.py")
    ap.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    ap.add_argument("--output", required=True, help="Raw candidate JSONL output")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="Optional generation limit for smoke tests; 0 means all remaining")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Append and skip generation IDs already in output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        ap.error("--temperature cannot be negative")
    torch.manual_seed(args.seed)

    jobs = []
    with open(args.queue, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in queue line {line_no}: {exc}")
            if job.get("status") != "pending_generation":
                continue
            jobs.append(job)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    already = read_existing_ids(out) if args.resume else set()
    jobs = [j for j in jobs if j.get("generation_id") not in already]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print(json.dumps({"jobs_generated": 0, "output": str(out), "status": "no pending jobs (or all IDs already present)"}, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.time()
    mode = "a" if args.resume else "w"
    with out.open(mode, encoding="utf-8") as f:
        for idx, job in enumerate(jobs, 1):
            user_prompt = (
                f'{job.get("prompt", "")}\n\n'
                f'Source question: {job.get("source_question", "")}\n'
                f'Source Bloom level: {job.get("source_bloom_level", "unknown")}\n'
                f'Target Bloom level: {job.get("target_bloom_level", "")}\n'
                f'Preferred opening verbs (ordered for diversity; choose only if semantically appropriate): {job.get("preferred_opening_verbs", "")}\n'
                "Return one JSON object only. Do not answer the question."
            )
            messages = [
                {"role": "system", "content": "You edit educational assessment questions. Preserve the source meaning and all supplied context; never invent missing facts. Choose a verb that fits the cognitive operation, not just the label."},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
            else:
                rendered = messages[0]["content"] + "\n\n" + messages[1]["content"]
                input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(model.device)
            generation_args = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_args.update({"temperature": args.temperature, "top_p": args.top_p})
            with torch.inference_mode():
                output_ids = model.generate(input_ids, **generation_args)
            raw = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            parsed = None
            try:
                start, end = raw.find("{"), raw.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
            record = {
                **job,
                "status": "raw_candidate_generated",
                "model_id": args.model,
                "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "generation_parameters": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "do_sample": args.temperature > 0,
                    "seed": args.seed,
                },
                "raw_model_output": raw,
                "parsed_candidate": parsed,
                "generation_parse_status": "parsed_json" if parsed is not None else "raw_only_parse_failed",
                "human_review_status": "not_reviewed",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{idx}/{len(jobs)}] {job.get('generation_id', 'unknown')} -> {record['generation_parse_status']}")
    print(json.dumps({
        "queue": args.queue,
        "model": args.model,
        "jobs_generated": len(jobs),
        "output": str(out),
        "elapsed_seconds": round(time.time() - started, 2),
        "status": "raw candidates only; human validation still required",
    }, indent=2))


if __name__ == "__main__":
    main()
