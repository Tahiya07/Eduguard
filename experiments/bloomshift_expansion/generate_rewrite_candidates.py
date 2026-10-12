#!/usr/bin/env python3
"""Generate raw candidate rewrites from a reviewed JSONL prompt queue using a local HF model.

Outputs are unvalidated model candidates. This script preserves raw generations and
records model/runtime metadata; it does not certify semantics or Bloom alignment.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="JSONL from build_generation_queue.py")
    ap.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    ap.add_argument("--output", required=True, help="Raw candidate JSONL output")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests; 0 means all")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be positive")
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
            if args.limit and len(jobs) >= args.limit:
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out.open("w", encoding="utf-8") as f:
        for idx, job in enumerate(jobs, 1):
            user_prompt = (
                f'{job.get("prompt", "")}\n\n'
                f'Source question: {job.get("source_question", "")}\n'
                f'Source Bloom level: {job.get("source_bloom_level", "unknown")}\n'
                f'Target Bloom level: {job.get("target_bloom_level", "")}\n'
                f'Preferred opening verbs: {job.get("preferred_opening_verbs", "")}\n'
                "Return one JSON object only. Do not answer the question."
            )
            messages = [
                {"role": "system", "content": "You are an educational assessment item editor. Preserve source meaning and context; never invent missing facts."},
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
