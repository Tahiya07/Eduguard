#!/usr/bin/env python
"""Local LLM-as-judge accuracy for an evaluate_rewrite.py run (no regeneration of model outputs).

ROUGE / BERTScore measure overlap with one reference; they are not accuracy. This scores every
prediction with a local judge model (default Qwen/Qwen2.5-7B-Instruct, 4-bit) under a fixed rubric,
greedy decoding, JSON output:

* QA (closed-book: the judge sees only the question, the reference answer and the model answer;
  no passage is given to the model or the judge): correct / partially_correct / incorrect.
  accuracy = share judged correct; lenient accuracy = correct or partially correct.
* Summarization: the judge sees the source text, the reference summary and the model summary.
  faithful = no statement contradicts or goes beyond the source; coverage 1-5 of the reference's
  main points; acceptable = faithful and coverage >= 3.
* Bloom rewrite: the judge sees only the rewritten question (NOT the requested level) and names its
  Bloom level; level accuracy = judged level equals the requested level. It also says whether the
  rewrite keeps the source question's topic. This is a second, independent instrument next to the
  fixed 0.5B classifier.

Judge reliability (computed once per dataset split, reused for base and LoRA):
  QA / summarization: the reference itself must be judged correct / faithful, and the reference of a
  different row must be judged incorrect / not faithful. Bloom: the judge's level accuracy on the
  human reference rewrites (the judge's ceiling, like the classifier ceiling).

Outputs in the results directory: judge_predictions.jsonl (resumable), judge_metrics.json,
judge_report.md; calibration in <config results_dir>/judge_calibration_<split>.json.

Usage (GPU; the judge is ~5.5 GB in 4-bit):
  python experiments/multitask_bloom_rewrite/scripts/judge_evaluation.py                 # LoRA results
  python experiments/multitask_bloom_rewrite/scripts/judge_evaluation.py --condition base
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
QA_VERDICTS = ("correct", "partially_correct", "incorrect")
MAX_SOURCE_TOKENS = 6000
SUMMARY_CALIBRATION_ROWS = 40  # long inputs; QA and Bloom calibrate on every row
SEED = 42
BOOTSTRAP = 2000
RELIABILITY_MIN = 0.9  # reference accepted AND mismatched reference rejected, per task

SYSTEM = ("You are a strict, impartial university examiner. You grade one item at a time against the rubric "
          "you are given and reply with a single JSON object only, no other text.")

QA_PROMPT = """Grade a student's answer to a closed-book university exam question.

<question>
{question}
</question>

<student_answer>
{answer}
</student_answer>

<reference_answer>
{reference}
</reference_answer>

Grade ONLY the text inside <student_answer>. The reference answer is one acceptable answer written by the teacher; use it to know what is correct.
Step 1 - "addresses_question": is the student answer about the question asked? An answer about a different question or topic is false, however accurate it is in itself.
Step 2 - "verdict":
- "correct": the student answer states the same key facts/conclusion as the reference and contains no significant factual error. Different wording, a different but valid explanation, or extra correct detail is fine. For calculations the final result must match the reference (rounding differences are fine).
- "partially_correct": some key point is right, but an essential part is missing or there is a minor error.
- "incorrect": wrong, contradicts the reference, does not answer the question, or is empty.
Judge correctness only, not style or length.

Reply with JSON: {{"addresses_question": true | false, "verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one short sentence about the student answer>"}}"""

SUMM_PROMPT = """Grade a summary of a university text.

<summary_to_grade>
{summary}
</summary_to_grade>

<source_text>
{source}
</source_text>

<reference_summary>
{reference}
</reference_summary>

Grade ONLY the text inside <summary_to_grade>; the reference summary was written by the author and shows the main points.
Step 1 - "about_source": is the summary to grade about this source text? A summary of a different text is false.
Step 2 - "faithful": true only if every statement in the summary to grade is supported by the source text (no contradictions, no invented facts, numbers or claims). Omissions do not make it unfaithful. A summary that repeats itself or is cut off can still be faithful.
Step 3 - "coverage": 1-5, how many of the main points of the reference summary the summary to grade covers (1 = almost none, 3 = about half, 5 = all).

Reply with JSON: {{"about_source": true | false, "faithful": true | false, "coverage": 1-5, "reason": "<one short sentence about the summary to grade>"}}"""

BLOOM_PROMPT = """Classify an exam question by the cognitive level of Bloom's revised taxonomy it requires.

Levels:
- Remember: recall facts, terms, definitions, formulas or lists from memory.
- Understand: explain, describe, summarise, interpret or give examples of an idea in one's own words.
- Apply: use a method, rule, formula or procedure in a given situation or on given data (e.g. calculate, solve, implement).
- Analyze: break something into parts, compare, examine relationships or causes, distinguish.
- Evaluate: judge, assess, justify, critique or decide using criteria.
- Create: design, develop, construct, propose or produce something new.
Classify by what the student must actually do, not by the first verb alone.

Original question (for topic only):
{source}

Question to classify:
{question}

Reply with JSON: {{"level": "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate" | "Create", "same_topic": true | false}}
("same_topic": does the question to classify stay on the topic of the original question?)"""


# ------------------------------------------------------------------ judge model
class Judge:
    def __init__(self, model_id: str, precision: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        cuda = torch.cuda.is_available()
        kwargs: dict[str, Any] = {}
        if precision in ("4bit", "8bit"):
            if not cuda:
                raise SystemExit(f"--precision {precision} needs a CUDA GPU; use --precision bf16/fp32 on CPU")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
                if precision == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        if "device_map" not in kwargs:
            self.model.to("cuda" if cuda else "cpu")
        self.device = next(self.model.parameters()).device

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tok(text or "", add_special_tokens=False)["input_ids"]
        return text if len(ids) <= max_tokens else self.tok.decode(ids[:max_tokens]) + " [...]"

    def ask(self, prompt: str, max_new_tokens: int = 120) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        # enable_thinking=False only affects templates that use it (Qwen3); others ignore the variable
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_json(raw: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*}", "}", m.group(0).replace("'", '"')))
        except json.JSONDecodeError:
            return None


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "false", "no"):
        return v.strip().lower() in ("true", "yes")
    return None


# ------------------------------------------------------------------ per-task judging
def judge_qa(j: Judge, question: str, reference: str, answer: str) -> dict[str, Any]:
    if not (answer or "").strip():
        return {"verdict": "incorrect", "reason": "empty answer", "parse_ok": True, "raw": ""}
    raw = j.ask(QA_PROMPT.format(question=question, reference=reference, answer=answer), max_new_tokens=160)
    d = parse_json(raw) or {}
    v = str(d.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if v not in QA_VERDICTS:
        hit = re.search(r"partially_correct|incorrect|correct", raw.lower().replace(" ", "_"))
        v = hit.group(0) if hit else None
    on_topic = as_bool(d.get("addresses_question"))
    if on_topic is False:
        v = "incorrect"  # gate: an answer to a different question is never correct
    return {"verdict": v, "addresses_question": on_topic, "reason": d.get("reason"), "parse_ok": v is not None, "raw": raw}


def judge_summary(j: Judge, source: str, reference: str, summary: str) -> dict[str, Any]:
    if not (summary or "").strip():
        return {"faithful": False, "coverage": 1, "reason": "empty summary", "parse_ok": True, "raw": ""}
    raw = j.ask(SUMM_PROMPT.format(source=j.truncate(source, MAX_SOURCE_TOKENS), reference=reference, summary=summary),
                max_new_tokens=160)
    d = parse_json(raw) or {}
    faithful = as_bool(d.get("faithful"))
    try:
        coverage = min(5, max(1, int(d.get("coverage"))))
    except (TypeError, ValueError):
        coverage = None
    about = as_bool(d.get("about_source"))
    if about is False:
        faithful, coverage = False, 1  # gate: a summary of a different text
    return {"faithful": faithful, "coverage": coverage, "about_source": about, "reason": d.get("reason"),
            "parse_ok": faithful is not None and coverage is not None, "raw": raw}


def judge_bloom(j: Judge, source: str, question: str) -> dict[str, Any]:
    if not (question or "").strip():
        return {"level": None, "same_topic": False, "parse_ok": True, "raw": ""}
    raw = j.ask(BLOOM_PROMPT.format(source=source, question=question), max_new_tokens=40)
    d = parse_json(raw) or {}
    level = str(d.get("level", "")).strip().title()
    if level not in LEVELS:
        hit = [lv for lv in LEVELS if lv.lower() in raw.lower()]
        level = hit[0] if len(hit) == 1 else None
    return {"level": level, "same_topic": as_bool(d.get("same_topic")), "parse_ok": level is not None, "raw": raw}


def judge_row(j: Judge, r: dict[str, Any]) -> dict[str, Any]:
    t = r["task"]
    if t == "qa":
        out = judge_qa(j, r["question"], r["reference"], r["prediction"])
        out.update(judge_correct_f=float(out["verdict"] == "correct"),
                   judge_lenient_f=float(out["verdict"] in ("correct", "partially_correct")))
    elif t == "summarization":
        out = judge_summary(j, r.get("article", ""), r["reference"], r["prediction"])
        out.update(judge_faithful_f=float(out["faithful"] is True),
                   judge_acceptable_f=float(out["faithful"] is True and (out["coverage"] or 0) >= 3))
    else:
        out = judge_bloom(j, r["source_question"], r["prediction"])
        out.update(judge_level_match_f=float(out["level"] == r["target_bloom_level"]),
                   judge_topic_f=float(out["same_topic"] is True))
    return {"id": r["id"], "task": t, **out}


# ------------------------------------------------------------------ statistics
def mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    rng = random.Random(SEED)
    n = len(values)
    boots = sorted(st.mean(rng.choices(values, k=n)) for _ in range(BOOTSTRAP))
    return {"mean": round(st.mean(values), 4), "ci95": [round(boots[int(0.025 * BOOTSTRAP)], 4),
                                                        round(boots[int(0.975 * BOOTSTRAP) - 1], 4)], "n": n}


def summarize(preds: list[dict[str, Any]], judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for r in preds:
        if r["id"] in judged:
            by_task[r["task"]].append((r, judged[r["id"]]))
    out: dict[str, Any] = {}
    if by_task.get("qa"):
        rows = by_task["qa"]
        groups = defaultdict(list)
        for r, jd in rows:
            groups[r.get("qa_type") or "unknown"].append(jd["judge_correct_f"])
        out["qa"] = {
            "accuracy": mean_ci([jd["judge_correct_f"] for _, jd in rows]),
            "lenient_accuracy": mean_ci([jd["judge_lenient_f"] for _, jd in rows]),
            "verdicts": dict(Counter(jd["verdict"] for _, jd in rows)),
            "unparsed": sum(not jd["parse_ok"] for _, jd in rows),
            "accuracy_by_qa_type": {k: {"n": len(v), "accuracy": round(st.mean(v), 4)} for k, v in sorted(groups.items())},
        }
    if by_task.get("summarization"):
        rows = by_task["summarization"]
        cov = [jd["coverage"] for _, jd in rows if jd["coverage"] is not None]
        out["summarization"] = {
            "faithful_rate": mean_ci([jd["judge_faithful_f"] for _, jd in rows]),
            "acceptable_rate": mean_ci([jd["judge_acceptable_f"] for _, jd in rows]),
            "mean_coverage_1_5": round(st.mean(cov), 3) if cov else None,
            "unparsed": sum(not jd["parse_ok"] for _, jd in rows),
        }
    if by_task.get("bloom_rewrite"):
        rows = by_task["bloom_rewrite"]
        per_level = {}
        for lv in LEVELS:
            v = [jd["judge_level_match_f"] for r, jd in rows if r["target_bloom_level"] == lv]
            per_level[lv] = {"n": len(v), "accuracy": round(st.mean(v), 4) if v else None}
        both = [jd["judge_level_match_f"] * float(bool(r.get("target_match"))) for r, jd in rows]
        out["bloom_rewrite"] = {
            "level_accuracy": mean_ci([jd["judge_level_match_f"] for _, jd in rows]),
            "topic_kept_rate": mean_ci([jd["judge_topic_f"] for _, jd in rows]),
            "judge_and_classifier_agree_on_target": mean_ci(both),
            "per_level": per_level,
            "unparsed": sum(not jd["parse_ok"] for _, jd in rows),
        }
    return out


# ------------------------------------------------------------------ calibration
def reliability(cal: dict[str, Any]) -> dict[str, bool]:
    """A task's judge scores are reported only if the judge accepts the references and rejects mismatched ones."""
    out = {}
    if "qa" in cal:
        c = cal["qa"]
        out["qa"] = min(c["reference_judged_correct"], c["other_rows_reference_judged_incorrect"]) >= RELIABILITY_MIN
    if "summarization" in cal:
        c = cal["summarization"]
        out["summarization"] = min(c["reference_judged_faithful"], c["other_rows_summary_judged_unfaithful"]) >= RELIABILITY_MIN
    return out


def calibrate(j: Judge, preds: list[dict[str, Any]], sample: dict[str, int] | None = None) -> dict[str, Any]:
    """Judge reliability on references only (independent of the evaluated system)."""
    rng = random.Random(SEED)
    cal: dict[str, Any] = {}
    sample = sample or {}

    def pick(task: str, default: int | None = None) -> list[dict[str, Any]]:
        rows = [r for r in preds if r["task"] == task]
        k = sample.get(task, default)
        return rng.sample(rows, k) if k is not None and k < len(rows) else rows

    qa = pick("qa")
    if qa:
        same = [judge_qa(j, r["question"], r["reference"], r["reference"])["verdict"] == "correct" for r in qa]
        others = [x["reference"] for x in qa]
        wrong = []
        for i, r in enumerate(qa):
            k = rng.choice([x for x in range(len(qa)) if x != i])
            wrong.append(judge_qa(j, r["question"], r["reference"], others[k])["verdict"] == "incorrect")
        cal["qa"] = {"n": len(qa), "reference_judged_correct": round(st.mean(same), 4),
                     "other_rows_reference_judged_incorrect": round(st.mean(wrong), 4)}
    summ = [r for r in preds if r["task"] == "summarization"]
    if summ:
        sample_rows = pick("summarization", SUMMARY_CALIBRATION_ROWS)
        same = [judge_summary(j, r.get("article", ""), r["reference"], r["reference"])["faithful"] is True for r in sample_rows]
        wrong = []
        for r in sample_rows:
            other = rng.choice([x for x in summ if x["id"] != r["id"]])
            wrong.append(judge_summary(j, r.get("article", ""), r["reference"], other["reference"])["faithful"] is False)
        cal["summarization"] = {"n": len(sample_rows), "reference_judged_faithful": round(st.mean(same), 4),
                                "other_rows_summary_judged_unfaithful": round(st.mean(wrong), 4)}
    bloom = pick("bloom_rewrite")
    if bloom:
        res = [(r["target_bloom_level"], judge_bloom(j, r["source_question"], r["reference"])["level"]) for r in bloom]
        cal["bloom_rewrite"] = {
            "n": len(res),
            "reference_level_accuracy": round(st.mean(float(a == b) for a, b in res), 4),
            "reference_per_level": {lv: round(st.mean(float(b == lv) for a, b in res if a == lv), 4)
                                    for lv in LEVELS if any(a == lv for a, _ in res)},
        }
    return cal


# ------------------------------------------------------------------ report
def fmt(m: dict[str, Any] | None) -> str:
    if not m or m.get("mean") is None:
        return "n/a"
    return f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"


def write_report(path: Path, meta: dict[str, Any], res: dict[str, Any], cal: dict[str, Any]) -> None:
    L = [f"# Judge evaluation ({meta['condition']}, {meta['split']})", "",
         f"Judge: `{meta['judge_model']}` ({meta['precision']}), greedy, fixed rubric; generated {meta['generated_utc']}.",
         "Accuracy here is the judge's verdict, not word overlap. Values are mean [95% bootstrap CI].", ""]
    q, s, b = res.get("qa"), res.get("summarization"), res.get("bloom_rewrite")
    L += ["| Task | Measure | Value |", "|---|---|---:|"]
    if q:
        L += [f"| Closed-book QA | Accuracy (judged correct) | {fmt(q['accuracy'])} |",
              f"| Closed-book QA | Lenient accuracy (correct or partially correct) | {fmt(q['lenient_accuracy'])} |"]
    if s:
        L += [f"| Summarization | Faithful (no unsupported claims) | {fmt(s['faithful_rate'])} |",
              f"| Summarization | Acceptable (faithful and coverage >= 3/5) | {fmt(s['acceptable_rate'])} |",
              f"| Summarization | Mean coverage (1-5) | {s['mean_coverage_1_5']} |"]
    if b:
        L += [f"| Bloom rewrite | Level accuracy (judge, blind to requested level) | {fmt(b['level_accuracy'])} |",
              f"| Bloom rewrite | Judge and 0.5B classifier both agree with requested level | {fmt(b['judge_and_classifier_agree_on_target'])} |",
              f"| Bloom rewrite | Topic kept | {fmt(b['topic_kept_rate'])} |"]
    if q:
        L += ["", "QA accuracy by question type:", "", "| QA type | N | Accuracy |", "|---|---:|---:|"]
        L += [f"| {k} | {v['n']} | {v['accuracy']} |" for k, v in q["accuracy_by_qa_type"].items()]
    if b:
        ref = cal.get("bloom_rewrite", {}).get("reference_per_level", {})
        L += ["", "Bloom level accuracy by requested level:", "",
              "| Level | N | Model (judge) | Human reference (judge ceiling) |", "|---|---:|---:|---:|"]
        L += [f"| {lv} | {v['n']} | {v['accuracy']} | {ref.get(lv, 'n/a')} |" for lv, v in b["per_level"].items() if v["n"]]
    rel = reliability(cal)
    bad = [t for t, ok in rel.items() if not ok]
    if bad:
        L[4:4] = [f"**WARNING: the judge is NOT RELIABLE for {', '.join(bad)}** (it must accept the reference and reject "
                  f"a mismatched one at least {RELIABILITY_MIN:.0%} of the time; see below). Do not report those numbers; "
                  "use a stronger judge or the human sheets.", ""]
    L += ["", "## Judge reliability (references only, same for every system)", ""]
    if "qa" in cal:
        c = cal["qa"]
        L.append(f"- QA (n={c['n']}): reference answer judged correct {c['reference_judged_correct']}; "
                 f"another question's reference judged incorrect {c['other_rows_reference_judged_incorrect']}.")
    if "summarization" in cal:
        c = cal["summarization"]
        L.append(f"- Summarization (n={c['n']}): reference judged faithful {c['reference_judged_faithful']}; "
                 f"another section's summary judged unfaithful {c['other_rows_summary_judged_unfaithful']}.")
    if "bloom_rewrite" in cal:
        L.append(f"- Bloom (n={cal['bloom_rewrite']['n']}): judge agrees with the requested level on the human reference "
                 f"rewrites {cal['bloom_rewrite']['reference_level_accuracy']}.")
    unparsed = {t: v.get("unparsed") for t, v in res.items()}
    L += ["", f"Unparseable judge replies (counted as not correct): {unparsed}.",
          "An LLM judge can be wrong; confirm on the human sheets (human_eval_*.csv) before final claims.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ main
def run(results_dir: Path, calibration_path: Path, judge_model: str, precision: str, condition: str,
        tasks: list[str] | None = None, limit: int | None = None, recalibrate: bool = False) -> dict[str, Any]:
    preds = [json.loads(l) for l in (results_dir / "predictions.jsonl").open(encoding="utf-8") if l.strip()]
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    split = metrics.get("dataset", {}).get("split", "validation")
    if tasks:
        preds = [r for r in preds if r["task"] in tasks]
    if limit:
        seen: Counter = Counter()
        kept = []
        for r in preds:
            seen[r["task"]] += 1
            if seen[r["task"]] <= limit:
                kept.append(r)
        preds = kept
    out_path = results_dir / "judge_predictions.jsonl"
    judged: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            d = json.loads(l)
            if d.get("judge_model") == judge_model:
                judged[d["id"]] = d
    todo = [r for r in preds if r["id"] not in judged]
    cal = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() and not recalibrate else None
    if cal and cal.get("judge_model") != judge_model:
        cal = None
    j = None
    if todo or cal is None:
        print(f"Loading judge {judge_model} ({precision}) ...", flush=True)
        j = Judge(judge_model, precision)
    if todo:
        print(f"Judging {len(todo)} predictions ({len(judged)} already done) ...", flush=True)
        t0 = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            for i, r in enumerate(todo, 1):
                d = {**judge_row(j, r), "judge_model": judge_model}
                judged[r["id"]] = d
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if cal is None and not limit and not tasks:
        print("Calibrating judge on references ...", flush=True)
        cal = {"judge_model": judge_model, "precision": precision, "split": split, **calibrate(j, preds)}
        calibration_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    cal = cal or {}
    res = summarize(preds, judged)
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "judge_model": judge_model, "precision": precision,
            "condition": condition, "split": split, "results_dir": str(results_dir), "calibration_file": str(calibration_path)}
    payload = {**meta, "results": res, "calibration": cal, "reliable": reliability(cal)}
    (results_dir / "judge_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(results_dir / "judge_report.md", meta, res, cal)
    if j is not None:
        del j
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return payload


def main() -> None:
    from paths import ACTIVE_TRAINING_CONFIG, CONFIG_DIR

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_DIR / ACTIVE_TRAINING_CONFIG))
    ap.add_argument("--condition", choices=["lora", "base"], default="lora")
    ap.add_argument("--results-dir", default=None, help="override the results directory")
    ap.add_argument("--judge-model", default=None, help=f"default: config judge.model or {JUDGE_MODEL}")
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16", "fp16", "fp32"], default=None,
                    help="default: config judge.precision or 4bit")
    ap.add_argument("--tasks", nargs="*", choices=["bloom_rewrite", "qa", "summarization"])
    ap.add_argument("--limit", type=int, help="judge only the first N rows per task (smoke test; no calibration)")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="quick reliability check only (30 QA, 6 summaries, 60 Bloom references); judges no predictions")
    a = ap.parse_args()
    config = Path(a.config) if Path(a.config).is_absolute() else REPO_ROOT / a.config
    cfg = json.loads(config.read_text(encoding="utf-8"))
    jcfg = cfg.get("judge", {})
    base_rd = Path(cfg["results_dir"])
    base_rd = base_rd if base_rd.is_absolute() else REPO_ROOT / base_rd
    rd = Path(a.results_dir) if a.results_dir else (base_rd if a.condition == "lora" else base_rd.with_name(base_rd.name + "_base"))
    rd = rd if rd.is_absolute() else REPO_ROOT / rd
    metrics = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    split = metrics.get("dataset", {}).get("split", "validation")
    if a.check:
        model, precision = a.judge_model or jcfg.get("model", JUDGE_MODEL), a.precision or jcfg.get("precision", "4bit")
        preds = [json.loads(l) for l in (rd / "predictions.jsonl").open(encoding="utf-8") if l.strip()]
        t0 = time.time()
        cal = calibrate(Judge(model, precision), preds, {"qa": 30, "summarization": 6, "bloom_rewrite": 60})
        print(json.dumps({"judge_model": model, "precision": precision, "seconds": round(time.time() - t0),
                          "calibration": cal, "reliable": reliability(cal)}, indent=2))
        return
    payload = run(rd, base_rd / f"judge_calibration_{split}.json", a.judge_model or jcfg.get("model", JUDGE_MODEL),
                  a.precision or jcfg.get("precision", "4bit"), a.condition, a.tasks, a.limit, a.recalibrate)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print((rd / "judge_report.md").read_text(encoding="utf-8"))
    del payload


if __name__ == "__main__":
    main()
