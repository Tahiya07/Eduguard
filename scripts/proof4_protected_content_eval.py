#!/usr/bin/env python3
"""
Proof ④ — Protected-content evaluation.

Reports the three paper-facing rates:

  * attack_block_rate   = blocked_attacks / n_attacks
  * leakage_rate        = leaked_attacks / n_attacks   (= attack success)
  * false_block_rate    = blocked_benign / n_benign    (= 1 - benign_allow_rate)

Modes:

  # Recompute from the locked artifact (no model calls)
  python scripts/proof4_protected_content_eval.py --from-artifact

  # Re-run the live PrivacyGuard taxonomy suite
  python scripts/proof4_protected_content_eval.py --rerun

  # Also include baseline ablation summary if present
  python scripts/proof4_protected_content_eval.py --from-artifact --include-ablation

Does not claim formal DP or cryptographic security.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "paper" / "proofs" / "proof4_privacy"
ARTIFACT = ROOT / "artifacts" / "model train results" / "privacy_guard_eval.json"
ABLATION = ROOT / "artifacts" / "model train results" / "privacy_benchmark_baselines.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rates_from_guard_payload(payload: dict[str, Any], source: str) -> dict[str, Any]:
    n_attack = int(payload.get("n_attack_prompts") or 0)
    n_benign = int(payload.get("n_student_benign") or 0)
    block = float(payload.get("student_attack_block_rate") or 0.0)
    success = float(payload.get("student_attack_success_rate") or 0.0)
    benign_allow = float(payload.get("student_benign_allow_rate") or 0.0)
    false_block = 1.0 - benign_allow

    # Prefer counts when reconstructible
    blocked_attacks = round(block * n_attack) if n_attack else None
    leaked_attacks = round(success * n_attack) if n_attack else None
    blocked_benign = round(false_block * n_benign) if n_benign else None

    return {
        "proof": "④ Protected-content evaluation",
        "timestamp_utc": utc_now(),
        "source": source,
        "simulation_mode": payload.get("simulation_mode"),
        "counts": {
            "n_attacks": n_attack,
            "n_benign": n_benign,
            "n_teacher_moderation": int(payload.get("n_teacher_moderation") or 0),
            "blocked_attacks": blocked_attacks,
            "leaked_attacks": leaked_attacks,
            "blocked_benign": blocked_benign,
            "allowed_benign": round(benign_allow * n_benign) if n_benign else None,
        },
        "metrics": {
            "attack_block_rate": block,
            "leakage_rate": success,
            "false_block_rate": false_block,
            "benign_allow_rate": benign_allow,
            "teacher_moderation_allow_rate": float(
                payload.get("teacher_moderation_allow_rate") or 0.0
            ),
        },
        "attack_category_summary": payload.get("attack_category_summary") or {},
        "paper_ready": bool(n_attack > 0 and n_benign > 0),
        "limitations": [
            "Empirical screening only — not a cryptographic security proof.",
            "Formal differential privacy is out of scope for this proof.",
            "High false_block_rate indicates over-blocking / utility collapse.",
        ],
    }


def from_artifact(path: Path = ARTIFACT) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "MISSING",
            "paper_ready": False,
            "reason": f"Artifact not found: {path}",
            "metrics": {
                "attack_block_rate": None,
                "leakage_rate": None,
                "false_block_rate": None,
            },
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = _rates_from_guard_payload(
        payload,
        source=str(path.relative_to(ROOT)).replace("\\", "/"),
    )
    report["status"] = "FROM_ARTIFACT"
    return report


def include_ablation(report: dict[str, Any], path: Path = ABLATION) -> dict[str, Any]:
    if not path.is_file():
        report["ablation"] = {"status": "MISSING", "path": str(path)}
        return report
    data = json.loads(path.read_text(encoding="utf-8"))
    baselines = data.get("baselines") or {}
    rows = []
    if isinstance(baselines, dict):
        for name, stats in baselines.items():
            rows.append(
                {
                    "name": name,
                    "attack_block_rate": stats.get("attack_block_rate"),
                    "leakage_rate": stats.get("attack_success_rate"),
                    "false_block_rate": (
                        None
                        if stats.get("benign_allow_rate") is None
                        else 1.0 - float(stats["benign_allow_rate"])
                    ),
                    "benign_allow_rate": stats.get("benign_allow_rate"),
                }
            )
    report["ablation"] = {
        "status": "OK",
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "n_cases": data.get("n_cases"),
        "n_attack_cases": data.get("n_attack_cases"),
        "n_benign_cases": data.get("n_benign_cases"),
        "baselines": rows,
    }
    return report


def rerun() -> dict[str, Any]:
    """Execute privacy/evaluate_privacy_guard.py logic and map to proof metrics."""
    from privacy.evaluate_privacy_guard import (  # type: ignore
        _attack_sets,
        _row,
        _student_benign_sets,
        _teacher_moderation_sets,
        _mean,
    )

    rows: list[dict[str, object]] = []
    for category, prompts in _attack_sets().items():
        for item in prompts:
            kind = "student_benign" if category == "benign_safe_alternative" else "student_attack"
            rows.append(_row(kind, category, item["query"], item["answer"], "student"))
    for item in _student_benign_sets():
        rows.append(_row("student_benign", "benign_study_help", item["query"], item["answer"], "student"))
    for item in _teacher_moderation_sets():
        rows.append(_row("teacher_moderation", "teacher_moderation", item["query"], item["answer"], "teacher"))

    attacks = [r for r in rows if r["kind"] == "student_attack"]
    benign = [r for r in rows if r["kind"] == "student_benign"]
    teacher = [r for r in rows if r["kind"] == "teacher_moderation"]

    from collections import defaultdict

    by_cat: dict[str, list] = defaultdict(list)
    for r in attacks:
        by_cat[str(r["category"])].append(r)
    attack_category_summary = {
        cat: {
            "n": len(items),
            "block_rate": _mean([0.0 if bool(i["allowed"]) else 1.0 for i in items]),
            "mean_risk_score": _mean([float(i["risk_score"]) for i in items]),
        }
        for cat, items in sorted(by_cat.items())
    }

    payload = {
        "simulation_mode": False,
        "n_rows": len(rows),
        "n_attack_prompts": len(attacks),
        "n_student_benign": len(benign),
        "n_teacher_moderation": len(teacher),
        "student_attack_block_rate": _mean([0.0 if bool(r["allowed"]) else 1.0 for r in attacks]),
        "student_attack_success_rate": _mean([1.0 if bool(r["allowed"]) else 0.0 for r in attacks]),
        "student_benign_allow_rate": _mean([1.0 if bool(r["allowed"]) else 0.0 for r in benign]),
        "teacher_moderation_allow_rate": _mean(
            [1.0 if bool(r["allowed"]) else 0.0 for r in teacher]
        ),
        "attack_category_summary": attack_category_summary,
    }
    report = _rates_from_guard_payload(payload, source="privacy.evaluate_privacy_guard (live rerun)")
    report["status"] = "RERUN"
    report["rows"] = rows
    return report


def _write_outputs(report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Drop bulky rows from the summary JSON if present
    summary = {k: v for k, v in report.items() if k != "rows"}
    out = OUT_DIR / "proof4_protected_content_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    m = report.get("metrics") or {}
    c = report.get("counts") or {}
    md = OUT_DIR / "proof4_protected_content_report.md"
    lines = [
        "# Proof ④ — Protected-content evaluation",
        "",
        f"- status: `{report.get('status')}`",
        f"- source: `{report.get('source')}`",
        f"- paper_ready (artifact present): `{report.get('paper_ready')}`",
        "",
        "## Headline rates",
        "",
        f"| Metric | Value | Count |",
        f"| --- | ---: | ---: |",
        f"| attack_block_rate | {m.get('attack_block_rate')} | "
        f"{c.get('blocked_attacks')}/{c.get('n_attacks')} |",
        f"| leakage_rate | {m.get('leakage_rate')} | "
        f"{c.get('leaked_attacks')}/{c.get('n_attacks')} |",
        f"| false_block_rate | {m.get('false_block_rate')} | "
        f"{c.get('blocked_benign')}/{c.get('n_benign')} |",
        f"| benign_allow_rate | {m.get('benign_allow_rate')} | "
        f"{c.get('allowed_benign')}/{c.get('n_benign')} |",
        "",
        "## Limitations",
        "",
    ]
    for lim in report.get("limitations") or []:
        lines.append(f"- {lim}")
    lines.append("")
    if report.get("ablation"):
        lines.append("## Ablation")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report["ablation"], indent=2))
        lines.append("```")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")

    if "rows" in report:
        rows_path = OUT_DIR / "proof4_protected_content_rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as f:
            for row in report["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Proof ④ protected-content rates")
    parser.add_argument("--from-artifact", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument(
        "--artifact",
        type=str,
        default=str(ARTIFACT),
        help="Path to privacy_guard_eval.json",
    )
    args = parser.parse_args()

    if not args.from_artifact and not args.rerun:
        parser.print_help()
        raise SystemExit(2)

    if args.from_artifact and args.rerun:
        print("Choose only one of --from-artifact or --rerun", file=sys.stderr)
        raise SystemExit(2)

    if args.from_artifact:
        art = Path(args.artifact)
        if not art.is_file():
            art = ROOT / args.artifact
        report = from_artifact(art)
    else:
        report = rerun()

    if args.include_ablation:
        report = include_ablation(report)

    out = _write_outputs(report)
    slim = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(slim, indent=2))
    print(f"[proof4] wrote {out}")
    if report.get("status") == "MISSING":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
