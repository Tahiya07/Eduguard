#!/usr/bin/env python3
"""
Proof ③ — RAG retrieval evaluation (Recall@1 / Recall@3 / Recall@5 + MRR).

This is the locked paper entry point for retrieval metrics. It does NOT invent
results. Modes:

  # Report what the existing smoke CSV can support (honest gaps)
  python scripts/proof3_rag_retrieval_eval.py --summarize-existing

  # Write an empty/expandable benchmark scaffold (no metrics claimed)
  python scripts/proof3_rag_retrieval_eval.py --write-scaffold

  # Run a real benchmark JSONL (requires local encoder + faiss)
  python scripts/proof3_rag_retrieval_eval.py --run-benchmark data/rag_benchmark/queries.jsonl

Benchmark JSONL schema (one object per line):
  {
    "query_id": "q001",
    "query": "...",
    "relevant_doc_ids": ["d01", "d02"],   # gold relevant passages
    "corpus": [                            # optional per-query corpus override
      {"doc_id": "d01", "text": "..."},
      {"doc_id": "d02", "text": "..."}
    ]
  }

Shared corpus file (optional): data/rag_benchmark/corpus.jsonl
  {"doc_id": "d01", "text": "..."}

Paper rule: do not cite smoke-only hit@k as a complete Recall@1/3/5+MRR proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "paper" / "proofs" / "proof3_rag"
SMOKE_CSV = ROOT / "artifacts" / "model train results" / "qwen_rag_eval_rows.csv"
SCAFFOLD_DIR = ROOT / "data" / "rag_benchmark"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize_existing(smoke_csv: Path = SMOKE_CSV) -> dict[str, Any]:
    """Derive whatever retrieval metrics exist in the smoke CSV — no invention."""
    if not smoke_csv.is_file():
        payload = {
            "status": "MISSING",
            "proof": "③ RAG retrieval",
            "paper_ready": False,
            "reason": f"Smoke CSV not found: {smoke_csv}",
            "required_for_paper": ["Recall@1", "Recall@3", "Recall@5", "MRR"],
            "available": {},
        }
        return payload

    rows: list[dict[str, str]] = []
    with smoke_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("task") or "").strip() == "academic_qa":
                rows.append(row)

    def _col_mean(name: str) -> float | None:
        vals = []
        for r in rows:
            raw = (r.get(name) or "").strip()
            if raw == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        return _mean(vals) if vals else None

    hit1 = _col_mean("retrieval_hit_at_1")
    hit3 = _col_mean("retrieval_hit_at_3")
    # These columns are not present in the smoke CSV
    has_r5 = any((r.get("retrieval_hit_at_5") or "").strip() for r in rows)
    has_mrr = any((r.get("retrieval_mrr") or "").strip() for r in rows)

    payload = {
        "status": "SMOKE_ONLY",
        "proof": "③ RAG retrieval",
        "paper_ready": False,
        "timestamp_utc": utc_now(),
        "source_artifact": str(smoke_csv.relative_to(ROOT)).replace("\\", "/"),
        "n_academic_qa_rows": len(rows),
        "available": {
            "hit_at_1_mean": hit1,
            "hit_at_3_mean": hit3,
            "note": "Smoke CSV uses hit_at_k column names; treat as feasibility only.",
        },
        "missing_for_proof": [],
        "required_for_paper": ["Recall@1", "Recall@3", "Recall@5", "MRR"],
    }
    if hit1 is None:
        payload["missing_for_proof"].append("Recall@1 / hit@1")
    if hit3 is None:
        payload["missing_for_proof"].append("Recall@3 / hit@3")
    if not has_r5:
        payload["missing_for_proof"].append("Recall@5")
    if not has_mrr:
        payload["missing_for_proof"].append("MRR")

    payload["interpretation"] = (
        "Existing artifact is a small academic_qa smoke set. "
        "It does NOT satisfy the paper proof requiring Recall@1/3/5 + MRR "
        "on a proper retrieval benchmark. Run --run-benchmark after filling "
        "data/rag_benchmark/."
    )
    return payload


def write_scaffold(out_dir: Path = SCAFFOLD_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    readme = out_dir / "README.md"

    # Seed with topics aligned to the smoke set — clearly marked non-final.
    corpus = [
        {
            "doc_id": "circuits_ohm",
            "text": (
                "Ohm's law states that voltage equals current times resistance. "
                "A resistor with constant resistance has a linear current-voltage relationship."
            ),
        },
        {
            "doc_id": "circuits_power",
            "text": (
                "Electrical power in a simple circuit can be computed as voltage times current. "
                "Power is measured in watts."
            ),
        },
        {
            "doc_id": "networks_tcp",
            "text": (
                "TCP provides reliable, ordered delivery using acknowledgements and retransmission. "
                "UDP has lower transport overhead but does not guarantee ordering or delivery."
            ),
        },
        {
            "doc_id": "algo_mergesort",
            "text": (
                "Merge sort is a divide-and-conquer algorithm with time complexity O(n log n) "
                "in the average and worst cases."
            ),
        },
        {
            "doc_id": "stats_pvalue",
            "text": (
                "A p-value is the probability, under the null hypothesis, of observing a result "
                "at least as extreme as the one obtained. Smaller p-values indicate stronger "
                "evidence against the null at a fixed significance level alpha."
            ),
        },
    ]
    queries = [
        {
            "query_id": "q_ohm",
            "query": "What relationship does Ohm's law define?",
            "relevant_doc_ids": ["circuits_ohm"],
        },
        {
            "query_id": "q_power",
            "query": "What unit is electrical power measured in?",
            "relevant_doc_ids": ["circuits_power"],
        },
        {
            "query_id": "q_tcp",
            "query": "Which protocol provides reliable ordered delivery?",
            "relevant_doc_ids": ["networks_tcp"],
        },
        {
            "query_id": "q_udp",
            "query": "What tradeoff does UDP make compared with TCP?",
            "relevant_doc_ids": ["networks_tcp"],
        },
        {
            "query_id": "q_merge",
            "query": "What is the time complexity of merge sort?",
            "relevant_doc_ids": ["algo_mergesort"],
        },
        {
            "query_id": "q_pval",
            "query": "What does a p-value represent in hypothesis testing?",
            "relevant_doc_ids": ["stats_pvalue"],
        },
    ]

    with corpus_path.open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with queries_path.open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme.write_text(
        "# RAG benchmark scaffold (Proof ③)\n\n"
        "This seed corpus/query set is for wiring Recall@1/3/5 + MRR evaluation.\n"
        "It is **not** automatically a journal-scale benchmark.\n\n"
        "Expand `corpus.jsonl` and `queries.jsonl`, then run:\n\n"
        "```bash\n"
        "python scripts/proof3_rag_retrieval_eval.py "
        "--run-benchmark data/rag_benchmark/queries.jsonl "
        "--corpus data/rag_benchmark/corpus.jsonl\n"
        "```\n\n"
        "Only mark `paper_ready=true` after a held-out, adequately sized corpus "
        "is locked and hashes recorded.\n",
        encoding="utf-8",
    )
    return {
        "status": "SCAFFOLD_WRITTEN",
        "corpus": str(corpus_path.relative_to(ROOT)).replace("\\", "/"),
        "queries": str(queries_path.relative_to(ROOT)).replace("\\", "/"),
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "paper_ready": False,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def _reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / float(i)
    return 0.0


def _recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return float(len(top & relevant) / len(relevant))


def run_benchmark(
    queries_path: Path,
    corpus_path: Path | None,
    top_k: int = 10,
) -> dict[str, Any]:
    queries = _load_jsonl(queries_path)
    shared_corpus: list[dict[str, Any]] = []
    if corpus_path is not None:
        shared_corpus = _load_jsonl(corpus_path)

    try:
        from retriever import PrivacyRetriever
    except Exception as e:
        return {
            "status": "FAILED",
            "paper_ready": False,
            "reason": f"Could not import PrivacyRetriever: {e}",
        }

    # Build default index from shared corpus if present
    doc_id_to_text: dict[str, str] = {
        str(d["doc_id"]): str(d["text"]) for d in shared_corpus if "doc_id" in d and "text" in d
    }

    per_query = []
    r1s, r3s, r5s, mrrs = [], [], [], []

    for q in queries:
        qid = str(q.get("query_id") or q.get("id") or "")
        query = str(q.get("query") or "").strip()
        relevant = {str(x) for x in (q.get("relevant_doc_ids") or [])}
        if not query or not relevant:
            raise ValueError(f"Query {qid!r} missing query text or relevant_doc_ids")

        local_corpus = q.get("corpus")
        if local_corpus:
            local_map = {
                str(d["doc_id"]): str(d["text"])
                for d in local_corpus
                if "doc_id" in d and "text" in d
            }
        else:
            local_map = doc_id_to_text

        if not local_map:
            raise ValueError(
                f"No corpus available for query {qid!r}. "
                "Provide --corpus or per-query corpus field."
            )

        # Preserve stable ordering
        ids = list(local_map.keys())
        texts = [local_map[i] for i in ids]

        retriever = PrivacyRetriever(lambda_privacy=0.0)
        retriever.build_index(texts)
        hits = retriever.retrieve(query, top_k=min(top_k, len(texts)))
        ranked_ids = []
        for h in hits:
            # retrieve returns doc_id as integer index into documents
            idx = int(getattr(h, "doc_id", -1))
            if 0 <= idx < len(ids):
                ranked_ids.append(ids[idx])

        r1 = _recall_at_k(ranked_ids, relevant, 1)
        r3 = _recall_at_k(ranked_ids, relevant, 3)
        r5 = _recall_at_k(ranked_ids, relevant, 5)
        mrr = _reciprocal_rank(ranked_ids, relevant)
        r1s.append(r1)
        r3s.append(r3)
        r5s.append(r5)
        mrrs.append(mrr)
        per_query.append(
            {
                "query_id": qid,
                "query": query,
                "relevant_doc_ids": sorted(relevant),
                "ranked_doc_ids": ranked_ids,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "mrr": mrr,
            }
        )

    n = len(per_query)
    # Heuristic: paper_ready only if expanded beyond seed scaffold size
    paper_ready = n >= 50 and len(doc_id_to_text) >= 50
    return {
        "status": "EXECUTED",
        "proof": "③ RAG retrieval",
        "timestamp_utc": utc_now(),
        "queries_path": str(queries_path.relative_to(ROOT)).replace("\\", "/")
        if queries_path.is_relative_to(ROOT)
        else str(queries_path),
        "corpus_path": (
            str(corpus_path.relative_to(ROOT)).replace("\\", "/")
            if corpus_path and corpus_path.is_relative_to(ROOT)
            else (str(corpus_path) if corpus_path else None)
        ),
        "n_queries": n,
        "n_corpus_docs": len(doc_id_to_text),
        "metrics": {
            "Recall@1": _mean(r1s),
            "Recall@3": _mean(r3s),
            "Recall@5": _mean(r5s),
            "MRR": _mean(mrrs),
        },
        "paper_ready": paper_ready,
        "paper_ready_rule": "paper_ready=true only if n_queries>=50 and n_corpus_docs>=50",
        "per_query": per_query,
    }


def _write_outputs(payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "proof3_rag_retrieval_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md = OUT_DIR / "proof3_rag_retrieval_report.md"
    lines = [
        "# Proof ③ — RAG retrieval report",
        "",
        f"- status: `{payload.get('status')}`",
        f"- paper_ready: `{payload.get('paper_ready')}`",
        f"- timestamp: `{payload.get('timestamp_utc', '')}`",
        "",
    ]
    metrics = payload.get("metrics") or {}
    if metrics:
        lines.append("## Metrics")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
        lines.append("")
    avail = payload.get("available") or {}
    if avail:
        lines.append("## Available from smoke artifact")
        for k, v in avail.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    missing = payload.get("missing_for_proof") or []
    if missing:
        lines.append("## Missing for paper proof")
        for m in missing:
            lines.append(f"- {m}")
        lines.append("")
    if payload.get("interpretation"):
        lines.append("## Interpretation")
        lines.append(str(payload["interpretation"]))
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Proof ③ RAG Recall@1/3/5 + MRR")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--write-scaffold", action="store_true")
    parser.add_argument("--run-benchmark", type=str, default="")
    parser.add_argument("--corpus", type=str, default=str(SCAFFOLD_DIR / "corpus.jsonl"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if not any([args.summarize_existing, args.write_scaffold, args.run_benchmark]):
        parser.print_help()
        raise SystemExit(2)

    payload: dict[str, Any] = {}
    if args.write_scaffold:
        payload = write_scaffold()
        print(json.dumps(payload, indent=2))
    if args.summarize_existing:
        payload = summarize_existing()
        out = _write_outputs(payload)
        print(json.dumps({k: v for k, v in payload.items() if k != "per_query"}, indent=2))
        print(f"[proof3] wrote {out}")
    if args.run_benchmark:
        qpath = Path(args.run_benchmark)
        if not qpath.is_file():
            qpath = ROOT / args.run_benchmark
        cpath = Path(args.corpus) if args.corpus else None
        if cpath and not cpath.is_file():
            cpath = ROOT / args.corpus if args.corpus else None
        if cpath and not cpath.is_file():
            cpath = None
        payload = run_benchmark(qpath, cpath, top_k=args.top_k)
        out = _write_outputs(payload)
        slim = {k: v for k, v in payload.items() if k != "per_query"}
        print(json.dumps(slim, indent=2))
        print(f"[proof3] wrote {out}")
        if payload.get("status") == "FAILED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
