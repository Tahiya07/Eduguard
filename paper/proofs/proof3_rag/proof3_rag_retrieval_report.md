# Proof ③ — RAG retrieval report

- status: `SMOKE_ONLY`
- paper_ready: `False`
- timestamp: `2026-09-21T19:06:06.829467+00:00`

## Available from smoke artifact
- hit_at_1_mean: 0.9
- hit_at_3_mean: 1.0
- note: Smoke CSV uses hit_at_k column names; treat as feasibility only.

## Missing for paper proof
- Recall@5
- MRR

## Interpretation
Existing artifact is a small academic_qa smoke set. It does NOT satisfy the paper proof requiring Recall@1/3/5 + MRR on a proper retrieval benchmark. Run --run-benchmark after filling data/rag_benchmark/.
