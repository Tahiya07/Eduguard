# RAG benchmark scaffold (Proof ③)

This seed corpus/query set is for wiring Recall@1/3/5 + MRR evaluation.
It is **not** automatically a journal-scale benchmark.

Expand `corpus.jsonl` and `queries.jsonl`, then run:

```bash
python scripts/proof3_rag_retrieval_eval.py --run-benchmark data/rag_benchmark/queries.jsonl --corpus data/rag_benchmark/corpus.jsonl
```

Only mark `paper_ready=true` after a held-out, adequately sized corpus is locked and hashes recorded.
