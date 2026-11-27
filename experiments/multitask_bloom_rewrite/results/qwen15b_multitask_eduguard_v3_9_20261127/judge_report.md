# Judge evaluation (lora, validation)

Judge: `Qwen/Qwen2.5-7B-Instruct` (4bit), greedy, fixed rubric; generated 2026-11-27T12:12:58.805200+00:00.
Accuracy here is the judge's verdict, not word overlap. Values are mean [95% bootstrap CI].
**WARNING: the judge is NOT RELIABLE for summarization** (it must accept the reference and reject a mismatched one at least 90% of the time; see below). Do not report those numbers; use a stronger judge or the human sheets.


| Task | Measure | Value |
|---|---|---:|
| Closed-book QA | Accuracy (judged correct) | 0.644 [0.569, 0.719] |
| Closed-book QA | Lenient accuracy (correct or partially correct) | 0.973 [0.945, 0.993] |
| Summarization | Faithful (no unsupported claims) | 0.966 [0.931, 0.993] |
| Summarization | Acceptable (faithful and coverage >= 3/5) | 0.966 [0.931, 0.993] |
| Summarization | Mean coverage (1-5) | 3.884 |
| Bloom rewrite | Level accuracy (judge, blind to requested level) | 0.780 [0.736, 0.824] |
| Bloom rewrite | Judge and 0.5B classifier both agree with requested level | 0.745 [0.698, 0.789] |
| Bloom rewrite | Topic kept | 0.997 [0.991, 1.000] |

QA accuracy by question type:

| QA type | N | Accuracy |
|---|---:|---:|
| analytical | 26 | 0.6538 |
| application | 31 | 0.6452 |
| calculation | 14 | 0.7143 |
| comparative | 24 | 0.625 |
| conceptual | 33 | 0.6667 |
| design | 2 | 0.5 |
| diagnosis | 4 | 0.5 |
| explanation | 3 | 0.6667 |
| process | 5 | 0.6 |
| reasoning | 2 | 0.5 |
| sequence | 2 | 0.5 |

Bloom level accuracy by requested level:

| Level | N | Model (judge) | Human reference (judge ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.6066 | 0.623 |
| Understand | 59 | 1.0 | 0.9492 |
| Apply | 52 | 0.8654 | 0.9423 |
| Analyze | 52 | 0.8846 | 0.8077 |
| Evaluate | 57 | 0.6316 | 0.6316 |
| Create | 60 | 0.7167 | 0.6833 |

## Judge reliability (references only, same for every system)

- QA (n=146): reference answer judged correct 1; another question's reference judged incorrect 1.
- Summarization (n=40): reference judged faithful 1; another section's summary judged unfaithful 0.275.
- Bloom (n=341): judge agrees with the requested level on the human reference rewrites 0.7683.

Unparseable judge replies (counted as not correct): {'qa': 0, 'summarization': 0, 'bloom_rewrite': 0}.
An LLM judge can be wrong; confirm on the human sheets (human_eval_*.csv) before final claims.
