# Judge evaluation (gguf, validation)

Judge: `Qwen/Qwen2.5-7B-Instruct` (4bit), greedy, fixed rubric; generated 2026-11-27T23:19:34.258307+00:00.
Accuracy here is the judge's verdict, not word overlap. Values are mean [95% bootstrap CI].
**WARNING: the judge is NOT RELIABLE for summarization** (it must accept the reference and reject a mismatched one at least 90% of the time; see below). Do not report those numbers; use a stronger judge or the human sheets.


| Task | Measure | Value |
|---|---|---:|
| Closed-book QA | Accuracy (judged correct) | 0.671 [0.596, 0.747] |
| Closed-book QA | Lenient accuracy (correct or partially correct) | 0.973 [0.945, 0.993] |
| Summarization | Faithful (no unsupported claims) | 0.959 [0.925, 0.986] |
| Summarization | Acceptable (faithful and coverage >= 3/5) | 0.959 [0.925, 0.986] |
| Summarization | Mean coverage (1-5) | 3.877 |
| Bloom rewrite | Level accuracy (judge, blind to requested level) | 0.821 [0.780, 0.859] |
| Bloom rewrite | Judge and 0.5B classifier both agree with requested level | 0.757 [0.713, 0.801] |
| Bloom rewrite | Topic kept | 1.000 [1.000, 1.000] |

QA accuracy by question type:

| QA type | N | Accuracy |
|---|---:|---:|
| analytical | 26 | 0.6154 |
| application | 31 | 0.7419 |
| calculation | 14 | 0.8571 |
| comparative | 24 | 0.6667 |
| conceptual | 33 | 0.6364 |
| design | 2 | 0.0 |
| diagnosis | 4 | 0.5 |
| explanation | 3 | 1.0 |
| process | 5 | 0.6 |
| reasoning | 2 | 0.5 |
| sequence | 2 | 0.5 |

Bloom level accuracy by requested level:

| Level | N | Model (judge) | Human reference (judge ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.4754 | 0.623 |
| Understand | 59 | 0.9661 | 0.9492 |
| Apply | 52 | 0.9615 | 0.9423 |
| Analyze | 52 | 0.8846 | 0.8077 |
| Evaluate | 57 | 0.8421 | 0.6316 |
| Create | 60 | 0.8333 | 0.6833 |

## Judge reliability (references only, same for every system)

- QA (n=146): reference answer judged correct 1; another question's reference judged incorrect 1.
- Summarization (n=40): reference judged faithful 1; another section's summary judged unfaithful 0.275.
- Bloom (n=341): judge agrees with the requested level on the human reference rewrites 0.7683.

Unparseable judge replies (counted as not correct): {'qa': 0, 'summarization': 0, 'bloom_rewrite': 1}.
An LLM judge can be wrong; confirm on the human sheets (human_eval_*.csv) before final claims.
