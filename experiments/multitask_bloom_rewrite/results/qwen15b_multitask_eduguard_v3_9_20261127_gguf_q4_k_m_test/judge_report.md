# Judge evaluation (gguf, test)

Judge: `Qwen/Qwen2.5-7B-Instruct` (4bit), greedy, fixed rubric; generated 2026-11-28T01:45:53.534374+00:00.
Accuracy here is the judge's verdict, not word overlap. Values are mean [95% bootstrap CI].
**WARNING: the judge is NOT RELIABLE for summarization** (it must accept the reference and reject a mismatched one at least 90% of the time; see below). Do not report those numbers; use a stronger judge or the human sheets.


| Task | Measure | Value |
|---|---|---:|
| Closed-book QA | Accuracy (judged correct) | 0.660 [0.578, 0.735] |
| Closed-book QA | Lenient accuracy (correct or partially correct) | 0.932 [0.891, 0.966] |
| Summarization | Faithful (no unsupported claims) | 0.912 [0.864, 0.952] |
| Summarization | Acceptable (faithful and coverage >= 3/5) | 0.912 [0.864, 0.952] |
| Summarization | Mean coverage (1-5) | 3.782 |
| Bloom rewrite | Level accuracy (judge, blind to requested level) | 0.778 [0.735, 0.821] |
| Bloom rewrite | Judge and 0.5B classifier both agree with requested level | 0.735 [0.690, 0.781] |
| Bloom rewrite | Topic kept | 1.000 [1.000, 1.000] |

QA accuracy by question type:

| QA type | N | Accuracy |
|---|---:|---:|
| analytical | 25 | 0.68 |
| application | 33 | 0.697 |
| calculation | 14 | 0.6429 |
| comparative | 26 | 0.6154 |
| conceptual | 31 | 0.5806 |
| design | 3 | 0.6667 |
| diagnosis | 1 | 1.0 |
| explanation | 3 | 1.0 |
| process | 4 | 0.5 |
| reasoning | 4 | 0.75 |
| sequence | 3 | 1.0 |

Bloom level accuracy by requested level:

| Level | N | Model (judge) | Human reference (judge ceiling) |
|---|---:|---:|---:|
| Remember | 64 | 0.4531 | 0.625 |
| Understand | 61 | 0.9508 | 0.9016 |
| Apply | 49 | 1.0 | 0.9796 |
| Analyze | 53 | 0.8491 | 0.7736 |
| Evaluate | 62 | 0.8387 | 0.7258 |
| Create | 62 | 0.6452 | 0.5161 |

## Judge reliability (references only, same for every system)

- QA (n=147): reference answer judged correct 1; another question's reference judged incorrect 0.9728.
- Summarization (n=40): reference judged faithful 1; another section's summary judged unfaithful 0.225.
- Bloom (n=351): judge agrees with the requested level on the human reference rewrites 0.7436.

Unparseable judge replies (counted as not correct): {'qa': 0, 'summarization': 0, 'bloom_rewrite': 0}.
An LLM judge can be wrong; confirm on the human sheets (human_eval_*.csv) before final claims.
