# Judge evaluation (base, test)

Judge: `Qwen/Qwen2.5-7B-Instruct` (4bit), greedy, fixed rubric; generated 2026-11-27T20:26:22.382123+00:00.
Accuracy here is the judge's verdict, not word overlap. Values are mean [95% bootstrap CI].
**WARNING: the judge is NOT RELIABLE for summarization** (it must accept the reference and reject a mismatched one at least 90% of the time; see below). Do not report those numbers; use a stronger judge or the human sheets.


| Task | Measure | Value |
|---|---|---:|
| Closed-book QA | Accuracy (judged correct) | 0.973 [0.946, 0.993] |
| Closed-book QA | Lenient accuracy (correct or partially correct) | 0.980 [0.952, 1.000] |
| Summarization | Faithful (no unsupported claims) | 1.000 [1.000, 1.000] |
| Summarization | Acceptable (faithful and coverage >= 3/5) | 1.000 [1.000, 1.000] |
| Summarization | Mean coverage (1-5) | 4.204 |
| Bloom rewrite | Level accuracy (judge, blind to requested level) | 0.613 [0.561, 0.661] |
| Bloom rewrite | Judge and 0.5B classifier both agree with requested level | 0.524 [0.473, 0.576] |
| Bloom rewrite | Topic kept | 0.992 [0.980, 1.000] |

QA accuracy by question type:

| QA type | N | Accuracy |
|---|---:|---:|
| analytical | 25 | 1.0 |
| application | 33 | 1.0 |
| calculation | 14 | 0.9286 |
| comparative | 26 | 1.0 |
| conceptual | 31 | 0.9677 |
| design | 3 | 0.6667 |
| diagnosis | 1 | 1.0 |
| explanation | 3 | 1.0 |
| process | 4 | 1.0 |
| reasoning | 4 | 1.0 |
| sequence | 3 | 0.6667 |

Bloom level accuracy by requested level:

| Level | N | Model (judge) | Human reference (judge ceiling) |
|---|---:|---:|---:|
| Remember | 64 | 0.1406 | 0.6406 |
| Understand | 61 | 0.8852 | 0.9016 |
| Apply | 49 | 0.6122 | 0.9796 |
| Analyze | 53 | 0.2642 | 0.7736 |
| Evaluate | 62 | 0.9516 | 0.7581 |
| Create | 62 | 0.7903 | 0.5323 |

## Judge reliability (references only, same for every system)

- QA (n=147): reference answer judged correct 1; another question's reference judged incorrect 0.9728.
- Summarization (n=40): reference judged faithful 1; another section's summary judged unfaithful 0.25.
- Bloom (n=351): judge agrees with the requested level on the human reference rewrites 0.755.

Unparseable judge replies (counted as not correct): {'qa': 0, 'summarization': 0, 'bloom_rewrite': 0}.
An LLM judge can be wrong; confirm on the human sheets (human_eval_*.csv) before final claims.
