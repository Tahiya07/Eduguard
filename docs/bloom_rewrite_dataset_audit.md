# Bloom target-rewrite dataset audit

**Research question:** which model, Qwen2.5-0.5B-Instruct or Qwen2.5-1.5B-Instruct, provides the best trade-off between Bloom *target-level rewrite* quality and deployment cost for EduGuard?

**Audit date:** 2026-08-26  
**Decision:** no public dataset provides genuine `source_question → target_level → target_rewrite` supervision. Strategy **B** is used: Figshare Bloom-labelled academic questions as a source pool, plus a deterministic transformation framework. Candidate Qwen models were **not** used to generate labels.

A dataset is accepted only if it can supervise **target-level transformation**, not merely Bloom classification or passage-to-question generation.

---

## Local Figshare inspection (confirmed present)

| Field | Finding |
|---|---|
| File | `data/figshare_bloom_v1.csv` |
| Rows | 2330 |
| Columns | `question`, `original_label`, `bloom_level` |
| Labels | Remember 321, Understand 915, Apply 285, Analyze 274, Evaluate 284, Create 251 |
| Original taxonomy | Knowledge / Comprehension / Application / Analysis / Evaluation / Synthesis |
| Exact duplicate questions | 0 |
| Normalized near-exact duplicates | 5 |
| Rewrite-pair columns | **none** |
| Dataset type | **CLASSIFICATION ONLY** |
| Can supervise target rewriting by itself | **No** |

`data/figshare_combined_dataset.csv` (2522 rows, columns `QUESTION`, `BT LEVEL`) is the same Figshare family with original Bloom names. It also has **no rewrite pairs**.

Existing `figshare_bloom_v1_{train,val,test}.csv` splits are **classifier** splits and were **not** reused. They have small normalized overlaps (train–val 2, train–test 1, val–test 1).

---

## Public dataset audit

### Figshare Exam Question Datasets (Gani & Sangodiah)

| Field | Value |
|---|---|
| Dataset | Exam Question Datasets (Figshare Bloom) |
| URL/source | https://doi.org/10.6084/m9.figshare.22597957.v3 |
| License | Open Figshare deposit used in published academic work; local copy already in-repo |
| Number of examples | 2330 mapped rows locally (`figshare_bloom_v1.csv`); 2522 in combined file |
| Bloom coverage | C1–C6 after mapping Knowledge→Remember and Synthesis→Create |
| Transformation-pair availability | **None** |
| Suitability | Source-question pool, labels, stratification, classifier evaluation |
| Problems | Classification only; Understand-heavy; incomplete template stems; duplicates/near-duplicates |
| Final decision | **Use as source pool only. Do not treat as rewrite supervision.** |

### CogBench (Kunuku / mouryat9)

| Field | Value |
|---|---|
| Dataset | mouryat9/CogBench |
| URL/source | https://huggingface.co/datasets/mouryat9/CogBench |
| License | CC-BY-4.0 |
| Number of examples | 27,099 questions |
| Bloom coverage | C1–C6 |
| Transformation-pair availability | **None**. Fields are `question_text`, `bloom_level`, `bloom_name`. Supported tasks are classification and *passage/level-conditioned question generation*, not rewriting an existing question to a requested target level. |
| Suitability | Classifier pretraining / QG-from-passage evaluation — **not** EduGuard rewrite SFT |
| Problems | Silver labels (~82% CCS accuracy) on most rows; no source→target pairs |
| Final decision | **Reject as rewrite supervision.** |

### CogBench (verifiable constraint benchmark)

| Field | Value |
|---|---|
| Dataset | cogbench/cogbench |
| URL/source | https://github.com/cogbench/cogbench |
| License | Project/benchmark license; 120 OpenStax passages |
| Number of examples | 120 passages, not rewrite pairs |
| Bloom coverage | C1–C6 as *generation targets from a passage* |
| Transformation-pair availability | **None**. Input is a passage, not an existing exam question. |
| Suitability | Constraint-based QG evaluation |
| Problems | Wrong task; too small; not source→target rewriting |
| Final decision | **Reject.** |

### CogBench (ACL educational QA cognitive alignment)

| Field | Value |
|---|---|
| Dataset | CogBench (Findings of ACL 2026, math cognitive alignment) |
| URL/source | https://aclanthology.org/2026.findings-acl.1068/ |
| License | ACL anthology paper resource |
| Number of examples | ~2.1K mathematics questions with multi-level *solutions/explanations* |
| Bloom coverage | Cognitive-level *explanations of answers*, not exam-question rewrites |
| Transformation-pair availability | Multiple solutions per item, **not** rewritten exam prompts |
| Suitability | Different task (answer explanation alignment) |
| Problems | Not Bloom exam-question transformation; not EduGuard’s student-facing rewrite job |
| Final decision | **Reject.** The name “CogBench” is overloaded; this is not a rewrite dataset. |

### CogBench (KwaiKEG cognitive dynamics)

| Field | Value |
|---|---|
| Dataset | kwaikeg/CogBench |
| URL/source | https://huggingface.co/datasets/kwaikeg/CogBench |
| License | See Hugging Face card |
| Number of examples | Attitude questionnaires over information-flow iterations |
| Bloom coverage | None |
| Transformation-pair availability | None |
| Suitability | Unrelated (cognitive dynamics / Likert attitudes) |
| Problems | Not educational exam rewriting |
| Final decision | **Reject.** |

### EduQuest

| Field | Value |
|---|---|
| Dataset | EduQuest |
| URL/source | https://github.com/aegonwolf/EduQuest |
| License | CC BY-NC-ND 4.0 (restrictive; no derivatives) |
| Number of examples | 68,248 questions + lecture texts |
| Bloom coverage | Cognitive-process and knowledge-dimension labels; not exclusive C1–C6 rewrite pairs |
| Transformation-pair availability | **None** (lecture↔question, not source-level→target-level rewrite) |
| Suitability | Question generation from lecture text |
| Problems | No transformation pairs; ND license is a poor fit for derived synthetic rewrite corpora |
| Final decision | **Reject as rewrite supervision.** |

### BloomXplain

| Field | Value |
|---|---|
| Dataset | BloomXplain |
| URL/source | https://zenodo.org/records/18336597 |
| License | Research dump / code on GitHub |
| Number of examples | STEM question–answer pairs with Bloom categories |
| Bloom coverage | C1–C6 for *explanations*, not question rewrites |
| Transformation-pair availability | None for exam-question transformation |
| Suitability | Explanation generation |
| Problems | Wrong output type (explanations vs student-facing rewritten questions) |
| Final decision | **Reject.** |

### BloomVQA

| Field | Value |
|---|---|
| Dataset | ygong/BloomVQA |
| URL/source | https://huggingface.co/datasets/ygong/BloomVQA |
| License | Research dataset (children’s picture stories) |
| Number of examples | 1,200 MC items / 20 stories |
| Bloom coverage | Comprehension levels on images |
| Transformation-pair availability | None |
| Suitability | Vision-language, early-childhood, not university exam rewriting |
| Problems | Wrong modality, wrong age, no rewrite pairs |
| Final decision | **Reject.** |

### Local `obe_dataset.csv`

| Field | Value |
|---|---|
| Dataset | `data/obe_dataset.csv` |
| URL/source | Local file |
| License | Unknown |
| Number of examples | ~100k rows |
| Bloom coverage | Column exists, but labels are inconsistent with the question text (e.g. “What is Continuity…?” labelled Understand; “Explain Eigenvectors briefly.” labelled Evaluate) |
| Transformation-pair availability | None |
| Suitability | Not trustworthy Bloom supervision |
| Problems | Label–text mismatch; mixed languages; looks synthetic/noisy |
| Final decision | **Reject.** |

### Local unlabelled question CSVs

| Field | Value |
|---|---|
| Dataset | `data/Data_Structure.csv`, `data/Introduction_to_Computers_and_Research.csv`, `data/Irrelevant_Questions.csv` |
| URL/source | Local |
| License | Unknown |
| Number of examples | Topic/score/question only |
| Bloom coverage | **No Bloom labels** |
| Transformation-pair availability | None |
| Suitability | Not usable for this experiment |
| Problems | No cognitive-level annotation |
| Final decision | **Reject.** |

---

## Strategy selected

**A (public transformation pairs):** not available after this audit.

**B (Bloom-labelled public questions + controlled synthetic transformation):** selected.

Synthetic targets are produced by `experiments/bloom_rewrite/bloom_target_policy.py`, not by Qwen2.5-0.5B or Qwen2.5-1.5B. That avoids circular training-on-own-outputs.

**Important limitation:** these are deterministic policy templates, not human-authored gold rewrites. They change the required student *task* (recall vs scenario vs analysis vs criteria-based judgment vs novel design). They should not be described as naturally occurring exam pairs. Human evaluation remains required before any production claim.
