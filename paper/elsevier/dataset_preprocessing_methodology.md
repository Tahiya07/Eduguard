# EduGuard Datasets: Preprocessing Methodology and Limitations

*Draft section text for the Elsevier manuscript (`manuscript.tex`). Numbers are taken from the dataset manifests and audit reports in the repository (paths in Section 7). Items in square brackets are for the authors to complete before submission.*

---

## 1. Overview

EduGuard is trained as a single multitask LoRA adapter on Qwen2.5-1.5B-Instruct [`qwen2024technical`; `hu2022lora`] for three academic tasks:

1. **Bloom rewriting.** Rewrite a source question so that it targets a specified level of the revised Bloom taxonomy [`bloom1956taxonomy`; `anderson2001revision`].
2. **Closed-book academic question answering (QA).** Answer a university-level question without a context passage.
3. **Academic summarization.** Summarize an academic or lecture-style passage.

Each task has its own source corpus and preprocessing pipeline. The three are combined only at the formatting stage (Section 5) into a task-separated dataset with one directory per task and split. The final training release is `multitask_bloom_rewrite_separate_eduguard_v3_3`.

**Table 1.** Final split sizes (rows).

| Task | Train | Validation | Test | Source |
|---|---:|---:|---:|---|
| Bloom rewriting | 2,345 | 295 | 280 | EduGuard Bloom v3.1 (600 authored examples) |
| Closed-book QA | 1,172 | 146 | 147 | EduGuard QA v2 (1,600 authored items) |
| Summarization | 592 | 64 | 57 | EduGuard Summarization v1 (713 records) |
| **Total** | **4,109** | **505** | **484** | |

---

## 2. Bloom rewriting dataset (EduGuard Bloom v3.1)

### 2.1 Source

The source is a set of 600 university-level questions from [N] disciplines. Each question carries a gold Bloom level and five transformations, one for each of the other five levels, giving 3,000 transformations. [Authors: describe who wrote the original 600 examples, how, and from which courses.]

### 2.2 Correction and repair

The corpus was corrected in successive passes. The original master files were never modified; every change was recorded field by field in a correction log.

- **Structural and semantic correction (passes 1–2).** All 600 examples and 3,000 transformations were re-read.
  - Six source labels were corrected:
    - Two Remember → Understand: classification tasks.
    - Four Create → Apply: standard derivations and proofs, labelled consistently with each other.
  - 713 transformation fields were revised. The main reasons:
    - 150 rows referred to "supplied" material that did not exist; they were made self-contained.
    - 127 Evaluate rows did not state the object or criterion being judged.
    - 56 Apply rows required only recognition.
    - 20 Create rows restated a concept instead of asking for a construction.
  - Every numeric item was recalculated. 93 items needed no change to their wording. The technical and arithmetic errors that were found were corrected, for example a mis-rounded discharge value and a mis-identified optimum path. Every new value introduced was confirmed by an independent script.
- **Repair (v3.0–v3.1).** All 586 eligible examples were reviewed and 1,408 of the 2,930 transformations were rewritten.
  - The rewriting rule assigns a level by the cognitive operation the learner must perform (recall; interpret/classify/compare/explain; execute; differentiate/organize/attribute; check/critique; generate/plan/produce), not by the leading verb.
  - Imperative and interrogative forms are allowed at every level, so that neither a verb nor a question mark acts as a label cue.
  - Content constraints: no new numbers or facts are introduced; source questions stay identical to the previous version; every value in a deliberately false claim is preserved.
- **Exclusions.** 14 examples were excluded (exclusion register), plus one row removed after a relabel. That leaves 586 examples × 5 target levels = 2,930 rows.

### 2.3 Splitting and leakage control

- **Pinned, grouped splits.** Splits are pinned to a fixed reference ID list and made at the level of source-question groups: train 469 examples / 323 groups; validation 59 / 40; test 58 / 42. No example, group or identical text appears in more than one split.
- **Contested labels.** Ten test rows from two examples whose source label is contested (Analyze vs. Apply) are excluded from training and scoring. This is why the test split has 280 rows instead of 290.
- **Challenge set.** A held-out challenge set was built from the test split: 348 items and 406 same-content adjacent-level pairs, disjoint from train and validation by ID, group and text. It contains slices targeting shortcut learning: misleading verbs, verbs used at several levels, and rare verbs.

### 2.4 Shortcut diagnostics

Simple predictors were fitted on train and scored on validation + test (Table 2). Their accuracy is reported to show how much of the label can be recovered from surface cues alone.

**Table 2.** Surface-cue predictability of the target level.

| Predictor | v2 | v3.1 |
|---|---:|---:|
| Leading verb only | 0.974 | 0.930 |
| Naive Bayes, verb masked | 0.749 | 0.716 |
| H(level \| verb), bits | 0.078 | 0.166 |

---

## 3. Closed-book QA dataset (EduGuard QA v2)

### 3.1 Source

1,600 authored question–answer items, each with:
- discipline, course and topic
- QA type (e.g. conceptual, application, comparative, calculation)
- difficulty
- a reference answer and key points

[Authors: describe how the items were authored.] The task is closed-book: the model sees only the question and must produce the reference answer.

### 3.2 Deduplication and leakage-safe splitting

Question similarity was defined as the maximum of two cosine similarities on TF-IDF vectors of the question text: character 3–5-grams (word-boundary aware) and word unigrams.

1. **Duplicates and redundancy.** Removed to a reserve set of 135 items:
   - 9 exact duplicate questions
   - 54 redundant near-twins (similarity ≥ 0.80)
   - 71 of the most redundant remaining items
   - 1 item whose reference answer was judged unsound
2. **Leakage groups.** Items with similarity ≥ 0.55 were joined transitively into 1,102 groups. Each group was assigned to a single split.
3. **Stratification.** Splits (≈80/10/10) were stratified by discipline group with a fixed seed (20261126). The final counts are 1,172 / 146 / 147.

The resulting maximum cross-split question similarity is 0.540 (train/validation), 0.550 (train/test) and 0.446 (validation/test).

### 3.3 Answer review

- **Held-out items.** Every validation and test item (293) received a model-assisted answer review. The full question and reference answer were read, calculations were recomputed, and claims were checked against standard textbook knowledge. All retained held-out items were judged correct.
- **Training items.** Of the 1,172 training items, 158 were reviewed (calculation and numerical items) and 1,014 were not.

### 3.4 Composition

The discipline mix was kept as authored. Computing accounts for 45.4% of the 1,465 retained items. The other eleven discipline groups each account for 2.0–8.1%.

---

## 4. Summarization dataset (EduGuard Summarization v1)

### 4.1 Design

Records are grouped into four length categories by the whitespace-token word count of the source passage:
- short: 300–600 words
- medium: 601–1,200 words
- long: 1,201–2,500 words
- very long: 2,501–4,000 words

Each record has an ID, department, course, topic, source text, reference summary, source type and review status.

### 4.2 Sources

**Table 3.** Composition by source type (all splits).

| Source type | Records | Median source words | Median summary words | Median compression |
|---|---:|---:|---:|---:|
| Original academic examples | 444 | 339 | 69 | 0.222 |
| arXiv scientific articles | 240 | 2,923 | 116 | 0.041 |
| Lecture-slide-style texts | 29 | 917 | 121 | 0.129 |
| **Total** | **713** | | | |

- **Original academic examples (444).**
  - 293 Short records (UAP-SUM-0001..0293) were provided by the authors. [Authors: describe their origin.]
  - 151 Medium records were written with model assistance to a multi-department topic plan. Across all sources the dataset covers 39 department labels, including the "Unspecified (arXiv)" label.
  - All 444 are marked `generated_unverified`.
- **Lecture-slide-style texts (29).** Contributed by the authors. Of 210 candidates, 29 were retained; the rest were dropped, mainly as duplicates or near-duplicates of existing records.
- **arXiv articles (240).** Drawn from the arXiv long-document summarization corpus of Cohan et al. (2018), in the `ccdv/arxiv-summarization` distribution: the article body is the source and the author abstract is the summary.
  - Only rows that had passed an earlier abstract-quality filter were eligible. That filter removed rows whose abstract contained unrecoverable `@xmath` placeholders, LaTeX residue or incomplete endings.
  - Of 390 eligible rows, 240 fell within the length ranges and were merged (medium 15, long 63, very long 162). 150 were rejected on length.
  - The original arXiv split of each paper was preserved; 9 surplus validation papers were moved to train.
  - No department or course was invented for these rows.
- **LecSumm (0).** The LecSumm corpus [cite] was examined and contributed no records:
  - It contains only 10 distinct lecture notes.
  - Eight exceed the 4,000-word ceiling, and source trimming was not permitted.
  - The remaining two are scraped course pages with embedded code.
  - The repository carries no licence.

### 4.3 Splitting and integrity checks

**Table 4.** Summarization split counts by category.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| Short | 235 | 29 | 29 |
| Medium | 188 | 4 | 3 |
| Long | 46 | 9 | 8 |
| Very long | 123 | 22 | 17 |
| **Total** | **592** | **64** | **57** |

Automated checks, all passed:
- exact schema keys
- unique IDs
- no exact duplicate sources or summaries
- no cross-split near-duplicates, using the same similarity measure as Section 3.2 with threshold 0.50

The observed maximum cross-split source similarity is 0.48.

### 4.4 Deviation from the planned design

The planned dataset had 1,465 records (Short 293, Medium 440, Long 513, Very long 219). Generation was stopped at 713 records, and the dataset was frozen as v1 in preference to adding further unverified generated text. The Long category is the most under-filled: 63 of 513 planned.

---

## 5. Multitask formatting and training interface

- **Prompt format.** Each record is rendered with the Qwen2.5 ChatML chat template as a system prompt, a user turn and an assistant turn. Each task, and each summarization format (lecture vs. article), has its own fixed system prompt.
  - Bloom rewriting: the user turn gives the source question, its level and the target level.
  - QA: the user turn gives the question only (closed-book).
  - Summarization: the user turn gives the source passage.
- **Loss masking.** Loss is computed only on assistant tokens; the prompt is masked. The tokenized prompt is required to be an exact prefix of the tokenized full example. Any row failing this check aborts preprocessing instead of being trained on silently. 5,098 masking checks passed.
- **Sequence length.** The maximum length is 8,192 tokens. No row is truncated.

  | Split | Median tokens | 99th percentile | Maximum |
  |---|---:|---:|---:|
  | Train | 979 | 5,284 | 7,217 |
  | Validation | 976 | 5,004 | 5,881 |
  | Test | 976 | 5,088 | 6,413 |

- **Task sampling.** The natural training mix is 57.1% Bloom, 28.5% QA and 14.4% summarization. Training samples tasks with probabilities 0.50 / 0.25 / 0.25, with replacement, so each summarization record is seen about 1.7 times per epoch. Validation and test sets are not resampled.
- **Integrity.** A manifest records SHA-256 hashes of every split file and of each task's upstream source. Training aborts on any count, hash, schema or leakage mismatch, and a full dataset QC is rerun before each training run.

---

## 6. Limitations

1. **Limited human verification.**
   - Bloom: corrections and rewrites were model-assisted (Claude), and checks were automated. No human or subject-matter expert has verified them. The release is a finalized candidate. Acceptance requires (a) expert resolution of two contested source labels, and (b) a human spot-check of about 10% of rewritten rows.
   - QA: reference answers were reviewed by a model, not by experts, and 1,014 training answers were not reviewed at all.
   - Summarization: 444 of 713 reference summaries were written by or with a model. No summary has been checked for factual consistency with its source, automatically or by humans.
2. **Model-generated content and circularity.** Part of the training targets was produced with a large language model. A smaller model fine-tuned on them may inherit that model's style and errors. Evaluating such a model against references from the same process can overstate quality. This is the main reason for reporting results on the held-out challenge set and on independent data.
3. **Surface cues in Bloom labels.** After repair, the leading verb still predicts the target level for 93% of held-out rows, because teachers conventionally phrase tasks this way. High overall accuracy therefore does not show that a model has learned the cognitive operations. The challenge slices are designed to test this, but they are small (10–47 items), so differences of about ±10 points are within noise.
4. **Disciplinary skew.** QA is computing-heavy (45%). Many disciplines have fewer than ten items, so their held-out quota is zero or one, and per-discipline results are not reliable. The Bloom and summarization corpora have similar uneven coverage.
5. **Summarization coverage and heterogeneity.**
   - The dataset is half its planned size.
   - Medium and Long are under-represented in validation and test (Medium 4 and 3, Long 9 and 8 records), so per-category scores there are unreliable.
   - The three source types differ systematically in register, length and compression ratio (0.04 for arXiv vs. 0.22 for authored examples), so length category and source type are confounded.
   - The arXiv sources are lowercased and tokenized and contain equation placeholders (median about 109 `@xmath` tokens per article). The model may learn this formatting for long inputs.
   - 118 Short records are below the nominal 300-word minimum, and 111 near-duplicate pairs remain within the training Short block. Both were kept as authored, and neither crosses a split.
6. **Lexical near-duplicate detection.** Leakage control uses TF-IDF similarity. Paraphrases with little word overlap may go undetected, so some semantic overlap between splits cannot be ruled out.
7. **Licensing and provenance.**
   - The arXiv articles carry individual author licences. The terms of the derived summarization corpus were not independently verified.
   - LecSumm was excluded partly because it has no licence.
   - Redistribution rights for the arXiv portion must be confirmed before public release. Alternatively, release IDs only.
8. **Language and setting.** All data are in English and reflect the curricula and conventions of [institution/country] university courses, with some local context such as currency in taka. Generalization to other languages, education systems or K–12 settings is untested.
9. **Small held-out sets.** Validation and test hold 505 and 484 rows across three tasks, which limits statistical power for task-, level- and category-level comparisons. Confidence intervals should be reported.

---

## 7. Reproducibility and data availability

| Artefact | Path |
|---|---|
| Bloom v3.1 + manifest | `data/bloom_rewrite_eduguard_v3_1/` |
| Bloom repair reports, challenge set | `final_corrected_dataset/repaired_v3_20261125/` |
| QA v2 + manifest + reserve | `data/qa_eduguard_v2/` |
| Summarization v1 + audit report + provenance logs | `data/summarization/` |
| Final multitask release + QC report | `data/multitask_bloom_rewrite_separate_eduguard_v3_3/` |
| Training config | `experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_eduguard_v3_3_20261126.json` |

[Authors: state the release location (e.g., Zenodo/Mendeley Data DOI), licence (e.g., CC BY 4.0 for authored content), and how the arXiv-derived portion is distributed.]

---

## 8. Suggested declaration (Elsevier generative-AI policy)

Elsevier requires disclosure of generative AI use. Its use in producing research data belongs in the Methods; its use in writing the manuscript belongs in the dedicated declaration. Suggested wording for Methods:

> Parts of the Bloom-rewriting and summarization datasets were created or revised with the assistance of a large language model (Claude, Anthropic). Reference answers in the QA held-out splits were reviewed with the same model. All model-assisted content was subjected to the automated checks described above; it has not been verified by human subject-matter experts, except where stated.

[Authors: adapt to what was actually done, and add the separate writing-assistance declaration if applicable.]

---

## 9. References to add to `references.bib`

```bibtex
@inproceedings{cohan2018discourse,
  author    = {Cohan, Arman and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Kim, Seokhwan and Chang, Walter and Goharian, Nazli},
  title     = {A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  pages     = {615--621},
  year      = {2018}
}
```

Already in `references.bib`: `bloom1956taxonomy`, `anderson2001revision`, `qwen2024technical`, `hu2022lora`. [Add a LecSumm citation if the exclusion is discussed in the paper.]
