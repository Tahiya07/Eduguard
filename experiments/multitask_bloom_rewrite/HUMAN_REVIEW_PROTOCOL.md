# BloomShift and generated-rewrite human review protocol

## Purpose

Use human judgments to assess target Bloom alignment and content preservation. Automated lexical checks and a Bloom classifier are screening signals only; neither is ground truth.

## Unit of review

Each row contains the source question, requested target Bloom level, and one rewrite. Hide model/checker predictions and the expected label where feasible during the initial rating. For dataset quality review, reviewers should see the source question and intended target level because those define the task.

## Required independent ratings

For every item, two reviewers independently assign:

1. **Target-level alignment (1–4):** 1 = clearly wrong cognitive operation; 2 = mostly below/above target; 3 = reasonable alignment; 4 = strong alignment with the target operation.
2. **Cognitive operation demonstrated:** Remember / Understand / Apply / Analyze / Evaluate / Create / unclear.
3. **Topic and context preservation (1–4):** whether subject, entities, scenario, scope, and essential constraints are retained.
4. **Meaning preservation (1–4):** whether the original academic intent remains intact while cognitive demand changes.
5. **Question/task usability (yes/no):** one understandable student-facing exam question or imperative, not an answer or meta response.
6. **Pedagogical quality (1–4):** clarity, answerability, and appropriateness for the intended level.
7. **Decision:** accept / revise / reject, plus a short reason for revise/reject.

## Bloom level operational definitions

- **Remember:** retrieve or recognize information.
- **Understand:** explain or interpret meaning.
- **Apply:** use knowledge or a procedure in a concrete case.
- **Analyze:** examine parts, relationships, evidence, assumptions, or causes.
- **Evaluate:** make and justify a judgment using criteria/evidence.
- **Create:** produce a coherent new design, plan, model, or solution under the source constraints.

Do not classify by the opening verb alone. Ask what a student would actually need to do to produce a successful answer.

## Adjudication and agreement

- Calibrate reviewers on a separate 20-item calibration set before the main annotation.
- Review the main set independently; do not discuss individual items before submitting ratings.
- Report raw agreement and a suitable chance-corrected statistic for categorical labels (e.g. Cohen's kappa for two raters; weighted kappa for ordinal ratings). State the statistic, sample size, confidence interval if available, and how missing ratings are handled.
- Adjudicate disagreements after the independent ratings are frozen. Preserve original ratings and adjudicated labels separately.
- Do not claim human validation for rows without recorded reviewer decisions and adjudication where required.

## Dataset split policy

Only rows with explicit `review_status=human_validated` and `training_eligible=true` may enter the final supervised corpus. Missing metadata is ineligible, not implicitly approved. Keep source groups disjoint across train/validation/test. Do not tune prompts or thresholds on the locked test set.

## Generated-model evaluation

Human-evaluate a predeclared sample from the locked test set. Report classifier agreement separately from human target-level accuracy. Treat lexical overlap as lexical overlap, not semantic similarity. Report automated acceptance as an automated proxy, never as a human-validated rewrite rate.
