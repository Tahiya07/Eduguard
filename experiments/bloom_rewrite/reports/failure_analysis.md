# Failure analysis

**Status: pending.** No model outputs exist yet.

After evaluation, classify every failed test item separately for 0.5B and 1.5B:

- WRONG_TARGET_LEVEL
- DECLARATIVE_ANSWER
- TOPIC_DRIFT
- MEANING_DRIFT
- TRIVIAL_VERB_SUBSTITUTION
- FORBIDDEN_COGNITIVE_OPERATION
- MULTI_LEVEL_TASK
- META_RESPONSE
- INVALID_QUESTION
- TOO_VAGUE
- OTHER

Pay particular attention to Apply, Analyze, Evaluate, and Create. Those are the weakest levels in the current production generator and must not be hidden behind overall accuracy.

The compare script writes failure counts into `results/comparison/paired_comparison.json` once both `predictions.jsonl` files exist.
