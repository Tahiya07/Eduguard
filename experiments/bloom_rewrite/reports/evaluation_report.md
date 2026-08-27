# Evaluation report

**Status: pending.** Neither generator has been trained, so there are no held-out metrics.

When training completes, evaluate **both** models on the same file `data/bloom_rewrite/test.jsonl` with the **same** greedy decoding (`temperature=0`, `max_new_tokens=96`) and the **same fixed** EduGuard Bloom classifier (`predict_bloom.QwenBloomPredictor`). Do not retrain the classifier for each generator.

Primary metric: target Bloom accuracy (`predicted_level == target_level`), plus per-target counts, source×target matrix, macro accuracy, macro F1, weighted F1.

Secondary metrics (validator, not verb lookup as Bloom proof): question validity, topic preservation, cognitive-task structure, meta-language rate, declarative-answer rate, trivial transformation rate, forbidden-operation rate, empty/invalid rate.

Do not treat lexical similarity or a Bloom verb as proof of target-level correctness.
