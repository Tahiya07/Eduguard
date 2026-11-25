# EduGuard Bloom dataset v3: training-ready export

This folder has the same layout and record types as `../../training_ready/` (v2). Use it the same way: load answer tasks through `scripts/eduguard_loader.py`.

What differs from v2:
- Question text comes from the repaired master `../corrected_dataset_v3.json` (v3.1): 1,408 of 2,930 transformations differ from v2. There is no per-level verb rule; rows may be imperative or interrogative.
- Example 252's source label changed from Analyze to Understand. Its Understand row was removed and an Analyze row added. The prior label is in `source_bloom_level_before_correction`.
- New fields (true when text differs from v2): `pass3_rewritten_levels` (examples) and `pass3_rewritten` (transformation and classification records). `schema.json` describes both.
- Split, group, eligibility, answer-source, provenance and false-claim markers are copied from v2 by example ID. `tests/reference_split_ids.json` is a byte-identical copy.

Review status: corrections are model-assisted (Claude) and not human-verified; checks are automated.

Verify the export from `repaired_v3_20261125/`:

    python -I -m unittest discover -s training_ready/tests -v
    python -I training_ready/scripts/check_leakage.py

The held-out challenge set is in `../challenge_set/`, not here. Never train, tune or select checkpoints on it.
