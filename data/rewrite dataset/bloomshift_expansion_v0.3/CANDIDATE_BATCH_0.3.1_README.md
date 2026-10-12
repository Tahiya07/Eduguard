# BloomShift expansion candidate batch 0.3.1

This file contains a manually drafted candidate batch based on source questions already present in `source_candidates_for_rewrite.csv`.

- Candidate records: 25
- Human validation: pending for every record
- Training eligibility: false for every record
- Some transformations are marked `conditional` and carry explicit semantic/context risks.
- This is a candidate batch, not a replacement for the existing train/validation/test sets and not a complete expansion of the 2,007-source pool.
- Do not report these candidates as human-validated or gold-standard data. Before training, review source fidelity, target Bloom alignment, feasibility, duplicates, and grouped splitting.

Every record retains the original source question and group identifier. The opening verb is included for auditing but does not itself prove cognitive-level alignment.
