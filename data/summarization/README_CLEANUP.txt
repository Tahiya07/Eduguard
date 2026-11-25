EduGuard dataset structural cleanup

Changes made:
- Training split reduced from 586 records to the 64 explicitly requested first-occurrence IDs.
- 522 duplicate training records removed. No retained training record was regenerated or rewritten.
- Retained IDs are unique (64/64).
- Retained source_text values are unique (64/64).
- Validation (73) and test (73) records were preserved.
- Exact source_text overlap across train/validation/test: 0 pairs for each split pair in this archive.
- review_status changed from curriculum_topic_verified to curriculum_verification_pending in all splits because official UAP syllabus/lecture-slide verification was not established by the available files.
- source_type was preserved as lecture_slide_text_style; this describes style, not proof of official provenance.
- Six retained law-related records carry jurisdiction-scope notes in an added jurisdiction_note field. Their source text was not altered.

Final split sizes: train=64, validation=73, test=73; total=210.
