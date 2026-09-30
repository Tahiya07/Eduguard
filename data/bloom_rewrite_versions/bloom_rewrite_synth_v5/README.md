# bloom_rewrite_synth_v5

Conservative deterministic Bloom-target rewrite supervision built directly from the authoritative Figshare corpus. No LLM generation or judging is used.

Rows are emitted only after source-anchor, target-cue, topic-recall, and unsupported-content validation. Target classes are balanced to the maximum validated count available in each split.
