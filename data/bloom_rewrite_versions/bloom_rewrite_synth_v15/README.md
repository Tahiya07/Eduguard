# bloom_rewrite_synth_v15

High-fidelity deterministic Bloom-target supervision built directly from data/figshare_bloom_v1.csv. Actual source task form is detected independently of the source label. Exact target matches are retained; only narrow, source-supported cross-level transformations are emitted. Unsupported transformations are excluded. No LLM is used.\n