# EduGuard Final Training Readiness

Date: 2026-10-23
Checkpoint: `d495644` (`main` matched `origin/main`)

## Verdict: NOT READY on the current host

Dataset, configuration, tokenizer, loader, and integrity validation pass. The current host is CPU-only with 7.72 GB total RAM and 1.0 GB available, below the active Qwen2.5-1.5B LoRA estimate of 28 GB total and 22 GB available. No training was started. A suitable GPU or higher-memory host is required.

## Dataset and semantic review

- Bloom split counts are **1,075 / 212 / 135** (train/validation/test); 1,422/1,422 canonical source mappings pass. IDs, source questions, labels, and split membership are preserved. There are no group leaks, exact input overlaps, or duplicate Bloom rewrites.
- The tokenization pass found 18 Bloom train rows whose `sft_text` system prompt had changed while the derived `prompt_text` and `generation_prompt` remained stale. I synchronized those two derived fields to each existing SFT prompt. **No question target, source question, label, ID, or split changed.** SFT prompt/target consistency now passes 4,266/4,266 task rows. Manifest and QC fingerprints were refreshed and validate.
- Reviewed all 15 remaining source-context records against the canonical split JSON, source IDs/topics/anchors, supervisor and reviewer sheets, audit queues, and matching original FigShare CSV entries. These sources contain no additional context; the prompts were retained unchanged rather than supplied with invented details.

| Records | Target levels | Missing source information | Disposition |
|---|---|---|---|
| `bloomshift_final_test_0105`–`0109` | Apply, Create, Evaluate, Remember, Understand | The word “this” has no identified subject/domain or supplied examples; canonical and CSV source repeat the same contextless sentence. | Retained; cannot ground a subject-specific rewrite. |
| `bloomshift_final_train_0503`, `_0504` | Apply, Evaluate | The terms to differentiate are absent; no list exists in canonical or original source. | Retained; terms cannot be inferred. |
| `bloomshift_final_train_0444`, `_0445` | Create, Evaluate | Genetic-algorithm problem instance and its inputs/objective/fitness/constraints are absent. | Retained; no instance can be reconstructed. |
| `exp034_figshare_row_107_evaluate` | Evaluate | No case, candidate actions, outcomes, or effectiveness objective is supplied. | Retained; no defensible comparison is possible. |
| `bloomshift_final_validation_0015`, `_0016` | Evaluate, Remember | Building specifications are referenced but absent. | Retained; no specification list or design brief exists locally. |
| `bloomshift_final_test_0042` | Remember | Analog-circuit specifications and component/behavior constraints are absent. | Retained; no circuit brief exists locally. |
| `bloomshift_final_train_0002`, `_0003` | Apply, Evaluate | Nested-loop output pattern is referenced but omitted (including in the original CSV question). | Retained; the intended output cannot be recovered. |

These 15 source limitations remain disclosed; this pass did not alter their targets or labels. Semantic review is rule-guided and single-annotator; it does not establish independent gold-label quality.

## Tokenizer, data loader, and configuration

- Loaded the exact configured tokenizer, `Qwen/Qwen2.5-1.5B-Instruct`, Hub revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Only `config.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json` were staged; **no model weights** were downloaded.
- Isolated environment: Python 3.11.9, PyTorch 2.3.1+cpu, Transformers 4.41.0, tokenizers 0.19.1, huggingface_hub 0.36.2, PEFT 0.11.1, Accelerate 0.33.0, Datasets 2.20.0, safetensors 0.4.3, sentencepiece 0.2.0. The project `.venv` was not changed.
- Existing `validate_multitask_dataset.py` passed for all **4,266 task records** at `max_seq_length=8192`: 3,225 train, 636 validation, 405 test; max sequence lengths were 8,188 / 8,161 / 8,085, with no overflow, no prompt consuming the target, and 4,266 valid assistant-only loss masks.
- Active `qwen15b_multitask_separate_20261016.json` and task-separated manifest validation pass. Six isolated task-loader/config tests pass.

## Tests and remaining blocker

- Full suite after the prompt-field repair: **60 passed, 0 failures, 0 errors, 0 skipped**.
- Dataset/config integrity: PASS; the refreshed manifest and QC hashes match actual files. The tokenizer QC report records PASS.
- Current host resource assessment: no CUDA device; 7.72 GB total RAM, 1.0 GB available, 131.56 GB free disk. The repository estimate is 28 GB CPU RAM for this run, with at least 22 GB available. Disk passes; memory does not.
- **Readiness:** dataset and preflight are ready, but starting the controlled training run on this host is blocked by insufficient compute memory and no GPU. Use a suitable GPU host or a host meeting the CPU memory estimate. No model was trained, and no commit or push was made.


Final fingerprints after preserving original JSONL line formatting:
- Manifest: `298bc0a62140a679c19387dd6b86e83d4c3ff86ad15444e5283bcf145362bf01`
- QC report: `658a4d45466ad9c57786406d8e43ecf4b4f1668e2064b84f0c1a58f8bd2f6e87`
- Bloom train JSONL: `161d96669cd3d37c58e75428b083e1936f5e40b47de19aaac132c51ee215d31a`
