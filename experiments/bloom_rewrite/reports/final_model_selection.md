# Final model selection

**NO MODEL WINNER — TRAINING/EVALUATION NOT COMPLETED.**

## Training / evaluation task (locked)

```
TRAINING TASK:     question + target_level → rewrite
SOURCE LEVEL:      metadata only
TARGET LEVEL:      generator input
EVALUATION TASK:   question + target_level → rewrite
```

### Format change

| | Format |
|---|---|
| **BEFORE (v1)** | `question + source Bloom + target Bloom → rewrite` |
| **AFTER (v2)** | `question + target Bloom → rewrite` |

**Why:** EduGuard production lets the user select only a target Bloom level. The generator must not require the original Bloom label. The classifier remains a separate validator.

## Dataset

| Field | Value |
|---|---|
| Version | `bloom_rewrite_synth_v2` |
| Hash | `b3725b77862868dcd3d7ad07f1d2e15ae41d6d9887e8510d5396de8c4e790bae` |
| Seed | 42 |
| Train / val / test | 6975 / 1491 / 1536 |
| Total | 10002 |
| Missing source→target cells | none (all 30 present) |
| Leakage | train∩val∩test = 0 (normalized, groups, source_ids) |
| Source Bloom in SFT prompts | **0** leaked records |

v1 archived at `data/bloom_rewrite_versions/bloom_rewrite_synth_v1/`.

## Sample SFT user message

```text
Original question:
Explain what virtual memory is.

Target Bloom level:
Analyze
```

(No `Original Bloom level:` / `Source Bloom level:` lines.)

## Configs

Both `qwen05b_lora.json` and `qwen15b_lora.json` use the same 2-input task via `prompt_format.py`.

## Production alignment

Live `build_targeted_rewrite_prompt()` is also 2-input (question + target). It additionally injects per-level task policy and examples; that wrapper difference is documented and does not change the trained function.

## Training readiness

Scripts and dataset are ready for 2-input SFT.

## Resource readiness

**TRAINING NOT STARTED — insufficient resources** (≈7.72 GB RAM, no CUDA). Do not train on this machine.

## Decision rule

Locked in `configs/decision_rule.json` before results: quality first, then cost; do not pick by size alone.
