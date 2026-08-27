# Dataset Quality Report

Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`

## Counts
```json
{
  "train": {
    "total": 17437,
    "by_task": {
      "bloom_rewrite": 6975,
      "qa": 5231,
      "summarization": 5231
    }
  },
  "validation": {
    "total": 8276,
    "by_task": {
      "bloom_rewrite": 1491,
      "qa": 5285,
      "summarization": 1500
    }
  },
  "test": {
    "total": 8321,
    "by_task": {
      "bloom_rewrite": 1536,
      "qa": 5285,
      "summarization": 1500
    }
  }
}
```

## Leakage
```json
{
  "ok": true,
  "details": {
    "bloom_rewrite": {
      "overlaps": {
        "train\u2229validation": 0,
        "train\u2229test": 0,
        "validation\u2229test": 0
      },
      "ok": true,
      "sizes": {
        "train": 1458,
        "validation": 313,
        "test": 313
      }
    },
    "qa": {
      "overlaps": {
        "train\u2229validation": 0,
        "train\u2229test": 0,
        "validation\u2229test": 0
      },
      "ok": true,
      "sizes": {
        "train": 5231,
        "validation": 5285,
        "test": 5285
      }
    },
    "summarization": {
      "overlaps": {
        "train\u2229validation": 0,
        "train\u2229test": 0,
        "validation\u2229test": 0
      },
      "ok": true,
      "sizes": {
        "train": 5231,
        "validation": 1500,
        "test": 1500
      }
    }
  },
  "identical_sft_text_across_splits": 12,
  "identical_sft_text_note": "Identical full ChatML strings across splits are recorded for template memorization risk; source-id/group leakage is the hard stop condition."
}
```

## Task balance
```json
{
  "mix_preset": "A",
  "requested": {
    "bloom_rewrite": 0.4,
    "qa": 0.3,
    "summarization": 0.3
  },
  "actual_train_proportions": {
    "bloom_rewrite": 0.4,
    "qa": 0.3,
    "summarization": 0.3
  },
  "actual_train_counts": {
    "bloom_rewrite": 6975,
    "qa": 5231,
    "summarization": 5231
  },
  "sensitivity_presets": {
    "A": {
      "bloom_rewrite": 0.4,
      "qa": 0.3,
      "summarization": 0.3
    },
    "B": {
      "bloom_rewrite": 0.5,
      "qa": 0.25,
      "summarization": 0.25
    },
    "C": {
      "bloom_rewrite": 0.35,
      "qa": 0.325,
      "summarization": 0.325
    }
  },
  "note": "Mix selection must use validation only; Mix A is pre-registered primary."
}
```

## Bloom QC sample
```json
{
  "sampled": 300,
  "accepted": 198,
  "accept_rate": 0.66,
  "rejection_reasons": {
    "invalid_question_format;answer_or_declarative;cognitive_invalid": 96,
    "topic_drift": 2,
    "meta_response": 2,
    "meta_response;answer_or_declarative;cognitive_invalid": 2
  },
  "note": "Validation is heuristic offline QC; Bloom synth_v2 was already policy-validated upstream."
}
```

## Provenance
- Bloom: bloom_rewrite_synth_v2 (synthetic transformations)
- QA: rajpurkar/squad (SQuAD 1.1)
- Summarization: ccdv/pubmed-summarization

Bloom rewrite supervision is synthetic and must be stated as such in the paper.
