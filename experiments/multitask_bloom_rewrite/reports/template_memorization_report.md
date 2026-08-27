# Template Memorization Report

Bloom rewrite targets are **synthetic**. This report measures template/n-gram overlap.

```json
{
  "dataset": "bloom_rewrite_synth_v2",
  "train_n": 6975,
  "test_n": 1536,
  "top_train_templates": [
    {
      "template": "assess the strengths and limitations",
      "count": 454
    },
    {
      "template": "given a concrete academic scenario",
      "count": 453
    },
    {
      "template": "propose an original academic artifact",
      "count": 439
    },
    {
      "template": "formulate a structured approach that",
      "count": 414
    },
    {
      "template": "name the essential elements involved",
      "count": 257
    },
    {
      "template": "design a new plan addressing",
      "count": 251
    },
    {
      "template": "analyze the components involved in",
      "count": 247
    },
    {
      "template": "design a new plan that",
      "count": 227
    },
    {
      "template": "use the relevant procedure for",
      "count": 224
    },
    {
      "template": "apply the appropriate method to",
      "count": 217
    },
    {
      "template": "state the main points concerning",
      "count": 206
    },
    {
      "template": "compare the internal parts of",
      "count": 201
    },
    {
      "template": "analyze how the components of",
      "count": 177
    },
    {
      "template": "identify the key components of",
      "count": 174
    },
    {
      "template": "name the primary characteristics of",
      "count": 172
    },
    {
      "template": "identify the key facts about",
      "count": 150
    },
    {
      "template": "summarize the academic idea behind",
      "count": 130
    },
    {
      "template": "describe the purpose and main",
      "count": 119
    },
    {
      "template": "describe what is meant by",
      "count": 54
    },
    {
      "template": "evaluate the effectiveness of the",
      "count": 20
    }
  ],
  "shared_leading5_templates": 128,
  "shared_leading5_templates_pct_of_test_templates": 0.2565,
  "shared_bigrams": 1973,
  "exact_rewrite_overlap_train_test": 3,
  "warning": "Bloom rewrite supervision is synthetic. High template overlap indicates risk of template memorization rather than general transformation learning."
}
```
