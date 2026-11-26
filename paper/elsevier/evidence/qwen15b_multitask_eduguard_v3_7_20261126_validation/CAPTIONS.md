# Draft figure and table captions

Run `qwen15b_multitask_eduguard_v3_7_20261126`, split `validation`. Generated 2026-11-26T20:38:08.927932+00:00.

- **figures/fig01_training_curves**: Training and validation loss (assistant tokens only) and learning-rate schedule. The adapter with the lowest validation loss (epoch 1, 1.361) was retained as the final model.
- **figures/figS1_gpu_memory**: Peak allocated and reserved GPU memory per logging interval during training (RTX 4080, 16 GB).
- **figures/fig02_dataset_splits**: Rows per task and split in the final multitask dataset.
- **tables/t1_dataset_splits**: Rows per task and split.
- **figures/fig03_token_lengths**: Token length of training examples (prompt + answer, Qwen2.5 tokenizer; boxes = quartiles, whiskers = min/max). No example exceeds the 8,192-token limit, so nothing is truncated.
- **figures/fig04_summarization_composition**: Summarization records by source-length category and source type (all splits).
- **tables/t2_summarization_categories**: Summarization records per length category and split.
- **tables/t3_main_results**: Held-out validation results. 95% percentile bootstrap CIs (2000 resamples); paired bootstrap for the difference.
- **figures/fig05_main_results**: Held-out validation performance per task with 95% bootstrap confidence intervals. Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier; 'fully validated (calibrated)' uses a bge-m3 topic check calibrated on training references; ROUGE-L to reference compares each rewrite with the teacher-written rewrite for the same level; length-controlled ROUGE-L removes Markdown and cuts each output to its reference's word count, identically for both systems.
- **figures/fig06_bloom_confusion**: Requested target level versus the Bloom classifier's prediction for each generated rewrite (validation). Cells show counts; shading is the row-normalised share.
- **figures/fig07_bloom_per_level**: Bloom rewriting target accuracy per requested level with 95% bootstrap CIs (validation).
- **tables/t4_bloom_per_level**: Bloom rewriting target accuracy per requested level (validation).
- **figures/fig08_source_target_accuracy**: EduGuard LoRA: target accuracy for each source-to-target level transformation (validation). Each cell shows accuracy and n; the diagonal is empty by design.
- **figures/fig09_bloom_failures**: Primary failure category of Bloom rewrites that did not pass full validation (validation).
- **figures/fig10_summarization_breakdown**: Summarization ROUGE-L by source type and by source-length category with 95% bootstrap CIs (validation). Small groups (n < 10) have wide intervals.
- **tables/t5_summarization_breakdown**: Summarization ROUGE-L by source type and length category (validation).
- **figures/fig11_qa_by_type**: Closed-book QA ROUGE-L by question type with 95% bootstrap CIs (validation).
- **tables/t6_qa_by_type**: Closed-book QA ROUGE-L by question type (validation).
- **figures/fig12_rouge_distributions**: Per-example ROUGE-L distributions (validation); boxes show quartiles, points are individual examples.
- **figures/fig13_classifier_ceiling**: Agreement of the fixed Bloom classifier with the requested level for model rewrites and for the human reference rewrites of the same items (validation). Low reference agreement (e.g. Remember, Analyze) marks a limit of the classifier, not of the model.
- **tables/t7_bloom_model_vs_reference**: Bloom rewriting checks applied identically to model and human reference rewrites (validation).
- **tables/t8_semantic_agreement**: Embedding-based agreement with references (validation).

## Not generated

- base: no predictions.jsonl under C:\Eduguard\experiments\multitask_bloom_rewrite\results\qwen15b_multitask_eduguard_v3_7_20261126_base
