# Draft figure and table captions

Run `qwen15b_multitask_eduguard_v3_9_20261127`, split `test`. Generated 2026-11-27T20:27:11.153196+00:00.

- **figures/fig01_training_curves**: Training and validation loss (assistant tokens only) and learning-rate schedule. The adapter with the lowest validation loss (epoch 1, 1.365) was retained as the final model.
- **figures/figS1_gpu_memory**: Peak allocated and reserved GPU memory per logging interval during training (RTX 4080, 16 GB).
- **figures/fig02_dataset_splits**: Rows per task and split in the final multitask dataset.
- **tables/t1_dataset_splits**: Rows per task and split.
- **figures/fig03_token_lengths**: Token length of training examples (prompt + answer, Qwen2.5 tokenizer; boxes = quartiles, whiskers = min/max). No example exceeds the 8,192-token limit, so nothing is truncated.
- **figures/fig04_summarization_composition**: Summarization records by source-length category and source type (all splits).
- **tables/t2_summarization_categories**: Summarization records per length category and split.
- **tables/t3_main_results**: Held-out test results. 95% percentile bootstrap CIs (2000 resamples); paired bootstrap for the difference.
- **figures/fig05_main_results**: Held-out test performance per task with 95% bootstrap confidence intervals. Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier; 'fully validated (calibrated)' uses a bge-m3 topic check calibrated on training references; ROUGE-L to reference compares each rewrite with the teacher-written rewrite for the same level; length-controlled ROUGE-L removes Markdown and cuts each output to its reference's word count, identically for both systems.
- **figures/fig06_bloom_confusion**: Requested target level versus the Bloom classifier's prediction for each generated rewrite (test). Cells show counts; shading is the row-normalised share.
- **figures/fig07_bloom_per_level**: Bloom rewriting target accuracy per requested level with 95% bootstrap CIs (test).
- **tables/t4_bloom_per_level**: Bloom rewriting target accuracy per requested level (test).
- **figures/fig08_source_target_accuracy**: Base Qwen2.5-1.5B: target accuracy for each source-to-target level transformation (test). Each cell shows accuracy and n; the diagonal is empty by design.
- **figures/fig09_bloom_failures**: Primary failure category of Bloom rewrites that did not pass full validation (test).
- **figures/fig10_summarization_breakdown**: Summarization ROUGE-L by source type and by source-length category with 95% bootstrap CIs (test). Small groups (n < 10) have wide intervals.
- **tables/t5_summarization_breakdown**: Summarization ROUGE-L by source type and length category (test).
- **figures/fig11_qa_by_type**: Closed-book QA ROUGE-L by question type with 95% bootstrap CIs (test).
- **tables/t6_qa_by_type**: Closed-book QA ROUGE-L by question type (test).
- **figures/fig12_rouge_distributions**: Per-example ROUGE-L distributions (test); boxes show quartiles, points are individual examples.

## Not generated

- qwen15b_multitask_eduguard_v3_9_20261127_base_test: judge not reliable for ['summarization'] - judge accuracy omitted there
- lora: no predictions.jsonl under C:\Eduguard\experiments\multitask_bloom_rewrite\results\qwen15b_multitask_eduguard_v3_9_20261127_test
