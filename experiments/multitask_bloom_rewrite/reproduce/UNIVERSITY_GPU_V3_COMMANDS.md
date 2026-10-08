# UNIVERSITY GPU — FINAL 1.5B MULTITASK PIPELINE

Use this document only for the finalized generator dataset and training stack.
Historical v3 commands/results are retained elsewhere for provenance and are not part of the final run.

Assume:
- D:\\Eduguard
- Python 3.10 environment
- RTX 4090 or another machine that passes the resource gate

```powershell
cd D:\\Eduguard

# 0) GPU sanity
.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# 1) Generate final corpus from BloomShift + SQuAD + BillSum
.\\.venv\\Scripts\\python.exe experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --bloom-dir "data/rewrite dataset/bloomshift_final_candidate" --output-dir data/multitask_bloom_rewrite_final --model-id "Qwen/Qwen2.5-1.5B-Instruct" --max-seq-length 8192 --seed 42

# 2) Strict dataset QC
.\\.venv\\Scripts\\python.exe experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_final --model-id "Qwen/Qwen2.5-1.5B-Instruct" --max-length 8192

# 3) One-task-per-task 1.5B sanity check; add --backward on GPU for gradient validation
.\\.venv\\Scripts\\python.exe experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_final

# 4) Resource gate
.\\.venv\\Scripts\\python.exe experiments/multitask_bloom_rewrite/scripts/check_resources.py --model-id "Qwen/Qwen2.5-1.5B-Instruct"

# 5) Final 1.5B LoRA training
.\\.venv\\Scripts\\python.exe experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json
```

The final corpus uses 40% Bloom transformation, 30% QA, and 30% summarization in training. Validation/test remain task-wise held-out splits. Checkpoint selection uses validation loss only.

Do not use the old qwen15b v3/sumfix configs or data/multitask_bloom_rewrite_v3 for the final run.
