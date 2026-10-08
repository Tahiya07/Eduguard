$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$Py = if (Test-Path "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe") { "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe" } else { "python" }

Write-Host "=== EduGuard final 1.5B multitask pipeline ==="
Write-Host "BloomShift: data/rewrite dataset/bloomshift_final_candidate"
Write-Host "Final corpus: data/multitask_bloom_rewrite_final"

& $Py experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --bloom-dir "data/rewrite dataset/bloomshift_final_candidate" --output-dir data/multitask_bloom_rewrite_final --model-id "Qwen/Qwen2.5-1.5B-Instruct" --max-seq-length 8192 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Final multitask dataset preparation failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_final --model-id "Qwen/Qwen2.5-1.5B-Instruct" --max-length 8192
if ($LASTEXITCODE -ne 0) { throw "Final multitask dataset QC failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_final
if ($LASTEXITCODE -ne 0) { throw "1.5B training sanity check failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/check_resources.py --model-id "Qwen/Qwen2.5-1.5B-Instruct"
if ($LASTEXITCODE -ne 0) { Write-Host "TRAINING NOT STARTED — INSUFFICIENT RESOURCES"; exit $LASTEXITCODE }

& $Py experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json
if ($LASTEXITCODE -ne 0) { throw "Final 1.5B LoRA training failed." }
