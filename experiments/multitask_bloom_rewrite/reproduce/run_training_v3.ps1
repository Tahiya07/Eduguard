$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$Py = if (Test-Path "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe") {
  "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe"
} else { "python" }

Write-Host "=== EduGuard Multi-task v3 training ==="
Write-Host "Canonical Bloom dataset: data/rewrite dataset"
Write-Host "Prepared corpus: data/multitask_bloom_rewrite_v3"
Write-Host ""

& $Py experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_v3.py `
  --locked-multitask-dir data/multitask_bloom_rewrite `
  --bloom-v3-dir "data/rewrite dataset" `
  --output-dir data/multitask_bloom_rewrite_v3 `
  --seed 42

if ($LASTEXITCODE -ne 0) { throw "Multi-task v3 dataset preparation failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/check_resources.py `
  --model-id "Qwen/Qwen2.5-1.5B-Instruct"

if ($LASTEXITCODE -ne 0) {
  Write-Host "TRAINING NOT STARTED — INSUFFICIENT RESOURCES"
  exit $LASTEXITCODE
}

& $Py experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_v3.json
