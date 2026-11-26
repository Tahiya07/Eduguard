$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$Py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$Config = "experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_eduguard_v3_4_20261126.json"

Write-Host "=== EduGuard Bloom-prioritized 1.5B follow-up ==="
Write-Host "Dataset: data/multitask_bloom_rewrite_separate_eduguard_v3_3"
Write-Host "Config: $Config"

& $Py experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_separate_eduguard_v3_3 --model-id "Qwen/Qwen2.5-1.5B-Instruct" --max-length 8192
if ($LASTEXITCODE -ne 0) { throw "Final multitask dataset QC failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/check_resources.py --config $Config --defer-vram-check-to-preflight
if ($LASTEXITCODE -ne 0) { Write-Host "TRAINING NOT STARTED - INSUFFICIENT RESOURCES"; exit $LASTEXITCODE }

& $Py experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config $Config --dataset-dir data/multitask_bloom_rewrite_separate_eduguard_v3_3 --backward
if ($LASTEXITCODE -ne 0) { throw "1.5B maximum-context GPU sanity check failed; training was not started." }

& $Py experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Final 1.5B LoRA training failed." }

& $Py experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config $Config --condition lora --split validation
if ($LASTEXITCODE -ne 0) { throw "Validation evaluation failed." }
