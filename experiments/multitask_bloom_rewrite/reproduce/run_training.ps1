$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
$Py = if (Test-Path "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe") {
  "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe"
} else { "python" }
& $Py experiments/multitask_bloom_rewrite/scripts/check_resources.py
if ($LASTEXITCODE -ne 0) {
  Write-Host "TRAINING NOT STARTED — INSUFFICIENT RESOURCES"
  exit $LASTEXITCODE
}
& $Py experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json
& $Py experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask.json
