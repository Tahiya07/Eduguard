$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
$Py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" -3 }
& $Py -m unittest experiments.multitask_bloom_rewrite.tests.test_evaluate_rewrite -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json --condition lora --smoke-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json --condition lora --limit 2
