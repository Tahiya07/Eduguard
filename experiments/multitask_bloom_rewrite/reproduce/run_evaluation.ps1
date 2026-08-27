$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
Write-Host "Evaluation scripts run after trained checkpoints exist."
Write-Host "Populate results/{qwen05b_base,qwen05b_lora,qwen15b_base,qwen15b_lora}/ from measured runs only."
exit 0
