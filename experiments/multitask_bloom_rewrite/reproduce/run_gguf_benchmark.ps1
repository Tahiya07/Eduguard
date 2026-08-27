$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
$Py = if (Test-Path "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe") {
  "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe"
} else { "python" }
# Example (after conversion):
# & $Py experiments/multitask_bloom_rewrite/scripts/benchmark_gguf.py --gguf experiments/multitask_bloom_rewrite/models/qwen05b_multitask.gguf
Write-Host "Provide --gguf paths after conversion. Refuses overwriting models/qwen.gguf."
exit 0
