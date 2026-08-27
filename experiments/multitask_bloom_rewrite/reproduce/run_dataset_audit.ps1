$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
$Py = if (Test-Path "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe") {
  "C:\Users\tahiy\AppData\Local\Programs\Python\Python310\python.exe"
} else { "python" }
& $Py experiments/multitask_bloom_rewrite/scripts/audit_datasets.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py experiments/multitask_bloom_rewrite/scripts/template_memorization_report.py
