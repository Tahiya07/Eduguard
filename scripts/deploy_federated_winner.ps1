param(
    [switch]$SkipLiveEval,
    [switch]$SkipMerge
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($root -match "scripts$") {
    $root = Split-Path -Parent $root
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python runtime missing: $python"
}

$argsList = @("experiments/federated/scripts/compare_fedavg_5r_vs_r20.py")
if ($SkipLiveEval) { $argsList += "--skip-live-eval" }
if ($SkipMerge) { $argsList += "--skip-merge" }

& $python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$deploy = Join-Path $root "artifacts\evaluation\deployment_recommendation.json"
if (-not (Test-Path $deploy)) {
    Write-Host "No deployment_recommendation.json produced."
    exit 0
}

$rec = Get-Content $deploy -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "Winner: $($rec.winner)"
Write-Host "Merge status: $($rec.merge.status)"
if ($rec.merge.merge_command) {
    Write-Host "Merge command (if needed): $($rec.merge.merge_command)"
}
Write-Host ""
Write-Host "To run EduGuard with the federated winner:"
Write-Host ('$env:BLOOM_MODEL_DIR = "' + $rec.bloom_model_dir + '"')
Write-Host ".\start_offline.ps1 -Development"
