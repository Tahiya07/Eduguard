param([switch]$Development)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$python = Join-Path $root ".venv\Scripts\python.exe"
$frontend = Join-Path $root "frontend"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:OFFLINE_MODE = "true"

$env:GENERATOR_MODEL_PATH = Join-Path $root "models\qwen.gguf"
$env:BLOOM_MODEL_DIR = Join-Path $root "models\qwen_bloom_merged0.5B"
$env:RETRIEVAL_ENCODER = Join-Path $root "models\bge-small"

$env:CORS_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"

if (-not (Test-Path $python)) {
    throw "Python runtime missing: $python"
}

if (-not (Test-Path (Join-Path $root "models\qwen.gguf"))) {
    throw "Qwen model missing."
}

if (-not (Test-Path (Join-Path $root "models\bge-small"))) {
    throw "BGE model missing."
}

if (-not (Test-Path (Join-Path $root "models\qwen_bloom_merged0.5B"))) {
    throw "Bloom model missing."
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    throw "Frontend dependencies missing."
}

Write-Host ""
Write-Host "========================================"
Write-Host "          EduGuard Offline"
Write-Host "========================================"
Write-Host ""

Write-Host "Starting backend..."

Start-Process `
    -FilePath $python `
    -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $root `
    -WindowStyle Hidden

Start-Sleep -Seconds 5

if (-not $Development) {
    Write-Host "Starting production frontend..."

    Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList "run start -- --hostname 127.0.0.1 --port 3000" `
        -WorkingDirectory $frontend `
        -WindowStyle Hidden
}
else {
    Write-Host "Starting development frontend..."

    Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList "run dev -- --hostname 127.0.0.1 --port 3000" `
        -WorkingDirectory $frontend `
 \       -WindowStyle Hidden
}

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "EduGuard is running locally:"
Write-Host "http://127.0.0.1:3000"
Write-Host ""
Start-Process "http://127.0.0.1:3000"