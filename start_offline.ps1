param([switch]$Development)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$python = Join-Path $root ".venv\Scripts\python.exe"
$frontend = Join-Path $root "frontend"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:OFFLINE_MODE = "true"

# v3.9 multitask generator (Q4_K_M export of best_adapter, verified by convert_to_gguf.py)
$env:GENERATOR_MODEL_PATH = Join-Path $root "models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf"
if (-not $env:GENERATOR_THREADS) { $env:GENERATOR_THREADS = "8" }
# Token budgets sized from measured v3.9 outputs (median / p99 tokens): rewrite 27 / 60, QA 45 / 107,
# summary 106 / 431. A cap only bounds the worst case (generation stops at end-of-turn), so these do
# not slow typical requests; caps below p99 cut answers mid-sentence and force summary re-generation.
if (-not $env:GENERATOR_CONTEXT_TOKENS) { $env:GENERATOR_CONTEXT_TOKENS = "4096" }  # long summary prompt + 384-token summary
if (-not $env:GENERATOR_ANSWER_TOKENS) { $env:GENERATOR_ANSWER_TOKENS = "160" }
if (-not $env:GENERATOR_SUMMARY_TOKENS) { $env:GENERATOR_SUMMARY_TOKENS = "384" }
if (-not $env:GENERATOR_MODERATION_TOKENS) { $env:GENERATOR_MODERATION_TOKENS = "96" }
if (-not $env:GENERATOR_REWRITE_TOKENS) { $env:GENERATOR_REWRITE_TOKENS = "96" }

# Final Bloom classifier: FedProx IID r20 best (Qwen2.5-0.5B).
$env:BLOOM_MODEL_DIR = Join-Path $root "models\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
# Retrieval encoder: local models\bge-small if present, otherwise the "bge-small" profile falls back
# to BAAI/bge-small-en-v1.5 from the Hugging Face cache (same model, same 384-dim embeddings).
if (Test-Path (Join-Path $root "models\bge-small")) {
    $env:RETRIEVAL_ENCODER = Join-Path $root "models\bge-small"
} else {
    $env:RETRIEVAL_ENCODER = "bge-small"
}

$env:CORS_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"

if (-not (Test-Path $python)) {
    throw "Python runtime missing: $python"
}

if (-not (Test-Path $env:GENERATOR_MODEL_PATH)) {
    throw "Qwen model missing at $($env:GENERATOR_MODEL_PATH)."
}

if (-not (Test-Path (Join-Path $root "models\bge-small"))) {
    $hfBge = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--BAAI--bge-small-en-v1.5"
    if (Test-Path $hfBge) {
        Write-Host "models\bge-small not found; using BAAI/bge-small-en-v1.5 from the Hugging Face cache."
    } else {
        Write-Warning "No BGE encoder found; document search falls back to keyword hashing. Download it once with: .venv\Scripts\python.exe -c `"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5')`""
    }
}

if (-not (Test-Path (Join-Path $env:BLOOM_MODEL_DIR "config.json"))) {
    throw "Bloom model missing at $($env:BLOOM_MODEL_DIR)."
}
$bloomWeights = @(
    (Join-Path $env:BLOOM_MODEL_DIR "model.safetensors"),
    (Join-Path $env:BLOOM_MODEL_DIR "pytorch_model.bin")
) | Where-Object { Test-Path $_ }
if (-not $bloomWeights) {
    throw "FedProx r20 merged classifier is selected, but no weight file is in $($env:BLOOM_MODEL_DIR). Place model.safetensors there. The folder currently has config and tokenizer files only."
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    throw "Frontend dependencies missing."
}

Write-Host ""
Write-Host "========================================"
Write-Host "          EduGuard Offline"
Write-Host "========================================"
Write-Host ""

# Prioritize the UI and launch both processes without fixed sleeps. The frontend
# can render while the backend's optional model warmup runs in its own thread.
if (-not $env:EDUGUARD_WARMUP_ON_STARTUP) { $env:EDUGUARD_WARMUP_ON_STARTUP = "1" }

if (-not $Development) {
    Write-Host "Starting production frontend..."
    $frontendProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList "run start -- --hostname 127.0.0.1 --port 3000" `
        -WorkingDirectory $frontend `
        -WindowStyle Hidden `
        -PassThru
}
else {
    Write-Host "Starting development frontend (Turbopack)..."
    $frontendProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList "run dev -- --hostname 127.0.0.1 --port 3000" `
        -WorkingDirectory $frontend `
        -WindowStyle Hidden `
        -PassThru
}

Write-Host "Starting backend in parallel..."
$backendProcess = Start-Process `
    -FilePath $python `
    -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru

# Poll for actual frontend readiness instead of imposing two fixed 5-second
# delays. The UI is opened as soon as Next responds; model readiness is separate.
$frontendReady = $false
$deadline = [DateTime]::UtcNow.AddSeconds(90)
while ([DateTime]::UtcNow -lt $deadline) {
    if ($frontendProcess.HasExited) {
        throw "Frontend process exited during startup (exit code $($frontendProcess.ExitCode)). Check the frontend terminal/log output."
    }
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:3000" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $frontendReady = $true
            break
        }
    } catch {
        # Next is still compiling or binding its port. Keep polling briefly.
    }
    Start-Sleep -Milliseconds 300
}

if (-not $frontendReady) {
    Write-Warning "Frontend did not respond within 90 seconds. It may still be compiling; opening the local URL anyway."
}
else {
    Write-Host "Frontend is responding."
}

try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "Backend health endpoint is responding."
} catch {
    Write-Host "Backend is still starting. The UI can open now; model readiness is reported separately by /models/status."
}

Write-Host ""
Write-Host "EduGuard local UI: http://127.0.0.1:3000"
Write-Host "Backend health:    http://127.0.0.1:8000/health"
Write-Host "Model status:     http://127.0.0.1:8000/models/status"
Write-Host ""
Start-Process "http://127.0.0.1:3000"
