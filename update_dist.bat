@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Refreshes frontend assets, runtime models and .env in an existing dist\EduGuard\.
rem It does NOT rebuild EduGuard.exe or the backend; use build_exe.bat after any Python code change.
rem Usage: update_dist.bat [--no-pause]

set "NO_PAUSE=%~1"
echo ========================================
echo     Updating EduGuard dist assets
echo ========================================

if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: dist\EduGuard\EduGuard.exe not found. Run build_exe.bat first.
    goto :fail
)
if not exist "dist\EduGuard\backend\EduGuardBackend.exe" (
    echo ERROR: dist\EduGuard\backend\EduGuardBackend.exe not found. Run build_exe.bat ^(the backend is now a folder build^).
    goto :fail
)

echo [1/4] Building and copying the standalone frontend...
set "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000"
set "NEXT_TELEMETRY_DISABLED=1"
pushd frontend
call npm run build
if errorlevel 1 (
    popd
    echo ERROR: npm run build failed.
    goto :fail
)
popd
robocopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone" /MIR /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 goto :fail

echo [2/4] Updating public assets...
rem public\ ships inside the standalone build (copied there by npm run build); remove the old duplicate.
if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
if not exist "dist\EduGuard\frontend\.next\standalone\public" goto :fail

echo [3/4] Updating runtime models...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 goto :fail

echo [4/4] Refreshing .env and clearing saved sessions...
rem Never ship uploaded material: sessions created while testing the package hold users' notes.
if exist "dist\EduGuard\data\web_sessions" rmdir /s /q "dist\EduGuard\data\web_sessions"
(
    echo # Defaults for the packaged app. EduGuard.exe sets the same values when it starts the backend.
    echo OFFLINE_MODE=true
    echo HF_HUB_OFFLINE=1
    echo TRANSFORMERS_OFFLINE=1
    echo HF_DATASETS_OFFLINE=1
    echo BLOOM_MODEL_SIZE=0.5b
    echo BLOOM_MODEL_DIR=models/qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged
    echo BLOOM_USE_QUANTIZED=false
    echo GENERATOR_MODEL_PATH=models/gguf/qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf
    echo GENERATOR_REWRITE_LORA_SCALE=0
    echo RETRIEVAL_ENCODER=bge-small
    echo GENERATOR_THREADS=8
    echo GENERATOR_CONTEXT_TOKENS=4096
    echo GENERATOR_ANSWER_TOKENS=160
    echo GENERATOR_SUMMARY_TOKENS=384
    echo GENERATOR_MODERATION_TOKENS=96
    echo GENERATOR_REWRITE_TOKENS=96
) > "dist\EduGuard\.env"

echo.
echo DIST UPDATE COMPLETED. EXEs unchanged; use build_exe.bat if Python code changed.
if /i not "%NO_PAUSE%"=="--no-pause" pause
exit /b 0

:fail
echo.
echo UPDATE FAILED.
if /i not "%NO_PAUSE%"=="--no-pause" pause
exit /b 1
