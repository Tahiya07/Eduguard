@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo     Updating EduGuard dist assets
echo ========================================
echo.
echo This refreshes frontend assets, runtime models, and .env.
echo It does NOT rebuild EduGuard.exe or EduGuardBackend.exe.
echo Use build_exe.bat for a full rebuild.
echo.

if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: dist\EduGuard\EduGuard.exe not found. Run build_exe.bat first.
    exit /b 1
)
if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: frontend\.next\standalone\server.js not found.
    echo Build the frontend with: cd frontend ^&^& npm run build
    exit /b 1
)
if not exist "node.exe" (
    echo ERROR: Root node.exe not found. Rebuild the launcher package.
    exit /b 1
)

echo [1/4] Updating standalone frontend...
if exist "dist\EduGuard\frontend\.next\standalone" rmdir /s /q "dist\EduGuard\frontend\.next\standalone"
robocopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone" /E /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 (
    echo ERROR: Failed to update standalone frontend.
    exit /b 1
)
if not exist "dist\EduGuard\frontend\.next\standalone\server.js" (
    echo ERROR: Packaged server.js missing after copy.
    exit /b 1
)

echo.
echo [2/4] Updating public assets...
if exist "frontend\public" (
    if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
    robocopy "frontend\public" "dist\EduGuard\frontend\public" /E /NFL /NDL /NJH /NJS /NC /NS /NP
    if errorlevel 8 (
        echo ERROR: Failed to update public assets.
        exit /b 1
    )
)

echo.
echo [3/4] Updating selected runtime models...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 (
    echo ERROR: Runtime model update failed.
    exit /b 1
)

echo.
echo [4/4] Refreshing portable .env...
(
    echo BLOOM_MODEL_SIZE=0.5b
    echo BLOOM_MODEL_DIR=models/qwen_bloom_merged0.5B
    echo BLOOM_USE_QUANTIZED=false
    echo GENERATOR_MODEL_PATH=models/qwen.gguf
    echo RETRIEVAL_ENCODER=bge-small
    echo OFFLINE_MODE=true
echo GENERATOR_THREADS=8
echo GENERATOR_CONTEXT_TOKENS=1024
echo GENERATOR_ANSWER_TOKENS=96
echo GENERATOR_SUMMARY_TOKENS=128
echo GENERATOR_MODERATION_TOKENS=48
echo GENERATOR_REWRITE_TOKENS=96
    echo HF_HUB_OFFLINE=1
    echo TRANSFORMERS_OFFLINE=1
    echo HF_DATASETS_OFFLINE=1
) > "dist\EduGuard\.env"

if not exist "dist\EduGuard\models\qwen.gguf" (
    echo ERROR: Packaged multitask GGUF missing.
    exit /b 1
)
if not exist "dist\EduGuard\models\qwen_bloom_merged0.5B\model.safetensors" (
    echo ERROR: Packaged Bloom classifier weights missing.
    exit /b 1
)
if not exist "dist\EduGuard\models\bge-small\model.safetensors" (
    echo ERROR: Packaged BGE encoder weights missing.
    exit /b 1
)

echo.
echo ========================================
echo DIST UPDATE COMPLETED - FILE CHECKS PASSED
echo ========================================
echo EXEs unchanged. Use build_exe.bat if Python launcher/backend code changed.
echo.
pause
exit /b 0
