@echo off
setlocal EnableExtensions

echo Updating EduGuard dist folder ^(runtime assets only^)...
echo.
echo This does NOT rebuild EXEs. Use build_exe.bat for a full rebuild.
echo.

if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: dist\EduGuard\EduGuard.exe not found. Run build_exe.bat first.
    exit /b 1
)

echo Step 1: Updating frontend standalone...
if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: frontend\.next\standalone\server.js not found.
    exit /b 1
)
if exist "dist\EduGuard\frontend\.next\standalone" rmdir /s /q "dist\EduGuard\frontend\.next\standalone"
xcopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone\" /E /I /Y >nul
echo.

echo Step 2: Updating public folder...
if exist "frontend\public" (
    if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
    xcopy "frontend\public" "dist\EduGuard\frontend\public\" /E /I /Y >nul
)
echo.

echo Step 3: Updating runtime models only...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 (
    echo ERROR: Runtime model copy failed.
    exit /b 1
)
echo.

echo Step 4: Refreshing portable .env...
(
    echo BLOOM_MODEL_SIZE=0.5b
    echo BLOOM_MODEL_DIR=models/qwen_bloom_fedprox_r20
    echo BLOOM_USE_QUANTIZED=false
    echo GENERATOR_MODEL_PATH=models/qwen.gguf
    echo RETRIEVAL_ENCODER=bge-small
    echo OFFLINE_MODE=true
    echo GENERATOR_THREADS=8
    echo GENERATOR_CONTEXT_TOKENS=512
    echo GENERATOR_ANSWER_TOKENS=64
    echo GENERATOR_SUMMARY_TOKENS=80
    echo GENERATOR_MODERATION_TOKENS=64
    echo GENERATOR_REWRITE_TOKENS=64
    echo HF_HUB_OFFLINE=1
    echo TRANSFORMERS_OFFLINE=1
    echo HF_DATASETS_OFFLINE=1
) > "dist\EduGuard\.env"
echo.

echo ========================================
echo Update complete
echo ========================================
echo Updated: frontend + lean models + .env
echo EXEs unchanged. Rebuild with build_exe.bat if backend/launcher code changed.
echo.
pause
endlocal
