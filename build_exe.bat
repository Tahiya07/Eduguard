@echo off
setlocal EnableExtensions

echo ========================================
echo       Building EduGuard (portable)
echo ========================================
echo.
echo FINAL deployed models only:
echo   1^) FedProx IID r20 best Bloom classifier ^(Qwen2.5-0.5B^)
echo   2^) Multitask v3 Q4_K_M generator GGUF ^(models\qwen.gguf^)
echo   +) lean BGE-small ^(lazy retrieval encoder^)
echo Centralized bloom merge / zips / onnx duplicates are NOT packaged.
echo.

echo Step 1: Building EduGuardBackend.exe...
.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm EduGuardBackend.spec
if errorlevel 1 (
    echo.
    echo ERROR: Backend build failed.
    exit /b 1
)
echo.

echo Step 2: Building EduGuard.exe launcher...
.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm EduGuard.spec
if errorlevel 1 (
    echo.
    echo ERROR: Main app build failed.
    exit /b 1
)
echo.

echo Step 3: Verifying main executable...
if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: EduGuard.exe was not created.
    exit /b 1
)
echo EduGuard.exe verified.
echo.

echo Step 4: Copying frontend standalone build...
if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: frontend\.next\standalone\server.js not found.
    echo Run: cd frontend ^&^& npm run build
    exit /b 1
)

if exist "dist\EduGuard\frontend\.next\standalone" rmdir /s /q "dist\EduGuard\frontend\.next\standalone"
xcopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone\" /E /I /Y >nul
if errorlevel 4 (
    echo ERROR: Failed to copy frontend standalone build.
    exit /b 1
)
echo Frontend standalone build copied.
echo.

echo Step 5: Copying frontend public folder...
if exist "frontend\public" (
    if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
    xcopy "frontend\public" "dist\EduGuard\frontend\public\" /E /I /Y >nul
)
echo Public folder ready.
echo.

echo Step 6: Copying FINAL runtime models only...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 (
    echo ERROR: Final model copy failed.
    exit /b 1
)
echo.

echo Step 7: Copying EduGuardBackend.exe...
if exist "dist\EduGuardBackend\EduGuardBackend.exe" (
    copy /Y "dist\EduGuardBackend\EduGuardBackend.exe" "dist\EduGuard\EduGuardBackend.exe" >nul
) else if exist "dist\EduGuardBackend.exe" (
    copy /Y "dist\EduGuardBackend.exe" "dist\EduGuard\EduGuardBackend.exe" >nul
) else (
    echo ERROR: EduGuardBackend.exe not found after PyInstaller.
    exit /b 1
)
echo Backend executable copied.
echo.

echo Step 8: Writing portable runtime env...
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
echo Portable .env written ^(FedProx r20 + multitask v3 GGUF^).
echo.

echo Step 9: Final verification...
if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: EduGuard.exe missing.
    exit /b 1
)
if not exist "dist\EduGuard\EduGuardBackend.exe" (
    echo ERROR: EduGuardBackend.exe missing.
    exit /b 1
)
if not exist "dist\EduGuard\frontend\.next\standalone\server.js" (
    echo ERROR: Next.js server.js missing.
    exit /b 1
)
if not exist "dist\EduGuard\models\qwen.gguf" (
    echo ERROR: Multitask v3 GGUF missing.
    exit /b 1
)
if not exist "dist\EduGuard\models\qwen_bloom_fedprox_r20\model.safetensors" (
    echo ERROR: FedProx r20 Bloom weights missing from package.
    exit /b 1
)
if not exist "dist\EduGuard\models\bge-small\model.safetensors" (
    echo ERROR: BGE encoder weights missing.
    exit /b 1
)
if exist "dist\EduGuard\models\qwen_bloom_merged0.5B" (
    echo ERROR: Centralized bloom merge must not be packaged as final.
    exit /b 1
)
if exist "dist\EduGuard\models\qwen_bloom_merged0.5B.zip" (
    echo ERROR: Zip archive must not be packaged.
    exit /b 1
)
if exist "dist\EduGuard\models\bge-small\onnx" (
    echo ERROR: BGE onnx duplicate must not be packaged.
    exit /b 1
)

echo.
echo ========================================
echo          BUILD COMPLETE
echo ========================================
echo.
echo Package: dist\EduGuard\
echo Final models:
echo   Bloom classifier = FedProx r20  -^> models\qwen_bloom_fedprox_r20
echo   Generator GGUF   = multitask v3 -^> models\qwen.gguf
echo.
echo Start with: dist\EduGuard\EduGuard.exe
echo.

pause
endlocal
