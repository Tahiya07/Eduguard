@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo       Building EduGuard - Portable
echo ========================================
echo.

rem Step 0: Verify required build inputs.
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found.
    exit /b 1
)
if not exist "node.exe" (
    echo ERROR: Root node.exe not found. Required for the portable frontend.
    exit /b 1
)
if not exist "frontend\package.json" (
    echo ERROR: frontend\package.json not found.
    exit /b 1
)
if not exist "frontend\node_modules\next\dist\bin\next" (
    echo ERROR: Frontend dependencies are missing.
    echo Run: cd frontend ^&^& npm ci
    exit /b 1
)

rem Step 1: Build Next.js standalone output, including static assets.
echo [1/7] Building Next.js frontend...
pushd frontend
call npm run build
if errorlevel 1 (
    popd
    echo ERROR: npm run build failed.
    exit /b 1
)
popd

if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: Next.js standalone server.js was not generated.
    exit /b 1
)
if not exist "frontend\.next\standalone\.next\static" (
    echo ERROR: Standalone Next.js static assets are missing.
    exit /b 1
)

rem Step 2: Build backend first. Models stay external to the EXE.
echo.
echo [2/7] Building EduGuardBackend.exe...
".venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm "EduGuardBackend.spec"
if errorlevel 1 (
    echo ERROR: Backend PyInstaller build failed.
    exit /b 1
)
if not exist "dist\EduGuardBackend\EduGuardBackend.exe" if not exist "dist\EduGuardBackend.exe" (
    echo ERROR: Backend executable was not generated.
    exit /b 1
)

rem Step 3: Build launcher. EduGuard.spec collects root node.exe.
echo.
echo [3/7] Building EduGuard.exe...
".venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm "EduGuard.spec"
if errorlevel 1 (
    echo ERROR: Launcher PyInstaller build failed.
    exit /b 1
)
if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: dist\EduGuard\EduGuard.exe was not generated.
    exit /b 1
)

rem Step 4: Copy the standalone frontend AFTER PyInstaller has created dist.
echo.
echo [4/7] Copying standalone frontend...
if exist "dist\EduGuard\frontend\.next\standalone" rmdir /s /q "dist\EduGuard\frontend\.next\standalone"
robocopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone" /E /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 (
    echo ERROR: Failed to copy standalone frontend.
    exit /b 1
)
if not exist "dist\EduGuard\frontend\.next\standalone\server.js" (
    echo ERROR: Packaged Next.js server.js is missing.
    exit /b 1
)
if not exist "dist\EduGuard\frontend\.next\standalone\.next\static" (
    echo ERROR: Packaged Next.js static assets are missing.
    exit /b 1
)

rem Keep public available at the path expected by the standalone app.
if exist "frontend\public" (
    if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
    robocopy "frontend\public" "dist\EduGuard\frontend\public" /E /NFL /NDL /NJH /NJS /NC /NS /NP
    if errorlevel 8 (
        echo ERROR: Failed to copy frontend public assets.
        exit /b 1
    )
)

rem Step 5: Copy only the explicitly selected runtime model artifacts.
echo.
echo [5/7] Copying final runtime models...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 (
    echo ERROR: Runtime model packaging failed.
    exit /b 1
)

rem Step 6: Install the separately built backend executable alongside launcher.
echo.
echo [6/7] Installing EduGuardBackend.exe...
if exist "dist\EduGuardBackend\EduGuardBackend.exe" (
    copy /Y "dist\EduGuardBackend\EduGuardBackend.exe" "dist\EduGuard\EduGuardBackend.exe" >nul
) else (
    copy /Y "dist\EduGuardBackend.exe" "dist\EduGuard\EduGuardBackend.exe" >nul
)
if errorlevel 1 (
    echo ERROR: Could not copy backend executable into dist\EduGuard.
    exit /b 1
)

rem Step 7: Write the portable runtime environment and validate package.
echo.
echo [7/7] Writing portable .env...
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

if not exist "dist\EduGuard\EduGuard.exe" goto :verify_failed
if not exist "dist\EduGuard\EduGuardBackend.exe" goto :verify_failed
if not exist "dist\EduGuard\node.exe" goto :verify_failed
if not exist "dist\EduGuard\frontend\.next\standalone\server.js" goto :verify_failed
if not exist "dist\EduGuard\frontend\.next\standalone\.next\static" goto :verify_failed
if not exist "dist\EduGuard\models\qwen.gguf" goto :verify_failed
if not exist "dist\EduGuard\models\qwen_bloom_merged0.5B\model.safetensors" goto :verify_failed
if not exist "dist\EduGuard\models\bge-small\model.safetensors" goto :verify_failed
if exist "dist\EduGuard\models\qwen_bloom_merged0.5B" (
    echo ERROR: Unintended centralized Bloom model found in package.
    exit /b 1
)
if exist "dist\EduGuard\models\qwen_bloom_merged0.5B.zip" (
    echo ERROR: Model archive must not be packaged.
    exit /b 1
)
if exist "dist\EduGuard\models\bge-small\onnx" (
    echo ERROR: Duplicate BGE ONNX assets must not be packaged.
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETED - FILE CHECKS PASSED
echo ========================================
echo Package: %CD%\dist\EduGuard\
echo Launcher: dist\EduGuard\EduGuard.exe
echo Backend:  dist\EduGuard\EduGuardBackend.exe
echo Frontend: dist\EduGuard\frontend\.next\standalone\server.js
echo Models:   dist\EduGuard\models\
echo.
echo Start with: dist\EduGuard\EduGuard.exe
echo This verifies package files, not a full runtime smoke test.
echo.
pause
exit /b 0

:verify_failed
echo.
echo ERROR: Final package verification failed.
echo One or more required files are missing under dist\EduGuard\.
exit /b 1
