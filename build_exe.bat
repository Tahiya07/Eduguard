@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Builds the portable EduGuard package in dist\EduGuard\:
rem   EduGuard.exe            launcher (starts frontend + backend, opens the browser)
rem   node.exe                Node runtime for the Next.js frontend
rem   backend\                EduGuardBackend.exe + its _internal\ libraries (folder build)
rem   frontend\               Next.js standalone server + static + public assets
rem   models\                 v3.9 GGUF generator, FedProx Bloom classifier, BGE encoder
rem   tesseract\              (optional) Tesseract OCR, copied if installed on this PC
rem   .env                    defaults (the launcher sets the same values explicitly)
rem Usage: build_exe.bat [--no-pause]

set "NO_PAUSE=%~1"

echo ========================================
echo       Building EduGuard - Portable
echo ========================================
echo.

rem ---- Step 0: required build inputs ---------------------------------------------------------
rem Build from .venv-build: same library versions as .venv but CPU-only PyTorch. The CUDA build of
rem PyTorch in .venv (needed for training) would add ~3.6 GB of GPU libraries the app never uses;
rem the generator runs on llama.cpp and uses a GPU through GENERATOR_GPU_LAYERS when its build supports it.
set "PY=.venv-build\Scripts\python.exe"
if not exist "%PY%" (
    echo WARNING: .venv-build not found; falling back to .venv ^(CUDA PyTorch, ~3.6 GB larger package^).
    set "PY=.venv\Scripts\python.exe"
)
if not exist "%PY%" (
    echo ERROR: no Python environment found ^(.venv-build or .venv^).
    goto :fail
)
"%PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed in %PY%. Run: %PY% -m pip install pyinstaller
    goto :fail
)
"%PY%" -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 3)" 2>nul
if errorlevel 3 echo WARNING: %PY% has CUDA PyTorch; the package will be ~3.6 GB larger.
if not exist "frontend\node_modules\next\dist\bin\next" (
    echo ERROR: Frontend dependencies are missing. Run: cd frontend ^&^& npm ci
    goto :fail
)
if not exist "node.exe" (
    for /f "delims=" %%N in ('where node 2^>nul') do if not exist "node.exe" copy /Y "%%N" "node.exe" >nul
)
if not exist "node.exe" (
    echo ERROR: node.exe not found in the project root or on PATH. Install Node.js or copy node.exe here.
    goto :fail
)

rem ---- Step 1: Next.js standalone build ------------------------------------------------------
echo [1/8] Building Next.js frontend...
rem NEXT_PUBLIC_* values are baked in at build time; the launcher serves the API on 127.0.0.1:8000.
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
if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: Next.js standalone server.js was not generated.
    goto :fail
)
if not exist "frontend\.next\standalone\.next\static" (
    echo ERROR: Standalone Next.js static assets are missing.
    goto :fail
)

rem ---- Step 2: backend (folder build) ---------------------------------------------------------
echo.
echo [2/8] Building backend (PyInstaller, this takes several minutes)...
"%PY%" -m PyInstaller --clean --noconfirm "EduGuardBackend.spec"
if errorlevel 1 (
    echo ERROR: Backend PyInstaller build failed.
    goto :fail
)
if not exist "dist\EduGuardBackend\EduGuardBackend.exe" (
    echo ERROR: dist\EduGuardBackend\EduGuardBackend.exe was not generated.
    goto :fail
)

rem ---- Step 3: launcher (recreates dist\EduGuard, so everything else is copied afterwards) -----
echo.
echo [3/8] Building EduGuard.exe launcher...
"%PY%" -m PyInstaller --clean --noconfirm "EduGuard.spec"
if errorlevel 1 (
    echo ERROR: Launcher PyInstaller build failed.
    goto :fail
)
if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: dist\EduGuard\EduGuard.exe was not generated.
    goto :fail
)
copy /Y "node.exe" "dist\EduGuard\node.exe" >nul || goto :fail

rem ---- Step 4: frontend ----------------------------------------------------------------------
echo.
echo [4/8] Copying standalone frontend...
robocopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone" /MIR /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 (
    echo ERROR: Failed to copy standalone frontend.
    goto :fail
)
rem public\ is already inside the standalone build (npm run build copies it to .next\standalone\public),
rem which is the only copy server.js serves; a second copy next to it was 15 MB of duplicates.
if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
if not exist "dist\EduGuard\frontend\.next\standalone\public" (
    echo ERROR: standalone build has no public\ folder. Run "npm run build" in frontend.
    goto :fail
)

rem ---- Step 5: models ------------------------------------------------------------------------
echo.
echo [5/8] Copying runtime models...
call "%~dp0scripts\copy_runtime_models.bat" "dist\EduGuard\models"
if errorlevel 1 (
    echo ERROR: Runtime model packaging failed.
    goto :fail
)

rem ---- Step 6: backend folder ----------------------------------------------------------------
echo.
echo [6/8] Installing backend folder...
if exist "dist\EduGuard\EduGuardBackend.exe" del /q "dist\EduGuard\EduGuardBackend.exe"
robocopy "dist\EduGuardBackend" "dist\EduGuard\backend" /MIR /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 (
    echo ERROR: Could not copy the backend folder into dist\EduGuard\backend.
    goto :fail
)

rem ---- Step 7: optional Tesseract OCR (JPG/PNG uploads) ----------------------------------------
echo.
echo [7/8] Tesseract OCR...
set "TESS_SRC="
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" set "TESS_SRC=%ProgramFiles%\Tesseract-OCR"
if not defined TESS_SRC if exist "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" set "TESS_SRC=%ProgramFiles(x86)%\Tesseract-OCR"
if defined TESS_SRC (
    rem Runtime only: tesseract.exe, its DLLs and English/OSD data. Training tools, docs and Java
    rem viewers in the install are not needed.
    if exist "dist\EduGuard\tesseract" rmdir /s /q "dist\EduGuard\tesseract"
    robocopy "%TESS_SRC%" "dist\EduGuard\tesseract" tesseract.exe *.dll /NFL /NDL /NJH /NJS /NC /NS /NP
    if errorlevel 8 goto :tess_fail
    rem English only. osd.traineddata (10 MB) is for page-orientation detection (--psm 0/1), which EduGuard never uses.
    robocopy "%TESS_SRC%\tessdata" "dist\EduGuard\tesseract\tessdata" eng.traineddata /NFL /NDL /NJH /NJS /NC /NS /NP
    if errorlevel 8 goto :tess_fail
    for %%D in (configs tessconfigs) do if exist "%TESS_SRC%\tessdata\%%D" robocopy "%TESS_SRC%\tessdata\%%D" "dist\EduGuard\tesseract\tessdata\%%D" /E /NFL /NDL /NJH /NJS /NC /NS /NP
    if not exist "dist\EduGuard\tesseract\tesseract.exe" goto :tess_fail
    echo   bundled runtime from %TESS_SRC%
) else (
    echo   WARNING: Tesseract not installed on this PC; JPG/PNG uploads will need it on the target PC.
    echo            Install with: winget install --id UB-Mannheim.TesseractOCR --exact, then rebuild.
)

rem ---- Step 8: .env and verification ---------------------------------------------------------
echo.
echo [8/8] Writing .env and verifying the package...
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

for %%F in (
    "EduGuard.exe"
    "node.exe"
    "backend\EduGuardBackend.exe"
    "backend\_internal"
    "backend\_internal\experiments\multitask_bloom_rewrite\prompts.py"
    "frontend\.next\standalone\server.js"
    "frontend\.next\standalone\.next\static"
    "frontend\.next\standalone\public"
    "models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf"
    "models\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged\model.safetensors"
    "models\bge-small\model.safetensors"
    ".env"
) do (
    if not exist "dist\EduGuard\%%~F" (
        echo ERROR: missing from package: dist\EduGuard\%%~F
        goto :fail
    )
)

echo.
echo ========================================
echo BUILD COMPLETED - FILE CHECKS PASSED
echo ========================================
echo Package:  %CD%\dist\EduGuard\
echo Start:    dist\EduGuard\EduGuard.exe
echo Ship the whole dist\EduGuard folder (zip it); EduGuard.exe does not work on its own.
echo.
if /i not "%NO_PAUSE%"=="--no-pause" pause
exit /b 0

:tess_fail
echo ERROR: Could not copy Tesseract.
goto :fail

:fail
echo.
echo BUILD FAILED.
if /i not "%NO_PAUSE%"=="--no-pause" pause
exit /b 1
