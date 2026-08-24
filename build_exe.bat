@echo off
setlocal

echo ========================================
echo       Building EduGuard
echo ========================================
echo.

echo Step 1: Building EduGuard.exe...
.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm EduGuard.spec
if errorlevel 1 (
    echo.
    echo ERROR: Main app build failed.
    exit /b 1
)
echo.

echo Step 2: Verifying main executable...
if not exist "dist\EduGuard\EduGuard.exe" (
    echo ERROR: EduGuard.exe was not created.
    exit /b 1
)
echo EduGuard.exe verified.
echo.

echo Step 3: Copying frontend standalone build...
if not exist "frontend\.next\standalone\server.js" (
    echo ERROR: frontend\.next\standalone\server.js not found.
    exit /b 1
)

xcopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\standalone\" /E /I /Y
if errorlevel 4 (
    echo ERROR: Failed to copy frontend standalone build.
    exit /b 1
)

echo Frontend standalone build copied.
echo.

echo Step 4: Copying frontend public folder...
if exist "frontend\public" (
    xcopy "frontend\public" "dist\EduGuard\frontend\public\" /E /I /Y
    if errorlevel 4 (
        echo ERROR: Failed to copy frontend public folder.
        exit /b 1
    )
)
echo Public folder copied.
echo.

echo Step 5: Copying models...
if not exist "models" (
    echo ERROR: models directory not found.
    exit /b 1
)

xcopy "models" "dist\EduGuard\models\" /E /I /Y
if errorlevel 4 (
    echo ERROR: Failed to copy models.
    exit /b 1
)
echo Models copied.
echo.

echo Step 6: Copying EduGuardBackend.exe...
if not exist "dist\EduGuardBackend.exe" (
    echo ERROR: dist\EduGuardBackend.exe not found.
    echo Build the backend first using:
    echo .venv\Scripts\python.exe -m PyInstaller --clean --noconfirm EduGuardBackend.spec
    exit /b 1
)

copy /Y "dist\EduGuardBackend.exe" "dist\EduGuard\EduGuardBackend.exe"
if errorlevel 1 (
    echo ERROR: Failed to copy EduGuardBackend.exe.
    exit /b 1
)
echo Backend executable copied.
echo.

echo Step 7: Final verification...

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
    echo ERROR: Qwen model missing.
    exit /b 1
)

echo.
echo ========================================
echo          BUILD COMPLETE
echo ========================================
echo.
echo EduGuard.exe              OK
echo EduGuardBackend.exe       OK
echo Next.js standalone       OK
echo Models                    OK
echo.
echo Package: dist\EduGuard\
echo.

pause
endlocal
