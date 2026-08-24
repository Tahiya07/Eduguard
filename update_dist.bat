@echo off
echo Updating EduGuard dist folder with latest changes...
echo.

echo Step 1: Copying updated frontend build...
if exist "dist\EduGuard\frontend\.next" rmdir /s /q "dist\EduGuard\frontend\.next"
xcopy "frontend\.next\standalone" "dist\EduGuard\frontend\.next\" /E /I /Y
echo.

echo Step 2: Copying updated public folder...
if exist "dist\EduGuard\frontend\public" rmdir /s /q "dist\EduGuard\frontend\public"
xcopy "frontend\public" "dist\EduGuard\frontend\public" /E /I /Y
echo.

echo Step 3: Copying updated models...
if exist "dist\EduGuard\models" rmdir /s /q "dist\EduGuard\models"
xcopy "models" "dist\EduGuard\models" /E /I /Y
echo.

echo Step 4: Copying updated backend code...
if exist "dist\EduGuardBackend" rmdir /s /q "dist\EduGuardBackend"
xcopy "backend" "dist\EduGuardBackend\backend" /E /I /Y
copy "backend_launcher.py" "dist\EduGuardBackend\"
copy "bloom_prompt.py" "dist\EduGuardBackend\"
copy "predict_bloom.py" "dist\EduGuardBackend\"
copy "qwen_gguf_cli.py" "dist\EduGuardBackend\"
copy "multi_slm.py" "dist\EduGuardBackend\"
echo.

echo ========================================
echo Update complete!
echo ========================================
echo.
echo Updated files in: dist\EduGuard\
echo - Latest frontend build
echo - Latest models  
echo - Latest backend code (including new bloom_prompt.py)
echo.
echo Note: EXEs remain unchanged. Use build_exe.bat to rebuild EXEs if needed.
echo.
pause