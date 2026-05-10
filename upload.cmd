@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Project History Upload Tool
echo ===================================================
echo.
echo This script will:
echo 1. Initialize a new git repo in 'upload_to_github'
echo 2. Generate 150+ commits with historical timestamps
echo 3. Push every commit to GitHub
echo.

:: Run the execution script
python execute_upload.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The upload process failed.
    echo Check 'upload_to_github\upload_log.txt' for details.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Everything has been uploaded to GitHub!
pause
