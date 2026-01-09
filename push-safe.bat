@echo off
REM Push Git avec gestion memoire (ferme VS Code temporairement)
REM Usage: push-safe.bat [branch]

setlocal enabledelayedexpansion

set BRANCH=%1
if "%BRANCH%"=="" set BRANCH=master

echo ============================================
echo ALICE Engine - Safe Push (RAM optimized)
echo ============================================
echo.

REM Verifier memoire disponible
echo [1/5] Checking available memory...
for /f "tokens=2 delims==" %%a in ('wmic OS get FreePhysicalMemory /value ^| find "="') do set FREE_KB=%%a
set /a FREE_MB=!FREE_KB!/1024
echo       %FREE_MB% MB available

if %FREE_MB% LSS 2000 (
    echo.
    echo [WARNING] Less than 2GB free RAM!
    echo           Pre-push hooks need ~1.5GB
    echo.

    REM Proposer de fermer VS Code
    tasklist /FI "IMAGENAME eq Code.exe" 2>NUL | find /I /N "Code.exe">NUL
    if !errorlevel!==0 (
        echo [INFO] VS Code is running and using ~1.5GB RAM
        choice /C ON /M "Close VS Code temporarily? (O=Yes, N=No)"
        if !errorlevel!==1 (
            echo [INFO] Closing VS Code...
            taskkill /F /IM Code.exe >NUL 2>&1
            timeout /t 2 >NUL
            set REOPEN_VSCODE=1
        )
    )
)

REM Forcer garbage collection Python avant
echo.
echo [2/5] Running garbage collection...
python -c "import gc; gc.collect()"

REM Lancer le push
echo.
echo [3/5] Pushing to %BRANCH%...
echo.
git push origin %BRANCH%
set PUSH_RESULT=!errorlevel!

REM Reouvrir VS Code si ferme
if defined REOPEN_VSCODE (
    echo.
    echo [4/5] Reopening VS Code...
    start "" code .
)

echo.
echo [5/5] Done!
if %PUSH_RESULT%==0 (
    echo       Push successful!
) else (
    echo       Push failed with code %PUSH_RESULT%
)

endlocal
pause
