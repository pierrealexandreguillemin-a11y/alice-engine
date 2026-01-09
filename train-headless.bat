@echo off
REM Entrainement ML SANS VS Code (economise 1-2 GB de RAM)
REM Usage: train-headless.bat [--no-mlflow]

echo ============================================
echo ALICE Engine - Training Mode (Headless)
echo ============================================
echo.

REM Verifier si VS Code tourne
tasklist /FI "IMAGENAME eq Code.exe" 2>NUL | find /I /N "Code.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [WARNING] VS Code est en cours d'execution!
    echo           Ferme-le pour liberer ~1.5 GB de RAM
    echo.
    choice /C ON /M "Continuer quand meme (O) ou Non (N)"
    if errorlevel 2 exit /b 1
)

echo.
echo [INFO] Memoire disponible:
powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem | Select-Object @{N='FreeGB';E={[math]::Round($_.FreePhysicalMemory/1MB,1)}}).FreeGB"
echo GB libres
echo.

echo [INFO] Lancement de l'entrainement...
python -m scripts.train_models_parallel %*

echo.
echo [INFO] Entrainement termine!
pause
