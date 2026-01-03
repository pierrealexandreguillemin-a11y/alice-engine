# setup-dev.ps1 - Configuration environnement dev Alice-Engine
# Usage: .\scripts\setup-dev.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Alice-Engine - Setup Dev Environment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Ajouter Python Scripts au PATH
$pythonScripts = "$env:APPDATA\Python\Python313\Scripts"
if (Test-Path $pythonScripts) {
    if ($env:PATH -notlike "*$pythonScripts*") {
        $env:PATH = "$env:PATH;$pythonScripts"
        Write-Host "[OK] Python Scripts ajoute au PATH" -ForegroundColor Green
    } else {
        Write-Host "[OK] Python Scripts deja dans PATH" -ForegroundColor Green
    }
} else {
    Write-Host "[!] Python Scripts non trouve: $pythonScripts" -ForegroundColor Yellow
}

# 2. Installer dependances
Write-Host ""
Write-Host "Installation des dependances..." -ForegroundColor Cyan
pip install -r requirements.txt -q
pip install -r requirements-dev.txt -q
Write-Host "[OK] Dependances installees" -ForegroundColor Green

# 3. Installer hooks pre-commit
Write-Host ""
Write-Host "Installation des hooks Git..." -ForegroundColor Cyan
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
Write-Host "[OK] Hooks installes" -ForegroundColor Green

# 4. Verifier outils
Write-Host ""
Write-Host "Verification des outils..." -ForegroundColor Cyan

$tools = @("ruff", "mypy", "pytest", "bandit", "radon", "xenon", "pydeps", "pre-commit")
foreach ($tool in $tools) {
    $result = Get-Command $tool -ErrorAction SilentlyContinue
    if ($result) {
        Write-Host "  [OK] $tool" -ForegroundColor Green
    } else {
        Write-Host "  [!] $tool non trouve" -ForegroundColor Yellow
    }
}

# 5. Ajouter PATH permanent (optionnel)
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup termine!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pour ajouter le PATH de facon permanente:" -ForegroundColor Yellow
Write-Host '  [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";' + $pythonScripts + '", "User")' -ForegroundColor Gray
Write-Host ""
Write-Host "Commandes disponibles:" -ForegroundColor Cyan
Write-Host "  make help      - Voir toutes les commandes"
Write-Host "  make quality   - Lint + Format + Typecheck"
Write-Host "  make test      - Lancer les tests"
Write-Host "  make all       - Validation complete"
Write-Host ""
