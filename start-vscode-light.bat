@echo off
REM Lance VS Code en mode leger (Claude Code uniquement)
REM Usage: start-vscode-light.bat [ultra]
REM   ultra = desactive AUSSI Pylance (mode minimal)

if "%1"=="ultra" (
    echo Mode ULTRA-LEGER: Pylance desactive
    code --disable-extension github.copilot ^
         --disable-extension github.copilot-chat ^
         --disable-extension continue.continue ^
         --disable-extension ms-python.python ^
         --disable-extension ms-python.vscode-pylance ^
         --disable-extension ms-python.debugpy ^
         --disable-extension ms-python.vscode-python-envs ^
         --disable-extension dbaeumer.vscode-eslint ^
         --disable-extension esbenp.prettier-vscode ^
         --disable-extension mongodb.mongodb-vscode ^
         --disable-extension ms-azuretools.vscode-containers ^
         --disable-extension ms-vscode.powershell ^
         --disable-extension remiscan.dependency-cruiser-ts ^
         --disable-extension tomoki1207.pdf ^
         --disable-extension mechatroner.rainbow-csv ^
         .
) else (
    echo Mode LEGER: Pylance actif en mode minimal
    code --disable-extension github.copilot ^
         --disable-extension github.copilot-chat ^
         --disable-extension continue.continue ^
         --disable-extension ms-python.debugpy ^
         --disable-extension dbaeumer.vscode-eslint ^
         --disable-extension esbenp.prettier-vscode ^
         --disable-extension mongodb.mongodb-vscode ^
         --disable-extension ms-azuretools.vscode-containers ^
         --disable-extension ms-vscode.powershell ^
         --disable-extension remiscan.dependency-cruiser-ts ^
         --disable-extension tomoki1207.pdf ^
         --disable-extension mechatroner.rainbow-csv ^
         .
)
