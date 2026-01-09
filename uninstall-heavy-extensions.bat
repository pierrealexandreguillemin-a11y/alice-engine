@echo off
echo Desinstallation des extensions lourdes...
call code --uninstall-extension github.copilot
call code --uninstall-extension github.copilot-chat
call code --uninstall-extension continue.continue
call code --uninstall-extension mongodb.mongodb-vscode
call code --uninstall-extension ms-azuretools.vscode-containers
call code --uninstall-extension remiscan.dependency-cruiser-ts
echo.
echo Termine! Redémarre VS Code.
pause
