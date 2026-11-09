@echo off
rem Double-click to copy the installed NvPresent64.dll into ..\ref for offline inspection.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get_nvpresent.ps1"
echo.
pause
