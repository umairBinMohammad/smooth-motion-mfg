@echo off
rem Records 75 seconds of display timing for GRB.exe with Intel PresentMon
rem (PresentMon-<version>-x64.exe in this folder, from
rem https://github.com/GameTechDev/PresentMon/releases). Output: ..\out\pm_capture.csv
setlocal
set PM=
for %%f in ("%~dp0PresentMon-*-x64.exe") do set PM=%%f
if "%PM%"=="" (
    echo PresentMon-*-x64.exe was not found in %~dp0
    pause
    exit /b 1
)
if not exist "%~dp0..\out" mkdir "%~dp0..\out"
echo Recording 75 seconds. Switch to the game now and press F10.
"%PM%" --process_name GRB.exe --output_file "%~dp0..\out\pm_capture.csv" --v1_metrics --qpc_time --timed 75 --terminate_after_timed --stop_existing_session --restart_as_admin
