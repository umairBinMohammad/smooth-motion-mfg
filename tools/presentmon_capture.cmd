@echo off
rem Records 25 seconds of display timing for GRB.exe with Intel PresentMon.
rem Put PresentMon-<version>-x64.exe in this tools folder first
rem (https://github.com/GameTechDev/PresentMon/releases).
rem Usage: presentmon_capture.cmd <label>     e.g.  presentmon_capture.cmd 2x
setlocal
set LABEL=%1
if "%LABEL%"=="" set LABEL=capture
set PM=
for %%f in ("%~dp0PresentMon-*-x64.exe") do set PM=%%f
if "%PM%"=="" (
    echo PresentMon-*-x64.exe was not found in %~dp0
    pause
    exit /b 1
)
if not exist "%~dp0..\out" mkdir "%~dp0..\out"
echo Recording 25 seconds to out\pm_%LABEL%.csv ...
"%PM%" --process_name GRB.exe --output_file "%~dp0..\out\pm_%LABEL%.csv" --timed 25 --terminate_after_timed --restart_as_admin
echo Done.
pause
