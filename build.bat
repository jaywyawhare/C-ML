@echo off
REM C-ML Build Launcher for Windows
REM This batch file detects PowerShell and runs the build script

echo C-ML Build System
echo ==================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: PowerShell not found
    echo Please install PowerShell or use build.ps1 directly
    exit /b 1
)

REM Run the PowerShell build script
powershell -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
