# C-ML Build Script for Windows
# Usage: .\build.ps1 [command]
# Commands: all, lib, frontend, install, clean, help

param(
    [Parameter(Position=0)]
    [string]$Command = "all"
)

# Colors for output
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

Write-Host "C-ML Build Script" -ForegroundColor Blue
Write-Host "==================" -ForegroundColor Blue
Write-Host "Platform: Windows`n" -ForegroundColor Green

# Check dependencies
function Check-Dependencies {
    Write-Info "Checking dependencies..."

    $missingDeps = @()

    # Check for CMake
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        $missingDeps += "cmake"
    }

    # Check for C compiler (either MSVC or MinGW)
    $hasMSVC = Get-Command cl -ErrorAction SilentlyContinue
    $hasMinGW = Get-Command gcc -ErrorAction SilentlyContinue

    if (-not $hasMSVC -and -not $hasMinGW) {
        $missingDeps += "Visual Studio or MinGW"
    }

    # Check for Python
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Warning-Custom "python not found - visualization server will not work"
    }

    # Check for Node.js
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warning-Custom "npm not found - frontend build will be skipped"
    }

    if ($missingDeps.Count -gt 0) {
        Write-Error-Custom "Missing required dependencies: $($missingDeps -join ', ')"
        Write-Host "`nInstall instructions:"
        Write-Host "  CMake:         https://cmake.org/download/"
        Write-Host "  Visual Studio: https://visualstudio.microsoft.com/"
        Write-Host "  MinGW:         https://www.mingw-w64.org/"
        Write-Host "  Python:        https://www.python.org/downloads/"
        Write-Host "  Node.js:       https://nodejs.org/"
        exit 1
    }

    Write-Success "All required dependencies found"
}

# Build the C library
function Build-Library {
    Write-Info "Building C library..."

    if (-not (Test-Path "build")) {
        New-Item -ItemType Directory -Path "build" | Out-Null
    }

    Set-Location build

    # Configure with CMake
    Write-Info "Configuring with CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Release

    # Build
    Write-Info "Compiling..."
    cmake --build . --config Release

    Set-Location ..
    Write-Success "Library built successfully"
}

# Build the frontend
function Build-Frontend {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warning-Custom "npm not found - skipping frontend build"
        return
    }

    Write-Info "Building frontend..."
    Set-Location viz-ui

    if (-not (Test-Path "node_modules")) {
        Write-Info "Installing npm dependencies..."
        npm install
    }

    npm run build
    Set-Location ..
    Write-Success "Frontend built successfully"
}

# Install the library
function Install-Library {
    Write-Info "Installing C-ML library..."

    # Check for admin privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Warning-Custom "Installation requires administrator privileges"
        Write-Host "Please run this script as Administrator or use:"
        Write-Host "  cmake --install build --prefix C:\cml"
        exit 1
    }

    Set-Location build
    cmake --install . --config Release
    Set-Location ..

    Write-Success "Library installed successfully"

    Write-Host "`nInstallation complete!" -ForegroundColor Green
    Write-Host "`nTo use the library:"
    Write-Host "  #include <cml/cml.h>"
    Write-Host "  cl your_code.c cml.lib"
    Write-Host "`nTo enable visualization:"
    Write-Host "  `$env:VIZ=`"1`""
    Write-Host "  .\your_program.exe"
}

# Clean build artifacts
function Clean-Build {
    Write-Info "Cleaning build artifacts..."

    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }

    if (Test-Path "viz-ui\dist") {
        Remove-Item -Recurse -Force "viz-ui\dist"
    }

    if (Test-Path "viz-ui\node_modules") {
        Write-Info "Removing node_modules..."
        Remove-Item -Recurse -Force "viz-ui\node_modules"
    }

    Write-Success "Clean complete"
}

# Run tests
function Run-Tests {
    Write-Info "Running tests..."
    Set-Location build
    ctest --output-on-failure
    Set-Location ..
    Write-Success "All tests passed"
}

# Show help
function Show-Help {
    Write-Host @"

C-ML Build Script for Windows

Usage: .\build.ps1 [command]

Commands:
  all         Build everything (library + frontend)
  lib         Build only the C library
  frontend    Build only the frontend
  install     Install the library (requires admin)
  test        Run tests
  clean       Clean build artifacts
  help        Show this help message

Examples:
  .\build.ps1 all          # Build library and frontend
  .\build.ps1 install      # Install to Program Files
  .\build.ps1 clean        # Clean all build files

Environment Variables:
  CMAKE_INSTALL_PREFIX    Installation prefix

"@
}

# Main command handler
switch ($Command.ToLower()) {
    "all" {
        Check-Dependencies
        Build-Library
        Build-Frontend
        Write-Success "Build complete!"
        Write-Host "`nNext steps:"
        Write-Host "  1. Run '.\build.ps1 install' as Administrator to install system-wide"
        Write-Host "  2. Or link against build\lib\Release\cml.lib directly"
    }
    "lib" {
        Check-Dependencies
        Build-Library
    }
    "frontend" {
        Build-Frontend
    }
    "install" {
        Install-Library
    }
    "test" {
        Run-Tests
    }
    "clean" {
        Clean-Build
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error-Custom "Unknown command: $Command"
        Show-Help
        exit 1
    }
}
