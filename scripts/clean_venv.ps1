param(
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath ".venv") {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}

Write-Host "Creating virtual environment using" $Python
& $Python -m venv .venv

$venvPath = (Resolve-Path .\.venv).Path
$venvPython = Join-Path $venvPath "Scripts\\python.exe"

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing requirements..."
& $venvPython -m pip install --requirement requirements.txt

Write-Host "Virtual environment ready at" $venvPath
