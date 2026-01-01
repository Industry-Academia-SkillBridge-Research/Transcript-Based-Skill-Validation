# PowerShell script to run the backend FastAPI server

Write-Host "=== Backend Setup and Run ===" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "backend\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv backend\.venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "backend\.venv\Scripts\Activate.ps1"

# Install/upgrade dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r backend\requirements.txt

# Change to backend directory
Set-Location backend

# Run the FastAPI server
Write-Host "`nStarting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press CTRL+C to stop the server`n" -ForegroundColor Yellow

uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

