# PowerShell script to set up both backend and frontend

Write-Host "=== Project Setup ===" -ForegroundColor Cyan

# Setup Backend
Write-Host "`n[1/2] Setting up Backend..." -ForegroundColor Yellow
if (-not (Test-Path "backend\.venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv backend\.venv
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}

Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Yellow
& "backend\.venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install -r backend\requirements.txt
Write-Host "Backend setup complete!" -ForegroundColor Green

# Setup Frontend
Write-Host "`n[2/2] Setting up Frontend..." -ForegroundColor Yellow
Set-Location frontend-react
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "Frontend setup complete!" -ForegroundColor Green
} else {
    Write-Host "Node modules already installed" -ForegroundColor Green
}

Set-Location ..

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "`nTo run the backend:" -ForegroundColor Cyan
Write-Host "  .\run-backend.ps1" -ForegroundColor White
Write-Host "`nTo run the frontend:" -ForegroundColor Cyan
Write-Host "  .\run-frontend.ps1" -ForegroundColor White
Write-Host "`nNote: You'll need two terminal windows - one for backend and one for frontend" -ForegroundColor Yellow

