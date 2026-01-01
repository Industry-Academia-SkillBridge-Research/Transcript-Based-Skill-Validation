# PowerShell script to run the frontend React app

Write-Host "=== Frontend Setup and Run ===" -ForegroundColor Cyan

# Change to frontend directory
Set-Location frontend-react

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Run the development server
Write-Host "`nStarting Vite development server..." -ForegroundColor Green
Write-Host "Press CTRL+C to stop the server`n" -ForegroundColor Yellow

npm run dev

