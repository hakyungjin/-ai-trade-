# Deploy Frontend to Firebase
# Usage: ./scripts/deploy-frontend.ps1

Write-Host "Starting frontend deployment to Firebase..." -ForegroundColor Cyan

# 1. Change to frontend directory
Set-Location -Path (Join-Path $PSScriptRoot "..\frontend")
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# 2. Set Firebase project
Write-Host "Setting Firebase project..." -ForegroundColor Yellow
firebase use ai-trader-e69ae

# 3. Build frontend
Write-Host "Building frontend..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

# 4. Deploy to Firebase
Write-Host "Deploying to Firebase... Please wait." -ForegroundColor Yellow
firebase deploy

if ($LASTEXITCODE -eq 0) {
    Write-Host "Frontend deployment completed!" -ForegroundColor Green
    Write-Host "App URL: https://ai-trader-e69ae.web.app" -ForegroundColor Cyan
} else {
    Write-Host "Deployment failed." -ForegroundColor Red
    exit 1
}
