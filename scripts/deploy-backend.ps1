# Deploy Backend to Cloud Run
# Usage: ./scripts/deploy-backend.ps1

Write-Host "Starting backend deployment to Cloud Run..." -ForegroundColor Cyan

$projectId = "gen-lang-client-0823293183"
$region = "asia-northeast3"
$serviceName = "crypto-backend"

# 1. Change to backend directory
Set-Location -Path (Join-Path $PSScriptRoot "..\backend")
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# 2. Deploy to Cloud Run
Write-Host "Deploying... Please wait." -ForegroundColor Yellow

gcloud run deploy $serviceName `
    --source . `
    --platform managed `
    --region $region `
    --allow-unauthenticated `
    --project $projectId

if ($LASTEXITCODE -eq 0) {
    Write-Host "Backend deployment completed!" -ForegroundColor Green
    Write-Host "Service URL: https://$serviceName-162307894443.$region.run.app" -ForegroundColor Cyan
} else {
    Write-Host "Deployment failed." -ForegroundColor Red
    exit 1
}
