# Check GCP Cloud Run Deployment Status
# Usage: .\check-deployment-status.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "gen-lang-client-0823293183",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast3"
)

Write-Output "=========================================="
Write-Output "  Checking Deployment Status"
Write-Output "=========================================="
Write-Output ""

# Set project
gcloud config set project $ProjectId --quiet

# Check backend service
Write-Output "Backend Service Status:"
Write-Output "----------------------"
gcloud run services describe crypto-backend --region $Region --format="table(status.url,status.conditions[0].status,status.latestReadyRevisionName)" 2>&1
Write-Output ""

# Check frontend service
Write-Output "Frontend Service Status:"
Write-Output "----------------------"
gcloud run services describe crypto-frontend --region $Region --format="table(status.url,status.conditions[0].status,status.latestReadyRevisionName)" 2>&1
Write-Output ""

# List all services
Write-Output "All Cloud Run Services:"
Write-Output "----------------------"
gcloud run services list --region $Region
Write-Output ""

# Check recent builds
Write-Output "Recent Builds:"
Write-Output "----------------------"
gcloud builds list --limit=5 --format="table(id,status,createTime,duration)"
Write-Output ""

