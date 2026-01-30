# GCP Cloud Run Deployment Script
# Usage: .\deploy-gcp.ps1 -ProjectId "gen-lang-client-0823293183" -Region "asia-northeast3"
# Or: .\deploy-gcp.ps1 (uses default project ID)

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "gen-lang-client-0823293183",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast3",
    
    [Parameter(Mandatory=$false)]
    [switch]$BackendOnly,
    
    [Parameter(Mandatory=$false)]
    [switch]$FrontendOnly
)

# Color output function
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "  GCP Cloud Run Deployment Script"
Write-ColorOutput Green "=========================================="
Write-Output "Project ID: $ProjectId"
Write-Output "Region: $Region"
Write-Output ""

# Check gcloud CLI
Write-Output "Checking gcloud CLI..."
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-ColorOutput Red "ERROR: gcloud CLI is not installed."
    Write-Output "Installation guide: https://cloud.google.com/sdk/docs/install"
    exit 1
}
Write-ColorOutput Green "OK: gcloud CLI found"
Write-Output ""

# Set project
Write-Output "Setting GCP project..."
gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Failed to set project"
    exit 1
}
Write-ColorOutput Green "OK: Project set successfully"
Write-Output ""

# Enable required APIs
Write-Output "Enabling required GCP APIs..."
$apis = @(
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "sqladmin.googleapis.com"
)

foreach ($api in $apis) {
    Write-Output "  - Enabling $api..."
    gcloud services enable $api --quiet
}
Write-ColorOutput Green "OK: APIs enabled"
Write-Output ""

# Load .env.production if exists
$envProdPath = "backend\.env.production"
if (Test-Path $envProdPath) {
    Write-Output "Loading environment variables from .env.production..."
    Get-Content $envProdPath | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($key -and $value) {
                Set-Item -Path "env:$key" -Value $value
                Write-Output "  Loaded: $key"
            }
        }
    }
    Write-Output ""
}

# Check environment variables
Write-Output "Checking environment variables..."
$requiredVars = @(
    "DATABASE_URL",
    "BINANCE_API_KEY",
    "BINANCE_SECRET_KEY"
)

$missingVars = @()
foreach ($var in $requiredVars) {
    if (-not (Get-Item "Env:$var" -ErrorAction SilentlyContinue)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-ColorOutput Yellow "WARNING: The following environment variables are not set:"
    foreach ($var in $missingVars) {
        Write-Output "  - $var"
    }
    Write-Output ""
    Write-Output "To set environment variables:"
    Write-Output "  `$env:DATABASE_URL = 'mysql+aiomysql://user:pass@host:3306/dbname'"
    Write-Output "  `$env:BINANCE_API_KEY = 'your_api_key'"
    Write-Output "  `$env:BINANCE_SECRET_KEY = 'your_secret_key'"
    Write-Output ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Deploy backend
if (-not $FrontendOnly) {
    Write-ColorOutput Cyan "=========================================="
    Write-ColorOutput Cyan "  Deploying Backend"
    Write-ColorOutput Cyan "=========================================="
    Write-Output ""
    
    # Prepare environment variables
    $envVars = @()
    if ($env:DATABASE_URL) {
        $envVars += "DATABASE_URL=$($env:DATABASE_URL)"
    }
    if ($env:BINANCE_API_KEY) {
        $envVars += "BINANCE_API_KEY=$($env:BINANCE_API_KEY)"
    }
    if ($env:BINANCE_SECRET_KEY) {
        $envVars += "BINANCE_SECRET_KEY=$($env:BINANCE_SECRET_KEY)"
    }
    if ($env:BINANCE_TESTNET) {
        $envVars += "BINANCE_TESTNET=$($env:BINANCE_TESTNET)"
    }
    
    $envVarsString = $envVars -join ","
    
    Write-Output "Deploying backend... (This may take a few minutes)"
    Write-Output ""
    
    Push-Location backend
    
    $deployCmd = "gcloud run deploy crypto-backend " +
                 "--source . " +
                 "--platform managed " +
                 "--region $Region " +
                 "--allow-unauthenticated " +
                 "--memory 2Gi " +
                 "--cpu 2 " +
                 "--timeout 600 " +
                 "--max-instances 10 " +
                 "--min-instances 0 " +
                 "--concurrency 80 " +
                 "--port 8080"
    
    if ($envVarsString) {
        $deployCmd += " --set-env-vars=`"$envVarsString`""
    }
    
    Invoke-Expression $deployCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "SUCCESS: Backend deployed!"
        
        # Get deployment URL
        $backendUrl = gcloud run services describe crypto-backend --region $Region --format "value(status.url)" 2>$null
        if ($backendUrl) {
            Write-ColorOutput Green "Backend URL: $backendUrl"
            $env:BACKEND_URL = $backendUrl
        }
    } else {
        Write-ColorOutput Red "ERROR: Backend deployment failed"
        Pop-Location
        exit 1
    }
    
    Pop-Location
    Write-Output ""
}

# Deploy frontend
if (-not $BackendOnly) {
    Write-ColorOutput Cyan "=========================================="
    Write-ColorOutput Cyan "  Deploying Frontend"
    Write-ColorOutput Cyan "=========================================="
    Write-Output ""
    
    # Check backend URL
    if (-not $env:BACKEND_URL) {
        Write-Output "Please enter the backend URL:"
        $backendUrl = Read-Host "Backend URL"
    } else {
        $backendUrl = $env:BACKEND_URL
    }
    
    Write-Output "Deploying frontend... (This may take a few minutes)"
    Write-Output ""
    
    Push-Location frontend
    
    $deployCmd = "gcloud run deploy crypto-frontend " +
                 "--source . " +
                 "--platform managed " +
                 "--region $Region " +
                 "--allow-unauthenticated " +
                 "--memory 512Mi " +
                 "--cpu 1 " +
                 "--build-arg=`"VITE_API_URL=$backendUrl`""
    
    Invoke-Expression $deployCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "SUCCESS: Frontend deployed!"
        
        # Get deployment URL
        $frontendUrl = gcloud run services describe crypto-frontend --region $Region --format "value(status.url)" 2>$null
        if ($frontendUrl) {
            Write-ColorOutput Green "Frontend URL: $frontendUrl"
        }
    } else {
        Write-ColorOutput Red "ERROR: Frontend deployment failed"
        Pop-Location
        exit 1
    }
    
    Pop-Location
    Write-Output ""
}

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "  Deployment Complete!"
Write-ColorOutput Green "=========================================="
Write-Output ""
Write-Output "You can check service status with:"
Write-Output "  gcloud run services list --region $Region"
Write-Output ""
