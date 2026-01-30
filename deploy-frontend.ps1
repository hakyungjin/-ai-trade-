# GCP Cloud Run Frontend Deployment Script
# Usage: .\deploy-frontend.ps1 -BackendUrl "https://crypto-backend-xxxxx-du.a.run.app" -Region "asia-northeast3"
# Or: .\deploy-frontend.ps1 (will prompt for backend URL)

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "gen-lang-client-0823293183",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast3",
    
    [Parameter(Mandatory=$false)]
    [string]$BackendUrl
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
Write-ColorOutput Green "  GCP Cloud Run Frontend Deployment"
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
    "cloudbuild.googleapis.com"
)

foreach ($api in $apis) {
    Write-Output "  - Enabling $api..."
    gcloud services enable $api --quiet
}
Write-ColorOutput Green "OK: APIs enabled"
Write-Output ""

# Load .env.production if exists
$envProdPath = "frontend\.env.production"
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

# Get backend URL from environment variable or parameter
if (-not $BackendUrl) {
    if ($env:VITE_API_URL) {
        $BackendUrl = $env:VITE_API_URL
        Write-ColorOutput Green "Using VITE_API_URL from .env.production: $BackendUrl"
    }
}

# Get backend URL
if (-not $BackendUrl) {
    # Try to get from existing service
    Write-Output "Checking for existing backend service..."
    $existingBackend = gcloud run services describe crypto-backend --region $Region --format "value(status.url)" 2>$null
    
    if ($existingBackend) {
        Write-ColorOutput Green "Found existing backend: $existingBackend"
        $BackendUrl = $existingBackend
        $useExisting = Read-Host "Use this backend URL? (y/n)"
        if ($useExisting -ne "y") {
            $BackendUrl = $null
        }
    }
    
    if (-not $BackendUrl) {
        Write-Output ""
        Write-Output "Please enter the backend URL:"
        Write-Output "Example: https://crypto-backend-xxxxx-du.a.run.app"
        $BackendUrl = Read-Host "Backend URL"
    }
}

# Validate backend URL
if (-not $BackendUrl -or $BackendUrl -notmatch "^https?://") {
    Write-ColorOutput Red "ERROR: Invalid backend URL format"
    Write-Output "URL must start with http:// or https://"
    exit 1
}

Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Deploying Frontend"
Write-ColorOutput Cyan "=========================================="
Write-Output ""
Write-Output "Backend URL: $BackendUrl"
Write-Output "Deploying frontend... (This may take a few minutes)"
Write-Output ""

Push-Location frontend

# Prepare build environment variables
$buildEnvVars = @()
if ($BackendUrl) {
    $buildEnvVars += "VITE_API_URL=$BackendUrl"
} elseif ($env:VITE_API_URL) {
    $buildEnvVars += "VITE_API_URL=$($env:VITE_API_URL)"
}

if ($env:VITE_WS_URL) {
    $buildEnvVars += "VITE_WS_URL=$($env:VITE_WS_URL)"
}

# Add any other VITE_* environment variables from .env.production
Get-ChildItem Env: | Where-Object { $_.Name -like "VITE_*" } | ForEach-Object {
    $varString = "$($_.Name)=$($_.Value)"
    if ($buildEnvVars -notcontains $varString) {
        $buildEnvVars += $varString
    }
}

$deployCmd = "gcloud run deploy crypto-frontend " +
             "--source . " +
             "--platform managed " +
             "--region $Region " +
             "--allow-unauthenticated " +
             "--memory 512Mi " +
             "--cpu 1"

# Add build environment variables
if ($buildEnvVars.Count -gt 0) {
    $buildEnvVarsString = $buildEnvVars -join ","
    $deployCmd += " --set-build-env-vars=`"$buildEnvVarsString`""
}

Invoke-Expression $deployCmd

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "SUCCESS: Frontend deployed!"
    
    # Get deployment URL
    $frontendUrl = gcloud run services describe crypto-frontend --region $Region --format "value(status.url)" 2>$null
    if ($frontendUrl) {
        Write-ColorOutput Green "Frontend URL: $frontendUrl"
        Write-Output ""
        Write-ColorOutput Cyan "=========================================="
        Write-ColorOutput Cyan "  Deployment Summary"
        Write-ColorOutput Cyan "=========================================="
        Write-Output "Frontend: $frontendUrl"
        Write-Output "Backend:  $BackendUrl"
        Write-Output ""
        Write-Output "Open your browser and visit: $frontendUrl"
    }
} else {
    Write-ColorOutput Red "ERROR: Frontend deployment failed"
    Pop-Location
    exit 1
}

Pop-Location
Write-Output ""

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "  Deployment Complete!"
Write-ColorOutput Green "=========================================="
Write-Output ""
Write-Output "You can check service status with:"
Write-Output "  gcloud run services list --region $Region"
Write-Output ""

