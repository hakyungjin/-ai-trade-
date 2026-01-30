# GCP Cloud Run Deployment Script with Supabase
# Usage: .\deploy-with-supabase.ps1
# Or: .\deploy-with-supabase.ps1 -SupabasePassword "your_password"

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "gen-lang-client-0823293183",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast3",
    
    [Parameter(Mandatory=$false)]
    [string]$SupabasePassword,
    
    [Parameter(Mandatory=$false)]
    [string]$BinanceApiKey,
    
    [Parameter(Mandatory=$false)]
    [string]$BinanceSecretKey
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
Write-ColorOutput Green "  GCP Cloud Run Deployment with Supabase"
Write-ColorOutput Green "=========================================="
Write-Output "Project ID: $ProjectId"
Write-Output "Region: $Region"
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

# Supabase connection string (override if provided via parameter)
if ($SupabasePassword) {
    $supabaseUrl = "postgresql+asyncpg://postgres:$SupabasePassword@db.vmiinfjxpnoevsehhzey.supabase.co:5432/postgres"
    $env:DATABASE_URL = $supabaseUrl
}

# Binance API keys (override if provided)
if ($BinanceApiKey) {
    $env:BINANCE_API_KEY = $BinanceApiKey
}
if ($BinanceSecretKey) {
    $env:BINANCE_SECRET_KEY = $BinanceSecretKey
}

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

# Check environment variables
Write-Output "Environment variables:"
if ($env:DATABASE_URL) {
    Write-Output "  DATABASE_URL: Configured"
} else {
    Write-ColorOutput Red "  DATABASE_URL: Not set (required)"
    Write-Output "Please set DATABASE_URL in backend/.env.production"
    exit 1
}
if ($env:BINANCE_API_KEY) {
    Write-Output "  BINANCE_API_KEY: Configured"
} else {
    Write-ColorOutput Yellow "  BINANCE_API_KEY: Not set (optional)"
}
if ($env:BINANCE_SECRET_KEY) {
    Write-Output "  BINANCE_SECRET_KEY: Configured"
} else {
    Write-ColorOutput Yellow "  BINANCE_SECRET_KEY: Not set (optional)"
}
Write-Output ""

# Deploy backend
Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Deploying Backend"
Write-ColorOutput Cyan "=========================================="
Write-Output ""

# Prepare environment variables
$envVars = @()

# DATABASE_URL 설정 (우선순위: 파라미터 > 환경변수 > .env.production)
if ($SupabasePassword) {
    $supabaseUrl = "postgresql+asyncpg://postgres:$SupabasePassword@db.vmiinfjxpnoevsehhzey.supabase.co:5432/postgres"
    $env:DATABASE_URL = $supabaseUrl
}

if ($env:DATABASE_URL) {
    $envVars += "DATABASE_URL=$($env:DATABASE_URL)"
} else {
    Write-ColorOutput Red "ERROR: DATABASE_URL is required"
    Write-Output "Please set DATABASE_URL in backend/.env.production or provide -SupabasePassword"
    exit 1
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
             "--port 8080 " +
             "--set-env-vars=`"$envVarsString`""

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

# Deploy frontend
Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Deploying Frontend"
Write-ColorOutput Cyan "=========================================="
Write-Output ""

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
        Write-Output ""
        Write-ColorOutput Cyan "=========================================="
        Write-ColorOutput Cyan "  Deployment Summary"
        Write-ColorOutput Cyan "=========================================="
        Write-Output "Frontend: $frontendUrl"
        Write-Output "Backend:  $backendUrl"
        Write-Output "Database: Supabase (PostgreSQL)"
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

