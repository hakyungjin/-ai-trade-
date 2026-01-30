# Firebase Hosting Frontend Deployment Script
# Usage: .\deploy-frontend-firebase.ps1
# Or: .\deploy-frontend-firebase.ps1 -ProjectId "your-firebase-project-id"

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "gen-lang-client-0823293183"
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
Write-ColorOutput Green "  Firebase Hosting Frontend Deployment"
Write-ColorOutput Green "=========================================="
Write-Output "Project ID: $ProjectId"
Write-Output ""

# Check Firebase CLI
Write-Output "Checking Firebase CLI..."
if (-not (Get-Command firebase -ErrorAction SilentlyContinue)) {
    Write-ColorOutput Red "ERROR: Firebase CLI is not installed."
    Write-Output "Installation: npm install -g firebase-tools"
    Write-Output "Then run: firebase login"
    exit 1
}
Write-ColorOutput Green "OK: Firebase CLI found"
Write-Output ""

# Check if logged in
Write-Output "Checking Firebase login status..."
$loginStatus = firebase projects:list 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Yellow "WARNING: Not logged in to Firebase"
    Write-Output "Please run: firebase login"
    exit 1
}
Write-ColorOutput Green "OK: Logged in to Firebase"
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

# Set Firebase project
Write-Output "Setting Firebase project..."
Push-Location frontend
firebase use $ProjectId
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Failed to set Firebase project"
    Pop-Location
    exit 1
}
Write-ColorOutput Green "OK: Firebase project set"
Write-Output ""

# Check if Firebase Hosting is initialized
Write-Output "Checking Firebase Hosting configuration..."
if (-not (Test-Path "firebase.json")) {
    Write-ColorOutput Yellow "WARNING: firebase.json not found. Initializing Firebase Hosting..."
    firebase init hosting --project $ProjectId --public dist --yes
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "ERROR: Failed to initialize Firebase Hosting"
        Pop-Location
        exit 1
    }
} else {
    Write-ColorOutput Green "OK: Firebase Hosting configuration found"
}

# Ensure Firebase Hosting is enabled for the project
Write-Output "Ensuring Firebase Hosting is enabled..."
firebase hosting:sites:list --project $ProjectId 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Yellow "WARNING: Firebase Hosting may not be enabled. Please enable it in Firebase Console:"
    Write-Output "  https://console.firebase.google.com/project/$ProjectId/hosting"
    Write-Output ""
    Write-Output "Or run: firebase init hosting"
}
Write-Output ""

# Build frontend
Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Building Frontend"
Write-ColorOutput Cyan "=========================================="
Write-Output ""

# Set environment variables for build
if ($env:VITE_API_URL) {
    $env:VITE_API_URL = $env:VITE_API_URL
    Write-Output "Using VITE_API_URL: $env:VITE_API_URL"
}
if ($env:VITE_WS_URL) {
    $env:VITE_WS_URL = $env:VITE_WS_URL
    Write-Output "Using VITE_WS_URL: $env:VITE_WS_URL"
}

Write-Output "Building frontend... (This may take a few minutes)"
Write-Output ""

npm run build

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Build failed"
    Pop-Location
    exit 1
}

Write-ColorOutput Green "OK: Build completed"
Write-Output ""

# Deploy to Firebase Hosting
Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Deploying to Firebase Hosting"
Write-ColorOutput Cyan "=========================================="
Write-Output ""

# Deploy to hosting
# Note: If you get "no site name or target name" error, 
# you need to enable Firebase Hosting in Firebase Console first
Write-Output "Deploying to Firebase Hosting..."
Write-Output "Note: If this fails, make sure Firebase Hosting is enabled in Firebase Console"
Write-Output "  https://console.firebase.google.com/project/$ProjectId/hosting"
Write-Output ""

firebase deploy --only hosting

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "SUCCESS: Frontend deployed to Firebase Hosting!"
    Write-Output ""
    
    Write-ColorOutput Green "Deployment URL: https://$ProjectId.web.app"
    Write-ColorOutput Green "Preview URL: https://$ProjectId.firebaseapp.com"
} else {
    Write-ColorOutput Red "ERROR: Firebase deployment failed"
    Write-Output ""
    Write-Output "Troubleshooting:"
    Write-Output "1. Make sure Firebase Hosting is enabled in Firebase Console"
    Write-Output "2. Run: firebase init hosting"
    Write-Output "3. Or create a site manually in Firebase Console"
    Pop-Location
    exit 1
}

Pop-Location
Write-Output ""

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "  Deployment Complete!"
Write-ColorOutput Green "=========================================="
Write-Output ""
Write-Output "Your frontend is now live at:"
Write-Output "  https://$ProjectId.web.app"
Write-Output "  https://$ProjectId.firebaseapp.com"
Write-Output ""

