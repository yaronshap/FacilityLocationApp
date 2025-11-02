# GitHub Setup Script for Facility Location Project
# Run this script from PowerShell in your project directory

Write-Host "================================" -ForegroundColor Cyan
Write-Host "GitHub Setup for Facility Location Project" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Make initial commit
Write-Host "Step 1: Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Facility Location Optimization App

- Streamlit app for 5 facility location problems
- LSCP, MCLP, P-Median, P-Center, SPLP implementations
- Integer Programming and Complete Enumeration methods
- Interactive visualizations and solution comparisons
- Unit tests for solver validation"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Initial commit created successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Error creating commit" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to GitHub: https://github.com/new" -ForegroundColor White
Write-Host "2. Create a new repository with these settings:" -ForegroundColor White
Write-Host "   - Repository name: facility-location-optimization" -ForegroundColor Gray
Write-Host "   - Description: Interactive Streamlit app for facility location optimization problems" -ForegroundColor Gray
Write-Host "   - Visibility: Public or Private (your choice)" -ForegroundColor Gray
Write-Host "   - DO NOT initialize with README, .gitignore, or license" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. After creating the repository, GitHub will show you commands." -ForegroundColor White
Write-Host "   Use these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/facility-location-optimization.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Replace YOUR_USERNAME with your actual GitHub username" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Current Git Status:" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
git log --oneline -1
Write-Host ""
git status
Write-Host ""
Write-Host "Your local repository is ready to push to GitHub!" -ForegroundColor Green

