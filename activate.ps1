#!/usr/bin/env pwsh
# Activate virtual environment and prepare for story generation

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

Write-Host ""
Write-Host "Virtual environment ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Yellow
Write-Host "  python run.py --model deepseek/deepseek-r1" -ForegroundColor Gray
Write-Host "  python run.py --model openai/gpt-5.2 --threads 10" -ForegroundColor Gray
Write-Host "  python run.py --help" -ForegroundColor Gray
Write-Host ""
