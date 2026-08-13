$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

if (-not $candidates) {
    Write-Host 'Python 3.10 or newer was not found. Install Python and try again.' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

$pythonBin = $candidates[0]
Write-Host 'Starting HeartLens. Press Ctrl+C to stop.' -ForegroundColor Green
& $pythonBin (Join-Path $appRoot 'app.py')
