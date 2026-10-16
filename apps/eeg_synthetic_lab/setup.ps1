param([string]$PythonExecutable = 'python', [switch]$Dev)
. (Join-Path $PSScriptRoot 'tools/app_env.ps1')
try {
    if (-not (Test-Path -LiteralPath $LabPython)) {
        $pythonCommand = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw 'Python was not found. Install Python 3.12, or pass -PythonExecutable with its full path.' }
        & $PythonExecutable -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Use Python 3.10-3.13; Python 3.12 was tested.'"
        if ($LASTEXITCODE -ne 0) { throw 'A compatible Python executable is required.' }
        & $PythonExecutable -m venv (Join-Path $LabRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the dedicated environment.' }
    }
    $requirements = if ($Dev) { 'requirements-dev.txt' } else { 'requirements-previews.txt' }
    Write-Host 'Installing pinned application packages. This initial setup may require internet access.'
    & $LabPython -m pip install --disable-pip-version-check --retries 1 -r (Join-Path $LabRoot $requirements)
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Check network/proxy access and rerun setup.ps1. The research environment was not changed.' }
    & $LabPython -m pip freeze | Set-Content -LiteralPath (Join-Path $LabRoot 'docs/installed-versions.txt') -Encoding utf8
    Write-Host 'Setup complete. Run prepare_data.ps1 once, then run.ps1.'
} catch { Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }
