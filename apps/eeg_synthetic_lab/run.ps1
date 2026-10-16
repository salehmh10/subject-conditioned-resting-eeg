param([int]$Port = 8501)
. (Join-Path $PSScriptRoot 'tools/app_env.ps1')
if (-not (Test-Path -LiteralPath $LabPython)) {
    Write-Host 'Application environment missing. Run setup.ps1 first.' -ForegroundColor Red
    exit 1
}
& $LabPython (Join-Path $LabRoot 'tools/launch.py') --port $Port
exit $LASTEXITCODE
