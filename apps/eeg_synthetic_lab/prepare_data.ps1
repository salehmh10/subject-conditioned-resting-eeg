param([switch]$SkipPreviews)
. (Join-Path $PSScriptRoot 'tools/app_env.ps1')
try {
    if (-not (Test-Path -LiteralPath $LabPython)) { throw 'The application environment is missing. Run setup.ps1 first.' }
    Push-Location $LabRoot
    try {
        & $LabPython (Join-Path $LabRoot 'tools/prepare_presentation_data.py')
        if ($LASTEXITCODE -ne 0) { throw 'Evidence preparation failed. See the source mapping and confirm the canonical artifacts are available.' }
        if (-not $SkipPreviews) {
            & $LabPython (Join-Path $LabRoot 'tools/export_train_previews.py')
            if ($LASTEXITCODE -ne 0) { Write-Warning 'Train previews are unavailable. Numeric result pages remain usable; inspect docs/PREVIEW_PROVENANCE.md.' }
        }
        & $LabPython (Join-Path $LabRoot 'tools/export_presentation_html.py')
        if ($LASTEXITCODE -ne 0) { throw 'HTML snapshot export failed.' }
    } finally { Pop-Location }
    Write-Host 'Presentation data and offline HTML are ready. Run run.ps1.'
} catch { Write-Host "Preparation failed: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }
