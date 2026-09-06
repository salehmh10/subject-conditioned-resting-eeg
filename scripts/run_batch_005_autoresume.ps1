param([switch]$Attached)
$ErrorActionPreference = 'Stop'
$scenarioRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $scenarioRepo
. (Join-Path $PSScriptRoot 'setup_scenario2_environment.ps1')
$scenarioPython = Join-Path $scenarioRepo '.venv\Scripts\python.exe'
$scenarioScript = Join-Path $scenarioRepo 'scripts\data\s2_14_autonomous_batch_supervisor.py'
if ($Attached) {
    & $scenarioPython -u $scenarioScript --batch-id 005
    exit $LASTEXITCODE
}
$scenarioProcess = Start-Process -FilePath $scenarioPython -ArgumentList @('-u', ('"' + $scenarioScript + '"'), '--batch-id', '005') -WorkingDirectory $scenarioRepo -WindowStyle Hidden -PassThru
Write-Output ('Final Batch 005 supervisor launched; process ID: ' + $scenarioProcess.Id)
