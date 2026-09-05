param([switch]$ForTests)
$ErrorActionPreference = 'Stop'
$scenarioRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$scenarioPython = Join-Path $scenarioRepo '.venv\Scripts\python.exe'
# -B makes the initial path lookup read-only before cache variables are set.
$scenarioLookup = "import sys; sys.path[:0]=[r'$scenarioRepo',r'$scenarioRepo\src']; from resting_eeg.data.environment import main; main()"
$scenarioJson = & $scenarioPython -B -c $scenarioLookup --repo-root $scenarioRepo --values-only
if ($LASTEXITCODE -ne 0) { throw 'Scenario 2 path lookup failed' }
$scenarioValues = $scenarioJson | ConvertFrom-Json
foreach ($property in $scenarioValues.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process')
}
$scenarioArguments = @('-c', $scenarioLookup, '--repo-root', $scenarioRepo)
if ($ForTests) { $scenarioArguments += '--for-tests' }
$scenarioEvidence = & $scenarioPython @scenarioArguments
if ($LASTEXITCODE -ne 0) { throw "Scenario 2 environment/system-drive guard failed: $scenarioEvidence" }
$scenarioEvidence
# Dot-source this script in EVERY shell used for Python, pytest, compileall,
# pip, verification, or streaming. No user/machine-wide variables are modified.
