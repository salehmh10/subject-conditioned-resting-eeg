$ErrorActionPreference = 'Stop'
$LabRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
foreach ($folder in @('.cache', '.tmp', 'exports', 'docs')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $LabRoot $folder) | Out-Null
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $LabRoot '.cache/pycache'
$env:PIP_CACHE_DIR = Join-Path $LabRoot '.cache/pip'
$env:TEMP = Join-Path $LabRoot '.tmp'
$env:TMP = $env:TEMP
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$env:STREAMLIT_SERVER_ADDRESS = '127.0.0.1'
$env:STREAMLIT_SERVER_HEADLESS = 'true'
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE = 'none'
$env:MPLCONFIGDIR = Join-Path $LabRoot '.cache/matplotlib'
$LabPython = Join-Path $LabRoot '.venv/Scripts/python.exe'
