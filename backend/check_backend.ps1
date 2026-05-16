$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$tempRoot = Join-Path $scriptDir "runtime\pytest-runs"
$runId = "{0}-{1}" -f $PID, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$baseTemp = Join-Path $tempRoot $runId

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing project virtualenv Python: $venvPython"
}

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:TEMP = $baseTemp
$env:TMP = $baseTemp
New-Item -ItemType Directory -Path $baseTemp -Force | Out-Null

Push-Location $scriptDir
try {
    & $venvPython -m pytest "--basetemp=$baseTemp" @args
    if ($LASTEXITCODE -ne 0) {
        throw "Backend pytest failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
