[CmdletBinding()]
param(
    [switch]$ForcePip,
    [switch]$Setup,
    [string]$Project,
    [int]$RunRounds = 0
)

$ErrorActionPreference = "Stop"

$RepoUrl = "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py -3"
    }
    throw "Python 3.10+ is required but was not found in PATH."
}

function Test-PythonVersion([string]$PythonCmd) {
    $code = 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    Invoke-Expression "$PythonCmd -c `"$code`""
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10+ is required."
    }
}

function Invoke-Checked([string]$Command) {
    Write-Host $Command -ForegroundColor DarkGray
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command"
    }
}

Write-Step "Checking Git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found in PATH."
}

Write-Step "Checking Python"
$PythonCmd = Get-PythonCommand
Test-PythonVersion $PythonCmd

$UseUv = $false
if (-not $ForcePip -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    $UseUv = $true
}

if ($UseUv) {
    Write-Step "Installing NightRunner with uv"
    Invoke-Checked "uv tool install --force `"$RepoUrl`""
} else {
    Write-Step "Installing NightRunner with pip"
    Invoke-Checked "$PythonCmd -m pip install --user --upgrade `"$RepoUrl`""
}

$RunnerCmd = $null
Write-Step "Verifying installation"
if (Get-Command nightrunner -ErrorAction SilentlyContinue) {
    Invoke-Checked "nightrunner --help"
    $RunnerCmd = "nightrunner"
} else {
    Write-Warning "nightrunner is not on PATH yet."
    Write-Host "You can still run:" -ForegroundColor Yellow
    Write-Host "$PythonCmd -m nightrunner.cli --help"
    Invoke-Checked "$PythonCmd -m nightrunner.cli --help"
    $RunnerCmd = "$PythonCmd -m nightrunner.cli"
}

if ($Setup) {
    Write-Step "Running NightRunner setup"
    $setupCmd = "$RunnerCmd setup"
    if ($Project) {
        $setupCmd += " --project `"$Project`""
    }
    Invoke-Checked $setupCmd
}

if ($RunRounds -gt 0) {
    Write-Step "Running NightRunner"
    $runCmd = "$RunnerCmd run --rounds $RunRounds"
    if ($Project) {
        $runCmd += " --project `"$Project`""
    }
    Invoke-Checked $runCmd
}

Write-Step "Done"
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "nightrunner auth login"
if ($Project) {
    Write-Host "cd `"$Project`""
} else {
    Write-Host "cd D:\MyDeepLearningProject"
}
Write-Host "nightrunner setup"
Write-Host "nightrunner run --rounds 8"
