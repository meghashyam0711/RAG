# GraphRAG one-line installer for Windows (PowerShell 5.1+ and 7+).
#   irm https://raw.githubusercontent.com/meghashyam0711/RAG/main/install.ps1 | iex
# Options (set before running):
#   $env:GRAPHRAG_HOME    = install folder            (default: $HOME\RAG)
#   $env:GRAPHRAG_CLIENTS = all | none | codex,cursor (default: all detected agents/IDEs)
#   $env:GRAPHRAG_REPO    = git URL or local path     (default: https://github.com/meghashyam0711/RAG)

$ErrorActionPreference = "Stop"
$Repo = if ($env:GRAPHRAG_REPO) { $env:GRAPHRAG_REPO } else { "https://github.com/meghashyam0711/RAG" }
$Dir = if ($env:GRAPHRAG_HOME) { $env:GRAPHRAG_HOME } else { Join-Path $HOME "RAG" }
$Clients = if ($env:GRAPHRAG_CLIENTS) { $env:GRAPHRAG_CLIENTS } else { "all" }

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Has($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function Refresh-Path {
    # Pick up tools installed during this session without reopening the terminal.
    $parts = @([Environment]::GetEnvironmentVariable("Path", "Machine"),
               [Environment]::GetEnvironmentVariable("Path", "User"),
               (Join-Path $HOME ".local\bin"), (Join-Path $HOME ".cargo\bin"), $env:Path)
    $env:Path = ($parts | Where-Object { $_ }) -join ";"
}
function Run($exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "'$exe $($argList -join ' ')' failed (exit $LASTEXITCODE)" }
}

Refresh-Path

# 1. uv (it also provides Python 3.11+ automatically)
if (-not (Has "uv")) {
    Step "Installing uv"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Refresh-Path
    if (-not (Has "uv")) { throw "uv was installed but is not on PATH. Open a new terminal and run this again." }
}

# 2. Get the code: git clone, or ZIP download when git is missing
if (Test-Path (Join-Path $Dir "pyproject.toml")) {
    Step "Found existing install in $Dir"
    if ((Has "git") -and (Test-Path (Join-Path $Dir ".git"))) { git -C $Dir pull --ff-only --quiet }
} else {
    if ((Test-Path $Dir) -and (Get-ChildItem $Dir -Force | Select-Object -First 1)) {
        throw "$Dir exists but is not this project. Delete it or set `$env:GRAPHRAG_HOME to another folder."
    }
    if (Has "git") {
        Step "Cloning $Repo into $Dir"
        Run "git" @("clone", "--depth", "1", "--quiet", $Repo, $Dir)
    } else {
        Step "git not found; downloading ZIP"
        $zip = Join-Path ([IO.Path]::GetTempPath()) "graphrag-src.zip"
        $tmp = Join-Path ([IO.Path]::GetTempPath()) "graphrag-src"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest "$($Repo.TrimEnd('/') -replace '\.git$','')/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive $zip $tmp -Force
        $src = (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName
        if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force }
        Move-Item $src $Dir
        Remove-Item $zip, $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 3. Install dependencies and self-test
Set-Location $Dir
Step "Installing dependencies (uv sync)"
Run "uv" @("sync", "--quiet")
Step "Self-test"
Run "uv" @("run", "graphrag", "doctor")

# 4. Register with coding agents / IDEs
if ($Clients -ne "none") {
    foreach ($c in ($Clients -split ",")) {
        Step "Registering MCP server: $c"
        & uv run graphrag install $c.Trim()
    }
}

Write-Host ""
Write-Host "Done. Installed in $Dir" -ForegroundColor Green
Write-Host "Restart your agent/IDE to load 'graphrag-reader'. Terminal use:"
Write-Host "  cd `"$Dir`""
Write-Host "  uv run graphrag ingest C:\path\to\file.pdf --domain technical"
Write-Host "  uv run graphrag ask `"your question`""
