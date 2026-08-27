<#
.SYNOPSIS
    Let Docker containers reach the Tailscale network and the LAN.

.DESCRIPTION
    Docker Desktop on Windows runs containers inside a WSL2 virtual machine with its
    own network stack. By default that VM reaches the public internet through NAT but
    has no route to the Windows host's own interfaces -- which is where Tailscale and
    the LAN live. Containers can therefore reach the internet and nothing local.

    It usually works anyway, because Tailscale installs a route into the WSL network
    when it starts and the VM inherits it. That breaks whenever the VM is recreated
    while Tailscale is down -- after a Docker Desktop crash, for instance -- and
    recycling the VM afterwards does not always restore it.

    Mirrored networking removes the guesswork: WSL shares the Windows network stack
    directly, so containers see every interface Windows sees, Tailscale included.

    Requires Windows 11 22H2 or later.

.NOTES
    This affects ALL WSL distributions, not just Docker's.

    To undo: delete %USERPROFILE%\.wslconfig, or restore the .bak file this script
    writes if you already had one, then run `wsl --shutdown` and restart Docker.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\enable-wsl-mirrored-networking.ps1
#>

$ErrorActionPreference = 'Stop'

# Run from the repo root regardless of where this was invoked from, so the docker
# checks at the end see the project's compose file and images.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$configPath = Join-Path $env:USERPROFILE '.wslconfig'

Write-Host ''
Write-Host 'WSL mirrored networking' -ForegroundColor Cyan
Write-Host '-----------------------'
Write-Host "  working from $repoRoot"

# --- 1. Preserve anything already there -------------------------------------
if (Test-Path $configPath) {
    $existing = Get-Content $configPath -Raw
    if ($existing -match 'networkingMode\s*=\s*mirrored') {
        Write-Host 'Already set to mirrored. Nothing to change.' -ForegroundColor Green
        Write-Host "  $configPath"
        Write-Host ''
        Write-Host 'If containers still cannot reach the tailnet, run `wsl --shutdown`'
        Write-Host 'and restart Docker Desktop, then re-run the check at the end of this script.'
        return
    }
    $backup = "$configPath.bak"
    Copy-Item $configPath $backup -Force
    Write-Host "Backed up your existing config to $backup" -ForegroundColor Yellow
    Write-Host 'Existing contents:'
    $existing -split "`n" | ForEach-Object { Write-Host "  $_" }
    Write-Host ''
    Write-Host 'This script only appends a [wsl2] networkingMode setting. Review the'
    Write-Host 'result afterwards if you had other settings in there.'
} else {
    Write-Host 'No .wslconfig exists; creating one.'
}

# --- 2. Write the setting ----------------------------------------------------
# dnsTunneling and autoProxy are the usual companions to mirrored mode: without
# them DNS and proxy resolution inside WSL can lag behind the host's.
$block = @'
[wsl2]
# Share the Windows network stack so containers see every interface Windows sees,
# including Tailscale. Without this, Docker's VM reaches the internet but not the
# tailnet or the LAN.
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
'@

# Written without a byte-order mark, deliberately. PowerShell 5.1's `-Encoding utf8`
# emits a BOM, and WSL's config parser does not tolerate one -- the file looks correct,
# `networkingMode=mirrored` is right there when you cat it, and WSL silently ignores the
# whole thing. Costly to diagnose, so it is worth the explicit encoder.
$content = if (Test-Path $configPath) {
    (Get-Content $configPath -Raw).TrimEnd() + "`r`n`r`n" + $block + "`r`n"
} else {
    $block + "`r`n"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($configPath, $content, $utf8NoBom)
Write-Host "Wrote $configPath" -ForegroundColor Green

# --- 3. Recreate the VM ------------------------------------------------------
Write-Host ''
Write-Host 'Shutting down WSL so the VM picks up the new setting...'
wsl --shutdown
Start-Sleep -Seconds 5

Write-Host 'Restarting Docker Desktop...'
$docker = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
if (Test-Path $docker) {
    Start-Process $docker
} else {
    Write-Host "Could not find Docker Desktop at $docker -- start it yourself." -ForegroundColor Yellow
}

# --- 4. Wait for the daemon --------------------------------------------------
Write-Host -NoNewline 'Waiting for the Docker daemon'
$up = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 10
    docker version --format '{{.Server.Version}}' *> $null
    if ($?) { $up = $true; break }
    Write-Host -NoNewline '.'
}
Write-Host ''
if (-not $up) {
    Write-Host 'Daemon did not come up within 5 minutes. Start Docker Desktop manually,' -ForegroundColor Yellow
    Write-Host 'then run the check below yourself.'
    Write-Host '  docker run --rm --entrypoint python di-app -c "import socket; socket.create_connection((''100.67.117.41'',443),8); print(''OK'')"'
    return
}
Write-Host 'Daemon is up.' -ForegroundColor Green

# --- 5. Prove it actually fixed the thing it was meant to fix ----------------
# The point was never the config file; it was container-to-tailnet reachability.
Write-Host ''
Write-Host 'Checking whether a container can now reach the tailnet...'
# Uses the project's own image so this does not pull anything, and so the probe runs
# in exactly the container the extractor will run in.
$py = "import socket" +
      "`ns = socket.socket(); s.settimeout(8)" +
      "`ntry:" +
      "`n    s.connect(('100.67.117.41', 443)); print('OK')" +
      "`nexcept Exception as e:" +
      "`n    print('FAILED', type(e).__name__)" +
      "`nfinally:" +
      "`n    s.close()"
$probe = docker run --rm --entrypoint python di-app -c $py

if ($probe -match 'OK') {
    Write-Host 'Container reached the tailnet. Docker can talk to the model server.' -ForegroundColor Green
} else {
    Write-Host "Container still cannot reach the tailnet: $probe" -ForegroundColor Red
    Write-Host ''
    Write-Host 'Things worth checking:'
    Write-Host '  - Tailscale is running and healthy:  tailscale status'
    Write-Host '  - Windows 11 22H2 or later (mirrored mode needs it):  winver'
    Write-Host '  - Docker Desktop > Settings > Resources > Network, and make sure'
    Write-Host '    nothing there overrides the WSL setting.'
}
Write-Host ''
