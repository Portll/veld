# Veld Windows service installer.
#
# Installs veld.exe as a Windows service so it survives Claude Code sessions,
# user logoffs, and reboots. Three supervisor backends are supported:
#   - nssm  (Non-Sucking Service Manager — small, console-app wrapper, default)
#   - shawl (Rust-written NSSM alternative, modern CLI)
#   - winsw (Jenkins-ecosystem Windows Service Wrapper, XML config, still widely
#            deployed in enterprise environments running Jenkins/legacy Windows
#            ops)
#
# Prerequisites (one of):
#   - nssm.exe  on PATH  (`winget install NSSM.NSSM`           or `choco install nssm`)
#   - shawl.exe on PATH  (`winget install mtkennerly.shawl`    or `cargo install shawl`)
#   - winsw.exe on PATH  (`winget install WinSW.WinSW`         or `choco install winsw`)
# Plus: veld.exe built (cargo build --release in the repo root).
#
# Usage:
#   .\install-service.ps1                          # default: NSSM
#   .\install-service.ps1 -Supervisor shawl        # use shawl instead
#   .\install-service.ps1 -Supervisor winsw        # use WinSW (XML config)
#   .\install-service.ps1 -Uninstall               # remove (auto-detects wrapper)
#   .\install-service.ps1 -BinaryPath C:\path\to\veld.exe  # custom veld binary
#
# After install: `Get-Service Veld` shows the service. Logs go to
# %ProgramData%\Veld\service.log.

param(
    [switch]$Uninstall,
    [switch]$Configure,
    [string]$BinaryPath,
    [string]$ServiceName = "Veld",
    [ValidateSet("nssm", "shawl", "winsw")]
    [string]$Supervisor = "nssm",
    # -Configure picks these up from %APPDATA%\veld\config.toml of the running user
    # by default; pass explicitly to override (e.g. when running elevated and the
    # current-user APPDATA differs from the target user's).
    [string]$ApiKey,
    [string]$StoragePath
)

$ErrorActionPreference = "Stop"

function Resolve-VeldBinary {
    param([string]$Hint)
    if ($Hint) { return $Hint }
    $repoBinary = Join-Path $PSScriptRoot "..\..\target\release\veld.exe"
    $repoBinary = [System.IO.Path]::GetFullPath($repoBinary)
    if (Test-Path $repoBinary) { return $repoBinary }
    $cmd = Get-Command veld.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Resolve-WrapperExe {
    # Find the supervisor wrapper exe (nssm/shawl/winsw) regardless of which
    # user is currently in $env:Path. Needed because elevated shells inherit
    # Administrator's PATH, which doesn't include user-scoped winget Links dirs.
    param([string]$Name)

    $cmd = Get-Command "$Name.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @()
    # All users' WinGet Links dirs
    foreach ($u in Get-ChildItem 'C:\Users' -Directory -ErrorAction SilentlyContinue) {
        $candidates += Join-Path $u.FullName "AppData\Local\Microsoft\WinGet\Links\$Name.exe"
        $candidates += Join-Path $u.FullName "AppData\Local\Microsoft\WindowsApps\$Name.exe"
    }
    # Chocolatey shim
    $candidates += "C:\ProgramData\chocolatey\bin\$Name.exe"
    # Native install locations
    $candidates += "C:\Program Files\$Name\$Name.exe"
    $candidates += "C:\Program Files (x86)\$Name\$Name.exe"
    # NSSM-specific (chocolatey-style nested under nssm-VERSION)
    if ($Name -eq "nssm") {
        $candidates += (Get-ChildItem 'C:\Program Files' -Filter "nssm-*" -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName "win64\nssm.exe" })
    }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Get-RunningUserConfig {
    # When called elevated, %USERPROFILE% is Administrator's. Find the real
    # user's veld config.toml by scanning C:\Users\*. Returns @{key=...; path=...}
    # or $null.
    foreach ($u in Get-ChildItem 'C:\Users' -Directory -ErrorAction SilentlyContinue) {
        $cfg = Join-Path $u.FullName 'AppData\Roaming\veld\config.toml'
        if (Test-Path $cfg) {
            $m = (Select-String -Path $cfg -Pattern '^api_key\s*=\s*"([^"]+)"').Matches
            if ($m) {
                return @{
                    Key      = $m.Groups[1].Value
                    Storage  = (Split-Path $cfg -Parent)
                    UserDir  = $u.FullName
                }
            }
        }
    }
    return $null
}

if ($Configure) {
    Write-Host "Configuring env vars for service '$ServiceName' ($Supervisor)..."
    if (-not $ApiKey -or -not $StoragePath) {
        $found = Get-RunningUserConfig
        if (-not $found) {
            Write-Error "Could not locate any user's %APPDATA%\veld\config.toml. Pass -ApiKey and -StoragePath explicitly."
            exit 1
        }
        if (-not $ApiKey)      { $ApiKey      = $found.Key }
        if (-not $StoragePath) { $StoragePath = $found.Storage }
        Write-Host "  resolved from $($found.UserDir): storage=$StoragePath key=$($ApiKey.Substring(0,8))..."
    }

    if ($Supervisor -eq "nssm") {
        $nssm = Resolve-WrapperExe "nssm"
        if (-not $nssm) {
            Write-Error "nssm.exe not found. Install: winget install NSSM.NSSM"
            exit 1
        }
        Write-Host "  using nssm: $nssm"
        # Stop and WAIT for SCM to reach Stopped before reconfiguring. If we set
        # AppEnvironmentExtra while the wrapper is still in StopPending, the
        # wrapper can get stuck (SCM marks it NOT_STOPPABLE) and the only
        # recovery is taskkill on the wrapper PID. PS cmdlets block on SCM
        # transitions by default; raw `nssm stop` does not.
        try {
            Stop-Service -Name $ServiceName -ErrorAction Stop
        } catch {
            Write-Warning "Stop-Service failed: $($_.Exception.Message). Retrying via nssm..."
            & $nssm stop $ServiceName 2>&1 | Out-Null
        }
        # Defensive: if SCM still reports a transitional state, poll briefly.
        for ($i = 0; $i -lt 20; $i++) {
            if ((Get-Service $ServiceName).Status -eq 'Stopped') { break }
            Start-Sleep -Milliseconds 500
        }
        & $nssm set $ServiceName AppEnvironmentExtra ":VELD_API_KEYS=$ApiKey" ":VELD_MEMORY_PATH=$StoragePath"
        Start-Service -Name $ServiceName
    } elseif ($Supervisor -eq "shawl") {
        # shawl stores env on the sc.exe binPath. Easiest: rewrite via shawl add --replace.
        Write-Warning "shawl reconfigure: stop service, re-run 'shawl add --replace' with --env flags, restart."
        Write-Warning "Reinstall via: .\install-service.ps1 -Uninstall; .\install-service.ps1 -Supervisor shawl -ApiKey ... -StoragePath ..."
        exit 1
    } else {
        $winswCopy = "$env:ProgramData\Veld\$ServiceName.exe"
        $winswXml  = "$env:ProgramData\Veld\$ServiceName.xml"
        if (-not (Test-Path $winswXml)) { Write-Error "WinSW config not found at $winswXml"; exit 1 }
        # Insert <env> entries into the XML (idempotent — replace existing ones)
        [xml]$x = Get-Content $winswXml
        foreach ($e in @($x.service.env)) { $x.service.RemoveChild($e) | Out-Null }
        foreach ($pair in @(@{n="VELD_API_KEYS"; v=$ApiKey}, @{n="VELD_MEMORY_PATH"; v=$StoragePath})) {
            $node = $x.CreateElement("env")
            $node.SetAttribute("name", $pair.n)
            $node.SetAttribute("value", $pair.v)
            $x.service.AppendChild($node) | Out-Null
        }
        $x.Save($winswXml)
        & $winswCopy stop  2>&1 | Out-Null
        & $winswCopy start
    }

    Start-Sleep -Seconds 2
    Get-Service $ServiceName
    Write-Host "`nLogs:         $env:ProgramData\Veld\service.log"
    Write-Host "Health check: curl http://127.0.0.1:3030/health"
    exit 0
}

if ($Uninstall) {
    Write-Host "Stopping and removing service '$ServiceName'..."
    # Stop via Windows SCM (works for all three wrappers)
    Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    # Try whichever supervisor's remove path is available, in order
    if (Get-Command nssm.exe -ErrorAction SilentlyContinue) {
        nssm.exe remove $ServiceName confirm 2>$null
    }
    # WinSW: invoke the renamed copy of winsw.exe to uninstall cleanly
    $winswCopy = "$env:ProgramData\Veld\$ServiceName.exe"
    if (Test-Path $winswCopy) {
        & $winswCopy uninstall 2>$null
        Remove-Item -Force "$env:ProgramData\Veld\$ServiceName.exe", "$env:ProgramData\Veld\$ServiceName.xml" -ErrorAction SilentlyContinue
    }
    # shawl + any leftover: sc.exe delete works either way
    sc.exe delete $ServiceName 2>$null | Out-Null
    Write-Host "Done."
    exit 0
}

$wrapperExe = Resolve-WrapperExe $Supervisor
if (-not $wrapperExe) {
    switch ($Supervisor) {
        "nssm"  { Write-Error "nssm.exe not found. Install: winget install NSSM.NSSM" }
        "shawl" { Write-Error "shawl.exe not found. Install: winget install mtkennerly.shawl  OR  cargo install shawl" }
        "winsw" { Write-Error "winsw.exe not found. Install: winget install WinSW.WinSW  OR  choco install winsw" }
    }
    exit 1
}
Write-Host "  using ${Supervisor}: $wrapperExe"

$BinaryPath = Resolve-VeldBinary $BinaryPath
if (-not $BinaryPath -or -not (Test-Path $BinaryPath)) {
    Write-Error "Could not locate veld.exe. Build the release binary or pass -BinaryPath."
    exit 1
}

$logDir = "$env:ProgramData\Veld"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}
$logPath = Join-Path $logDir "service.log"
$appDir  = Split-Path $BinaryPath -Parent

# Resolve api key + storage path from the running user's veld config (the service
# will run as LocalSystem and won't find them otherwise — would fall back to a
# default dev key and the wrong storage location).
if (-not $ApiKey -or -not $StoragePath) {
    $found = Get-RunningUserConfig
    if ($found) {
        if (-not $ApiKey)      { $ApiKey      = $found.Key }
        if (-not $StoragePath) { $StoragePath = $found.Storage }
        Write-Host "  resolved config from $($found.UserDir)"
    } else {
        Write-Warning "No %APPDATA%\veld\config.toml found. Service will fall back to default dev key."
    }
}

Write-Host "Installing service '$ServiceName' via $Supervisor -> $BinaryPath"

if ($Supervisor -eq "nssm") {
    nssm.exe install $ServiceName $BinaryPath server
    nssm.exe set $ServiceName AppDirectory     $appDir
    nssm.exe set $ServiceName AppStdout        $logPath
    nssm.exe set $ServiceName AppStderr        $logPath
    nssm.exe set $ServiceName AppRotateFiles   1
    nssm.exe set $ServiceName AppRotateOnline  1
    nssm.exe set $ServiceName AppRotateBytes   10485760
    nssm.exe set $ServiceName Start            SERVICE_AUTO_START
    nssm.exe set $ServiceName DisplayName      "Veld - Agentic Memory"
    nssm.exe set $ServiceName Description      "Persistent cognitive memory server for AI agents."
    if ($ApiKey -and $StoragePath) {
        nssm.exe set $ServiceName AppEnvironmentExtra ":VELD_API_KEYS=$ApiKey" ":VELD_MEMORY_PATH=$StoragePath"
    }

    Write-Host "Starting service..."
    nssm.exe start $ServiceName
} elseif ($Supervisor -eq "shawl") {
    # shawl. Syntax: shawl add --name N [options] -- COMMAND ARGS...
    # `--log` writes the wrapped process's stdout+stderr to a file;
    # `--log-rotate 10mb` rotates at 10MB. `sc.exe config` sets autostart afterward.
    $shawlArgs = @(
        'add',
        '--name', $ServiceName,
        '--cwd', $appDir,
        '--log', $logPath,
        '--log-rotate', '10mb',
        '--log-retain', '4'
    )
    if ($ApiKey)      { $shawlArgs += @('--env', "VELD_API_KEYS=$ApiKey") }
    if ($StoragePath) { $shawlArgs += @('--env', "VELD_MEMORY_PATH=$StoragePath") }
    $shawlArgs += @('--', $BinaryPath, 'server')
    & shawl.exe @shawlArgs
    sc.exe config $ServiceName start= auto | Out-Null
    sc.exe description $ServiceName "Persistent cognitive memory server for AI agents." | Out-Null

    Write-Host "Starting service..."
    Start-Service -Name $ServiceName
} else {
    # winsw. Convention: copy WinSW.exe to <ServiceName>.exe next to a
    # <ServiceName>.xml config; the renamed exe IS the service binary.
    $winswSrc  = (Get-Command winsw.exe).Source
    $winswCopy = Join-Path $logDir "$ServiceName.exe"
    $winswXml  = Join-Path $logDir "$ServiceName.xml"

    Copy-Item -Force -Path $winswSrc -Destination $winswCopy

    $envXml = ""
    if ($ApiKey)      { $envXml += "  <env name=`"VELD_API_KEYS`" value=`"$ApiKey`"/>`n" }
    if ($StoragePath) { $envXml += "  <env name=`"VELD_MEMORY_PATH`" value=`"$StoragePath`"/>`n" }
    $xml = @"
<service>
  <id>$ServiceName</id>
  <name>Veld - Agentic Memory</name>
  <description>Persistent cognitive memory server for AI agents.</description>
  <executable>$BinaryPath</executable>
  <arguments>server</arguments>
  <workingdirectory>$appDir</workingdirectory>
  <startmode>Automatic</startmode>
  <onfailure action="restart" delay="10 sec"/>
$envXml  <log mode="roll-by-size">
    <sizeThreshold>10240</sizeThreshold>
    <keepFiles>4</keepFiles>
  </log>
  <logpath>$logDir</logpath>
</service>
"@
    Set-Content -Path $winswXml -Value $xml -Encoding UTF8

    & $winswCopy install
    Write-Host "Starting service..."
    & $winswCopy start
}

Start-Sleep -Seconds 2
Get-Service $ServiceName
Write-Host "`nLogs:         $logPath"
Write-Host "Health check: curl http://127.0.0.1:3030/health"
Write-Host "Uninstall:    .\install-service.ps1 -Uninstall"
