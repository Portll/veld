# Veld installer for Windows
# Usage: irm https://raw.githubusercontent.com/Portll/veld/main/scripts/install.ps1 | iex
#
# Environment variables:
#   VELD_VERSION         - version to install (default: latest)
#   VELD_DIR             - installation directory (default: %LOCALAPPDATA%\Veld\bin)
#   VELD_NO_MODIFY_PATH  - set to 1 to skip PATH modification

$ErrorActionPreference = "Stop"

$Repo = "Portll/veld"
$Version = $env:VELD_VERSION
$InstallDir = if ($env:VELD_DIR) { $env:VELD_DIR } else { Join-Path $env:LOCALAPPDATA "Veld\bin" }

function Write-Info  { param($msg) Write-Host "  info  " -NoNewline -ForegroundColor Cyan;  Write-Host $msg }
function Write-Ok    { param($msg) Write-Host "    ok  " -NoNewline -ForegroundColor Green; Write-Host $msg }
function Write-Warn  { param($msg) Write-Host "  warn  " -NoNewline -ForegroundColor Yellow; Write-Host $msg }
function Write-Err   { param($msg) Write-Host "  error " -NoNewline -ForegroundColor Red;   Write-Host $msg; exit 1 }

# ─── Resolve version ───────────────────────────────────────────────────────
function Resolve-LatestVersion {
    if ($Version) {
        $script:Version = $Version -replace '^v', ''
        Write-Info "Installing version: v$Version"
        return
    }

    Write-Info "Fetching latest release..."
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        $script:Version = $release.tag_name -replace '^v', ''
        Write-Ok "Latest version: v$Version"
    } catch {
        Write-Err "Could not determine latest version. Set VELD_VERSION manually."
    }
}

# ─── Download ───────────────────────────────────────────────────────────────
function Get-Binary {
    param($AssetName, $DestName)

    $url = "https://github.com/$Repo/releases/download/v$Version/$AssetName"
    $dest = Join-Path $tmpDir $DestName

    Write-Info "Downloading $AssetName..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    } catch {
        Write-Err "Download failed: $url`n  $_"
    }

    $size = (Get-Item $dest).Length
    if ($size -lt 1024) {
        $content = Get-Content $dest -Raw -ErrorAction SilentlyContinue
        if ($content -match '<html') {
            Write-Err "Download failed: $AssetName returned HTML (release v$Version may not exist)"
        }
    }

    return $dest
}

# ─── Main ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Veld Installer" -ForegroundColor White -BackgroundColor DarkCyan
Write-Host ""

# Check architecture
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
if ($arch -ne "X64") {
    Write-Err "Unsupported architecture: $arch. Windows ARM64 prebuilt binaries are not published (bun-windows-arm64 unsupported by the MCP server build). Build from source: cargo build --release --target aarch64-pc-windows-msvc"
}
Write-Info "Detected platform: x86_64-windows"

Resolve-LatestVersion

# Create temp dir
$tmpDir = Join-Path $env:TEMP "veld-install-$(Get-Random)"
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

try {
    # Download binaries
    $veldPath  = Get-Binary "veld-x86_64-windows.exe" "veld.exe"
    $mcpPath   = Get-Binary "veld-mcp-windows-x64.exe" "veld-mcp.exe"

    # Install
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item $veldPath  (Join-Path $InstallDir "veld.exe")     -Force
    Copy-Item $mcpPath   (Join-Path $InstallDir "veld-mcp.exe") -Force

    Write-Ok "Installed veld.exe     -> $InstallDir\veld.exe"
    Write-Ok "Installed veld-mcp.exe -> $InstallDir\veld-mcp.exe"
} finally {
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
}

# ─── PATH setup ─────────────────────────────────────────────────────────────
if ($env:VELD_NO_MODIFY_PATH -ne "1") {
    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$InstallDir*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$InstallDir;$userPath", "User")
        $env:PATH = "$InstallDir;$env:PATH"
        Write-Ok "Added $InstallDir to user PATH"
        Write-Warn "Restart your terminal for PATH changes to take effect"
    }
}

# ─── Done ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Veld v$Version installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Get started:" -ForegroundColor White
Write-Host "    veld init       " -NoNewline -ForegroundColor Cyan; Write-Host "# First-time setup (config + AI model download)"
Write-Host "    veld server     " -NoNewline -ForegroundColor Cyan; Write-Host "# Start the memory server (port 3030)"
Write-Host "    veld tui        " -NoNewline -ForegroundColor Cyan; Write-Host "# Launch the dashboard"
Write-Host "    veld doctor     " -NoNewline -ForegroundColor Cyan; Write-Host "# Verify installation"
Write-Host ""
Write-Host "  Docs: https://github.com/$Repo" -ForegroundColor DarkGray
Write-Host ""
