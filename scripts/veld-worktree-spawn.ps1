#!/usr/bin/env pwsh
# veld-worktree-spawn.ps1 — spawn a sibling git worktree per agent x branch.
#
# Convention:
#   <parent-of-main-worktree>/<repo-name>-<branch-slug>
# e.g. main worktree c:/Repositories/Portll/veld + branch w5/journaled-writer
#      -> c:/Repositories/Portll/veld-w5-journaled-writer
#
# Branch resolution:
#   - local branch exists:          git worktree add <path> <branch>
#   - origin/<branch> exists:       git worktree add <path> -b <branch> --track origin/<branch>
#   - neither:                      git worktree add <path> -b <branch> origin/main
#
# Conflicts:
#   - target path already exists -> abort, suggest `git worktree remove`.
#
# Slugging:
#   - lowercased, [^a-z0-9]+ collapsed to '-', trimmed. '/', '_', '.' all become '-'.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Branch,

    [string]$Agent,

    [switch]$Register,

    [switch]$Help
)

$ErrorActionPreference = 'Stop'

function Show-Help {
    @'
Usage: veld-worktree-spawn.ps1 <branch> [--Agent <id>] [--Register] [--Help]

  <branch>     Branch to check out in the new worktree (e.g. w5/journaled-writer).
  --Agent      Agent id to tag the session with. Defaults to a random 8-char hex.
  --Register   POST a session record to Veld's /api/remember (kind=session).
               Honours $env:VELD_API (default http://127.0.0.1:8080).
  --Help       Show this help.

Creates a sibling worktree at <parent>/<repo>-<branch-slug>, copies .claude/,
.vscode/, .mcp.json, and sleight/ (if present), and prints the cd command.
'@ | Write-Host
}

if ($Help -or -not $Branch) { Show-Help; if (-not $Branch) { exit 1 } else { exit 0 } }

function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }
function Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Note($msg) { Write-Host $msg -ForegroundColor DarkGray }

# Verify we're inside a git repo (refuse otherwise).
$topLevel = $null
try { $topLevel = (& git rev-parse --show-toplevel 2>$null).Trim() } catch {}
if (-not $topLevel) { Fail "not inside a git repository" }

# Find the *main* worktree (siblings should be relative to it, not whatever
# linked worktree we happen to be in right now).
$mainWorktree = $null
$porc = & git worktree list --porcelain
foreach ($line in $porc) {
    if ($line -like 'worktree *') { $mainWorktree = $line.Substring(9).Trim(); break }
}
if (-not $mainWorktree) { $mainWorktree = $topLevel }
$mainWorktree = (Resolve-Path -LiteralPath $mainWorktree).Path

$parentDir = Split-Path -Parent $mainWorktree
$repoName  = Split-Path -Leaf   $mainWorktree

# Slug: lowercase, non-alnum -> '-', collapse, trim.
$slug = $Branch.ToLowerInvariant()
$slug = [regex]::Replace($slug, '[^a-z0-9]+', '-')
$slug = $slug.Trim('-')
if (-not $slug) { Fail "branch '$Branch' slugs to empty string" }

$targetPath = Join-Path $parentDir "$repoName-$slug"

if (Test-Path -LiteralPath $targetPath) {
    Fail "target path already exists: $targetPath`n       remove it first with: git worktree remove `"$targetPath`""
}

# Agent id: default to random 8-char hex.
if (-not $Agent) {
    $bytes = New-Object byte[] 4
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $Agent = -join ($bytes | ForEach-Object { $_.ToString('x2') })
}

# Branch resolution.
$hasLocal  = $false
$hasRemote = $false
& git show-ref --verify --quiet "refs/heads/$Branch"           2>$null; if ($LASTEXITCODE -eq 0) { $hasLocal  = $true }
& git show-ref --verify --quiet "refs/remotes/origin/$Branch"  2>$null; if ($LASTEXITCODE -eq 0) { $hasRemote = $true }

Info "main worktree : $mainWorktree"
Info "new worktree  : $targetPath"
Info "branch        : $Branch  (local=$hasLocal, origin=$hasRemote)"
Info "agent id      : $Agent"

if ($hasLocal) {
    & git worktree add -- "$targetPath" "$Branch"
} elseif ($hasRemote) {
    & git worktree add -b "$Branch" --track -- "$targetPath" "origin/$Branch"
} else {
    & git worktree add -b "$Branch" -- "$targetPath" "origin/main"
}
if ($LASTEXITCODE -ne 0) { Fail "git worktree add failed" }

# Idempotent per-file copy: skip if dest exists.
function Copy-IfMissing($srcRoot, $dstRoot, $label) {
    if (-not (Test-Path -LiteralPath $srcRoot)) { return }
    if (-not (Test-Path -LiteralPath $dstRoot)) { New-Item -ItemType Directory -Path $dstRoot -Force | Out-Null }
    $srcItem = Get-Item -LiteralPath $srcRoot
    if ($srcItem.PSIsContainer) {
        Get-ChildItem -LiteralPath $srcRoot -Recurse -Force | ForEach-Object {
            $rel = $_.FullName.Substring($srcItem.FullName.Length).TrimStart('\','/')
            # Skip nested worktree directories - those are other agents' state.
            if ($rel -like 'worktrees\*' -or $rel -like 'worktrees/*') { return }
            $dst = Join-Path $dstRoot $rel
            if ($_.PSIsContainer) {
                if (-not (Test-Path -LiteralPath $dst)) { New-Item -ItemType Directory -Path $dst -Force | Out-Null }
            } else {
                if (Test-Path -LiteralPath $dst) {
                    Note "[skip] $label/$rel exists"
                } else {
                    $dstParent = Split-Path -Parent $dst
                    if (-not (Test-Path -LiteralPath $dstParent)) { New-Item -ItemType Directory -Path $dstParent -Force | Out-Null }
                    Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
                }
            }
        }
    } else {
        if (Test-Path -LiteralPath $dstRoot) { Note "[skip] $label exists" }
        else { Copy-Item -LiteralPath $srcRoot -Destination $dstRoot -Force }
    }
}

Info "copying agent config -> $targetPath"
Copy-IfMissing (Join-Path $mainWorktree '.claude')  (Join-Path $targetPath '.claude')  '.claude'
Copy-IfMissing (Join-Path $mainWorktree '.vscode')  (Join-Path $targetPath '.vscode')  '.vscode'
Copy-IfMissing (Join-Path $mainWorktree 'sleight')  (Join-Path $targetPath 'sleight')  'sleight'
Copy-IfMissing (Join-Path $mainWorktree '.mcp.json') (Join-Path $targetPath '.mcp.json') '.mcp.json'

# Optional Veld session registration.
if ($Register) {
    $api = if ($env:VELD_API) { $env:VELD_API } else { 'http://127.0.0.1:8080' }
    $startedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $payload = [ordered]@{
        kind    = 'session'
        tags    = @('agent-session', "agent:$Agent", "branch:$Branch")
        content = "Spawned worktree $targetPath on branch $Branch for agent $Agent"
        facets  = @{
            agent_session = [ordered]@{
                worktree_path = $targetPath
                branch        = $Branch
                agent_id      = $Agent
                started_at    = $startedAt
                parent_repo   = $mainWorktree
            }
        }
    } | ConvertTo-Json -Depth 6 -Compress
    try {
        Invoke-RestMethod -Method Post -Uri "$api/api/remember" -ContentType 'application/json' -Body $payload -TimeoutSec 5 | Out-Null
        Info "registered session with Veld at $api"
    } catch {
        Write-Host "warn: failed to register session ($_). Worktree is still ready." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Worktree ready." -ForegroundColor Green
Write-Host "Next:"
Write-Host "  cd `"$targetPath`"; claude"
