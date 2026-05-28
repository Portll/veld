# Fetch external knowledge sources for alignment corpus expansion.
#
# Each source is a sparse, shallow git clone of just the doc subtree —
# typically <20 MB per repo. Re-running the script updates existing clones
# rather than re-fetching from scratch.
#
# Run from `evaluations/alignment/pairs/sources/external/` or from the repo
# root. Writes to `evaluations/alignment/pairs/_external/` (gitignored).

$ErrorActionPreference = "Stop"

# Resolve repo root regardless of where the script is invoked from.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path "$scriptDir/../../../../.."
$externalDir = Join-Path $repoRoot "evaluations/alignment/pairs/_external"

if (-not (Test-Path $externalDir)) {
    New-Item -ItemType Directory -Path $externalDir -Force | Out-Null
}

# (folder, repo URL, subtree path to sparse-checkout)
$sources = @(
    @{ Folder = "mdn-content";       Repo = "https://github.com/mdn/content.git";                Sub = "files/en-us/web/javascript" },
    @{ Folder = "cpython";           Repo = "https://github.com/python/cpython.git";             Sub = "Doc" },
    @{ Folder = "rust-by-example";   Repo = "https://github.com/rust-lang/rust-by-example.git";  Sub = "src" },
    @{ Folder = "react.dev";         Repo = "https://github.com/reactjs/react.dev.git";          Sub = "src/content" },
    @{ Folder = "TypeScript-Website"; Repo = "https://github.com/microsoft/TypeScript-Website.git"; Sub = "packages/documentation" }
)

foreach ($src in $sources) {
    $dest = Join-Path $externalDir $src.Folder
    if (Test-Path $dest) {
        Write-Host "[fetch] $($src.Folder) exists — pulling updates"
        Push-Location $dest
        try {
            git pull --depth=1 origin HEAD 2>&1 | Out-Host
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "[fetch] cloning $($src.Repo) (sparse: $($src.Sub))"
        git clone --depth=1 --filter=blob:none --sparse $src.Repo $dest 2>&1 | Out-Host
        Push-Location $dest
        try {
            git sparse-checkout set $src.Sub 2>&1 | Out-Host
        } finally {
            Pop-Location
        }
    }
}

Write-Host ""
Write-Host "[fetch] done. Sources available under $externalDir"
Write-Host "[fetch] next: run alignment-collect with the TOML specs in this directory."
