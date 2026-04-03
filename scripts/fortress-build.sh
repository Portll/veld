#!/usr/bin/env bash
#
# fortress-build.sh — Fractal binary obfuscation pipeline for shodh-memory
#
# Produces a distribution binary that is maximally resistant to reverse engineering.
#
# Layers:
#   L0: Compile-time string encryption (obf! macro, per-build key)
#   L1: Compiler hardening (LTO=fat, codegen-units=1, strip, panic=abort)
#   L2: Runtime protection (anti-debug, custom panic handler, integrity checks)
#   L3: Post-build processing (this script — strip metadata, pack, verify)
#
# Usage:
#   ./scripts/fortress-build.sh [--target <triple>] [--upx] [--verify-only <binary>]
#
# Output:
#   target/fortress/veld  (obfuscated server binary)
#   target/fortress/shodh                (obfuscated CLI binary)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# shellcheck source=./setup-libclang-env.sh
source "$SCRIPT_DIR/setup-libclang-env.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TARGET=""
USE_UPX=false
VERIFY_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift 2 ;;
        --upx) USE_UPX=true; shift ;;
        --verify-only) VERIFY_ONLY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Verify-only mode ───────────────────────────────────────────────────────
if [[ -n "$VERIFY_ONLY" ]]; then
    echo -e "${CYAN}═══ FORTRESS VERIFICATION: $VERIFY_ONLY ═══${NC}"
    verify_binary "$VERIFY_ONLY"
    exit $?
fi

# ─── Phase 1: Build with fortress profile ────────────────────────────────────
echo -e "${CYAN}═══ FORTRESS BUILD: Phase 1 — Compile ═══${NC}"
echo "  Profile:       fortress (LTO=fat, codegen-units=1, strip=symbols)"
echo "  Features:      fortress (anti-debug, panic handler, string encryption)"
echo "  Overflow:      disabled (removes check branches)"
echo "  Debug info:    zero"
echo ""

BUILD_ARGS=(
    build
    --profile fortress
    --features fortress
)

if [[ -n "$TARGET" ]]; then
    BUILD_ARGS+=(--target "$TARGET")
    BINARY_DIR="target/$TARGET/fortress"
else
    BINARY_DIR="target/fortress"
fi

"$SCRIPT_DIR/cargo-dev.sh" "${BUILD_ARGS[@]}"

echo -e "${GREEN}  ✓ Compilation complete${NC}"

# ─── Phase 2: Identify binaries ──────────────────────────────────────────────
echo -e "${CYAN}═══ FORTRESS BUILD: Phase 2 — Post-process ═══${NC}"

BINARIES=()
for name in veld shodh; do
    bin="$BINARY_DIR/$name"
    if [[ -f "$bin" ]]; then
        BINARIES+=("$bin")
    fi
done

if [[ ${#BINARIES[@]} -eq 0 ]]; then
    echo -e "${RED}  ✗ No binaries found in $BINARY_DIR${NC}"
    exit 1
fi

for bin in "${BINARIES[@]}"; do
    echo ""
    echo -e "${YELLOW}  Processing: $(basename "$bin")${NC}"
    ORIG_SIZE=$(stat -f%z "$bin" 2>/dev/null || stat -c%s "$bin" 2>/dev/null)

    # 2a. Additional strip passes
    # strip -x removes local symbols that `strip = "symbols"` might miss
    if command -v strip &>/dev/null; then
        strip -x "$bin" 2>/dev/null || true
        echo "    Strip -x: done"
    fi

    # 2b. Remove Mach-O code signature (we'll re-sign after)
    # on macOS, codesign adds metadata that reveals toolchain info
    if [[ "$(uname)" == "Darwin" ]]; then
        codesign --remove-signature "$bin" 2>/dev/null || true
        echo "    Remove code signature: done"
    fi

    # 2c. UPX packing (optional — compresses + adds unpacking stub)
    if $USE_UPX && command -v upx &>/dev/null; then
        upx --best --ultra-brute "$bin" 2>/dev/null || {
            echo -e "    ${YELLOW}UPX packing skipped (binary format not supported)${NC}"
        }
        echo "    UPX pack: done"
    fi

    # 2d. Re-sign on macOS (required for execution after stripping)
    if [[ "$(uname)" == "Darwin" ]]; then
        codesign -s - "$bin" 2>/dev/null || true
        echo "    Ad-hoc re-sign: done"
    fi

    FINAL_SIZE=$(stat -f%z "$bin" 2>/dev/null || stat -c%s "$bin" 2>/dev/null)
    REDUCTION=$(( (ORIG_SIZE - FINAL_SIZE) * 100 / ORIG_SIZE ))
    echo -e "    Size: ${ORIG_SIZE} → ${FINAL_SIZE} bytes (${REDUCTION}% reduction)"
done

# ─── Phase 3: Verification ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══ FORTRESS BUILD: Phase 3 — Verify ═══${NC}"

LEAKED=0
for bin in "${BINARIES[@]}"; do
    name=$(basename "$bin")
    echo ""
    echo -e "  ${YELLOW}Verifying: $name${NC}"

    # 3a. Check for leaked API route strings
    ROUTES=$(strings "$bin" 2>/dev/null | grep -c '/api/' || true)
    if [[ $ROUTES -gt 0 ]]; then
        echo -e "    ${RED}✗ LEAK: $ROUTES API route strings found${NC}"
        strings "$bin" | grep '/api/' | head -5 | while read -r line; do
            echo -e "      ${RED}→ $line${NC}"
        done
        LEAKED=$((LEAKED + ROUTES))
    else
        echo -e "    ${GREEN}✓ No API route strings leaked${NC}"
    fi

    # 3b. Check for leaked environment variable names
    ENVVARS=$(strings "$bin" 2>/dev/null | grep -c 'SHODH_' || true)
    if [[ $ENVVARS -gt 0 ]]; then
        echo -e "    ${RED}✗ LEAK: $ENVVARS SHODH_ env var strings found${NC}"
        strings "$bin" | grep 'SHODH_' | head -5 | while read -r line; do
            echo -e "      ${RED}→ $line${NC}"
        done
        LEAKED=$((LEAKED + ENVVARS))
    else
        echo -e "    ${GREEN}✓ No SHODH_ env var strings leaked${NC}"
    fi

    # 3c. Check for leaked source file paths
    SRCPATHS=$(strings "$bin" 2>/dev/null | grep -cE 'src/|\.rs:' || true)
    if [[ $SRCPATHS -gt 0 ]]; then
        echo -e "    ${RED}✗ LEAK: $SRCPATHS source path strings found${NC}"
        strings "$bin" | grep -E 'src/|\.rs:' | head -5 | while read -r line; do
            echo -e "      ${RED}→ $line${NC}"
        done
        LEAKED=$((LEAKED + SRCPATHS))
    else
        echo -e "    ${GREEN}✓ No source path strings leaked${NC}"
    fi

    # 3d. Check for leaked error messages with internal details
    ERRS=$(strings "$bin" 2>/dev/null | grep -ciE 'failed to|error:|panic|unwrap|internal' || true)
    if [[ $ERRS -gt 10 ]]; then
        echo -e "    ${YELLOW}⚠ WARNING: $ERRS potential error message strings${NC}"
        LEAKED=$((LEAKED + ERRS))
    else
        echo -e "    ${GREEN}✓ Error messages minimized ($ERRS strings)${NC}"
    fi

    # 3e. Symbol count
    SYMS=$(nm "$bin" 2>/dev/null | wc -l || echo 0)
    echo -e "    Symbols remaining: ${SYMS}"

    # 3f. Total readable strings
    TOTAL_STRINGS=$(strings "$bin" 2>/dev/null | wc -l || echo 0)
    echo -e "    Total readable strings: ${TOTAL_STRINGS}"

    # 3g. Binary size
    SIZE=$(stat -f%z "$bin" 2>/dev/null || stat -c%s "$bin" 2>/dev/null)
    SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc)
    echo -e "    Binary size: ${SIZE_MB} MB"
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
if [[ $LEAKED -gt 0 ]]; then
    echo -e "${YELLOW}  Fortress build complete with $LEAKED leaked strings.${NC}"
    echo -e "${YELLOW}  These are expected for non-obf!() wrapped strings.${NC}"
    echo -e "${YELLOW}  Wrap critical strings with obf!() macro to encrypt.${NC}"
else
    echo -e "${GREEN}  Fortress build complete — zero string leaks detected.${NC}"
fi
echo -e "${CYAN}  Binaries: ${BINARIES[*]}${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
