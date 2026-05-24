#!/usr/bin/env bash
# Veld installer
# Usage: curl -sSf https://raw.githubusercontent.com/Portll/veld/main/scripts/install.sh | bash
#
# Environment variables:
#   VELD_VERSION   - version to install (default: latest)
#   VELD_DIR       - installation directory (default: ~/.veld/bin)
#   VELD_NO_MODIFY_PATH - set to 1 to skip PATH modification

set -euo pipefail

REPO="Portll/veld"
VERSION="${VELD_VERSION:-}"
INSTALL_DIR="${VELD_DIR:-$HOME/.veld/bin}"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${CYAN}info${NC}  %s\n" "$1"; }
ok()    { printf "${GREEN}  ok${NC}  %s\n" "$1"; }
warn()  { printf "${YELLOW}warn${NC}  %s\n" "$1"; }
err()   { printf "${RED}error${NC} %s\n" "$1" >&2; exit 1; }

# ─── Detect platform ────────────────────────────────────────────────────────
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)  os="linux"  ;;
        Darwin*) os="macos"  ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) err "Unsupported OS: $(uname -s)" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)  arch="x86_64"  ;;
        aarch64|arm64) arch="aarch64" ;;
        *) err "Unsupported architecture: $(uname -m)" ;;
    esac

    # Windows ARM64 release artifacts are not built (bun-windows-arm64 unsupported).
    # The veld binary itself cross-compiles fine — build manually if needed:
    #   cargo build --release --target aarch64-pc-windows-msvc
    if [ "$os" = "windows" ] && [ "$arch" != "x86_64" ]; then
        err "Windows ${arch} prebuilt binaries unavailable. Build from source: cargo build --release --target aarch64-pc-windows-msvc"
    fi

    PLATFORM="${os}"
    ARCH="${arch}"
    info "Detected platform: ${ARCH}-${PLATFORM}"
}

# ─── Resolve version ───────────────────────────────────────────────────────
resolve_version() {
    if [ -n "$VERSION" ]; then
        # Strip leading 'v' if present
        VERSION="${VERSION#v}"
        info "Installing version: v${VERSION}"
        return
    fi

    info "Fetching latest release..."
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"

    if command -v curl &>/dev/null; then
        VERSION=$(curl -sSf "$api_url" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    elif command -v wget &>/dev/null; then
        VERSION=$(wget -qO- "$api_url" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    else
        err "Neither curl nor wget found. Install one and retry."
    fi

    if [ -z "$VERSION" ]; then
        err "Could not determine latest version. Set VELD_VERSION manually."
    fi

    ok "Latest version: v${VERSION}"
}

# ─── Build asset names ──────────────────────────────────────────────────────
build_asset_names() {
    case "${PLATFORM}" in
        windows)
            VELD_ASSET="veld-x86_64-windows.exe"
            MCP_ASSET="veld-mcp-windows-x64.exe"
            VELD_BIN="veld.exe"
            MCP_BIN="veld-mcp.exe"
            ;;
        macos)
            VELD_ASSET="veld-${ARCH}-macos"
            local mcp_arch="${ARCH/x86_64/x64}"
            mcp_arch="${mcp_arch/aarch64/arm64}"
            MCP_ASSET="veld-mcp-darwin-${mcp_arch}"
            VELD_BIN="veld"
            MCP_BIN="veld-mcp"
            ;;
        linux)
            VELD_ASSET="veld-${ARCH}-linux"
            local mcp_arch="${ARCH/x86_64/x64}"
            mcp_arch="${mcp_arch/aarch64/arm64}"
            MCP_ASSET="veld-mcp-linux-${mcp_arch}"
            VELD_BIN="veld"
            MCP_BIN="veld-mcp"
            ;;
    esac

    BASE_URL="https://github.com/${REPO}/releases/download/v${VERSION}"
}

# ─── Download ───────────────────────────────────────────────────────────────
download() {
    local url="$1" dest="$2"

    info "Downloading $(basename "$url")..."
    if command -v curl &>/dev/null; then
        curl -fSL --progress-bar "$url" -o "$dest"
    elif command -v wget &>/dev/null; then
        wget --show-progress -q "$url" -O "$dest"
    fi
}

# ─── Install ────────────────────────────────────────────────────────────────
install_binaries() {
    mkdir -p "$INSTALL_DIR"

    local tmpdir
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    download "${BASE_URL}/${VELD_ASSET}" "${tmpdir}/${VELD_BIN}"
    download "${BASE_URL}/${MCP_ASSET}" "${tmpdir}/${MCP_BIN}"

    # Verify downloads are not HTML error pages
    for bin in "${tmpdir}/${VELD_BIN}" "${tmpdir}/${MCP_BIN}"; do
        if [ ! -f "$bin" ] || [ ! -s "$bin" ]; then
            err "Download failed: $(basename "$bin") is empty"
        fi
        if head -c 6 "$bin" | grep -qi '<html'; then
            err "Download failed: $(basename "$bin") is an HTML error page (release v${VERSION} may not exist)"
        fi
    done

    cp "${tmpdir}/${VELD_BIN}" "${INSTALL_DIR}/${VELD_BIN}"
    cp "${tmpdir}/${MCP_BIN}" "${INSTALL_DIR}/${MCP_BIN}"
    chmod +x "${INSTALL_DIR}/${VELD_BIN}" "${INSTALL_DIR}/${MCP_BIN}"

    ok "Installed veld     → ${INSTALL_DIR}/${VELD_BIN}"
    ok "Installed veld-mcp → ${INSTALL_DIR}/${MCP_BIN}"
}

# ─── PATH setup ─────────────────────────────────────────────────────────────
setup_path() {
    if [ "${VELD_NO_MODIFY_PATH:-0}" = "1" ]; then
        return
    fi

    # Check if already in PATH
    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) return ;;
    esac

    local shell_name profile_file=""
    shell_name="$(basename "${SHELL:-/bin/sh}")"

    case "$shell_name" in
        bash)
            if [ -f "$HOME/.bash_profile" ]; then
                profile_file="$HOME/.bash_profile"
            else
                profile_file="$HOME/.bashrc"
            fi
            ;;
        zsh)  profile_file="$HOME/.zshrc" ;;
        fish) profile_file="$HOME/.config/fish/config.fish" ;;
        *)    profile_file="$HOME/.profile" ;;
    esac

    if [ -n "$profile_file" ]; then
        local line
        if [ "$shell_name" = "fish" ]; then
            line="set -gx PATH \"${INSTALL_DIR}\" \$PATH"
        else
            line="export PATH=\"${INSTALL_DIR}:\$PATH\""
        fi

        # Don't duplicate
        if [ -f "$profile_file" ] && grep -qF "$INSTALL_DIR" "$profile_file" 2>/dev/null; then
            return
        fi

        printf '\n# Veld\n%s\n' "$line" >> "$profile_file"
        ok "Added ${INSTALL_DIR} to PATH in ${profile_file}"
        warn "Restart your shell or run: source ${profile_file}"
    fi
}

# ─── Post-install hint ──────────────────────────────────────────────────────
post_install() {
    printf "\n${BOLD}${GREEN}Veld v${VERSION} installed successfully!${NC}\n\n"
    printf "  Get started:\n"
    printf "    ${CYAN}veld init${NC}       # First-time setup (config + AI model download)\n"
    printf "    ${CYAN}veld server${NC}     # Start the memory server (port 3030)\n"
    printf "    ${CYAN}veld tui${NC}        # Launch the dashboard\n"
    printf "    ${CYAN}veld doctor${NC}     # Verify installation\n\n"
    printf "  Docs: https://github.com/${REPO}\n\n"
}

# ─── Main ───────────────────────────────────────────────────────────────────
main() {
    printf "\n${BOLD}Veld Installer${NC}\n\n"

    detect_platform
    resolve_version
    build_asset_names
    install_binaries
    setup_path
    post_install
}

main
