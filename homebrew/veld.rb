# Reference Homebrew formula for veld.
#
# Asset names and class name aligned with .github/workflows/release.yml.
# To revive Homebrew distribution:
#   1. Create tap repo: github.com/Portll/homebrew-veld
#   2. Copy this file there as Formula/veld.rb
#   3. Update `version`, refresh the SHA256s with `shasum -a 256 <asset>` per release
#   4. Users install with: brew tap Portll/veld && brew install veld

class Veld < Formula
  desc "Cognitive memory system for AI agents — local, private, neuroscience-inspired"
  homepage "https://github.com/Portll/veld"
  version "0.7.7"
  license "BUSL-1.1"

  # Release assets are raw, unpackaged binaries (matching install.sh / install.ps1).
  # Homebrew handles a single-file `url:` correctly — the file lands in the cellar
  # root and `bin.install` picks it up.
  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-aarch64-macos"
      sha256 "REPLACE_WITH_AARCH64_MACOS_SHA256"
    else
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-x86_64-macos"
      sha256 "REPLACE_WITH_X86_64_MACOS_SHA256"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-aarch64-linux"
      sha256 "REPLACE_WITH_AARCH64_LINUX_SHA256"
    else
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-x86_64-linux"
      sha256 "REPLACE_WITH_X86_64_LINUX_SHA256"
    end
  end

  def install
    # Homebrew's single-file download lands at a path matching the URL basename.
    # Stage it as `veld` so the resulting bin/veld is correctly named.
    if OS.mac?
      asset = Hardware::CPU.arm? ? "veld-aarch64-macos" : "veld-x86_64-macos"
    else
      asset = Hardware::CPU.arm? ? "veld-aarch64-linux" : "veld-x86_64-linux"
    end
    mv asset, "veld"
    bin.install "veld"
  end

  def post_install
    ohai "Run 'veld init' to complete first-time setup"
  end

  def caveats
    <<~EOS
      Veld has been installed. Get started:

        veld init       # First-time setup (creates config, downloads ONNX model)
        veld server     # Start the memory server (default port 3030)
        veld tui        # Launch the dashboard (requires veld-tui binary)

      The MCP server binary (veld-mcp) is not bundled in this formula.
      Install it separately from the release page or via the install.sh script.

      Claude Code integration:
        claude mcp add veld -- npx -y @veld/memory-mcp

      Documentation: https://github.com/Portll/veld
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/veld --version")
  end
end
