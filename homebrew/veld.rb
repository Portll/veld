# Reference Homebrew formula for veld.
#
# NOTE: This file is archival/reference-only for the last public Homebrew release
# line. It is not kept in sync with the current `v0.7.6-unstable` repository
# branch and should not be treated as the source of truth for this branch.
#
# To use:
#   1. Create a tap repo: github.com/Portll/homebrew-veld
#   2. Copy this file there as Formula/veld.rb
#   3. Update the version, URLs, and SHA256 hashes for each release
#
# Users install with:
#   brew tap Portll/veld
#   brew install veld

class VeldMemory < Formula
  desc "Cognitive memory system for AI agents — local, private, neuroscience-inspired"
  homepage "https://github.com/Portll/veld"
  version "0.1.91"
  license "BUSL-1.1"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-macos-arm64.tar.gz"
      sha256 "15baa1cb6546fbd50e7e31d3865caf6b8a7d8188813179fbafe6707d839cd419"
    else
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-macos-x64.tar.gz"
      sha256 "6e4068f77f7abb5dc2cc3dd7ce56a276bf0a359b51e2ed04b923f6acdad6fad1"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-linux-arm64.tar.gz"
      sha256 "d3a3dc2aedd853cebbbf82106e1d0039f071cfcfedfc84f6419577dc3d578ee3"
    else
      url "https://github.com/Portll/veld/releases/download/v#{version}/veld-linux-x64.tar.gz"
      sha256 "c07692e8f53d5b1515ae25e0035bc2db55ebf039e7051d3e877720b98b9718d9"
    end
  end

  def install
    bin.install "veld"
    bin.install "veld"
    bin.install "veld-tui"
    lib.install Dir["*.dylib"] if OS.mac?
    lib.install Dir["*.so*"] if OS.linux?
  end

  def post_install
    ohai "Run 'veld init' to complete first-time setup"
  end

  def caveats
    <<~EOS
      Veld has been installed. Get started:

        veld init       # First-time setup (creates config, downloads AI model)
        veld server     # Start the memory server
        veld tui        # Launch the dashboard
        veld status     # Check server health

      Claude Code integration:
        claude mcp add veld -- npx -y @veld/memory-mcp

      Documentation: https://veld.com/docs
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/veld version")
  end
end
