#!/usr/bin/env node
/**
 * Postinstall script for @veld/memory-mcp
 *
 * Downloads the appropriate veld binary for the current platform
 * from GitHub releases.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execFileSync } = require('child_process');

const VERSION = require('../package.json').version;
const REPO = 'Portll/veld';
const BIN_DIR = path.join(__dirname, '..', 'bin');

// Platform detection
function getPlatformInfo() {
  const platform = process.platform;
  const arch = process.arch;

  if (platform === 'linux' && arch === 'x64') {
    return { name: 'veld-linux-x64', ext: '.tar.gz', binary: 'veld' };
  } else if (platform === 'linux' && arch === 'arm64') {
    return { name: 'veld-linux-arm64', ext: '.tar.gz', binary: 'veld' };
  } else if (platform === 'darwin' && arch === 'x64') {
    return { name: 'veld-macos-x64', ext: '.tar.gz', binary: 'veld' };
  } else if (platform === 'darwin' && arch === 'arm64') {
    return { name: 'veld-macos-arm64', ext: '.tar.gz', binary: 'veld' };
  } else if (platform === 'win32' && arch === 'x64') {
    return { name: 'veld-windows-x64', ext: '.zip', binary: 'veld.exe' };
  } else {
    return null;
  }
}

// Download file with redirect following
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const request = (url) => {
      https.get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          // Follow redirect
          request(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download: ${response.statusCode}`));
          return;
        }

        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(dest, () => {});
        reject(err);
      });
    };

    request(url);
  });
}

// Extract archive
function extract(archive, dest, platformInfo) {
  if (platformInfo.ext === '.tar.gz') {
    execFileSync('tar', ['-xzf', archive, '-C', dest], { stdio: 'inherit' });
  } else if (platformInfo.ext === '.zip') {
    // Use PowerShell on Windows
    execFileSync('powershell', ['-Command', `Expand-Archive -Path '${archive}' -DestinationPath '${dest}' -Force`], { stdio: 'inherit' });
  }
}

async function main() {
  const platformInfo = getPlatformInfo();

  if (!platformInfo) {
    console.log('[veld] Unsupported platform:', process.platform, process.arch);
    console.log('[veld] You will need to run the server manually.');
    return;
  }

  console.log('[veld] Installing server binary for', process.platform, process.arch);

  // Create bin directory
  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }

  const binaryPath = path.join(BIN_DIR, platformInfo.binary);

  // Check if already installed
  if (fs.existsSync(binaryPath)) {
    console.log('[veld] Binary already installed at', binaryPath);
    return;
  }

  // Download URL
  const downloadUrl = `https://github.com/${REPO}/releases/download/v${VERSION}/${platformInfo.name}${platformInfo.ext}`;
  const archivePath = path.join(BIN_DIR, `${platformInfo.name}${platformInfo.ext}`);

  console.log('[veld] Downloading from', downloadUrl);

  try {
    await download(downloadUrl, archivePath);
    console.log('[veld] Downloaded archive');

    // Extract
    extract(archivePath, BIN_DIR, platformInfo);
    console.log('[veld] Extracted binary');

    // Clean up archive
    fs.unlinkSync(archivePath);

    // Make executable (Unix)
    if (process.platform !== 'win32') {
      fs.chmodSync(binaryPath, 0o755);
    }

    console.log('[veld] Server binary installed at', binaryPath);
  } catch (err) {
    console.error('[veld] Failed to install binary:', err.message);
    console.log('[veld] You can manually download from:', `https://github.com/${REPO}/releases`);
  }
}

main();
