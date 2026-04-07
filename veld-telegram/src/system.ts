import { exec } from "child_process";
import { promisify } from "util";
import { readFile, stat, readdir, unlink } from "fs/promises";
import { join, basename } from "path";
import { tmpdir } from "os";
import si from "systeminformation";

const execAsync = promisify(exec);

export async function takeScreenshot(): Promise<Buffer> {
  const tempPath = join(tmpdir(), `screenshot_${Date.now()}.png`);
  console.log(`[screenshot] Platform: ${process.platform}, saving to: ${tempPath}`);

  try {
    if (process.platform === "win32") {
      const scriptPath = join(tmpdir(), `ss_${Date.now()}.ps1`);
      const psScript = `
[void][Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')
[void][Reflection.Assembly]::LoadWithPartialName('System.Drawing')
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($bounds.Location, (New-Object System.Drawing.Point(0,0)), $bounds.Size)
$bmp.Save('${tempPath.replace(/\\/g, "/")}')
$g.Dispose()
$bmp.Dispose()
`;
      const { writeFile: writeFs } = await import("fs/promises");
      await writeFs(scriptPath, psScript, "utf-8");
      console.log(`[screenshot] Script written to: ${scriptPath}`);

      const { stdout, stderr } = await execAsync(`powershell -ExecutionPolicy Bypass -File "${scriptPath}"`, { timeout: 15000 });
      console.log(`[screenshot] PowerShell output: ${stdout || "(none)"}, stderr: ${stderr || "(none)"}`);
      await unlink(scriptPath).catch(() => {});
    } else if (process.platform === "darwin") {
      await execAsync(`screencapture -x ${tempPath}`);
    } else {
      await execAsync(`import -window root ${tempPath}`);
    }

    const buffer = await readFile(tempPath);
    console.log(`[screenshot] Read ${buffer.length} bytes`);
    await unlink(tempPath).catch(() => {});
    return buffer;
  } catch (error: any) {
    console.error(`[screenshot] Error:`, error);
    throw new Error(`Screenshot failed: ${error.message}`);
  }
}

export async function runCommand(command: string): Promise<string> {
  try {
    const { stdout, stderr } = await execAsync(command, {
      timeout: 30000,
      maxBuffer: 1024 * 1024,
      shell: process.platform === "win32" ? "powershell.exe" : "/bin/bash",
    });
    const output = stdout || stderr || "(no output)";
    return output.slice(0, 4000);
  } catch (error: any) {
    if (error.killed) {
      return "Command timed out (30s limit)";
    }
    return `Error: ${error.message}\n${error.stderr || ""}`.slice(0, 4000);
  }
}

export async function getSystemStatus(): Promise<string> {
  const [cpu, mem, disk, os, time] = await Promise.all([
    si.currentLoad(),
    si.mem(),
    si.fsSize(),
    si.osInfo(),
    si.time(),
  ]);

  const usedMem = ((mem.used / mem.total) * 100).toFixed(1);
  const mainDisk = disk[0];
  const usedDisk = mainDisk
    ? ((mainDisk.used / mainDisk.size) * 100).toFixed(1)
    : "N/A";

  const uptimeHours = Math.floor(time.uptime / 3600);
  const uptimeMins = Math.floor((time.uptime % 3600) / 60);

  return `
*System Status*

*OS:* ${os.distro} ${os.release}
*Hostname:* ${os.hostname}

*CPU:* ${cpu.currentLoad.toFixed(1)}%
*RAM:* ${usedMem}% (${formatBytes(mem.used)} / ${formatBytes(mem.total)})
*Disk:* ${usedDisk}% used

*Uptime:* ${uptimeHours}h ${uptimeMins}m
`.trim();
}

export async function getFileInfo(filePath: string): Promise<{
  exists: boolean;
  isFile: boolean;
  isDir: boolean;
  size: number;
  name: string;
}> {
  try {
    const stats = await stat(filePath);
    return {
      exists: true,
      isFile: stats.isFile(),
      isDir: stats.isDirectory(),
      size: stats.size,
      name: basename(filePath),
    };
  } catch {
    return { exists: false, isFile: false, isDir: false, size: 0, name: "" };
  }
}

export async function readFileContent(filePath: string): Promise<Buffer> {
  return readFile(filePath);
}

export async function listDirectory(dirPath: string): Promise<string> {
  try {
    const entries = await readdir(dirPath, { withFileTypes: true });
    const lines = entries.slice(0, 50).map((entry) => {
      const icon = entry.isDirectory() ? "📁" : "📄";
      return `${icon} ${entry.name}`;
    });
    if (entries.length > 50) {
      lines.push(`... and ${entries.length - 50} more`);
    }
    return lines.join("\n");
  } catch (error: any) {
    return `Error: ${error.message}`;
  }
}

export async function getProcessList(): Promise<string> {
  const processes = await si.processes();
  const top = processes.list
    .sort((a, b) => b.cpu - a.cpu)
    .slice(0, 15)
    .map((p) => `${p.name.slice(0, 20).padEnd(20)} CPU:${p.cpu.toFixed(1)}% MEM:${p.mem.toFixed(1)}%`)
    .join("\n");
  return `*Top Processes:*\n\`\`\`\n${top}\n\`\`\``;
}

export async function killProcess(pid: number): Promise<string> {
  try {
    process.kill(pid);
    return `Process ${pid} killed`;
  } catch (error: any) {
    return `Failed to kill ${pid}: ${error.message}`;
  }
}

export async function lockScreen(): Promise<string> {
  try {
    if (process.platform === "win32") {
      await execAsync("rundll32.exe user32.dll,LockWorkStation");
    } else if (process.platform === "darwin") {
      await execAsync("pmset displaysleepnow");
    } else {
      await execAsync("xdg-screensaver lock");
    }
    return "Screen locked";
  } catch (error: any) {
    return `Failed to lock: ${error.message}`;
  }
}

export async function openApp(appName: string): Promise<string> {
  try {
    const open = (await import("open")).default;
    await open(appName);
    return `Opened: ${appName}`;
  } catch (error: any) {
    return `Failed to open ${appName}: ${error.message}`;
  }
}

export async function getClipboard(): Promise<string> {
  try {
    if (process.platform === "win32") {
      const { stdout } = await execAsync("powershell Get-Clipboard", { encoding: "utf8" });
      return stdout.trim() || "(empty)";
    } else if (process.platform === "darwin") {
      const { stdout } = await execAsync("pbpaste");
      return stdout.trim() || "(empty)";
    } else {
      const { stdout } = await execAsync("xclip -selection clipboard -o");
      return stdout.trim() || "(empty)";
    }
  } catch {
    return "(clipboard unavailable)";
  }
}

export async function setClipboard(text: string): Promise<string> {
  try {
    if (process.platform === "win32") {
      await execAsync(`powershell Set-Clipboard -Value "${text.replace(/"/g, '`"')}"`);
    } else if (process.platform === "darwin") {
      await execAsync(`echo "${text}" | pbcopy`);
    } else {
      await execAsync(`echo "${text}" | xclip -selection clipboard`);
    }
    return "Clipboard updated";
  } catch (error: any) {
    return `Failed: ${error.message}`;
  }
}

export async function shutdown(restart: boolean = false): Promise<string> {
  try {
    if (process.platform === "win32") {
      const cmd = restart ? "shutdown /r /t 5" : "shutdown /s /t 5";
      await execAsync(cmd);
    } else {
      const cmd = restart ? "sudo shutdown -r +1" : "sudo shutdown -h +1";
      await execAsync(cmd);
    }
    return restart ? "Restarting in 5 seconds..." : "Shutting down in 5 seconds...";
  } catch (error: any) {
    return `Failed: ${error.message}`;
  }
}

export async function cancelShutdown(): Promise<string> {
  try {
    if (process.platform === "win32") {
      await execAsync("shutdown /a");
    } else {
      await execAsync("sudo shutdown -c");
    }
    return "Shutdown cancelled";
  } catch (error: any) {
    return `Failed: ${error.message}`;
  }
}

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(1)}GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)}MB`;
}
