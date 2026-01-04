import { spawn } from "child_process";
import { config } from "./config";

interface Memory {
  id: string;
  content: string;
  memory_type: string;
  relevance_score?: number;
}

interface ProactiveContextResponse {
  memories: Memory[];
  total_found: number;
}

// Get context from shodh-memory
export async function getMemoryContext(userMessage: string): Promise<string> {
  try {
    const response = await fetch(`${config.shodh.apiUrl}/api/proactive_context`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: "telegram-user",
        context: userMessage,
        auto_ingest: false,
        max_results: 5,
        semantic_threshold: 0.5,
      }),
    });

    if (!response.ok) return "";

    const data = (await response.json()) as ProactiveContextResponse;
    if (!data.memories?.length) return "";

    return data.memories.map((m) => `[${m.memory_type}] ${m.content}`).join("\n");
  } catch {
    return "";
  }
}

// Store conversation in memory
export async function rememberConversation(userMessage: string, response: string): Promise<void> {
  try {
    await fetch(`${config.shodh.apiUrl}/api/remember`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: "telegram-user",
        content: `User: ${userMessage}\nClaude: ${response}`,
        memory_type: "Conversation",
        tags: ["telegram", "chat"],
      }),
    });
  } catch {
    // Ignore errors
  }
}

// Call Claude Code CLI
export async function askClaude(userMessage: string, memoryContext: string = ""): Promise<string> {
  return new Promise((resolve, reject) => {
    let systemContext = `You are Claude, running via Telegram to control a Windows laptop remotely.
You have FULL CLI access - you can run any command, read/write files, take screenshots, etc.

IMPORTANT: If the user asks you to DO something - actually DO IT, don't just explain.

SPECIAL TELEGRAM COMMANDS (the bot will execute these):
- [SCREENSHOT] - Takes and sends a screenshot to Telegram
- [SEND_FILE:path] - Sends a file to Telegram, e.g. [SEND_FILE:C:/Users/Varun Sharma/Desktop/notes.txt]
- [LIST_DIR:path] - Lists directory contents, e.g. [LIST_DIR:C:/Users/Varun Sharma/Desktop]

Use these commands in your response when the user asks for screenshots, files, or directory listings.
You can also run any Bash/PowerShell command directly using your tools.

Keep responses brief - this is mobile chat.`;

    if (memoryContext) {
      systemContext += `\n\nRelevant context from memory:\n${memoryContext}`;
    }

    const fullPrompt = `${systemContext}\n\nUser: ${userMessage}`;

    const claude = spawn("claude", ["-p", "--dangerously-skip-permissions"], {
      shell: true,
      env: { ...process.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      claude.kill();
      resolve("Response timed out. Try a simpler question.");
    }, 60000); // 60 second timeout

    claude.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    claude.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    claude.on("close", (code) => {
      clearTimeout(timeout);
      if (timedOut) return;

      if (code !== 0 && !stdout) {
        resolve(`Error: ${stderr || "Claude CLI failed"}`);
        return;
      }

      // Truncate long responses for Telegram
      let response = stdout.trim();
      if (response.length > 4000) {
        response = response.slice(0, 3900) + "\n\n... (truncated)";
      }

      resolve(response || "No response from Claude");
    });

    claude.on("error", (err) => {
      clearTimeout(timeout);
      resolve(`Failed to run Claude CLI: ${err.message}`);
    });

    claude.stdin.write(fullPrompt);
    claude.stdin.end();
  });
}

// Execute a command via Claude Code (gives Claude full CLI access)
export async function executeWithClaude(task: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const prompt = `Execute this task on the laptop: ${task}

Do it now - don't just explain. Actually run the commands needed.
Keep the response brief - just confirm what you did.`;

    const claude = spawn("claude", ["-p", "--dangerously-skip-permissions"], {
      shell: true,
      cwd: process.cwd(),
      env: { ...process.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      claude.kill();
      resolve("Task timed out (60s limit)");
    }, 60000);

    claude.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    claude.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    claude.on("close", (code) => {
      clearTimeout(timeout);
      if (timedOut) return;

      let response = stdout.trim();
      if (response.length > 4000) {
        response = response.slice(0, 3900) + "\n\n... (truncated)";
      }

      resolve(response || stderr || "Task completed (no output)");
    });

    claude.on("error", (err) => {
      clearTimeout(timeout);
      resolve(`Failed: ${err.message}`);
    });

    claude.stdin.write(prompt);
    claude.stdin.end();
  });
}
