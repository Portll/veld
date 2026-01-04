import TelegramBot from "node-telegram-bot-api";
import { config, isOwner, verifyPin } from "./config";
import {
  takeScreenshot,
  runCommand,
  getSystemStatus,
  getFileInfo,
  readFileContent,
  listDirectory,
  getProcessList,
  killProcess,
  lockScreen,
  openApp,
  getClipboard,
  setClipboard,
  shutdown,
  cancelShutdown,
} from "./system";
import { askClaude, executeWithClaude, getMemoryContext, rememberConversation } from "./claude";

if (!config.telegram.token) {
  console.error("❌ TELEGRAM_BOT_TOKEN is required");
  console.error("   Get one from @BotFather on Telegram");
  process.exit(1);
}

if (!config.telegram.ownerId) {
  console.error("❌ OWNER_ID is required");
  console.error("   Send /id to @userinfobot to get your Telegram user ID");
  process.exit(1);
}

const bot = new TelegramBot(config.telegram.token, { polling: true });

console.log(`
🤖 shodh-telegram starting...
👤 Owner ID: ${config.telegram.ownerId}
🔐 PIN protection: ${config.security.pin ? "enabled" : "disabled"}
`);

// Debug: Log ALL incoming messages
bot.on("message", (msg) => {
  console.log(`[debug] Received: "${msg.text}" from ${msg.from?.id}`);
});

// Security middleware
function ownerOnly(
  handler: (msg: TelegramBot.Message, match: RegExpExecArray | null) => Promise<void>
) {
  return async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    if (!isOwner(msg.from?.id || 0)) {
      await bot.sendMessage(msg.chat.id, "⛔ Access denied. Owner only.");
      console.log(`⚠️ Unauthorized access attempt from ${msg.from?.id}`);
      return;
    }
    try {
      await handler(msg, match);
    } catch (error: any) {
      console.error("Command error:", error);
      await bot.sendMessage(msg.chat.id, `❌ Error: ${error.message}`);
    }
  };
}

// /start - Welcome message
bot.onText(/\/start/, ownerOnly(async (msg) => {
  await bot.sendMessage(msg.chat.id, `
*shodh-telegram* 🎛️

Your laptop + WhatsApp command center.

*WhatsApp (Keshav):*
/inbox - WhatsApp message summary

*Laptop Control:*
/screenshot - Take screenshot
/status - System info
/run <cmd> - Execute command
/do <task> - Claude does the task

*Files:*
/file <path> - Send file
/ls <path> - List directory

*System:*
/ps - Process list
/kill <pid> - Kill process
/lock - Lock screen
/open <app> - Open application

/help - Full command list
`, { parse_mode: "Markdown" });
}));

// /inbox - Get WhatsApp inbox summary from Keshav
bot.onText(/\/inbox/, ownerOnly(async (msg) => {
  await bot.sendMessage(msg.chat.id, "📬 Fetching WhatsApp inbox...");

  try {
    const response = await fetch(`${config.shodh.apiUrl}/api/recall/tags`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: "keshav",
        tags: ["whatsapp-inbox"],
        limit: 20,
      }),
    });

    if (!response.ok) {
      await bot.sendMessage(msg.chat.id, "❌ Failed to fetch inbox");
      return;
    }

    const data = await response.json() as { memories: Array<{ experience: { content: string }; tags: string[] }> };

    if (!data.memories?.length) {
      await bot.sendMessage(msg.chat.id, "📭 No new WhatsApp messages.");
      return;
    }

    const inlineButtons: Array<{ text: string; url: string }> = [];

    for (const mem of data.memories) {
      try {
        const rawContent = mem.experience?.content || "";
        const content = rawContent.replace("INBOX: ", "");
        const item = JSON.parse(content);
        const msgPreview = item.message?.substring(0, 50) || "(no message)";
        const priority = item.priority === "high" ? "🔴" : item.hasActionItem ? "📋" : "💬";
        const contact = item.contact || "Unknown";
        const phone = item.phone || "";

        if (phone && phone !== "status") {
          inlineButtons.push({
            text: `${priority} ${contact}: ${msgPreview.substring(0, 25)}${msgPreview.length > 25 ? '...' : ''}`,
            url: `https://wa.me/${phone}`,
          });
        }
      } catch {
        // Skip unparseable
      }
    }

    if (inlineButtons.length === 0) {
      await bot.sendMessage(msg.chat.id, "📭 No actionable WhatsApp messages.");
      return;
    }

    const keyboard = inlineButtons.map(btn => [{ text: btn.text, url: btn.url }]);

    await bot.sendMessage(
      msg.chat.id,
      `📬 *WhatsApp Inbox* (${inlineButtons.length} messages)\n\nTap to open chat:`,
      {
        parse_mode: "Markdown",
        reply_markup: { inline_keyboard: keyboard },
      }
    );
  } catch (error: any) {
    await bot.sendMessage(msg.chat.id, `❌ Error: ${error.message}`);
  }
}));

// /help
bot.onText(/\/help/, ownerOnly(async (msg) => {
  await bot.sendMessage(msg.chat.id, `
*📱 shodh-telegram*

*📬 WhatsApp*
/inbox - Messages from Keshav (tap to open chat)

*🤖 Claude AI*
/do <task> - Claude executes task on laptop
Or just type naturally - Claude responds

*📸 Screen & Files*
/screenshot - Capture screen
/file <path> - Send any file
/ls <path> - List directory

*💻 System*
/status - CPU, RAM, disk, uptime
/ps - Top processes
/kill <pid> - Kill a process
/run <cmd> - Run shell command

*🔧 Control*
/lock - Lock screen
/open <app/url> - Open app or URL
/clip - Get clipboard
/setclip <text> - Set clipboard

*⚡ Power*
/shutdown [pin] - Shutdown
/restart [pin] - Restart
/cancel - Cancel shutdown

_Claude can also send files, screenshots, and list directories when you ask._
`, { parse_mode: "Markdown" });
}));

// /screenshot
bot.onText(/\/screenshot/, ownerOnly(async (msg) => {
  console.log("[cmd] /screenshot triggered");
  await bot.sendMessage(msg.chat.id, "📸 Taking screenshot...");
  const image = await takeScreenshot();
  console.log(`[cmd] Screenshot captured, size: ${image.length} bytes`);
  await bot.sendPhoto(msg.chat.id, image, { caption: `Screenshot - ${new Date().toLocaleTimeString()}` });
  console.log("[cmd] Screenshot sent");
}));

// /status
bot.onText(/\/status/, ownerOnly(async (msg) => {
  const status = await getSystemStatus();
  await bot.sendMessage(msg.chat.id, status, { parse_mode: "Markdown" });
}));

// /run <command>
bot.onText(/\/run (.+)/, ownerOnly(async (msg, match) => {
  const command = match?.[1];
  if (!command) return;

  await bot.sendMessage(msg.chat.id, `⚙️ Running: \`${command}\``, { parse_mode: "Markdown" });
  const output = await runCommand(command);
  await bot.sendMessage(msg.chat.id, `\`\`\`\n${output}\n\`\`\``, { parse_mode: "Markdown" });
}));

// /file <path>
bot.onText(/\/file (.+)/, ownerOnly(async (msg, match) => {
  const filePath = match?.[1]?.trim();
  if (!filePath) return;

  const info = await getFileInfo(filePath);

  if (!info.exists) {
    await bot.sendMessage(msg.chat.id, `❌ File not found: ${filePath}`);
    return;
  }

  if (info.isDir) {
    await bot.sendMessage(msg.chat.id, `📁 That's a directory. Use /ls ${filePath}`);
    return;
  }

  if (info.size > 50 * 1024 * 1024) {
    await bot.sendMessage(msg.chat.id, `❌ File too large (${(info.size / 1024 / 1024).toFixed(1)}MB). Telegram limit is 50MB.`);
    return;
  }

  await bot.sendMessage(msg.chat.id, `📤 Sending ${info.name}...`);
  const content = await readFileContent(filePath);
  await bot.sendDocument(msg.chat.id, content, {}, { filename: info.name });
}));

// /ls <path>
bot.onText(/\/ls(?: (.+))?/, ownerOnly(async (msg, match) => {
  const dirPath = match?.[1]?.trim() || ".";
  const listing = await listDirectory(dirPath);
  await bot.sendMessage(msg.chat.id, `*${dirPath}*\n\n${listing}`, { parse_mode: "Markdown" });
}));

// /ps - Process list
bot.onText(/\/ps/, ownerOnly(async (msg) => {
  const list = await getProcessList();
  await bot.sendMessage(msg.chat.id, list, { parse_mode: "Markdown" });
}));

// /kill <pid>
bot.onText(/\/kill (\d+)/, ownerOnly(async (msg, match) => {
  const pid = parseInt(match?.[1] || "0");
  if (!pid) return;
  const result = await killProcess(pid);
  await bot.sendMessage(msg.chat.id, result);
}));

// /lock
bot.onText(/\/lock/, ownerOnly(async (msg) => {
  const result = await lockScreen();
  await bot.sendMessage(msg.chat.id, `🔒 ${result}`);
}));

// /open <app>
bot.onText(/\/open (.+)/, ownerOnly(async (msg, match) => {
  const app = match?.[1]?.trim();
  if (!app) return;
  const result = await openApp(app);
  await bot.sendMessage(msg.chat.id, result);
}));

// /clip - Get clipboard
bot.onText(/\/clip$/, ownerOnly(async (msg) => {
  const content = await getClipboard();
  await bot.sendMessage(msg.chat.id, `📋 Clipboard:\n\`\`\`\n${content.slice(0, 3000)}\n\`\`\``, { parse_mode: "Markdown" });
}));

// /setclip <text>
bot.onText(/\/setclip (.+)/, ownerOnly(async (msg, match) => {
  const text = match?.[1];
  if (!text) return;
  const result = await setClipboard(text);
  await bot.sendMessage(msg.chat.id, `📋 ${result}`);
}));

// /shutdown [pin]
bot.onText(/\/shutdown(?: (\d+))?/, ownerOnly(async (msg, match) => {
  const pin = match?.[1] || "";
  if (!verifyPin(pin)) {
    await bot.sendMessage(msg.chat.id, "🔐 PIN required: /shutdown <pin>");
    return;
  }
  const result = await shutdown(false);
  await bot.sendMessage(msg.chat.id, `⚡ ${result}`);
}));

// /restart [pin]
bot.onText(/\/restart(?: (\d+))?/, ownerOnly(async (msg, match) => {
  const pin = match?.[1] || "";
  if (!verifyPin(pin)) {
    await bot.sendMessage(msg.chat.id, "🔐 PIN required: /restart <pin>");
    return;
  }
  const result = await shutdown(true);
  await bot.sendMessage(msg.chat.id, `🔄 ${result}`);
}));

// /cancel - Cancel shutdown
bot.onText(/\/cancel/, ownerOnly(async (msg) => {
  const result = await cancelShutdown();
  await bot.sendMessage(msg.chat.id, result);
}));


// /do <task> - Execute task with Claude Code
bot.onText(/\/do (.+)/, ownerOnly(async (msg, match) => {
  const task = match?.[1];
  if (!task) return;

  await bot.sendMessage(msg.chat.id, "🤖 Claude is working on it...");
  const result = await executeWithClaude(task);
  await bot.sendMessage(msg.chat.id, result, { parse_mode: "Markdown" }).catch(() => 
    bot.sendMessage(msg.chat.id, result) // Fallback without markdown
  );
}));

// Parse and execute special commands from Claude's response
async function executeSpecialCommands(chatId: number, response: string): Promise<string> {
  let cleanResponse = response;

  // [SCREENSHOT] - Take and send screenshot
  if (response.includes("[SCREENSHOT]")) {
    cleanResponse = cleanResponse.replace(/\[SCREENSHOT\]/g, "");
    try {
      const image = await takeScreenshot();
      await bot.sendPhoto(chatId, image, { caption: `📸 Screenshot - ${new Date().toLocaleTimeString()}` });
    } catch (err: any) {
      await bot.sendMessage(chatId, `❌ Screenshot failed: ${err.message}`);
    }
  }

  // [SEND_FILE:path] - Send a file
  const fileMatches = response.matchAll(/\[SEND_FILE:([^\]]+)\]/g);
  for (const match of fileMatches) {
    const filePath = match[1].trim();
    cleanResponse = cleanResponse.replace(match[0], "");
    try {
      const info = await getFileInfo(filePath);
      if (!info.exists) {
        await bot.sendMessage(chatId, `❌ File not found: ${filePath}`);
      } else if (info.isDir) {
        await bot.sendMessage(chatId, `📁 That's a directory. Use [LIST_DIR:${filePath}]`);
      } else if (info.size > 50 * 1024 * 1024) {
        await bot.sendMessage(chatId, `❌ File too large (${(info.size / 1024 / 1024).toFixed(1)}MB)`);
      } else {
        const content = await readFileContent(filePath);
        await bot.sendDocument(chatId, content, {}, { filename: info.name });
      }
    } catch (err: any) {
      await bot.sendMessage(chatId, `❌ Failed to send file: ${err.message}`);
    }
  }

  // [LIST_DIR:path] - List directory contents
  const dirMatches = response.matchAll(/\[LIST_DIR:([^\]]+)\]/g);
  for (const match of dirMatches) {
    const dirPath = match[1].trim();
    cleanResponse = cleanResponse.replace(match[0], "");
    try {
      const listing = await listDirectory(dirPath);
      await bot.sendMessage(chatId, `📁 *${dirPath}*\n\n${listing}`, { parse_mode: "Markdown" });
    } catch (err: any) {
      await bot.sendMessage(chatId, `❌ Failed to list directory: ${err.message}`);
    }
  }

  return cleanResponse.trim();
}

// Handle plain text messages - send to Claude
bot.on("message", async (msg) => {
  if (!msg.text || msg.text.startsWith("/")) return;
  if (!isOwner(msg.from?.id || 0)) return;

  try {
    await bot.sendChatAction(msg.chat.id, "typing");

    // Get memory context
    const memoryContext = await getMemoryContext(msg.text);

    // Ask Claude
    const response = await askClaude(msg.text, memoryContext);

    // Store in memory
    await rememberConversation(msg.text, response);

    // Execute special commands and get cleaned response
    const cleanResponse = await executeSpecialCommands(msg.chat.id, response);

    // Send text response if any remains
    if (cleanResponse) {
      await bot.sendMessage(msg.chat.id, cleanResponse, { parse_mode: "Markdown" }).catch(() =>
        bot.sendMessage(msg.chat.id, cleanResponse)
      );
    }
  } catch (error: any) {
    await bot.sendMessage(msg.chat.id, `❌ Error: ${error.message}`);
  }
});



console.log("✅ Bot is running! Send /start to your bot on Telegram.");
