import type { WASocket, WAMessage } from "@whiskeysockets/baileys";
import { config, shouldReply } from "./config";
import { getProactiveContext, remember } from "./memory";
import { generateResponse, type Message } from "./llm";

async function sendTelegramAlert(message: string): Promise<void> {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_OWNER_ID;

  if (!token || !chatId) return;

  try {
    await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text: message,
        parse_mode: "Markdown",
      }),
    });
  } catch (err) {
    console.error("Failed to send Telegram alert:", err);
  }
}

const conversationHistory = new Map<string, Message[]>();
const MAX_HISTORY = 10;

interface InboxItem {
  timestamp: Date;
  contact: string;
  phone: string;
  message: string;
  response: string;
  hasActionItem: boolean;
  priority: "high" | "normal" | "low";
}

const inbox: InboxItem[] = [];

function detectPriority(text: string, contactName: string): "high" | "normal" | "low" {
  const urgentWords = ["urgent", "asap", "emergency", "important", "call me", "immediately", "जरूरी", "फौरन"];
  const lowPriorityWords = ["fyi", "no rush", "whenever", "just sharing"];

  const lowerText = text.toLowerCase();

  if (urgentWords.some(w => lowerText.includes(w))) return "high";
  if (lowPriorityWords.some(w => lowerText.includes(w))) return "low";
  return "normal";
}

function detectActionItem(text: string): boolean {
  const actionWords = ["please", "can you", "need you to", "remind", "call", "send", "check", "book", "schedule", "pay", "transfer", "buy", "get", "bring", "karo", "kar do", "bhejo", "batao"];
  const lowerText = text.toLowerCase();
  return actionWords.some(w => lowerText.includes(w));
}

export function getInboxSummary(): string {
  if (inbox.length === 0) {
    return "No messages while you were away.";
  }

  const highPriority = inbox.filter(i => i.priority === "high");
  const actionItems = inbox.filter(i => i.hasActionItem);
  const others = inbox.filter(i => i.priority !== "high" && !i.hasActionItem);

  let summary = `📬 *Inbox Summary* (${inbox.length} messages)\n\n`;

  if (highPriority.length > 0) {
    summary += `🔴 *Urgent (${highPriority.length}):*\n`;
    highPriority.forEach(item => {
      summary += `• ${item.contact}: "${item.message.substring(0, 50)}${item.message.length > 50 ? '...' : ''}"\n`;
    });
    summary += "\n";
  }

  if (actionItems.length > 0) {
    summary += `📋 *Action Items (${actionItems.length}):*\n`;
    actionItems.forEach(item => {
      if (item.priority !== "high") {
        summary += `• ${item.contact}: "${item.message.substring(0, 50)}${item.message.length > 50 ? '...' : ''}"\n`;
      }
    });
    summary += "\n";
  }

  if (others.length > 0) {
    summary += `💬 *Other Messages (${others.length}):*\n`;
    others.slice(0, 5).forEach(item => {
      summary += `• ${item.contact}: "${item.message.substring(0, 40)}${item.message.length > 40 ? '...' : ''}"\n`;
    });
    if (others.length > 5) {
      summary += `  ...and ${others.length - 5} more\n`;
    }
  }

  return summary;
}

export function clearInbox(): void {
  inbox.length = 0;
}

function getHistory(jid: string): Message[] {
  return conversationHistory.get(jid) || [];
}

function addToHistory(jid: string, role: "user" | "assistant", content: string) {
  const history = getHistory(jid);
  history.push({ role, content });

  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  conversationHistory.set(jid, history);
}

function extractMessageText(message: WAMessage): string | null {
  const msg = message.message;
  if (!msg) return null;

  if (msg.conversation) {
    return msg.conversation;
  }
  if (msg.extendedTextMessage?.text) {
    return msg.extendedTextMessage.text;
  }
  if (msg.imageMessage?.caption) {
    return `[Image] ${msg.imageMessage.caption}`;
  }
  if (msg.videoMessage?.caption) {
    return `[Video] ${msg.videoMessage.caption}`;
  }
  if (msg.documentMessage?.caption) {
    return `[Document] ${msg.documentMessage.caption}`;
  }

  return null;
}

function getContactName(message: WAMessage): string {
  return message.pushName || message.key.remoteJid?.split("@")[0] || "Unknown";
}

export async function handleMessage(
  sock: WASocket,
  message: WAMessage
): Promise<void> {
  const jid = message.key.remoteJid;
  if (!jid) return;

  // Ignore Status broadcasts (like Instagram Stories)
  if (jid === "status@broadcast") {
    return;
  }

  if (message.key.fromMe) return;

  const isGroup = jid.endsWith("@g.us");

  if (!shouldReply(jid, isGroup)) {
    if (config.debug) {
      console.log(`Skipping message from ${jid} (not in allowed list)`);
    }
    return;
  }

  const text = extractMessageText(message);
  if (!text) {
    if (config.debug) {
      console.log(`No text content in message from ${jid}`);
    }
    return;
  }

  const contactName = getContactName(message);
  const rawUserId = jid.split("@")[0];
  const userId = config.shodh.sharedUserId || rawUserId;

  console.log(`\n📩 Message from ${contactName} (${userId}): ${text}`);

  try {
    await sock.sendPresenceUpdate("composing", jid);

    const memoryContext = await getProactiveContext(
      userId,
      `${contactName} says: ${text}`
    );

    // Always log memory status
    if (memoryContext) {
      console.log(`📚 Memory retrieved (${memoryContext.length} chars):\n${memoryContext.substring(0, 500)}${memoryContext.length > 500 ? '...' : ''}`);
    } else {
      console.log(`📚 No memory context found for user: ${userId}`);
    }

    const history = getHistory(jid);
    const response = await generateResponse(text, memoryContext, history);

    addToHistory(jid, "user", text);
    addToHistory(jid, "assistant", response);

    const priority = detectPriority(text, contactName);
    const hasActionItem = detectActionItem(text);

    // Add to inbox for summary
    inbox.push({
      timestamp: new Date(),
      contact: contactName,
      phone: rawUserId,
      message: text,
      response,
      hasActionItem,
      priority,
    });

    // Store conversation
    await remember(
      userId,
      `${contactName}: ${text}\nKeshav: ${response}`,
      "Conversation",
      ["whatsapp", "keshav", contactName.toLowerCase().replace(/\s+/g, "-")]
    );

    // Store in inbox for cross-bot access (Telegram can read this)
    const inboxEntry = JSON.stringify({
      time: new Date().toISOString(),
      contact: contactName,
      phone: rawUserId,
      message: text,
      priority,
      hasActionItem,
    });
    await remember(
      userId,
      `INBOX: ${inboxEntry}`,
      "Context",
      ["whatsapp-inbox", "unread", priority === "high" ? "urgent" : "normal"]
    );

    // Store action items separately for easy retrieval
    if (hasActionItem) {
      await remember(
        userId,
        `ACTION ITEM from ${contactName} (${rawUserId}): ${text}`,
        "Task",
        ["whatsapp", "action-item", "pending", contactName.toLowerCase().replace(/\s+/g, "-")]
      );
      console.log(`📋 Action item detected from ${contactName}`);
    }

    if (priority === "high") {
      console.log(`🔴 URGENT message from ${contactName}`);
      await sendTelegramAlert(
        `🔴 *URGENT WhatsApp*\n\n*${contactName}*: ${text.substring(0, 200)}${text.length > 200 ? '...' : ''}\n\n[Open Chat](https://wa.me/${rawUserId})`
      );
    } else if (hasActionItem) {
      await sendTelegramAlert(
        `📋 *Action Item*\n\n*${contactName}*: ${text.substring(0, 200)}${text.length > 200 ? '...' : ''}\n\n[Open Chat](https://wa.me/${rawUserId})`
      );
    }

    if (config.whatsapp.typingDelay > 0) {
      await new Promise((resolve) =>
        setTimeout(resolve, config.whatsapp.typingDelay)
      );
    }

    await sock.sendPresenceUpdate("paused", jid);

    await sock.sendMessage(jid, { text: response });

    console.log(`✅ Replied to ${contactName}: ${response.substring(0, 100)}...`);
  } catch (error) {
    console.error(`❌ Error handling message from ${contactName}:`, error);
    await sock.sendPresenceUpdate("paused", jid);
  }
}
