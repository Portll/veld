export type LLMProvider = "claude-cli" | "groq" | "anthropic" | "openai" | "ollama";

export const config = {
  shodh: {
    apiUrl: process.env.SHODH_API_URL || "http://127.0.0.1:3030",
    apiKey: process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key",
    sharedUserId: process.env.SHODH_SHARED_USER_ID || "keshav",
  },
  llm: {
    provider: (process.env.LLM_PROVIDER || "claude-cli") as LLMProvider,
    groq: {
      apiKey: process.env.GROQ_API_KEY || "",
      model: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
    },
    anthropic: {
      apiKey: process.env.ANTHROPIC_API_KEY || "",
      model: process.env.CLAUDE_MODEL || "claude-sonnet-4-20250514",
    },
    openai: {
      apiKey: process.env.OPENAI_API_KEY || "",
      model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    },
    ollama: {
      baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
      model: process.env.OLLAMA_MODEL || "llama3.2",
    },
  },
  whatsapp: {
    allowedContacts: process.env.ALLOWED_CONTACTS?.split(",").filter(Boolean) || [],
    blockedContacts: process.env.BLOCKED_CONTACTS?.split(",").filter(Boolean) || [],
    autoReplyAll: process.env.AUTO_REPLY_ALL !== "false",
    replyToGroups: process.env.REPLY_TO_GROUPS === "true",
    typingDelay: parseInt(process.env.TYPING_DELAY || "1000"),
    systemPrompt: process.env.SYSTEM_PROMPT || `You are Keshav, Varun's personal AI assistant responding via WhatsApp.

Your personality:
- Friendly, helpful, and efficient
- You speak naturally like a real assistant would
- Keep responses concise - this is WhatsApp, not email
- Use context from memory to personalize responses
- Remember past conversations and reference them when relevant

You help Varun with:
- Answering questions and providing information
- Reminders and scheduling
- Quick research and lookups
- General assistance

If someone other than Varun messages, be polite but brief. Prioritize Varun's requests.
If you don't know something, say so honestly.`,
  },
  debug: process.env.DEBUG === "true",
};

export function shouldReply(jid: string, isGroup: boolean): boolean {
  if (isGroup && !config.whatsapp.replyToGroups) {
    return false;
  }

  const contact = jid.split("@")[0];

  if (config.whatsapp.blockedContacts.includes(contact)) {
    return false;
  }

  if (config.whatsapp.autoReplyAll) {
    return true;
  }

  if (config.whatsapp.allowedContacts.length > 0) {
    return config.whatsapp.allowedContacts.includes(contact);
  }

  return false;
}
