import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";
import OpenAI from "openai";
import { spawn } from "child_process";
import { config } from "./config";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

let anthropicClient: Anthropic | null = null;
let groqClient: Groq | null = null;
let openaiClient: OpenAI | null = null;

function buildSystemPrompt(memoryContext: string): string {
  let systemPrompt = config.whatsapp.systemPrompt;
  if (memoryContext) {
    systemPrompt += `\n\n## Relevant context from memory:\n${memoryContext}`;
  }
  return systemPrompt;
}

async function generateWithClaudeCLI(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[]
): Promise<string> {
  const systemPrompt = buildSystemPrompt(memoryContext);

  let fullPrompt = `<system>\n${systemPrompt}\n</system>\n\n`;

  for (const msg of conversationHistory) {
    fullPrompt += `<${msg.role}>\n${msg.content}\n</${msg.role}>\n\n`;
  }

  fullPrompt += `<user>\n${userMessage}\n</user>`;

  return new Promise((resolve, reject) => {
    // Use --dangerously-skip-permissions to avoid interactive prompts
    // and --verbose to see what's happening
    const claude = spawn("claude", ["-p", "--dangerously-skip-permissions"], {
      shell: true,
      env: { ...process.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    claude.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    claude.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    claude.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Claude CLI exited with code ${code}: ${stderr}`));
        return;
      }
      resolve(stdout.trim());
    });

    claude.on("error", (err) => {
      reject(new Error(`Failed to spawn claude CLI: ${err.message}`));
    });

    claude.stdin.write(fullPrompt);
    claude.stdin.end();
  });
}

async function generateWithGroq(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[]
): Promise<string> {
  if (!groqClient) {
    if (!config.llm.groq.apiKey) {
      throw new Error("GROQ_API_KEY is required");
    }
    groqClient = new Groq({ apiKey: config.llm.groq.apiKey });
  }

  const messages: Groq.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: buildSystemPrompt(memoryContext) },
    ...conversationHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: userMessage },
  ];

  const response = await groqClient.chat.completions.create({
    model: config.llm.groq.model,
    messages,
    max_tokens: 1024,
  });

  return response.choices[0]?.message?.content || "No response generated.";
}

async function generateWithAnthropic(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[]
): Promise<string> {
  if (!anthropicClient) {
    if (!config.llm.anthropic.apiKey) {
      throw new Error("ANTHROPIC_API_KEY is required");
    }
    anthropicClient = new Anthropic({ apiKey: config.llm.anthropic.apiKey });
  }

  const messages: Anthropic.MessageParam[] = [
    ...conversationHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user" as const, content: userMessage },
  ];

  const response = await anthropicClient.messages.create({
    model: config.llm.anthropic.model,
    max_tokens: 1024,
    system: buildSystemPrompt(memoryContext),
    messages,
  });

  const textBlock = response.content.find((block) => block.type === "text");
  if (textBlock && textBlock.type === "text") {
    return textBlock.text;
  }

  return "No response generated.";
}

async function generateWithOpenAI(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[]
): Promise<string> {
  if (!openaiClient) {
    if (!config.llm.openai.apiKey) {
      throw new Error("OPENAI_API_KEY is required");
    }
    openaiClient = new OpenAI({ apiKey: config.llm.openai.apiKey });
  }

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: buildSystemPrompt(memoryContext) },
    ...conversationHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: userMessage },
  ];

  const response = await openaiClient.chat.completions.create({
    model: config.llm.openai.model,
    messages,
    max_tokens: 1024,
  });

  return response.choices[0]?.message?.content || "No response generated.";
}

async function generateWithOllama(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[]
): Promise<string> {
  const messages = [
    { role: "system", content: buildSystemPrompt(memoryContext) },
    ...conversationHistory.map((m) => ({
      role: m.role,
      content: m.content,
    })),
    { role: "user", content: userMessage },
  ];

  const response = await fetch(`${config.llm.ollama.baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: config.llm.ollama.model,
      messages,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status}`);
  }

  const data = await response.json();
  return data.message?.content || "No response generated.";
}

export async function generateResponse(
  userMessage: string,
  memoryContext: string,
  conversationHistory: Message[] = []
): Promise<string> {
  try {
    switch (config.llm.provider) {
      case "claude-cli":
        return await generateWithClaudeCLI(userMessage, memoryContext, conversationHistory);
      case "groq":
        return await generateWithGroq(userMessage, memoryContext, conversationHistory);
      case "anthropic":
        return await generateWithAnthropic(userMessage, memoryContext, conversationHistory);
      case "openai":
        return await generateWithOpenAI(userMessage, memoryContext, conversationHistory);
      case "ollama":
        return await generateWithOllama(userMessage, memoryContext, conversationHistory);
      default:
        throw new Error(`Unknown LLM provider: ${config.llm.provider}`);
    }
  } catch (error) {
    console.error("LLM error:", error);
    throw error;
  }
}
