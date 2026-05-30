#!/usr/bin/env bun
/**
 * Veld status line for Claude Code (portable, cross-platform).
 *
 * Claude Code invokes this on each render and pipes a JSON status payload on
 * stdin. We do two things:
 *   1. POST the session's context-window usage to one or more veld servers
 *      (`/api/context_status`, a PUBLIC route — no API key needed). This is what
 *      lights up the "Sessions" card in the veld monitor / TUI.
 *   2. Print a one-line status string (first stdout line becomes the status).
 *
 * This replaces the bash+jq `scripts/veld-statusline.sh`, which cannot run on
 * Windows. Bun is already the runtime used by `hooks/memory-hook.ts`, so it is
 * guaranteed present wherever the hooks work.
 *
 * Server routing: posts to every URL in `VELD_STATUSLINE_URLS` (comma-separated),
 * falling back to `VELD_API_URL`, then `http://127.0.0.1:3030`. Posting to several
 * instances is cheap (fire-and-forget) and means the session shows under whichever
 * veld instance the monitor is currently viewing.
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const POST_TIMEOUT_MS = 1200;

/** Platform veld config path (mirrors hooks/memory-hook.ts). */
function veldConfigPath(): string {
  const xdg = process.env.XDG_CONFIG_HOME;
  if (xdg) return join(xdg, "veld", "config.toml");
  if (process.platform === "win32") {
    const appdata = process.env.APPDATA;
    if (appdata) return join(appdata, "veld", "config.toml");
  } else if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", "veld", "config.toml");
  }
  return join(homedir(), ".config", "veld", "config.toml");
}

/**
 * Resolve the veld API key (VELD_API_KEY env → `api_key` in config.toml → "").
 * The context-status POST route is authenticated (only the GET is public), so the
 * status line must present a key or the server returns 401 and the session never
 * registers. Mirrors `resolveApiKey()` in hooks/memory-hook.ts.
 */
function resolveApiKey(): string {
  const envKey = process.env.VELD_API_KEY;
  if (envKey && envKey.trim()) return envKey.trim();
  const path = veldConfigPath();
  if (!existsSync(path)) return "";
  try {
    for (const raw of readFileSync(path, "utf-8").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq === -1 || line.slice(0, eq).trim() !== "api_key") continue;
      const value = line.slice(eq + 1).split("#")[0].trim().replace(/^["']|["']$/g, "");
      if (value) return value;
    }
  } catch {
    // unreadable config — fall through to empty
  }
  return "";
}

function resolveUrls(): string[] {
  const raw =
    process.env.VELD_STATUSLINE_URLS ||
    process.env.VELD_API_URL ||
    "http://127.0.0.1:3030";
  return raw
    .split(",")
    .map((s) => s.trim().replace(/\/+$/, ""))
    .filter((s) => s.length > 0);
}

async function readStdin(): Promise<string> {
  try {
    return await Bun.stdin.text();
  } catch {
    return "";
  }
}

function num(v: unknown, fallback = 0): number {
  const n = typeof v === "string" ? parseInt(v, 10) : (v as number);
  return Number.isFinite(n) ? (n as number) : fallback;
}

async function postStatus(
  url: string,
  apiKey: string,
  body: Record<string, unknown>
): Promise<void> {
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), POST_TIMEOUT_MS);
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers["X-API-Key"] = apiKey;
    await fetch(`${url}/api/context_status`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(t);
  } catch {
    // Fire-and-forget: the status line must never fail because a server is down.
  }
}

async function main() {
  const rawInput = await readStdin();
  let input: Record<string, any> = {};
  try {
    input = rawInput ? JSON.parse(rawInput) : {};
  } catch {
    input = {};
  }

  const sessionId: string = input?.session_id || "unknown";
  const model: string =
    input?.model?.display_name || input?.model?.id || "Claude";
  const cwd: string =
    input?.workspace?.current_dir || input?.cwd || process.cwd() || "";

  const cw = input?.context_window || {};
  const budget = num(cw?.context_window_size, 200000);
  const usage = cw?.current_usage || {};
  const tokensUsed =
    num(usage?.input_tokens) +
    num(usage?.cache_creation_input_tokens) +
    num(usage?.cache_read_input_tokens);

  const percent = budget > 0 ? Math.round((tokensUsed * 100) / budget) : 0;

  // Fire the status POST(s) to every configured instance, in parallel, but bound
  // total wall-clock so the status line stays snappy.
  const body = {
    session_id: String(sessionId).slice(0, 128),
    tokens_used: tokensUsed,
    tokens_budget: budget,
    current_dir: cwd,
    model,
  };
  const apiKey = resolveApiKey();
  await Promise.race([
    Promise.allSettled(resolveUrls().map((u) => postStatus(u, apiKey, body))),
    new Promise((r) => setTimeout(r, POST_TIMEOUT_MS + 100)),
  ]);

  // ---- formatted status line --------------------------------------------
  const fmt = (n: number) => (n >= 1000 ? `${Math.floor(n / 1000)}k` : `${n}`);
  // ANSI color by usage band (green < 50, yellow < 80, red otherwise).
  const color = percent < 50 ? "\x1b[32m" : percent < 80 ? "\x1b[33m" : "\x1b[31m";
  const reset = "\x1b[0m";
  const dir = cwd ? cwd.split(/[\\/]/).filter(Boolean).pop() || "" : "";

  process.stdout.write(
    `${color}${percent}%${reset} ${fmt(tokensUsed)}/${fmt(budget)} | ${model}${dir ? ` | ${dir}` : ""}`
  );
}

main();
