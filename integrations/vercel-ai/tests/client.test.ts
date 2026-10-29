import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_BASE_URL, VeldClient, VeldError } from "../src/client.js";
import { BASE_URL, fakeFetch, json, recallMemory } from "./helpers.js";

function client(handler: Parameters<typeof fakeFetch>[0], options = {}) {
  const fake = fakeFetch(handler);
  const veld = new VeldClient({
    baseUrl: BASE_URL,
    apiKey: "test-key-7f3a",
    userId: "app-user",
    fetch: fake.fetch,
    ...options,
  });
  return { veld, calls: fake.calls };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("configuration", () => {
  it("reads the server, key and user from the environment", () => {
    vi.stubEnv("VELD_API_URL", "https://veld.example.com/");
    vi.stubEnv("VELD_API_KEY", "env-key");
    vi.stubEnv("VELD_USER_ID", "env-user");
    const veld = new VeldClient();
    expect(veld.baseUrl).toBe("https://veld.example.com");
    expect(veld.userId).toBe("env-user");
  });

  it("defaults to the local server and the default user", () => {
    vi.stubEnv("VELD_API_URL", undefined);
    vi.stubEnv("VELD_USER_ID", undefined);
    const veld = new VeldClient({ apiKey: "k" });
    expect(veld.baseUrl).toBe(DEFAULT_BASE_URL);
    expect(DEFAULT_BASE_URL).toBe("http://127.0.0.1:3030");
    expect(veld.userId).toBe("default");
  });

  it("refuses to start without an API key", () => {
    vi.stubEnv("VELD_API_KEY", undefined);
    expect(() => new VeldClient()).toThrow(/API key required/);
  });

  it("warns when a remote server is reached over plain HTTP", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.stubEnv("VELD_ALLOW_HTTP", undefined);
    new VeldClient({ apiKey: "k", baseUrl: "http://veld.example.com" });
    new VeldClient({ apiKey: "k", baseUrl: "http://127.0.0.1:3030" });
    new VeldClient({ apiKey: "k", baseUrl: "https://veld.example.com" });
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("does not expose the API key", () => {
    const veld = new VeldClient({ apiKey: "secret-key-5521" });
    expect(JSON.stringify(veld)).not.toContain("secret-key-5521");
    expect(Object.values(veld)).not.toContain("secret-key-5521");
  });
});

describe("routes", () => {
  it("remember posts the body with the API key and returns the id", async () => {
    const { veld, calls } = client(() => json(200, { id: "mem-1", success: true }));
    const reply = await veld.remember({
      content: "The user prefers dark mode",
      memoryType: "Decision",
      tags: ["ui"],
      sourceType: "ai_generated",
    });
    expect(reply.id).toBe("mem-1");
    expect(calls[0]).toMatchObject({
      url: `${BASE_URL}/api/remember`,
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": "test-key-7f3a" },
      body: {
        user_id: "app-user",
        content: "The user prefers dark mode",
        memory_type: "Decision",
        tags: ["ui"],
        source_type: "ai_generated",
      },
    });
  });

  it("remember omits fields that were not given", async () => {
    const { veld, calls } = client(() => json(200, { id: "m", success: true }));
    await veld.remember({ content: "x", tags: [] });
    expect(calls[0]?.body).toEqual({ user_id: "app-user", content: "x" });
  });

  it("remember treats an unsuccessful reply as a failure", async () => {
    const { veld } = client(() => json(200, { id: "", success: false }));
    await expect(veld.remember({ content: "x" })).rejects.toMatchObject({ code: "NOT_STORED" });
  });

  it("recall posts the query and returns the memories", async () => {
    const memory = recallMemory("m-1", "Deadline is Friday");
    const { veld, calls } = client(() => json(200, { memories: [memory], count: 1 }));
    const reply = await veld.recall({ query: "deadline", limit: 3, mode: "semantic", tags: ["a"] });
    expect(reply.memories).toEqual([memory]);
    expect(calls[0]?.body).toEqual({
      user_id: "app-user",
      query: "deadline",
      limit: 3,
      mode: "semantic",
      tags: ["a"],
    });
  });

  it("recall rejects a reply without a memories list", async () => {
    const { veld } = client(() => json(200, { count: 0 }));
    await expect(veld.recall({ query: "q" })).rejects.toMatchObject({ code: "INVALID_RESPONSE" });
  });

  it("list sends tags comma-separated", async () => {
    const { veld, calls } = client(() => json(200, { memories: [], total: 0 }));
    await veld.list({ limit: 10, offset: 20, type: "Decision", tags: ["a", "b"] });
    expect(calls[0]).toMatchObject({
      path: "/api/memories",
      body: { user_id: "app-user", limit: 10, offset: 20, type: "Decision", tags: "a,b" },
    });
  });

  it("forget deletes by id for this user", async () => {
    const { veld, calls } = client(() => json(200, { success: true, id: "a/b", message: "" }));
    await veld.forget("a/b");
    expect(calls[0]).toMatchObject({
      method: "DELETE",
      path: "/api/memory/a%2Fb?user_id=app-user",
      body: undefined,
    });
  });
});

describe("errors", () => {
  it("carries the server's status, code and message", async () => {
    const { veld } = client(() => json(401, { code: "INVALID_API_KEY", message: "Invalid API key" }));
    const error = await veld.recall({ query: "q" }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(VeldError);
    expect(error).toMatchObject({ status: 401, code: "INVALID_API_KEY" });
    expect((error as Error).message).toBe(
      "veld POST /api/recall returned 401 INVALID_API_KEY: Invalid API key",
    );
  });

  it("maps a plain-text error body", async () => {
    const { veld } = client(() => new Response("slow down", { status: 429 }));
    await expect(veld.remember({ content: "x" })).rejects.toMatchObject({
      status: 429,
      code: "HTTP_ERROR",
      message: "veld POST /api/remember returned 429 HTTP_ERROR: slow down",
    });
  });

  it("maps a success reply that is not JSON", async () => {
    const { veld } = client(() => new Response("<html>", { status: 200 }));
    await expect(veld.recall({ query: "q" })).rejects.toMatchObject({
      status: 200,
      code: "INVALID_RESPONSE",
    });
  });

  it("maps an unreachable server", async () => {
    const { veld } = client(() => {
      throw new TypeError("fetch failed");
    });
    await expect(veld.recall({ query: "q" })).rejects.toMatchObject({
      status: undefined,
      code: "CONNECTION_FAILED",
    });
  });

  it("times out a server that does not answer", async () => {
    const { veld } = client(
      (call) =>
        new Promise<Response>((_, reject) => {
          call.signal?.addEventListener("abort", () => reject(call.signal?.reason));
        }),
      { timeoutMs: 20 },
    );
    await expect(veld.recall({ query: "q" })).rejects.toMatchObject({ code: "TIMEOUT" });
  });

  it("passes a caller's abort through unchanged", async () => {
    const controller = new AbortController();
    const { veld } = client(
      (call) =>
        new Promise<Response>((_, reject) => {
          call.signal?.addEventListener("abort", () => reject(call.signal?.reason));
        }),
    );
    const pending = veld.recall({ query: "q" }, controller.signal);
    controller.abort();
    const error = await pending.catch((e: unknown) => e);
    expect(error).not.toBeInstanceOf(VeldError);
    expect((error as Error).name).toBe("AbortError");
  });
});
