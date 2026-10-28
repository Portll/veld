import type {
  DeleteMemoryResponse,
  ErrorResponse,
  ListMemoriesRequest,
  ListResponse,
  MemoryType,
  RecallRequest,
  RecallResponse,
  RememberRequest,
  RememberResponse,
  RetrievalMode,
  SourceType,
} from "./types.js";

/** Where the client goes when neither `baseUrl` nor VELD_API_URL is set. */
export const DEFAULT_BASE_URL = "http://127.0.0.1:3030";
const DEFAULT_TIMEOUT_MS = 10_000;

export interface VeldClientOptions {
  /** Server URL. Default: VELD_API_URL, else http://127.0.0.1:3030. */
  baseUrl?: string;
  /** API key sent as X-API-Key. Default: VELD_API_KEY. Required. */
  apiKey?: string;
  /** Veld user whose memories are read and written. Default: VELD_USER_ID, else "default". */
  userId?: string;
  /** Per-request timeout in milliseconds. Default: 10000. */
  timeoutMs?: number;
  /** fetch to use instead of the global one. */
  fetch?: typeof globalThis.fetch;
}

export interface RememberInput {
  content: string;
  memoryType?: MemoryType;
  tags?: string[];
  sourceType?: SourceType;
}

export interface RecallInput {
  query: string;
  limit?: number;
  mode?: RetrievalMode;
  /** Return only memories carrying at least one of these tags. */
  tags?: string[];
}

export interface ListInput {
  limit?: number;
  offset?: number;
  /** Memory type, e.g. "Decision". */
  type?: string;
  /** Case-insensitive substring of content or tags. */
  query?: string;
  /** Return only memories carrying every one of these tags. */
  tags?: string[];
}

/**
 * A veld call that failed. `status` is the HTTP status, absent when no reply
 * arrived. `code` is the server's error code (e.g. `INVALID_API_KEY`), or one of
 * `TIMEOUT`, `CONNECTION_FAILED`, `INVALID_RESPONSE`, `NOT_STORED`, `HTTP_ERROR`.
 */
export class VeldError extends Error {
  readonly status: number | undefined;
  readonly code: string;

  constructor(message: string, code: string, status?: number) {
    super(message);
    this.name = "VeldError";
    this.code = code;
    this.status = status;
  }
}

function isLocalHost(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return ["127.0.0.1", "localhost", "[::1]", "::1", "0.0.0.0"].includes(host);
  } catch {
    return false;
  }
}

function errorFields(text: string): { code: string; message: string } {
  try {
    const body = JSON.parse(text) as Partial<ErrorResponse> & { error?: string };
    if (body && typeof body === "object") {
      return {
        code: typeof body.code === "string" ? body.code : "HTTP_ERROR",
        message: body.message ?? body.error ?? text,
      };
    }
  } catch {
    // A proxy or rate limiter in front of veld may answer in plain text.
  }
  return { code: "HTTP_ERROR", message: text };
}

/**
 * HTTP client for the veld routes the AI SDK integration needs. It talks only
 * to the configured server and sends nothing anywhere else.
 */
export class VeldClient {
  readonly baseUrl: string;
  readonly userId: string;
  readonly #apiKey: string;
  readonly #timeoutMs: number;
  readonly #fetch: typeof globalThis.fetch;

  constructor(options: VeldClientOptions = {}) {
    const env = typeof process === "undefined" ? {} : process.env;
    const apiKey = options.apiKey ?? env.VELD_API_KEY;
    if (!apiKey) {
      throw new Error("API key required. Pass apiKey or set VELD_API_KEY.");
    }
    this.#apiKey = apiKey;
    this.baseUrl = (options.baseUrl ?? env.VELD_API_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.userId = options.userId ?? env.VELD_USER_ID ?? "default";
    this.#timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);

    if (
      this.baseUrl.startsWith("http://") &&
      !isLocalHost(this.baseUrl) &&
      env.VELD_ALLOW_HTTP !== "true"
    ) {
      console.warn(
        "[veld] WARNING: Using HTTP for a non-localhost server sends the API key in clear text. " +
          "Use an https:// URL, or set VELD_ALLOW_HTTP=true to suppress this warning.",
      );
    }
  }

  /** Store a memory (`POST /api/remember`) and return its id. */
  async remember(input: RememberInput, signal?: AbortSignal): Promise<RememberResponse> {
    const body: RememberRequest = {
      user_id: this.userId,
      content: input.content,
      ...(input.memoryType && { memory_type: input.memoryType }),
      ...(input.tags && input.tags.length > 0 && { tags: input.tags }),
      ...(input.sourceType && { source_type: input.sourceType }),
    };
    const reply = await this.#request<RememberResponse>("POST", "/api/remember", body, signal);
    if (!reply || reply.success !== true || typeof reply.id !== "string" || !reply.id) {
      throw new VeldError("veld did not store the memory", "NOT_STORED");
    }
    return reply;
  }

  /** Recall memories relevant to a query (`POST /api/recall`), best first. */
  async recall(input: RecallInput, signal?: AbortSignal): Promise<RecallResponse> {
    const body: RecallRequest = {
      user_id: this.userId,
      query: input.query,
      ...(input.limit !== undefined && { limit: input.limit }),
      ...(input.mode && { mode: input.mode }),
      ...(input.tags && input.tags.length > 0 && { tags: input.tags }),
    };
    const reply = await this.#request<RecallResponse>("POST", "/api/recall", body, signal);
    if (!reply || !Array.isArray(reply.memories)) {
      throw new VeldError("veld returned a recall reply without a memories list", "INVALID_RESPONSE");
    }
    return reply;
  }

  /** List memories (`POST /api/memories`); `total` counts every match. */
  async list(input: ListInput = {}, signal?: AbortSignal): Promise<ListResponse> {
    const body: ListMemoriesRequest = {
      user_id: this.userId,
      ...(input.limit !== undefined && { limit: input.limit }),
      ...(input.offset !== undefined && { offset: input.offset }),
      ...(input.type && { type: input.type }),
      ...(input.query && { query: input.query }),
      ...(input.tags && input.tags.length > 0 && { tags: input.tags.join(",") }),
    };
    const reply = await this.#request<ListResponse>("POST", "/api/memories", body, signal);
    if (!reply || !Array.isArray(reply.memories)) {
      throw new VeldError("veld returned a list reply without a memories list", "INVALID_RESPONSE");
    }
    return reply;
  }

  /** Delete one memory (`DELETE /api/memory/{id}`). */
  async forget(memoryId: string, signal?: AbortSignal): Promise<DeleteMemoryResponse> {
    const path =
      `/api/memory/${encodeURIComponent(memoryId)}` +
      `?user_id=${encodeURIComponent(this.userId)}`;
    return this.#request<DeleteMemoryResponse>("DELETE", path, undefined, signal);
  }

  async #request<T>(
    method: string,
    path: string,
    body: object | undefined,
    signal: AbortSignal | undefined,
  ): Promise<T> {
    const timeout = AbortSignal.timeout(this.#timeoutMs);
    const route = `${method} ${path.split("?")[0]}`;
    let response: Response;
    try {
      response = await this.#fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { "Content-Type": "application/json", "X-API-Key": this.#apiKey },
        ...(body !== undefined && { body: JSON.stringify(body) }),
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      if (timeout.aborted) {
        throw new VeldError(`veld ${route} timed out after ${this.#timeoutMs} ms`, "TIMEOUT");
      }
      const reason = error instanceof Error ? error.message : String(error);
      throw new VeldError(`veld ${route} failed: ${reason}`, "CONNECTION_FAILED");
    }

    const text = await response.text();
    if (!response.ok) {
      const { code, message } = errorFields(text);
      throw new VeldError(
        `veld ${route} returned ${response.status} ${code}: ${message}`,
        code,
        response.status,
      );
    }
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new VeldError(
        `veld ${route} returned a body that is not JSON`,
        "INVALID_RESPONSE",
        response.status,
      );
    }
  }
}
