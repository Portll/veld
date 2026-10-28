/**
 * Wire shapes of the veld HTTP routes this package calls, as stated in the
 * generated schema reference (docs/src/reference/schemas.md, built from
 * src/handlers). Requests list only the fields this package sends.
 */

/** Memory types `POST /api/remember` accepts; the server matches them case-insensitively. */
export const MEMORY_TYPES = [
  "Observation",
  "Decision",
  "Learning",
  "Error",
  "Discovery",
  "Pattern",
  "Context",
  "Task",
  "Conversation",
] as const;

export type MemoryType = (typeof MEMORY_TYPES)[number];

/** `RecallRequest.mode`. */
export type RetrievalMode = "semantic" | "associative" | "hybrid";

/**
 * Provenance a client may declare. Trusted provenance (user, file, system) is
 * established by the server and refused when a payload asserts it.
 */
export type SourceType = "api" | "external_api" | "web" | "ai_generated" | "inferred";

/** `POST /api/remember` body (`RememberRequest`). */
export interface RememberRequest {
  user_id: string;
  content: string;
  memory_type?: MemoryType;
  tags?: string[];
  source_type?: SourceType;
}

/** `POST /api/remember` reply (`RememberResponse`). */
export interface RememberResponse {
  id: string;
  success: boolean;
}

/** `POST /api/recall` body (`RecallRequest`). */
export interface RecallRequest {
  user_id: string;
  query: string;
  limit?: number;
  mode?: RetrievalMode;
  /** Any-match tag filter. */
  tags?: string[];
}

/** `RecallExperience`. `tags` holds the caller's tags and the entities veld extracted. */
export interface RecallExperience {
  content: string;
  memory_type?: string | null;
  tags: string[];
}

/** `RecallMemory`. Optional fields are omitted when empty. */
export interface RecallMemory {
  id: string;
  experience: RecallExperience;
  importance: number;
  created_at: string;
  score: number;
  tier: string;
  source_type?: string;
  source_trusted?: boolean;
  /** True when untrusted content matches prompt-injection directives. */
  injection_suspected?: boolean;
}

/** `POST /api/recall` reply (`RecallResponse`), the fields this package reads. */
export interface RecallResponse {
  memories: RecallMemory[];
  count: number;
  tokens_estimated?: number;
}

/** `POST /api/memories` body (`ListMemoriesRequest`). */
export interface ListMemoriesRequest {
  user_id: string;
  limit?: number;
  offset?: number;
  type?: string;
  query?: string;
  /** Comma-separated; every tag must be present, matched exactly. */
  tags?: string;
}

/** `ListMemoryItem`. `content` is a preview when `content_truncated` is true. */
export interface ListMemoryItem {
  id: string;
  content: string;
  content_truncated: boolean;
  content_length: number;
  memory_type: string;
  importance: number;
  tags: string[];
  created_at: string;
  tier: string;
}

/** `POST /api/memories` reply (`ListResponse`). */
export interface ListResponse {
  memories: ListMemoryItem[];
  /** Matches before `limit`/`offset` were applied. */
  total: number;
}

/** `DELETE /api/memory/{memory_id}` reply (`DeleteMemoryResponse`). */
export interface DeleteMemoryResponse {
  success: boolean;
  id: string;
  message: string;
}

/** Error body every route returns (`ErrorResponse`). */
export interface ErrorResponse {
  code: string;
  message: string;
  details?: string;
  request_id?: string;
}
