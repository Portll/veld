// Type mirrors of the Rust IPC command return shapes, for contract safety.
// Grows per build phase alongside the Rust commands.

export interface AppInfo {
  name: string;
  version: string;
}

export type ViewName = "tree" | "workspaces" | "scope" | "connections" | "recall";

export interface GhAccount {
  login: string;
  is_default: boolean;
}

export interface RepoNode {
  path: string;
  canonical_path: string;
  project_hash: string;
  repo: string;
  org: string;
  account: string;
  boundary: "internal" | "external";
  remote_url: string | null;
  is_symlink: boolean;
  is_worktree: boolean;
}

export interface ServerHealth {
  reachable: boolean;
  status: string | null;
  rtt_ms: number | null;
}
