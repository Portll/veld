//! Read-only discovery of the local environment the Manager surfaces:
//! GitHub accounts (from gh's `hosts.yml`), repositories + worktrees under
//! `~/Repositories`, their org/boundary classification, and Veld server health.
//! Every function is side-effect free — filesystem reads, `git` shell-outs, and
//! a GET to the Veld `/health` endpoint.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

/// Match the Claude Code hooks' `echo "$dir" | md5 | cut -c1-12`: md5 of the
/// path PLUS the trailing newline `echo` appends, truncated to 12 hex chars.
pub fn project_hash(path: &str) -> String {
    let digest = md5::compute(format!("{path}\n").as_bytes());
    format!("{digest:x}").chars().take(12).collect()
}

// ── GitHub accounts (gh hosts.yml) ───────────────────────────────────────────

/// A GitHub account/user known to the gh CLI.
#[derive(Debug, Serialize)]
pub struct GhAccount {
    pub login: String,
    pub is_default: bool,
}

/// Parse `~/.config/gh/hosts.yml` for the github.com users + the active user.
/// The file's shape is shallow and stable, so a tiny indent-aware scan avoids a
/// (deprecated) YAML dependency.
pub fn list_gh_accounts() -> Vec<GhAccount> {
    let path = home().join(".config/gh/hosts.yml");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut logins: Vec<String> = Vec::new();
    let mut active: Option<String> = None;
    let mut in_github = false;
    let mut in_users = false;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if indent == 0 {
            in_github = t == "github.com:";
            in_users = false;
            continue;
        }
        if !in_github {
            continue;
        }
        if indent <= 4 {
            // direct children of `github.com:` — `git_protocol`, `users:`, `user:`
            in_users = t == "users:";
            if let Some(rest) = t.strip_prefix("user:") {
                active = Some(rest.trim().trim_matches('"').to_string());
            }
            continue;
        }
        // deeper than 4 → a login key under `users:` (its own children ignored)
        if in_users && indent == 8 && t.ends_with(':') {
            let login = t.trim_end_matches(':').trim().to_string();
            if !login.is_empty() {
                logins.push(login);
            }
        }
    }
    logins
        .into_iter()
        .map(|login| GhAccount {
            is_default: active.as_deref() == Some(login.as_str()),
            login,
        })
        .collect()
}

// ── Repositories ─────────────────────────────────────────────────────────────

/// A discovered repository (or worktree) and its classification.
#[derive(Debug, Serialize)]
pub struct RepoNode {
    pub path: String,
    pub canonical_path: String,
    pub project_hash: String,
    pub repo: String,
    pub org: String,
    pub account: String,
    pub boundary: String, // "internal" | "external"
    pub remote_url: Option<String>,
    pub is_symlink: bool,
    pub is_worktree: bool,
}

fn is_git_repo(dir: &Path) -> bool {
    // `.git` as a directory (normal repo) or a file (worktree gitdir pointer)
    dir.join(".git").exists()
}

fn git_output(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Parse a git remote URL into `(host, org, repo)`.
/// Handles `git@github.com:Org/Repo.git` and `https://github.com/Org/Repo.git`.
fn parse_remote(url: &str) -> Option<(String, String, String)> {
    let rest = if let Some(r) = url.strip_prefix("git@") {
        r.replacen(':', "/", 1)
    } else {
        url.trim_start_matches("https://")
            .trim_start_matches("http://")
            .to_string()
    };
    let rest = rest.trim_end_matches(".git");
    let mut parts = rest.split('/');
    let host = parts.next()?.to_string();
    let org = parts.next()?.to_string();
    let repo = parts.next()?.to_string();
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    Some((host, org, repo))
}

/// Portll = internal; everything else (Gourmet-Fire, gourmetfire, …) = external.
fn boundary_for(org: &str) -> &'static str {
    if org.eq_ignore_ascii_case("Portll") {
        "internal"
    } else {
        "external"
    }
}

/// Best-effort account owner for an org login.
fn account_for(org: &str) -> String {
    match org.to_ascii_lowercase().as_str() {
        "portll" => "Portll".to_string(),
        "gourmet-fire" | "gourmetfire" => "gourmetfire".to_string(),
        other if other.is_empty() => "unknown".to_string(),
        _ => org.to_string(),
    }
}

fn node_for(dir: &Path) -> Option<RepoNode> {
    if !is_git_repo(dir) {
        return None;
    }
    let path = dir.to_string_lossy().to_string();
    let canonical_path = std::fs::canonicalize(dir)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.clone());
    let is_symlink = std::fs::symlink_metadata(dir)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    // A worktree carries a `.git` FILE (a gitdir pointer), not a directory.
    let is_worktree = dir.join(".git").is_file();

    let remote_url = git_output(dir, &["remote", "get-url", "origin"]);
    let (org, repo) = match remote_url.as_deref().and_then(parse_remote) {
        Some((_host, org, repo)) => (org, repo),
        None => (String::new(), dir.file_name()?.to_string_lossy().to_string()),
    };

    Some(RepoNode {
        project_hash: project_hash(&path),
        boundary: boundary_for(&org).to_string(),
        account: account_for(&org),
        org,
        repo,
        remote_url,
        is_symlink,
        is_worktree,
        canonical_path,
        path,
    })
}

/// Discover repositories under `~/Repositories` and `~/Repositories/External/<org>`.
pub fn scan_repos() -> Vec<RepoNode> {
    let root = home().join("Repositories");
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            }
        }
    }
    let external = root.join("External");
    if let Ok(rd) = std::fs::read_dir(&external) {
        for org_e in rd.flatten() {
            let org_p = org_e.path();
            if !org_p.is_dir() {
                continue;
            }
            if let Ok(rd2) = std::fs::read_dir(&org_p) {
                for e in rd2.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        dirs.push(p);
                    }
                }
            }
        }
    }

    let mut nodes: Vec<RepoNode> = dirs.iter().filter_map(|d| node_for(d)).collect();
    nodes.sort_by(|a, b| (&a.org, &a.repo).cmp(&(&b.org, &b.repo)));
    nodes
}

// ── Veld server health ───────────────────────────────────────────────────────

/// Reachability + round-trip of the Veld `/health` endpoint.
#[derive(Debug, Serialize)]
pub struct ServerHealth {
    pub reachable: bool,
    pub status: Option<String>,
    pub rtt_ms: Option<u128>,
}

/// GET `{host}/health` with a short timeout. Never errors — reports reachability.
pub async fn server_health(host: &str) -> ServerHealth {
    let url = format!("{}/health", host.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            return ServerHealth {
                reachable: false,
                status: None,
                rtt_ms: None,
            }
        }
    };
    let start = std::time::Instant::now();
    match client.get(&url).send().await {
        Ok(resp) => {
            let rtt = start.elapsed().as_millis();
            let ok = resp.status().is_success();
            let status = resp
                .text()
                .await
                .ok()
                .map(|t| t.chars().take(120).collect());
            ServerHealth {
                reachable: ok,
                status,
                rtt_ms: Some(rtt),
            }
        }
        Err(_) => ServerHealth {
            reachable: false,
            status: None,
            rtt_ms: None,
        },
    }
}
