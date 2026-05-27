//! GitHub connector — fetches issues / PRs / repo files via the REST API.
//!
//! # Auth
//!
//! Personal access tokens only. No OAuth dance — the user passes a token
//! at construction time (Veld stays local-first; remote auth is the
//! caller's problem).
//!
//! # Scope
//!
//! Minimal viable surface: issue body, pull request body + diff title list,
//! single repo file, and repo README. Each returns an [`IngestPayload`]
//! that callers feed into the existing ingest pipeline via
//! [`crate::ingest::extractors::extract`].

use anyhow::{anyhow, Result};
use base64::Engine as _;
use serde::Deserialize;

const GITHUB_API: &str = "https://api.github.com";
const USER_AGENT: &str = "veld-ingest/1.0";

/// Content fetched from GitHub, ready to feed into the ingest extractor.
#[derive(Debug, Clone)]
pub struct GithubContent {
    /// The canonical URL of the resource (issue link, PR link, blob link)
    pub url: String,
    /// Title or path for display + entity hints
    pub title: String,
    /// Plain-text body to ingest
    pub body: String,
    /// Suggested input format for the ingest extractor
    /// (`"markdown"` for issues / PRs / README, `"code"` for non-markdown
    /// file blobs).
    pub format_hint: &'static str,
}

/// Blocking GitHub client. Holds the personal access token and HTTP client.
pub struct GithubClient {
    token: String,
    client: reqwest::blocking::Client,
}

impl GithubClient {
    /// Construct a new client. `token` must be a GitHub personal access
    /// token (classic or fine-grained) with read access to the target
    /// resources. Passing an empty token is allowed but rate-limits will
    /// be tight and private repos will 404.
    pub fn new(token: impl Into<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(std::time::Duration::from_secs(20))
            .build()?;
        Ok(Self {
            token: token.into(),
            client,
        })
    }

    fn get_json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        let url = format!("{GITHUB_API}{path}");
        let mut req = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28");
        if !self.token.is_empty() {
            req = req.bearer_auth(&self.token);
        }
        let resp = req.send()?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "GitHub API {} → {}",
                url,
                resp.status().as_u16()
            ));
        }
        Ok(resp.json::<T>()?)
    }

    /// Fetch a single issue's body + metadata.
    pub fn fetch_issue(&self, owner: &str, repo: &str, number: u64) -> Result<GithubContent> {
        let path = format!("/repos/{owner}/{repo}/issues/{number}");
        let issue: IssueOrPr = self.get_json(&path)?;
        let body = issue.body.unwrap_or_default();
        let mut combined = format!("# {}\n\n{}", issue.title, body);
        if !issue.labels.is_empty() {
            let labels: Vec<String> =
                issue.labels.iter().map(|l| l.name.clone()).collect();
            combined.push_str(&format!("\n\nLabels: {}", labels.join(", ")));
        }
        Ok(GithubContent {
            url: issue.html_url,
            title: issue.title,
            body: combined,
            format_hint: "markdown",
        })
    }

    /// Fetch a pull request's body, title, and changed-file list.
    pub fn fetch_pull_request(
        &self,
        owner: &str,
        repo: &str,
        number: u64,
    ) -> Result<GithubContent> {
        // Same shape as issue body
        let issue_path = format!("/repos/{owner}/{repo}/pulls/{number}");
        let pr: IssueOrPr = self.get_json(&issue_path)?;

        let files_path = format!("/repos/{owner}/{repo}/pulls/{number}/files");
        let files: Vec<PrFile> = self.get_json(&files_path).unwrap_or_default();

        let mut combined = format!("# {}\n\n{}", pr.title, pr.body.unwrap_or_default());
        if !files.is_empty() {
            combined.push_str("\n\n## Changed Files\n");
            for f in &files {
                combined.push_str(&format!(
                    "- `{}` (+{}/-{})\n",
                    f.filename, f.additions, f.deletions
                ));
            }
        }
        Ok(GithubContent {
            url: pr.html_url,
            title: pr.title,
            body: combined,
            format_hint: "markdown",
        })
    }

    /// Fetch a single file from a repo at a ref (branch / tag / sha).
    /// If `ref_` is `None`, the repo's default branch is used.
    pub fn fetch_file(
        &self,
        owner: &str,
        repo: &str,
        path: &str,
        ref_: Option<&str>,
    ) -> Result<GithubContent> {
        let suffix = ref_.map(|r| format!("?ref={r}")).unwrap_or_default();
        let api_path = format!("/repos/{owner}/{repo}/contents/{path}{suffix}");
        let content: FileContent = self.get_json(&api_path)?;
        let decoded = decode_b64(&content.content)?;
        let format_hint = format_from_filename(path);
        Ok(GithubContent {
            url: content.html_url,
            title: path.to_string(),
            body: decoded,
            format_hint,
        })
    }

    /// Fetch the repo's README at the default branch.
    pub fn fetch_readme(&self, owner: &str, repo: &str) -> Result<GithubContent> {
        let path = format!("/repos/{owner}/{repo}/readme");
        let content: FileContent = self.get_json(&path)?;
        let decoded = decode_b64(&content.content)?;
        Ok(GithubContent {
            url: content.html_url,
            title: format!("{owner}/{repo} README"),
            body: decoded,
            format_hint: "markdown",
        })
    }
}

fn decode_b64(s: &str) -> Result<String> {
    // GitHub returns base64 with embedded newlines
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&clean)
        .map_err(|e| anyhow!("GitHub returned invalid base64: {e}"))?;
    String::from_utf8(bytes).map_err(|e| anyhow!("GitHub file is not valid UTF-8: {e}"))
}

fn format_from_filename(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".md") || lower.ends_with(".markdown") {
        "markdown"
    } else if lower.ends_with(".json") {
        "json"
    } else if lower.ends_with(".csv") {
        "csv"
    } else if lower.ends_with(".html") || lower.ends_with(".htm") {
        "html"
    } else if lower.ends_with(".rs")
        || lower.ends_with(".py")
        || lower.ends_with(".ts")
        || lower.ends_with(".js")
        || lower.ends_with(".go")
        || lower.ends_with(".java")
        || lower.ends_with(".c")
        || lower.ends_with(".cpp")
        || lower.ends_with(".h")
    {
        "code"
    } else {
        "plaintext"
    }
}

// =============================================================================
// API response types — minimal subset of fields actually used
// =============================================================================

#[derive(Deserialize)]
struct IssueOrPr {
    title: String,
    body: Option<String>,
    html_url: String,
    #[serde(default)]
    labels: Vec<Label>,
}

#[derive(Deserialize)]
struct Label {
    name: String,
}

#[derive(Deserialize)]
struct FileContent {
    content: String,
    html_url: String,
}

#[derive(Deserialize)]
struct PrFile {
    filename: String,
    #[serde(default)]
    additions: u64,
    #[serde(default)]
    deletions: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_filename_dispatch() {
        assert_eq!(format_from_filename("README.md"), "markdown");
        assert_eq!(format_from_filename("config.json"), "json");
        assert_eq!(format_from_filename("data.csv"), "csv");
        assert_eq!(format_from_filename("page.html"), "html");
        assert_eq!(format_from_filename("lib.rs"), "code");
        assert_eq!(format_from_filename("LICENSE"), "plaintext");
    }

    #[test]
    fn decode_b64_handles_newlines() {
        // GitHub formats base64 with line breaks every 60 chars
        let input = "SGVs\nbG8s\nIHdv\ncmxk"; // "Hello, world"
        let out = decode_b64(input).unwrap();
        assert_eq!(out, "Hello, world");
    }

    #[test]
    fn client_constructs_with_empty_token() {
        // Validates the builder path; no network call.
        assert!(GithubClient::new("").is_ok());
    }
}
