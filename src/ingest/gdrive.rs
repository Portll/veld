//! Google Drive connector — fetches file content via the Drive v3 API.
//!
//! # Auth
//!
//! Bearer-token only. The caller obtains an OAuth2 access token (or service-
//! account token) out-of-band and passes it to [`DriveClient::new`]. Veld
//! does not run the OAuth dance — that's the caller's concern (CLI helper,
//! external service, etc.). This keeps the connector dependency-free and
//! avoids storing refresh tokens in the memory store.
//!
//! # Scope
//!
//! Minimal viable surface:
//!   - `fetch_text_file(file_id)` — download a Google Doc / plain text /
//!     markdown file as text. Google Docs are exported as plain text;
//!     non-Google files use the raw `alt=media` download.
//!   - `fetch_metadata(file_id)` — file name + mime type, used by callers
//!     to decide ingest format.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::time::Duration;

const DRIVE_API: &str = "https://www.googleapis.com/drive/v3";

/// Content fetched from Google Drive, ready to feed into the ingest extractor.
#[derive(Debug, Clone)]
pub struct DriveContent {
    /// File ID
    pub id: String,
    /// File name (used as title + entity hint)
    pub name: String,
    /// Plain-text body
    pub body: String,
    /// Suggested input format for the ingest extractor
    pub format_hint: &'static str,
}

/// Blocking Drive client.
pub struct DriveClient {
    token: String,
    client: reqwest::blocking::Client,
}

impl DriveClient {
    /// Construct with an OAuth2 access token (must already be exchanged —
    /// no refresh logic here).
    pub fn new(token: impl Into<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("veld-ingest/1.0")
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            token: token.into(),
            client,
        })
    }

    /// Fetch file metadata (name + mimeType).
    pub fn fetch_metadata(&self, file_id: &str) -> Result<DriveMetadata> {
        let url = format!("{DRIVE_API}/files/{file_id}?fields=id,name,mimeType");
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "GDrive metadata {} → {}",
                file_id,
                resp.status().as_u16()
            ));
        }
        Ok(resp.json()?)
    }

    /// Download file content as text. Google-native types
    /// (`application/vnd.google-apps.*`) are exported as `text/plain`; other
    /// files use raw `alt=media` download.
    pub fn fetch_text_file(&self, file_id: &str) -> Result<DriveContent> {
        let meta = self.fetch_metadata(file_id)?;

        let url = if meta.mime_type.starts_with("application/vnd.google-apps.") {
            format!("{DRIVE_API}/files/{file_id}/export?mimeType=text/plain")
        } else {
            format!("{DRIVE_API}/files/{file_id}?alt=media")
        };

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "GDrive content {} → {}",
                file_id,
                resp.status().as_u16()
            ));
        }
        let body = resp.text()?;
        let format_hint = format_from_mime(&meta.mime_type, &meta.name);

        Ok(DriveContent {
            id: meta.id,
            name: meta.name,
            body,
            format_hint,
        })
    }
}

fn format_from_mime(mime: &str, name: &str) -> &'static str {
    let mime_lower = mime.to_ascii_lowercase();
    if mime_lower.contains("markdown") || name.to_ascii_lowercase().ends_with(".md") {
        "markdown"
    } else if mime_lower.contains("json") {
        "json"
    } else if mime_lower.contains("csv") {
        "csv"
    } else if mime_lower.contains("html") {
        "html"
    } else if mime_lower.contains("pdf") {
        "pdf"
    } else {
        "plaintext"
    }
}

/// File metadata response.
#[derive(Debug, Clone, Deserialize)]
pub struct DriveMetadata {
    pub id: String,
    pub name: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_mime_dispatch() {
        assert_eq!(format_from_mime("text/markdown", "notes.md"), "markdown");
        assert_eq!(
            format_from_mime("text/plain", "README.md"),
            "markdown",
            "filename extension wins when mime is generic"
        );
        assert_eq!(format_from_mime("application/json", "x.json"), "json");
        assert_eq!(format_from_mime("text/csv", "rows.csv"), "csv");
        assert_eq!(format_from_mime("application/pdf", "x.pdf"), "pdf");
        assert_eq!(format_from_mime("text/plain", "notes.txt"), "plaintext");
    }

    #[test]
    fn client_constructs_with_empty_token() {
        // No network call — just validates the builder path
        assert!(DriveClient::new("").is_ok());
    }
}
