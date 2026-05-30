//! Multi-format text extraction pipeline
//!
//! Converts various input formats (Markdown, JSON, CSV, code, HTML) into
//! plain text suitable for memory storage. Extracts structural metadata
//! (headings, entity hints, word/line counts) to enrich downstream
//! embedding and graph processing.
//!
//! No external crates — parsing uses only stdlib + `regex`.

pub mod extractors;
pub mod gdrive;
pub mod github;
pub mod tabular;

use anyhow::Result;

/// Supported input formats for text extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    PlainText,
    Markdown,
    Json,
    Csv,
    Code,
    Html,
    Pdf,
    Unknown,
}

impl InputFormat {
    /// Machine-readable format label for metadata tagging.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PlainText => "plaintext",
            Self::Markdown => "markdown",
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Code => "code",
            Self::Html => "html",
            Self::Pdf => "pdf",
            Self::Unknown => "unknown",
        }
    }
}

/// Detect format from an optional filename extension, falling back to
/// content sniffing when no extension is available or recognized.
pub fn detect_format(filename: Option<&str>, content: &str) -> InputFormat {
    // Extension-based detection (highest priority)
    if let Some(name) = filename {
        if let Some(ext) = name.rsplit('.').next() {
            match ext.to_ascii_lowercase().as_str() {
                "md" | "markdown" => return InputFormat::Markdown,
                "json" | "jsonl" => return InputFormat::Json,
                "csv" | "tsv" => return InputFormat::Csv,
                "html" | "htm" => return InputFormat::Html,
                "pdf" => return InputFormat::Pdf,
                "txt" | "log" => return InputFormat::PlainText,
                "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp"
                | "cc" | "h" | "hpp" | "rb" | "sh" | "bash" | "zsh" | "sql" | "toml" | "yaml"
                | "yml" | "swift" | "kt" | "scala" | "lua" | "r" | "pl" | "ex" | "exs"
                | "zig" | "nim" | "v" | "d" | "cs" | "fs" | "ml" | "hs" | "erl" | "elm"
                | "clj" | "lisp" | "proto" | "graphql" | "tf" | "hcl" => {
                    return InputFormat::Code;
                }
                _ => {}
            }
        }
    }

    // Content sniffing fallback

    // PDF magic bytes (binary content — check before trimming)
    if content.starts_with("%PDF-") {
        return InputFormat::Pdf;
    }

    let trimmed = content.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        // Quick JSON plausibility check: must close with matching bracket
        let last_non_ws = content.trim_end().as_bytes().last().copied();
        if (trimmed.starts_with('{') && last_non_ws == Some(b'}'))
            || (trimmed.starts_with('[') && last_non_ws == Some(b']'))
        {
            return InputFormat::Json;
        }
    }
    if trimmed.starts_with("<!") || trimmed.starts_with("<html") || trimmed.starts_with("<HTML") {
        return InputFormat::Html;
    }
    // Markdown heuristic: presence of ATX headings
    if trimmed.starts_with("# ")
        || trimmed.contains("\n# ")
        || trimmed.contains("\n## ")
        || trimmed.contains("\n### ")
    {
        return InputFormat::Markdown;
    }

    InputFormat::PlainText
}

/// Extract clean text and metadata from content in the given format.
pub fn extract_text(content: &str, format: InputFormat) -> Result<ExtractedContent> {
    extractors::extract(content, format)
}

/// Result of text extraction: cleaned text plus structural metadata.
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// The extracted plain text, ready for embedding/storage.
    pub text: String,
    /// Structural metadata harvested during extraction.
    pub metadata: ContentMetadata,
}

/// Metadata harvested from the input document during extraction.
#[derive(Debug, Clone, Default)]
pub struct ContentMetadata {
    /// Detected format label (e.g. "markdown", "json").
    pub format: String,
    /// Document title, if one could be inferred.
    pub title: Option<String>,
    /// Headings found in the document (Markdown `#`, HTML `<h1>`–`<h6>`).
    pub headings: Vec<String>,
    /// Entity hints extracted from the structure (JSON keys, CSV columns,
    /// function/class names in code).
    pub entities_hint: Vec<String>,
    /// Total line count of extracted text.
    pub line_count: usize,
    /// Total word count of extracted text.
    pub word_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_format ────────────────────────────────────────────────────

    #[test]
    fn detect_by_extension() {
        assert_eq!(
            detect_format(Some("notes.md"), "anything"),
            InputFormat::Markdown
        );
        assert_eq!(
            detect_format(Some("data.json"), "anything"),
            InputFormat::Json
        );
        assert_eq!(
            detect_format(Some("sheet.csv"), "anything"),
            InputFormat::Csv
        );
        assert_eq!(
            detect_format(Some("page.html"), "anything"),
            InputFormat::Html
        );
        assert_eq!(
            detect_format(Some("main.rs"), "anything"),
            InputFormat::Code
        );
        assert_eq!(
            detect_format(Some("app.py"), "anything"),
            InputFormat::Code
        );
        assert_eq!(
            detect_format(Some("readme.txt"), "anything"),
            InputFormat::PlainText
        );
    }

    #[test]
    fn detect_by_content_sniffing() {
        assert_eq!(
            detect_format(None, r#"{"key": "value"}"#),
            InputFormat::Json
        );
        assert_eq!(
            detect_format(None, r#"[{"a":1}]"#),
            InputFormat::Json
        );
        assert_eq!(
            detect_format(None, "<!DOCTYPE html><html></html>"),
            InputFormat::Html
        );
        assert_eq!(
            detect_format(None, "# My Title\nSome text"),
            InputFormat::Markdown
        );
        assert_eq!(
            detect_format(None, "Just plain text here."),
            InputFormat::PlainText
        );
    }

    #[test]
    fn detect_unknown_extension_falls_through() {
        // .xyz is unknown, content is plain text
        assert_eq!(
            detect_format(Some("file.xyz"), "hello world"),
            InputFormat::PlainText
        );
    }

    #[test]
    fn format_as_str() {
        assert_eq!(InputFormat::Markdown.as_str(), "markdown");
        assert_eq!(InputFormat::Json.as_str(), "json");
        assert_eq!(InputFormat::Code.as_str(), "code");
    }

    // ── extract_text smoke ───────────────────────────────────────────────

    #[test]
    fn extract_plaintext_passthrough() {
        let content = "Hello world.\nSecond line.";
        let result = extract_text(content, InputFormat::PlainText).unwrap();
        assert_eq!(result.text, content);
        assert_eq!(result.metadata.line_count, 2);
        assert_eq!(result.metadata.word_count, 4);
        assert_eq!(result.metadata.format, "plaintext");
    }
}
