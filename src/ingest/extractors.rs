//! Format-specific text extraction implementations
//!
//! Each extractor converts a specific format into clean plain text and
//! populates [`ContentMetadata`] with structural hints (headings, entity
//! names, column headers, etc.).
//!
//! All parsing is done with stdlib + `regex` — no external crates.

use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;

use super::{ContentMetadata, ExtractedContent, InputFormat};

// =============================================================================
// DISPATCH
// =============================================================================

/// Dispatch extraction to the appropriate format handler.
pub fn extract(content: &str, format: InputFormat) -> Result<ExtractedContent> {
    match format {
        InputFormat::PlainText | InputFormat::Unknown => extract_plaintext(content),
        InputFormat::Markdown => extract_markdown(content),
        InputFormat::Json => extract_json(content),
        InputFormat::Csv => extract_csv(content),
        InputFormat::Code => extract_code(content),
        InputFormat::Html => extract_html(content),
    }
}

// =============================================================================
// SHARED HELPERS
// =============================================================================

fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.lines().count()
}

/// Deduplicate a list of strings (case-insensitive), preserving first occurrence.
fn dedup_case_insensitive(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            if seen.contains(&lower) {
                false
            } else {
                seen.insert(lower);
                true
            }
        })
        .collect()
}

// =============================================================================
// PLAIN TEXT
// =============================================================================

fn extract_plaintext(content: &str) -> Result<ExtractedContent> {
    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::PlainText.as_str().to_string(),
            line_count: count_lines(content),
            word_count: count_words(content),
            ..Default::default()
        },
        text: content.to_string(),
    })
}

// =============================================================================
// MARKDOWN
// =============================================================================

/// Regex patterns compiled once for Markdown extraction.
struct MarkdownPatterns {
    heading: Regex,
    bold_italic: Regex,
    bold: Regex,
    italic: Regex,
    strikethrough: Regex,
    inline_code: Regex,
    link: Regex,
    image: Regex,
    footnote_ref: Regex,
    html_tag: Regex,
}

fn md_patterns() -> &'static MarkdownPatterns {
    static INSTANCE: OnceLock<MarkdownPatterns> = OnceLock::new();
    INSTANCE.get_or_init(|| MarkdownPatterns {
        heading: Regex::new(r"^(#{1,6})\s+(.+)$").unwrap(),
        bold_italic: Regex::new(r"\*\*\*(.+?)\*\*\*|___(.+?)___").unwrap(),
        bold: Regex::new(r"\*\*(.+?)\*\*|__(.+?)__").unwrap(),
        italic: Regex::new(r"\*(.+?)\*|_(.+?)_").unwrap(),
        strikethrough: Regex::new(r"~~(.+?)~~").unwrap(),
        inline_code: Regex::new(r"`([^`]+)`").unwrap(),
        link: Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap(),
        image: Regex::new(r"!\[([^\]]*)\]\([^\)]+\)").unwrap(),
        footnote_ref: Regex::new(r"\[\^[^\]]+\]").unwrap(),
        html_tag: Regex::new(r"<[^>]+>").unwrap(),
    })
}

fn extract_markdown(content: &str) -> Result<ExtractedContent> {
    let pat = md_patterns();
    let mut output_lines: Vec<String> = Vec::new();
    let mut headings: Vec<String> = Vec::new();
    let mut title: Option<String> = None;
    let mut in_code_block = false;

    for line in content.lines() {
        // Track fenced code blocks
        if line.trim_start().starts_with("```") || line.trim_start().starts_with("~~~") {
            in_code_block = !in_code_block;
            // Omit the fence line itself, but preserve block content
            continue;
        }

        if in_code_block {
            // Preserve code block content as-is
            output_lines.push(line.to_string());
            continue;
        }

        // Headings
        if let Some(caps) = pat.heading.captures(line) {
            let level = caps.get(1).map_or(0, |m| m.as_str().len());
            let text = caps.get(2).map_or("", |m| m.as_str()).trim().to_string();
            if level == 1 && title.is_none() {
                title = Some(text.clone());
            }
            headings.push(text.clone());
            output_lines.push(text);
            continue;
        }

        // Horizontal rules
        let trimmed = line.trim();
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            continue;
        }

        // Strip Markdown table separators
        if trimmed.starts_with('|') && trimmed.contains("---") {
            continue;
        }

        // Process inline Markdown for everything else
        let processed = strip_markdown_inline(line, pat);
        output_lines.push(processed);
    }

    let text = output_lines.join("\n");
    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::Markdown.as_str().to_string(),
            title,
            headings,
            line_count: count_lines(&text),
            word_count: count_words(&text),
            ..Default::default()
        },
        text,
    })
}

/// Strip inline Markdown formatting from a single line.
fn strip_markdown_inline(line: &str, pat: &MarkdownPatterns) -> String {
    let mut s = line.to_string();

    // Images before links (images are `![alt](url)`, links are `[text](url)`)
    s = pat
        .image
        .replace_all(&s, |caps: &regex::Captures| {
            caps.get(1).map_or("", |m| m.as_str()).to_string()
        })
        .to_string();

    // Links: keep link text
    s = pat
        .link
        .replace_all(&s, |caps: &regex::Captures| {
            caps.get(1).map_or("", |m| m.as_str()).to_string()
        })
        .to_string();

    // Bold+italic → text
    s = pat
        .bold_italic
        .replace_all(&s, |caps: &regex::Captures| {
            caps.get(1)
                .or(caps.get(2))
                .map_or("", |m| m.as_str())
                .to_string()
        })
        .to_string();

    // Bold → text
    s = pat
        .bold
        .replace_all(&s, |caps: &regex::Captures| {
            caps.get(1)
                .or(caps.get(2))
                .map_or("", |m| m.as_str())
                .to_string()
        })
        .to_string();

    // Italic → text
    s = pat
        .italic
        .replace_all(&s, |caps: &regex::Captures| {
            caps.get(1)
                .or(caps.get(2))
                .map_or("", |m| m.as_str())
                .to_string()
        })
        .to_string();

    // Strikethrough → text
    s = pat
        .strikethrough
        .replace_all(&s, "$1")
        .to_string();

    // Inline code → preserve content
    s = pat.inline_code.replace_all(&s, "$1").to_string();

    // Footnote references
    s = pat.footnote_ref.replace_all(&s, "").to_string();

    // Strip any remaining inline HTML
    s = pat.html_tag.replace_all(&s, "").to_string();

    // Strip blockquote markers
    if s.trim_start().starts_with('>') {
        s = s.trim_start().trim_start_matches('>').trim_start().to_string();
    }

    // Strip list markers (unordered: - * +, ordered: 1. 2.)
    let list_trimmed = s.trim_start();
    if list_trimmed.starts_with("- ")
        || list_trimmed.starts_with("* ")
        || list_trimmed.starts_with("+ ")
    {
        s = list_trimmed[2..].to_string();
    } else if let Some(rest) = try_strip_ordered_list(list_trimmed) {
        s = rest;
    }

    s
}

/// Attempt to strip an ordered list marker (e.g. "1. ") and return the remainder.
fn try_strip_ordered_list(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    // Consume digits
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        return None;
    }
    // Expect '. '
    if bytes[i] == b'.' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
        Some(s[i + 2..].to_string())
    } else {
        None
    }
}

// =============================================================================
// JSON
// =============================================================================

fn extract_json(content: &str) -> Result<ExtractedContent> {
    let value: serde_json::Value = serde_json::from_str(content)?;
    let mut lines: Vec<String> = Vec::new();
    let mut entities: Vec<String> = Vec::new();

    flatten_json(&value, &mut String::new(), &mut lines, &mut entities, 0);

    let entities = dedup_case_insensitive(entities);

    let text = lines.join("\n");
    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::Json.as_str().to_string(),
            entities_hint: entities,
            line_count: count_lines(&text),
            word_count: count_words(&text),
            ..Default::default()
        },
        text,
    })
}

/// Recursively flatten a JSON value into "key.path: value" lines.
/// Collects top-level and second-level keys as entity hints.
fn flatten_json(
    value: &serde_json::Value,
    prefix: &mut String,
    lines: &mut Vec<String>,
    entities: &mut Vec<String>,
    depth: usize,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                // Collect keys at depth 0 and 1 as entity hints
                if depth <= 1 {
                    entities.push(key.clone());
                }

                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        flatten_json(val, &mut path.clone(), lines, entities, depth + 1);
                    }
                    _ => {
                        let display = json_scalar_to_string(val);
                        lines.push(format!("{}: {}", path, display));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            // For small arrays, enumerate items. For large arrays, summarize.
            if arr.len() > 100 {
                lines.push(format!(
                    "{}: [array of {} items]",
                    if prefix.is_empty() {
                        "root"
                    } else {
                        prefix.as_str()
                    },
                    arr.len()
                ));
                // Still process a sample (first 5) for entity extraction
                for (i, item) in arr.iter().take(5).enumerate() {
                    let idx_path = if prefix.is_empty() {
                        format!("[{}]", i)
                    } else {
                        format!("{}[{}]", prefix, i)
                    };
                    flatten_json(item, &mut idx_path.clone(), lines, entities, depth + 1);
                }
            } else {
                for (i, item) in arr.iter().enumerate() {
                    let idx_path = if prefix.is_empty() {
                        format!("[{}]", i)
                    } else {
                        format!("{}[{}]", prefix, i)
                    };
                    flatten_json(item, &mut idx_path.clone(), lines, entities, depth + 1);
                }
            }
        }
        _ => {
            let display = json_scalar_to_string(value);
            lines.push(format!(
                "{}: {}",
                if prefix.is_empty() {
                    "value"
                } else {
                    prefix.as_str()
                },
                display
            ));
        }
    }
}

fn json_scalar_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

// =============================================================================
// CSV
// =============================================================================

fn extract_csv(content: &str) -> Result<ExtractedContent> {
    let mut lines_iter = content.lines();

    // Detect delimiter: tab or comma
    let first_line = match lines_iter.next() {
        Some(l) => l,
        None => {
            return Ok(ExtractedContent {
                text: String::new(),
                metadata: ContentMetadata {
                    format: InputFormat::Csv.as_str().to_string(),
                    ..Default::default()
                },
            });
        }
    };

    let delimiter = if first_line.contains('\t') {
        '\t'
    } else {
        ','
    };

    let headers: Vec<String> = parse_csv_row(first_line, delimiter);
    let mut output_lines: Vec<String> = Vec::new();
    let mut row_count: usize = 0;

    for line in lines_iter {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        row_count += 1;

        if row_count > 100 {
            // Summarize: stop adding individual rows
            continue;
        }

        let fields = parse_csv_row(line, delimiter);
        let mut parts: Vec<String> = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            let col = headers.get(i).map(|h| h.as_str()).unwrap_or("?");
            parts.push(format!("{}: {}", col, field));
        }
        output_lines.push(parts.join(", "));
    }

    if row_count > 100 {
        output_lines.push(format!("... ({} total rows, showing first 100)", row_count));
    }

    let entities = dedup_case_insensitive(headers.clone());
    let text = output_lines.join("\n");

    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::Csv.as_str().to_string(),
            entities_hint: entities,
            line_count: count_lines(&text),
            word_count: count_words(&text),
            ..Default::default()
        },
        text,
    })
}

/// Simple CSV row parser that handles quoted fields.
fn parse_csv_row(line: &str, delimiter: char) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    // Escaped quote
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delimiter {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(ch);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// =============================================================================
// CODE
// =============================================================================

/// Regex patterns for extracting declarations from source code.
struct CodePatterns {
    declarations: Regex,
}

fn code_patterns() -> &'static CodePatterns {
    static INSTANCE: OnceLock<CodePatterns> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        // Matches common declaration keywords followed by the identifier name.
        // Covers: Rust, Python, JavaScript/TypeScript, Go, Java/C#, C/C++, Ruby,
        //         Kotlin, Swift, Scala, Elixir, Haskell, Zig, etc.
        //
        // Pattern explanation:
        //   \b(keyword)\s+       — keyword boundary + whitespace
        //   ([A-Za-z_]\w*)       — identifier (starts with letter/underscore)
        //
        // We also match `impl` blocks and trait definitions.
        CodePatterns {
            declarations: Regex::new(
                r"(?m)(?:^|\s)(?:pub\s+(?:async\s+)?(?:unsafe\s+)?|async\s+|export\s+(?:default\s+)?|abstract\s+|static\s+|final\s+|private\s+|protected\s+|internal\s+|override\s+|virtual\s+|const\s+|(?:unsafe\s+)?)?\b(fn|def|function|class|struct|interface|trait|impl|enum|type|module|mod|object|record|protocol|extension|typealias|typedef)\s+([A-Za-z_]\w*)",
            )
            .unwrap(),
        }
    })
}

fn extract_code(content: &str) -> Result<ExtractedContent> {
    let pat = code_patterns();
    let mut entities: Vec<String> = Vec::new();

    for caps in pat.declarations.captures_iter(content) {
        if let Some(name) = caps.get(2) {
            let name_str = name.as_str().to_string();
            // Skip single-character names and common test/helper names
            if name_str.len() > 1 {
                entities.push(name_str);
            }
        }
    }

    let entities = dedup_case_insensitive(entities);

    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::Code.as_str().to_string(),
            entities_hint: entities,
            line_count: count_lines(content),
            word_count: count_words(content),
            ..Default::default()
        },
        // Preserve code as-is for embedding — stripping syntax would lose meaning
        text: content.to_string(),
    })
}

// =============================================================================
// HTML
// =============================================================================

fn extract_html(content: &str) -> Result<ExtractedContent> {
    let mut output = String::with_capacity(content.len());
    let mut headings: Vec<String> = Vec::new();
    let mut title: Option<String> = None;

    // State machine for HTML tag stripping
    let mut in_tag = false;
    let mut tag_name = String::new();
    let mut capturing_tag_name = false;
    let mut is_closing_tag = false;
    let mut in_heading = false;
    let mut heading_text = String::new();
    let mut in_title = false;
    let mut title_text = String::new();
    let mut in_script_or_style = false;
    let mut in_entity = false;
    let mut entity_buf = String::new();

    for ch in content.chars() {
        if in_entity {
            entity_buf.push(ch);
            if ch == ';' {
                // Resolve common HTML entities
                let resolved = resolve_html_entity(&entity_buf);
                if in_heading {
                    heading_text.push_str(&resolved);
                } else if in_title {
                    title_text.push_str(&resolved);
                } else if !in_script_or_style {
                    output.push_str(&resolved);
                }
                in_entity = false;
                entity_buf.clear();
            } else if entity_buf.len() > 10 || ch.is_whitespace() {
                // Not a real entity, emit as-is
                let buf = std::mem::take(&mut entity_buf);
                if in_heading {
                    heading_text.push_str(&buf);
                } else if in_title {
                    title_text.push_str(&buf);
                } else if !in_script_or_style {
                    output.push_str(&buf);
                }
                in_entity = false;
            }
            continue;
        }

        if in_tag {
            if ch == '>' {
                in_tag = false;
                capturing_tag_name = false;

                let tag_lower = tag_name.to_lowercase();

                if is_closing_tag {
                    match tag_lower.as_str() {
                        "title" => {
                            in_title = false;
                            if title.is_none() {
                                let t = title_text.trim().to_string();
                                if !t.is_empty() {
                                    title = Some(t);
                                }
                            }
                            title_text.clear();
                        }
                        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                            in_heading = false;
                            let h = heading_text.trim().to_string();
                            if !h.is_empty() {
                                // Use first <h1> as title fallback when no <title> exists
                                if title.is_none() && tag_lower == "h1" {
                                    title = Some(h.clone());
                                }
                                headings.push(h);
                            }
                            heading_text.clear();
                            output.push('\n');
                        }
                        "script" | "style" => {
                            in_script_or_style = false;
                        }
                        _ => {}
                    }
                } else {
                    match tag_lower.as_str() {
                        "title" => {
                            in_title = true;
                            title_text.clear();
                        }
                        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                            in_heading = true;
                            heading_text.clear();
                            output.push('\n');
                        }
                        "script" | "style" => {
                            in_script_or_style = true;
                        }
                        "br" | "br/" => {
                            output.push('\n');
                        }
                        "p" | "div" | "li" | "tr" | "blockquote" | "section" | "article"
                        | "header" | "footer" | "nav" | "main" | "aside" => {
                            // Add newline before block elements
                            if !output.ends_with('\n') {
                                output.push('\n');
                            }
                        }
                        _ => {}
                    }
                }

                tag_name.clear();
            } else if ch == '/' && capturing_tag_name && tag_name.is_empty() {
                // Slash immediately after '<': this is a closing tag (e.g. </h1>)
                is_closing_tag = true;
            } else if capturing_tag_name {
                if ch.is_whitespace() || ch == '/' {
                    capturing_tag_name = false;
                } else {
                    tag_name.push(ch);
                }
            }
        } else if ch == '<' {
            in_tag = true;
            capturing_tag_name = true;
            is_closing_tag = false;
            tag_name.clear();
        } else if ch == '&' {
            in_entity = true;
            entity_buf.clear();
            entity_buf.push('&');
        } else if in_script_or_style {
            // Skip script/style content
        } else if in_heading {
            heading_text.push(ch);
            output.push(ch);
        } else if in_title {
            title_text.push(ch);
        } else {
            output.push(ch);
        }
    }

    // Collapse multiple blank lines into single newlines
    let text = collapse_blank_lines(&output);

    Ok(ExtractedContent {
        metadata: ContentMetadata {
            format: InputFormat::Html.as_str().to_string(),
            title,
            headings,
            line_count: count_lines(&text),
            word_count: count_words(&text),
            ..Default::default()
        },
        text,
    })
}

/// Resolve common HTML character entities to their text equivalents.
fn resolve_html_entity(entity: &str) -> String {
    match entity {
        "&amp;" => "&".to_string(),
        "&lt;" => "<".to_string(),
        "&gt;" => ">".to_string(),
        "&quot;" => "\"".to_string(),
        "&apos;" => "'".to_string(),
        "&nbsp;" => " ".to_string(),
        "&mdash;" => "\u{2014}".to_string(),
        "&ndash;" => "\u{2013}".to_string(),
        "&hellip;" => "\u{2026}".to_string(),
        "&copy;" => "\u{00A9}".to_string(),
        "&reg;" => "\u{00AE}".to_string(),
        "&trade;" => "\u{2122}".to_string(),
        _ => {
            // Try numeric entity: &#123; or &#x1F;
            if entity.starts_with("&#x") || entity.starts_with("&#X") {
                let hex = &entity[3..entity.len() - 1];
                if let Ok(code) = u32::from_str_radix(hex, 16) {
                    if let Some(ch) = char::from_u32(code) {
                        return ch.to_string();
                    }
                }
            } else if entity.starts_with("&#") {
                let num = &entity[2..entity.len() - 1];
                if let Ok(code) = num.parse::<u32>() {
                    if let Some(ch) = char::from_u32(code) {
                        return ch.to_string();
                    }
                }
            }
            // Unknown entity — pass through
            entity.to_string()
        }
    }
}

/// Collapse runs of blank lines into single newlines and trim.
fn collapse_blank_lines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_blank = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_blank && !result.is_empty() {
                result.push('\n');
                prev_blank = true;
            }
        } else {
            if !result.is_empty() && !prev_blank {
                result.push('\n');
            }
            result.push_str(trimmed);
            prev_blank = false;
        }
    }

    result
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Markdown ─────────────────────────────────────────────────────────

    #[test]
    fn markdown_headings_extracted() {
        let md = "# Title\n\nSome text.\n\n## Section One\n\nMore text.\n\n### Sub Section";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert_eq!(result.metadata.title, Some("Title".to_string()));
        assert_eq!(result.metadata.headings.len(), 3);
        assert_eq!(result.metadata.headings[0], "Title");
        assert_eq!(result.metadata.headings[1], "Section One");
        assert_eq!(result.metadata.headings[2], "Sub Section");
    }

    #[test]
    fn markdown_inline_formatting_stripped() {
        let md = "This is **bold** and *italic* and `code` text.";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert_eq!(result.text, "This is bold and italic and code text.");
    }

    #[test]
    fn markdown_links_keep_text() {
        let md = "Visit [Rust](https://rust-lang.org) today.";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert_eq!(result.text, "Visit Rust today.");
    }

    #[test]
    fn markdown_code_block_preserved() {
        let md = "Before\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nAfter";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert!(result.text.contains("fn main()"));
        assert!(result.text.contains("println!"));
        // Fence lines should be omitted
        assert!(!result.text.contains("```"));
    }

    #[test]
    fn markdown_list_markers_stripped() {
        let md = "- item one\n- item two\n1. numbered\n2. also numbered";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert!(result.text.contains("item one"));
        assert!(result.text.contains("numbered"));
        assert!(!result.text.contains("- item"));
        assert!(!result.text.contains("1. "));
    }

    #[test]
    fn markdown_images_alt_text() {
        let md = "An image: ![Alt text](image.png) here.";
        let result = extract(md, InputFormat::Markdown).unwrap();
        assert_eq!(result.text, "An image: Alt text here.");
    }

    // ── JSON ─────────────────────────────────────────────────────────────

    #[test]
    fn json_object_flattened() {
        let json = r#"{"name": "Alice", "age": 30, "address": {"city": "NYC"}}"#;
        let result = extract(json, InputFormat::Json).unwrap();
        assert!(result.text.contains("name: Alice"));
        assert!(result.text.contains("age: 30"));
        assert!(result.text.contains("address.city: NYC"));
        assert!(result.metadata.entities_hint.contains(&"name".to_string()));
        assert!(result.metadata.entities_hint.contains(&"age".to_string()));
    }

    #[test]
    fn json_array_flattened() {
        let json = r#"[{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]"#;
        let result = extract(json, InputFormat::Json).unwrap();
        assert!(result.text.contains("id: 1"));
        assert!(result.text.contains("val: b"));
    }

    #[test]
    fn json_large_array_summarized() {
        // 150 items — should be summarized
        let items: Vec<String> = (0..150).map(|i| format!(r#"{{"n":{}}}"#, i)).collect();
        let json = format!("[{}]", items.join(","));
        let result = extract(&json, InputFormat::Json).unwrap();
        assert!(result.text.contains("150 items"));
    }

    // ── CSV ──────────────────────────────────────────────────────────────

    #[test]
    fn csv_basic() {
        let csv = "Name,Age,City\nAlice,30,NYC\nBob,25,LA";
        let result = extract(csv, InputFormat::Csv).unwrap();
        assert!(result.text.contains("Name: Alice"));
        assert!(result.text.contains("Age: 30"));
        assert!(result.text.contains("City: LA"));
        assert!(result
            .metadata
            .entities_hint
            .contains(&"Name".to_string()));
    }

    #[test]
    fn csv_quoted_fields() {
        let csv = "Name,Description\n\"Smith, John\",\"Has a, comma\"\nJane,Simple";
        let result = extract(csv, InputFormat::Csv).unwrap();
        assert!(result.text.contains("Name: Smith, John"));
        assert!(result.text.contains("Description: Has a, comma"));
    }

    #[test]
    fn csv_tsv_detection() {
        let tsv = "Name\tAge\nAlice\t30";
        let result = extract(tsv, InputFormat::Csv).unwrap();
        assert!(result.text.contains("Name: Alice"));
        assert!(result.text.contains("Age: 30"));
    }

    // ── Code ─────────────────────────────────────────────────────────────

    #[test]
    fn code_rust_entities() {
        let code = r#"
pub fn extract_text(content: &str) -> Result<String> {
    todo!()
}

struct Config {
    path: String,
}

pub async fn handle_request() {}

impl Config {
    fn new() -> Self { todo!() }
}
"#;
        let result = extract(code, InputFormat::Code).unwrap();
        let ents = &result.metadata.entities_hint;
        assert!(ents.contains(&"extract_text".to_string()));
        assert!(ents.contains(&"Config".to_string()));
        assert!(ents.contains(&"handle_request".to_string()));
        assert!(ents.contains(&"new".to_string()));
        // Code preserved as-is
        assert!(result.text.contains("pub fn extract_text"));
    }

    #[test]
    fn code_python_entities() {
        let code = "def process_data(items):\n    pass\n\nclass DataProcessor:\n    pass\n";
        let result = extract(code, InputFormat::Code).unwrap();
        let ents = &result.metadata.entities_hint;
        assert!(ents.contains(&"process_data".to_string()));
        assert!(ents.contains(&"DataProcessor".to_string()));
    }

    #[test]
    fn code_js_entities() {
        let code = "function fetchData() {}\nclass ApiClient {}\ninterface Options {}";
        let result = extract(code, InputFormat::Code).unwrap();
        let ents = &result.metadata.entities_hint;
        assert!(ents.contains(&"fetchData".to_string()));
        assert!(ents.contains(&"ApiClient".to_string()));
        assert!(ents.contains(&"Options".to_string()));
    }

    // ── HTML ─────────────────────────────────────────────────────────────

    #[test]
    fn html_basic_stripping() {
        let html = "<html><body><h1>Hello</h1><p>World</p></body></html>";
        let result = extract(html, InputFormat::Html).unwrap();
        assert!(result.text.contains("Hello"));
        assert!(result.text.contains("World"));
        assert!(!result.text.contains("<"));
        assert_eq!(result.metadata.headings, vec!["Hello"]);
    }

    #[test]
    fn html_title_extracted() {
        let html = "<html><head><title>My Page</title></head><body>Content</body></html>";
        let result = extract(html, InputFormat::Html).unwrap();
        assert_eq!(result.metadata.title, Some("My Page".to_string()));
        assert!(result.text.contains("Content"));
        // Title text should NOT appear in body
        assert!(!result.text.contains("My Page"));
    }

    #[test]
    fn html_script_style_stripped() {
        let html = "<p>Before</p><script>var x = 1;</script><style>.a{}</style><p>After</p>";
        let result = extract(html, InputFormat::Html).unwrap();
        assert!(result.text.contains("Before"));
        assert!(result.text.contains("After"));
        assert!(!result.text.contains("var x"));
        assert!(!result.text.contains(".a{"));
    }

    #[test]
    fn html_entities_resolved() {
        let html = "<p>A &amp; B &lt; C &gt; D</p>";
        let result = extract(html, InputFormat::Html).unwrap();
        assert!(result.text.contains("A & B < C > D"));
    }

    #[test]
    fn html_nested_headings() {
        let html = "<h1>Title</h1><p>Text</p><h2>Sub</h2><p>More</p><h3>Deep</h3>";
        let result = extract(html, InputFormat::Html).unwrap();
        assert_eq!(result.metadata.headings.len(), 3);
        assert_eq!(result.metadata.headings[0], "Title");
        assert_eq!(result.metadata.headings[1], "Sub");
        assert_eq!(result.metadata.headings[2], "Deep");
        assert_eq!(result.metadata.title, Some("Title".to_string()));
    }
}
