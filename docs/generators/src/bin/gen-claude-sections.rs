//! gen-claude-sections — mirror named sections from CLAUDE.md into the docs
//! site. The mapping is hard-coded: each entry pairs a CLAUDE.md `## Section`
//! anchor with an output markdown path.
//!
//! Behaviour:
//!   - If the section exists in CLAUDE.md, mirror its body into the output
//!     path with a "GENERATED FROM" header.
//!   - If the section does NOT exist in CLAUDE.md, leave the output file
//!     untouched. (Useful while LLM-Wiki Phase 1+5 are landing the sections.)
//!   - If the output file does not exist yet AND the section is missing,
//!     write a placeholder noting the section is pending.

use anyhow::Result;
use std::path::PathBuf;
use veld_docs_generators::{docs_src_root, generated_header, read_source, repo_root, write_output};

struct Mirror {
    section_header: &'static str,  // e.g. "## Epistemic Hygiene"
    output_rel_path: &'static str, // relative to docs/src/
}

const MIRRORS: &[Mirror] = &[
    Mirror {
        section_header: "## Epistemic Hygiene",
        output_rel_path: "architecture/epistemic-hygiene.md",
    },
    Mirror {
        section_header: "## Page Contract",
        output_rel_path: "schema/page-contract.md",
    },
    Mirror {
        section_header: "## Encoding Conventions",
        output_rel_path: "guides/encoding-conventions.md",
    },
    Mirror {
        section_header: "## Scale & Migration",
        output_rel_path: "guides/scale-and-migration.md",
    },
    Mirror {
        section_header: "## Tool Coupling",
        output_rel_path: "guides/tool-coupling.md",
    },
    Mirror {
        section_header: "## Schema Version & CHANGELOG",
        output_rel_path: "schema/changelog.md",
    },
];

/// Extract the body of a `## Section` from a markdown document. The body is
/// everything from the first line after the header up to (but not including)
/// the next `## ` line or end-of-file.
fn extract_section(source: &str, header: &str) -> Option<String> {
    let mut iter = source.lines();
    let mut body: Vec<&str> = Vec::new();
    let mut started = false;
    for line in iter.by_ref() {
        if !started {
            if line.trim_end() == header {
                started = true;
            }
            continue;
        }
        // Stop at next top-level section.
        if line.starts_with("## ") && !line.starts_with("### ") {
            break;
        }
        body.push(line);
    }
    if !started {
        return None;
    }
    Some(body.join("\n").trim().to_string())
}

const REPO_BLOB_URL: &str = "https://github.com/Portll/veld/blob/main/";
const REPO_TREE_URL: &str = "https://github.com/Portll/veld/tree/main/";
const BOOK_SOURCE_PREFIX: &str = "docs/src/";

/// CLAUDE.md links resolve from the repository root, but the mirrored page lives under
/// docs/src/, where they point at nothing and fail the book's link check. A book page
/// becomes a page-relative link; any other repository path becomes a GitHub link.
fn rewrite_links(body: &str, output_rel_path: &str) -> String {
    let depth = output_rel_path.matches('/').count();
    let mut in_fence = false;
    let mut lines = Vec::new();
    for line in body.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if in_fence {
            lines.push(line.to_string());
            continue;
        }
        let mut out = String::with_capacity(line.len());
        for (i, segment) in line.split('`').enumerate() {
            if i > 0 {
                out.push('`');
            }
            if i % 2 == 1 {
                out.push_str(segment);
            } else {
                rewrite_inline_links(segment, depth, &mut out);
            }
        }
        lines.push(out);
    }
    lines.join("\n")
}

fn rewrite_inline_links(text: &str, depth: usize, out: &mut String) {
    let mut rest = text;
    while let Some(open) = rest.find("](") {
        let target_start = open + 2;
        let Some(len) = rest[target_start..].find(')') else {
            break;
        };
        out.push_str(&rest[..target_start]);
        out.push_str(&rewrite_target(
            &rest[target_start..target_start + len],
            depth,
        ));
        rest = &rest[target_start + len..];
    }
    out.push_str(rest);
}

fn rewrite_target(target: &str, depth: usize) -> String {
    let is_external_or_relative = target.contains("://")
        || target.starts_with("mailto:")
        || target.starts_with('#')
        || target.starts_with('/')
        || target.starts_with("./")
        || target.starts_with("../");
    if is_external_or_relative {
        return target.to_string();
    }
    if let Some(book_path) = target.strip_prefix(BOOK_SOURCE_PREFIX) {
        return format!("{}{book_path}", "../".repeat(depth));
    }
    let path = target.split('#').next().unwrap_or(target);
    let base = if path.ends_with('/') {
        REPO_TREE_URL
    } else {
        REPO_BLOB_URL
    };
    format!("{base}{target}")
}

fn main() -> Result<()> {
    let claude_md = repo_root().join("CLAUDE.md");
    let source = read_source(&claude_md)?;

    let mut mirrored = 0usize;
    let mut missing = 0usize;
    let mut skipped = 0usize;

    for m in MIRRORS {
        let output_path: PathBuf = docs_src_root().join(m.output_rel_path);
        match extract_section(&source, m.section_header) {
            Some(body) if !body.is_empty() => {
                let title = m.section_header.trim_start_matches("## ");
                let mut out = generated_header(
                    &format!("CLAUDE.md ({})", m.section_header),
                    "gen-claude-sections",
                );
                out.push_str(&format!("# {title}\n\n"));
                out.push_str(
                    "> Mirrored from CLAUDE.md. Edit the source there, not this file.\n\n",
                );
                out.push_str(&rewrite_links(&body, m.output_rel_path));
                out.push_str("\n");
                write_output(&output_path, &out)?;
                mirrored += 1;
            }
            _ => {
                // Section missing in CLAUDE.md. Leave existing output untouched
                // if present (hand-authored placeholder); otherwise emit
                // a minimal placeholder.
                if !output_path.exists() {
                    let placeholder = generated_header(
                        &format!("CLAUDE.md ({})", m.section_header),
                        "gen-claude-sections",
                    ) + &format!(
                        "# {} (pending)\n\nThe `{}` section is not yet present in CLAUDE.md. \
                         This page will be auto-populated by `gen-claude-sections` once the \
                         section lands (LLM-Wiki Phase 1 / Phase 5).\n",
                        m.section_header.trim_start_matches("## "),
                        m.section_header
                    );
                    write_output(&output_path, &placeholder)?;
                    missing += 1;
                } else {
                    skipped += 1;
                }
            }
        }
    }

    eprintln!(
        "gen-claude-sections: {mirrored} mirrored, {missing} placeholders written, {skipped} existing pages untouched"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::rewrite_links;

    #[test]
    fn book_pages_become_page_relative() {
        assert_eq!(
            rewrite_links("[0007](docs/src/decisions/0007-x.md#gates)", "guides/a.md"),
            "[0007](../decisions/0007-x.md#gates)"
        );
    }

    #[test]
    fn repository_files_and_directories_link_to_github() {
        assert_eq!(
            rewrite_links(
                "[c](src/constants.rs#L42) and [d](src/storage/)",
                "guides/a.md"
            ),
            "[c](https://github.com/Portll/veld/blob/main/src/constants.rs#L42) and \
             [d](https://github.com/Portll/veld/tree/main/src/storage/)"
        );
    }

    #[test]
    fn code_spans_fences_and_absolute_links_are_untouched() {
        let body = "`[x](src/a.rs)` [`Dials`](docs/src/d.md) [w](https://e.org/a)\n\
                    ```md\n[y](src/b.rs)\n```";
        assert_eq!(
            rewrite_links(body, "guides/a.md"),
            "`[x](src/a.rs)` [`Dials`](../d.md) [w](https://e.org/a)\n\
             ```md\n[y](src/b.rs)\n```"
        );
    }
}
