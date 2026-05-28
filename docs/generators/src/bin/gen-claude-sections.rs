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
    section_header: &'static str, // e.g. "## Epistemic Hygiene"
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
                out.push_str("> Mirrored from CLAUDE.md. Edit the source there, not this file.\n\n");
                out.push_str(&body);
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
