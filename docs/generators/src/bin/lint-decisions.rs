//! lint-decisions — validate ADR frontmatter and ID continuity across
//! `docs/src/decisions/*.md`. Outputs nothing on success. On failure, prints
//! findings and exits non-zero.
//!
//! Checks:
//!   - Each `NNNN-*.md` file has YAML frontmatter with `id`, `title`,
//!     `status`, `date`.
//!   - `id` value matches the filename prefix.
//!   - IDs are continuous (1, 2, 3, ...) with no gaps and no duplicates.
//!   - `status` is one of: accepted, superseded, deferred, draft.
//!   - `date` parses as YYYY-MM-DD.
//!   - `index.md` exists.

use regex::Regex;
use std::collections::BTreeMap;
use std::process::ExitCode;
use walkdir::WalkDir;

use veld_docs_generators::{docs_src_root, repo_root};

const VALID_STATUSES: &[&str] = &["accepted", "superseded", "deferred", "draft", "living"];

#[derive(Debug)]
struct Finding {
    file: String,
    issue: String,
}

fn extract_frontmatter(source: &str) -> Option<&str> {
    // YAML frontmatter: starts with --- and ends with ---.
    let body = source.strip_prefix("---")?;
    let body = body.strip_prefix('\n').unwrap_or(body);
    let end = body.find("\n---")?;
    Some(&body[..end])
}

fn get_field<'a>(fm: &'a str, key: &str) -> Option<&'a str> {
    for line in fm.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix(&format!("{key}:")) {
            return Some(rest.trim());
        }
    }
    None
}

fn lint_file(path: &std::path::Path, findings: &mut Vec<Finding>) -> Option<u32> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let rel = path
        .strip_prefix(repo_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    if filename == "index.md" {
        return None; // index has minimal frontmatter; not part of ID continuity
    }

    let Ok(source) = std::fs::read_to_string(path) else {
        findings.push(Finding {
            file: rel,
            issue: "unreadable".into(),
        });
        return None;
    };

    let Some(fm) = extract_frontmatter(&source) else {
        findings.push(Finding {
            file: rel.clone(),
            issue: "missing YAML frontmatter (must start with `---`)".into(),
        });
        return None;
    };

    // Required fields.
    let mut id: Option<u32> = None;
    for key in &["id", "title", "status", "date"] {
        match get_field(fm, key) {
            Some(v) if !v.is_empty() => {
                if *key == "id" {
                    if let Ok(n) = v.parse::<u32>() {
                        id = Some(n);
                    } else {
                        findings.push(Finding {
                            file: rel.clone(),
                            issue: format!("id `{v}` is not an unsigned integer"),
                        });
                    }
                }
                if *key == "status" && !VALID_STATUSES.contains(&v) {
                    findings.push(Finding {
                        file: rel.clone(),
                        issue: format!(
                            "status `{v}` not in valid set ({})",
                            VALID_STATUSES.join(", ")
                        ),
                    });
                }
                if *key == "date" {
                    let date_re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
                    if !date_re.is_match(v) {
                        findings.push(Finding {
                            file: rel.clone(),
                            issue: format!("date `{v}` is not YYYY-MM-DD"),
                        });
                    }
                }
            }
            _ => findings.push(Finding {
                file: rel.clone(),
                issue: format!("missing required frontmatter field `{key}`"),
            }),
        }
    }

    // Verify id matches filename prefix.
    if let Some(id_val) = id {
        let prefix_re = Regex::new(r"^(\d{4})-").unwrap();
        if let Some(cap) = prefix_re.captures(&filename) {
            let prefix: u32 = cap[1].parse().unwrap();
            if prefix != id_val {
                findings.push(Finding {
                    file: rel.clone(),
                    issue: format!("id `{id_val}` does not match filename prefix `{prefix:04}`"),
                });
            }
        }
    }

    id
}

fn main() -> ExitCode {
    let decisions_dir = docs_src_root().join("decisions");
    let mut findings: Vec<Finding> = Vec::new();
    let mut ids: BTreeMap<u32, String> = BTreeMap::new();

    if !decisions_dir.exists() {
        eprintln!(
            "lint-decisions: no decisions directory at {}",
            decisions_dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let index = decisions_dir.join("index.md");
    if !index.exists() {
        findings.push(Finding {
            file: "docs/src/decisions/index.md".into(),
            issue: "missing index page".into(),
        });
    }

    for entry in WalkDir::new(&decisions_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
    {
        let path = entry.path();
        if let Some(id) = lint_file(path, &mut findings) {
            if let Some(existing) = ids.get(&id) {
                findings.push(Finding {
                    file: path.to_string_lossy().to_string(),
                    issue: format!("duplicate id {id} (already used by `{existing}`)"),
                });
            } else {
                ids.insert(
                    id,
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string(),
                );
            }
        }
    }

    // Continuity check.
    if let (Some(&first), Some(&last)) = (ids.keys().next(), ids.keys().next_back()) {
        for expected in first..=last {
            if !ids.contains_key(&expected) {
                findings.push(Finding {
                    file: "docs/src/decisions/".into(),
                    issue: format!("missing decision id {expected:04} (gap in sequence)"),
                });
            }
        }
    }

    if findings.is_empty() {
        eprintln!(
            "lint-decisions: ✓ all {} decisions are valid",
            ids.len()
        );
        return ExitCode::SUCCESS;
    }

    eprintln!("lint-decisions: {} finding(s)", findings.len());
    for f in &findings {
        eprintln!("  - {}: {}", f.file, f.issue);
    }
    ExitCode::FAILURE
}
