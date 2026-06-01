//! gen-changelog — produce `docs/src/changelog.md` from git tags + commit log.
//!
//! Strategy:
//!   - Walk `git tag --sort=-creatordate` for `v*` tags (newest first).
//!   - Between each adjacent pair of tags, run `git log <older>..<newer>
//!     --pretty=...` and group commits by conventional-commit prefix
//!     (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, etc.).
//!   - When there are commits between the newest tag and `HEAD`, emit an
//!     "Unreleased" section at the top.
//!
//! Trigger: pre-commit (cheap) and CI. Re-running is idempotent so long as
//! the git history is identical.
//!
//! Failure modes:
//!   - No tags at all → emit only an "Unreleased" section (does not fail).
//!   - `git` not on PATH → loud error.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::process::Command;

use veld_docs_generators::{docs_src_root, generated_header, write_output};

fn git_tags() -> Result<Vec<String>> {
    let out = Command::new("git")
        .args(["tag", "--sort=-creatordate"])
        .output()
        .context("running `git tag` (is git on PATH?)")?;
    if !out.status.success() {
        anyhow::bail!(
            "git tag failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8(out.stdout)?
        .lines()
        .filter(|l| l.starts_with('v'))
        .map(String::from)
        .collect())
}

fn commits_between(from: &str, to: &str) -> Result<Vec<String>> {
    // %s = subject, %h = short hash. We keep both so the changelog cites a
    // navigable hash next to every line.
    let out = Command::new("git")
        .args([
            "log",
            &format!("{from}..{to}"),
            "--pretty=format:%s (%h)",
        ])
        .output()
        .context("running `git log`")?;
    if !out.status.success() {
        anyhow::bail!(
            "git log {from}..{to} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8(out.stdout)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect())
}

/// Group commit lines by their conventional-commit prefix (`feat`, `fix`,
/// etc.). Commits without a recognisable prefix are bucketed under `misc`.
fn group_by_prefix(commits: &[String]) -> BTreeMap<String, Vec<&String>> {
    let mut out: BTreeMap<String, Vec<&String>> = BTreeMap::new();
    for c in commits {
        // Take the substring up to the first `:` or `(`, whichever comes
        // first — covers both `feat: x` and `feat(scope): x`.
        let prefix = c
            .split(|ch: char| ch == ':' || ch == '(')
            .next()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| {
                !s.is_empty()
                    && s.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
            })
            .unwrap_or_else(|| "misc".to_string());
        out.entry(prefix).or_default().push(c);
    }
    out
}

fn render_section(label: &str, commits: &[String], out: &mut String) {
    if commits.is_empty() {
        return;
    }
    out.push_str(&format!("## {label}\n\n"));
    let groups = group_by_prefix(commits);
    for (prefix, lines) in &groups {
        out.push_str(&format!("### {prefix}\n\n"));
        for line in lines {
            out.push_str(&format!("- {line}\n"));
        }
        out.push('\n');
    }
}

fn main() -> Result<()> {
    let tags = git_tags()?;
    let mut body = generated_header("git log + tags", "gen-changelog");
    body.push_str("# Changelog\n\n");
    body.push_str(
        "Generated from git history. Sections are conventional-commit prefixes; \
         hashes link to the commit on origin.\n\n",
    );

    // 1. Unreleased section: HEAD..(newest tag if any).
    let unreleased = match tags.first() {
        Some(tip) => commits_between(tip, "HEAD")?,
        None => commits_between("", "HEAD").unwrap_or_default(),
    };
    if !unreleased.is_empty() {
        render_section("Unreleased", &unreleased, &mut body);
    }

    // 2. Per-release sections: walk adjacent pairs newest-first.
    for win in tags.windows(2) {
        let (newer, older) = (&win[0], &win[1]);
        let commits = commits_between(older, newer)?;
        if commits.is_empty() {
            continue;
        }
        render_section(newer, &commits, &mut body);
    }

    // 3. Oldest tag (no predecessor): just list the tag and a note.
    if let Some(oldest) = tags.last() {
        body.push_str(&format!(
            "## {oldest}\n\nFirst tagged release tracked by this changelog. \
             For pre-{oldest} history see `git log`.\n",
        ));
    } else {
        body.push_str(
            "No tagged releases exist yet — only the *Unreleased* section above \
             is populated.\n",
        );
    }

    let output_path = docs_src_root().join("changelog.md");
    write_output(&output_path, &body)?;
    eprintln!(
        "gen-changelog: {} tags, {} unreleased commits",
        tags.len(),
        unreleased.len()
    );
    Ok(())
}
