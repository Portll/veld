//! Git tab — Phase A in-Veld git viewer.
//!
//! Read-only. Surfaces:
//!   * the current branch in the active worktree
//!   * sibling worktrees and the branch checked out in each
//!   * the most recent commits on the selected branch with author,
//!     timestamp, subject and file-change count
//!
//! There are no diff bodies and no write operations (no push, no
//! checkout, no merge). Refreshes run off the UI thread.
//!
//! Architecture note: this module owns *only* the data acquisition
//! and the rendering for the Git tab. Selection state, focus and
//! cached snapshots live on `AppState` so existing key-handling
//! and tick patterns can drive them without touching this file.

use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, TimeZone, Utc};
use git2::{BranchType, DiffOptions, Repository, Sort};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::types::{AppState, CommitInfo, GitPane, GitState, WorktreeInfo};

/// Maximum number of commits to load per branch view.
pub const MAX_COMMITS: usize = 50;

// ---------------------------------------------------------------------------
// Data acquisition
// ---------------------------------------------------------------------------

impl GitState {
    /// Build a fresh `GitState` snapshot for the repository that contains
    /// `start_path`. `start_path` is typically the process current
    /// directory; `Repository::discover` walks up to find the .git dir,
    /// which also handles being inside a worktree.
    ///
    /// `branch_for_commits` selects which branch's history to load. When
    /// `None`, the active worktree's current branch is used.
    pub fn from_repo(
        start_path: &Path,
        branch_for_commits: Option<&str>,
    ) -> Result<Self, git2::Error> {
        let repo = Repository::discover(start_path)?;

        // ---------- current branch (in *this* worktree) ----------
        let head = repo.head().ok();
        let current_branch = head
            .as_ref()
            .and_then(|r| r.shorthand().map(|s| s.to_string()))
            .filter(|name| name != "HEAD");

        // ---------- branches ----------
        let mut branches: Vec<String> = Vec::new();
        for entry in repo.branches(Some(BranchType::Local))? {
            let (branch, _) = entry?;
            if let Some(name) = branch.name()?.map(|s| s.to_string()) {
                branches.push(name);
            }
        }
        branches.sort();
        branches.dedup();

        // ---------- worktrees (main + linked) ----------
        let worktrees = collect_worktrees(&repo, start_path)?;

        // ---------- recent commits on the chosen branch ----------
        let branch_to_walk = branch_for_commits
            .map(|s| s.to_string())
            .or_else(|| current_branch.clone());

        let recent_commits = match branch_to_walk.as_deref() {
            Some(name) => load_recent_commits(&repo, name)?,
            None => Vec::new(),
        };

        Ok(GitState {
            current_branch,
            branches,
            worktrees,
            recent_commits,
            last_refreshed: Instant::now(),
        })
    }
}

/// Resolve the canonical filesystem path of every worktree on this
/// repository, including the main one. Marks the active worktree based
/// on whether `active_hint` is a prefix of (or equal to) the worktree's
/// path.
fn collect_worktrees(repo: &Repository, active_hint: &Path) -> Result<Vec<WorktreeInfo>, git2::Error> {
    let mut out: Vec<WorktreeInfo> = Vec::new();

    // The main worktree's filesystem path. `Repository::path()` returns
    // the gitdir (e.g. `<root>/.git` for the main repo or
    // `<root>/.git/worktrees/<name>` when opened from a linked one).
    // When in a linked worktree, `<gitdir>/commondir` is a small text
    // file containing a relative path back to the canonical `.git`
    // directory — that is what we use here so the result is the same
    // regardless of which worktree the TUI was started in.
    let gitdir = repo.path().to_path_buf();
    let common_gitdir = read_commondir(&gitdir).unwrap_or(gitdir.clone());
    let main_workdir: Option<PathBuf> = common_gitdir
        .parent()
        .map(|p: &Path| p.to_path_buf())
        .or_else(|| repo.workdir().map(|p| p.to_path_buf()));

    let canonical_active = canonicalize_or_keep(active_hint);

    if let Some(main_path) = main_workdir {
        let canonical_main = canonicalize_or_keep(&main_path);
        let branch = branch_for_workdir(&canonical_main).ok().flatten();
        let name = path_display_name(&canonical_main);
        let is_active = paths_equivalent(&canonical_main, &canonical_active);
        out.push(WorktreeInfo {
            path: canonical_main.to_string_lossy().into_owned(),
            name,
            branch,
            is_active,
        });
    }

    for wt_name in repo.worktrees()?.iter().flatten() {
        let wt = match repo.find_worktree(wt_name) {
            Ok(wt) => wt,
            Err(_) => continue,
        };
        let wt_path = wt.path();
        let canonical_wt = canonicalize_or_keep(wt_path);
        let branch = branch_for_workdir(&canonical_wt).ok().flatten();
        let name = wt_name.to_string();
        let is_active = paths_equivalent(&canonical_wt, &canonical_active);
        out.push(WorktreeInfo {
            path: canonical_wt.to_string_lossy().into_owned(),
            name,
            branch,
            is_active,
        });
    }

    // Stable ordering: active first, then alphabetically by name.
    out.sort_by(|a, b| match (a.is_active, b.is_active) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.cmp(&b.name),
    });

    Ok(out)
}

/// Read the branch shorthand for a given working directory by opening
/// that working directory as its own repository. Returns `Ok(None)` for
/// a detached HEAD.
fn branch_for_workdir(workdir: &Path) -> Result<Option<String>, git2::Error> {
    let repo = Repository::open(workdir)?;
    let head = match repo.head() {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };
    Ok(head
        .shorthand()
        .map(|s| s.to_string())
        .filter(|s| s != "HEAD"))
}

fn canonicalize_or_keep(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Resolve `<gitdir>/commondir` to the canonical .git path used by all
/// worktrees of a repository. Returns `None` when the file is missing
/// (i.e. `gitdir` already *is* the common gitdir).
fn read_commondir(gitdir: &Path) -> Option<PathBuf> {
    let marker = gitdir.join("commondir");
    let rel = std::fs::read_to_string(&marker).ok()?;
    let rel = rel.trim();
    if rel.is_empty() {
        return None;
    }
    let joined = gitdir.join(rel);
    Some(canonicalize_or_keep(&joined))
}

fn paths_equivalent(a: &Path, b: &Path) -> bool {
    a == b
}

fn path_display_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

/// Walk up to `MAX_COMMITS` commits on a local branch and return them in
/// newest-first order.
fn load_recent_commits(repo: &Repository, branch_name: &str) -> Result<Vec<CommitInfo>, git2::Error> {
    let branch = repo.find_branch(branch_name, BranchType::Local)?;
    let oid = match branch.get().target() {
        Some(oid) => oid,
        None => return Ok(Vec::new()),
    };

    let mut walk = repo.revwalk()?;
    walk.set_sorting(Sort::TIME)?;
    walk.push(oid)?;

    let mut out = Vec::with_capacity(MAX_COMMITS);
    for (idx, step) in walk.enumerate() {
        if idx >= MAX_COMMITS {
            break;
        }
        let commit_oid = step?;
        let commit = repo.find_commit(commit_oid)?;

        let author = commit.author();
        let when = author.when();
        let time = Utc
            .timestamp_opt(when.seconds(), 0)
            .single()
            .unwrap_or_else(Utc::now);

        let subject = commit
            .summary()
            .unwrap_or("(no message)")
            .lines()
            .next()
            .unwrap_or("")
            .to_string();

        let files_changed = file_change_count(repo, &commit).unwrap_or(0);

        out.push(CommitInfo {
            short_oid: commit
                .as_object()
                .short_id()
                .ok()
                .and_then(|s| s.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| commit_oid.to_string()[..7.min(commit_oid.to_string().len())].to_string()),
            subject,
            author: author.name().unwrap_or("unknown").to_string(),
            time,
            files_changed,
        });
    }
    Ok(out)
}

/// Count files changed vs the commit's first parent. Returns `Ok(0)` for
/// root commits and rolls library errors up.
fn file_change_count(repo: &Repository, commit: &git2::Commit) -> Result<usize, git2::Error> {
    let tree = commit.tree()?;
    let parent_tree = if commit.parent_count() == 0 {
        None
    } else {
        Some(commit.parent(0)?.tree()?)
    };

    let mut opts = DiffOptions::new();
    opts.skip_binary_check(true);

    let diff = repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut opts))?;
    Ok(diff.deltas().len())
}

// ---------------------------------------------------------------------------
// Selection helpers — operate on AppState. Public for the key-handling
// layer in main.rs.
// ---------------------------------------------------------------------------

pub fn move_up(state: &mut AppState) {
    let pane = state.git_focus;
    let len = pane_len(state, pane);
    if len == 0 {
        return;
    }
    let cur = pane_selected_mut(state, pane);
    if *cur > 0 {
        *cur -= 1;
    }
}

pub fn move_down(state: &mut AppState) {
    let pane = state.git_focus;
    let len = pane_len(state, pane);
    if len == 0 {
        return;
    }
    let max = len - 1;
    let cur = pane_selected_mut(state, pane);
    if *cur < max {
        *cur += 1;
    }
}

pub fn cycle_pane(state: &mut AppState) {
    state.git_focus = state.git_focus.next();
}

fn pane_len(state: &AppState, pane: GitPane) -> usize {
    let g = match state.git_state.as_ref() {
        Some(g) => g,
        None => return 0,
    };
    match pane {
        GitPane::Worktrees => g.worktrees.len(),
        GitPane::Branches => g.branches.len(),
        GitPane::Commits => g.recent_commits.len(),
    }
}

fn pane_selected_mut(state: &mut AppState, pane: GitPane) -> &mut usize {
    match pane {
        GitPane::Worktrees => &mut state.git_selected_worktree,
        GitPane::Branches => &mut state.git_selected_branch,
        GitPane::Commits => &mut state.git_selected_commit,
    }
}

/// Returns the branch the commits pane should be showing.
///
/// Selection precedence: the branch currently highlighted in the
/// Branches pane, falling back to the active worktree's current branch.
pub fn selected_branch_name(state: &AppState) -> Option<String> {
    state
        .git_state
        .as_ref()
        .and_then(|g| {
            g.branches
                .get(state.git_selected_branch)
                .cloned()
                .or_else(|| g.current_branch.clone())
        })
}

/// Clamp selection indices in case the snapshot shrank (e.g. a worktree
/// was deleted between refreshes).
pub fn clamp_selections(state: &mut AppState) {
    if let Some(ref g) = state.git_state {
        if !g.worktrees.is_empty() && state.git_selected_worktree >= g.worktrees.len() {
            state.git_selected_worktree = g.worktrees.len() - 1;
        }
        if !g.branches.is_empty() && state.git_selected_branch >= g.branches.len() {
            state.git_selected_branch = g.branches.len() - 1;
        }
        if !g.recent_commits.is_empty() && state.git_selected_commit >= g.recent_commits.len() {
            state.git_selected_commit = g.recent_commits.len() - 1;
        } else if g.recent_commits.is_empty() {
            state.git_selected_commit = 0;
        }

        // If we have a current branch and no explicit branch selection
        // has been moved off of it, snap selection onto the current
        // branch on first load.
        if state.git_commits_for_branch.is_none() {
            if let Some(ref cur) = g.current_branch {
                if let Some(idx) = g.branches.iter().position(|b| b == cur) {
                    state.git_selected_branch = idx;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Colour the tab uses for emphasis. Picked to read on both themes.
const ACCENT: Color = Color::Rgb(255, 183, 130);
const DIM: Color = Color::Rgb(140, 140, 145);
const ACTIVE_MARK: Color = Color::Rgb(180, 230, 180);

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.border()))
        .title(Span::styled(
            " Git ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Top status line: current branch + staleness.
    let status_height = 1u16;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(status_height),
            Constraint::Min(5),
        ])
        .split(inner);

    render_status_line(f, chunks[0], state);

    // Three-pane horizontal layout: worktrees (left), branches (middle),
    // commits (right).
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(22),
            Constraint::Percentage(22),
            Constraint::Percentage(56),
        ])
        .split(chunks[1]);

    render_worktree_pane(f, panes[0], state);
    render_branch_pane(f, panes[1], state);
    render_commit_pane(f, panes[2], state);
}

fn render_status_line(f: &mut Frame, area: Rect, state: &AppState) {
    let mut spans: Vec<Span> = Vec::new();

    if let Some(ref g) = state.git_state {
        spans.push(Span::styled(" branch ", Style::default().fg(DIM)));
        spans.push(Span::styled(
            g.current_branch.clone().unwrap_or_else(|| "(detached)".into()),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!(
                "{} worktree(s), {} branch(es), {} commit(s)",
                g.worktrees.len(),
                g.branches.len(),
                g.recent_commits.len()
            ),
            Style::default().fg(DIM),
        ));
        spans.push(Span::raw("  "));
        let secs = g.last_refreshed.elapsed().as_secs();
        spans.push(Span::styled(
            format!("refreshed {}s ago", secs),
            Style::default().fg(DIM),
        ));
    } else if state.git_refreshing {
        spans.push(Span::styled(
            " loading git state… ",
            Style::default().fg(ACCENT),
        ));
    } else if let Some(ref err) = state.git_error {
        spans.push(Span::styled(
            format!(" git error: {} ", err),
            Style::default().fg(Color::Red),
        ));
    } else {
        spans.push(Span::styled(" (no git state yet) ", Style::default().fg(DIM)));
    }

    if state.git_refreshing && state.git_state.is_some() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled("refreshing…", Style::default().fg(ACCENT)));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_worktree_pane(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.git_focus == GitPane::Worktrees;
    let block = pane_block(" Worktrees ", focused, state);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let g = match state.git_state.as_ref() {
        Some(g) => g,
        None => return,
    };

    if g.worktrees.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("(none)", Style::default().fg(DIM))),
            inner,
        );
        return;
    }

    let lines: Vec<Line> = g
        .worktrees
        .iter()
        .enumerate()
        .map(|(i, wt)| {
            let is_selected = i == state.git_selected_worktree;
            let prefix = if wt.is_active { "* " } else { "  " };
            let style = base_row_style(is_selected && focused);
            let mut spans = vec![
                Span::styled(
                    prefix,
                    Style::default()
                        .fg(if wt.is_active { ACTIVE_MARK } else { DIM }),
                ),
                Span::styled(truncate(&wt.name, 18), style),
            ];
            if let Some(ref b) = wt.branch {
                spans.push(Span::raw("  "));
                spans.push(Span::styled(
                    truncate(b, 24),
                    Style::default().fg(DIM),
                ));
            }
            Line::from(spans)
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

fn render_branch_pane(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.git_focus == GitPane::Branches;
    let block = pane_block(" Branches ", focused, state);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let g = match state.git_state.as_ref() {
        Some(g) => g,
        None => return,
    };

    if g.branches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("(none)", Style::default().fg(DIM))),
            inner,
        );
        return;
    }

    let current = g.current_branch.as_deref();

    let lines: Vec<Line> = g
        .branches
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let is_selected = i == state.git_selected_branch;
            let is_current = Some(b.as_str()) == current;
            let prefix = if is_current { "● " } else { "  " };
            let style = base_row_style(is_selected && focused);
            Line::from(vec![
                Span::styled(
                    prefix,
                    Style::default().fg(if is_current { ACTIVE_MARK } else { DIM }),
                ),
                Span::styled(truncate(b, 26), style),
            ])
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

fn render_commit_pane(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.git_focus == GitPane::Commits;
    let title = match selected_branch_name(state) {
        Some(b) => format!(" Commits — {} ", b),
        None => " Commits ".to_string(),
    };
    let block = pane_block(&title, focused, state);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let g = match state.git_state.as_ref() {
        Some(g) => g,
        None => return,
    };

    if g.recent_commits.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "(no commits on selected branch)",
                Style::default().fg(DIM),
            )),
            inner,
        );
        return;
    }

    let header = Row::new(vec![
        Cell::from(Span::styled("oid", Style::default().fg(DIM))),
        Cell::from(Span::styled("when", Style::default().fg(DIM))),
        Cell::from(Span::styled("author", Style::default().fg(DIM))),
        Cell::from(Span::styled("Δ", Style::default().fg(DIM))),
        Cell::from(Span::styled("subject", Style::default().fg(DIM))),
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let now = Utc::now();
    let rows: Vec<Row> = g
        .recent_commits
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let is_selected = i == state.git_selected_commit && focused;
            let when = format_when(now, c.time);
            let row_style = base_row_style(is_selected);
            Row::new(vec![
                Cell::from(Span::styled(c.short_oid.clone(), Style::default().fg(ACCENT))),
                Cell::from(Span::raw(when)),
                Cell::from(Span::raw(truncate(&c.author, 16))),
                Cell::from(Span::raw(format!("{:>3}", c.files_changed))),
                Cell::from(Span::raw(truncate(&c.subject, 120))),
            ])
            .style(row_style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(16),
            Constraint::Length(4),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .column_spacing(1);

    f.render_widget(table, inner);
}

fn pane_block<'a>(title: &str, focused: bool, state: &AppState) -> Block<'a> {
    let border_color = if focused {
        ACCENT
    } else {
        state.theme.border()
    };
    let title_style = if focused {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(DIM)
    };
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(title.to_string(), title_style))
}

fn base_row_style(selected: bool) -> Style {
    if selected {
        Style::default()
            .bg(Color::Rgb(40, 40, 55))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn format_when(now: DateTime<Utc>, t: DateTime<Utc>) -> String {
    let delta = now.signed_duration_since(t);
    let secs = delta.num_seconds();
    if secs < 0 {
        // Future-dated commit — show the date.
        return t.format("%Y-%m-%d").to_string();
    }
    if secs < 60 {
        return format!("{}s", secs);
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{}m", mins);
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{}h", hours);
    }
    let days = hours / 24;
    if days < 30 {
        return format!("{}d", days);
    }
    t.format("%Y-%m-%d").to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the repo root if the working directory of the test
    /// process is inside a git repository, otherwise `None`. We rely on
    /// `Repository::discover` rather than shelling out so the test works
    /// the same in CI and locally.
    fn current_repo_root() -> Option<PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let repo = Repository::discover(&cwd).ok()?;
        repo.workdir().map(|p| p.to_path_buf())
    }

    #[test]
    fn from_repo_returns_populated_state_for_current_repo() {
        let root = match current_repo_root() {
            Some(r) => r,
            None => {
                // Not in a git checkout (e.g. some package-only build).
                // Skip — we cannot meaningfully assert anything.
                eprintln!("skipping: not in a git repo");
                return;
            }
        };

        let state = GitState::from_repo(&root, None).expect("from_repo should succeed in a checkout");

        // At minimum we should have one worktree (the main one) and one
        // local branch, and the last_refreshed instant should be recent.
        assert!(!state.worktrees.is_empty(), "expected at least the main worktree");
        assert!(!state.branches.is_empty(), "expected at least one local branch");
        assert!(state.last_refreshed.elapsed().as_secs() < 5);
        // The active worktree must be flagged.
        assert!(state.worktrees.iter().any(|w| w.is_active), "expected exactly one active worktree");

        // If we have a current branch, walking its commits should
        // produce something (this repo has thousands of commits).
        if state.current_branch.is_some() {
            assert!(
                !state.recent_commits.is_empty(),
                "expected commits on current branch"
            );
            assert!(state.recent_commits.len() <= MAX_COMMITS);
            // Subject should be non-empty for the tip commit.
            let tip = &state.recent_commits[0];
            assert!(!tip.subject.is_empty());
            assert!(!tip.short_oid.is_empty());
        }
    }

    #[test]
    fn truncate_handles_unicode_and_ascii() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello w…");
        // multibyte
        let s = "αβγδεζηθ";
        let out = truncate(s, 4);
        assert!(out.chars().count() <= 4);
    }

    #[test]
    fn format_when_buckets_are_sensible() {
        let now = Utc::now();
        let secs_ago = now - chrono::Duration::seconds(10);
        assert_eq!(format_when(now, secs_ago), "10s");
        let mins_ago = now - chrono::Duration::minutes(5);
        assert_eq!(format_when(now, mins_ago), "5m");
        let hours_ago = now - chrono::Duration::hours(3);
        assert_eq!(format_when(now, hours_ago), "3h");
        let days_ago = now - chrono::Duration::days(2);
        assert_eq!(format_when(now, days_ago), "2d");
    }
}
