//! Loops tab: the self-evaluation loops that write their state to `~/.config/veld/loops/` — the
//! knowledge-base generators and the Qwen work lanes. Each loop says what it attempted, what its
//! gate accepted and refused, why it refused most, and how that has moved hour by hour.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use ratatui::prelude::*;
use ratatui::widgets::*;
use serde_json::Value;

use crate::types::AppState;

const ACCENT: Color = Color::Rgb(130, 200, 255);
const GOOD: Color = Color::Rgb(140, 220, 140);
const BAD: Color = Color::Rgb(240, 130, 130);
const REFRESH: Duration = Duration::from_secs(2);

static CACHE: OnceLock<Mutex<(Option<Instant>, Vec<Value>)>> = OnceLock::new();

fn loops_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("veld").join("loops"))
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn number(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in path {
        cur = cur.get(*k)?;
    }
    cur.as_f64()
}

fn count(v: &Value, path: &[&str]) -> String {
    number(v, path).map_or_else(|| "–".to_string(), |n| format!("{}", n as i64))
}

fn percent(v: &Value, path: &[&str]) -> String {
    number(v, path).map_or_else(|| "–".to_string(), |n| format!("{:.0}%", n * 100.0))
}

fn is_lane(v: &Value) -> bool {
    text(v, "kind") == "lane"
}

/// Every loop's snapshot, the lanes first and then the knowledge bases by id, read from disk at
/// most every two seconds.
pub fn snapshots() -> Vec<Value> {
    let cell = CACHE.get_or_init(|| Mutex::new((None, Vec::new())));
    let mut guard = match cell.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.0.is_none_or(|t| t.elapsed() >= REFRESH) {
        guard.1 = loops_dir().map(|d| read_from(&d)).unwrap_or_default();
        guard.0 = Some(Instant::now());
    }
    guard.1.clone()
}

fn read_from(dir: &Path) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<Value> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| fs::read_to_string(p).ok())
        .filter_map(|t| serde_json::from_str::<Value>(&t).ok())
        .collect();
    out.sort_by_key(|v| (!is_lane(v), text(v, "id")));
    out
}

fn age(updated: &str) -> String {
    let Ok(t) = chrono::DateTime::parse_from_rfc3339(updated) else {
        return "–".to_string();
    };
    let secs = (chrono::Utc::now() - t.with_timezone(&chrono::Utc))
        .num_seconds()
        .max(0);
    match secs {
        s if s < 120 => format!("{s}s ago"),
        s if s < 7200 => format!("{}m ago", s / 60),
        s => format!("{}h ago", s / 3600),
    }
}

/// The loop the selection points at: the Up and Down keys move `scroll_offset`, wrapped to the list.
fn selected(state: &AppState, n: usize) -> usize {
    if n == 0 {
        0
    } else {
        state.scroll_offset % n
    }
}

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    render_loops(f, area, state, &snapshots());
}

fn render_loops(f: &mut Frame, area: Rect, state: &AppState, loops: &[Value]) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.border()))
        .title(Span::styled(
            " Loops — self-evaluation ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if loops.is_empty() {
        let dir = loops_dir()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        let msg = Paragraph::new(vec![
            Line::from(Span::styled("No loop has reported yet.", Style::default().add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from(format!("A loop writes its snapshot to {dir}/<id>.json and every attempt to <id>.events.jsonl.")),
        ])
        .wrap(Wrap { trim: false });
        f.render_widget(msg, inner);
        return;
    }

    let table_height = (loops.len() as u16 + 3).min(inner.height / 2).max(4);
    let [top, bottom] =
        Layout::vertical([Constraint::Length(table_height), Constraint::Min(8)]).areas(inner);
    let sel = selected(state, loops.len());
    render_table(f, top, loops, sel);
    render_detail(f, bottom, &loops[sel]);
}

fn render_table(f: &mut Frame, area: Rect, loops: &[Value], sel: usize) {
    let header = Row::new([
        "Loop",
        "Accepted",
        "Refused",
        "Waiting",
        "Accept rate",
        "First try",
        "Errors",
        "Updated",
    ])
    .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));
    let rows = loops.iter().enumerate().map(|(i, v)| {
        let (accepted, refused, waiting, rate, first) = if is_lane(v) {
            (
                count(v, &["counts", "passed"]),
                count(v, &["counts", "failed"]),
                count(v, &["counts", "readyJobs"]),
                percent(v, &["rates", "gatePass"]),
                percent(v, &["rates", "landedAsDrafted"]),
            )
        } else {
            (
                count(v, &["counts", "accepted"]),
                count(v, &["counts", "rejected"]),
                count(v, &["counts", "pending"]),
                percent(v, &["rates", "accepted"]),
                percent(v, &["rates", "acceptedFirstTry"]),
            )
        };
        let style = if i == sel {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        Row::new(vec![
            Cell::from(text(v, "title")),
            Cell::from(accepted).style(Style::default().fg(GOOD)),
            Cell::from(refused).style(Style::default().fg(BAD)),
            Cell::from(waiting),
            Cell::from(rate),
            Cell::from(first),
            Cell::from(count(v, &["counts", "errors"])),
            Cell::from(age(&text(v, "updatedAt"))),
        ])
        .style(style)
    });
    let widths = [
        Constraint::Min(30),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(7),
        Constraint::Length(9),
    ];
    f.render_widget(Table::new(rows, widths).header(header), area);
}

fn render_detail(f: &mut Frame, area: Rect, v: &Value) {
    let [stats, history] =
        Layout::vertical([Constraint::Min(6), Constraint::Length(5)]).areas(area);
    let [left, mid, right] = Layout::horizontal([
        Constraint::Percentage(30),
        Constraint::Percentage(35),
        Constraint::Percentage(35),
    ])
    .areas(stats);
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);

    let mut facts = vec![Line::from(Span::styled(text(v, "title"), bold))];
    if is_lane(v) {
        facts.extend([
            Line::from(format!(
                "runs {}  passed {}  failed {}",
                count(v, &["counts", "runs"]),
                count(v, &["counts", "passed"]),
                count(v, &["counts", "failed"])
            )),
            Line::from(format!(
                "landed {}  dropped {}",
                count(v, &["counts", "landed"]),
                count(v, &["counts", "dropped"])
            )),
            Line::from(format!(
                "gate pass {}  landed as drafted {}",
                percent(v, &["rates", "gatePass"]),
                percent(v, &["rates", "landedAsDrafted"])
            )),
            Line::from(format!(
                "correction min per landing {}",
                number(v, &["rates", "correctionMinPerLanding"])
                    .map_or("–".into(), |n| format!("{n:.1}"))
            )),
            Line::from(format!(
                "ready {} jobs, {} min of wall clock",
                count(v, &["counts", "readyJobs"]),
                count(v, &["counts", "readyWallMin"])
            )),
            Line::from(format!(
                "Qwen hours {}  lanes {}",
                number(v, &["counts", "qwenHours"]).map_or("–".into(), |n| format!("{n:.1}")),
                count(v, &["lanes"])
            )),
        ]);
        if let Some(fams) = v.get("byFamily").and_then(Value::as_object) {
            facts.push(Line::from(""));
            for (name, fam) in fams {
                facts.push(Line::from(Span::styled(
                    format!(
                        "{name}: {} runs, {} pass",
                        count(fam, &["runs"]),
                        percent(fam, &["passRate"])
                    ),
                    dim,
                )));
            }
        }
    } else {
        facts.extend([
            Line::from(format!("namespace {}", text(v, "namespace"))),
            Line::from(format!(
                "targets {}  accepted {}  refused {}",
                count(v, &["counts", "targets"]),
                count(v, &["counts", "accepted"]),
                count(v, &["counts", "rejected"])
            )),
            Line::from(format!(
                "rows in veld {}  attempts {}",
                count(v, &["counts", "rows"]),
                count(v, &["counts", "attempts"])
            )),
            Line::from(format!(
                "accept rate {}  first try {}",
                percent(v, &["rates", "accepted"]),
                percent(v, &["rates", "acceptedFirstTry"])
            )),
            Line::from(format!(
                "attempts per target {}",
                number(v, &["rates", "attemptsPerTarget"])
                    .map_or("–".into(), |n| format!("{n:.2}"))
            )),
        ]);
    }
    f.render_widget(
        Paragraph::new(facts)
            .block(Block::default().borders(Borders::RIGHT))
            .wrap(Wrap { trim: true }),
        left,
    );

    let mut reasons = vec![Line::from(Span::styled("Why the gate refused", bold))];
    for r in v
        .get("topReasons")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let label = r.get(0).and_then(Value::as_str).unwrap_or("");
        let n = r.get(1).and_then(Value::as_i64).unwrap_or(0);
        reasons.push(Line::from(vec![
            Span::styled(format!("{n:>4} "), Style::default().fg(BAD)),
            Span::raw(label.to_string()),
        ]));
    }
    f.render_widget(
        Paragraph::new(reasons)
            .block(Block::default().borders(Borders::RIGHT))
            .wrap(Wrap { trim: true }),
        mid,
    );

    let mut recent = vec![Line::from(Span::styled("Recent", bold))];
    for e in v
        .get("recent")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(right.height.saturating_sub(1) as usize)
    {
        let outcome = text(e, "outcome");
        let colour = match outcome.as_str() {
            "accepted" | "passed" | "kb-accepted" => GOOD,
            "rejected" | "failed" | "kb-rejected" | "error" => BAD,
            _ => Color::Gray,
        };
        let what = if outcome.is_empty() {
            text(e, "kind")
        } else {
            outcome
        };
        recent.push(Line::from(vec![
            Span::styled(format!("{what:<9} "), Style::default().fg(colour)),
            Span::raw(text(e, "target")),
        ]));
    }
    f.render_widget(Paragraph::new(recent).wrap(Wrap { trim: true }), right);

    let key = if is_lane(v) { "passed" } else { "accepted" };
    let points: Vec<u64> = v
        .get("trajectory")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|p| p.get(key).and_then(Value::as_u64).unwrap_or(0))
        .collect();
    let spark = Sparkline::default()
        .block(Block::default().borders(Borders::TOP).title(Span::styled(
            format!(" {key} per hour, last {} hours ", points.len()),
            dim,
        )))
        .data(&points)
        .style(Style::default().fg(GOOD));
    f.render_widget(spark, history);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn sample() -> Vec<Value> {
        vec![
            json!({ "id": "qwen-lane", "kind": "lane", "title": "Qwen work lanes", "updatedAt": "2026-10-02T07:00:00Z",
                    "counts": { "runs": 10, "passed": 4, "failed": 5, "errors": 1, "readyJobs": 7 }, "rates": { "gatePass": 0.444, "landedAsDrafted": 0.25 },
                    "trajectory": [{ "at": "2026-10-02T06:00Z", "passed": 2 }, { "at": "2026-10-02T07:00Z", "passed": 2 }] }),
            json!({ "id": "cobolwork-utility", "kind": "kb", "title": "cobolwork's utility knowledge base", "namespace": "cobolwork", "updatedAt": "2026-10-02T07:00:00Z",
                    "counts": { "targets": 34, "accepted": 3, "rejected": 1, "pending": 30, "attempts": 6, "errors": 0 }, "rates": { "accepted": 0.75, "acceptedFirstTry": 0.667 },
                    "topReasons": [["source.quote", 2]], "recent": [{ "kind": "attempt", "target": "IDCAMS/DELETE", "outcome": "accepted" }] }),
        ]
    }

    fn screen(loops: &[Value], selected: usize) -> String {
        let mut terminal = Terminal::new(TestBackend::new(120, 32)).unwrap();
        let mut state = AppState::new();
        state.scroll_offset = selected;
        terminal
            .draw(|f| render_loops(f, f.area(), &state, loops))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn every_loop_is_a_row_and_the_selected_one_shows_its_refusals() {
        let out = screen(&sample(), 1);
        assert!(out.contains("Accept rate"));
        assert!(out.contains("Qwen work lanes"));
        assert!(out.contains("cobolwork's utility knowledge base"));
        assert!(out.contains("Why the gate refused"));
        assert!(out.contains("source.quote"));
        assert!(out.contains("IDCAMS/DELETE"));
    }

    #[test]
    fn a_lane_shows_its_gate_pass_rate_and_buffer() {
        let out = screen(&sample(), 0);
        assert!(out.contains("44%"));
        assert!(out.contains("ready 7 jobs"));
    }

    #[test]
    fn no_loop_reported_says_where_loops_write() {
        assert!(screen(&[], 0).contains("No loop has reported yet"));
    }

    #[test]
    fn snapshots_are_read_lane_first_and_a_broken_file_is_skipped() {
        let dir = std::env::temp_dir().join(format!("veld-loops-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for v in sample().iter().rev() {
            fs::write(
                dir.join(format!("{}.json", v["id"].as_str().unwrap())),
                v.to_string(),
            )
            .unwrap();
        }
        fs::write(dir.join("broken.json"), "{ not json").unwrap();
        let ids: Vec<String> = read_from(&dir).iter().map(|v| text(v, "id")).collect();
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(ids, vec!["qwen-lane", "cobolwork-utility"]);
    }
}
