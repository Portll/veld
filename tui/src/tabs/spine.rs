//! Spine tab: Spine installed from its GitHub releases and managed through `veldctl spine`, the
//! same lib/spine.cjs Veld Desktop uses. Keys: o open · i install/update · u uninstall (y confirms,
//! Y proceeds even with active sessions, n cancels) · r refresh. Actions run off the UI thread.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use ratatui::prelude::*;
use ratatui::widgets::*;
use serde_json::Value;

use crate::types::AppState;

const ACCENT: Color = Color::Rgb(130, 200, 255);
const GOOD: Color = Color::Rgb(140, 220, 140);
const BAD: Color = Color::Rgb(240, 130, 130);
const WARN: Color = Color::Rgb(232, 163, 61);
const REFRESH: Duration = Duration::from_secs(600);

#[derive(Default)]
struct Tab {
    status: Option<Value>,
    error: Option<String>,
    busy: Option<&'static str>,
    last: Option<String>,
    confirm: bool,
    fetched: Option<Instant>,
}

static TAB: OnceLock<Mutex<Tab>> = OnceLock::new();

fn tab() -> std::sync::MutexGuard<'static, Tab> {
    TAB.get_or_init(|| Mutex::new(Tab::default())).lock().unwrap_or_else(|p| p.into_inner())
}

fn veldctl() -> PathBuf {
    std::env::var_os("VELDCTL")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/bin/veldctl")).filter(|p| p.exists()))
        .unwrap_or_else(|| PathBuf::from("veldctl"))
}

fn call(args: &[&str]) -> Result<Value, String> {
    let out = Command::new(veldctl()).arg("spine").args(args).arg("--json").output().map_err(|e| format!("veldctl: {e}"))?;
    if out.status.success() {
        let text = String::from_utf8_lossy(&out.stdout);
        serde_json::from_str(text.trim().lines().last().unwrap_or("")).map_err(|e| format!("veldctl output: {e}"))
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().trim_start_matches("error:").trim().to_string())
    }
}

fn refresh_now() {
    let r = call(&["status"]);
    let mut t = tab();
    match r {
        Ok(v) => {
            t.status = Some(v);
            t.error = None;
        }
        Err(e) => t.error = Some(e),
    }
}

pub fn refresh() {
    tab().fetched = Some(Instant::now());
    std::thread::spawn(refresh_now);
}

pub fn action(sub: &'static str, force: bool) {
    {
        let mut t = tab();
        if t.busy.is_some() {
            return;
        }
        t.busy = Some(sub);
        t.confirm = false;
    }
    std::thread::spawn(move || {
        let mut args = vec![sub];
        if force {
            args.push("--force");
        }
        let msg = match call(&args) {
            Ok(v) => {
                let steps = v.get("steps").and_then(Value::as_array).map(|s| s.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" · "));
                format!("{sub}: {}", steps.unwrap_or_else(|| "done".into()))
            }
            Err(e) => format!("{sub} failed: {e}"),
        };
        {
            let mut t = tab();
            t.busy = None;
            t.last = Some(msg);
        }
        refresh_now();
    });
}

pub fn arm_uninstall() {
    tab().confirm = true;
}

pub fn confirming() -> bool {
    tab().confirm
}

pub fn cancel() {
    tab().confirm = false;
}

fn s<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(v, |cur, k| cur.get(*k))
}

fn kv(k: &str, v: String, colour: Color) -> Line<'static> {
    Line::from(vec![Span::styled(format!("  {k:<10}"), Style::default().fg(Color::DarkGray)), Span::styled(v, Style::default().fg(colour))])
}

pub fn render(f: &mut Frame, area: Rect, _state: &AppState) {
    if tab().fetched.map_or(true, |t| t.elapsed() > REFRESH) {
        refresh();
    }
    let t = tab();
    let mut lines = vec![Line::from(Span::styled("  Spine — installed from GitHub releases, managed by veld", Style::default().fg(ACCENT).bold())), Line::default()];
    match (&t.status, &t.error) {
        (Some(st), _) => {
            let str_at = |p: &[&str]| s(st, p).and_then(Value::as_str).map(str::to_string);
            let runtime = str_at(&["runtime"]);
            lines.push(kv("installed", runtime.clone().unwrap_or_else(|| "no".into()), if runtime.is_some() { GOOD } else { WARN }));
            lines.push(kv("app", format!("{} ({})", str_at(&["app"]).unwrap_or_else(|| "none".into()), str_at(&["appPath"]).unwrap_or_default()), Color::Gray));
            let mcp = match (s(st, &["mcp", "registered"]).and_then(Value::as_bool), s(st, &["mcp", "managed"]).and_then(Value::as_bool)) {
                (Some(true), Some(true)) => ("managed by veld".to_string(), GOOD),
                (Some(true), _) => (format!("{} (not managed by veld)", str_at(&["mcp", "command"]).unwrap_or_default()), Color::Gray),
                _ => ("not registered".to_string(), WARN),
            };
            lines.push(kv("mcp", mcp.0, mcp.1));
            let latest = str_at(&["latest", "version"]).map_or_else(|| str_at(&["latestError"]).unwrap_or_else(|| "unknown".into()), |v| format!("{v} ({})", str_at(&["latest", "publishedAt"]).unwrap_or_default()));
            lines.push(kv("latest", latest, Color::Gray));
            if s(st, &["updateAvailable"]).and_then(Value::as_bool) == Some(true) {
                lines.push(kv("", "update available — press i".into(), WARN));
            }
        }
        (None, Some(e)) => lines.push(kv("status", e.clone(), BAD)),
        (None, None) => lines.push(kv("status", "reading…".into(), Color::Gray)),
    }
    lines.push(Line::default());
    if let Some(b) = t.busy {
        lines.push(kv("working", format!("{b}…"), ACCENT));
    } else if let Some(m) = &t.last {
        lines.push(kv("last", m.clone(), if m.contains("failed") { BAD } else { GOOD }));
    }
    if t.confirm {
        lines.push(Line::default());
        lines.push(kv("confirm", "uninstall Spine? y = yes · Y = even with active sessions · n = cancel".into(), WARN));
    }
    lines.push(Line::default());
    lines.push(Line::from(Span::styled("  o open · i install/update · u uninstall · r refresh", Style::default().fg(Color::DarkGray))));
    f.render_widget(Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Spine ")).wrap(Wrap { trim: false }), area);
}
