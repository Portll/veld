//! Stateless ratatui widgets over [`veld_status_core::StatusSnapshot`].
//!
//! Each public function takes a `Frame`, a target `Rect`, and a snapshot reference,
//! and renders one panel. Composition is the caller's responsibility — this crate
//! does not own layout.

use chrono::Utc;
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};
use veld_status_core::{
    ActivityEntry, ContextSession, GraphStats, ReachState, ServerHealth, StatusSnapshot,
    TierStats, TodoStats,
};

const ACCENT: Color = Color::Rgb(255, 140, 50);
const GREEN: Color = Color::Rgb(120, 200, 120);
const YELLOW: Color = Color::Rgb(230, 200, 90);
const RED: Color = Color::Rgb(220, 90, 90);
const DIM: Color = Color::Rgb(120, 120, 140);
const BG_BORDER: Color = Color::Rgb(60, 60, 80);

fn block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BG_BORDER))
        .title(Span::styled(
            format!(" {} ", title),
            Style::default().fg(DIM),
        ))
}

fn reach_color(state: &ReachState) -> Color {
    match state {
        ReachState::Reachable => GREEN,
        ReachState::Unhealthy(_) => YELLOW,
        ReachState::Unreachable(_) => RED,
        ReachState::Unknown => DIM,
    }
}

fn reach_label(state: &ReachState) -> &'static str {
    match state {
        ReachState::Reachable => "online",
        ReachState::Unhealthy(_) => "unhealthy",
        ReachState::Unreachable(_) => "offline",
        ReachState::Unknown => "probing",
    }
}

fn humanize_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

/// Top panel: reachability dot, version, uptime, RTT, storage backend.
pub fn server_panel(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let server = &snap.server;
    let dot_color = reach_color(&server.state);
    let label = reach_label(&server.state);

    let mut lines = vec![Line::from(vec![
        Span::styled("● ", Style::default().fg(dot_color).add_modifier(Modifier::BOLD)),
        Span::styled(
            label,
            Style::default().fg(dot_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(&snap.base_url, Style::default().fg(Color::White)),
    ])];

    let rtt = server
        .rtt_ms
        .map(|r| format!("{} ms", r))
        .unwrap_or_else(|| "—".to_string());
    let uptime = server
        .uptime_secs()
        .map(humanize_uptime)
        .unwrap_or_else(|| "—".to_string());
    let version = server.version.as_deref().unwrap_or("—");
    let backend = server.effective_storage_backend.as_deref().unwrap_or("—");

    lines.push(field_line("version", version));
    lines.push(field_line("uptime", &uptime));
    lines.push(field_line("rtt", &rtt));
    lines.push(field_line("storage", backend));
    if let (Some(total), Some(cached)) = (server.users_count, server.users_in_cache) {
        lines.push(field_line("users", &format!("{} ({} cached)", total, cached)));
    }
    if let ReachState::Unhealthy(msg) | ReachState::Unreachable(msg) = &server.state {
        lines.push(Line::from(Span::styled(
            truncate(msg, area.width.saturating_sub(4) as usize),
            Style::default().fg(RED),
        )));
    }

    let p = Paragraph::new(lines)
        .block(block("Server"))
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

/// Memory tier breakdown + index-health stripe.
pub fn tiers_panel(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let tiers = &snap.memory;
    render_tiers(f, area, tiers);
}

fn render_tiers(f: &mut Frame, area: Rect, tiers: &TierStats) {
    let total = tiers.total.max(1);
    let pct = |n: u64| ((n as f64 / total as f64) * 100.0).round() as u16;
    let health_color = if tiers.index_healthy { GREEN } else { RED };
    let health_label = if tiers.index_healthy {
        "index healthy"
    } else {
        "index lag"
    };

    let lines = vec![
        field_line("total", &format_count(tiers.total)),
        field_line(
            "working",
            &format!("{} ({}%)", format_count(tiers.working), pct(tiers.working)),
        ),
        field_line(
            "session",
            &format!("{} ({}%)", format_count(tiers.session), pct(tiers.session)),
        ),
        field_line(
            "long-term",
            &format!(
                "{} ({}%)",
                format_count(tiers.long_term),
                pct(tiers.long_term)
            ),
        ),
        field_line("retrievals", &format_count(tiers.total_retrievals)),
        Line::from(vec![
            Span::styled("● ", Style::default().fg(health_color)),
            Span::styled(health_label, Style::default().fg(health_color)),
            Span::styled(
                format!("  vec={}/{}", tiers.vector_index, tiers.total),
                Style::default().fg(DIM),
            ),
        ]),
    ];

    let p = Paragraph::new(lines).block(block("Memory"));
    f.render_widget(p, area);
}

/// Active Claude Code sessions with a token-usage gauge each.
pub fn sessions_panel(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let outer = block("Sessions");
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    if snap.sessions.is_empty() {
        let p = Paragraph::new(Line::from(Span::styled(
            "no active Claude Code sessions",
            Style::default().fg(DIM),
        )))
        .alignment(Alignment::Left);
        f.render_widget(p, inner);
        return;
    }

    // Each session gets two rows: header + gauge. Skip overflow gracefully.
    let row_h: u16 = 2;
    let max_sessions = (inner.height / row_h) as usize;
    for (i, session) in snap.sessions.iter().take(max_sessions).enumerate() {
        let y = inner.y + (i as u16) * row_h;
        let row = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: row_h,
        };
        render_session(f, row, session);
    }
}

fn render_session(f: &mut Frame, area: Rect, session: &ContextSession) {
    if area.height < 2 {
        return;
    }
    let header_area = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: 1,
    };
    let gauge_area = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: 1,
    };

    let pct = session.percent_used.min(100);
    let gauge_color = if pct < 50 {
        GREEN
    } else if pct < 80 {
        YELLOW
    } else {
        RED
    };

    let model = session.model.as_deref().unwrap_or("?");
    let task = session.current_task.as_deref().unwrap_or("");
    let session_label = if session.session_id.len() > 8 {
        &session.session_id[..8]
    } else {
        &session.session_id
    };

    let header = Line::from(vec![
        Span::styled(
            format!("{}  ", session_label),
            Style::default().fg(DIM),
        ),
        Span::styled(model.to_string(), Style::default().fg(ACCENT)),
        Span::styled("  ", Style::default()),
        Span::styled(
            truncate(task, area.width as usize),
            Style::default().fg(Color::White),
        ),
    ]);
    f.render_widget(Paragraph::new(header), header_area);

    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(gauge_color).bg(Color::Rgb(30, 30, 40)))
        .ratio((pct as f64) / 100.0)
        .label(format!(
            "{}% {}k/{}k",
            pct,
            session.tokens_used / 1000,
            session.tokens_budget / 1000
        ));
    f.render_widget(gauge, gauge_area);
}

/// Live activity tail. Newest at top.
pub fn activity_tail(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let outer = block("Activity");
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    if snap.recent.is_empty() {
        let p = Paragraph::new(Line::from(Span::styled(
            "waiting for events…",
            Style::default().fg(DIM),
        )));
        f.render_widget(p, inner);
        return;
    }

    let items: Vec<ListItem> = snap
        .recent
        .iter()
        .take(inner.height as usize)
        .map(|e| ListItem::new(render_activity_line(e, inner.width as usize)))
        .collect();
    f.render_widget(List::new(items), inner);
}

/// Optional fifth panel: todo counts. Useful in monitor when there's vertical room.
pub fn todos_panel(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let todos = &snap.todos;
    let lines = vec![
        field_line("total", &todos.total.to_string()),
        field_line("in-progress", &todos.in_progress.to_string()),
        field_line("blocked", &todos.blocked.to_string()),
        field_line(
            "overdue",
            &if todos.overdue > 0 {
                format!("{}", todos.overdue)
            } else {
                "0".to_string()
            },
        ),
        field_line("done", &todos.done.to_string()),
    ];
    let p = Paragraph::new(lines).block(block("Todos"));
    f.render_widget(p, area);
}

/// Optional sixth panel: graph counts.
pub fn graph_panel(f: &mut Frame, area: Rect, snap: &StatusSnapshot) {
    let g = &snap.graph;
    let lines = vec![
        field_line("entities", &format_count(g.entities)),
        field_line("relations", &format_count(g.relationships)),
    ];
    let p = Paragraph::new(lines).block(block("Graph"));
    f.render_widget(p, area);
}

fn render_activity_line(e: &ActivityEntry, width: usize) -> Line<'static> {
    let color = event_color(&e.event_type);
    let icon = event_icon(&e.event_type);
    let elapsed = (Utc::now() - e.timestamp).num_seconds();
    let time = if elapsed < 0 {
        e.timestamp.format("%H:%M").to_string()
    } else if elapsed < 60 {
        format!("{}s", elapsed)
    } else if elapsed < 3600 {
        format!("{}m", elapsed / 60)
    } else {
        format!("{}h", elapsed / 3600)
    };
    let preview = e.preview.clone().unwrap_or_default();
    let mtype = e.memory_type.clone().unwrap_or_default();
    let mtype_segment = if mtype.is_empty() {
        String::new()
    } else {
        format!("[{}] ", mtype)
    };
    let trailing = truncate(
        &format!("{}{}", mtype_segment, preview),
        width.saturating_sub(14),
    );

    Line::from(vec![
        Span::styled(
            format!("{:>4} ", time),
            Style::default().fg(DIM),
        ),
        Span::styled(
            format!("{} ", icon),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{:<14} ", truncate(&e.event_type, 14)),
            Style::default().fg(color),
        ),
        Span::styled(trailing, Style::default().fg(Color::White)),
    ])
}

fn event_color(ty: &str) -> Color {
    match ty {
        "CREATE" | "TODO_CREATE" => GREEN,
        "RETRIEVE" | "PROACTIVE_CONTEXT" => Color::Rgb(255, 200, 150),
        "DELETE" | "TODO_DELETE" => RED,
        "UPDATE" | "TODO_UPDATE" => YELLOW,
        "GRAPH_UPDATE" => Color::Magenta,
        "CONSOLIDATE" => Color::Rgb(180, 200, 255),
        "STRENGTHEN" | "TODO_COMPLETE" => Color::Rgb(200, 255, 200),
        "DECAY" => Color::Gray,
        "PROMOTE" => Color::LightYellow,
        "CONTEXT_UPDATE" => Color::Rgb(180, 220, 255),
        _ => Color::White,
    }
}

fn event_icon(ty: &str) -> &'static str {
    match ty {
        "CREATE" => "●",
        "RETRIEVE" => "◎",
        "DELETE" => "○",
        "UPDATE" => "◐",
        "GRAPH_UPDATE" => "◆",
        "CONSOLIDATE" => "⟳",
        "STRENGTHEN" => "↑",
        "DECAY" => "↓",
        "PROMOTE" => "⇧",
        "TODO_CREATE" => "□",
        "TODO_UPDATE" => "◧",
        "TODO_COMPLETE" => "☑",
        "TODO_DELETE" => "☒",
        "CONTEXT_UPDATE" => "◉",
        _ => "•",
    }
}

fn field_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{:<10}", label),
            Style::default().fg(DIM),
        ),
        Span::styled(value.to_string(), Style::default().fg(Color::White)),
    ])
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", prefix)
}

/// Re-export for downstream consumers that need to query graph counts directly.
pub fn graph_summary(snap: &StatusSnapshot) -> &GraphStats {
    &snap.graph
}

/// Re-export for downstream consumers wanting raw todo counts.
pub fn todo_summary(snap: &StatusSnapshot) -> &TodoStats {
    &snap.todos
}

/// Re-export for downstream consumers wanting raw server health.
pub fn server_summary(snap: &StatusSnapshot) -> &ServerHealth {
    &snap.server
}
