//! Dials tab — the runtime cognitive-tuning surface in the monitor.
//!
//! v1 is an informational view: it documents the dial groups and how to set them
//! (`PUT /api/dials`, MCP `set_dials`, or `VELD_*` env). Live value display and
//! in-tab toggling are a planned follow-up that hooks the TUI's async-refresh path;
//! the dials are already fully controllable via the HTTP API and MCP tools.

use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::types::AppState;

const ACCENT: Color = Color::Rgb(255, 183, 130);

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.border()))
        .title(Span::styled(
            " Dials ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let dim = Style::default().fg(Color::DarkGray);
    let head = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
    let bold = Style::default().add_modifier(Modifier::BOLD);

    let lines = vec![
        Line::from(Span::styled("Cognitive Dials — runtime tuning", head)),
        Line::from(""),
        Line::from(Span::styled(
            "Reconsolidation (plasticity) — runs in background maintenance, off the recall hot path:",
            bold,
        )),
        Line::from("  reconsolidation_enabled        master on/off (labile-window close + re-encode)"),
        Line::from("  reconsolidation_window_secs    labile window length (default 300s)"),
        Line::from("  re_encode_graph_strengthen     strengthen graph edges to retrieval-context entities"),
        Line::from("  re_encode_entity_enrich        merge new context entities into the memory"),
        Line::from("  re_encode_embedding_nudge      blend the memory embedding toward the context"),
        Line::from(""),
        Line::from(Span::styled(
            "Retrieval (fast <-> accurate) — applied per query:",
            bold,
        )),
        Line::from("  refiner_mode / use_reranking   cross-encoder / LLM refiner stage"),
        Line::from("  candidate_count / rerank_count candidate pool + rerank depth"),
        Line::from("  rrf_weights                    bm25 / vector / graph fusion weights"),
        Line::from(""),
        Line::from(Span::styled("Set via:", bold)),
        Line::from("  PUT /api/dials   ·   MCP set_dials   ·   VELD_RECONSOLIDATION_* / VELD_DIALS_* env"),
        Line::from(Span::styled(
            "  GET /api/dials   ·   MCP get_dials   — read current values",
            dim,
        )),
        Line::from(""),
        Line::from(Span::styled(
            "All reconsolidation toggles default OFF (inert until enabled).",
            dim,
        )),
        Line::from(Span::styled(
            format!("user: {}", state.current_user),
            dim,
        )),
    ];

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}
