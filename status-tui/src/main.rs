//! `veld-status` — a lightweight terminal status monitor for a Veld server.
//!
//! Reads the API key from `$VELD_API_KEY` or the platform config file, picks the
//! first user from `GET /api/users`, then renders a 2x3 grid of status panels
//! refreshed from [`veld_status_core`].

use std::io;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Terminal;
use veld_status_core::{StatusClient, StatusClientConfig};
use veld_status_widget::{
    activity_tail, graph_panel, server_panel, sessions_panel, tiers_panel, todos_panel,
};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:3030";
const DEFAULT_DEV_API_KEY: &str = "sk-veld-dev-local-testing-key";

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(io::stderr)
        .try_init();

    let base_url = std::env::var("VELD_SERVER_URL")
        .or_else(|_| std::env::var("VELD_API_URL"))
        .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());

    let api_key = resolve_api_key();
    let user_id = pick_user_id(&base_url, &api_key).await?;

    let config = StatusClientConfig::new(base_url, api_key, user_id)
        .with_refresh_interval(Duration::from_secs(1))
        .with_request_timeout(Duration::from_secs(5));
    let client = StatusClient::spawn(config).context("could not spawn status client")?;

    run_tui(client.snapshot()).await
}

fn resolve_api_key() -> String {
    if let Ok(key) = std::env::var("VELD_API_KEY") {
        if !key.trim().is_empty() {
            return key;
        }
    }
    let path = veld_status_core::default_config_path();
    if path.exists() {
        if let Ok(key) = veld_status_core::load_api_key_from(&path) {
            return key;
        }
    }
    DEFAULT_DEV_API_KEY.to_string()
}

async fn pick_user_id(base_url: &str, api_key: &str) -> Result<String> {
    if let Ok(user) = std::env::var("VELD_USER_ID") {
        if !user.trim().is_empty() {
            return Ok(user);
        }
    }
    let users = StatusClient::list_users(base_url, api_key, Duration::from_secs(5))
        .await
        .with_context(|| format!("could not list users at {}", base_url))?;
    users
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("server has no users; set VELD_USER_ID or create one"))
}

async fn run_tui(
    snapshot: std::sync::Arc<parking_lot::RwLock<veld_status_core::StatusSnapshot>>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, SetTitle("veld-status"))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = render_loop(&mut terminal, &snapshot).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

async fn render_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    snapshot: &std::sync::Arc<parking_lot::RwLock<veld_status_core::StatusSnapshot>>,
) -> Result<()> {
    let tick = Duration::from_millis(250);
    loop {
        {
            let snap = snapshot.read().clone();
            terminal.draw(|f| draw(f, &snap))?;
        }

        if event::poll(tick)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('Q') => return Ok(()),
                        _ => {}
                    }
                }
            }
        }
    }
}

fn draw(f: &mut ratatui::Frame, snap: &veld_status_core::StatusSnapshot) {
    let area = f.area();

    // Row layout: top row (server/memory/sessions), middle (todos/graph), bottom (activity tail)
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(rows[0]);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    server_panel(f, top[0], snap);
    tiers_panel(f, top[1], snap);
    sessions_panel(f, top[2], snap);
    todos_panel(f, middle[0], snap);
    graph_panel(f, middle[1], snap);
    activity_tail(f, rows[2], snap);

    // Footer hint
    let hint = format!(
        " user: {}    q/Esc: quit ",
        if snap.user_id.is_empty() {
            "?"
        } else {
            &snap.user_id
        }
    );
    let footer = ratatui::widgets::Paragraph::new(hint).alignment(ratatui::layout::Alignment::Left);
    f.render_widget(footer, rows[3]);
}
