#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

//! Veld Monitor — tray + window GUI over [`veld_status_core`].

mod app;
mod tray;

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,veld_monitor=info")),
        )
        .try_init();

    app::run();
}
