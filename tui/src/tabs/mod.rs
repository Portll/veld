//! Tab modules for the Veld TUI.
//!
//! Each submodule here owns a single tab's data refresh logic and its
//! render entry point. The dispatching layer in `widgets.rs::render_main`
//! routes to the right `render_*` per `ViewMode`.

pub mod git;
