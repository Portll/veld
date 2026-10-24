//! Static-docs hosting — serves the mdBook output (`docs/book/`) at `/docs/`
//! on the public router so the TUI (and direct browser access) can pull up the
//! sidecar without leaving veld.
//!
//! Resolution order for the docs directory:
//!   1. `VELD_DOCS_DIR` env var (absolute or repo-relative path).
//!   2. `docs/book/` relative to the veld process CWD.
//!
//! When the resolved directory does not exist, we mount a single fallback page
//! that explains how to build the docs rather than returning a bare 404.
//!
//! Security note: the docs site is public by construction — it documents the
//! project, contains no per-tenant data, and never reads `?user_id=`. This
//! handler does NOT violate the `public_router_has_no_per_user_handlers`
//! invariant.

use axum::{
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use std::path::PathBuf;
use tower_http::services::ServeDir;

#[cfg(feature = "embedded-docs")]
use axum::extract::Path as AxumPath;
#[cfg(feature = "embedded-docs")]
use rust_embed::Embed;

/// Compile-time embed of `docs/book/` — populated when the `embedded-docs`
/// feature is enabled. The folder is relative to the workspace root at
/// build time; the macro fails loudly if it's missing, which is the right
/// behaviour for release builds (CI must run `mdbook build docs` before
/// compiling).
///
/// In debug builds, `rust-embed` reads from the filesystem at request time
/// rather than baking bytes in — iteration on docs content stays fast.
#[cfg(feature = "embedded-docs")]
#[derive(Embed)]
#[folder = "docs/book/"]
struct EmbeddedDocs;

/// Resolve the docs directory from env or the default repo-relative path.
pub fn docs_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VELD_DOCS_DIR") {
        if !d.trim().is_empty() {
            return PathBuf::from(d);
        }
    }
    PathBuf::from("docs").join("book")
}

/// Friendly placeholder served when the docs directory is missing or the
/// requested file is not found inside it. Plain HTML — no external assets,
/// renders cleanly even without CSS.
async fn docs_not_built() -> impl IntoResponse {
    let resolved = docs_dir();
    let body = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Veld docs — not built</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 4rem auto; padding: 0 1.5rem; color: #222; line-height: 1.5; }}
  code, pre {{ background: #f4f4f5; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  pre {{ padding: 0.75rem 1rem; overflow-x: auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
  .muted {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Docs not built yet</h1>
<p>Veld is configured to serve the static mdBook output from <code>{resolved}</code>, but no <code>index.html</code> was found there.</p>
<p>Build the docs once:</p>
<pre>cargo install --locked mdbook mdbook-mermaid mdbook-toc
mdbook build docs</pre>
<p>Then reload this page. The docs will keep being served automatically on every restart of <code>veld server</code> as long as <code>docs/book/</code> exists.</p>
<p class="muted">Override the directory with the <code>VELD_DOCS_DIR</code> environment variable. Public site (built by CI): <a href="https://portll.github.io/veld/">portll.github.io/veld</a>.</p>
</body>
</html>"#,
        resolved = escape_html(&resolved.display().to_string()),
    );
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(body),
    )
        .into_response()
}

/// Build the `/docs` route subtree. Returns a `Router<S>` for any state type
/// `S` so the caller can merge it into a stateful public router.
///
/// Resolution priority (highest first):
///   1. Filesystem at `docs_dir()` — used in dev and when the operator wants
///      to override embedded content.
///   2. `embedded-docs` feature, when enabled — serves from `EmbeddedDocs`.
///   3. The "not built" placeholder.
///
/// The router always exists — never a bare 404 for the `/docs` root.
pub fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let dir = docs_dir();
    if dir.is_dir() {
        // ServeDir serves index.html for "/" and falls back to the
        // appropriate not-found handler when a requested file is missing.
        let serve = ServeDir::new(&dir)
            .append_index_html_on_directories(true)
            .not_found_service(get(docs_not_built));
        return Router::new().nest_service("/docs", serve);
    }

    // Filesystem absent — fall through to embedded bytes when the feature is
    // on, otherwise to the placeholder. Either way, the route surface is the
    // same: `/docs` plus a catch-all for `/docs/<path>`.
    #[cfg(feature = "embedded-docs")]
    {
        Router::new()
            .route("/docs", get(embedded_index))
            .route("/docs/", get(embedded_index))
            .route("/docs/{*rest}", get(embedded_file))
    }
    #[cfg(not(feature = "embedded-docs"))]
    {
        Router::new()
            .route("/docs", get(docs_not_built))
            .route("/docs/{*rest}", get(docs_not_built))
    }
}

#[cfg(feature = "embedded-docs")]
async fn embedded_index() -> impl IntoResponse {
    serve_embedded("index.html").into_response()
}

#[cfg(feature = "embedded-docs")]
async fn embedded_file(AxumPath(rest): AxumPath<String>) -> impl IntoResponse {
    // `rest` is everything after `/docs/`. Strip a leading slash defensively.
    let key = rest.trim_start_matches('/');
    // Directory-style requests (no extension) get the directory's index.
    let lookup = if key.is_empty() || key.ends_with('/') {
        format!("{key}index.html")
    } else if !key.contains('.') {
        format!("{key}/index.html")
    } else {
        key.to_string()
    };
    serve_embedded(&lookup).into_response()
}

#[cfg(feature = "embedded-docs")]
fn serve_embedded(key: &str) -> axum::response::Response {
    match EmbeddedDocs::get(key) {
        Some(file) => {
            let mime = file.metadata.mimetype().to_string();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                file.data.into_owned(),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            Html(embedded_not_found_body(key)),
        )
            .into_response(),
    }
}

/// The 404 page for a path the embedded bundle lacks. The path comes from the request URL, so it is
/// escaped: unescaped, `/docs/x.html"><script>…` ran on veld's origin.
#[cfg_attr(not(feature = "embedded-docs"), allow(dead_code))]
fn embedded_not_found_body(key: &str) -> String {
    format!(
        "<!doctype html><meta charset=utf-8><title>Veld docs — 404</title>\
         <p style=\"font-family: system-ui, sans-serif; max-width: 40em; margin: 4rem auto\">\
         The path <code>{key}</code> is not present in the embedded docs bundle. \
         If you're running a custom build, set <code>VELD_DOCS_DIR</code> to a directory \
         produced by <code>mdbook build docs</code>.</p>",
        key = escape_html(key),
    )
}

/// Text made safe to place in HTML element content or a quoted attribute.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Public-facing URL for the docs root, relative to the server's bind address.
/// Used by the TUI to construct `http://<host>:<port>/docs/`.
pub const DOCS_URL_PATH: &str = "/docs/";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_requested_path_is_escaped_in_the_embedded_404_page() {
        let body = embedded_not_found_body("x.html\"><script>alert(document.domain)</script>");
        assert!(
            !body.contains("<script>"),
            "markup from the URL must not reach the page: {body}"
        );
        assert!(body.contains(
            "<code>x.html&quot;&gt;&lt;script&gt;alert(document.domain)&lt;/script&gt;</code>"
        ));
    }

    #[test]
    fn escaping_covers_the_five_html_metacharacters_and_nothing_else() {
        assert_eq!(
            escape_html(r#"<a href="x" title='y'>&</a>"#),
            "&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;"
        );
        assert_eq!(escape_html("guide/intro.html"), "guide/intro.html");
    }
}
