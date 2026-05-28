//! gen-http-api — parse `src/handlers/router.rs` and emit a markdown reference
//! page listing every HTTP endpoint with method, path, and handler.
//!
//! Strategy: walk the syn AST looking for chained `.route("/path", METHOD(handler))`
//! calls. The Axum router uses this pattern throughout. Non-literal paths
//! (rare) are skipped with a warning comment.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use syn::visit::Visit;
use syn::{Expr, ExprCall, ExprMethodCall, ExprPath, Lit};

use veld_docs_generators::{
    docs_src_root, generated_header, read_source, repo_root, write_output,
};

#[derive(Debug, Clone)]
struct Route {
    method: String,
    path: String,
    handler: String,
}

struct RouteCollector {
    routes: Vec<Route>,
    skipped: usize,
}

impl<'ast> Visit<'ast> for RouteCollector {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "route" && call.args.len() == 2 {
            // First arg: literal string for the path.
            let path = match &call.args[0] {
                Expr::Lit(lit) => match &lit.lit {
                    Lit::Str(s) => Some(s.value()),
                    _ => None,
                },
                _ => None,
            };

            // Second arg: METHOD(handler) call where METHOD is get/post/put/delete/patch.
            let (method, handler) = match &call.args[1] {
                Expr::Call(ExprCall { func, args, .. }) => {
                    let m = match func.as_ref() {
                        Expr::Path(ExprPath { path, .. }) => {
                            path.segments.last().map(|s| s.ident.to_string())
                        }
                        _ => None,
                    };
                    let h = args.first().and_then(|a| match a {
                        Expr::Path(ExprPath { path, .. }) => Some(
                            path.segments
                                .iter()
                                .map(|s| s.ident.to_string())
                                .collect::<Vec<_>>()
                                .join("::"),
                        ),
                        _ => None,
                    });
                    (m, h)
                }
                _ => (None, None),
            };

            match (path, method, handler) {
                (Some(p), Some(m), Some(h)) => self.routes.push(Route {
                    method: m.to_uppercase(),
                    path: p,
                    handler: h,
                }),
                _ => self.skipped += 1,
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

fn render(routes: &[Route], skipped: usize) -> String {
    let mut out = generated_header("src/handlers/router.rs", "gen-http-api");
    out.push_str("# HTTP API\n\n");
    out.push_str(&format!(
        "Veld exposes **{}** HTTP routes. Every route (except `/health/*` probes) requires API-key authentication via the `X-API-Key` header.\n\n",
        routes.len()
    ));
    if skipped > 0 {
        out.push_str(&format!(
            "> Warning: {skipped} route(s) used a non-literal path string and were skipped by the generator. \
             Inspect [src/handlers/router.rs](https://github.com/Portll/veld/blob/main/src/handlers/router.rs) directly for those.\n\n"
        ));
    }
    out.push_str("Base URL: `http://127.0.0.1:3030` (default; configurable via `VELD_BIND_ADDR`).\n\n");

    // Group by path prefix for readability: /api/X/* → "X", /health/* → "health", etc.
    let mut groups: BTreeMap<String, Vec<&Route>> = BTreeMap::new();
    for r in routes {
        let group = match r.path.strip_prefix("/api/") {
            Some(rest) => {
                let seg = rest.split('/').next().unwrap_or("misc");
                if seg.is_empty() {
                    "misc".to_string()
                } else {
                    seg.to_string()
                }
            }
            None => match r.path.strip_prefix('/') {
                Some(rest) => rest.split('/').next().unwrap_or("misc").to_string(),
                None => "misc".to_string(),
            },
        };
        groups.entry(group).or_default().push(r);
    }

    for (group, rs) in &groups {
        out.push_str(&format!("## /{}\n\n", group));
        out.push_str("| Method | Path | Handler |\n|---|---|---|\n");
        let mut sorted = rs.clone();
        sorted.sort_by(|a, b| a.path.cmp(&b.path));
        for r in sorted {
            out.push_str(&format!(
                "| `{}` | `{}` | `{}` |\n",
                r.method, r.path, r.handler
            ));
        }
        out.push('\n');
    }

    out.push_str(
        "---\n\n*Handlers live in `src/handlers/*.rs`. For request/response shapes, see the corresponding handler source or run `veld serve` and call `OPTIONS /api/...` (where supported).*\n",
    );
    out
}

fn main() -> Result<()> {
    let router_path = repo_root().join("src").join("handlers").join("router.rs");
    let source = read_source(&router_path)?;
    let file = syn::parse_file(&source)
        .with_context(|| format!("parsing {}", router_path.display()))?;

    let mut collector = RouteCollector {
        routes: Vec::new(),
        skipped: 0,
    };
    collector.visit_file(&file);

    if collector.routes.is_empty() {
        anyhow::bail!(
            "extracted zero routes from {} — the router pattern may have changed; \
             inspect the generator",
            router_path.display()
        );
    }

    let output_path = docs_src_root().join("reference").join("http-api.md");
    write_output(&output_path, &render(&collector.routes, collector.skipped))?;

    eprintln!(
        "gen-http-api: extracted {} routes ({} skipped)",
        collector.routes.len(),
        collector.skipped
    );
    Ok(())
}
