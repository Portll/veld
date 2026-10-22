//! `veld serve` (rmcp, src/mcp/) and the TypeScript server (mcp-server/index.ts) expose one
//! tool set. The TypeScript half is read from its ListTools registration; the Rust half is
//! read from the live router, so a tool that is declared but not routed fails as well as one
//! that exists on one side only. `mcp-server/tests/tool-parity.test.ts` is the same check run
//! from the other side's toolchain.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Tool names inside the `ListToolsRequestSchema` handler: each registration is a
/// `name: "<tool>",` line of its own; schema properties called `name` are objects, not strings.
fn typescript_tool_names() -> BTreeSet<String> {
    let source = read(&repo_root().join("mcp-server/index.ts"));
    let start = source
        .find("setRequestHandler(ListToolsRequestSchema")
        .expect("mcp-server/index.ts registers ListToolsRequestSchema");
    let region = &source[start..];
    let end = region[1..]
        .find("setRequestHandler(")
        .map(|i| i + 1)
        .unwrap_or(region.len());
    region[..end]
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("name: \"")
                .and_then(|rest| rest.strip_suffix("\","))
                .map(str::to_string)
        })
        .collect()
}

/// Every `#[tool(` attribute in the Rust server is followed by the `fn <name>(` it decorates.
fn rust_source_tool_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for file in ["tools.rs", "tools_memory.rs", "tools_work.rs"] {
        let source = read(&repo_root().join("src/mcp").join(file));
        let mut rest = source.as_str();
        while let Some(at) = rest.find("#[tool(") {
            rest = &rest[at + "#[tool(".len()..];
            let fn_at = rest
                .find("fn ")
                .expect("a #[tool] attribute decorates a fn");
            let ident: String = rest[fn_at + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.insert(ident);
        }
    }
    names
}

fn live_router_tool_names() -> BTreeSet<String> {
    let server = veld::mcp::VeldMcpServer::new(
        "http://127.0.0.1:1".to_string(),
        "parity-test-key".to_string(),
        "parity".to_string(),
    );
    server.tool_names().into_iter().collect()
}

#[tokio::test]
async fn typescript_and_rust_servers_expose_the_same_tools() {
    let typescript = typescript_tool_names();
    let rust = live_router_tool_names();
    let only_typescript: Vec<&String> = typescript.difference(&rust).collect();
    let only_rust: Vec<&String> = rust.difference(&typescript).collect();
    assert!(
        only_typescript.is_empty() && only_rust.is_empty(),
        "tools only in mcp-server/index.ts: {only_typescript:?}; tools only in veld serve: {only_rust:?}"
    );
    assert!(
        typescript.len() >= 60,
        "expected the shared tool set to hold at least 60 tools, found {}",
        typescript.len()
    );
}

#[tokio::test]
async fn every_declared_rust_tool_is_routed() {
    let declared = rust_source_tool_names();
    let routed = live_router_tool_names();
    let unrouted: Vec<&String> = declared.difference(&routed).collect();
    let undeclared: Vec<&String> = routed.difference(&declared).collect();
    assert!(
        unrouted.is_empty() && undeclared.is_empty(),
        "declared but not routed: {unrouted:?}; routed but not found in src/mcp: {undeclared:?}"
    );
}
