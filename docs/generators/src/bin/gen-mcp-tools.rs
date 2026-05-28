//! gen-mcp-tools — parse `mcp-server/index.ts` and emit a markdown reference
//! page listing every MCP tool with description and schema.
//!
//! The TypeScript registrations use a consistent pattern:
//!
//!     server.registerTool(
//!       "tool_name",
//!       { description: "...", inputSchema: z.object({...}) },
//!       async (...) => { ... }
//!     )
//!
//! or via the rmcp-style:
//!
//!     name: "tool_name",
//!     description: "...",
//!     ...
//!
//! We use targeted regex to extract `name`+`description` pairs. If the count
//! drops below an expected lower-bound (45), we fail loudly.

use anyhow::Result;
use regex::Regex;

use veld_docs_generators::{
    docs_src_root, generated_header, read_source, repo_root, write_output,
};

const MIN_EXPECTED_TOOLS: usize = 45;

#[derive(Debug, Clone)]
struct McpTool {
    name: String,
    description: String,
}

fn extract_tools(source: &str) -> Vec<McpTool> {
    // Match either:
    //   name: "tool_name", ... description: "..."
    // or:
    //   registerTool("tool_name", { description: "..." }, ...)
    //
    // We do this in two passes and dedupe by name (server name "veld" is excluded).
    let pair_re =
        Regex::new(r#"name:\s*"([a-z_][a-z0-9_]*)"\s*,[\s\S]{0,400}?description:\s*"([^"]+)""#)
            .expect("regex");
    let register_re =
        Regex::new(r#"registerTool\(\s*"([a-z_][a-z0-9_]*)"\s*,[\s\S]{0,400}?description:\s*"([^"]+)""#)
            .expect("regex");

    let mut seen: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for cap in pair_re.captures_iter(source) {
        let name = cap[1].to_string();
        if name == "veld" {
            continue;
        } // server name, not a tool
        let desc = cap[2].to_string();
        seen.entry(name).or_insert(desc);
    }
    for cap in register_re.captures_iter(source) {
        let name = cap[1].to_string();
        if name == "veld" {
            continue;
        }
        let desc = cap[2].to_string();
        seen.entry(name).or_insert(desc);
    }

    seen.into_iter()
        .map(|(name, description)| McpTool { name, description })
        .collect()
}

fn render(tools: &[McpTool]) -> String {
    let mut out = generated_header("mcp-server/index.ts", "gen-mcp-tools");
    out.push_str("# MCP Tools\n\n");
    out.push_str(&format!(
        "The TypeScript MCP server (`@veld/memory-mcp`) exposes **{}** tools over the HTTP API. The Rust binary (`veld serve`) exposes the same tools via stdio MCP using `rmcp`.\n\n",
        tools.len()
    ));
    out.push_str("Tools are listed alphabetically. For full parameter schemas, see [mcp-server/index.ts](https://github.com/Portll/veld/blob/main/mcp-server/index.ts).\n\n");
    out.push_str("| Tool | Description |\n|---|---|\n");
    let mut sorted: Vec<&McpTool> = tools.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in sorted {
        // Escape pipes inside descriptions for markdown table cells.
        let safe_desc = t.description.replace('|', "\\|").replace('\n', " ");
        out.push_str(&format!("| `{}` | {} |\n", t.name, safe_desc));
    }
    out.push_str("\n---\n\n*To use these from Claude Code or VS Code Copilot, see the [Claude Code integration guide](../guides/claude-code-integration.md) or [VS Code Copilot guide](../guides/vscode-copilot.md).*\n");
    out
}

fn main() -> Result<()> {
    let index_path = repo_root().join("mcp-server").join("index.ts");
    let source = read_source(&index_path)?;
    let tools = extract_tools(&source);

    if tools.len() < MIN_EXPECTED_TOOLS {
        anyhow::bail!(
            "extracted only {} MCP tools from {}, expected at least {}. The registration pattern may have changed; inspect the generator.",
            tools.len(),
            index_path.display(),
            MIN_EXPECTED_TOOLS
        );
    }

    let output_path = docs_src_root().join("reference").join("mcp-tools.md");
    write_output(&output_path, &render(&tools))?;

    eprintln!("gen-mcp-tools: extracted {} tools", tools.len());
    Ok(())
}
