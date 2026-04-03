//! Unified Veld binary - MCP server + Claude Code hooks
//!
//! Usage:
//!   shodh serve              - Run as MCP server (stdio transport)
//!   shodh hook session-start - Output session start hook JSON
//!   shodh hook prompt <msg>  - Output prompt submit hook JSON
//!
//! Both modes use the same core memory functionality, ready for future MCP push.

pub mod client;
pub mod tools;
pub mod types;

use anyhow::Result;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    model::{
        Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
    },
    tool_handler, ServerHandler, ServiceExt,
};
use std::sync::Arc;

use client::{AsyncApiClient, BlockingApiClient};
use types::*;

// =============================================================================
// MCP SERVER
// =============================================================================

#[derive(Debug, Clone)]
pub struct ShodhMcpServer {
    pub(crate) client: Arc<AsyncApiClient>,
    tool_router: ToolRouter<Self>,
}

impl ShodhMcpServer {
    pub fn new(api_url: String, api_key: String, user_id: String) -> Self {
        // Delegate to tools::create which has access to the #[tool_router]-generated
        // associated function within the same impl block.
        Self::create(api_url, api_key, user_id)
    }
}

#[tool_handler]
impl ServerHandler for ShodhMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Veld Memory - persistent cognitive memory with causal reasoning. \
                 Use proactive_context at session start to surface relevant memories. \
                 Use remember to store decisions, learnings, errors. \
                 Use recall to search memories. \
                 Use lineage_trace to understand 'why' - trace causal chains backward/forward. \
                 Use lineage_link to explicitly connect cause→effect memories. \
                 Use lineage_confirm/reject to improve inference accuracy."
                    .to_string(),
            ),
        }
    }
}

// =============================================================================
// PUBLIC ENTRY POINT
// =============================================================================

/// Run the MCP server over stdio transport.
pub async fn run_mcp_server(api_url: String, api_key: String, user_id: String) -> Result<()> {
    eprintln!("Starting Veld MCP server...");
    eprintln!("  API URL: {}", api_url);
    eprintln!("  User ID: {}", user_id);

    let server = ShodhMcpServer::new(api_url, api_key, user_id);
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

// =============================================================================
// HOOK OUTPUT
// =============================================================================

fn output_hook(event_name: &str, context: &str) {
    let output = HookOutput {
        hook_specific_output: HookSpecificOutput {
            hook_event_name: event_name.to_string(),
            additional_context: context.to_string(),
        },
    };
    println!("{}", serde_json::to_string(&output).unwrap());
}

// =============================================================================
// HOOK HANDLERS
// =============================================================================

pub fn handle_session_start(api_url: &str, api_key: &str, user_id: &str, project_dir: Option<&str>) {
    let client = BlockingApiClient::new(api_url.to_string(), api_key.to_string());

    let dir_name = project_dir
        .and_then(|p| std::path::Path::new(p).file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get proactive context
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: format!("Starting session in {dir_name}"),
            max_results: 3,
            auto_ingest: false,
        },
    );

    // Get pending todos
    let todos_result: Result<ListTodosResponse> = client.post(
        "/api/todos",
        &ListTodosRequest {
            user_id: user_id.to_string(),
            status: vec!["todo".to_string(), "in_progress".to_string()],
        },
    );

    // Build context string
    let mut context_parts = vec!["## Veld Memory Context Restored\n".to_string()];

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("### Relevant Memories:".to_string());
            for mem in ctx.memories.iter().take(3) {
                context_parts.push(format!(
                    "- [{}] {}: {}",
                    mem.memory_type,
                    &mem.id[..8.min(mem.id.len())],
                    mem.content.chars().take(200).collect::<String>()
                ));
            }
            context_parts.push(String::new());
        }
    }

    if let Ok(todos) = todos_result {
        let in_progress: Vec<_> = todos
            .todos
            .iter()
            .filter(|t| t.status == "in_progress")
            .collect();
        let pending: Vec<_> = todos.todos.iter().filter(|t| t.status == "todo").collect();

        if !in_progress.is_empty() || !pending.is_empty() {
            context_parts.push("### Pending Todos:".to_string());

            if !in_progress.is_empty() {
                context_parts.push("**In Progress:**".to_string());
                for todo in in_progress.iter().take(5) {
                    context_parts.push(format!("- ⏳ {}", todo.content));
                }
            }

            if !pending.is_empty() {
                context_parts.push("**Todo:**".to_string());
                for todo in pending.iter().take(5) {
                    let priority = todo.priority.as_deref().unwrap_or("");
                    let prefix = match priority {
                        "urgent" => "🔴",
                        "high" => "🟠",
                        "medium" => "🟡",
                        _ => "⚪",
                    };
                    context_parts.push(format!("- {} {}", prefix, todo.content));
                }
            }
        }
    }

    output_hook("SessionStart", &context_parts.join("\n"));
}

pub fn handle_prompt_submit(api_url: &str, api_key: &str, user_id: &str, message: &str) {
    let client = BlockingApiClient::new(api_url.to_string(), api_key.to_string());

    // Get proactive context based on user message
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: message.to_string(),
            max_results: 5,
            auto_ingest: true, // Store the context for implicit feedback
        },
    );

    let mut context_parts = Vec::new();

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("## Relevant Memories (auto-surfaced)\n".to_string());
            for mem in ctx.memories.iter() {
                let relevance = (mem.relevance_score * 100.0) as u32;
                context_parts.push(format!(
                    "- [{}%] **{}**: {}",
                    relevance,
                    mem.memory_type,
                    mem.content.chars().take(300).collect::<String>()
                ));
            }
        }
    }

    // Only output if we have relevant context
    if !context_parts.is_empty() {
        output_hook("UserPromptSubmit", &context_parts.join("\n"));
    } else {
        // Output empty hook (no context to inject)
        output_hook("UserPromptSubmit", "");
    }
}

/// Launch Claude Code with Veld Cortex proxy
pub async fn handle_claude_launch(port: u16, args: Vec<String>) -> Result<()> {
    let server_url = format!("http://127.0.0.1:{port}");

    // Check if server is running
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_url = format!("{server_url}/health");
    let server_running = client.get(&health_url).send().await.is_ok();

    if !server_running {
        eprintln!("🧠 Starting Veld memory server on port {port}...");

        // Start server in background
        let exe_path = std::env::current_exe()?;
        let server_binary = exe_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot find executable directory"))?
            .join("veld");

        #[cfg(windows)]
        let server_binary = server_binary.with_extension("exe");

        if !server_binary.exists() {
            // Try finding in PATH
            eprintln!("⚠️  veld not found at {:?}", server_binary);
            eprintln!("   Please ensure veld is installed and in PATH");
            std::process::exit(1);
        }

        let mut cmd = std::process::Command::new(&server_binary);
        cmd.env("SHODH_PORT", port.to_string());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0); // Detach from parent
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            const DETACHED_PROCESS: u32 = 0x00000008;
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }

        #[allow(clippy::zombie_processes)] // Intentionally detached background server
        cmd.spawn().expect("Failed to start veld");

        // Wait for server to be ready
        eprintln!("   Waiting for server to be ready...");
        let mut server_ready = false;
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if client.get(&health_url).send().await.is_ok() {
                eprintln!("   ✓ Server ready");
                server_ready = true;
                break;
            }
        }
        if !server_ready {
            eprintln!("   ✗ Server failed to start within 3 seconds on port {port}");
            std::process::exit(1);
        }
    } else {
        eprintln!("🧠 Veld memory server already running on port {port}");
    }

    // Launch claude with ANTHROPIC_API_BASE pointing to Cortex proxy
    eprintln!("🚀 Launching Claude Code with Veld Cortex...");
    eprintln!("   ANTHROPIC_API_BASE={}", server_url);
    eprintln!();

    let mut claude_cmd = std::process::Command::new("claude");
    claude_cmd.env("ANTHROPIC_API_BASE", &server_url);
    claude_cmd.args(&args);

    // Replace current process with claude
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = claude_cmd.exec();
        eprintln!("Failed to exec claude: {}", err);
        std::process::exit(1);
    }

    #[cfg(windows)]
    {
        // On Windows, npm-installed commands need cmd /c to resolve .cmd wrappers
        let mut cmd = std::process::Command::new("cmd");
        cmd.arg("/c").arg("claude");
        cmd.env("ANTHROPIC_API_BASE", &server_url);
        cmd.args(&args);
        let status = cmd.status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}
