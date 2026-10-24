//! Every tool `veld serve` exposes is called once over MCP against a veld server that runs
//! inside this test, so a tool whose request body or response type drifts from its HTTP
//! handler fails here rather than on a user's first call. `tests/mcp_tool_parity.rs` checks
//! the tool NAMES against the TypeScript server; this checks that each Rust tool ANSWERS.
//!
//! The server is the protected router behind the API-key middleware on an ephemeral port,
//! with MiniLM in hash mode (no model files, no network) as in `recall_engine_parity.rs`.
//! The MCP client is rmcp's own over an in-memory duplex pipe, so the exchange is the real
//! JSON-RPC the binary speaks, not a call into the router.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{json, Value};
use tempfile::TempDir;
use veld::config::ServerConfig;
use veld::handlers::MultiUserMemoryManager;
use veld::server_runtime::ServerRuntime;

const USER: &str = "parity";
const API_KEY: &str = "sk-veld-test-mcp-roundtrip";
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Tools whose server-side precondition this test cannot meet. A refusal from the server
/// is accepted for them; a request or response that fails to encode or decode is not.
const PRECONDITION_TOOLS: &[&str] = &[
    "dataset_create",
    "dataset_list",
    "dataset_query",
    "dataset_link",
    "gap_mapper",
];

fn pin_environment(model_dir: &Path) {
    for (name, value) in [
        ("VELD_EMBEDDER", "minilm".to_string()),
        ("VELD_MODEL_PATH", model_dir.display().to_string()),
        ("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", "1".to_string()),
        ("VELD_OFFLINE", "true".to_string()),
        ("VELD_AUTO_DOWNLOAD_MODELS", "false".to_string()),
        ("VELD_EMBEDDING_API_URL", "http://127.0.0.1:9".to_string()),
        ("VELD_ONNX_THREADS", "1".to_string()),
        ("RAYON_NUM_THREADS", "1".to_string()),
        ("VELD_LAZY_CROSS_ENCODER", "1".to_string()),
        ("VELD_API_KEYS", API_KEY.to_string()),
        ("VELD_DEV_API_KEY", API_KEY.to_string()),
    ] {
        env::set_var(name, value);
    }
    for name in [
        "VELD_ENV",
        "VELD_ADMIN_API_KEY",
        "VELD_STORAGE_BACKEND",
        "VELD_REPLICATE_FROM",
    ] {
        env::remove_var(name);
    }
}

struct Live {
    client: RunningService<RoleClient, ()>,
    http: reqwest::Client,
    base_url: String,
    called: BTreeSet<String>,
}

impl Live {
    async fn call(&mut self, name: &str, arguments: Value) -> Result<String> {
        self.called.insert(name.to_string());
        let arguments = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => return Err(anyhow!("{name}: arguments must be an object, got {other}")),
        };
        let mut request = CallToolRequestParams::new(name.to_string());
        request.arguments = arguments;
        let result = tokio::time::timeout(CALL_TIMEOUT, self.client.call_tool(request))
            .await
            .with_context(|| format!("{name}: no answer within {CALL_TIMEOUT:?}"))?;
        match result {
            Ok(result) => {
                let text: String = result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .map(|t| t.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n");
                if result.is_error == Some(true) {
                    Err(anyhow!("{name} answered an error result: {text}"))
                } else {
                    Ok(text)
                }
            }
            Err(e) => Err(anyhow!("{name} failed: {e}")),
        }
    }

    /// The tool must succeed.
    async fn ok(&mut self, name: &str, arguments: Value) -> Result<String> {
        let text = self.call(name, arguments).await?;
        eprintln!("[{name}] {}", first_line(&text));
        Ok(text)
    }

    /// The tool may be refused by the server, but must not fail to encode or decode.
    async fn answers(&mut self, name: &str, arguments: Value) -> Result<()> {
        match self.call(name, arguments).await {
            Ok(text) => {
                eprintln!("[{name}] {}", first_line(&text));
                Ok(())
            }
            Err(e) => {
                let message = e.to_string();
                for sign in [
                    "decoding response body",
                    "missing field",
                    "invalid type",
                    "expected",
                ] {
                    anyhow::ensure!(
                        !message.contains(sign),
                        "{name}: the reply did not match the tool's response type: {message}"
                    );
                }
                eprintln!("[{name}] refused as expected: {}", first_line(&message));
                Ok(())
            }
        }
    }

    async fn get_json(&self, path: &str) -> Result<Value> {
        let response = self
            .http
            .get(format!("{}{path}", self.base_url))
            .header("X-API-Key", API_KEY)
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "GET {path}: {}",
            response.status()
        );
        Ok(response.json().await?)
    }

    async fn post_json(&self, path: &str, body: Value) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}{path}", self.base_url))
            .header("X-API-Key", API_KEY)
            .json(&body)
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "POST {path}: {}",
            response.status()
        );
        Ok(response.json().await?)
    }
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("")
}

/// The first UUID in `text`. Works on bytes so a multi-byte glyph before the id does not
/// put a slice boundary inside a character.
fn find_uuid(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    (0..bytes.len().saturating_sub(35)).find_map(|i| {
        let window = &bytes[i..i + 36];
        let shape = window.iter().enumerate().all(|(j, b)| match j {
            8 | 13 | 18 | 23 => *b == b'-',
            _ => b.is_ascii_hexdigit(),
        });
        shape
            .then(|| std::str::from_utf8(window).ok().map(str::to_string))
            .flatten()
    })
}

fn find_short_todo_id(text: &str) -> Option<String> {
    text.split(|c: char| c.is_whitespace() || c == '[' || c == ']' || c == '(' || c == ')')
        .find(|word| {
            word.len() == 8
                && word.starts_with("SHO-")
                && word[4..].bytes().all(|b| b.is_ascii_hexdigit())
        })
        .map(str::to_string)
}

fn find_backup_id(text: &str) -> Option<u32> {
    text.lines()
        .find_map(|line| line.split("Backup ID:").nth(1))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|id| id.parse().ok())
}

async fn start_server(data_dir: &Path) -> Result<String> {
    let config = ServerConfig {
        storage_path: data_dir.to_path_buf(),
        ..ServerConfig::default()
    };
    let manager = Arc::new(
        MultiUserMemoryManager::new(data_dir.to_path_buf(), config)
            .context("starting the memory manager")?,
    );
    let runtime = ServerRuntime::from_manager(manager);
    let app = runtime
        .protected_routes(false)
        .layer(axum::middleware::from_fn(veld::auth::auth_middleware));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("test server stopped: {e}");
        }
    });
    Ok(format!("http://{address}"))
}

async fn connect_mcp(base_url: &str) -> Result<RunningService<RoleClient, ()>> {
    let (server_io, client_io) = tokio::io::duplex(1 << 20);
    let server =
        veld::mcp::VeldMcpServer::new(base_url.to_string(), API_KEY.to_string(), USER.to_string());
    tokio::spawn(async move {
        match server.serve(server_io).await {
            Ok(running) => {
                let _ = running.waiting().await;
            }
            Err(e) => eprintln!("mcp server stopped: {e}"),
        }
    });
    let client = ().serve(client_io).await.context("mcp client handshake")?;
    Ok(client)
}

fn seed_project_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir.join("src"))?;
    fs::write(
        dir.join("README.md"),
        "# Parity fixture\n\nA small project the seed tool reads during the MCP round-trip test.\n",
    )?;
    fs::write(
        dir.join("src/lib.rs"),
        "//! Fixture crate.\n\npub fn answer() -> u32 {\n    42\n}\n",
    )?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_tool_answers_over_mcp() -> Result<()> {
    let model_dir = TempDir::new()?;
    pin_environment(model_dir.path());
    let data_dir = TempDir::new()?;
    let project_dir = TempDir::new()?;
    seed_project_dir(project_dir.path())?;

    let base_url = start_server(data_dir.path()).await?;
    let client = connect_mcp(&base_url).await?;
    let mut live = Live {
        client,
        http: reqwest::Client::new(),
        base_url: base_url.clone(),
        called: BTreeSet::new(),
    };

    let listed: BTreeSet<String> = live
        .client
        .list_all_tools()
        .await?
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();

    // Memories, recall and feedback.
    let first = live
        .ok("remember", json!({"content": "The auth middleware reads the X-API-Key header, not a cookie.", "type": "Decision", "tags": ["auth", "decision"]}))
        .await?;
    let first_id = find_uuid(&first).context("remember did not name the stored id")?;
    let second = live
        .ok("remember", json!({"content": "Recall merges vector, keyword and graph candidates before reranking.", "type": "Learning", "tags": ["retrieval"]}))
        .await?;
    let second_id = find_uuid(&second).context("second remember did not name the stored id")?;
    live.ok("recall", json!({"query": "how does auth work", "limit": 5}))
        .await?;
    live.ok(
        "proactive_context",
        json!({"context": "working on the auth middleware"}),
    )
    .await?;
    live.ok("recall_by_tags", json!({"tags": ["auth"], "limit": 5}))
        .await?;
    live.ok("context_summary", json!({})).await?;
    live.ok("list_memories", json!({"limit": 10})).await?;
    live.ok("read_memory", json!({"memory_id": first_id}))
        .await?;
    live.ok("memory_stats", json!({})).await?;
    live.ok(
        "reinforce_memories",
        json!({"memory_ids": [first_id], "outcome": "helpful"}),
    )
    .await?;
    live.ok("pin_memory", json!({"memory_id": first_id}))
        .await?;
    live.ok("unpin_memory", json!({"memory_id": first_id}))
        .await?;
    live.ok("demote_to_cold", json!({"memory_id": second_id}))
        .await?;
    live.ok("promote_to_hot", json!({"memory_id": second_id}))
        .await?;
    live.ok(
        "move_to_tier",
        json!({"memory_id": second_id, "target_tier": "session"}),
    )
    .await?;

    // Lineage.
    let link = live
        .ok(
            "lineage_link",
            json!({"from_memory_id": first_id, "to_memory_id": second_id, "relation": "RelatedTo"}),
        )
        .await?;
    let edge_to_confirm = find_uuid(&link).context("lineage_link did not name the edge")?;
    let link_back = live
        .ok("lineage_link", json!({"from_memory_id": second_id, "to_memory_id": first_id, "relation": "InformedBy"}))
        .await?;
    let edge_to_reject =
        find_uuid(&link_back).context("second lineage_link did not name the edge")?;
    live.ok(
        "lineage_trace",
        json!({"memory_id": second_id, "direction": "both"}),
    )
    .await?;
    live.ok("lineage_confirm", json!({"edge_id": edge_to_confirm}))
        .await?;
    live.ok("lineage_reject", json!({"edge_id": edge_to_reject}))
        .await?;
    live.ok("lineage_stats", json!({})).await?;

    // Graph analysis, dials, datasets.
    live.ok("gap_analyze", json!({})).await?;
    live.answers("gap_mapper", json!({})).await?;
    live.ok("get_dials", json!({})).await?;
    live.ok("set_dials", json!({"use_reranking": false}))
        .await?;
    live.answers("dataset_create", json!({"schema": {"name": "parity_rows", "columns": [{"name": "id", "ty": "I64", "nullable": false}], "primary_key": ["id"]}})).await?;
    live.answers("dataset_list", json!({})).await?;
    live.answers("dataset_query", json!({"name": "parity_rows"}))
        .await?;
    live.answers("dataset_link", json!({"name": "parity_rows", "row_pk": {"columns": {"id": 1}}, "kind": "memory", "target_id": first_id})).await?;

    // Index, facts, consolidation, seeding, tokens.
    live.ok("verify_index", json!({})).await?;
    live.ok("repair_index", json!({})).await?;
    live.ok("consolidation_report", json!({})).await?;
    live.ok("fact_narratives", json!({"limit": 5})).await?;
    live.ok(
        "facts_preview_purge",
        json!({"pattern": "no fact carries this text"}),
    )
    .await?;
    live.ok(
        "seed_project",
        json!({"project_path": project_dir.path().display().to_string(), "max_files": 10}),
    )
    .await?;
    live.ok("token_status", json!({})).await?;
    live.ok("reset_token_session", json!({})).await?;

    // Reminders.
    live.ok("set_reminder", json!({"content": "Re-run the soak once the migration lands.", "trigger_type": "duration", "after_seconds": 3600})).await?;
    live.ok("list_reminders", json!({})).await?;
    let reminders = live
        .post_json("/api/reminders", json!({"user_id": USER}))
        .await?;
    let reminder_id = reminders["reminders"][0]["id"]
        .as_str()
        .context("the reminder list is empty after set_reminder")?
        .to_string();
    live.ok("dismiss_reminder", json!({"reminder_id": reminder_id}))
        .await?;

    // Projects, todos, subtasks, comments.
    live.ok(
        "add_project",
        json!({"name": "Parity", "prefix": "PAR", "description": "Round-trip fixture"}),
    )
    .await?;
    live.ok("list_projects", json!({})).await?;
    let todo = live
        .ok("add_todo", json!({"content": "Verify every MCP tool answers", "project": "Parity", "priority": "high", "tags": ["ci"]}))
        .await?;
    let todo_id = find_short_todo_id(&todo).context("add_todo did not show the todo's short id")?;
    let sibling = live
        .ok(
            "add_todo",
            json!({"content": "Second todo for reordering", "project": "Parity"}),
        )
        .await?;
    let sibling_id =
        find_short_todo_id(&sibling).context("second add_todo did not show the todo's short id")?;
    live.ok("list_todos", json!({})).await?;
    live.ok(
        "list_todos",
        json!({"query": "MCP tool", "status": ["todo", "in_progress"]}),
    )
    .await?;
    live.ok("update_todo", json!({"todo_id": todo_id, "notes": "Driven by tests/mcp_tool_roundtrip.rs", "status": "in_progress"})).await?;
    live.ok(
        "reorder_todo",
        json!({"todo_id": sibling_id, "direction": "up"}),
    )
    .await?;
    live.ok("list_subtasks", json!({"parent_id": todo_id}))
        .await?;
    live.ok(
        "add_todo_comment",
        json!({"todo_id": todo_id, "content": "First comment", "comment_type": "note"}),
    )
    .await?;
    live.ok("list_todo_comments", json!({"todo_id": todo_id}))
        .await?;
    let comments = live
        .get_json(&format!("/api/todos/{todo_id}/comments?user_id={USER}"))
        .await?;
    let comment_id = comments["comments"][0]["id"]
        .as_str()
        .context("the comment list is empty after add_todo_comment")?
        .to_string();
    live.ok(
        "update_todo_comment",
        json!({"todo_id": todo_id, "comment_id": comment_id, "content": "First comment, revised"}),
    )
    .await?;
    live.ok(
        "delete_todo_comment",
        json!({"todo_id": todo_id, "comment_id": comment_id}),
    )
    .await?;
    live.ok("complete_todo", json!({"todo_id": todo_id}))
        .await?;
    live.ok("delete_todo", json!({"todo_id": sibling_id}))
        .await?;
    live.ok("todo_stats", json!({})).await?;
    live.ok("archive_project", json!({"project": "Parity"}))
        .await?;
    live.ok(
        "delete_project",
        json!({"project": "Parity", "delete_todos": true}),
    )
    .await?;

    // Backups last: restore replaces the live store, and forget needs a memory to drop.
    let created = live.ok("backup_create", json!({})).await?;
    let backup_id = find_backup_id(&created).context("backup_create did not show the backup id")?;
    live.ok("backup_list", json!({})).await?;
    live.ok("backup_verify", json!({"backup_id": backup_id}))
        .await?;
    live.ok("backup_purge", json!({"keep_count": 5})).await?;
    live.ok("backup_restore", json!({"backup_id": backup_id}))
        .await?;
    live.ok("memory_stats", json!({})).await?;
    live.ok("forget", json!({"id": first_id})).await?;

    let uncalled: Vec<&String> = listed.difference(&live.called).collect();
    assert!(
        uncalled.is_empty(),
        "tools the server lists that this test never called: {uncalled:?}"
    );
    let unknown: Vec<&String> = live.called.difference(&listed).collect();
    assert!(
        unknown.is_empty(),
        "tools called that the server does not list: {unknown:?}"
    );
    assert!(
        PRECONDITION_TOOLS.iter().all(|t| listed.contains(*t)),
        "the precondition list names a tool the server does not have"
    );

    live.client.cancel().await?;
    Ok(())
}
