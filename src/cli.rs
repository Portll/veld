//! Unified veld CLI — server, TUI, MCP, hooks, and management commands.
//!
//! Usage:
//!   veld server              - Start the HTTP API server
//!   veld tui                 - Launch the TUI dashboard
//!   veld serve               - Run as MCP server (stdio transport)
//!   veld init                - First-time setup wizard
//!   veld status              - Check server health
//!   veld doctor              - Diagnose common issues
//!   veld hook session-start  - Output session start hook JSON
//!   veld hook prompt <msg>   - Output prompt submit hook JSON
//!   veld hook commit         - Sync the latest git commit into Veld memory
//!   veld claude [args...]    - Launch Claude Code with Veld memory
//!   veld version             - Print version and build info
//!   veld export / import / migrate - Verified backend migration (ADR 0006)

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, ErrorCode, Implementation, ProtocolVersion, ServerCapabilities,
        ServerInfo,
    },
    schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf, sync::Arc};
use veld::config::StorageBackend;

// =============================================================================
// CLI STRUCTURE
// =============================================================================

const LONG_ABOUT: &str = "\
Veld - Agentic Memory.

One binary for everything: run the server, launch the TUI, serve MCP tools,
manage configuration, and diagnose issues.

Getting started:
  veld init        Set up storage directory, API key, and ONNX runtime
  veld server      Start the HTTP API server on localhost:3030
  veld tui         Launch the terminal dashboard
  veld status      Check if the server is running";

#[derive(Parser)]
#[command(name = "veld")]
#[command(about = "Veld - Agentic Memory")]
#[command(long_about = LONG_ABOUT)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP API server
    Server {
        /// Bind address (use 0.0.0.0 for network access)
        #[arg(short = 'H', long, env = "VELD_HOST", default_value = "127.0.0.1")]
        host: String,

        /// Port number to listen on
        #[arg(short, long, env = "VELD_PORT", default_value_t = 3030)]
        port: u16,

        /// Storage directory for the selected backend data
        #[arg(
            short,
            long = "storage",
            env = "VELD_MEMORY_PATH",
            default_value_os_t = veld::config::default_storage_path()
        )]
        storage_path: PathBuf,

        /// Storage engine: `rocksdb` (the default when built in) or `redb`
        #[arg(long, env = "VELD_STORAGE_BACKEND", default_value_t = veld::config::default_requested_storage_backend())]
        storage_backend: StorageBackend,

        /// Production mode: deny-all CORS fallback, backup auto-enable, and startup safety warnings
        #[arg(long, env = "VELD_ENV")]
        production: bool,

        /// Rate limit: max requests per second per client (0 = disabled)
        #[arg(long, env = "VELD_RATE_LIMIT", default_value_t = 4000)]
        rate_limit: u64,

        /// Maximum concurrent requests before load shedding
        #[arg(long, env = "VELD_MAX_CONCURRENT", default_value_t = 200)]
        max_concurrent: usize,
    },

    /// Launch the TUI dashboard
    Tui {
        /// API URL for the Veld server
        #[arg(long, env = "VELD_SERVER_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,
    },

    /// Run as MCP server (stdio transport)
    Serve {
        /// API URL for the Veld server
        #[arg(long, env = "VELD_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for Veld operations
        #[arg(long, env = "VELD_USER_ID", default_value = "claude-code")]
        user_id: String,

        /// Whether the MCP component is enabled. Set to false to disable
        /// `veld serve` centrally (config.toml `mcp = false`, or VELD_MCP_ENABLED=false).
        #[arg(
            long = "mcp-enabled",
            env = "VELD_MCP_ENABLED",
            default_value_t = true,
            action = clap::ArgAction::Set
        )]
        mcp_enabled: bool,
    },

    /// First-time setup — create config, generate API key, download models
    Init,

    /// Check server health and status
    Status {
        /// API URL for the Veld server
        #[arg(long, env = "VELD_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,
    },

    /// Diagnose common issues (storage, ONNX, port, server health)
    Doctor,

    /// Stop a running veld server by its port
    Stop {
        /// Port the target server is listening on (e.g. --port 3032)
        #[arg(long)]
        port: u16,
    },

    /// Output Claude Code hook JSON
    Hook {
        #[command(subcommand)]
        hook_type: HookType,
    },

    /// Launch Claude Code with the Veld proxy
    Claude {
        /// Port for the Veld server
        #[arg(long, default_value = "3030")]
        port: u16,

        /// Additional arguments to pass to claude
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Print version and build information
    Version,

    /// Export one user's stores, or every store, to a directory of verified dumps
    Export {
        /// User id, or `all` for every user plus the shared stores
        #[arg(long)]
        user: String,

        /// Directory to write into; must be absent or empty
        #[arg(long)]
        out: PathBuf,

        /// Backend to read (never modified)
        #[arg(long, env = "VELD_STORAGE_BACKEND")]
        backend: StorageBackend,

        /// Write records no decoder accepts, byte for byte, to this directory instead of
        /// failing. They are NOT exported and are reported as quarantined.
        #[arg(long = "quarantine-undecodable")]
        quarantine: Option<PathBuf>,

        /// Data directory
        #[arg(long = "storage", env = "VELD_MEMORY_PATH", default_value_os_t = veld::config::default_storage_path())]
        storage_path: PathBuf,
    },

    /// Import a dump file or export directory into empty targets, verifying each against its trailer
    Import {
        /// A `.veldport` dump, or a directory written by `veld export`
        #[arg(long = "in")]
        input: PathBuf,

        /// Backend to write
        #[arg(long, env = "VELD_STORAGE_BACKEND")]
        backend: StorageBackend,

        /// Data directory
        #[arg(long = "storage", env = "VELD_MEMORY_PATH", default_value_os_t = veld::config::default_storage_path())]
        storage_path: PathBuf,
    },

    /// One-off: seal intent-log frames and slow-store payloads written before they were encrypted. Stop the server first.
    ResealJournal {
        /// Storage backend the tenants use (decides where each store keystore lives)
        #[arg(long, env = "VELD_STORAGE_BACKEND", default_value_t = veld::config::default_requested_storage_backend())]
        backend: StorageBackend,

        /// User id, or `all` for every tenant with an intent log
        #[arg(long, default_value = "all")]
        user: String,

        /// Count the plaintext frames and write nothing
        #[arg(long)]
        dry_run: bool,

        /// Data directory
        #[arg(long = "storage", env = "VELD_MEMORY_PATH", default_value_os_t = veld::config::default_storage_path())]
        storage_path: PathBuf,
    },

    /// Manage the keyring: recovery codes, device factors, passphrase and key rotation
    Keys {
        #[command(subcommand)]
        action: KeysAction,

        /// Data directory
        #[arg(long = "storage", global = true, env = "VELD_MEMORY_PATH", default_value_os_t = veld::config::default_storage_path())]
        storage_path: PathBuf,
    },

    /// Copy stores from one backend to the other in the same data directory, verified; the source is never modified
    Migrate {
        #[arg(long)]
        from: StorageBackend,

        #[arg(long)]
        to: StorageBackend,

        /// User id, or `all` for every user plus the shared stores
        #[arg(long, default_value = "all")]
        user: String,

        /// Report per-unit record counts on both backends and write nothing
        #[arg(long)]
        dry_run: bool,

        /// Write records no decoder accepts, byte for byte, to this directory instead of
        /// refusing. They are NOT migrated and are reported as quarantined.
        #[arg(long = "quarantine-undecodable")]
        quarantine: Option<PathBuf>,

        /// Data directory
        #[arg(long = "storage", env = "VELD_MEMORY_PATH", default_value_os_t = veld::config::default_storage_path())]
        storage_path: PathBuf,
    },
}

/// Secrets come from the named environment variable, or are prompted for on a terminal.
#[derive(Subcommand)]
enum KeysAction {
    /// Show the keyring and every keystore it manages, with their factors (nothing is unsealed)
    Status,
    /// Create the keyring under VELD_MASTER_PASSPHRASE, enrol every keystore in it and issue a recovery code
    Init {
        /// Write the recovery kit to this file (created 0600; never overwritten) instead of printing the code
        #[arg(long)]
        kit: Option<PathBuf>,
        #[arg(long)]
        label: Option<String>,
    },
    /// Issue another recovery code; earlier codes stay valid until revoked
    AddRecoveryCode {
        #[arg(long)]
        kit: Option<PathBuf>,
        #[arg(long)]
        label: Option<String>,
        /// Unlock with a recovery code (VELD_RECOVERY_CODE) instead of the passphrase
        #[arg(long)]
        with_recovery_code: bool,
    },
    /// Revoke a recovery code or device factor by id (see `status`)
    Revoke {
        id: String,
        #[arg(long)]
        with_recovery_code: bool,
    },
    /// Re-wrap the keyring and every keystore under VELD_NEW_MASTER_PASSPHRASE
    RotatePassphrase,
    /// Start a new data-key epoch in every keystore; earlier records stay readable
    RotateDataKeys {
        #[arg(long)]
        with_recovery_code: bool,
    },
    /// Replace every master key; recovery codes and device factors are dropped and a new code issued
    RotateMasterKeys {
        #[arg(long)]
        kit: Option<PathBuf>,
        #[arg(long)]
        label: Option<String>,
    },
    /// Lost passphrase: open the keyring with VELD_RECOVERY_CODE and re-seal everything under VELD_NEW_MASTER_PASSPHRASE
    Recover {
        /// Where to write the replacement recovery kit
        #[arg(long)]
        kit: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum HookType {
    /// Session start hook - restore context
    SessionStart {
        /// API URL for the Veld server
        #[arg(long, env = "VELD_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for Veld operations
        #[arg(long, env = "VELD_USER_ID", default_value = "claude-code")]
        user_id: String,

        /// Project directory (from CLAUDE_PROJECT_DIR)
        #[arg(long, env = "CLAUDE_PROJECT_DIR")]
        project_dir: Option<String>,
    },

    /// User prompt submit hook - inject relevant context
    Prompt {
        /// The user's message
        message: String,

        /// API URL for the Veld server
        #[arg(long, env = "VELD_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for Veld operations
        #[arg(long, env = "VELD_USER_ID", default_value = "claude-code")]
        user_id: String,
    },

    /// Git post-commit hook — sync the latest commit message into Veld memory
    Commit {
        /// API URL for the Veld server
        #[arg(long, env = "VELD_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "VELD_API_KEY",
            hide_env_values = true,
            default_value = "sk-veld-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for Veld operations
        #[arg(long, env = "VELD_USER_ID", default_value = "claude-code")]
        user_id: String,
    },
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "fortress")]
    veld::fortress::init();

    // Bridge config.toml (written by `veld init`) into the environment before clap
    // parses `env = ...` args and before ServerConfig/auth read the environment.
    veld::config::load_config_file_into_env();

    let cli = Cli::parse();

    match cli.command {
        // =====================================================================
        // NEW: veld server — start the HTTP API server in-process
        // =====================================================================
        Commands::Server {
            host,
            port,
            storage_path,
            storage_backend,
            production,
            rate_limit,
            max_concurrent,
        } => {
            // server::run() builds its own tokio runtime, so we need to exit
            // the current one first. Drop the async context and call synchronously.
            // Since server::run() is blocking and builds its own runtime, we
            // spawn it in a blocking task and await.
            let result = tokio::task::spawn_blocking(move || {
                veld::server::run(veld::server::ServerRunConfig {
                    host,
                    port,
                    storage_path,
                    storage_backend,
                    production,
                    rate_limit,
                    max_concurrent,
                })
            })
            .await?;
            result?;
        }

        // =====================================================================
        // NEW: veld tui — launch the TUI dashboard
        // =====================================================================
        Commands::Tui { api_url, api_key } => {
            handle_tui(&api_url, &api_key)?;
        }

        // =====================================================================
        // EXISTING: veld serve — MCP server
        // =====================================================================
        Commands::Serve {
            api_url,
            api_key,
            user_id,
            mcp_enabled,
        } => {
            if !mcp_enabled {
                eprintln!("Veld MCP component is disabled — not starting `veld serve`.");
                eprintln!("  Enable it with `mcp = true` in config.toml or VELD_MCP_ENABLED=true.");
                return Ok(());
            }

            eprintln!("Starting veld MCP server...");
            eprintln!("  API URL: {}", api_url);
            eprintln!("  User ID: {}", user_id);

            let server = VeldMcpServer::new(api_url, api_key, user_id);
            let service = server.serve(rmcp::transport::stdio()).await?;
            service.waiting().await?;
        }

        // =====================================================================
        // NEW: veld init — first-time setup
        // =====================================================================
        Commands::Init => {
            handle_init()?;
        }

        // =====================================================================
        // NEW: veld status — health check
        // =====================================================================
        Commands::Status { api_url, api_key } => {
            handle_status(&api_url, &api_key)?;
        }

        // =====================================================================
        // NEW: veld doctor — diagnostics
        // =====================================================================
        Commands::Doctor => {
            handle_doctor()?;
        }

        // =====================================================================
        // NEW: veld stop — stop a running server by port
        // =====================================================================
        Commands::Stop { port } => {
            handle_stop(port)?;
        }

        // =====================================================================
        // EXISTING: veld hook — Claude Code hooks
        // =====================================================================
        // The hook handlers use reqwest::blocking, which builds its own runtime. Dropping that
        // runtime inside this #[tokio::main] context panics with "Cannot drop a runtime in a
        // context where blocking is not allowed", so each handler runs on its own thread.
        Commands::Hook { hook_type } => match hook_type {
            HookType::SessionStart {
                api_url,
                api_key,
                user_id,
                project_dir,
            } => {
                run_off_runtime(move || {
                    handle_session_start(&api_url, &api_key, &user_id, project_dir.as_deref())
                });
            }

            HookType::Prompt {
                message,
                api_url,
                api_key,
                user_id,
            } => {
                run_off_runtime(move || {
                    handle_prompt_submit(&api_url, &api_key, &user_id, &message)
                });
            }

            HookType::Commit {
                api_url,
                api_key,
                user_id,
            } => {
                run_off_runtime(move || handle_commit(&api_url, &api_key, &user_id));
            }
        },

        // =====================================================================
        // EXISTING: veld claude — launch Claude Code
        // =====================================================================
        Commands::Claude { port, args } => {
            handle_claude_launch(port, args).await?;
        }

        // =====================================================================
        // NEW: veld version — version and build info
        // =====================================================================
        Commands::Version => {
            handle_version();
        }

        Commands::Export {
            user,
            out,
            backend,
            quarantine,
            storage_path,
        } => {
            let scope = veld::storage::migrate::Scope::parse(&user)?;
            let reports = tokio::task::spawn_blocking(move || {
                veld::keyring::install(&storage_path)?;
                veld::storage::migrate::export(
                    &storage_path,
                    backend,
                    &scope,
                    &out,
                    quarantine.as_deref(),
                )
            })
            .await??;
            println!("{}", serde_json::to_string_pretty(&reports)?);
            for r in reports.iter().filter_map(|r| r.quarantine.as_ref()) {
                warn_quarantined(r);
            }
        }

        Commands::Import {
            input,
            backend,
            storage_path,
        } => {
            let reports = tokio::task::spawn_blocking(move || {
                veld::keyring::install(&storage_path)?;
                veld::storage::migrate::import(&storage_path, backend, &input)
            })
            .await??;
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }

        Commands::Keys {
            action,
            storage_path,
        } => {
            let report = tokio::task::spawn_blocking(move || keys(&storage_path, action)).await??;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }

        Commands::ResealJournal {
            backend,
            user,
            dry_run,
            storage_path,
        } => {
            let reports = tokio::task::spawn_blocking(move || {
                veld::keyring::install(&storage_path)?;
                reseal_journals(&storage_path, backend, &user, dry_run)
            })
            .await??;
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }

        Commands::Migrate {
            from,
            to,
            user,
            dry_run,
            quarantine,
            storage_path,
        } => {
            let scope = veld::storage::migrate::Scope::parse(&user)?;
            let report = tokio::task::spawn_blocking(move || {
                veld::keyring::install(&storage_path)?;
                veld::storage::migrate::migrate(
                    &storage_path,
                    from,
                    to,
                    &scope,
                    dry_run,
                    quarantine.as_deref(),
                )
            })
            .await??;
            println!("{}", serde_json::to_string_pretty(&report)?);
            for r in &report.quarantined {
                warn_quarantined(r);
            }
            if dry_run && report.quarantine_requested {
                for u in &report.units {
                    if let Some(n) = u
                        .source
                        .as_ref()
                        .map(|s| s.decode_failures)
                        .filter(|n| *n > 0)
                    {
                        eprintln!("WARNING: {}: {n} undecodable record(s) would be QUARANTINED, not migrated", u.unit);
                    }
                }
            }
            if report.blocked() {
                anyhow::bail!(
                    "dry run: at least one unit is blocked; a real migration would refuse"
                );
            }
        }
    }

    Ok(())
}

fn warn_quarantined(r: &veld::storage::migrate::QuarantineReport) {
    eprintln!(
        "WARNING: {}: {} undecodable record(s) QUARANTINED, not exported: {} (raw bytes in {}, sha256 listing in {})",
        r.unit,
        r.records,
        r.ids.join(", "),
        r.dump.display(),
        r.listing.display()
    );
}

// =============================================================================
// NEW COMMAND HANDLERS
// =============================================================================

fn handle_tui(api_url: &str, api_key: &str) -> Result<()> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine executable directory"))?;

    let tui_name = if cfg!(windows) {
        "veld-tui.exe"
    } else {
        "veld-tui"
    };

    let tui_binary = exe_dir.join(tui_name);

    if !tui_binary.exists() {
        // Try PATH lookup
        let from_path = which_binary(tui_name);
        if let Some(path) = from_path {
            return exec_tui(&path, api_url, api_key);
        }

        eprintln!("Error: veld-tui not found");
        eprintln!();
        eprintln!("  Looked in: {}", exe_dir.display());
        eprintln!("  Also checked: PATH");
        eprintln!();
        eprintln!("  If you intentionally installed the last public Homebrew release:");
        eprintln!("    brew reinstall veld");
        eprintln!("    (Homebrew tracks the public release line, not this unstable branch tip)");
        eprintln!();
        eprintln!("  If installed from GitHub releases:");
        eprintln!("    Download the release archive — it includes veld-tui");
        eprintln!("    Place veld-tui alongside the veld binary");
        std::process::exit(1);
    }

    exec_tui(&tui_binary, api_url, api_key)
}

fn exec_tui(tui_binary: &std::path::Path, api_url: &str, api_key: &str) -> Result<()> {
    let status = std::process::Command::new(tui_binary)
        .env("VELD_SERVER_URL", api_url)
        .env("VELD_API_KEY", api_key)
        .status()?;

    std::process::exit(status.code().unwrap_or(1));
}

fn which_binary(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let full = dir.join(name);
            if full.is_file() {
                Some(full)
            } else {
                None
            }
        })
    })
}

fn handle_init() -> Result<()> {
    eprintln!();
    eprintln!("  Veld Setup");
    eprintln!("  ══════════════════");
    eprintln!();

    // 1. Create storage directory
    let storage = veld::config::default_storage_path();
    std::fs::create_dir_all(&storage)?;
    eprintln!("  ✓ Storage directory: {}", storage.display());

    // 2. Create config directory + generate API key
    let config_dir = config_directory();
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        eprintln!("  ✓ Config exists: {}", config_path.display());
    } else {
        let api_key = generate_api_key();
        let config_content = format!(
            "# Veld Configuration\n\
             # Generated by `veld init`\n\
             \n\
             api_key = \"{api_key}\"\n\
             host = \"127.0.0.1\"\n\
             port = 3030\n\
             # storage = \"{}\"  # Uncomment to override default\n\
             # mcp = true  # Set to false to disable the Veld MCP component (`veld serve`)\n",
            storage.display()
        );
        std::fs::write(&config_path, config_content)?;
        restrict_config_permissions(&config_path);
        eprintln!("  ✓ Config created: {}", config_path.display());
        // The full key was just written to the config file with restricted permissions — that is
        // where the user retrieves it. Printing it here lands it in every captured stderr, shell
        // transcript and CI log this command runs under, so only a recognisable prefix is shown.
        eprintln!(
            "  ✓ API key generated: {}… (full key stored in {})",
            &api_key[..api_key.len().min(8)],
            config_path.display()
        );
    }

    // 3. Pre-download ONNX runtime + embedding model
    eprintln!();
    eprintln!("  Downloading ONNX runtime and embedding model...");
    eprintln!("  (this only happens once, ~40MB total)");
    eprintln!();
    veld::embeddings::minilm::pre_init_ort_runtime(false);
    eprintln!("  ✓ ONNX runtime ready");

    // 4. Print next steps
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════╗");
    eprintln!("  ║           Setup complete!                  ║");
    eprintln!("  ╚═══════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Next steps:");
    eprintln!();
    eprintln!("    veld server        Start the Veld server");
    eprintln!("    veld tui           Launch the TUI dashboard");
    eprintln!("    veld status        Check server health");
    eprintln!();
    eprintln!("  For Claude Code / Cursor:");
    eprintln!();
    eprintln!("    claude mcp add veld -- npx -y @veld/memory-mcp");
    eprintln!();
    eprintln!("  Documentation: https://github.com/Portll/veld");
    eprintln!();

    Ok(())
}

fn handle_status(api_url: &str, api_key: &str) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;

    let resp = client
        .get(format!("{api_url}/health"))
        .header("X-API-Key", api_key)
        .send();

    match resp {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json()?;

            eprintln!();
            eprintln!("  Veld Server: RUNNING");
            eprintln!("  ════════════════════════════");
            eprintln!();

            if let Some(version) = body.get("version").and_then(|v| v.as_str()) {
                eprintln!("  Version:  {}", version);
            }
            if let Some(backend) = body
                .get("effective_storage_backend")
                .and_then(|v| v.as_str())
            {
                eprintln!("  Backend:  {}", backend);
            }
            if let Some(requested) = body
                .get("requested_storage_backend")
                .and_then(|v| v.as_str())
            {
                if body
                    .get("effective_storage_backend")
                    .and_then(|v| v.as_str())
                    != Some(requested)
                {
                    eprintln!("  Requested: {}", requested);
                }
            }
            if let Some(users) = body.get("users_count").and_then(|v| v.as_u64()) {
                eprintln!("  Users:    {}", users);
            }
            if let Some(cache_size) = body.get("users_in_cache").and_then(|v| v.as_u64()) {
                eprintln!("  Cached:   {}", cache_size);
            }

            eprintln!("  URL:      {}", api_url);
            eprintln!();
        }
        Ok(r) => {
            eprintln!("  Server returned: {}", r.status());
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!();
            eprintln!("  Veld Server: NOT RUNNING");
            eprintln!("  ════════════════════════════════");
            eprintln!();
            eprintln!("  Could not connect to {}", api_url);
            eprintln!();
            eprintln!("  Start it with:");
            eprintln!("    veld server");
            eprintln!();
            eprintln!("  Or diagnose with:");
            eprintln!("    veld doctor");
            eprintln!();
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Stop a running veld server listening on `port`.
///
/// Resolves the pid from the instance registry first (and removes that registry
/// file), falling back to an OS port→pid lookup for servers that predate the
/// registry. Termination is graceful on Unix (SIGTERM → the server's shutdown
/// handler deregisters cleanly) and forced on Windows.
fn handle_stop(port: u16) -> Result<()> {
    use std::process::Command;

    // 1) Registry lookup — also lets us clean up the stale registry file.
    let mut pid: Option<u32> = None;
    let dir = veld::instance_registry::instances_dir();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(info) =
                    serde_json::from_str::<veld::instance_registry::InstanceInfo>(&text)
                {
                    if info.port == port {
                        pid = Some(info.pid);
                        let _ = std::fs::remove_file(&path); // best-effort cleanup
                        break;
                    }
                }
            }
        }
    }

    // 2) Fallback: ask the OS which pid is listening on the port (covers servers
    //    that predate the registry, e.g. an older :3030 build).
    if pid.is_none() {
        pid = pid_listening_on_port(port);
    }

    let Some(pid) = pid else {
        anyhow::bail!("no veld server found on port {port} (not in the registry, and nothing is listening there)");
    };

    println!("Stopping veld server on port {port} (pid {pid})…");
    let stopped = if cfg!(windows) {
        Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        Command::new("kill")
            .arg(pid.to_string())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    if stopped {
        println!("✓ stopped veld on port {port} (pid {pid})");
        Ok(())
    } else {
        anyhow::bail!(
            "failed to terminate pid {pid} (port {port}) — it may need elevated permissions"
        )
    }
}

/// Best-effort: find the pid of the process listening on `127.0.0.1:port`.
#[cfg(windows)]
fn pid_listening_on_port(port: u16) -> Option<u32> {
    let out = std::process::Command::new("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = format!(":{port}");
    for line in text.lines() {
        if !line.contains("LISTENING") {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        // PROTO  Local-Address  Foreign-Address  State  PID
        if cols.len() >= 5 && cols[1].ends_with(&needle) {
            if let Ok(pid) = cols[cols.len() - 1].parse::<u32>() {
                return Some(pid);
            }
        }
    }
    None
}

#[cfg(not(windows))]
fn pid_listening_on_port(port: u16) -> Option<u32> {
    let out = std::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{port}"), "-sTCP:LISTEN"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .next()
        .and_then(|l| l.trim().parse::<u32>().ok())
}

fn handle_doctor() -> Result<()> {
    eprintln!();
    eprintln!("  Veld Doctor");
    eprintln!("  ═══════════════════");
    eprintln!();

    let mut all_ok = true;

    // 1. Storage directory
    let storage = veld::config::default_storage_path();
    let config = veld::config::ServerConfig::from_env();
    if storage.exists() {
        eprintln!("  ✓ Storage directory exists: {}", storage.display());
    } else {
        match std::fs::create_dir_all(&storage) {
            Ok(()) => {
                eprintln!("  ✓ Storage directory created: {}", storage.display());
                // Clean up — we just tested writability
                let _ = std::fs::remove_dir(&storage);
            }
            Err(e) => {
                eprintln!(
                    "  ✗ Storage directory NOT writable: {} ({})",
                    storage.display(),
                    e
                );
                all_ok = false;
            }
        }
    }

    eprintln!(
        "  ✓ Backend target: {} (runtime: {})",
        config.requested_storage_backend, config.effective_storage_backend
    );

    // 2. Config
    let config_dir = config_directory();
    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        eprintln!("  ✓ Config file exists: {}", config_path.display());
    } else {
        eprintln!("  ⚠ No config file (run `veld init` to create one)");
    }

    // 3. ONNX runtime
    eprintln!("  … Checking ONNX runtime");
    veld::embeddings::minilm::pre_init_ort_runtime(false);
    eprintln!("  ✓ ONNX runtime loads OK");

    // 4. Port availability
    let port: u16 = std::env::var("VELD_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3030);

    let addr = format!("127.0.0.1:{port}");
    match std::net::TcpListener::bind(&addr) {
        Ok(_listener) => {
            eprintln!("  ✓ Port {port} is available");
        }
        Err(_) => {
            // Port in use — could be our server or something else
            eprintln!("  ⚠ Port {port} is in use");
        }
    }

    // 5. Server health (if running)
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    match client.get(format!("http://127.0.0.1:{port}/health")).send() {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().unwrap_or_default();
            let version = body
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            eprintln!("  ✓ Server is running (v{version})");
        }
        Ok(r) => {
            eprintln!("  ⚠ Port {port} responds but returned {}", r.status());
            all_ok = false;
        }
        Err(_) => {
            eprintln!("  ⚠ Server is not running on port {port}");
        }
    }

    // Summary
    eprintln!();
    if all_ok {
        eprintln!("  All checks passed.");
    } else {
        eprintln!("  Some issues found. Run `veld init` to fix setup issues.");
    }
    eprintln!();

    Ok(())
}

fn handle_version() {
    eprintln!("veld {}", env!("VELD_VERSION_FULL"));
    eprintln!("  Platform: {}", std::env::consts::OS);
    eprintln!("  Arch:     {}", std::env::consts::ARCH);
    eprintln!(
        "  Target backend: {}",
        veld::config::default_requested_storage_backend()
    );
    eprintln!(
        "  Runtime backend: {}",
        veld::config::effective_storage_backend_for_current_build(
            veld::config::default_requested_storage_backend()
        )
    );
    eprintln!("  License:  AGPL-3.0-or-later (or commercial — see LICENSING.md)");
    eprintln!("  Repo:     https://github.com/Portll/veld");
}

// =============================================================================
// UTILITY HELPERS
// =============================================================================

fn config_directory() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("veld"))
        .unwrap_or_else(|| PathBuf::from(".veld"))
}

fn generate_api_key() -> String {
    // Random 128-bit key. A timestamp-derived key is guessable within the window
    // the user ran `veld init`; this matches auth::default_dev_api_key().
    format!("sk-veld-{}", uuid::Uuid::new_v4().simple())
}

/// Restrict the config file to owner-only access on Unix.
///
/// `config.toml` holds the API key in plaintext; world-readable permissions would
/// expose it to other local users. On Windows the per-user `%APPDATA%` directory is
/// already ACL-scoped to the user, so no action is needed there.
fn restrict_config_permissions(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(err) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            eprintln!(
                "  ⚠ Could not restrict permissions on {}: {err}",
                path.display()
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

// =============================================================================
// API CLIENT
// =============================================================================

/// HTTP client for the veld API (async version for MCP tools)
#[derive(Clone, Debug)]
struct AsyncApiClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    user_id: String,
}

impl AsyncApiClient {
    fn new(base_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            user_id,
        }
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json().await?)
    }
}

/// HTTP client for the veld API (blocking version for hooks)
#[derive(Clone, Debug)]
struct BlockingApiClient {
    client: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
}

impl BlockingApiClient {
    fn new(base_url: String, api_key: String) -> Self {
        Self::with_timeout(base_url, api_key, None)
    }

    /// Construct a client with an optional request timeout. Hooks that must not
    /// hang a developer's workflow (e.g. the git post-commit hook) set a timeout.
    fn with_timeout(
        base_url: String,
        api_key: String,
        timeout: Option<std::time::Duration>,
    ) -> Self {
        let mut builder = reqwest::blocking::Client::builder();
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            client,
            base_url,
            api_key,
        }
    }

    fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json()?)
    }
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

#[derive(Serialize)]
struct ProactiveContextRequest {
    user_id: String,
    context: String,
    max_results: u32,
    auto_ingest: bool,
}

#[derive(Deserialize)]
struct ProactiveContextResponse {
    memories: Vec<SurfacedMemory>,
}

// /api/proactive_context names the score `score`; the old `relevance_score` mirror failed to
// deserialise, and the session-start hook discards the error, so it restored no memories at all.
#[derive(Deserialize)]
struct SurfacedMemory {
    id: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    memory_type: String,
    #[serde(default, alias = "relevance_score")]
    score: f32,
}

#[derive(Serialize)]
struct ListTodosRequest {
    user_id: String,
    status: Vec<String>,
}

#[derive(Deserialize)]
struct ListTodosResponse {
    todos: Vec<Todo>,
}

#[derive(Deserialize)]
struct Todo {
    #[allow(dead_code)]
    id: String,
    content: String,
    status: String,
    priority: Option<String>,
    #[allow(dead_code)]
    project: Option<String>,
}

// The server fills absent fields from serde defaults but rejects an explicit null for a
// non-Option field (RecallRequest.mode and limit, RememberRequest.tags), so None is omitted.
#[derive(Serialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct RememberResponse {
    id: String,
}

#[derive(Serialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<String>,
}

#[derive(Deserialize)]
struct RecallResponse {
    memories: Vec<RecalledMemory>,
}

// /api/recall nests the experience and names the score `score`; a flat {content, memory_type,
// similarity} mirror deserialised nothing and the tool reported "error decoding response body".
#[derive(Deserialize)]
struct RecalledMemory {
    id: String,
    experience: RecalledExperience,
    #[serde(default)]
    score: f32,
}

#[derive(Deserialize)]
struct RecalledExperience {
    #[serde(default)]
    content: String,
    #[serde(default)]
    memory_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    tags: Vec<String>,
}

// =============================================================================
// HOOK OUTPUT
// =============================================================================

#[derive(Serialize)]
struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: HookSpecificOutput,
}

#[derive(Serialize)]
struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    hook_event_name: String,
    #[serde(rename = "additionalContext")]
    additional_context: String,
}

/// Run a blocking hook handler on its own thread, off the tokio runtime.
fn run_off_runtime<F: FnOnce() + Send + 'static>(f: F) {
    match std::thread::spawn(f).join() {
        Ok(()) => {}
        Err(e) => std::panic::resume_unwind(e),
    }
}

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

fn handle_session_start(api_url: &str, api_key: &str, user_id: &str, project_dir: Option<&str>) {
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

fn handle_prompt_submit(api_url: &str, api_key: &str, user_id: &str, message: &str) {
    let client = BlockingApiClient::new(api_url.to_string(), api_key.to_string());

    // Get proactive context based on user message
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: message.to_string(),
            max_results: 5,
            auto_ingest: true,
        },
    );

    let mut context_parts = Vec::new();

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("## Relevant Memories (auto-surfaced)\n".to_string());
            for mem in ctx.memories.iter() {
                let relevance = (mem.score * 100.0) as u32;
                context_parts.push(format!(
                    "- [{}%] **{}**: {}",
                    relevance,
                    mem.memory_type,
                    mem.content.chars().take(300).collect::<String>()
                ));
            }
        }
    }

    if !context_parts.is_empty() {
        output_hook("UserPromptSubmit", &context_parts.join("\n"));
    } else {
        output_hook("UserPromptSubmit", "");
    }
}

/// Latest commit metadata gathered from `git`.
struct CommitInfo {
    short_hash: String,
    author: String,
    message: String,
    branch: String,
    repo: String,
    files: String,
}

/// Read the HEAD commit via `git`. Returns `None` when the current directory is
/// not a git repository or `git` is unavailable — the caller treats that as
/// "nothing to sync" rather than an error.
fn read_head_commit() -> Option<CommitInfo> {
    fn git(args: &[&str]) -> Option<String> {
        let output = std::process::Command::new("git").args(args).output().ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    // Unit-separator (\x1f) delimited so the message body can contain newlines,
    // quotes, or any other text without breaking the parse.
    let raw = git(&["log", "-1", "--format=%h%x1f%an%x1f%B"])?;
    let mut parts = raw.splitn(3, '\u{1f}');
    let short_hash = parts.next()?.to_string();
    let author = parts.next()?.to_string();
    let message = parts.next()?.trim().to_string();
    if short_hash.is_empty() {
        return None;
    }

    let branch =
        git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let files =
        git(&["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).unwrap_or_default();
    let repo = git(&["rev-parse", "--show-toplevel"])
        .and_then(|path| {
            std::path::Path::new(&path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    Some(CommitInfo {
        short_hash,
        author,
        message,
        branch,
        repo,
        files,
    })
}

/// Git post-commit hook — sync the latest commit message into Veld memory.
///
/// Reads HEAD via `git` and stores it as a memory through `/api/remember`.
/// Never fails the surrounding commit: a missing server or absent repo is
/// reported to stderr and swallowed. Uses a short request timeout so a slow or
/// unreachable server cannot hang the developer's `git commit`.
fn handle_commit(api_url: &str, api_key: &str, user_id: &str) {
    let Some(commit) = read_head_commit() else {
        // Not a git repo, or git unavailable — nothing to sync.
        return;
    };

    let files_section = if commit.files.is_empty() {
        String::new()
    } else {
        format!("\n\nFiles changed:\n{}", commit.files)
    };
    let content = format!(
        "Git commit {} on branch {} of {} by {}\n\n{}{}",
        commit.short_hash, commit.branch, commit.repo, commit.author, commit.message, files_section
    );

    let client = BlockingApiClient::with_timeout(
        api_url.to_string(),
        api_key.to_string(),
        Some(std::time::Duration::from_secs(10)),
    );

    let result: Result<RememberResponse> = client.post(
        "/api/remember",
        &RememberRequest {
            user_id: user_id.to_string(),
            content,
            memory_type: None,
            tags: Some(vec![
                "git".to_string(),
                "commit".to_string(),
                commit.repo.clone(),
                commit.branch.clone(),
            ]),
        },
    );

    match result {
        Ok(resp) => eprintln!(
            "[veld] Synced commit {} to memory ({})",
            commit.short_hash, resp.id
        ),
        Err(err) => eprintln!(
            "[veld] Commit {} not synced — is the Veld server running? ({err})",
            commit.short_hash
        ),
    }
}

// =============================================================================
// MCP TOOL PARAMETER TYPES
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RememberParams {
    /// The content to remember
    content: String,
    /// Type of memory (Observation, Decision, Learning, etc.)
    #[serde(rename = "type")]
    memory_type: Option<String>,
    /// Optional tags for categorization
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallParams {
    /// Natural language search query
    query: String,
    /// Maximum number of results (default: 5)
    limit: Option<u32>,
    /// Retrieval mode: semantic, associative, or hybrid
    mode: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ProactiveContextParams {
    /// Current conversation context
    context: String,
    /// Maximum memories to surface (default: 5)
    max_results: Option<u32>,
    /// Auto-store context for feedback (default: true)
    auto_ingest: Option<bool>,
}

// =============================================================================
// LINEAGE MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageTraceParams {
    /// Memory ID to trace lineage from
    memory_id: String,
    /// Direction: "backward" (find causes), "forward" (find effects), "both"
    direction: Option<String>,
    /// Maximum depth to traverse (default: 10)
    max_depth: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageConfirmParams {
    /// ID of the inferred edge to confirm
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageRejectParams {
    /// ID of the inferred edge to reject
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageLinkParams {
    /// Source memory ID (the cause/origin)
    from_memory_id: String,
    /// Target memory ID (the effect/result)
    to_memory_id: String,
    /// Relation type: Caused, ResolvedBy, InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo
    relation: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageStatsParams {
    /// Optional - leave empty to get stats for current user
    #[serde(default)]
    _placeholder: Option<String>,
}

// Lineage API request types
#[derive(Serialize)]
struct LineageTraceRequest {
    user_id: String,
    memory_id: String,
    direction: String,
    max_depth: u32,
}

#[derive(Serialize)]
struct LineageEdgeRequest {
    user_id: String,
    edge_id: String,
}

#[derive(Serialize)]
struct LineageAddEdgeRequest {
    user_id: String,
    from_memory_id: String,
    to_memory_id: String,
    relation: String,
}

#[derive(Serialize)]
struct LineageStatsRequest {
    user_id: String,
}

// Lineage API response types
#[derive(Deserialize)]
struct LineageTraceResponse {
    root: String,
    direction: String,
    edges: Vec<LineageEdgeInfo>,
    path: Vec<String>,
    depth: usize,
}

#[derive(Deserialize)]
struct LineageEdgeInfo {
    #[allow(dead_code)]
    id: String,
    from: String,
    to: String,
    relation: String,
    confidence: f32,
    source: String,
}

#[derive(Deserialize)]
struct LineageConfirmResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageRejectResponse {
    message: String,
    #[allow(dead_code)]
    deleted: bool,
}

#[derive(Deserialize)]
struct LineageAddResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageStatsResponse {
    total_edges: usize,
    inferred_edges: usize,
    confirmed_edges: usize,
    explicit_edges: usize,
    total_branches: usize,
    active_branches: usize,
    edges_by_relation: std::collections::HashMap<String, usize>,
    avg_confidence: f32,
}

// =============================================================================
// MCP SERVER
// =============================================================================

#[derive(Debug, Clone)]
struct VeldMcpServer {
    client: Arc<AsyncApiClient>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl VeldMcpServer {
    fn new(api_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: Arc::new(AsyncApiClient::new(api_url, api_key, user_id)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a memory for future recall. Use this to remember important information, decisions, user preferences, project context, or anything you want to recall later."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RememberResponse> = self
            .client
            .post(
                "/api/remember",
                &RememberRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    memory_type: params.memory_type,
                    tags: params.tags,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Stored memory: {}",
                resp.id
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Search memories using semantic, associative, or hybrid retrieval. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'hybrid' (combines both)."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RecallResponse> = self
            .client
            .post(
                "/api/recall",
                &RecallRequest {
                    user_id: self.client.user_id.clone(),
                    query: params.query,
                    limit: params.limit,
                    mode: params.mode,
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Found {} memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "**[{}]** {} (similarity: {:.0}%)\n{}\n\n",
                        mem.experience.memory_type,
                        &mem.id[..8.min(mem.id.len())],
                        mem.score * 100.0,
                        mem.experience.content
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "REQUIRED: Call this to surface relevant memories based on current context. Enables automatic memory surfacing and implicit feedback learning."
    )]
    async fn proactive_context(
        &self,
        Parameters(params): Parameters<ProactiveContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ProactiveContextResponse> = self
            .client
            .post(
                "/api/proactive_context",
                &ProactiveContextRequest {
                    user_id: self.client.user_id.clone(),
                    context: params.context,
                    max_results: params.max_results.unwrap_or(5),
                    auto_ingest: params.auto_ingest.unwrap_or(true),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Surfaced {} relevant memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "- [{}%] **{}**: {}\n",
                        (mem.score * 100.0) as u32,
                        mem.memory_type,
                        mem.content.chars().take(200).collect::<String>()
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // LINEAGE TOOLS - Causal Memory Tracking
    // =========================================================================

    #[tool(
        description = "Trace the causal lineage of a memory. Find what caused it (backward), what it led to (forward), or both. Useful for understanding 'why' something happened."
    )]
    async fn lineage_trace(
        &self,
        Parameters(params): Parameters<LineageTraceParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageTraceResponse> = self
            .client
            .post(
                "/api/lineage/trace",
                &LineageTraceRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    direction: params.direction.unwrap_or_else(|| "backward".to_string()),
                    max_depth: params.max_depth.unwrap_or(10),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Lineage Trace** ({})\n\nRoot: {}\nDepth: {}\n\n",
                    resp.direction, resp.root, resp.depth
                );

                if resp.edges.is_empty() {
                    output.push_str("No causal connections found.\n");
                } else {
                    output.push_str("**Causal Chain:**\n");
                    for edge in &resp.edges {
                        let confidence = (edge.confidence * 100.0) as u32;
                        let source_icon = match edge.source.as_str() {
                            "Confirmed" => "✓",
                            "Explicit" => "⚡",
                            _ => "?",
                        };
                        output.push_str(&format!(
                            "  {} --[{} {}% {}]--> {}\n",
                            &edge.from[..8.min(edge.from.len())],
                            edge.relation,
                            confidence,
                            source_icon,
                            &edge.to[..8.min(edge.to.len())]
                        ));
                    }

                    output.push_str(&format!("\n**Path:** {}\n", resp.path.join(" → ")));
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Confirm an inferred causal relationship between memories. This improves the system's confidence and learning."
    )]
    async fn lineage_confirm(
        &self,
        Parameters(params): Parameters<LineageConfirmParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageConfirmResponse> = self
            .client
            .post(
                "/api/lineage/confirm",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✓ Confirmed edge: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Reject an incorrectly inferred causal relationship. This helps the system learn better inference patterns."
    )]
    async fn lineage_reject(
        &self,
        Parameters(params): Parameters<LineageRejectParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageRejectResponse> = self
            .client
            .post(
                "/api/lineage/reject",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✗ Rejected edge: {}",
                resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Create an explicit causal link between two memories. Relations: Caused (Error→Todo), ResolvedBy (Todo→Learning), InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo."
    )]
    async fn lineage_link(
        &self,
        Parameters(params): Parameters<LineageLinkParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageAddResponse> = self
            .client
            .post(
                "/api/lineage/link",
                &LineageAddEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    from_memory_id: params.from_memory_id,
                    to_memory_id: params.to_memory_id,
                    relation: params.relation,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "⚡ Created link: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Get statistics about the causal lineage graph - edge counts, relation types, confidence distribution."
    )]
    async fn lineage_stats(
        &self,
        Parameters(_params): Parameters<LineageStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageStatsResponse> = self
            .client
            .post(
                "/api/lineage/stats",
                &LineageStatsRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = "**Lineage Graph Statistics**\n\n".to_string();
                output.push_str(&format!("**Edges:** {}\n", resp.total_edges));
                output.push_str(&format!("  ✓ Confirmed: {}\n", resp.confirmed_edges));
                output.push_str(&format!("  ? Inferred: {}\n", resp.inferred_edges));
                output.push_str(&format!("  ⚡ Explicit: {}\n", resp.explicit_edges));
                output.push_str(&format!(
                    "Average Confidence: {:.1}%\n\n",
                    resp.avg_confidence * 100.0
                ));
                output.push_str(&format!(
                    "**Branches:** {} total, {} active\n\n",
                    resp.total_branches, resp.active_branches
                ));

                if !resp.edges_by_relation.is_empty() {
                    output.push_str("**By Relation Type:**\n");
                    let mut relations: Vec<_> = resp.edges_by_relation.iter().collect();
                    relations.sort_by(|a, b| b.1.cmp(a.1));
                    for (relation, count) in relations {
                        output.push_str(&format!("  {}: {}\n", relation, count));
                    }
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for VeldMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.protocol_version = ProtocolVersion::V_2024_11_05;
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::from_build_env();
        info.instructions = Some(
            "Veld Memory - persistent cognitive memory with causal reasoning. \
             Use proactive_context at session start to surface relevant memories. \
             Use remember to store decisions, learnings, errors. \
             Use recall to search memories. \
             Use lineage_trace to understand 'why' - trace causal chains backward/forward. \
             Use lineage_link to explicitly connect cause→effect memories. \
             Use lineage_confirm/reject to improve inference accuracy."
                .to_string(),
        );
        info
    }
}

// =============================================================================
// CLAUDE LAUNCH
// =============================================================================

/// Launch Claude Code with Veld Cortex proxy
async fn handle_claude_launch(port: u16, args: Vec<String>) -> Result<()> {
    let server_url = format!("http://127.0.0.1:{port}");

    // Check if server is running
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_url = format!("{server_url}/health");
    let server_running = client.get(&health_url).send().await.is_ok();

    if !server_running {
        eprintln!("🧠 Starting Veld server on port {port}...");

        // Start server in background
        let exe_path = std::env::current_exe()?;
        let server_binary = exe_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot find executable directory"))?
            .join("veld");

        #[cfg(windows)]
        let server_binary = server_binary.with_extension("exe");

        if !server_binary.exists() {
            eprintln!("⚠️  veld not found at {:?}", server_binary);
            eprintln!("   Please ensure veld is installed and in PATH");
            std::process::exit(1);
        }

        let mut cmd = std::process::Command::new(&server_binary);
        cmd.env("VELD_PORT", port.to_string());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            const DETACHED_PROCESS: u32 = 0x00000008;
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }

        #[allow(clippy::zombie_processes)]
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
        eprintln!("🧠 Veld server already running on port {port}");
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
        let mut cmd = std::process::Command::new("cmd");
        cmd.arg("/c").arg("claude");
        cmd.env("ANTHROPIC_API_BASE", &server_url);
        cmd.args(&args);
        let status = cmd.status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the server's `/api/remember` response shape is `{id, success}`
    /// (see `handlers::remember::RememberResponse`). An earlier CLI mirror
    /// required `message` instead of `success`, which made the post-commit hook
    /// emit "error decoding response body" on every commit. Keep this test in
    /// lockstep with the server's response struct.
    #[test]
    fn remember_response_deserializes_server_shape() {
        // Exact shape the server emits today.
        let body = r#"{"id":"mem-abc123","success":true}"#;
        let resp: RememberResponse = serde_json::from_str(body).expect("must deserialize");
        assert_eq!(resp.id, "mem-abc123");
    }

    #[test]
    fn requests_with_no_optional_fields_deserialize_as_the_server_types() {
        let recall = serde_json::to_value(RecallRequest {
            user_id: "u".into(),
            query: "q".into(),
            limit: None,
            mode: None,
        })
        .unwrap();
        let parsed: veld::handlers::types::RecallRequest = serde_json::from_value(recall)
            .expect("server must accept a recall with no limit or mode");
        assert_eq!(parsed.query, "q");

        let remember = serde_json::to_value(RememberRequest {
            user_id: "u".into(),
            content: "c".into(),
            memory_type: None,
            tags: None,
        })
        .unwrap();
        let parsed: veld::handlers::types::RememberRequest =
            serde_json::from_value(remember).expect("server must accept a remember with no tags");
        assert!(parsed.tags.is_empty());
    }

    /// Regression: `/api/recall` nests the experience and names the score `score`.
    /// Keep this body in lockstep with `handlers::recall`'s response.
    #[test]
    fn recall_response_deserializes_server_shape() {
        let body = r#"{"memories":[{"id":"mem-1","experience":{"content":"hello","memory_type":"Context","tags":["a"]},"importance":0.5,"created_at":"2026-09-16T00:00:00Z","score":0.94,"tier":"hot","writer":"portll"}],"count":1,"tokens_estimated":12,"metacognition":null}"#;
        let resp: RecallResponse = serde_json::from_str(body).expect("must deserialize");
        assert_eq!(resp.memories.len(), 1);
        assert_eq!(resp.memories[0].experience.content, "hello");
        assert_eq!(resp.memories[0].experience.memory_type, "Context");
        assert!((resp.memories[0].score - 0.94).abs() < f32::EPSILON);
    }

    /// Regression: `/api/proactive_context` names the score `score`. The session-start hook
    /// discards a decode error, so a mismatch here restores no memories and says nothing.
    #[test]
    fn proactive_context_response_deserializes_server_shape() {
        let body = r#"{"memories":[{"id":"mem-1","content":"hello","memory_type":"Context","score":0.81,"importance":0.5,"created_at":"2026-09-16T00:00:00Z","tags":["a"],"tier":"hot","relevance_reason":"entity match","hours_since_created":1.0,"access_count":2,"activation_level":0.3,"retrieval_trigger":"entity","intrusion_score":0.0}],"memory_count":1,"reminder_count":0,"todo_count":0,"latency_ms":3}"#;
        let resp: ProactiveContextResponse = serde_json::from_str(body).expect("must deserialize");
        assert_eq!(resp.memories.len(), 1);
        assert_eq!(resp.memories[0].content, "hello");
        assert!((resp.memories[0].score - 0.81).abs() < f32::EPSILON);
    }

    #[test]
    fn remember_response_tolerates_extra_fields() {
        // Future server additions must not break the hook.
        let body = r#"{"id":"mem-xyz","success":true,"message":"ok","latency_ms":42}"#;
        let resp: RememberResponse = serde_json::from_str(body).expect("must deserialize");
        assert_eq!(resp.id, "mem-xyz");
    }
}

fn secret(env: &str, prompt: &str) -> anyhow::Result<zeroize::Zeroizing<String>> {
    if let Ok(v) = std::env::var(env) {
        if !v.is_empty() {
            return Ok(zeroize::Zeroizing::new(v));
        }
    }
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        anyhow::bail!("{env} is not set and there is no terminal to ask on");
    }
    eprint!("{prompt}: ");
    #[cfg(unix)]
    let echo_off = std::process::Command::new("stty")
        .arg("-echo")
        .stdin(std::process::Stdio::inherit())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let mut line = zeroize::Zeroizing::new(String::new());
    let read = std::io::stdin().read_line(&mut line);
    #[cfg(unix)]
    if echo_off {
        let _ = std::process::Command::new("stty")
            .arg("echo")
            .stdin(std::process::Stdio::inherit())
            .status();
        eprintln!();
    }
    read?;
    let value = zeroize::Zeroizing::new(line.trim_end_matches(['\r', '\n']).to_string());
    if value.is_empty() {
        anyhow::bail!("{prompt}: nothing entered");
    }
    Ok(value)
}

/// Write a recovery kit to `path` (0600, refusing to overwrite), or leave it in the
/// report to print.
fn deliver_kit(
    issued: veld::keyring::IssuedCode,
    path: Option<&std::path::Path>,
) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context as _;
    let Some(path) = path else {
        return Ok(serde_json::to_value(issued)?);
    };
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options
        .open(path)
        .with_context(|| format!("create {path:?} (an existing kit is never overwritten)"))?;
    file.write_all(issued.kit.as_bytes())?;
    file.sync_all()?;
    Ok(serde_json::json!({ "id": issued.id, "label": issued.label, "kit": path }))
}

fn keys(storage: &std::path::Path, action: KeysAction) -> anyhow::Result<serde_json::Value> {
    use veld::keyring;
    use veld::keystore::Factor;
    const PASS: &str = "VELD_MASTER_PASSPHRASE";
    const NEW_PASS: &str = "VELD_NEW_MASTER_PASSPHRASE";
    const CODE: &str = "VELD_RECOVERY_CODE";
    let unlock_secret = |with_code: bool| {
        if with_code {
            secret(CODE, "Recovery code").map(|s| (true, s))
        } else {
            secret(PASS, "Passphrase").map(|s| (false, s))
        }
    };
    fn factor(unlock: &(bool, zeroize::Zeroizing<String>)) -> Factor<'_> {
        match unlock {
            (true, code) => Factor::RecoveryCode(code),
            (false, passphrase) => Factor::Passphrase(passphrase),
        }
    }
    Ok(match action {
        KeysAction::Status => serde_json::to_value(keyring::status(storage)?)?,
        KeysAction::Init { kit, label } => {
            let pass = secret(PASS, "Passphrase")?;
            let report = keyring::init(storage, &pass, label.as_deref())?;
            let mut value = serde_json::to_value(&report)?;
            value["recovery"] = deliver_kit(report.recovery, kit.as_deref())?;
            value
        }
        KeysAction::AddRecoveryCode {
            kit,
            label,
            with_recovery_code,
        } => {
            let unlock = unlock_secret(with_recovery_code)?;
            let issued = keyring::add_recovery_code(storage, &factor(&unlock), label.as_deref())?;
            deliver_kit(issued, kit.as_deref())?
        }
        KeysAction::Revoke {
            id,
            with_recovery_code,
        } => {
            let unlock = unlock_secret(with_recovery_code)?;
            serde_json::to_value(keyring::revoke(storage, &factor(&unlock), &id)?)?
        }
        KeysAction::RotatePassphrase => {
            let old = secret(PASS, "Current passphrase")?;
            let new = secret(NEW_PASS, "New passphrase")?;
            serde_json::to_value(keyring::rotate_passphrase(storage, &old, &new)?)?
        }
        KeysAction::RotateDataKeys { with_recovery_code } => {
            let unlock = unlock_secret(with_recovery_code)?;
            serde_json::to_value(keyring::rotate_data_keys(storage, &factor(&unlock))?)?
        }
        KeysAction::RotateMasterKeys { kit, label } => {
            let pass = secret(PASS, "Passphrase")?;
            let report = keyring::rotate_master_keys(storage, &pass, label.as_deref())?;
            let mut value = serde_json::to_value(&report)?;
            value["recovery"] = deliver_kit(report.recovery, kit.as_deref())?;
            value
        }
        KeysAction::Recover { kit } => {
            let code = secret(CODE, "Recovery code")?;
            let new = secret(NEW_PASS, "New passphrase")?;
            let report = keyring::recover(storage, &Factor::RecoveryCode(&code), &new)?;
            let mut value = serde_json::to_value(&report)?;
            if let Some(issued) = report.reissued {
                value["reissued"] = deliver_kit(issued, kit.as_deref())?;
            }
            value
        }
    })
}

/// Each tenant's journal is resealed under the key the server would use for it: the
/// global keystore when VELD_KEYSTORE_PATH is set, otherwise the tenant's store
/// keystore. A tenant with neither keeps a plaintext journal by configuration and is
/// reported as skipped; no keystore is created here.
fn reseal_journals(
    storage: &std::path::Path,
    backend: StorageBackend,
    user: &str,
    dry_run: bool,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let global = veld::keystore::SecretCryptor::from_env("journal")?;
    let tenants: Vec<std::path::PathBuf> = if user == "all" {
        let mut dirs: Vec<_> = std::fs::read_dir(storage)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("intent.log").is_file())
            .collect();
        dirs.sort();
        dirs
    } else {
        vec![storage.join(veld::path_guard::sanitize_component(user, "user id")?)]
    };
    let mut reports = Vec::new();
    for tenant in tenants {
        let log = tenant.join("intent.log");
        if !log.is_file() {
            anyhow::bail!("{log:?} does not exist");
        }
        let store_keystore = veld::keystore::store_keystore_path(&tenant, backend);
        let cryptor = match &global {
            Some(c) => Some(c.clone()),
            None if store_keystore.exists() => {
                veld::keystore::SecretCryptor::from_record_keystore(&store_keystore, "journal")?
            }
            None => None,
        };
        let slow_store = tenant.join("slow_store.db");
        let slow_cryptor = match &cryptor {
            Some(_) if slow_store.is_file() => {
                match veld::keystore::SecretCryptor::from_env("slow-store")? {
                    Some(c) => Some(c),
                    None => veld::keystore::SecretCryptor::from_record_keystore(
                        &store_keystore,
                        "slow-store",
                    )?,
                }
            }
            _ => None,
        };
        match cryptor {
            Some(c) => {
                let mut report = serde_json::to_value(
                    veld::intent_log::reseal::reseal_plaintext_frames(&log, &c, dry_run)?,
                )?;
                if let Some(sc) = slow_cryptor {
                    let (rows, plaintext) = veld::memory::slow_store::reseal_plaintext_payloads(
                        &slow_store,
                        sc,
                        dry_run,
                    )?;
                    report["slow_store"] = serde_json::json!({
                        "path": slow_store, "rows": rows, "plaintext_rows": plaintext, "sealed": !dry_run,
                    });
                }
                reports.push(report);
            }
            None => reports.push(serde_json::json!({
                "path": log,
                "skipped": "no keystore for this tenant: its journal is plaintext by configuration",
            })),
        }
    }
    Ok(reports)
}
