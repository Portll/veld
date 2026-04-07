//! Veld Server — standalone binary entry point.
//!
//! This is a thin wrapper around `veld::server::run()`.
//! For the unified CLI, use `veld server` instead.
//!
//! Usage:
//!   veld [OPTIONS]
//!
//! Options:
//!   -H, --host <HOST>         Bind address [env: VELD_HOST] [default: 127.0.0.1]
//!   -p, --port <PORT>         Port number [env: VELD_PORT] [default: 3030]
//!   -s, --storage <PATH>      Storage directory [env: VELD_MEMORY_PATH]
//!   -h, --help                Print help
//!   -V, --version             Print version

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::Parser;
use veld::config::StorageBackend;
use std::path::PathBuf;

const LONG_ABOUT: &str = r#"
Veld is the Earth substrate behind Veld, featuring:

  • 3-tier memory (Working → Session → LongTerm) with automatic promotion
  • Hebbian learning - memories that help get stronger, misleading ones decay
  • Knowledge graph with spreading activation for associative retrieval
  • Vector search (MiniLM embeddings + Vamana/DiskANN index)
  • 100% offline - no cloud, no API keys needed for core functionality

The server exposes a REST API for Veld remember/recall operations. After starting:

  Health check:  curl http://localhost:3030/health
  Store memory:  curl -X POST http://localhost:3030/api/remember \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-veld-dev-local-testing-key" \
                   -d '{"user_id":"test","content":"Hello world"}'
  Search:        curl -X POST http://localhost:3030/api/recall \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-veld-dev-local-testing-key" \
                   -d '{"user_id":"test","query":"hello"}'
"#;

const AFTER_HELP: &str = r#"
INTEGRATION:
  Unified CLI:   veld server | veld tui | veld serve
  Claude Code:   claude mcp add veld -- npx -y @veld/memory-mcp
  Python:        pip install veld
  TUI:           veld tui

EXAMPLES:
  veld                          # Start with defaults
  veld -H 0.0.0.0 -p 8080      # Custom host and port
  veld --production -s /var/lib/veld  # Production mode

DOCUMENTATION:
  GitHub:  https://github.com/Portll/veld
"#;

/// Veld Server - Earth substrate for Veld
#[derive(Parser)]
#[command(name = "veld")]
#[command(version, about, long_about = LONG_ABOUT, after_help = AFTER_HELP)]
struct Cli {
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

    /// Requested storage backend (`redb` target, `rocksdb` legacy compatibility)
    #[arg(long, env = "VELD_STORAGE_BACKEND", default_value = "redb")]
    storage_backend: StorageBackend,

    /// Production mode: stricter CORS, automatic backups enabled
    #[arg(long, env = "VELD_ENV")]
    production: bool,

    /// Rate limit: max requests per second per client
    #[arg(long, env = "VELD_RATE_LIMIT", default_value_t = 4000)]
    rate_limit: u64,

    /// Maximum concurrent requests before load shedding
    #[arg(long, env = "VELD_MAX_CONCURRENT", default_value_t = 200)]
    max_concurrent: usize,
}

fn main() -> Result<()> {
    // Fortress: install anti-debug + custom panic handler BEFORE anything else.
    // Must be the first code to execute — debugger attachment at startup is caught.
    #[cfg(feature = "fortress")]
    veld::fortress::init();

    let cli = Cli::parse();

    veld::server::run(veld::server::ServerRunConfig {
        host: cli.host,
        port: cli.port,
        storage_path: cli.storage_path,
      storage_backend: cli.storage_backend,
        production: cli.production,
        rate_limit: cli.rate_limit,
        max_concurrent: cli.max_concurrent,
    })
}
