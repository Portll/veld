//! Shodh-Memory Server — standalone binary entry point.
//!
//! This is a thin wrapper around `shodh_memory::server::run()`.
//! For the unified CLI, use `shodh server` instead.
//!
//! Usage:
//!   shodh-memory-server [OPTIONS]
//!
//! Options:
//!   -H, --host <HOST>         Bind address [env: SHODH_HOST] [default: 127.0.0.1]
//!   -p, --port <PORT>         Port number [env: SHODH_PORT] [default: 3030]
//!   -s, --storage <PATH>      Storage directory [env: SHODH_MEMORY_PATH]
//!   -h, --help                Print help
//!   -V, --version             Print version

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

const LONG_ABOUT: &str = r#"
Shodh-Memory is a cognitive memory system for AI agents, featuring:

  • 3-tier memory (Working → Session → LongTerm) with automatic promotion
  • Hebbian learning - memories that help get stronger, misleading ones decay
  • Knowledge graph with spreading activation for associative retrieval
  • Vector search (MiniLM embeddings + Vamana/DiskANN index)
  • 100% offline - no cloud, no API keys needed for core functionality

The server exposes a REST API for memory operations. After starting:

  Health check:  curl http://localhost:3030/health
  Store memory:  curl -X POST http://localhost:3030/api/remember \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-shodh-dev-local-testing-key" \
                   -d '{"user_id":"test","content":"Hello world"}'
  Search:        curl -X POST http://localhost:3030/api/recall \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-shodh-dev-local-testing-key" \
                   -d '{"user_id":"test","query":"hello"}'
"#;

const AFTER_HELP: &str = r#"
INTEGRATION:
  Unified CLI:   shodh server | shodh tui | shodh serve
  Claude Code:   claude mcp add shodh-memory -- npx -y @shodh/memory-mcp
  Python:        pip install shodh-memory
  TUI:           shodh tui

EXAMPLES:
  shodh-memory-server                          # Start with defaults
  shodh-memory-server -H 0.0.0.0 -p 8080      # Custom host and port
  shodh-memory-server --production -s /var/lib/shodh  # Production mode

DOCUMENTATION:
  GitHub:  https://github.com/varun29ankuS/shodh-memory
"#;

/// Shodh-Memory Server - Cognitive Memory for AI Agents
#[derive(Parser)]
#[command(name = "shodh-memory-server")]
#[command(version, about, long_about = LONG_ABOUT, after_help = AFTER_HELP)]
struct Cli {
    /// Bind address (use 0.0.0.0 for network access)
    #[arg(short = 'H', long, env = "SHODH_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on
    #[arg(short, long, env = "SHODH_PORT", default_value_t = 3030)]
    port: u16,

    /// Storage directory for RocksDB data
    #[arg(
        short,
        long = "storage",
        env = "SHODH_MEMORY_PATH",
        default_value_os_t = shodh_memory::config::default_storage_path()
    )]
    storage_path: PathBuf,

    /// Production mode: stricter CORS, automatic backups enabled
    #[arg(long, env = "SHODH_ENV")]
    production: bool,

    /// Rate limit: max requests per second per client
    #[arg(long, env = "SHODH_RATE_LIMIT", default_value_t = 4000)]
    rate_limit: u64,

    /// Maximum concurrent requests before load shedding
    #[arg(long, env = "SHODH_MAX_CONCURRENT", default_value_t = 200)]
    max_concurrent: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    shodh_memory::server::run(shodh_memory::server::ServerRunConfig {
        host: cli.host,
        port: cli.port,
        storage_path: cli.storage_path,
        production: cli.production,
        rate_limit: cli.rate_limit,
        max_concurrent: cli.max_concurrent,
    })
}
