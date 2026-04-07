//! Custom panic handler that reveals nothing about the binary's internals.
//!
//! In fortress mode, panics produce zero useful reverse-engineering information:
//! - No source file paths
//! - No line numbers
//! - No function names
//! - No panic message content
//! - Just an opaque error code and immediate abort

use std::sync::atomic::{AtomicU64, Ordering};

static PANIC_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Install the fortress panic handler. Replaces the default Rust panic handler
/// with one that emits only an opaque error code and immediately aborts.
pub fn install_panic_handler() {
    std::panic::set_hook(Box::new(|_info| {
        let n = PANIC_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Opaque error code — hash of counter prevents correlation
        let code = n.wrapping_mul(0x517cc1b727220a95) ^ 0x6c62272e07bb0142;
        let hex = format!("Ex{:016x}\n", code);
        // Write to stderr without format machinery
        let _ = std::io::Write::write_all(&mut std::io::stderr(), hex.as_bytes());
        std::process::abort();
    }));
}
