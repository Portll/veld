//! Fortress: fractal binary obfuscation for distribution builds.
//!
//! Build: `cargo build --profile fortress --features fortress`
//!
//! Layers of protection (fractal — obfuscation at every scale):
//!
//! **Layer 0 — Compile-time string encryption:**
//! All sensitive strings (API routes, env vars, error messages) are XOR-encrypted
//! at compile time with a per-build random key. The `obf!()` macro encrypts at
//! compile time, decrypts at runtime. `strings` on the binary reveals nothing.
//!
//! **Layer 1 — Compiler hardening (Cargo profile):**
//! LTO=fat + codegen-units=1 + opt-level=3 monomorphizes the entire program into
//! a single LLVM module. Function boundaries are destroyed by inlining. No unwind
//! tables (panic=abort). No overflow checks. No debug info. No symbols.
//!
//! **Layer 2 — Runtime protection:**
//! Custom panic handler (no source paths/line numbers). Anti-debug detection
//! (ptrace/sysctl). Binary integrity verification (embedded hash).
//!
//! **Layer 3 — Trace elimination:**
//! In fortress mode, ALL tracing macros compile to no-ops. Zero log strings
//! survive in the binary. Error messages are replaced with numeric codes.
//!
//! **Layer 4 — Post-build processing (scripts/fortress-build.sh):**
//! Additional strip passes, Mach-O/ELF metadata cleanup, UPX packing,
//! verification via `strings` to confirm no leaks.

mod antidebug;
mod panic;
mod strings;

pub use antidebug::check_debugger;
pub use panic::install_panic_handler;
pub use strings::*;

/// Initialize all fortress protections. Call at the very start of main().
pub fn init() {
    install_panic_handler();
    antidebug::continuous_check();
}
