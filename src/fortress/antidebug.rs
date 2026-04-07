//! Anti-debugging and anti-tampering protections.

/// Check if a debugger is currently attached.
pub fn check_debugger() -> bool {
    platform_check() || timing_check()
}

/// Spawn a background thread that periodically checks for debugger attachment.
pub fn continuous_check() {
    std::thread::Builder::new()
        .name(String::new())
        .stack_size(32 * 1024)
        .spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(3));
            if check_debugger() {
                corrupt_and_exit();
            }
        })
        .ok();
}

fn corrupt_and_exit() -> ! {
    let mut garbage = [0xDEu8; 4096];
    for (i, byte) in garbage.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(0x9E).wrapping_add(0x37);
    }
    unsafe {
        std::ptr::write_volatile(&mut garbage as *mut _, garbage);
    }
    std::process::exit(137);
}

// macOS: check parent process for known debugger names
#[cfg(target_os = "macos")]
fn platform_check() -> bool {
    let ppid = std::process::id();
    if let Ok(output) = std::process::Command::new("/bin/ps")
        .args(["-o", "comm=", "-p"])
        .arg(ppid.to_string())
        .output()
    {
        let comm = String::from_utf8_lossy(&output.stdout);
        let comm_lower = comm.trim().to_lowercase();
        // Known debugger process names
        return ["lldb", "gdb", "debugserver", "dtrace", "instruments", "frida"]
            .iter()
            .any(|d| comm_lower.contains(d));
    }
    false
}

// Linux: check /proc/self/status for TracerPid
#[cfg(target_os = "linux")]
fn platform_check() -> bool {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("TracerPid:") {
                if let Some(pid_str) = line.split(':').nth(1) {
                    if let Ok(pid) = pid_str.trim().parse::<u32>() {
                        return pid != 0;
                    }
                }
            }
        }
    }
    false
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_check() -> bool {
    false
}

/// Timing-based detection: debugger single-stepping causes 1000x+ slowdown.
fn timing_check() -> bool {
    let start = std::time::Instant::now();
    let mut state: u64 = 0x243F6A8885A308D3;
    for i in 0..100u64 {
        state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i);
        std::hint::black_box(&state);
    }
    start.elapsed().as_millis() > 5
}
