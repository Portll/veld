//! Restart a veld instance: stop it gracefully, give whatever started it a moment
//! to bring it back, and otherwise relaunch it from a launch recipe with the
//! environment the stopped process had.
//!
//! The environment matters more than the command line: a server started by a
//! desktop app or a shell carries its embedder, keyring and backend settings in
//! `VELD_*` variables, and a relaunch without them comes up healthy over the wrong
//! embedder. So the new process's environment is layered, later layers winning:
//! the operator's env file (`~/.config/veld/server.env`, as Veld Desktop reads it),
//! the live process's own `VELD_*` variables read before it is stopped (macOS
//! `ps -E`, Linux `/proc`), the recipe's variables, the registry's host, port and
//! data path, and finally the caller's overrides. Values are held in memory only.
//!
//! A supervisor is never guessed from the environment: a process spawned by a
//! launchd job inherits `XPC_SERVICE_NAME` without being the job. It is observed
//! instead, as a new registry entry on the same port after the old pid is gone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::discovery::{discover_instances, DiscoveredInstance};
use crate::{Result, StatusError};

const EXIT_TIMEOUT: Duration = Duration::from_secs(120);
/// Veld Desktop respawns two seconds after exit and the server registers within
/// another second; launchd and systemd are faster. Measured 2026-10-18.
const SUPERVISOR_GRACE: Duration = Duration::from_secs(10);
const READY_TIMEOUT: Duration = Duration::from_secs(90);
const POLL: Duration = Duration::from_millis(500);

/// A way to start a server: the command line, the directory and the variables.
/// Saved by the monitor for an instance; also what [`inspect_process`] returns.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchSpec {
    pub exe: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

pub struct RestartRequest {
    pub base_url: String,
    /// The running instance, from the registry. None starts a stopped server.
    pub instance: Option<DiscoveredInstance>,
    /// A recipe chosen by the caller; else the registry's recipe is used.
    pub recipe: Option<LaunchSpec>,
    /// `KEY=VALUE` files read as the base environment layer, in order.
    pub env_files: Vec<PathBuf>,
    /// Set on the new process over every other layer, e.g.
    /// `VELD_STORAGE_BACKEND` after a backend swap.
    pub env_overrides: Vec<(String, String)>,
    /// Appended with the new process's stdout and stderr.
    pub log_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestartOutcome {
    pub base_url: String,
    /// Pid of the process this call started; None when a supervisor restarted it.
    pub pid: Option<u32>,
    /// The server came back on its own after the stop: something else owns it.
    pub supervised: bool,
    pub log_path: Option<String>,
    /// `VELD_*` names the old process had that could not be given to the new one.
    pub missing_env: Vec<String>,
    /// The variables that were given to the new process, by name.
    pub carried_env: Vec<String>,
    /// Where the recipe came from: `registry`, `recipe` or `supervisor`.
    pub recipe_source: String,
}

fn fail(msg: impl Into<String>) -> StatusError {
    StatusError::Restart(msg.into())
}

/// Keep a variable from the old process only when it configures veld itself.
#[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
fn carried(key: &str) -> bool {
    key.starts_with("VELD_") || key == "RUST_LOG"
}

fn is_var_name(k: &str) -> bool {
    !k.is_empty()
        && !k.starts_with(|c: char| c.is_ascii_digit())
        && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse a `KEY=VALUE` file the way Veld Desktop reads `server.env`: blank and
/// `#` lines skipped, one matching pair of surrounding quotes removed.
pub fn parse_env_file(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|raw| {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            let key = key.trim();
            if !is_var_name(key) {
                return None;
            }
            let value = value.trim();
            let unquoted = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);
            Some((key.to_string(), unquoted.to_string()))
        })
        .collect()
}

/// `text` with `key` set to `value`: the first live `key=` line is rewritten in
/// place and later ones dropped; without one, a line is appended. Comments and
/// every other line are kept byte for byte.
pub fn rewrite_env_var(text: &str, key: &str, value: &str) -> String {
    let mut replaced = false;
    let mut lines: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        let names_key = !line.starts_with('#')
            && line
                .split_once('=')
                .is_some_and(|(k, _)| k.trim() == key);
        if names_key {
            if !replaced {
                lines.push(format!("{key}={value}"));
                replaced = true;
            }
        } else {
            lines.push(raw.to_string());
        }
    }
    if !replaced {
        lines.push(format!("{key}={value}"));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

fn read_env_file(path: &Path) -> Result<Vec<(String, String)>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(parse_env_file(&text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(fail(format!("read {}: {e}", path.display()))),
    }
}

/// `ps -E` reports the exec-time environment of a process of the same user. It is
/// empty for Apple platform binaries, which never run a veld server, and misses
/// variables the server set on itself later (the `config.toml` bridge), which the
/// new process re-reads from the file anyway.
#[cfg(target_os = "macos")]
fn capture_env(pid: u32) -> Option<Vec<(String, String)>> {
    let out = Command::new("ps")
        .args(["-Eww", "-o", "command=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut vars: Vec<(String, String)> = Vec::new();
    for tok in text.trim_end().split(' ') {
        match tok.split_once('=') {
            Some((k, v)) if is_var_name(k) => vars.push((k.to_string(), v.to_string())),
            _ => {
                if let Some((_, v)) = vars.last_mut() {
                    v.push(' ');
                    v.push_str(tok);
                }
            }
        }
    }
    Some(vars.into_iter().filter(|(k, _)| carried(k)).collect())
}

#[cfg(target_os = "linux")]
fn capture_env(pid: u32) -> Option<Vec<(String, String)>> {
    let raw = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    Some(
        raw.split(|b| *b == 0)
            .filter_map(|entry| String::from_utf8(entry.to_vec()).ok())
            .filter_map(|entry| entry.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .filter(|(k, _)| carried(k))
            .collect(),
    )
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn capture_env(_pid: u32) -> Option<Vec<(String, String)>> {
    None
}

/// The command line of a live process: `ps -o command=` without `-E`, so the
/// environment is not in it. Arguments containing spaces come back split.
#[cfg(target_os = "macos")]
fn capture_cmdline(pid: u32) -> Option<Vec<String>> {
    let out = Command::new("ps")
        .args(["-ww", "-o", "command=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let words: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    (!words.is_empty()).then_some(words)
}

#[cfg(target_os = "macos")]
fn capture_cwd(pid: u32) -> Option<String> {
    let out = Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-d", "cwd", "-Fn"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix('n'))
        .map(str::to_string)
}

#[cfg(target_os = "linux")]
fn capture_cmdline(pid: u32) -> Option<Vec<String>> {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let words: Vec<String> = raw
        .split(|b| *b == 0)
        .filter(|w| !w.is_empty())
        .map(|w| String::from_utf8_lossy(w).to_string())
        .collect();
    (!words.is_empty()).then_some(words)
}

#[cfg(target_os = "linux")]
fn capture_cwd(pid: u32) -> Option<String> {
    std::fs::read_link(format!("/proc/{pid}/cwd"))
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn capture_cmdline(_pid: u32) -> Option<Vec<String>> {
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn capture_cwd(_pid: u32) -> Option<String> {
    None
}

/// How a live process was started, read from the OS: the recipe a server from
/// before the registry's `launch` field can be restarted with. Blocking.
pub fn inspect_process(pid: u32) -> Option<LaunchSpec> {
    let mut words = capture_cmdline(pid)?;
    let exe = words.remove(0);
    Some(LaunchSpec {
        exe,
        args: words,
        cwd: capture_cwd(pid),
        env: capture_env(pid).unwrap_or_default().into_iter().collect(),
    })
}

fn pid_alive(pid: u32) -> bool {
    if cfg!(windows) {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    } else {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// SIGTERM on Unix, which the server answers with its graceful shutdown; forced on
/// Windows, where it has no signal handler.
fn terminate(pid: u32) -> Result<()> {
    let ok = if cfg!(windows) {
        Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
    } else {
        Command::new("kill").arg(pid.to_string()).status()
    }
    .map(|s| s.success())
    .unwrap_or(false);
    if ok {
        Ok(())
    } else {
        Err(fail(format!("could not terminate pid {pid}")))
    }
}

async fn healthy(http: &reqwest::Client, base_url: &str) -> bool {
    http.get(format!("{base_url}/health"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// A live registry entry on `port` from a process other than `old_pid`.
fn reregistered(port: u16, old_pid: u32) -> Option<u32> {
    discover_instances()
        .into_iter()
        .find(|i| i.port == port && i.pid != 0 && i.pid != old_pid)
        .map(|i| i.pid)
}

async fn wait_for_exit(pid: u32) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < EXIT_TIMEOUT {
        if !tokio::task::spawn_blocking(move || pid_alive(pid))
            .await
            .map_err(|e| fail(e.to_string()))?
        {
            return Ok(());
        }
        tokio::time::sleep(POLL).await;
    }
    Err(fail(format!(
        "pid {pid} is still running {}s after being asked to stop",
        EXIT_TIMEOUT.as_secs()
    )))
}

async fn wait_for_health(http: &reqwest::Client, base_url: &str) -> bool {
    let start = Instant::now();
    while start.elapsed() < READY_TIMEOUT {
        if healthy(http, base_url).await {
            return true;
        }
        tokio::time::sleep(POLL).await;
    }
    false
}

fn open_log(path: &Path) -> Result<std::fs::File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| fail(format!("create {}: {e}", dir.display())))?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| fail(format!("open {}: {e}", path.display())))
}

fn spawn_detached(mut cmd: Command, log: std::fs::File) -> Result<std::process::Child> {
    let err = log
        .try_clone()
        .map_err(|e| fail(format!("clone log handle: {e}")))?;
    cmd.stdin(Stdio::null()).stdout(log).stderr(err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }
    cmd.spawn().map_err(|e| fail(format!("spawn: {e}")))
}

fn reap_in_background(mut child: std::process::Child) {
    std::thread::spawn(move || {
        let _ = child.wait();
    });
}

/// Stop the instance, if one is running, and bring it back. Resolves once
/// `/health` answers again or with the reason it did not.
pub async fn restart_instance(req: RestartRequest) -> Result<RestartOutcome> {
    let base_url = req.base_url.trim_end_matches('/').to_string();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    let mut captured: Vec<(String, String)> = Vec::new();
    if let Some(inst) = &req.instance {
        let pid = inst.pid;
        if pid == 0 {
            return Err(fail("the registry entry carries no pid"));
        }
        captured = tokio::task::spawn_blocking(move || capture_env(pid))
            .await
            .map_err(|e| fail(e.to_string()))?
            .unwrap_or_default();

        terminate(pid)?;
        wait_for_exit(pid).await?;

        let port = inst.port;
        let grace_end = Instant::now() + SUPERVISOR_GRACE;
        while Instant::now() < grace_end {
            let back = tokio::task::spawn_blocking(move || reregistered(port, pid))
                .await
                .map_err(|e| fail(e.to_string()))?;
            if back.is_some() || healthy(&http, &base_url).await {
                if !wait_for_health(&http, &base_url).await {
                    return Err(fail(format!(
                        "something restarted the server (pid {}) but it has not answered /health after {}s",
                        back.map(|p| p.to_string()).unwrap_or_else(|| "unknown".into()),
                        READY_TIMEOUT.as_secs()
                    )));
                }
                return Ok(RestartOutcome {
                    base_url,
                    pid: None,
                    supervised: true,
                    log_path: None,
                    missing_env: Vec::new(),
                    carried_env: Vec::new(),
                    recipe_source: "supervisor".into(),
                });
            }
            tokio::time::sleep(POLL).await;
        }
    } else if healthy(&http, &base_url).await {
        return Err(fail(format!(
            "{base_url} answers /health but is not in the instance registry; it is a server too old to register, stop it by hand first"
        )));
    }

    let registry_recipe = req.instance.as_ref().and_then(|i| i.launch.as_ref());
    let (recipe, recipe_source) = match (&req.recipe, registry_recipe) {
        (Some(spec), _) => (spec.clone(), "recipe"),
        (None, Some(r)) => (
            LaunchSpec {
                exe: r.exe.clone(),
                args: r.args.clone(),
                cwd: r.cwd.clone(),
                env: BTreeMap::new(),
            },
            "registry",
        ),
        (None, None) => {
            return Err(fail(if req.instance.is_some() {
                "stopped, but the registry entry has no launch recipe (written by servers from 0.7.40 on) and none is saved for this instance: save one in the Launch card, or start it by hand with `veld server`"
            } else {
                "no launch recipe is saved for this instance and it is not running: save one in the Launch card"
            }));
        }
    };

    let mut cmd = Command::new(&recipe.exe);
    cmd.args(&recipe.args);
    if let Some(cwd) = recipe.cwd.as_deref().filter(|d| Path::new(d).is_dir()) {
        cmd.current_dir(cwd);
    }
    let mut given: Vec<String> = Vec::new();
    let mut layer = |cmd: &mut Command, key: &str, value: &str| {
        if !value.is_empty() {
            cmd.env(key, value);
            given.push(key.to_string());
        }
    };
    for file in &req.env_files {
        for (k, v) in read_env_file(file)? {
            layer(&mut cmd, &k, &v);
        }
    }
    for (k, v) in &captured {
        layer(&mut cmd, k, v);
    }
    for (k, v) in &recipe.env {
        layer(&mut cmd, k, v);
    }
    if let Some(inst) = &req.instance {
        layer(&mut cmd, "VELD_HOST", &inst.host);
        layer(&mut cmd, "VELD_PORT", &inst.port.to_string());
        layer(&mut cmd, "VELD_MEMORY_PATH", &inst.data_path);
    }
    for (k, v) in &req.env_overrides {
        layer(&mut cmd, k, v);
    }
    let missing_env: Vec<String> = registry_recipe
        .map(|r| r.env_names.clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|n| !given.contains(n) && std::env::var_os(n).is_none())
        .collect();
    let mut carried_env = given;
    carried_env.sort();
    carried_env.dedup();

    let log = open_log(&req.log_path)?;
    let mut child = spawn_detached(cmd, log)?;
    let new_pid = child.id();

    let start = Instant::now();
    while start.elapsed() < READY_TIMEOUT {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(fail(format!(
                "the new server (pid {new_pid}) exited with {status} before answering /health; see {}",
                req.log_path.display()
            )));
        }
        if healthy(&http, &base_url).await {
            reap_in_background(child);
            return Ok(RestartOutcome {
                base_url,
                pid: Some(new_pid),
                supervised: false,
                log_path: Some(req.log_path.display().to_string()),
                missing_env,
                carried_env,
                recipe_source: recipe_source.into(),
            });
        }
        tokio::time::sleep(POLL).await;
    }
    reap_in_background(child);
    Err(fail(format!(
        "the new server (pid {new_pid}) has not answered /health after {}s; it is still running, see {}",
        READY_TIMEOUT.as_secs(),
        req.log_path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_veld_configuration_is_carried() {
        assert!(carried("VELD_EMBEDDING_API_URL"));
        assert!(carried("RUST_LOG"));
        assert!(!carried("PATH"));
        assert!(!carried("HOME"));
    }

    #[test]
    fn env_file_parses_like_veld_desktop() {
        let text = "# comment\n\nVELD_EMBEDDER=http\nVELD_URL=\"http://x:1\"  \n  QUOTED='a b'\n9BAD=1\nnoeq\n";
        let vars = parse_env_file(text);
        assert_eq!(
            vars,
            vec![
                ("VELD_EMBEDDER".to_string(), "http".to_string()),
                ("VELD_URL".to_string(), "http://x:1".to_string()),
                ("QUOTED".to_string(), "a b".to_string()),
            ]
        );
    }

    #[test]
    fn rewrite_keeps_every_other_line() {
        let text = "# live store\nVELD_STORAGE_BACKEND=rocksdb\nVELD_EMBEDDER=http\n";
        let out = rewrite_env_var(text, "VELD_STORAGE_BACKEND", "redb");
        assert_eq!(out, "# live store\nVELD_STORAGE_BACKEND=redb\nVELD_EMBEDDER=http\n");
        let appended = rewrite_env_var("VELD_EMBEDDER=http", "VELD_STORAGE_BACKEND", "redb");
        assert_eq!(appended, "VELD_EMBEDDER=http\nVELD_STORAGE_BACKEND=redb\n");
        let deduped = rewrite_env_var("A=1\nA=2\n", "A", "3");
        assert_eq!(deduped, "A=3\n");
    }

    // Live run against a scratch server named by VELD_RESTART_E2E_BASE_URL: inspect
    // it, stop it, relaunch it from its registry recipe, and leave the new one up.
    //   cargo test -- --ignored e2e_restart --nocapture
    #[tokio::test]
    #[ignore]
    async fn e2e_restart_relaunches_a_scratch_server() {
        let Ok(base_url) = std::env::var("VELD_RESTART_E2E_BASE_URL") else {
            eprintln!("VELD_RESTART_E2E_BASE_URL not set; skipping");
            return;
        };
        let key = base_url.trim_end_matches('/').to_string();
        let inst = discover_instances()
            .into_iter()
            .find(|i| i.base_url.trim_end_matches('/') == key)
            .expect("the scratch server is in the registry");
        let old_pid = inst.pid;
        let spec = inspect_process(old_pid).expect("inspect the scratch server");
        eprintln!("inspected: {} {:?} cwd={:?} env={:?}", spec.exe, spec.args, spec.cwd, spec.env.keys().collect::<Vec<_>>());
        assert!(spec.exe.ends_with("veld"));
        assert_eq!(spec.args, vec!["server".to_string()]);
        assert!(spec.env.contains_key("VELD_PORT"));

        let home = std::env::var("HOME").unwrap_or_default();
        let env_file = PathBuf::from(home).join(".config").join("veld").join("server.env");
        let out = restart_instance(RestartRequest {
            base_url: key.clone(),
            instance: Some(inst),
            recipe: None,
            env_files: if env_file.is_file() { vec![env_file] } else { vec![] },
            env_overrides: vec![],
            log_path: std::env::temp_dir().join("veld-restart-e2e.log"),
        })
        .await
        .expect("restart succeeds");
        eprintln!("outcome: {out:?}");
        assert!(!out.supervised, "nothing supervises the scratch server");
        assert_eq!(out.recipe_source, "registry");
        let new_pid = out.pid.expect("a pid was started");
        assert_ne!(new_pid, old_pid);
        // Nothing the old process was actually started with may be dropped. Names the
        // server set on itself after exec can show up here from older registry entries.
        for name in &out.missing_env {
            assert!(!spec.env.contains_key(name), "{name} was in the old environment but not carried");
        }
        assert!(out.carried_env.iter().any(|k| k == "VELD_PORT"));
        let back = discover_instances()
            .into_iter()
            .find(|i| i.base_url.trim_end_matches('/') == key)
            .expect("the relaunched server registered");
        assert_eq!(back.pid, new_pid);
        assert!(back.launch.is_some());
    }

    // Run only as the child of the test below: stays alive while it is inspected.
    #[test]
    #[ignore]
    fn child_waits_for_capture() {
        if std::env::var_os("VELD_RESTART_TEST_CHILD").is_some() {
            std::thread::sleep(Duration::from_secs(20));
        }
    }

    // The OS reports the environment a process was exec'd with, never one changed
    // later from inside, and macOS hides it for platform binaries such as /bin/sleep.
    // So the child is this test binary, which is neither.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn inspects_a_live_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let dir = std::env::temp_dir();
        let mut child = Command::new(&exe)
            .args(["restart::tests::child_waits_for_capture", "--exact", "--ignored"])
            .current_dir(&dir)
            .env("VELD_RESTART_TEST_CHILD", "1")
            .env("VELD_RESTART_TEST_VAR", "two words")
            .env("UNRELATED_TEST_VAR", "x")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn the test binary");
        let spec = inspect_process(child.id());
        let _ = child.kill();
        let _ = child.wait();
        let spec = spec.expect("read the child's command line and environment");
        assert_eq!(spec.exe, exe.to_string_lossy());
        assert_eq!(spec.args[0], "restart::tests::child_waits_for_capture");
        assert_eq!(spec.env.get("VELD_RESTART_TEST_VAR").map(String::as_str), Some("two words"));
        assert!(spec.env.keys().all(|k| carried(k)));
        let cwd = spec.cwd.expect("cwd is readable");
        let canonical = |p: &str| std::fs::canonicalize(p).unwrap_or_else(|_| PathBuf::from(p));
        assert_eq!(canonical(&cwd), canonical(&dir.to_string_lossy()));
    }
}
