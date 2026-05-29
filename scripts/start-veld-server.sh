#!/usr/bin/env bash
# start-veld-server.sh — operator entrypoint for the Veld memory server.
#
# Manages the locally-built `target/release/veld` binary: start (fg/bg), stop,
# status, restart, rate-limit reset, and a health + rate-limit watchdog that can
# be installed as a launchd agent for persistence across logins.
#
# Env (VELD_* preferred; SHODH_* still accepted by the binary via config alias
# promotion). All have sane defaults:
#   VELD_PORT          (3030)        server port
#   VELD_DEV_API_KEY   (dev key)     API key for /api/* + the watchdog probe
#   VELD_BINARY        (target/release/veld)
#   VELD_MEMORY_PATH   (unset)       storage dir; unset → binary's default_storage_path
#   VELD_RATE_LIMIT    (10000)       governor rps ceiling
#   VELD_RATE_BURST    (20000)       governor burst
#   VELD_WATCHDOG_INTERVAL (30)      seconds between watchdog checks
#   VELD_MAX_RL_RESETS (3)           rate-limit auto-restarts before deferring to a human
#
# NOTE on the rate-limit watchdog: current Veld (src/rate_limit_governance.rs)
# already caps wait_time and exposes an in-process ResetHandle, so the unbounded
# "Wait for {huge N}s" drift should not occur on an up-to-date binary. The
# watchdog's rate-limit auto-recovery is defence-in-depth and also rescues older
# binaries (e.g. vendored 0.5.5+121) whose governor lacks the cap.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ────────────────────────────────────────────────────
PORT="${VELD_PORT:-${SHODH_PORT:-3030}}"
API_KEY="${VELD_DEV_API_KEY:-${SHODH_DEV_API_KEY:-sk-veld-dev-local}}"
BINARY="${VELD_BINARY:-$PROJECT_ROOT/target/release/veld}"
RATE_LIMIT="${VELD_RATE_LIMIT:-${SHODH_RATE_LIMIT:-10000}}"
RATE_BURST="${VELD_RATE_BURST:-${SHODH_RATE_BURST:-20000}}"
WATCHDOG_INTERVAL="${VELD_WATCHDOG_INTERVAL:-30}"
HEALTH_TIMEOUT="${VELD_HEALTH_TIMEOUT:-5}"
MAX_CONSECUTIVE_FAILURES="${VELD_MAX_FAILURES:-2}"
MAX_RL_RESETS="${VELD_MAX_RL_RESETS:-3}"

HEALTH_URL="http://127.0.0.1:$PORT/health"
PID_FILE="$PROJECT_ROOT/.veld-server.pid"
LOG_FILE="$PROJECT_ROOT/.veld-server.log"
WATCHDOG_PID_FILE="$PROJECT_ROOT/.veld-watchdog.pid"
WATCHDOG_AGENT_LABEL="com.portll.veld-watchdog"

# ── Colours / helpers ─────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { printf "${CYAN}[veld]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[veld]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[veld]${NC} %s\n" "$*"; }
error() { printf "${RED}[veld]${NC} %s\n" "$*" >&2; }

get_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid; pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  kill -0 "$pid" 2>/dev/null || { rm -f "$PID_FILE"; return 1; }
  echo "$pid"
}

# Populate the global ENV_ARGS array with the env the binary runs under. An
# array (not word-splitting) keeps space-containing paths intact. VELD_MEMORY_PATH
# is included only when set, so an unset value lets the binary pick its default.
build_env_args() {
  ENV_ARGS=(
    VELD_DEV_API_KEY="$API_KEY"
    VELD_PORT="$PORT"
    VELD_RATE_LIMIT="$RATE_LIMIT"
    VELD_RATE_BURST="$RATE_BURST"
  )
  local mp="${VELD_MEMORY_PATH:-${SHODH_MEMORY_PATH:-}}"
  [ -n "$mp" ] && ENV_ARGS+=(VELD_MEMORY_PATH="$mp")
}

# /health is NOT rate-limited, so it cannot reveal a governor trip on its own.
is_healthy() {
  local r; r=$(curl -sf --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" 2>/dev/null) || return 1
  echo "$r" | grep -q '"status"' 2>/dev/null || return 1
  return 0
}

# Probe a rate-limited /api/* endpoint for the trip signal is_healthy() can't see.
# Returns 0 if the governor is tripped ("Too Many Requests" body), 1 otherwise.
is_rate_limited() {
  local r
  r=$(curl -s --max-time "$HEALTH_TIMEOUT" \
    -X POST "http://127.0.0.1:$PORT/api/relevant" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"user_id":"veld-watchdog-probe","context":"watchdog rate-limit probe","limit":1}' 2>/dev/null) || return 1
  echo "$r" | grep -q "Too Many Requests" 2>/dev/null && return 0
  return 1
}

# ── Commands ──────────────────────────────────────────────────
preflight() {
  if [ ! -x "$BINARY" ]; then
    error "Veld binary not found/executable: $BINARY"
    error "  Build it first:  cargo build --release   (or set VELD_BINARY)"
    exit 1
  fi
}

cmd_start_foreground() {
  preflight
  info "Starting Veld (foreground) on :$PORT — rate limit ${RATE_LIMIT}/s, burst ${RATE_BURST}"
  build_env_args
  exec env "${ENV_ARGS[@]}" "$BINARY"
}

cmd_start_background() {
  preflight
  if get_pid >/dev/null 2>&1; then
    warn "Veld already running (PID $(get_pid))"; return 0
  fi
  info "Starting Veld (background) on :$PORT"
  build_env_args
  nohup env "${ENV_ARGS[@]}" "$BINARY" >> "$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  local i
  for i in $(seq 1 20); do
    is_healthy && { ok "Veld ready (PID $(cat "$PID_FILE")) — log: $LOG_FILE"; return 0; }
    sleep 0.5
  done
  warn "Started (PID $(cat "$PID_FILE")) but health not yet responding — check $LOG_FILE"
}

cmd_stop() {
  # Stop the watchdog first so it does not resurrect the server.
  if [ -f "$WATCHDOG_PID_FILE" ]; then
    kill "$(cat "$WATCHDOG_PID_FILE")" 2>/dev/null || true
    rm -f "$WATCHDOG_PID_FILE"
  fi
  local pid; if pid=$(get_pid); then
    info "Stopping Veld (PID $pid)"
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    ok "Stopped"
  else
    info "Veld not running"
  fi
}

cmd_status() {
  local pid; if pid=$(get_pid); then
    if is_healthy; then ok "Veld running + healthy (PID $pid) on :$PORT"; else warn "Veld process up (PID $pid) but /health not responding"; fi
  else
    warn "Veld not running"
  fi
}

# Clear governor rate-limit state by graceful restart. Persistent memory is
# unaffected. Up-to-date binaries also expose an in-process ResetHandle, but a
# restart is the portable clear that works on any binary version.
cmd_reset_rate_limit() {
  warn "Clearing rate-limit state via graceful restart at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cmd_stop
  sleep 1
  cmd_start_background
}

cmd_watchdog() {
  if [ -f "$WATCHDOG_PID_FILE" ]; then
    local wd; wd=$(cat "$WATCHDOG_PID_FILE")
    if kill -0 "$wd" 2>/dev/null; then warn "Watchdog already running (PID $wd) — restarting it"; kill "$wd" 2>/dev/null || true; sleep 1; fi
    rm -f "$WATCHDOG_PID_FILE"
  fi
  get_pid >/dev/null 2>&1 || cmd_start_background
  info "Starting watchdog (interval ${WATCHDOG_INTERVAL}s, max failures $MAX_CONSECUTIVE_FAILURES, max RL resets $MAX_RL_RESETS)"
  (
    local failures=0
    local rl_trips=0
    trap 'exit 0' TERM INT
    while true; do
      sleep "$WATCHDOG_INTERVAL"
      local pid
      if ! pid=$(get_pid); then
        warn "[watchdog] server gone — restarting"; cmd_start_background; failures=0; continue
      fi
      if is_healthy; then
        # Green /health does not rule out a governor trip (it is not rate-limited).
        # On a trip, do a server-only graceful restart — NOT cmd_reset_rate_limit,
        # whose cmd_stop would also kill this watchdog.
        if is_rate_limited; then
          rl_trips=$((rl_trips + 1))
          if [ "$rl_trips" -le "$MAX_RL_RESETS" ]; then
            warn "[watchdog] rate-limiter tripped (/health green, /api 429) — graceful restart ($rl_trips/$MAX_RL_RESETS) PID $pid"
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WATCHDOG: rate-limit trip — restart to clear governor (PID $pid)" >> "$LOG_FILE"
            kill "$pid" 2>/dev/null || true; sleep 1; rm -f "$PID_FILE"; cmd_start_background
          else
            error "[watchdog] still tripping after $MAX_RL_RESETS restarts — leaving up for manual investigation"
          fi
          failures=0; continue
        fi
        [ "$failures" -gt 0 ] && ok "[watchdog] recovered after $failures failed checks"
        failures=0; rl_trips=0
      else
        failures=$((failures + 1))
        warn "[watchdog] health failed ($failures/$MAX_CONSECUTIVE_FAILURES) PID $pid"
        if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
          error "[watchdog] unresponsive — force-restarting"
          echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WATCHDOG: force-restart after $failures failed checks (PID $pid)" >> "$LOG_FILE"
          kill -9 "$pid" 2>/dev/null || true; sleep 1; rm -f "$PID_FILE"; cmd_start_background; failures=0
        fi
      fi
    done
  ) &
  echo "$!" > "$WATCHDOG_PID_FILE"
  ok "Watchdog started (PID $!)"
}

# Foreground supervisor for launchd: cmd_watchdog backgrounds its loop and
# returns, which launchd would treat as the job exiting. Block on the loop so
# launchd (KeepAlive) can supervise + revive it.
cmd_watchdog_foreground() {
  cmd_watchdog
  local wd; wd=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null) || { error "watchdog PID file missing"; return 1; }
  trap 'kill "$wd" 2>/dev/null || true; exit 0' TERM INT
  while kill -0 "$wd" 2>/dev/null; do sleep "$WATCHDOG_INTERVAL"; done
}

cmd_install_watchdog() {
  local dir="$HOME/Library/LaunchAgents"; local plist="$dir/${WATCHDOG_AGENT_LABEL}.plist"
  mkdir -p "$dir"
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${WATCHDOG_AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${SCRIPT_DIR}/start-veld-server.sh</string>
    <string>--watchdog-fg</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${PROJECT_ROOT}/.veld-watchdog-agent.log</string>
  <key>StandardErrorPath</key><string>${PROJECT_ROOT}/.veld-watchdog-agent.log</string>
</dict>
</plist>
PLIST
  ok "Wrote launchd agent: $plist"
  launchctl unload "$plist" 2>/dev/null || true
  if launchctl load "$plist" 2>/dev/null; then ok "Loaded ${WATCHDOG_AGENT_LABEL}"; else error "launchctl load failed — load manually: launchctl load \"$plist\""; return 1; fi
}

cmd_uninstall_watchdog() {
  local plist="$HOME/Library/LaunchAgents/${WATCHDOG_AGENT_LABEL}.plist"
  launchctl unload "$plist" 2>/dev/null || true
  rm -f "$plist" && ok "Removed ${WATCHDOG_AGENT_LABEL}"
}

# ── Main ──────────────────────────────────────────────────────
case "${1:-foreground}" in
  --bg|bg|background)               cmd_start_background ;;
  --stop|stop)                      cmd_stop ;;
  --status|status)                  cmd_status ;;
  --restart|restart)                cmd_stop; sleep 1; cmd_start_background ;;
  --reset-rl|reset-rl)              cmd_reset_rate_limit ;;
  --watchdog|watchdog)              cmd_watchdog ;;
  --watchdog-fg|watchdog-fg)        cmd_watchdog_foreground ;;
  --install-watchdog|install-watchdog)     cmd_install_watchdog ;;
  --uninstall-watchdog|uninstall-watchdog) cmd_uninstall_watchdog ;;
  --fg|fg|foreground|"")            cmd_start_foreground ;;
  --help|-h|help)
    echo "Usage: $0 [--fg|--bg|--stop|--status|--restart|--reset-rl|--watchdog|--install-watchdog|--uninstall-watchdog]"
    ;;
  *)
    error "Unknown action: $1"
    echo "Usage: $0 [--fg|--bg|--stop|--status|--restart|--reset-rl|--watchdog|--install-watchdog|--uninstall-watchdog]"
    exit 1 ;;
esac
