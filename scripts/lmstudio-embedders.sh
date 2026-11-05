#!/bin/sh
# Keeps LM Studio serving veld's embedders. Idempotent: starts the server if it is not up and loads
# each model that `lms ps` does not list. Run by launchd at login and every five minutes
# (packaging/macos/net.portll.lmstudio-embedders.plist).
#
#   EMBEDDER_MODELS   models to keep loaded (default nomic v1.5 and bge-m3)
#   LMS               the LM Studio CLI (default ~/.lmstudio/bin/lms)
#   LMS_QUERY_SECS    bound on `lms ps` and `lms server status` (default 15)
#   LMS_LOAD_SECS     bound on `lms server start` and on one `lms load` (default 300)
#   LMS_STRAY_SECS    a query of this CLI older than this is a stray and is killed (default 600)
#
# Every lms call runs under an alarm. An unbounded `lms ps` hung for a day on 2026-10-29 while the
# server answered, and launchd starts no new run while one is alive, so the job stalled behind it.
# Strays left by earlier unbounded runs are reaped; a `lms load` is never reaped.
set -u
LMS="${LMS:-$HOME/.lmstudio/bin/lms}"
MODELS="${EMBEDDER_MODELS:-text-embedding-nomic-embed-text-v1.5 text-embedding-bge-m3}"
QUERY_SECS="${LMS_QUERY_SECS:-15}"
LOAD_SECS="${LMS_LOAD_SECS:-300}"
STRAY_SECS="${LMS_STRAY_SECS:-600}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
bounded() { secs=$1; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }

# Queries of "$LMS" older than STRAY_SECS, by pid. ps etime is [[dd-]hh:]mm:ss.
stray_pids() {
    ps -eo pid=,etime=,command= | awk -v max="$STRAY_SECS" -v lms="$LMS" '
        {
            pid = $1; t = $2; cmd = $0
            sub(/^[ \t]*[0-9]+[ \t]+[^ \t]+[ \t]+/, "", cmd)
            if (index(cmd, lms " ps") != 1 && index(cmd, lms " server status") != 1) next
            days = 0
            if (index(t, "-")) { split(t, dt, "-"); days = dt[1]; t = dt[2] }
            n = split(t, hms, ":"); secs = 0
            for (i = 1; i <= n; i++) secs = secs * 60 + hms[i]
            if (days * 86400 + secs >= max) print pid
        }'
}

strays=$(stray_pids | tr '\n' ' ')
if [ -n "$strays" ]; then
    echo "$(stamp) reaping stray lms queries: $strays"
    # shellcheck disable=SC2086 # one argument per pid
    kill $strays 2>/dev/null
fi

if ! bounded "$QUERY_SECS" "$LMS" server status 2>/dev/null | grep -q "running"; then
    echo "$(stamp) server not running; starting"
    bounded "$LOAD_SECS" "$LMS" server start >/dev/null 2>&1 || echo "$(stamp) server start failed"
fi
if ! loaded=$(bounded "$QUERY_SECS" "$LMS" ps 2>/dev/null); then
    echo "$(stamp) lms ps failed or exceeded ${QUERY_SECS}s; loading nothing this run"
    exit 1
fi
for m in $MODELS; do
    if ! printf '%s\n' "$loaded" | grep -q "^$m "; then
        echo "$(stamp) loading $m"
        bounded "$LOAD_SECS" "$LMS" load "$m" --identifier "$m" -y >/dev/null 2>&1 || echo "$(stamp) load failed: $m"
    fi
done
