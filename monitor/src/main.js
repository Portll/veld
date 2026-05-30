// Veld Monitor frontend — minimal vanilla JS.
// Subscribes to "snapshot-updated" events from the Rust side, falls back to
// polling get_snapshot() if the event channel hasn't delivered for >5s.

// ─── Diagnostics ────────────────────────────────────────────────────────────
// Accumulated log of script-level events. Hidden by default; toggle with
// Ctrl+D (or Cmd+D on macOS) to pop it out as an overlay. Errors auto-pop.

const diagLog = [];
let diagAutoShow = false;

function diagPush(level, msg) {
  diagLog.push(`[${new Date().toLocaleTimeString()}] ${level} ${msg}`);
  if (diagLog.length > 200) diagLog.shift();
  if (level === "ERROR") {
    diagAutoShow = true;
    renderDiag();
  } else if (isDiagVisible()) {
    renderDiag();
  }
}

function isDiagVisible() {
  const el = document.getElementById("__diag");
  return !!el && el.style.display !== "none";
}

function renderDiag() {
  let el = document.getElementById("__diag");
  if (!el) {
    el = document.createElement("pre");
    el.id = "__diag";
    el.style.cssText =
      "color:#fff;background:#7a1f1f;font:11px/1.4 monospace;padding:8px;margin:0;" +
      "white-space:pre-wrap;position:fixed;top:0;left:0;right:0;z-index:99999;" +
      "border-bottom:1px solid #000;max-height:40vh;overflow:auto;cursor:pointer;";
    el.title = "Click or press Ctrl+D to dismiss";
    el.addEventListener("click", () => {
      diagAutoShow = false;
      el.style.display = "none";
    });
    (document.body || document.documentElement).prepend(el);
  }
  el.textContent = diagLog.join("\n");
  el.style.display = "block";
}

function toggleDiag() {
  const el = document.getElementById("__diag");
  if (!el || el.style.display === "none") {
    renderDiag();
  } else {
    el.style.display = "none";
    diagAutoShow = false;
  }
}

window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d") {
    e.preventDefault();
    toggleDiag();
  }
});

// Back-compat shim — older call sites still write to showDebug.
function showDebug(msg) {
  diagPush("ERROR", msg);
}

diagPush("info", "main.js loaded — userAgent: " + navigator.userAgent);

window.addEventListener("error", (e) => {
  diagPush("ERROR", `${e.message}\n  at ${e.filename}:${e.lineno}:${e.colno}`);
});
window.addEventListener("unhandledrejection", (e) => {
  diagPush("ERROR", `Unhandled rejection: ${e.reason}`);
});

if (!window.__TAURI__) {
  diagPush(
    "ERROR",
    "window.__TAURI__ is undefined — `withGlobalTauri: true` is missing or didn't take effect."
  );
  throw new Error("Tauri global not injected");
}
diagPush("info", "__TAURI__ keys: " + Object.keys(window.__TAURI__).join(", "));

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

let lastUpdateAt = 0;

function fmtCount(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

function fmtUptime(secs) {
  if (secs == null) return "—";
  if (secs < 60) return secs + "s";
  if (secs < 3600) return Math.floor(secs / 60) + "m " + (secs % 60) + "s";
  if (secs < 86400) {
    return Math.floor(secs / 3600) + "h " + Math.floor((secs % 3600) / 60) + "m";
  }
  return Math.floor(secs / 86400) + "d " + Math.floor((secs % 86400) / 3600) + "h";
}

function stateKind(state) {
  if (!state) return "unknown";
  if (typeof state === "string") return state;
  return state.kind || "unknown";
}

function render(snap) {
  lastUpdateAt = Date.now();
  const state = stateKind(snap.server.state);

  // Header
  const dot = document.getElementById("dot");
  dot.className = "dot " + state;
  document.getElementById("state-label").textContent = state;
  document.getElementById("base-url").textContent = snap.base_url || "";
  document.getElementById("rtt").textContent =
    snap.server.rtt_ms != null ? snap.server.rtt_ms + " ms" : "— ms";
  // Compute uptime from first_seen on the JS side because the server-derived
  // value isn't included in the wire snapshot (it's a method on ServerHealth).
  let uptimeSecs = null;
  if (snap.server.first_seen) {
    uptimeSecs = Math.max(
      0,
      Math.floor((Date.now() - new Date(snap.server.first_seen).getTime()) / 1000)
    );
  }
  document.getElementById("uptime").textContent = fmtUptime(uptimeSecs);
  document.getElementById("version").textContent = snap.server.version || "—";

  // Memory
  const m = snap.memory;
  const memGrid = document.getElementById("memory-grid");
  memGrid.innerHTML = "";
  appendField(memGrid, "Total", fmtCount(m.total));
  appendField(memGrid, "Working", fmtCount(m.working));
  appendField(memGrid, "Session", fmtCount(m.session));
  appendField(memGrid, "Long-term", fmtCount(m.long_term));
  appendField(memGrid, "Retrievals", fmtCount(m.total_retrievals));

  const stripe = document.getElementById("index-stripe");
  stripe.className = "index-stripe " + (m.index_healthy ? "healthy" : "lag");
  stripe.textContent = (m.index_healthy ? "index healthy" : "index lag") +
    `  vec=${m.vector_index}/${m.total}`;

  // Todos
  const t = snap.todos;
  const todosGrid = document.getElementById("todos-grid");
  todosGrid.innerHTML = "";
  appendField(todosGrid, "Total", String(t.total));
  appendField(todosGrid, "In progress", String(t.in_progress));
  appendField(todosGrid, "Blocked", String(t.blocked));
  appendField(todosGrid, "Overdue", String(t.overdue));
  appendField(todosGrid, "Done", String(t.done));

  // Sessions
  const sessionsEl = document.getElementById("sessions-list");
  sessionsEl.innerHTML = "";
  if (!snap.sessions.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "no active Claude Code sessions";
    sessionsEl.appendChild(p);
  } else {
    for (const s of snap.sessions) renderSession(sessionsEl, s);
  }

  // Activity
  const activityEl = document.getElementById("activity-list");
  activityEl.innerHTML = "";
  if (!snap.recent.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "waiting for events…";
    activityEl.appendChild(li);
  } else {
    for (const e of snap.recent.slice(0, 80)) renderActivity(activityEl, e);
  }
}

function appendField(grid, label, value) {
  const l = document.createElement("span");
  l.className = "label";
  l.textContent = label;
  const v = document.createElement("span");
  v.className = "value";
  v.textContent = value;
  grid.appendChild(l);
  grid.appendChild(v);
}

function renderSession(parent, s) {
  const wrap = document.createElement("div");
  wrap.className = "session";

  const header = document.createElement("div");
  header.className = "session-header";
  const idSpan = document.createElement("span");
  idSpan.className = "id";
  idSpan.textContent = (s.session_id || "?").slice(0, 8);
  const modelSpan = document.createElement("span");
  modelSpan.className = "model";
  modelSpan.textContent = s.model || "?";
  const taskSpan = document.createElement("span");
  taskSpan.className = "task";
  taskSpan.textContent = s.current_task || "";
  header.append(idSpan, modelSpan, taskSpan);

  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("div");
  const pct = Math.min(100, Math.max(0, s.percent_used || 0));
  const tone = pct < 50 ? "" : pct < 80 ? "warn" : "danger";
  fill.className = "bar-fill" + (tone ? " " + tone : "");
  fill.style.width = pct + "%";
  fill.title = `${pct}%  ${(s.tokens_used / 1000).toFixed(1)}k / ${(s.tokens_budget / 1000).toFixed(0)}k`;
  bar.appendChild(fill);

  wrap.append(header, bar);
  parent.appendChild(wrap);
}

function renderActivity(parent, e) {
  const li = document.createElement("li");
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = relTime(e.timestamp);
  const kind = document.createElement("span");
  kind.className = "kind ev-" + e.event_type.toLowerCase();
  kind.textContent = e.event_type;
  const preview = document.createElement("span");
  preview.className = "preview";
  const mt = e.memory_type ? `[${e.memory_type}] ` : "";
  preview.textContent = mt + (e.preview || "");
  li.append(ts, kind, preview);
  parent.appendChild(li);
}

function relTime(iso) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.floor((Date.now() - t) / 1000);
  if (diff < 0) return new Date(t).toLocaleTimeString().slice(0, 5);
  if (diff < 60) return diff + "s";
  if (diff < 3600) return Math.floor(diff / 60) + "m";
  if (diff < 86400) return Math.floor(diff / 3600) + "h";
  return Math.floor(diff / 86400) + "d";
}

// ---- Cognitive Dials ----------------------------------------------------
let currentDials = null;

const DIAL_TOGGLES = [
  ["reconsolidation_enabled", "Reconsolidation (master)"],
  ["re_encode_graph_strengthen", "Re-encode · graph strengthen"],
  ["re_encode_entity_enrich", "Re-encode · entity enrich"],
  ["re_encode_embedding_nudge", "Re-encode · embedding nudge"],
];
const DIAL_NUMBERS = [
  ["reconsolidation_window_secs", "Labile window (s)", 30, 3600, 1],
  ["graph_boost_scale", "Graph boost scale", 0, 1, 0.05],
  ["embedding_blend", "Embedding blend", 0, 0.5, 0.01],
];

function fmtOpt(v) {
  return v === null || v === undefined ? "—" : String(v);
}

function dialsStatus(msg, isError) {
  const el = document.getElementById("dials-status");
  if (!el) return;
  el.textContent = msg || "";
  el.style.color = isError ? "#dc5a5a" : "";
}

async function saveDials(updated) {
  dialsStatus("saving…", false);
  try {
    currentDials = await invoke("set_dials", { dials: updated });
    renderDialsBody();
    dialsStatus("saved", false);
    setTimeout(() => dialsStatus("", false), 1500);
  } catch (err) {
    dialsStatus("save failed: " + (err?.message || err), true);
  }
}

function renderDialsBody() {
  const body = document.getElementById("dials-body");
  if (!body) return;
  if (!currentDials) {
    body.innerHTML = '<p class="muted">no dials</p>';
    return;
  }
  body.innerHTML = "";

  for (const [key, label] of DIAL_TOGGLES) {
    const row = document.createElement("label");
    row.className = "dial-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!currentDials[key];
    cb.addEventListener("change", () =>
      saveDials({ ...currentDials, [key]: cb.checked })
    );
    const span = document.createElement("span");
    span.textContent = label;
    row.append(cb, span);
    body.appendChild(row);
  }

  for (const [key, label, min, max, step] of DIAL_NUMBERS) {
    const row = document.createElement("label");
    row.className = "dial-row";
    const span = document.createElement("span");
    span.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.min = String(min);
    inp.max = String(max);
    inp.step = String(step);
    inp.value = currentDials[key];
    inp.addEventListener("change", () => {
      const v = parseFloat(inp.value);
      if (Number.isNaN(v)) return;
      saveDials({ ...currentDials, [key]: v });
    });
    row.append(span, inp);
    body.appendChild(row);
  }

  const ro = document.createElement("p");
  ro.className = "muted dial-ro";
  ro.textContent =
    "retrieval: candidate_count=" + fmtOpt(currentDials.candidate_count) +
    " · rerank_count=" + fmtOpt(currentDials.rerank_count) +
    " · use_reranking=" + fmtOpt(currentDials.use_reranking) +
    " · refiner_mode=" + fmtOpt(currentDials.refiner_mode);
  body.appendChild(ro);
}

async function pullDials() {
  try {
    currentDials = await invoke("get_dials");
    renderDialsBody();
  } catch (err) {
    const body = document.getElementById("dials-body");
    if (body)
      body.innerHTML =
        '<p class="muted">dials unavailable: ' + (err?.message || err) + "</p>";
  }
}

async function pull() {
  try {
    const snap = await invoke("get_snapshot");
    diagPush(
      "info",
      `pull ok — state=${JSON.stringify(snap.server?.state)} ` +
        `total=${snap.memory?.total} user=${snap.user_id}`
    );
    render(snap);
  } catch (err) {
    diagPush("ERROR", "invoke(get_snapshot) failed: " + (err?.message || err));
    console.error("get_snapshot failed", err);
  }
}

async function init() {
  try {
    await listen("snapshot-updated", (event) => render(event.payload));
    diagPush("info", "listen('snapshot-updated') subscribed");
  } catch (err) {
    diagPush(
      "ERROR",
      "listen('snapshot-updated') failed: " + (err?.message || err)
    );
    return;
  }
  await pull();

  const dialsRefresh = document.getElementById("dials-refresh");
  if (dialsRefresh) dialsRefresh.addEventListener("click", pullDials);
  await pullDials();

  setInterval(() => {
    if (Date.now() - lastUpdateAt > 5000) pull();
  }, 2000);
}

init();
