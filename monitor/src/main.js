// Veld Monitor frontend — minimal vanilla JS.
// Subscribes to "snapshot-updated" events from the Rust side, falls back to
// polling get_snapshot() if the event channel hasn't delivered for >5s.

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

async function pull() {
  try {
    const snap = await invoke("get_snapshot");
    render(snap);
  } catch (err) {
    console.error("get_snapshot failed", err);
  }
}

async function init() {
  await listen("snapshot-updated", (event) => render(event.payload));
  await pull();
  // Safety-net polling: if the event channel goes >5s without delivery, pull.
  setInterval(() => {
    if (Date.now() - lastUpdateAt > 5000) pull();
  }, 2000);
}

init();
