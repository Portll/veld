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
let currentBaseUrl = "";
let currentUserId = "";
// Multi-instance: the fleet of discovered veld instances + which tab is active.
let instances = [];
let activeBaseUrl = "";

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
  if (snap.base_url) currentBaseUrl = snap.base_url;
  if (snap.user_id) currentUserId = snap.user_id;
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

// Each dial carries display metadata: a one-line `desc` shown under the control,
// and `info` (suggested values for different uses) surfaced via the ⓘ hover tooltip.
const DIAL_TOGGLES = [
  {
    key: "reconsolidation_enabled",
    label: "Reconsolidation (master)",
    desc: "Closes the labile window and re-encodes recalled memories during background maintenance.",
    info: "On = adaptive consolidation runs in maintenance. Off = fully inert (default). Master gate for the layers below.",
  },
  {
    key: "re_encode_graph_strengthen",
    label: "Re-encode · graph strengthen",
    desc: "Strengthens graph edges between a recalled memory and the entities in its retrieval context.",
    info: "On for richer associative recall over time. Off to keep edges stable / if the graph is noisy.",
  },
  {
    key: "re_encode_entity_enrich",
    label: "Re-encode · entity enrich",
    desc: "Adds newly-seen retrieval-context entities to the memory's own entity set.",
    info: "On to grow associations as a memory is re-used. Off to keep each memory's entities fixed.",
  },
  {
    key: "re_encode_embedding_nudge",
    label: "Re-encode · embedding nudge",
    desc: "Nudges a memory's embedding toward how it has actually been retrieved (semantic drift).",
    info: "On for usage-adaptive recall (heaviest layer). Off to keep embeddings fixed (safest, default).",
  },
];
const DIAL_NUMBERS = [
  {
    key: "reconsolidation_window_secs",
    label: "Labile window",
    unit: "s",
    min: 30,
    max: 3600,
    step: 1,
    desc: "How long a recalled memory stays re-encodable before the window closes and re-encode fires.",
    info: "60 = rapid testing · 300 = interactive chat (default) · 1800 = long research · 3600 = max.",
  },
  {
    key: "graph_boost_scale",
    label: "Graph boost scale",
    unit: "",
    min: 0,
    max: 1,
    step: 0.05,
    desc: "Strength of the graph-edge reinforcement applied when the window closes.",
    info: "0.2 = gentle · 0.5 = balanced (default) · 1.0 = aggressive edge strengthening.",
  },
  {
    key: "embedding_blend",
    label: "Embedding blend",
    unit: "",
    min: 0,
    max: 0.5,
    step: 0.01,
    desc: "Fraction the embedding moves toward the retrieval context on each nudge.",
    info: "0.05 = subtle (default) · 0.1 = moderate · 0.3 = strong drift (use with care).",
  },
  {
    key: "max_active_shadows",
    label: "Max active shadows",
    unit: "",
    min: 1,
    max: 500,
    step: 1,
    desc: "How many memories can be in the labile (re-encodable) window at once before the oldest are evicted.",
    info: "20 = default · higher = more parallel reconsolidation (more memory/CPU) · lower = tighter focus.",
  },
  {
    key: "working_memory_threshold",
    label: "Working-memory threshold",
    unit: "",
    min: 1,
    max: 100,
    step: 1,
    desc: "Recall count at/above which a memory is kept labile as 'working memory' instead of closing at the window.",
    info: "5 = default · lower = close windows sooner · higher = hold frequently-recalled memories labile longer.",
  },
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
    currentDials = await invoke("set_dials_for", {
      baseUrl: activeBaseUrl,
      dials: updated,
    });
    renderDialsBody();
    dialsStatus("saved", false);
    setTimeout(() => dialsStatus("", false), 1500);
  } catch (err) {
    dialsStatus("save failed: " + (err?.message || err), true);
  }
}

// A small ⓘ glyph whose native tooltip carries the "suggested values" text.
function makeInfoIcon(text) {
  const icon = document.createElement("span");
  icon.className = "dial-info";
  icon.textContent = "ⓘ";
  icon.title = text;
  icon.setAttribute("aria-label", text);
  icon.tabIndex = 0;
  return icon;
}

// "<label> ⓘ" — the info icon sits immediately after every label.
function makeDialLabel(label, info) {
  const wrap = document.createElement("span");
  wrap.className = "dial-label";
  const name = document.createElement("span");
  name.className = "dial-name";
  name.textContent = label;
  wrap.append(name, makeInfoIcon(info));
  return wrap;
}

function renderDialsBody() {
  const body = document.getElementById("dials-body");
  if (!body) return;
  if (!currentDials) {
    body.innerHTML = '<p class="muted">no dials</p>';
    return;
  }
  body.innerHTML = "";

  for (const d of DIAL_TOGGLES) {
    const dial = document.createElement("div");
    dial.className = "dial";

    // Head row: label + ⓘ on the left, checkbox pushed to the right.
    const head = document.createElement("label");
    head.className = "dial-head";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "dial-toggle";
    cb.checked = !!currentDials[d.key];
    cb.addEventListener("change", () =>
      saveDials({ ...currentDials, [d.key]: cb.checked })
    );
    head.append(makeDialLabel(d.label, d.info), cb);

    const desc = document.createElement("p");
    desc.className = "dial-desc";
    desc.textContent = d.desc;

    dial.append(head, desc);
    body.appendChild(dial);
  }

  for (const d of DIAL_NUMBERS) {
    const dial = document.createElement("div");
    dial.className = "dial";

    const head = document.createElement("div");
    head.className = "dial-head";

    // Control: range bracket to the LEFT of the input field.
    const control = document.createElement("div");
    control.className = "dial-control";
    const range = document.createElement("span");
    range.className = "dial-range";
    range.textContent =
      "[" + d.min + "–" + d.max + (d.unit ? " " + d.unit : "") + "]";
    const inp = document.createElement("input");
    inp.type = "number";
    inp.className = "dial-number";
    inp.min = String(d.min);
    inp.max = String(d.max);
    inp.step = String(d.step);
    inp.value = currentDials[d.key];
    inp.addEventListener("change", () => {
      let v = parseFloat(inp.value);
      if (Number.isNaN(v)) return;
      v = Math.min(d.max, Math.max(d.min, v));
      inp.value = String(v);
      saveDials({ ...currentDials, [d.key]: v });
    });
    control.append(range, inp);

    head.append(makeDialLabel(d.label, d.info), control);

    const desc = document.createElement("p");
    desc.className = "dial-desc";
    desc.textContent = d.desc;

    dial.append(head, desc);
    body.appendChild(dial);
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
    currentDials = await invoke("get_dials_for", { baseUrl: activeBaseUrl });
    renderDialsBody();
  } catch (err) {
    // Null out so the self-heal loop retries once the server is reachable again.
    currentDials = null;
    const msg = String(err?.message || err || "");
    const hint = /\b404\b/.test(msg)
      ? "this server build lacks /api/dials — rebuild the server from this branch"
      : "server unreachable — retrying…";
    const body = document.getElementById("dials-body");
    if (body) body.innerHTML = '<p class="muted">dials unavailable (' + hint + ")</p>";
  }
}

async function pull() {
  try {
    // Snapshot for the active instance/tab (falls back to pinned server-side).
    const snap = await invoke("get_snapshot_for", { baseUrl: activeBaseUrl });
    render(snap);
  } catch (err) {
    diagPush("ERROR", "get_snapshot_for failed: " + (err?.message || err));
  }
}

// ---- Multi-instance tabs ------------------------------------------------
async function pullInstances() {
  try {
    instances = await invoke("list_instances");
  } catch (err) {
    diagPush("ERROR", "list_instances failed: " + (err?.message || err));
    return;
  }
  // Default to the pinned (first) instance; recover if the active one vanished.
  if (instances.length && !instances.some((i) => i.base_url === activeBaseUrl)) {
    activeBaseUrl = instances[0].base_url;
  }
  renderTabs();
}

function renderTabs() {
  const bar = document.getElementById("tabs");
  if (!bar) return;
  // Only show the tab bar once more than one instance is discovered.
  if (instances.length <= 1) {
    bar.innerHTML = "";
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  bar.innerHTML = "";
  for (const inst of instances) {
    const tab = document.createElement("button");
    tab.className = "tab" + (inst.base_url === activeBaseUrl ? " active" : "");
    tab.title = inst.base_url + (inst.pinned ? " (configured)" : "");
    const dot = document.createElement("span");
    dot.className = "tab-dot " + (inst.online ? "on" : "off");
    const lbl = document.createElement("span");
    lbl.textContent = (inst.pinned ? "★ " : "") + (inst.label || inst.base_url);
    tab.append(dot, lbl);
    tab.addEventListener("click", () => setActiveInstance(inst.base_url));
    bar.appendChild(tab);
  }
}

function setActiveInstance(base) {
  if (base === activeBaseUrl) return;
  activeBaseUrl = base;
  currentDials = null; // force a dials reload for the newly-selected instance
  renderTabs();
  pull();
  pullDials();
}

// ---- First-run setup panel ----------------------------------------------
async function setupFirstRun() {
  let firstRun = false;
  try {
    firstRun = await invoke("first_run_check");
  } catch {
    return;
  }
  const panel = document.getElementById("first-run");
  if (!panel || !firstRun) return;
  panel.classList.remove("hidden");

  const seedBtn = document.getElementById("fr-seed-btn");
  const seedPath = document.getElementById("fr-seed-path");
  const seedStatus = document.getElementById("fr-seed-status");
  if (seedBtn && seedPath && seedStatus) {
    seedBtn.addEventListener("click", async () => {
      const p = seedPath.value.trim();
      if (!p) {
        seedStatus.textContent = "enter a project path";
        seedStatus.className = "fr-status err";
        return;
      }
      seedStatus.textContent = "seeding…";
      seedStatus.className = "fr-status";
      try {
        await invoke("seed_project", { baseUrl: activeBaseUrl, projectPath: p });
        seedStatus.textContent = "✓ seed started for " + p;
        seedStatus.className = "fr-status ok";
      } catch (err) {
        seedStatus.textContent = "seed failed: " + (err?.message || err);
        seedStatus.className = "fr-status err";
      }
    });
  }

  const hookBtn = document.getElementById("fr-hook-btn");
  const hookPath = document.getElementById("fr-hook-path");
  const hookStatus = document.getElementById("fr-hook-status");
  if (hookBtn && hookPath && hookStatus) {
    hookBtn.addEventListener("click", async () => {
      const p = hookPath.value.trim();
      if (!p) {
        hookStatus.textContent = "enter a project directory";
        hookStatus.className = "fr-status err";
        return;
      }
      try {
        const res = await invoke("claude_hook_status", { projectDir: p });
        if (res.configured) {
          hookStatus.textContent = "✓ veld hooks configured (" + res.path + ")";
          hookStatus.className = "fr-status ok";
        } else {
          hookStatus.textContent =
            "✗ not configured at " + res.path +
            " — copy hooks/claude-settings.json there and set VELD_HOOKS_DIR";
          hookStatus.className = "fr-status err";
        }
      } catch (err) {
        hookStatus.textContent = "check failed: " + (err?.message || err);
        hookStatus.className = "fr-status err";
      }
    });
  }

  // "Browse…" buttons → native folder picker (Tauri dialog) → fill the input.
  const wireBrowse = (btnId, input) => {
    const btn = document.getElementById(btnId);
    if (!btn || !input) return;
    btn.addEventListener("click", async () => {
      try {
        const dir = await invoke("pick_directory");
        if (dir) input.value = dir;
      } catch (err) {
        diagPush("ERROR", "pick_directory failed: " + (err?.message || err));
      }
    });
  };
  wireBrowse("fr-seed-browse", seedPath);
  wireBrowse("fr-hook-browse", hookPath);

  const done = document.getElementById("fr-done");
  if (done) {
    done.addEventListener("click", async () => {
      try {
        await invoke("mark_initialized");
      } catch (err) {
        diagPush("ERROR", "mark_initialized failed: " + (err?.message || err));
      }
      panel.classList.add("hidden");
    });
  }
}

async function init() {
  try {
    // The emit stream is the pinned instance; only render it when the pinned tab
    // is the active one (other tabs are driven by polling get_snapshot_for).
    await listen("snapshot-updated", (event) => {
      const snap = event.payload;
      if (snap && (!activeBaseUrl || snap.base_url === activeBaseUrl)) render(snap);
    });
    diagPush("info", "listen('snapshot-updated') subscribed");
  } catch (err) {
    diagPush(
      "ERROR",
      "listen('snapshot-updated') failed: " + (err?.message || err)
    );
    return;
  }
  await pullInstances(); // sets activeBaseUrl to the pinned instance
  await pull();

  const dialsRefresh = document.getElementById("dials-refresh");
  if (dialsRefresh) dialsRefresh.addEventListener("click", pullDials);
  await pullDials();

  // Graph viewer: load the server's /graph/view in an in-window iframe overlay
  // (not a separate OS window). URL derived from the live snapshot's base_url.
  const openGraph = document.getElementById("open-graph");
  const graphOverlay = document.getElementById("graph-overlay");
  const graphFrame = document.getElementById("graph-frame");
  const graphClose = document.getElementById("graph-close");
  const graphReload = document.getElementById("graph-reload");
  function loadGraphFrame() {
    const base = (currentBaseUrl || "").replace(/\/+$/, "");
    if (!base) {
      diagPush("ERROR", "graph: no base_url known yet — is the server up?");
      return;
    }
    const user = encodeURIComponent(currentUserId || "claude-code");
    // Cache-bust so Reload always re-requests (recovers after the server was down).
    graphFrame.src = `${base}/graph/view?user_id=${user}&_t=${Date.now()}`;
  }
  if (openGraph && graphOverlay && graphFrame) {
    openGraph.addEventListener("click", () => {
      loadGraphFrame();
      graphOverlay.classList.remove("hidden");
    });
  }
  if (graphReload && graphFrame) graphReload.addEventListener("click", loadGraphFrame);
  if (graphClose && graphOverlay && graphFrame) {
    graphClose.addEventListener("click", () => {
      graphOverlay.classList.add("hidden");
      graphFrame.src = "about:blank"; // stop the 3D render loop + SSE while hidden
    });
  }

  await setupFirstRun();

  setInterval(() => {
    pullInstances(); // refresh the tab list as instances come/go
    // Poll the active instance. For the pinned tab the emit also refreshes (1s);
    // for other tabs this is the only driver. Always poll if the emit is stale.
    const activeIsPinned = instances.find((i) => i.base_url === activeBaseUrl)?.pinned;
    if (!activeIsPinned || Date.now() - lastUpdateAt > 5000) pull();
    // Self-heal the dials card after a server flap/old-build window. Only retries
    // while unavailable (currentDials === null), so it never clobbers a live edit.
    if (!currentDials) pullDials();
  }, 3000);
}

init();
