// Data tab: backups, MIF export/import and the storage backend of the instance in
// the active tab.
//
// Backups and export/import authenticate with the monitor's API key. The backend
// swap calls the admin-gated storage routes: the admin key comes from the monitor's
// environment (VELD_ADMIN_API_KEY) or is asked for and kept in this module's memory.

import { h, when, fmtBytes, makeDialogs } from "./ui.js";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const state = { view: "backups", adminKey: null, adminFromEnv: false, adapters: null, importFile: null, restarting: false };

const base = () => (window.veldActiveBaseUrl?.() || "").replace(/\/+$/, "");
const userId = () => window.veldUserId?.() || "";
const $ = (id) => document.getElementById(id);
const BACKENDS = ["rocksdb", "redb"];

const { ask, show, confirmAction, flash } = makeDialogs({ modalId: "data-modal", flashId: "data-flash" });

class ApiError extends Error {
  constructor(status, body) {
    super(describe(status, body));
    this.status = status;
    this.code = body?.error;
    this.body = body;
  }
}

function describe(status, body) {
  if (typeof body === "string") return body;
  if (body?.detail) return body.detail;
  if (body?.message) return body.message;
  if (body?.error === "admin_key_not_configured") return "The server has no VELD_ADMIN_API_KEY set; the storage routes stay closed until it does.";
  if (body?.error === "unauthorized") return "The admin key was refused.";
  if (body?.error) return String(body.error);
  return `HTTP ${status}`;
}

async function api(method, path, { body, admin } = {}) {
  const res = await invoke("data_request", {
    baseUrl: base(),
    method,
    path,
    adminKey: admin ? state.adminKey : null,
    body: body ?? null,
  });
  if (res.status >= 400) throw new ApiError(res.status, res.body);
  return res.body;
}

const table = (heads, rows) => h("table", { class: "sec-table" },
  h("thead", {}, h("tr", {}, heads.map((t) => h("th", {}, t)))),
  h("tbody", {}, rows));

const stamp = () => new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "").replace(/(\d{8})(\d{4})/, "$1-$2");

// ── Backups ─────────────────────────────────────────────────────────────

async function renderBackups(body) {
  const user = userId();
  const res = await api("POST", "/api/backup/list", { body: { user_id: user } });
  const backups = (res.backups || []).slice().sort((a, b) => b.backup_id - a.backup_id);
  const reload = () => render();

  const verify = (b) => async () => {
    try {
      const out = await api("POST", "/api/backup/verify", { body: { user_id: user, backup_id: b.backup_id } });
      flash(`Backup ${b.backup_id}: ${out.message}`, out.is_valid ? "ok" : "error");
    } catch (err) { flash(err.message, "error"); }
  };
  const restore = (b) => async () => {
    const v = await ask(`Restore backup ${b.backup_id}`, [
      { name: "max_lsn", label: "Stop the intent log at LSN (optional)", inputmode: "numeric", optional: true },
    ], { note: `Replaces the live memories, graph and secondary stores of ${user} with this backup from ${when(b.created_at)}. The server should be restarted afterwards.`, submit: "Restore" });
    if (!v) return;
    if (!(await confirmAction("Restore over live data", `Everything ${user} stored after ${when(b.created_at)} is lost. Continue?`, "Restore"))) return;
    try {
      const maxLsn = v.max_lsn.trim() ? Number(v.max_lsn) : null;
      const out = await api("POST", "/api/backup/restore", { body: { user_id: user, backup_id: b.backup_id, max_lsn: maxLsn } });
      await show("Restored", h("p", {}, out.message), h("p", { class: "sec-note" }, "Stores: " + (out.restored_stores || []).join(", ")));
      reload();
    } catch (err) { flash(err.message, "error"); }
  };

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, `Backups of ${user}`),
    backups.length
      ? table(["Id", "Created", "Type", "Size", "Memories", "Secondary", ""], backups.map((b) => h("tr", { title: "sha256 " + b.checksum },
          h("td", {}, b.backup_id),
          h("td", {}, when(b.created_at)),
          h("td", {}, typeof b.backup_type === "string" ? b.backup_type.toLowerCase() : "full"),
          h("td", {}, fmtBytes(b.size_bytes)),
          h("td", {}, b.memory_count),
          h("td", {}, (b.secondary_stores || []).length ? `${b.secondary_stores.length} (${fmtBytes(b.secondary_size_bytes)})` : "—"),
          h("td", { class: "sec-row-actions" },
            h("button", { type: "button", class: "sec-small", onclick: verify(b) }, "Verify"),
            h("button", { type: "button", class: "sec-small sec-danger", onclick: restore(b) }, "Restore…")))))
      : h("p", {}, "No backups yet."),
    h("div", { class: "sec-actions" },
      h("button", { type: "button", onclick: async () => {
        flash("Creating backup…", "info");
        try {
          const out = await api("POST", "/api/backup/create", { body: { user_id: user } });
          if (!out.success) return flash(out.message, "error");
          flash(`Backup ${out.backup.backup_id} created: ${out.backup.memory_count} memories, ${fmtBytes(out.backup.size_bytes)}.`, "ok");
          reload();
        } catch (err) { flash(err.message, "error"); }
      } }, "Create backup"),
      h("button", { type: "button", class: "sec-secondary", disabled: !backups.length, onclick: async () => {
        const v = await ask("Purge old backups", [{ name: "keep", label: "Backups to keep (newest first)", inputmode: "numeric", value: "5" }], { submit: "Purge" });
        if (!v) return;
        const keep = Number(v.keep);
        if (!Number.isInteger(keep) || keep < 0) return flash("Enter a whole number.", "error");
        try {
          const out = await api("POST", "/api/backup/purge", { body: { user_id: user, keep_count: keep } });
          flash(`Purged ${out.purged_count} backup(s).`, "ok");
          reload();
        } catch (err) { flash(err.message, "error"); }
      } }, "Purge old…"),
      h("button", { type: "button", class: "sec-secondary", onclick: reload }, "Refresh"))));
}

// ── Export / import ─────────────────────────────────────────────────────

async function adapters() {
  if (!state.adapters) {
    const res = await api("GET", "/api/mif/adapters");
    state.adapters = res.adapters || [];
  }
  return state.adapters;
}

function formatOptions(list, { auto } = {}) {
  const seen = new Set(["veld"]);
  const opts = [{ value: "veld", label: "veld (MIF JSON)" }];
  for (const a of list) {
    if (seen.has(a.format)) continue;
    seen.add(a.format);
    opts.push({ value: a.format, label: `${a.name} (${a.format})` });
  }
  return auto ? [{ value: "", label: "auto-detect" }, ...opts] : opts;
}

function exportFileName(user, format) {
  const ext = format === "veld" || format === "json" ? "json" : format === "markdown" ? "md" : "txt";
  return `veld-export-${user}-${stamp()}.${ext}`;
}

async function renderExport(body) {
  const user = userId();
  const list = await adapters();

  const exportForm = h("form", { class: "sec-form" });
  const format = h("select", { name: "format" }, formatOptions(list).map((o) => h("option", { value: o.value }, o.label)));
  const graph = h("input", { type: "checkbox", name: "include_graph" });
  graph.checked = true;
  const embeddings = h("input", { type: "checkbox", name: "include_embeddings" });
  const redact = h("input", { type: "checkbox", name: "redact_pii" });
  const since = h("input", { type: "datetime-local", name: "since" });
  exportForm.append(
    h("label", {}, h("span", {}, "Format"), format),
    h("label", { class: "sec-check" }, graph, h("span", {}, "Include the knowledge graph")),
    h("label", { class: "sec-check" }, embeddings, h("span", {}, "Include embeddings (much larger)")),
    h("label", { class: "sec-check" }, redact, h("span", {}, "Redact PII")),
    h("label", {}, h("span", {}, "Only memories since (optional)"), since),
    h("button", { type: "submit" }, "Export to file…"));
  exportForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    flash("Exporting…", "info");
    try {
      const req = {
        user_id: user,
        format: format.value,
        include_graph: graph.checked,
        include_embeddings: embeddings.checked,
        redact_pii: redact.checked,
        since: since.value ? new Date(since.value).toISOString() : null,
      };
      const out = await api("POST", "/api/export/mif", { body: req });
      const wrapped = out && typeof out.data === "string";
      const contents = wrapped ? out.data : JSON.stringify(out, null, 2);
      const memories = wrapped ? out.memory_count : out.memories?.length ?? 0;
      const todos = wrapped ? out.todo_count : out.todos?.length ?? 0;
      const path = await invoke("save_export_file", { suggestedName: exportFileName(user, format.value), contents });
      if (path) flash(`Saved ${memories} memories and ${todos} todos to ${path}`, "ok");
      else flash("Export not saved.", "warn");
    } catch (err) { flash(err.message, "error"); }
  });
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, `Export ${user}`),
    h("p", { class: "sec-note" }, "Memories, todos, projects and reminders in the Memory Interchange Format. The file is written readable by you only."),
    exportForm));

  const importForm = h("form", { class: "sec-form" });
  const chosen = h("span", { class: "sec-mono" }, state.importFile ? state.importFile.path : "no file chosen");
  const importFormat = h("select", { name: "format" }, formatOptions(list, { auto: true }).map((o) => h("option", { value: o.value }, o.label)));
  const skip = h("input", { type: "checkbox", name: "skip_duplicates" });
  skip.checked = true;
  const importBtn = h("button", { type: "submit", disabled: !state.importFile }, "Import into " + user);
  importForm.append(
    h("div", { class: "sec-actions" },
      h("button", { type: "button", class: "sec-secondary", onclick: async () => {
        try {
          const picked = await invoke("pick_import_file");
          if (!picked) return;
          state.importFile = picked;
          chosen.textContent = picked.path;
          importBtn.disabled = false;
        } catch (err) { flash(err.message || String(err), "error"); }
      } }, "Choose file…"),
      chosen),
    h("label", {}, h("span", {}, "Format"), importFormat),
    h("label", { class: "sec-check" }, skip, h("span", {}, "Skip memories whose content is already stored")),
    importBtn);
  importForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.importFile) return;
    if (!(await confirmAction("Import into live data", `Records from ${state.importFile.path} are written into ${user} on ${base()}. Imported memories keep their original ids and timestamps.`, "Import"))) return;
    let data = state.importFile.contents;
    try { data = JSON.parse(data); } catch { /* a non-JSON adapter format travels as text */ }
    flash("Importing…", "info");
    try {
      const out = await api("POST", "/api/import/mif", {
        body: { user_id: user, skip_duplicates: skip.checked, format: importFormat.value || null, data },
      });
      const r = out.result;
      await show(out.success ? "Imported" : "Imported with errors",
        table(["", "Count"], [
          ["Memories", r.memories_imported], ["Todos", r.todos_imported], ["Projects", r.projects_imported],
          ["Reminders", r.reminders_imported], ["Entities", r.entities_imported], ["Edges", r.edges_imported],
          ["Duplicates skipped", r.duplicates_skipped],
        ].map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v)))),
        r.errors?.length ? h("pre", { class: "sec-code" }, r.errors.slice(0, 50).join("\n")) : null);
      state.importFile = null;
      render();
    } catch (err) { flash(err.message, "error"); }
  });
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, `Import into ${user}`),
    importForm));
}

// ── Backend ─────────────────────────────────────────────────────────────

const unitLabel = (u) => (typeof u === "string" ? "shared stores" : `user ${u.user}`);
const other = (backend) => BACKENDS.find((b) => b !== backend) || BACKENDS[0];

function restartSummary(out) {
  if (out.supervised) return `${out.base_url} stopped and came back on its own: whatever started it restarted it.`;
  let text = `${out.base_url} is back (pid ${out.pid}, ${out.recipe_source} recipe); output in ${out.log_path}.`;
  if (out.missing_env?.length) text += ` Not carried over: ${out.missing_env.join(", ")}.`;
  return text;
}

// Stop the active instance, if it runs, and bring it back; `storageBackend` is pinned
// on the new process when a swap was just made. Resolves to true when the server
// answered again.
async function restartServer({ storageBackend = null, reason, running = true } = {}) {
  if (state.restarting) return flash("A restart is already running.", "warn"), false;
  const text = (reason ? reason + " " : "") + (running
    ? `${base()} is asked to stop, and if nothing else brings it back it is started again with the environment it had. Clients see it offline for a few seconds.`
    : `${base()} is started from the saved launch recipe.`);
  if (!(await confirmAction(running ? "Restart the server" : "Start the server", text, running ? "Restart" : "Start"))) return false;
  state.restarting = true;
  flash("Restarting… waiting for /health.", "info");
  try {
    const out = await invoke("restart_server", { baseUrl: base(), storageBackend });
    flash(restartSummary(out), out.missing_env?.length ? "warn" : "ok");
    return true;
  } catch (err) {
    flash(err.message || String(err), "error");
    return false;
  } finally {
    state.restarting = false;
  }
}

async function ensureAdminKey() {
  if (state.adminFromEnv || state.adminKey) return true;
  const v = await ask("Admin key", [{ name: "key", label: "VELD_ADMIN_API_KEY of the server", type: "password" }],
    { note: "The storage routes are gated by the server's admin key. It is kept in this window's memory only." });
  if (!v || !v.key.trim()) return false;
  state.adminKey = v.key.trim();
  return true;
}

function reportView(report) {
  const units = report.units || [];
  const imported = report.imported || [];
  const quarantined = report.quarantined || [];
  const count = (x) => (x ? x.records : "—");
  return h("div", {},
    h("p", {}, `${report.from} → ${report.to}` + (report.dry_run ? " (dry run, nothing written)" : "")),
    table(["Unit", `On ${report.from}`, `On ${report.to}`, "Blocked"], units.map((u) => h("tr", {},
      h("td", {}, unitLabel(u.unit)),
      h("td", {}, count(u.source) + (u.source?.decode_failures ? ` (${u.source.decode_failures} undecodable)` : "")),
      h("td", {}, count(u.target)),
      h("td", { class: u.blocked?.length ? "sec-bad" : "" }, (u.blocked || []).join("; ") || "")))),
    imported.length ? h("div", {},
      h("h3", {}, "Written and verified"),
      table(["Unit", "Records", "Frames", "Digest"], imported.map((i) => h("tr", {},
        h("td", {}, unitLabel(i.unit)), h("td", {}, i.records), h("td", {}, `${i.frames} = ${i.verified_frames}`),
        h("td", { class: "sec-mono" }, i.digest === i.verified_digest ? "match" : "MISMATCH"))))) : null,
    quarantined.length ? h("p", { class: "sec-warn" },
      "Quarantined, not migrated: " + quarantined.map((q) => `${unitLabel(q.unit)} ${q.records} record(s) in ${q.dump}`).join("; ")) : null);
}

async function runMigrate(from, to, scope, dryRun, quarantine) {
  try {
    const body = { from, to, user: scope, dry_run: dryRun };
    if (quarantine) body.quarantine = `swap-${stamp()}`;
    const out = await api("POST", "/api/admin/storage/migrate", { body, admin: true });
    return out.report;
  } catch (err) {
    if (err.code === "blocked" && err.body?.report) return err.body.report;
    throw err;
  }
}

// ── Launch recipe ───────────────────────────────────────────────────────

const specSummary = (s) => [s.exe, ...(s.args || [])].join(" ");
const envLines = (env) => Object.entries(env || {}).map(([k, v]) => `${k}=${v}`).join("\n");

function parseEnvLines(text) {
  const env = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) throw new Error(`not KEY=VALUE: ${line}`);
    const key = line.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) throw new Error(`not a variable name: ${key}`);
    env[key] = line.slice(eq + 1).trim();
  }
  return env;
}

// Edit a recipe in a dialog; resolves to the spec or null.
async function editRecipe(initial, note) {
  const v = await ask("Launch recipe", [
    { name: "exe", label: "Executable", value: initial.exe || "", placeholder: "/usr/local/bin/veld" },
    { name: "args", label: "Arguments (space-separated)", value: (initial.args || []).join(" "), optional: true, placeholder: "server --port 3030" },
    { name: "cwd", label: "Working directory (optional)", value: initial.cwd || "", optional: true },
    { name: "env", label: "Environment, one KEY=VALUE per line (optional)", value: envLines(initial.env), multiline: true, rows: 7 },
  ], { note, submit: "Save" });
  if (!v) return null;
  let env;
  try { env = parseEnvLines(v.env); } catch (err) { flash(err.message, "error"); return null; }
  return { exe: v.exe.trim(), args: v.args.trim() ? v.args.trim().split(/\s+/) : [], cwd: v.cwd.trim() || null, env };
}

async function renderLaunchCard(body) {
  const info = await invoke("launch_recipe", { baseUrl: base() });
  const reload = () => render();
  const save = async (spec) => {
    try {
      await invoke("set_launch_recipe", { baseUrl: base(), spec });
      flash(spec ? "Recipe saved." : "Saved recipe forgotten.", "ok");
      reload();
    } catch (err) { flash(err.message || String(err), "error"); }
  };
  const uses = info.saved ? "the saved recipe" : info.registry ? "the registry's recipe" : "nothing: no recipe is known";
  const rows = [];
  if (info.registry) {
    rows.push(h("dt", {}, "Registry"), h("dd", {}, specSummary(info.registry),
      info.registry.cwd ? h("span", { class: "sec-mono" }, ` in ${info.registry.cwd}`) : null,
      h("div", { class: "sec-note" }, "Variables: " + (info.registry.env_names?.length ? info.registry.env_names.join(", ") : "none"))));
  }
  if (info.saved) {
    rows.push(h("dt", {}, "Saved"), h("dd", {}, specSummary(info.saved),
      info.saved.cwd ? h("span", { class: "sec-mono" }, ` in ${info.saved.cwd}`) : null,
      h("div", { class: "sec-note" }, "Variables: " + (Object.keys(info.saved.env || {}).join(", ") || "none"))));
  }
  rows.push(h("dt", {}, "server.env"), h("dd", {}, info.server_env_names.length
    ? h("span", {}, info.server_env_names.join(", "), h("span", { class: "sec-mono" }, ` from ${info.server_env_path}`))
    : h("span", { class: "sec-note" }, "none at " + info.server_env_path)));

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Launch"),
    h("p", { class: "sec-note" }, `A restart stops the server, waits for whatever started it, and otherwise starts it again using ${uses}. The operator's server.env is the base environment, the live process's variables and the recipe's come on top.`),
    h("dl", { class: "sec-kv" }, rows),
    h("div", { class: "sec-actions" },
      h("button", { type: "button", disabled: !info.running, title: info.running ? "" : "needs a running server", onclick: async () => {
        try {
          const spec = await invoke("capture_launch_recipe", { baseUrl: base() });
          const edited = await editRecipe(spec, "Read from the running process: command line, directory and VELD_* variables (values included). Saved readable by you only.");
          if (edited) await save(edited);
        } catch (err) { flash(err.message || String(err), "error"); }
      } }, "Capture from running server…"),
      h("button", { type: "button", class: "sec-secondary", onclick: async () => {
        const edited = await editRecipe(info.saved || info.registry || {}, "How to start this server. Variables here win over server.env and the live process's.");
        if (edited) await save(edited);
      } }, info.saved ? "Edit saved…" : "Write a recipe…"),
      info.saved ? h("button", { type: "button", class: "sec-secondary", onclick: async () => {
        if (await confirmAction("Forget the saved recipe", "Restarts fall back to the registry's recipe.", "Forget")) await save(null);
      } }, "Forget saved") : null,
      !info.running ? h("button", { type: "button", disabled: !info.saved, title: info.saved ? "" : "save a recipe first", onclick: async () => {
        if (await restartServer({ running: false })) reload();
      } }, "Start server…") : null)));
}

async function renderBackend(body) {
  let health;
  try {
    health = await api("GET", "/health");
  } catch (err) {
    body.append(h("p", { class: "sec-flash warn" }, `${base()} is not answering: ${err.message}`));
    await renderLaunchCard(body);
    return;
  }
  const cfg = await invoke("config_backend");
  const serving = health.effective_storage_backend;
  const requested = health.requested_storage_backend;
  const census = health.tenants_with_data || {};
  const user = userId();

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Storage backend"),
    h("p", {}, `Serving ${serving}` + (requested !== serving ? ` (requested ${requested}: this build cannot serve it)` : "") + ` · veld ${health.version || "?"}`),
    table(["Backend", "Tenants with data"], BACKENDS.map((b) => h("tr", {},
      h("td", {}, b + (b === serving ? " (serving)" : "")),
      h("td", {}, b === "redb" && census.redb == null ? "unknown to this build" : census[b] ?? "—")))),
    h("p", { class: "sec-note" }, census.measured_at ? "Counted at server start, " + when(census.measured_at) + "." : ""),
    h("p", {}, "Next start: ", h("strong", {}, cfg.server_env_backend || cfg.backend || "build default"), " ",
      cfg.server_env_backend
        ? h("span", { class: "sec-mono" }, `VELD_STORAGE_BACKEND in ${cfg.server_env_path}` + (cfg.backend && cfg.backend !== cfg.server_env_backend ? ` (overrides ${cfg.backend} in config.toml)` : ""))
        : h("span", { class: "sec-mono" }, cfg.path)),
    h("div", { class: "sec-actions" },
      h("button", { type: "button", class: "sec-secondary", onclick: async () => {
        if (await restartServer()) render();
      } }, "Restart server…"))));

  await renderLaunchCard(body);

  const target = h("select", {}, BACKENDS.map((b) => h("option", { value: b }, b)));
  target.value = other(serving);
  const scope = h("select", {}, [h("option", { value: "all" }, "every user + shared stores"), h("option", { value: user }, "only " + user)]);
  const quarantine = h("input", { type: "checkbox" });
  const result = h("div", {});
  const refresh = () => render();

  const guard = () => {
    if (target.value === serving) { flash(`The server is already serving ${target.value}.`, "warn"); return false; }
    return true;
  };

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Swap backend"),
    h("p", { class: "sec-note" }, "Copies every store from the serving backend to the other one in the same data directory, verifying each copy against its digest. The source is never modified. Then the next start is pointed at the copy; restart veld to serve it."),
    h("div", { class: "sec-form" },
      h("label", {}, h("span", {}, "Copy to"), target),
      h("label", {}, h("span", {}, "Scope"), scope),
      h("label", { class: "sec-check" }, quarantine, h("span", {}, "Set undecodable records aside instead of refusing (they are not migrated)"))),
    h("div", { class: "sec-actions" },
      h("button", { type: "button", class: "sec-secondary", onclick: async () => {
        if (!guard() || !(await ensureAdminKey())) return;
        flash("Counting…", "info");
        try {
          const report = await runMigrate(serving, target.value, scope.value, true, quarantine.checked);
          result.replaceChildren(reportView(report));
          flash(report.units?.some((u) => u.blocked?.length) ? "A real migration would refuse: see Blocked." : "Dry run complete.", "ok");
        } catch (err) { if (err.status === 401) state.adminKey = null; flash(err.message, "error"); }
      } }, "1 · Dry run"),
      h("button", { type: "button", onclick: async () => {
        if (!guard() || !(await ensureAdminKey())) return;
        if (!(await confirmAction(`Copy ${scope.value === "all" ? "everything" : user} to ${target.value}`,
          `Stores on ${target.value} must be empty; the copy on ${serving} stays as it is. This walks every record and can take a while.`, "Migrate"))) return;
        flash("Migrating…", "info");
        try {
          const report = await runMigrate(serving, target.value, scope.value, false, quarantine.checked);
          result.replaceChildren(reportView(report));
          const blocked = report.units?.some((u) => u.blocked?.length);
          flash(blocked ? "Refused: see Blocked." : `Copied to ${target.value} and verified. Point the next start at it below.`, blocked ? "error" : "ok");
        } catch (err) { if (err.status === 401) state.adminKey = null; flash(err.message, "error"); }
      } }, "2 · Migrate"),
      h("button", { type: "button", onclick: async () => {
        const alsoEnv = cfg.server_env_backend ? ` and VELD_STORAGE_BACKEND=${target.value} to ${cfg.server_env_path}, which Veld Desktop hands to the server` : "";
        if (!(await confirmAction(`Serve ${target.value} on the next start`,
          `Writes storage_backend = "${target.value}" to ${cfg.path}${alsoEnv}. A VELD_STORAGE_BACKEND set elsewhere in the server's environment still wins. The boot guard refuses an empty ${target.value} while ${serving} holds data.`, "Write"))) return;
        let written;
        try {
          written = await invoke("set_config_backend", { backend: target.value });
        } catch (err) { return flash(err.message || String(err), "error"); }
        flash(`Next start serves ${written.backend}` + (written.server_env_backend ? " (config.toml and server.env)." : "."), "ok");
        await restartServer({ storageBackend: written.backend, reason: `Serve ${written.backend} now?` });
        refresh();
      } }, "3 · Set next start"),
      !state.adminFromEnv ? h("button", { type: "button", class: "sec-link", onclick: async () => {
        state.adminKey = null;
        await ensureAdminKey();
      } }, state.adminKey ? "Change admin key…" : "Enter admin key…") : null),
    result));
}

// ── Replication ─────────────────────────────────────────────────────────

async function renderReplication(body) {
  if (!(await ensureAdminKey())) {
    body.append(h("p", { class: "sec-flash warn" }, "The replication routes need this server's admin key."));
    return;
  }
  let st;
  try {
    st = await api("GET", "/api/admin/replication/status", { admin: true });
  } catch (err) {
    if (err.status === 401) state.adminKey = null;
    body.append(h("p", { class: "sec-flash error" }, err.message));
    return;
  }
  const reload = () => render();
  const f = st.following;

  const instances = await invoke("list_instances").catch(() => []);
  const others = instances.filter((i) => i.base_url.replace(/\/+$/, "") !== base());

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Following"),
    f ? h("div", {},
      h("p", {}, h("strong", {}, f.target.source), ` since ${when(f.target.since)} · ${f.pages} pages · ${f.polls} polls finished · last ${when(f.last_poll)}`),
      f.last_error ? h("p", { class: "sec-flash error" }, "Leader unreachable: " + f.last_error) : null,
      table(["Tenant", "Applied LSN", "Behind", "Frames consumed", "Journaled here", "State"], Object.entries(f.users).map(([u, p]) => {
        const behind = p.leader_head_lsn == null ? "—" : Math.max(0, p.leader_head_lsn - 1 - (p.applied_lsn ?? -1));
        const state = p.diverged ? h("span", { class: "sec-bad", title: p.diverged }, "diverged")
          : p.last_error ? h("span", { class: "sec-warn", title: p.last_error }, `retrying (${p.failures})`)
          : behind === 0 ? "in sync" : "syncing";
        return h("tr", {},
          h("td", {}, u), h("td", {}, p.applied_lsn ?? "—"), h("td", {}, behind),
          h("td", {}, p.applied_records), h("td", {}, p.journaled_records ?? 0), h("td", {}, state));
      })),
      Object.values(f.users).some((p) => p.diverged)
        ? h("p", { class: "sec-note" }, "A diverged tenant's leader log moved backwards (restore, rollback or recreated log). It is no longer synced. To resync it, stop this follower, remove the tenant's directory here, and start it again; it replays from the start.")
        : null,
      h("div", { class: "sec-actions" },
        h("button", { type: "button", class: "sec-danger", onclick: async () => {
          if (!(await confirmAction("Stop following", `${base()} keeps what it has and stops tailing ${f.target.source}. The cursor is kept, so following again resumes.`, "Stop"))) return;
          try { await api("POST", "/api/admin/replication/unfollow", { admin: true, body: {} }); flash("Stopped following.", "ok"); reload(); }
          catch (err) { flash(err.message, "error"); }
        } }, "Stop following")))
    : h("div", {},
      h("p", { class: "sec-note" }, "Not following anything. A follower tails a leader's intent logs and applies every memory frame, ids and timestamps intact; it starts empty and converges. One direction, memories only: todos, reminders and projects are not journaled."),
      h("div", { class: "sec-actions" },
        h("button", { type: "button", onclick: async () => {
          const options = others.map((i) => ({ value: i.base_url, label: `${i.label || i.base_url} (${i.base_url})` }));
          const v = await ask("Follow an instance", [
            options.length ? { name: "pick", label: "Leader", options: [...options, { value: "", label: "other URL…" }] } : null,
            { name: "source", label: "Leader URL (if not listed)", placeholder: "http://127.0.0.1:3031", optional: true },
            { name: "admin_key", label: "Leader's admin key (VELD_ADMIN_API_KEY)", type: "password" },
            { name: "api_key", label: "Leader's API key (blank: same as this instance)", type: "password", optional: true },
          ].filter(Boolean), { note: `${base()} becomes the follower. Writes made here are not sent back.`, submit: "Follow" });
          if (!v) return;
          const source = (v.pick || v.source || "").trim();
          if (!source) return flash("Pick a leader or enter its URL.", "error");
          try {
            await api("POST", "/api/admin/replication/follow", { admin: true, body: { source, admin_key: v.admin_key, api_key: v.api_key || null } });
            flash(`Following ${source}.`, "ok");
            reload();
          } catch (err) { flash(err.message, "error"); }
        } }, "Follow an instance…")))));

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "This instance as a leader"),
    h("p", { class: "sec-note" }, `${st.base_url}: another instance follows it by its URL and this server's admin key.`),
    st.users.length
      ? table(["User", "Intent log"], st.users.map((u) => h("tr", {}, h("td", {}, u.user_id), h("td", {}, fmtBytes(u.log_bytes)))))
      : h("p", {}, "No users yet."),
    h("div", { class: "sec-actions" }, h("button", { type: "button", class: "sec-secondary", onclick: reload }, "Refresh"))));
}

// ── Shell ───────────────────────────────────────────────────────────────

const VIEWS = { backups: ["Backups", renderBackups], export: ["Export / Import", renderExport], backend: ["Backend", renderBackend], replication: ["Replication", renderReplication] };

async function render() {
  const nav = $("data-nav");
  const body = $("data-body");
  nav.replaceChildren();
  body.replaceChildren();
  $("data-where").textContent = base() + (userId() ? " · " + userId() : "");
  for (const [key, [label]] of Object.entries(VIEWS)) {
    nav.append(h("button", { type: "button", class: "sec-tab" + (state.view === key ? " active" : ""),
      onclick: () => { state.view = key; render(); } }, label));
  }
  if (!userId()) {
    body.append(h("p", { class: "sec-flash warn" }, "Waiting for the first snapshot of this instance…"));
    return;
  }
  try {
    await VIEWS[state.view][1](body);
  } catch (err) {
    body.append(h("p", { class: "sec-flash error" }, err.message || String(err)));
  }
}

export function setupData() {
  const overlay = $("data-overlay");
  invoke("admin_key_configured").then((v) => { state.adminFromEnv = !!v; }).catch(() => {});
  $("open-data")?.addEventListener("click", () => {
    overlay.classList.remove("hidden");
    render();
  });
  $("data-close")?.addEventListener("click", () => {
    overlay.classList.add("hidden");
    $("data-modal").classList.add("hidden");
  });
  window.addEventListener("veld-active-instance", () => {
    state.adapters = null;
    state.importFile = null;
    if (!overlay.classList.contains("hidden")) render();
  });
  // Restarts started from the tray report here when the panel is open.
  listen("server-restarted", (event) => {
    if (overlay.classList.contains("hidden")) return;
    const p = event.payload || {};
    if (p.ok) flash(restartSummary(p.ok), p.ok.missing_env?.length ? "warn" : "ok");
    else flash("Restart failed: " + p.error, "error");
  }).catch(() => {});
}
