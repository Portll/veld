// Security tab: sign in to veld's user-auth surface, then manage the keyring and
// the logins of the instance in the active tab.
//
// The session token lives in this module's memory only. Every change asks for a
// fresh 2FA code (step-up); every key change also asks for something that opens the
// keyring: the passphrase, a recovery code, or Touch ID on this Mac.

const { invoke } = window.__TAURI__.core;

const state = { session: null, me: null, view: "keys", enclave: null };

const base = () => (window.veldActiveBaseUrl?.() || "").replace(/\/+$/, "");
const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

class ApiError extends Error {
  constructor(status, body) {
    super(body?.message || (typeof body === "string" ? body : `HTTP ${status}`));
    this.status = status;
    this.code = body?.code;
  }
}

async function api(method, path, { body, stepUp } = {}) {
  const res = await invoke("security_request", {
    baseUrl: base(),
    method,
    path,
    session: state.session,
    stepUp: stepUp || null,
    body: body ?? null,
  });
  if (res.status >= 400) {
    const err = new ApiError(res.status, res.body);
    if (err.code === "INVALID_SESSION" && state.session) signOut("Your session ended. Sign in again.");
    throw err;
  }
  return res.body;
}

// ── Modals ──────────────────────────────────────────────────────────────

function modal(title, body, actions) {
  const root = $("sec-modal");
  root.replaceChildren(
    h("div", { class: "sec-modal-card", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("h3", {}, title), body, h("div", { class: "sec-modal-actions" }, actions)),
  );
  root.classList.remove("hidden");
  const first = root.querySelector("input, select, button");
  first?.focus();
  return () => root.classList.add("hidden");
}

// Ask for some values. Resolves to an object keyed by field name, or null if cancelled.
function ask(title, fields, { submit = "Continue", note } = {}) {
  return new Promise((resolve) => {
    const inputs = {};
    const form = h("form", { class: "sec-form" },
      note ? h("p", { class: "sec-note" }, note) : null,
      fields.map((f) => {
        const input = f.options
          ? h("select", { name: f.name }, f.options.map((o) => h("option", { value: o.value }, o.label)))
          : h("input", {
              name: f.name, type: f.type || "text", placeholder: f.placeholder || "",
              autocomplete: f.autocomplete || "off", inputmode: f.inputmode, required: f.optional ? false : true,
            });
        if (f.value !== undefined) input.value = f.value;
        inputs[f.name] = input;
        return h("label", {}, h("span", {}, f.label), input);
      }));
    let close;
    const done = (value) => { close(); resolve(value); };
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      done(Object.fromEntries(Object.entries(inputs).map(([k, el]) => [k, el.value])));
    });
    close = modal(title, form, [
      h("button", { type: "button", class: "sec-secondary", onclick: () => done(null) }, "Cancel"),
      h("button", { type: "button", onclick: () => form.requestSubmit() }, submit),
    ]);
  });
}

function show(title, ...content) {
  return new Promise((resolve) => {
    let close;
    close = modal(title, h("div", {}, content), [
      h("button", { type: "button", onclick: () => { close(); resolve(); } }, "Done"),
    ]);
  });
}

function confirmAction(title, text, verb) {
  return new Promise((resolve) => {
    let close;
    close = modal(title, h("p", {}, text), [
      h("button", { type: "button", class: "sec-secondary", onclick: () => { close(); resolve(false); } }, "Cancel"),
      h("button", { type: "button", class: "sec-danger", onclick: () => { close(); resolve(true); } }, verb),
    ]);
  });
}

function flash(text, kind = "info") {
  const el = $("sec-flash");
  el.textContent = text;
  el.className = "sec-flash " + kind;
  clearTimeout(flash.timer);
  flash.timer = setTimeout(() => (el.className = "sec-flash hidden"), 6000);
}

// Run a change behind step-up. A session stepped up in the last few minutes passes
// without a code; otherwise ask, and ask again when the code is refused.
async function stepUp(title, run) {
  try {
    return await run(null);
  } catch (err) {
    if (err.code !== "STEP_UP_REQUIRED") throw err;
  }
  for (let attempt = 0; attempt < 3; attempt++) {
    const v = await ask(title, [
      { name: "code", label: "2FA code", placeholder: "123456", inputmode: "numeric", autocomplete: "one-time-code" },
    ], { submit: "Confirm", note: attempt ? "That code was refused or already used. Wait for the next one." : null });
    if (!v) return null;
    try {
      return await run(v.code.trim());
    } catch (err) {
      if (err.code === "STEP_UP_REQUIRED" || err.code === "STEP_UP_REPLAYED") continue;
      throw err;
    }
  }
  return null;
}

// Something that opens the keyring, as the API body's `unlock`.
async function unlockFactor(status, purpose) {
  const enclaveFactors = (status?.keyring?.factors || []).filter((f) => f.provider === "secure-enclave");
  const options = [
    { value: "passphrase", label: "Passphrase" },
    { value: "recovery_code", label: "Recovery code" },
  ];
  if (enclaveFactors.length && state.enclave?.touch_id) {
    for (const f of enclaveFactors) options.push({ value: "se:" + f.id, label: "Touch ID — " + (f.label || f.id) });
  }
  const pick = await ask("Open the keyring", [
    { name: "with", label: "With", options },
  ], { note: purpose });
  if (!pick) return null;
  if (pick.with.startsWith("se:")) {
    const id = pick.with.slice(3);
    const released = await invoke("enclave", { action: "unlock", id });
    return { secure_enclave: { id, secret: released.secret } };
  }
  const secret = await ask(pick.with === "passphrase" ? "Passphrase" : "Recovery code", [
    { name: "value", label: pick.with === "passphrase" ? "Passphrase" : "Recovery code", type: "password" },
  ]);
  return secret ? { [pick.with]: secret.value } : null;
}

async function showKit(issued, title = "Recovery kit") {
  const save = async () => {
    const path = await invoke("save_secret_file", {
      suggestedName: `veld-recovery-kit-${issued.id}.txt`,
      contents: issued.kit,
    });
    if (path) flash("Saved to " + path, "ok");
  };
  await show(title,
    h("p", {}, "This code opens the keyring if the passphrase is lost. It is shown once and never stored by veld."),
    h("pre", { class: "sec-code" }, issued.code.match(/.{1,6}/g).join("-")),
    h("p", { class: "sec-note" }, "Keep it offline and apart from backups of the data directory. Id: " + issued.id),
    h("button", { type: "button", onclick: save }, "Save kit…"));
}

// ── Sign in ─────────────────────────────────────────────────────────────

function signOut(message) {
  const token = state.session;
  state.session = null;
  state.me = null;
  if (token) invoke("security_request", { baseUrl: base(), method: "POST", path: "/api/user_auth/logout", session: token, stepUp: null, body: null }).catch(() => {});
  render();
  if (message) flash(message, "warn");
}

async function refreshMe() {
  state.me = await api("GET", "/api/user_auth/me");
}

function renderSignIn(body) {
  const form = h("form", { class: "sec-form sec-signin" },
    h("h3", {}, "Sign in to " + (base() || "veld")),
    h("label", {}, h("span", {}, "Username"), h("input", { name: "username", required: true, autocomplete: "username" })),
    h("label", {}, h("span", {}, "Password"), h("input", { name: "password", type: "password", required: true, autocomplete: "current-password" })),
    h("label", {}, h("span", {}, "2FA code"), h("input", { name: "totp", inputmode: "numeric", autocomplete: "one-time-code", placeholder: "if enrolled" })),
    h("button", { type: "submit" }, "Sign in"),
    h("p", { class: "sec-note" }, "Key and login management needs an admin account with 2FA. The first account registered on a server becomes its admin."),
    h("button", { type: "button", class: "sec-link", onclick: bootstrap }, "Create the first admin…"));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(form);
    try {
      const res = await api("POST", "/api/user_auth/login", {
        body: { username: f.get("username"), password: f.get("password"), totp: f.get("totp") || null },
      });
      state.session = res.session_token;
      await refreshMe();
      state.view = state.me.user.role === "admin" ? "keys" : "account";
      render();
    } catch (err) {
      flash(err.code === "USER_AUTH_DISABLED"
        ? "User auth is off on this server: set VELD_USER_AUTH_ENABLED=true and restart it."
        : err.message, "error");
    }
  });
  body.append(form);
}

async function bootstrap() {
  const v = await ask("Create the first admin", [
    { name: "username", label: "Username", autocomplete: "username" },
    { name: "password", label: "Password", type: "password", autocomplete: "new-password" },
  ], { note: "Works only while the server has no accounts. Enrol 2FA right after signing in." });
  if (!v) return;
  try {
    await api("POST", "/api/user_auth/register", { body: v });
    flash("Admin created. Sign in.", "ok");
  } catch (err) {
    flash(err.message, "error");
  }
}

// ── Keys ────────────────────────────────────────────────────────────────

async function keyChange(title, purpose, call, status) {
  try {
    const unlock = await unlockFactor(status, purpose);
    if (!unlock) return null;
    return await stepUp(title, (code) => call(unlock, code));
  } catch (err) {
    flash(err.message || String(err), "error");
    return null;
  }
}

async function renderKeys(body) {
  const res = await api("GET", "/api/user_auth/admin/keys/status");
  const status = res.status;
  if (state.enclave === null) {
    state.enclave = await invoke("enclave", { action: "probe" }).catch(() => ({ secure_enclave: false, touch_id: false }));
  }
  const reload = () => render();

  if (!status.keyring) {
    body.append(h("section", { class: "sec-card" },
      h("h3", {}, "Keyring"),
      h("p", {}, "No keyring yet. A keyring holds the recovery codes, Touch ID and passkeys that can open every keystore on this server, including tenants created later."),
      h("button", { type: "button", onclick: async () => {
        const v = await ask("Set up the keyring", [
          { name: "passphrase", label: "Server passphrase", type: "password" },
          { name: "label", label: "Label for the first recovery code", value: "initial", optional: true },
        ], { note: "The passphrase the server unseals with (VELD_MASTER_PASSPHRASE)." });
        if (!v) return;
        try {
          const out = await stepUp("Confirm keyring setup", (code) =>
            api("POST", "/api/user_auth/admin/keys/init", { body: { passphrase: v.passphrase, label: v.label || null }, stepUp: code }));
          if (out) {
            await showKit(out.recovery, "Keyring created — first recovery kit");
            reload();
          }
        } catch (err) { flash(err.message, "error"); }
      } }, "Set up keyring…")));
  } else {
    if (!res.keyring_unsealed) {
      body.append(h("p", { class: "sec-flash warn" }, "The keyring exists but this server started before it: restart veld so tenants open through it."));
    }
    const factors = status.keyring.factors || [];
    const revoke = (f) => async () => {
      if (!(await confirmAction("Revoke " + (f.label || f.id), "It will no longer open the keyring. Copies of the keyring made before now still open with it.", "Revoke"))) return;
      const out = await keyChange("Confirm revocation", "Revoke " + f.id, (unlock, code) =>
        api("POST", "/api/user_auth/admin/keys/revoke", { body: { unlock, id: f.id }, stepUp: code }), status);
      if (out) { flash("Revoked " + f.id, "ok"); reload(); }
    };
    body.append(h("section", { class: "sec-card" },
      h("h3", {}, "Keyring factors"),
      h("table", { class: "sec-table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Kind"), h("th", {}, "Label"), h("th", {}, "Added"), h("th", {}, ""))),
        h("tbody", {}, factors.map((f) => h("tr", {},
          h("td", {}, f.provider),
          h("td", { title: f.id }, f.label || f.id),
          h("td", {}, when(f.created_at)),
          h("td", {}, f.provider === "passphrase" ? "" : h("button", { type: "button", class: "sec-small sec-danger", onclick: revoke(f) }, "Revoke")))))),
      h("div", { class: "sec-actions" },
        h("button", { type: "button", onclick: async () => {
          const label = await ask("New recovery code", [{ name: "label", label: "Label", placeholder: "e.g. safe at home", optional: true }]);
          if (!label) return;
          const out = await keyChange("Confirm new recovery code", "Issue a recovery code", (unlock, code) =>
            api("POST", "/api/user_auth/admin/keys/recovery_codes", { body: { unlock, label: label.label || null }, stepUp: code }), status);
          if (out) { await showKit(out); reload(); }
        } }, "New recovery kit…"),
        h("button", { type: "button", disabled: !state.enclave?.touch_id, title: state.enclave?.touch_id ? "" : "Touch ID or the veld-enclave helper is not available on this Mac", onclick: async () => {
          const label = await ask("Add Touch ID", [{ name: "label", label: "Label", value: "Touch ID", optional: true }],
            { note: "Creates a key in this Mac's Secure Enclave that only your current fingerprints can use. Adding or removing a fingerprint later invalidates it." });
          if (!label) return;
          const out = await keyChange("Confirm Touch ID", "Enrol Touch ID", async (unlock, code) => {
            const released = await invoke("enclave", { action: "enrol", label: label.label || "Touch ID" });
            return api("POST", "/api/user_auth/admin/keys/factors", {
              body: { unlock, provider: "secure-enclave", id: released.id, label: released.label, secret: released.secret }, stepUp: code });
          }, status);
          if (out) { flash("Touch ID enrolled", "ok"); reload(); }
        } }, "Add Touch ID…"),
        h("button", { type: "button", onclick: async () => {
          const label = await ask("Add a passkey", [{ name: "label", label: "Label", placeholder: "e.g. YubiKey 5C", optional: true }],
            { note: "Opens your browser to create a passkey with the PRF extension (iCloud Keychain on macOS 15+, most security keys)." });
          if (!label) return;
          const out = await keyChange("Confirm passkey", "Enrol a passkey", (unlock, code) =>
            api("POST", "/api/user_auth/admin/keys/passkey", { body: { unlock, label: label.label || null }, stepUp: code }), status);
          if (!out) return;
          const pageUrl = base().replace("://127.0.0.1", "://localhost").replace("://[::1]", "://localhost") + out.path;
          await invoke("open_in_browser", { url: pageUrl });
          flash("Finish in the browser…", "info");
          for (let i = 0; i < 150; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            const poll = await api("GET", "/api/user_auth/admin/keys/passkey/" + out.ticket).catch(() => null);
            if (!poll || poll.state === "waiting") continue;
            if (poll.state === "done") flash("Passkey enrolled", "ok");
            else flash("Passkey not enrolled: " + poll.error, "error");
            return reload();
          }
          flash("The passkey step timed out.", "error");
        } }, "Add passkey…")),
    ));

    body.append(h("section", { class: "sec-card" },
      h("h3", {}, "Rotation"),
      h("div", { class: "sec-actions" },
        h("button", { type: "button", onclick: async () => {
          const v = await ask("Rotate the passphrase", [
            { name: "old", label: "Current passphrase", type: "password" },
            { name: "new", label: "New passphrase", type: "password" },
            { name: "again", label: "New passphrase again", type: "password" },
          ], { note: "Re-wraps the keyring and every keystore. Set the new passphrase where veld reads it before the next restart." });
          if (!v) return;
          if (v.new !== v.again) return flash("The new passphrases differ.", "error");
          try {
            const out = await stepUp("Confirm passphrase rotation", (code) =>
              api("POST", "/api/user_auth/admin/keys/rotate_passphrase", { body: { old: v.old, new: v.new }, stepUp: code }));
            if (out) flash(`Passphrase rotated on the keyring and ${out.keystores.length} keystore(s).`, "ok");
          } catch (err) { flash(err.message, "error"); }
        } }, "Rotate passphrase…"),
        h("button", { type: "button", onclick: async () => {
          const out = await keyChange("Confirm data-key rotation", "Start a new data-key epoch everywhere", (unlock, code) =>
            api("POST", "/api/user_auth/admin/keys/rotate_data_keys", { body: { unlock }, stepUp: code }), status);
          if (out) { flash(`New data-key epoch in ${out.length} keystore(s).`, "ok"); reload(); }
        } }, "Rotate data keys…"),
        h("button", { type: "button", class: "sec-danger", onclick: async () => {
          if (!(await confirmAction("Rotate every master key",
            "Every master key is replaced. Data, purpose keys and the index carry over, but every recovery code, Touch ID and passkey is dropped; a new recovery kit is issued.", "Rotate"))) return;
          const v = await ask("Passphrase", [{ name: "passphrase", label: "Server passphrase", type: "password" }]);
          if (!v) return;
          try {
            const out = await stepUp("Confirm master-key rotation", (code) =>
              api("POST", "/api/user_auth/admin/keys/rotate_master_keys", { body: { passphrase: v.passphrase, label: null }, stepUp: code }));
            if (out) { await showKit(out.recovery, "Master keys rotated — new recovery kit"); reload(); }
          } catch (err) { flash(err.message, "error"); }
        } }, "Rotate master keys…"))));
  }

  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Keystores"),
    h("table", { class: "sec-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Scope"), h("th", {}, "Data-key epochs"), h("th", {}, "Keyring"), h("th", {}, "Generation"))),
      h("tbody", {}, status.keystores.map((k) => h("tr", { title: k.path },
        h("td", {}, k.scope),
        h("td", {}, k.error ? h("span", { class: "sec-bad" }, k.error) : `${k.epochs.join(", ")} (active ${k.active_epoch})`),
        h("td", {}, k.ring ? "✓" : h("span", { class: "sec-bad" }, "not enrolled")),
        h("td", {}, k.generation ?? "—")))))));
}

// ── Users ───────────────────────────────────────────────────────────────

async function renderUsers(body) {
  const users = await api("GET", "/api/user_auth/admin/users");
  const act = (title, path, username, after) => async () => {
    try {
      const out = await stepUp(title, (code) => api("POST", path, { body: { username }, stepUp: code }));
      if (out) { if (after) await after(out); render(); }
    } catch (err) { flash(err.message, "error"); }
  };
  const sessionsFor = (u) => async () => {
    const list = await api("GET", "/api/user_auth/admin/sessions?username=" + encodeURIComponent(u.username));
    const rows = list.map((s) => h("tr", {},
      h("td", {}, s.id), h("td", {}, when(s.created_at)), h("td", {}, when(s.expires_at)),
      h("td", {}, s.current ? "this session" : h("button", { type: "button", class: "sec-small", onclick: async () => {
        try {
          const out = await stepUp("Revoke session", (code) =>
            api("POST", "/api/user_auth/admin/sessions/revoke", { body: { username: u.username, session_id: s.id }, stepUp: code }));
          if (out) flash("Session revoked", "ok");
        } catch (err) { flash(err.message, "error"); }
      } }, "Revoke"))));
    await show("Sessions of " + u.username, list.length
      ? h("table", { class: "sec-table" }, h("thead", {}, h("tr", {}, h("th", {}, "Id"), h("th", {}, "Started"), h("th", {}, "Expires"), h("th", {}, ""))), h("tbody", {}, rows))
      : h("p", {}, "No active sessions."));
  };
  const self = state.me.user.username;
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Users"),
    h("table", { class: "sec-table" },
      h("thead", {}, h("tr", {}, ["User", "Role", "2FA", "Last sign-in", "State", "Sessions", ""].map((t) => h("th", {}, t)))),
      h("tbody", {}, users.map((u) => h("tr", {},
        h("td", {}, u.username),
        h("td", {}, u.role),
        h("td", {}, u.totp),
        h("td", {}, when(u.last_login_at)),
        h("td", {}, u.disabled_at ? h("span", { class: "sec-bad" }, "disabled")
          : u.locked_until ? h("span", { class: "sec-warn" }, "locked until " + when(u.locked_until)) : "active"),
        h("td", {}, h("button", { type: "button", class: "sec-small sec-secondary", onclick: sessionsFor(u) }, String(u.sessions))),
        h("td", { class: "sec-row-actions" }, u.username === self ? h("span", { class: "sec-note" }, "you") : [
          u.disabled_at
            ? h("button", { type: "button", class: "sec-small", onclick: act("Enable " + u.username, "/api/user_auth/admin/users/enable", u.username) }, "Enable")
            : h("button", { type: "button", class: "sec-small sec-danger", onclick: act("Disable " + u.username, "/api/user_auth/admin/users/disable", u.username) }, "Disable"),
          u.locked_until ? h("button", { type: "button", class: "sec-small", onclick: act("Unlock " + u.username, "/api/user_auth/admin/users/unlock", u.username) }, "Unlock") : null,
          h("button", { type: "button", class: "sec-small", onclick: act("Reset password of " + u.username, "/api/user_auth/admin/users/reset_password", u.username,
            (out) => show("Temporary password", h("p", {}, "Give this to " + u.username + " once; their sessions were ended."), h("pre", { class: "sec-code" }, out.temporary_password))) }, "Reset password"),
          u.totp !== "none" ? h("button", { type: "button", class: "sec-small", onclick: act("Reset 2FA of " + u.username, "/api/user_auth/admin/users/reset_2fa", u.username) }, "Reset 2FA") : null,
        ]))))),
    h("div", { class: "sec-actions" },
      h("button", { type: "button", onclick: async () => {
        const v = await ask("Add a user", [
          { name: "username", label: "Username" },
          { name: "password", label: "Initial password", type: "password", autocomplete: "new-password" },
        ]);
        if (!v) return;
        try { await api("POST", "/api/user_auth/register", { body: v }); flash("Added " + v.username, "ok"); render(); }
        catch (err) { flash(err.message, "error"); }
      } }, "Add user…"))));
}

// ── Audit ───────────────────────────────────────────────────────────────

async function renderAudit(body) {
  const events = await api("GET", "/api/user_auth/admin/audit?limit=200");
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Audit trail"),
    events.length ? h("table", { class: "sec-table" },
      h("thead", {}, h("tr", {}, ["When", "Who", "Action", "Target", "Detail"].map((t) => h("th", {}, t)))),
      h("tbody", {}, events.map((e) => h("tr", {},
        h("td", {}, when(e.at)), h("td", {}, e.actor), h("td", {}, e.action), h("td", {}, e.target || ""),
        h("td", { class: "sec-mono" }, e.detail && e.detail !== "{}" ? e.detail : "")))))
      : h("p", {}, "Nothing recorded yet.")));
}

// ── Account ─────────────────────────────────────────────────────────────

async function renderAccount(body) {
  await refreshMe();
  const { user, sessions } = state.me;
  const enrol = async () => {
    try {
      const out = await api("POST", "/api/user_auth/2fa/enroll");
      const secret = new URL(out.provisioning_uri).searchParams.get("secret");
      const v = await ask("Enrol 2FA", [{ name: "totp", label: "Code from your authenticator", inputmode: "numeric" }], {
        note: "Add this account to an authenticator app with the setup key " + secret + ", then enter the code it shows. Store the recovery codes below.",
      });
      if (!v) return;
      await api("POST", "/api/user_auth/2fa/confirm", { body: { totp: v.totp } });
      await show("2FA enrolled — recovery codes", h("pre", { class: "sec-code" }, out.recovery_codes.join("\n")));
      render();
    } catch (err) { flash(err.message, "error"); }
  };
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Your account"),
    h("p", {}, `${user.username} · ${user.role} · 2FA ${user.totp} · ${user.recovery_codes_remaining} recovery codes left`),
    user.totp !== "active" ? h("p", { class: "sec-flash warn" }, "Enrol 2FA: administering keys and logins needs it.") : null,
    h("div", { class: "sec-actions" },
      user.totp !== "active" ? h("button", { type: "button", onclick: enrol }, "Enrol 2FA…") : null,
      h("button", { type: "button", onclick: async () => {
        const v = await ask("Change password", [
          { name: "current_password", label: "Current password", type: "password", autocomplete: "current-password" },
          { name: "new_password", label: "New password", type: "password", autocomplete: "new-password" },
        ]);
        if (!v) return;
        try {
          const call = (code) => api("POST", "/api/user_auth/password", { body: v, stepUp: code });
          const out = user.totp === "active" ? await stepUp("Confirm password change", call) : await call(null);
          if (out) flash(`Password changed; ${out.sessions_revoked} other session(s) ended.`, "ok");
        } catch (err) { flash(err.message, "error"); }
      } }, "Change password…"),
      user.totp === "active" ? h("button", { type: "button", onclick: async () => {
        try {
          const out = await stepUp("New recovery codes", (code) => api("POST", "/api/user_auth/recovery_codes", { stepUp: code }));
          if (out) await show("New recovery codes", h("p", {}, "The old codes no longer work."), h("pre", { class: "sec-code" }, out.recovery_codes.join("\n")));
          render();
        } catch (err) { flash(err.message, "error"); }
      } }, "New login recovery codes…") : null,
      h("button", { type: "button", class: "sec-secondary", onclick: () => signOut() }, "Sign out"))));
  body.append(h("section", { class: "sec-card" },
    h("h3", {}, "Your sessions"),
    h("table", { class: "sec-table" },
      h("thead", {}, h("tr", {}, ["Id", "Started", "Expires", ""].map((t) => h("th", {}, t)))),
      h("tbody", {}, sessions.map((s) => h("tr", {},
        h("td", {}, s.id), h("td", {}, when(s.created_at)), h("td", {}, when(s.expires_at)),
        h("td", {}, s.current ? "this one" : h("button", { type: "button", class: "sec-small", onclick: async () => {
          try { await api("POST", "/api/user_auth/sessions/revoke", { body: { session_id: s.id } }); render(); }
          catch (err) { flash(err.message, "error"); }
        } }, "End"))))))));
}

// ── Shell ───────────────────────────────────────────────────────────────

const VIEWS = { keys: ["Keys", renderKeys, true], users: ["Users", renderUsers, true], audit: ["Audit", renderAudit, true], account: ["Account", renderAccount, false] };

async function render() {
  const nav = $("sec-nav");
  const body = $("sec-body");
  nav.replaceChildren();
  body.replaceChildren();
  $("sec-where").textContent = base();
  if (!state.session) return renderSignIn(body);
  const admin = state.me?.user?.role === "admin";
  for (const [key, [label, , adminOnly]] of Object.entries(VIEWS)) {
    if (adminOnly && !admin) continue;
    nav.append(h("button", { type: "button", class: "sec-tab" + (state.view === key ? " active" : ""),
      onclick: () => { state.view = key; render(); } }, label));
  }
  if (!admin) state.view = "account";
  try {
    await VIEWS[state.view][1](body);
  } catch (err) {
    body.append(h("p", { class: "sec-flash error" }, err.code === "STEP_UP_UNAVAILABLE" || err.code === "FORBIDDEN"
      ? err.message + " (Account tab)" : err.message || String(err)));
  }
}

export function setupSecurity() {
  const overlay = $("security-overlay");
  $("open-security")?.addEventListener("click", () => {
    overlay.classList.remove("hidden");
    render();
  });
  $("security-close")?.addEventListener("click", () => {
    overlay.classList.add("hidden");
    $("sec-modal").classList.add("hidden");
  });
  window.addEventListener("veld-active-instance", () => {
    if (state.session) signOut("Switched instance: sign in to " + base() + ".");
    state.enclave = null;
  });
}
