// DOM helpers and modal dialogs shared by the overlay panels (Security, Data).

export function h(tag, attrs = {}, ...children) {
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

export const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

export function fmtBytes(n) {
  if (n == null) return "—";
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + " GiB";
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + " MiB";
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + " KiB";
  return n + " B";
}

// Dialogs bound to one overlay's modal root and flash line.
export function makeDialogs({ modalId, flashId }) {
  const $ = (id) => document.getElementById(id);

  function modal(title, body, actions) {
    const root = $(modalId);
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
  // A field with `checkbox: true` resolves to a boolean; `multiline: true` is a textarea.
  function ask(title, fields, { submit = "Continue", note } = {}) {
    return new Promise((resolve) => {
      const inputs = {};
      const form = h("form", { class: "sec-form" },
        note ? h("p", { class: "sec-note" }, note) : null,
        fields.map((f) => {
          const input = f.options
            ? h("select", { name: f.name }, f.options.map((o) => h("option", { value: o.value }, o.label)))
            : f.multiline
            ? h("textarea", { name: f.name, rows: f.rows || 6, placeholder: f.placeholder || "", spellcheck: "false" })
            : h("input", {
                name: f.name, type: f.checkbox ? "checkbox" : f.type || "text", placeholder: f.placeholder || "",
                autocomplete: f.autocomplete || "off", inputmode: f.inputmode,
                required: f.optional || f.checkbox ? false : true,
              });
          if (f.checkbox) input.checked = !!f.value;
          else if (f.value !== undefined) input.value = f.value;
          inputs[f.name] = input;
          return f.checkbox
            ? h("label", { class: "sec-check" }, input, h("span", {}, f.label))
            : h("label", {}, h("span", {}, f.label), input);
        }));
      let close;
      const done = (value) => { close(); resolve(value); };
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        done(Object.fromEntries(Object.entries(inputs).map(([k, el]) =>
          [k, el.type === "checkbox" ? el.checked : el.value])));
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

  let flashTimer;
  function flash(text, kind = "info") {
    const el = $(flashId);
    el.textContent = text;
    el.className = "sec-flash " + kind;
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => (el.className = "sec-flash hidden"), 6000);
  }

  return { modal, ask, show, confirmAction, flash };
}
