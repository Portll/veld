import { invoke } from "@tauri-apps/api/core";
import type { AppInfo, GhAccount, RepoNode, ServerHealth, ViewName } from "./types";

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
};

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] ?? c));
}

/** Account → Org → Repo tree from the discovery commands. Folders arrive in P2. */
async function renderTree(): Promise<void> {
  const content = $("content");
  content.innerHTML = '<p class="placeholder">scanning ~/Repositories…</p>';
  const [accounts, repos] = await Promise.all([
    invoke<GhAccount[]>("list_gh_accounts"),
    invoke<RepoNode[]>("scan_repos"),
  ]);

  const byAccount = new Map<string, Map<string, RepoNode[]>>();
  for (const a of accounts) byAccount.set(a.login, new Map());
  for (const r of repos) {
    const acct = r.account || "unknown";
    const org = r.org || "(no remote)";
    if (!byAccount.has(acct)) byAccount.set(acct, new Map());
    const orgs = byAccount.get(acct)!;
    if (!orgs.has(org)) orgs.set(org, []);
    orgs.get(org)!.push(r);
  }

  const out: string[] = ['<div class="tree">'];
  for (const acct of [...byAccount.keys()].sort()) {
    const isDefault = accounts.find((a) => a.login === acct)?.is_default;
    out.push(
      `<div class="acct"><div class="node acct-node">⦿ ${esc(acct)}` +
        (isDefault ? ' <span class="tag">default</span>' : "") +
        "</div>",
    );
    const orgs = byAccount.get(acct)!;
    for (const org of [...orgs.keys()].sort()) {
      const list = orgs.get(org)!.sort((a, b) => a.repo.localeCompare(b.repo));
      const boundary = list[0]?.boundary ?? "external";
      out.push(
        `<div class="org"><div class="node org-node">${esc(org)}/ ` +
          `<span class="badge ${boundary}">${boundary}</span></div>`,
      );
      for (const r of list) {
        const marks =
          (r.is_worktree ? '<span class="mark">worktree</span>' : "") +
          (r.is_symlink ? '<span class="mark">symlink</span>' : "");
        out.push(
          `<div class="repo"><div class="node repo-node">${esc(r.repo)} ${marks}` +
            `<span class="hash" title="${esc(r.path)}">${r.project_hash}</span></div></div>`,
        );
      }
      out.push("</div>");
    }
    out.push("</div>");
  }
  out.push("</div>");
  out.push(
    `<p class="muted">${repos.length} repos · ${byAccount.size} accounts · project_hash matches CLAUDE_PROJECT_HASH</p>`,
  );
  content.innerHTML = out.join("");
}

function renderView(view: ViewName): void {
  if (view === "tree") {
    void renderTree();
  } else {
    $("content").innerHTML = `<p class="placeholder">${view} view — coming next.</p>`;
  }
}

function wireRail(): void {
  document.querySelectorAll<HTMLButtonElement>("#rail button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#rail button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderView((btn.dataset.view as ViewName) ?? "tree");
    });
  });
}

async function refreshHealth(): Promise<void> {
  try {
    const h = await invoke<ServerHealth>("server_health");
    $("dot").className = "dot " + (h.reachable ? "ok" : "err");
    $("status-text").textContent = h.reachable ? `veld · ${h.rtt_ms ?? "?"}ms` : "veld unreachable";
  } catch {
    $("dot").className = "dot err";
    $("status-text").textContent = "veld error";
  }
}

async function boot(): Promise<void> {
  wireRail();
  try {
    const info = await invoke<AppInfo>("app_info");
    $("app-title").textContent = `${info.name} v${info.version}`;
  } catch (err) {
    $("content").innerHTML = `<pre class="error">${esc(String(err))}</pre>`;
    return;
  }
  await refreshHealth();
  window.setInterval(() => void refreshHealth(), 5000);
  renderView("tree");
}

void boot();
