// Knowledge bases in veld, written by a local model and kept only where a deterministic gate confirms
// them: cobolwork's utility knowledge base (rows the feed's own gate checks against IBM's pages), and
// one knowledge base per project (rows whose quote must stand verbatim in the repository at HEAD).
// Sources go into veld first; each attempt retrieves its passages from veld, proposes rows, is gated,
// and is fed its rejections back. Every attempt and verdict is a loop event, and each knowledge base
// keeps a snapshot in ~/.config/veld/loops for veld-desktop and `veld tui`.
import * as PATHS from '../paths.mjs';
import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync, appendFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { request } from 'node:http';
import * as veld from './veld.mjs';
import { writeLoop, appendEvent, readEvents, trajectory, tally } from './loops.mjs';

const HERE = PATHS.KB_HOME;
const COBOLWORK = '/Users/portll/Repositories/Portll/cobolwork-lang-20261002';
const SOURCES = join(HERE, 'sources');
const OUT = join(HERE, 'out');
// Accepted rows veld could not take when they were written; flushPending sends them once it can.
const PENDING = join(HERE, 'out', 'pending-upserts.jsonl');
const REPOS = {
  commitwork: '/Users/portll/Repositories/Portll/commitwork',
  cobolwork: '/Users/portll/Repositories/Portll/cobolwork',
  ironwork: '/Users/portll/Repositories/Portll/ironwork',
  substrate: '/Users/portll/Repositories/Portll/substrate',
  spine: '/Users/portll/Repositories/Portll/spine',
  veld: '/Users/portll/Repositories/Portll/veld',
};
const PROJECT_CAP = 600;

// The commit a project's knowledge is read at: origin/main where the clone has it, so a stale local
// main does not stand for the project, else HEAD.
const refs = new Map();
function refOf(project) {
  if (!refs.has(project)) {
    let ref = 'HEAD';
    try { execFileSync('git', ['-C', REPOS[project], 'rev-parse', '--verify', '-q', 'origin/main'], { stdio: 'ignore' }); ref = 'origin/main'; } catch { /* no remote main */ }
    refs.set(project, ref);
  }
  return refs.get(project);
}
const ATTEMPTS = 4;
const KINDS = ['fact', 'procedure', 'decision', 'invariant', 'caution'];

// LM Studio takes response_format json_schema or text only; a strict schema needs every property required.
const objectOf = (properties) => ({ type: 'object', properties, required: Object.keys(properties), additionalProperties: false });

const readJson = (p, d = null) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return d; } };

// ── sources ──────────────────────────────────────────────────────────────────

// Passages of at most `max` characters, split at line ends.
function passages(text, max = 3500) {
  const out = [];
  let cur = '';
  for (const line of text.split('\n')) {
    if (cur.length + line.length + 1 > max && cur) { out.push(cur); cur = ''; }
    cur += (cur ? '\n' : '') + line.slice(0, max);
  }
  if (cur.trim()) out.push(cur);
  return out;
}

// The IBM pages the utility knowledge base cites, each as feed/verify.mjs expects a source: a text
// file named for the document, in the sources directory the gate is pointed at.
function utilityDocs() {
  const map = readJson(join(HERE, 'sources-utility.json'), {});
  const pages = new Map();
  for (const f of readdirSync(join(HERE, 'ibm', 'pages'))) {
    const p = readJson(join(HERE, 'ibm', 'pages', f));
    if (p) pages.set(p.href, p);
  }
  const docOf = (href) => `zos31-${href.split('/')[0].replace('com.ibm.zos.v3r1.', '')}-${href.split('/').pop().replace('.htm', '')}`;
  mkdirSync(SOURCES, { recursive: true });
  const docs = new Map();
  for (const [href, p] of pages) {
    const doc = docOf(href);
    if (!existsSync(join(SOURCES, `${doc}.txt`))) writeFileSync(join(SOURCES, `${doc}.txt`), p.text);
    docs.set(href, { doc, title: p.title, url: p.url, retrieved: p.retrieved, text: p.text });
  }
  return { map, docs };
}

const SKIP = /(^|\/)(node_modules|vendor|evaluations|fixtures|\.claude|target|dist|out|tmp)\//i;
const SKIP_FILE = /(changelog|licen[cs]e|third[-_]party|handoff|code_of_conduct|security\.md$)/i;

// A project's documentation as sections under their headings, the README and docs/ first.
function projectSections(project) {
  const repo = REPOS[project];
  const files = execFileSync('git', ['-C', repo, 'ls-tree', '-r', '--name-only', refOf(project)], { encoding: 'utf8', maxBuffer: 64 << 20 }).split('\n').filter((f) => f.endsWith('.md'))
    .filter((f) => !SKIP.test(f) && !SKIP_FILE.test(f));
  const rank = (f) => (/^readme\.md$/i.test(f) ? 0 : /^docs\//i.test(f) ? 1 : !f.includes('/') ? 2 : 3);
  files.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const out = [];
  for (const path of files) {
    let text;
    try { text = execFileSync('git', ['-C', repo, 'show', `${refOf(project)}:${path}`], { encoding: 'utf8', maxBuffer: 16 << 20 }); } catch { continue; }
    let heading = path;
    let body = [];
    let n = 0;
    const flush = () => {
      const t = body.join('\n').trim();
      if (t.length >= 400) out.push({ id: `${project}:${path}#${n}`, project, path, heading, text: t.slice(0, 6000) });
      n++;
      body = [];
    };
    for (const line of text.split('\n')) {
      const h = /^#{1,3}\s+(.+)/.exec(line);
      if (h) { flush(); heading = h[1].trim(); continue; }
      body.push(line);
    }
    flush();
    if (out.length >= PROJECT_CAP) break;
  }
  return out.slice(0, PROJECT_CAP);
}

// Puts a knowledge base's sources into veld, resumably: each passage is upserted by its external id,
// retried after a pause when veld stalls, and recorded in ingested/<kb>.progress.json once stored.
export async function ingest(kbId, { onProgress = () => {} } = {}) {
  const kb = KBS[kbId];
  const progressPath = join(HERE, 'ingested', `${kbId}.progress.json`);
  mkdirSync(join(HERE, 'ingested'), { recursive: true });
  const done = new Set(readJson(progressPath, []));
  const all = kb.sourcePassages();
  let stored = 0;
  for (const s of all) {
    if (done.has(s.externalId)) continue;
    for (let attempt = 1; ; attempt++) {
      try {
        await veld.upsert({ userId: kb.sourceNamespace, externalId: s.externalId, content: s.content, tags: s.tags, memoryType: 'Context' });
        break;
      } catch (e) {
        if (attempt >= 4) throw e;
        await new Promise((r) => setTimeout(r, 5000 * attempt));
      }
    }
    done.add(s.externalId);
    stored++;
    if (stored % 20 === 0) { writeFileSync(progressPath, JSON.stringify([...done])); onProgress(done.size, all.length); }
  }
  writeFileSync(progressPath, JSON.stringify([...done]));
  appendEvent(kbId, { kind: 'ingest', passages: all.length, stored, namespace: kb.sourceNamespace });
  return all.length;
}

// ── knowledge bases ─────────────────────────────────────────────────────────

const SECRET = /(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|\b(password|passwd|pwd|secret|api[_-]?key|token)\s*[:=]\s*\S{6,}|\b[0-9a-f]{32,}\b)/i;
const norm = (s) => String(s).replace(/[‘’ʼ]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, ' ').trim();

const utility = {
  id: 'cobolwork-utility',
  title: "cobolwork's utility knowledge base",
  project: 'cobolwork',
  namespace: 'cobolwork',
  sourceNamespace: 'kb-src-cobolwork',
  targets() {
    return readJson(join(COBOLWORK, 'feed', 'worklists', 'utility.json'), []).map((t) => ({ ...t, id: t.id }));
  },
  sourcePassages() {
    const { docs } = utilityDocs();
    const out = [];
    for (const d of docs.values()) passages(d.text).forEach((p, k) => out.push({
      externalId: `${d.doc}#${k}`, tags: ['kb-source', `kb:${this.id}`, `doc:${d.doc}`], content: `[${d.doc}] ${d.title}\nURL: ${d.url}\nRetrieved: ${d.retrieved}\n\n${p}`,
    }));
    return out;
  },
  docsFor(target) {
    const { map, docs } = utilityDocs();
    return (map[target.id] || []).map((h) => docs.get(h)).filter(Boolean);
  },
  query: (t) => `${t.utility} ${t.parameter} ${t.note || ''}`.trim(),
  recallTags(target) { return this.docsFor(target).map((d) => `doc:${d.doc}`); },
  schema: objectOf({
    kind: { type: 'string', enum: ['utility'] },
    utility: { type: 'string' },
    parameter: { type: 'string' },
    effect: { type: 'string' },
    severity: { type: 'string', enum: ['crit', 'high', 'med', 'low', 'info'] },
    rationale: { type: 'string' },
    jcl: { type: 'string' },
    source: objectOf({ doc: { type: 'string' }, url: { type: 'string' }, retrieved: { type: 'string' }, quote: { type: 'string' } }),
  }),
  system: `You describe one parameter of one z/OS utility and what makes it worth seeing in a security review, as one JSON object. Fields:
- kind: "utility"
- utility: the program name, 1 to 8 characters (IDCAMS, IKJEFT01, SORT, IEBGENER, IEFBR14, IEBCOPY, ADRDSSU, FTP)
- parameter: the parameter, command or statement named in the task
- effect: one sentence, what it does to data or to access
- severity: one of crit, high, med, low, info - how much a reviewer should care when a job uses it
- rationale: at least 60 characters, why it matters in a security review of batch jobs
- jcl: a JCL fragment invoking the utility with that parameter. Statements begin with // in columns 1 and 2, the name field is 1 to 8 characters, the operation is a real JCL operation (JOB, EXEC, DD, ...), nothing exceeds 72 columns, a continued operand resumes between columns 4 and 16. Use \\n between lines.
- source: { doc, url, retrieved, quote } where doc, url and retrieved are copied exactly from one passage header below, and quote is at least 40 characters copied VERBATIM from that passage's text - the gate checks it character for character.
Answer with the JSON object only. Do not invent: everything must come from the passages.`,
  user(target, passagesText, problems) {
    return `Utility: ${target.utility}\nParameter: ${target.parameter}${target.note ? `\nContext: ${target.note}` : ''}\n\nPassages retrieved from veld:\n\n${passagesText}${problems.length ? `\n\nYour previous answer was rejected by the gate. Fix every one of these:\n${problems.map((p) => `- ${p}`).join('\n')}` : ''}`;
  },
  async gate(target, answer) {
    const { verifyRow } = await import(join(COBOLWORK, 'feed', 'verify.mjs'));
    const row = answer;
    const problems = verifyRow(row, { sourcesDir: SOURCES });
    const allowed = this.docsFor(target).map((d) => d.doc);
    if (row && row.source && !allowed.includes(row.source.doc)) problems.push(`source.doc: must be one of ${allowed.join(', ')}`);
    if (row && row.utility && row.utility !== target.utility) problems.push(`utility: must be ${target.utility}`);
    if (row && row.parameter && String(row.parameter).toUpperCase() !== target.parameter.toUpperCase()) problems.push(`parameter: must be ${target.parameter}`);
    return { rows: problems.length ? [] : [row], problems };
  },
  stored(target, row) {
    return {
      externalId: `kb:utility:${target.id}`,
      tags: ['kb', 'kb:utility', `utility:${row.utility}`, `parameter:${row.parameter}`, `severity:${row.severity}`],
      content: `${row.utility} ${row.parameter}: ${row.effect}\nSeverity ${row.severity}. ${row.rationale}\nJCL:\n${row.jcl}\nSource: ${row.source.doc} (${row.source.url}), retrieved ${row.source.retrieved}: "${row.source.quote}"`,
    };
  },
};

function projectKb(project) {
  return {
    id: `project-${project}`,
    title: `${project}'s project knowledge base`,
    project,
    namespace: project,
    sourceNamespace: `kb-src-${project}`,
    standsAlone: true,
    targets() { return projectSections(project); },
    sourcePassages() {
      return projectSections(project).map((s) => ({ externalId: s.id, tags: ['kb-source', `kb:${this.id}`, `path:${s.path}`], content: `[${s.path} § ${s.heading}]\n${s.text}` }));
    },
    query: (t) => `${t.heading} ${t.text.slice(0, 200)}`,
    recallTags: (t) => [`path:${t.path}`],
    schema: objectOf({
      rows: {
        type: 'array', minItems: 1, maxItems: 3,
        items: objectOf({ topic: { type: 'string' }, kind: { type: 'string', enum: KINDS }, claim: { type: 'string' }, quote: { type: 'string' } }),
      },
    }),
    system: `You turn one section of a project's documentation into one to three knowledge rows that a developer or an AI agent working on the project would want recalled. Answer with a JSON object { "rows": [ ... ] }, each row:
- topic: a short noun phrase, under 80 characters
- kind: one of fact, procedure, decision, invariant, caution
- claim: 40 to 500 characters, in your own words, saying what is true, how to do something, what was decided and why, what must always hold, or what to avoid. Specific to this project; no generic advice.
- quote: 40 to 400 characters copied VERBATIM from the section text (the gate checks it against the file in the repository, ignoring only whitespace), the sentence that establishes the claim.
Never include a credential, key or token. Rows that only restate a heading are refused. Answer with the JSON object only.`,
    user(target, passagesText, problems) {
      return `Project: ${project}\nFile: ${target.path}\nSection: ${target.heading}\n\nSection text:\n${target.text}\n\nRelated passages retrieved from veld:\n\n${passagesText}${problems.length ? `\n\nYour previous answer was rejected by the gate. Fix every one of these:\n${problems.map((p) => `- ${p}`).join('\n')}` : ''}`;
    },
    async gate(target, answer) {
      const problems = [];
      const rows = [];
      const list = Array.isArray(answer?.rows) ? answer.rows : null;
      if (!list || !list.length) return { rows, problems: ['rows: give one to three rows'] };
      let file;
      try { file = norm(execFileSync('git', ['-C', REPOS[project], 'show', `${refOf(project)}:${target.path}`], { encoding: 'utf8', maxBuffer: 16 << 20 })); } catch { return { rows, problems: [`${target.path} is not in the repository at ${refOf(project)}`] }; }
      list.slice(0, 3).forEach((r, k) => {
        const p = [];
        const at = `rows[${k}]`;
        if (typeof r.topic !== 'string' || !r.topic.trim() || r.topic.length > 80) p.push(`${at}.topic: a phrase under 80 characters`);
        if (!KINDS.includes(r.kind)) p.push(`${at}.kind: one of ${KINDS.join(', ')}`);
        if (typeof r.claim !== 'string' || r.claim.length < 40 || r.claim.length > 500) p.push(`${at}.claim: 40 to 500 characters`);
        if (typeof r.quote !== 'string' || r.quote.length < 40 || r.quote.length > 400) p.push(`${at}.quote: 40 to 400 characters`);
        else if (!file.includes(norm(r.quote))) p.push(`${at}.quote: does not appear verbatim in ${target.path}`);
        if (typeof r.claim === 'string' && typeof r.quote === 'string' && norm(r.claim).includes(norm(r.quote).slice(0, 60))) p.push(`${at}.claim: say it in your own words, not by copying the quote`);
        if (SECRET.test(`${r.claim} ${r.quote}`)) p.push(`${at}: looks like it carries a credential`);
        if (p.length) problems.push(...p);
        else rows.push({ project, path: target.path, heading: target.heading, ...r });
      });
      return { rows, problems };
    },
    stored(target, row, k) {
      return {
        externalId: `kb:${target.id}:${k}`,
        tags: ['kb', 'kb:project', `path:${row.path}`, `kind:${row.kind}`],
        content: `${row.topic}: ${row.claim}\nSource: ${row.path} § ${row.heading}: "${row.quote}"`,
      };
    },
  };
}

export const KBS = Object.fromEntries([utility, ...Object.keys(REPOS).map(projectKb)].map((k) => [k.id, k]));

// ── one target ──────────────────────────────────────────────────────────────

function chat(host, system, user, schema) {
  const body = JSON.stringify({
    model: 'qwen/qwen3.8-27b', temperature: 0.3, max_tokens: 2400, reasoning_effort: 'none',
    response_format: { type: 'json_schema', json_schema: { name: 'kb_answer', strict: true, schema } },
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
  });
  return new Promise((resolve, reject) => {
    const req = request(`${host.url}/chat/completions`, { method: 'POST', agent: false, headers: { 'content-type': 'application/json', authorization: `Bearer ${host.key}` } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        if (res.statusCode !== 200) return reject(new Error(`model answered ${res.statusCode}: ${text.slice(0, 200)}`));
        try {
          const j = JSON.parse(text);
          resolve({ content: j.choices?.[0]?.message?.content || '', tokens: j.usage?.completion_tokens || 0 });
        } catch (e) { reject(e); }
      });
    });
    req.setTimeout(15 * 60 * 1000, () => req.destroy(new Error('model timed out')));
    req.on('error', reject);
    req.end(body);
  });
}

const parseAnswer = (text) => {
  const cleaned = String(text).replace(/^\s*```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
  return JSON.parse(cleaned);
};

// One target through retrieve, propose, gate and feed back, up to ATTEMPTS times. Every attempt is
// an event; accepted rows go into veld and into out/<kb>.jsonl for review.
export async function flushPending() {
  if (!existsSync(PENDING)) return 0;
  const rows = readFileSync(PENDING, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const left = [];
  for (const m of rows) { try { await veld.upsert(m); } catch { left.push(m); } }
  writeFileSync(PENDING, left.map((m) => JSON.stringify(m) + '\n').join(''));
  return rows.length - left.length;
}

export async function runKbJob(job, host) {
  const kb = KBS[job.kb];
  const target = kb.targets().find((t) => t.id === job.target);
  if (!target) return { accepted: 0, attempts: 0, tokens: 0, gone: `${job.target} is no longer a target of ${job.kb}` };
  let problems = [];
  let tokens = 0;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    const t0 = Date.now();
    // A project target carries its own section text, so veld's related passages are a help it can
    // draft without; the utility KB quotes the passages themselves and waits for them.
    let found = [];
    try {
      found = await veld.recall({ userId: kb.sourceNamespace, query: kb.query(target), tags: kb.recallTags(target), limit: 6 });
    } catch (e) {
      if (!kb.standsAlone) {
        appendEvent(kb.id, { kind: 'attempt', target: target.id, attempt, outcome: 'error', reasons: [`veld: ${e.message}`] });
        writeSnapshot(kb.id);
        throw e;
      }
    }
    if (!found.length && !kb.standsAlone) {
      appendEvent(kb.id, { kind: 'attempt', target: target.id, attempt, outcome: 'error', reasons: ['veld returned no passage for this target'] });
      writeSnapshot(kb.id);
      throw new veld.VeldUnavailable(`veld returned no passage for ${target.id}`);
    }
    const passagesText = found.map((m) => m.content).join('\n\n---\n\n');
    let answer = null;
    let verdict;
    let r;
    try {
      r = await chat(host, kb.system, kb.user(target, passagesText, problems), kb.schema);
    } catch (e) {
      appendEvent(kb.id, { kind: 'attempt', target: target.id, attempt, outcome: 'error', reasons: [`model: ${String(e.message).slice(0, 200)}`] });
      writeSnapshot(kb.id);
      throw e;
    }
    tokens += r.tokens;
    try {
      answer = parseAnswer(r.content);
      verdict = await kb.gate(target, answer);
    } catch (e) {
      verdict = { rows: [], problems: [`the answer was not usable JSON: ${String(e.message).slice(0, 120)}`] };
    }
    const sec = Math.round((Date.now() - t0) / 1000);
    if (verdict.rows.length) {
      mkdirSync(OUT, { recursive: true });
      for (const [k, row] of verdict.rows.entries()) {
        const s = kb.stored(target, row, k);
        const memory = { userId: kb.namespace, externalId: s.externalId, content: s.content, tags: s.tags, memoryType: 'Learning' };
        try { await veld.upsert(memory); } catch { appendFileSync(PENDING, JSON.stringify(memory) + '\n'); }
        appendFileSync(join(OUT, `${kb.id}.jsonl`), JSON.stringify({ target: target.id, ...row }) + '\n');
      }
      appendEvent(kb.id, { kind: 'attempt', target: target.id, attempt, outcome: 'accepted', rows: verdict.rows.length, partial: verdict.problems.length ? verdict.problems.slice(0, 4) : undefined, sec });
      writeSnapshot(kb.id);
      return { accepted: verdict.rows.length, attempts: attempt, tokens };
    }
    problems = verdict.problems;
    appendEvent(kb.id, { kind: 'attempt', target: target.id, attempt, outcome: 'rejected', reasons: problems.slice(0, 6), sec });
  }
  appendEvent(kb.id, { kind: 'verdict', target: target.id, outcome: 'rejected', reasons: problems.slice(0, 6) });
  writeSnapshot(kb.id);
  return { accepted: 0, attempts: ATTEMPTS, tokens };
}

// ── the loop's self-evaluation ──────────────────────────────────────────────

// What the loop knows about itself: how many targets ended accepted or rejected, how often the first
// attempt was enough, which gate rules reject most, and how that moves hour by hour.
export function writeSnapshot(kbId) {
  const kb = KBS[kbId];
  const events = readEvents(kbId);
  const attempts = events.filter((e) => e.kind === 'attempt');
  const last = new Map();
  for (const e of attempts) last.set(e.target, e);
  for (const e of events.filter((x) => x.kind === 'verdict')) last.set(e.target, e);
  const accepted = [...last.values()].filter((e) => e.outcome === 'accepted');
  const rejected = [...last.values()].filter((e) => e.outcome === 'rejected' && e.kind === 'verdict');
  const firstTry = accepted.filter((e) => e.attempt === 1).length;
  let total = null;
  try { total = kb.targets().length; } catch { total = null; }
  const reasonKey = (r) => String(r).replace(/rows\[\d\]\./, '').replace(/'[^']*'/g, "'…'").split(':')[0].slice(0, 60);
  writeLoop(kbId, {
    kind: 'kb',
    title: kb.title,
    namespace: kb.namespace,
    counts: {
      targets: total, accepted: accepted.length, rejected: rejected.length,
      pending: total == null ? null : Math.max(0, total - accepted.length - rejected.length),
      attempts: attempts.length, rows: accepted.reduce((n, e) => n + (e.rows || 1), 0),
      errors: attempts.filter((e) => e.outcome === 'error').length,
    },
    rates: {
      acceptedFirstTry: accepted.length ? +(firstTry / accepted.length).toFixed(3) : null,
      accepted: accepted.length + rejected.length ? +(accepted.length / (accepted.length + rejected.length)).toFixed(3) : null,
      attemptsPerTarget: last.size ? +(attempts.length / last.size).toFixed(2) : null,
    },
    topReasons: tally(attempts.filter((e) => e.outcome !== 'accepted').flatMap((e) => e.reasons || []), reasonKey),
    trajectory: trajectory(attempts, (e) => e.outcome),
    recent: events.slice(-15).reverse(),
  });
}
