// Drains the PL/I job bank through one local Qwen host: build the prompt from the job and the
// feat/pli head, get the draft, gate it in a throwaway worktree, and queue a repair when it fails.
//
//   node runner.mjs --host local|m4 [--once]
//
// bank/<id>.json is the job; state/<id>.json its status; out/<id>/ the prompt, reply, files and
// gate; ledger.jsonl one row per run; status.json the queue and its ready buffer in minutes.
import * as PATHS from '../paths.mjs';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, existsSync, readdirSync, rmSync, appendFileSync, openSync, closeSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync, spawnSync } from 'node:child_process';
import { request } from 'node:http';
const { runPrompt } = await import(join(PATHS.SPINE_REPO, 'prompt-runner.mjs'));

const ROOT = PATHS.LANE_HOME;
// A track is one reader being built: its worktree and branch, its corpus and measure, the paths a
// draft may write and the prompt template. A job names its track; PL/I is the default.
const LANG = PATHS.LANG_HOME;
const TRACKS = {
  pli: { repo: '/Users/portll/Repositories/Portll/cobolwork-pli-20261001', branch: 'feat/pli', corpus: join(ROOT, 'corpus'), synthetic: ['jcf608_PacificNationalBank'],
    measure: './diag/pli-measure.mjs', allowed: /^(lib\/pli\/|test\/pli-[A-Za-z0-9_-]+\.test\.mjs$|diag\/pli-|bench\/pli\/)/, template: 'cobolwork_pli' },
  ims: { repo: '/Users/portll/Repositories/Portll/cobolwork-lang-20261002', branch: 'feat/ims-db2', corpus: join(LANG, 'corpus-ims'), synthetic: ['jcf608_PacificNationalBank'],
    measure: './diag/ims-measure.mjs', allowed: /^(lib\/ims\/|test\/ims-[A-Za-z0-9_-]+\.test\.mjs$)/, template: 'cobolwork_lang' },
  db2: { repo: '/Users/portll/Repositories/Portll/cobolwork-lang-20261002', branch: 'feat/ims-db2', corpus: join(LANG, 'corpus-db2'), synthetic: ['jcf608_PacificNationalBank'],
    measure: './diag/db2-measure.mjs', allowed: /^(lib\/db2\/|test\/db2-[A-Za-z0-9_-]+\.test\.mjs$)/, template: 'cobolwork_lang' },
  hlasm: { repo: '/Users/portll/Repositories/Portll/cobolwork-hlasm-20261003', branch: 'feat/hlasm', corpus: join(LANG, 'corpus-hlasm'), synthetic: [],
    measure: './diag/hlasm-statement-measure.mjs', allowed: /^(lib\/hlasm\/|test\/hlasm-[A-Za-z0-9_-]+\.test\.mjs$|bench\/cases\/hlasm-[a-z0-9-]+\/[A-Za-z0-9_.$#@-]+$|bench\/hlasm-locate\/[a-z0-9-]+\.asm$)/, template: 'cobolwork_lang' },
};
const trackOf = (job) => ({ name: job.track || 'pli', ...TRACKS[job.track || 'pli'] });
const PLI = trackOf({});
const MAX_ROUNDS = 2;
const LOW_BUFFER_MIN = 240;
const args = process.argv.slice(2);
const HOST = args.includes('--host') ? args[args.indexOf('--host') + 1] : 'local';
// Two lanes per machine against one LM Studio slot: while one lane gates a draft or waits on veld,
// the other's request is already queued, so the model never waits for the next one.
const HOSTS = {
  local: { url: 'http://127.0.0.1:1234/v1', key: 'LMSTUDIO_LOCAL_API_KEY' },
  'local-b': { url: 'http://127.0.0.1:1234/v1', key: 'LMSTUDIO_LOCAL_API_KEY' },
  m4: { url: 'http://192.168.1.4:1234/v1', key: 'LMSTUDIO_M4PRO_API_KEY' },
  'm4-b': { url: 'http://192.168.1.4:1234/v1', key: 'LMSTUDIO_M4PRO_API_KEY' },
  // Groq takes only jobs that name it; the LM Studio lanes leave those alone.
  groq: { provider: 'groq' },
};
const mine = (job) => (HOSTS[HOST].provider || null) === (job.provider || null);
for (const d of ['bank', 'state', 'out', 'metrics']) mkdirSync(join(ROOT, d), { recursive: true });

const now = () => new Date().toISOString();
const log = (m) => appendFileSync(join(ROOT, `runner-${HOST}.log`), `${now()} ${m}\n`);
const readJson = (p, d = null) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return d; } };
const git = (t, ...a) => execFileSync('git', ['-C', t.repo, ...a], { encoding: 'utf8', maxBuffer: 64 << 20 });
const headSha = (t = PLI) => git(t, 'rev-parse', t.branch).trim();
const atHead = (t, file) => { try { return git(t, 'show', `${t.branch}:${file}`); } catch { return null; } };
const samplesFile = (t) => join(ROOT, 'metrics', t.name === 'pli' ? 'head-samples.json' : `head-samples-${t.name}.json`);

function httpFetch(url, { method = 'GET', headers = {}, body, signal } = {}) {
  return new Promise((resolve, reject) => {
    // A fresh connection per request: a pooled keep-alive socket the server has closed answers ECONNRESET.
    const req = request(url, { method, headers, agent: false }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => { const text = Buffer.concat(chunks).toString('utf8'); resolve({ status: res.statusCode, ok: res.statusCode >= 200 && res.statusCode < 300, text: async () => text, json: async () => JSON.parse(text) }); });
      res.on('error', reject);
    });
    req.setTimeout(0);
    req.on('error', reject);
    if (signal) signal.addEventListener('abort', () => req.destroy(signal.reason || new Error('aborted')), { once: true });
    if (body) req.write(body);
    req.end();
  });
}

const jobs = () => readdirSync(join(ROOT, 'bank')).filter((f) => f.endsWith('.json')).map((f) => readJson(join(ROOT, 'bank', f))).filter(Boolean);
const stateOf = (id) => readJson(join(ROOT, 'state', `${id}.json`), { status: 'pending' });
const setState = (id, patch) => writeFileSync(join(ROOT, 'state', `${id}.json`), JSON.stringify({ ...stateOf(id), ...patch, updatedAt: now() }, null, 1));
const DONE = new Set(['landed', 'dropped']);

// A knowledge-base job needs veld up and its sources ingested; health is asked at most once a minute.
const KB = PATHS.KB_HOME;
let veldUp = { at: 0, ok: false };
function veldHealthy() {
  if (Date.now() - veldUp.at > 60000) {
    const r = spawnSync('curl', ['-s', '-m', '5', '-o', '/dev/null', '-w', '%{http_code}', 'http://127.0.0.1:3030/health'], { encoding: 'utf8' });
    veldUp = { at: Date.now(), ok: r.stdout === '200' };
  }
  return veldUp.ok;
}

function needsMet(job) {
  // A project KB drafts from its own section text and queues its rows while veld is busy; the utility
  // KB quotes veld's passages, so it waits for veld.
  if (job.type === 'kb') return existsSync(join(KB, 'ingested', `${job.kb}.json`)) && (job.kb.startsWith('project-') || veldHealthy());
  for (const n of job.needs || []) {
    const body = atHead(trackOf(job), n.file);
    if (body == null || (n.contains && !body.includes(n.contains))) return false;
  }
  for (const a of job.after || []) if (!DONE.has(stateOf(a).status)) return false;
  return true;
}

function ledgerRows() {
  const p = join(ROOT, 'ledger.jsonl');
  return existsSync(p) ? readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
}

function estimateMin(job, rows) {
  const same = rows.filter((r) => r.type === (job.type || 'draft') && r.sec).map((r) => r.sec / 60).sort((a, b) => a - b);
  if (same.length >= 3) return Math.round(same[Math.floor(same.length / 2)] + 1);
  return job.estimateMin || 15;
}

// Lanes whose log moved in the last 20 minutes; the buffer in wall-clock time divides by them.
function lanes() {
  const fresh = Object.keys(HOSTS).filter((h) => { try { return Date.now() - statSync(join(ROOT, `runner-${h}.log`)).mtimeMs < 20 * 60 * 1000; } catch { return false; } });
  return Math.max(1, fresh.length);
}

function queue() {
  const rows = ledgerRows();
  const all = jobs().map((j) => ({ j, s: stateOf(j.id) }));
  const pending = all.filter(({ s }) => s.status === 'pending');
  const ready = pending.filter(({ j }) => needsMet(j)).sort((a, b) => (b.j.priority || 0) - (a.j.priority || 0) || String(a.j.id).localeCompare(String(b.j.id)));
  const blocked = pending.filter((x) => !ready.includes(x));
  const minutes = (xs) => xs.reduce((n, { j }) => n + estimateMin(j, rows), 0);
  const count = (st) => all.filter(({ s }) => s.status === st).length;
  return {
    ready, blocked,
    status: {
      at: now(), host: HOST, head: headSha(),
      readyJobs: ready.length, readyMin: minutes(ready), lanes: lanes(), readyWallMin: Math.round(minutes(ready) / lanes()), blockedJobs: blocked.length, blockedMin: minutes(blocked),
      running: all.filter(({ s }) => s.status === 'running').map(({ j, s }) => ({ id: j.id, host: s.host, since: s.startedAt })),
      passedAwaitingReview: count('passed'), failed: count('failed'), landed: count('landed'), dropped: count('dropped'),
      runs: rows.length, qwenHours: +(rows.reduce((n, r) => n + (r.sec || 0), 0) / 3600).toFixed(2),
      next: ready.slice(0, 8).map(({ j }) => j.id),
    },
  };
}

function writeStatus(extra = {}) {
  const { status } = queue();
  writeFileSync(join(ROOT, 'status.json'), JSON.stringify({ ...status, ...extra }, null, 1));
  exportLaneLoop(status).catch((e) => log(`loop export failed: ${e.message}`));
  return status;
}

// The lane's self-evaluation, for veld-desktop and veld tui: how drafts fare at the gate by family,
// what review made of the ones that passed, and how much work is queued.
const FAMILY = (id) => (id.startsWith('k-') ? 'knowledge base' : /^p-|^f\d/.test(id) ? 'benchmark pair' : /^[ib]\d|^d\d|^c\d|^e\d/.test(id) ? 'parser or rule' : /^h-/.test(id) ? 'hill-climb' : /^a-/.test(id) ? 'audit' : 'other');
async function exportLaneLoop(status) {
  const { writeLoop, trajectory, tally } = await import(new URL('../kb/loops.mjs', import.meta.url));
  const rows = ledgerRows();
  const runs = rows.filter((r) => r.type !== 'review');
  const reviews = rows.filter((r) => r.type === 'review');
  const byFamily = {};
  for (const r of runs) {
    const f = (byFamily[FAMILY(r.id)] ||= { runs: 0, passed: 0, failed: 0, errors: 0 });
    f.runs++;
    if (r.status === 'passed' || r.status === 'kb-accepted') f.passed++;
    else if (r.status === 'error') f.errors++;
    else f.failed++;
  }
  for (const f of Object.values(byFamily)) f.passRate = f.runs - f.errors ? +(f.passed / (f.runs - f.errors)).toFixed(3) : null;
  const landed = reviews.filter((r) => r.status === 'landed');
  const outcome = (r) => (r.status === 'passed' || r.status === 'kb-accepted' ? 'passed' : r.status === 'error' ? 'error' : 'failed');
  writeLoop('qwen-lane', {
    kind: 'lane',
    title: 'Qwen work lanes (local and m4)',
    lanes: status.lanes,
    counts: {
      runs: runs.length, passed: runs.filter((r) => outcome(r) === 'passed').length, failed: runs.filter((r) => outcome(r) === 'failed').length,
      errors: runs.filter((r) => outcome(r) === 'error').length, landed: landed.length, dropped: reviews.filter((r) => r.status === 'dropped').length,
      readyJobs: status.readyJobs, readyWallMin: status.readyWallMin, qwenHours: status.qwenHours,
    },
    rates: {
      gatePass: runs.length ? +(runs.filter((r) => outcome(r) === 'passed').length / Math.max(1, runs.filter((r) => outcome(r) !== 'error').length)).toFixed(3) : null,
      landedAsDrafted: landed.length ? +(landed.filter((r) => r.accepted === 'all').length / landed.length).toFixed(3) : null,
      correctionMinPerLanding: landed.length ? +(landed.reduce((n, r) => n + (r.correctionMin || 0), 0) / landed.length).toFixed(1) : null,
    },
    byFamily,
    topReasons: tally(reviews.filter((r) => r.status === 'dropped'), (r) => String(r.note || '').split(/[;:(]/)[0].slice(0, 60)),
    trajectory: trajectory(runs.filter((r) => r.startedAt).map((r) => ({ at: r.startedAt, outcome: outcome(r) })), (e) => e.outcome),
    recent: runs.slice(-15).reverse().map((r) => ({ at: r.startedAt, target: r.id, outcome: r.status, host: r.host, sec: r.sec })),
  });
}

// A job is taken by creating its lock file exclusively, so two lanes never draft the same job.
function take(job) {
  const lock = join(ROOT, 'state', `${job.id}.lock`);
  try { closeSync(openSync(lock, 'wx')); } catch { return false; }
  setState(job.id, { status: 'running', host: HOST, startedAt: now() });
  return true;
}

function fence(lang, body) { return `\`\`\`${lang}\n${body.replace(/\n$/, '')}\n\`\`\``; }

function context(job) {
  const parts = [`# Task ${job.id}: ${job.title}\n\n${job.spec}`];
  const track = trackOf(job);
  const samples = readJson(samplesFile(track), { samples: {} }).samples;
  for (const c of job.context || []) {
    if (c.file) {
      const body = atHead(track, c.file);
      parts.push(body == null ? `--- ${c.file} (does not exist yet)` : `--- ${c.file}\n${fence('js', body)}`);
    } else if (c.reference) {
      const text = existsSync(c.reference) ? readFileSync(c.reference, 'utf8') : '';
      const at = c.from ? text.indexOf(c.from) : 0;
      const end = c.to ? text.indexOf(c.to, at + 1) : -1;
      if (at >= 0 && text) parts.push(`--- ${c.title || 'Reference'} (follow this grammar; refuse what it does not show)\n${fence('text', text.slice(at, end > at ? end : at + (c.max || 6000)).slice(0, c.max || 6000))}`);
    } else if (c.samples) {
      const list = (samples[c.samples] || []).slice(0, c.n || 20);
      if (list.length) parts.push(`--- Corpus statements of kind ${c.samples} the reader does not parse yet (file:line, status, reason, then the source lines)\n${list.map((s) => `${s.file}:${s.line} ${s.status}${s.reason ? ` (${s.reason})` : ''}\n${fence('pli', s.text)}`).join('\n')}`);
    } else if (c.text) parts.push(c.text);
  }
  if (job.repairOf) {
    const prev = join(ROOT, 'out', job.repairOf);
    const gate = existsSync(join(prev, 'gate.txt')) ? readFileSync(join(prev, 'gate.txt'), 'utf8') : '';
    const files = readJson(join(prev, 'files.json'), []);
    parts.push(`# Your previous draft failed the gate\n\nFix it and reply with every file again, in full. Keep what worked.\n\n--- gate output\n${fence('text', gate.slice(-6000))}`);
    for (const f of files) parts.push(`--- your previous ${f.path}\n${fence('js', f.body)}`);
  }
  return parts.join('\n\n');
}

const PATH_LINE = /^\s*(?:[#*>`-]+\s*)*(?:File:\s*)?`?((?:lib|test|diag|bench)\/[A-Za-z0-9_./$#@-]+\.(?:mjs|js|json|pli|cbl|md|jcl|sql|ddl|dbd|psb|inc|asm|mlc|mac|cpy))`?\**\s*:?\s*$/;


export function extractFiles(reply) {
  const lines = reply.split('\n');
  const files = [];
  let pending = null;
  for (let k = 0; k < lines.length; k++) {
    const m = PATH_LINE.exec(lines[k]);
    if (m) { pending = m[1]; continue; }
    if (/^\s*```/.test(lines[k])) {
      const start = k + 1;
      let end = start;
      while (end < lines.length && !/^\s*```\s*$/.test(lines[end])) end++;
      let body = lines.slice(start, end).join('\n');
      let path = pending;
      const head = /^\/\/\s*((?:lib|test|diag|bench)\/\S+\.(?:mjs|js))\s*$/.exec(lines[start] || '');
      if (!path && head) { path = head[1]; body = lines.slice(start + 1, end).join('\n'); }
      if (path) files.push({ path, body: body.endsWith('\n') ? body : `${body}\n` });
      pending = null;
      k = end;
    }
  }
  const last = new Map();
  for (const f of files) last.set(f.path, f);
  return [...last.values()];
}

function run(cmd, a, opts = {}) {
  const r = spawnSync(cmd, a, { encoding: 'utf8', timeout: opts.timeout || 300000, cwd: opts.cwd, maxBuffer: 64 << 20, env: { ...process.env, ...(opts.env || {}) } });
  return { code: r.status, out: `${r.stdout || ''}${r.stderr || ''}`, timedOut: r.error && r.error.code === 'ETIMEDOUT' };
}

// Each failing test with its error and the stack frames inside lib/ or test/, so a repair can see
// where the error was thrown and not only that it was.
function failureLines(out, dir) {
  const lines = out.replaceAll(`file://${dir}/`, '').replaceAll(`${dir}/`, '').split('\n');
  const keep = [];
  let frames = 0;
  for (const l of lines) {
    if (/^\s*✖/.test(l)) { keep.push(l); frames = 0; continue; }
    if (/AssertionError|Error( \[[A-Z_]+\])?:|Cannot find|^\s*(expected|actual)|^\s*[+-] /.test(l)) { keep.push(l); continue; }
    if (/^\s*at .*\b(lib|test)\//.test(l) && frames < 3) { keep.push(l); frames++; }
  }
  return [...new Set(keep)].slice(0, 120).join('\n');
}

// The model cannot count columns, so a drafted PL/I file is moved inside the default margins (2,72)
// before the gate: a line starting in column 1 gains a space, and a long one is broken at a blank
// outside any string literal (a one-line comment is rewrapped instead). The gate still refuses what
// this cannot place.
function withinPliMargins(body) {
  const out = [];
  for (let line of body.split('\n')) {
    if (/^\S/.test(line) && !/^[*%]PROCESS\b/i.test(line)) line = ` ${line}`;
    line = line.replace(/\s+$/, '');
    while (line.length > 72) {
      const indent = line.length - line.trimStart().length;
      const comment = /^(\s*)\/\*\s*(.*?)\s*\*\/$/.exec(line);
      if (comment) {
        const words = comment[2].split(/\s+/);
        const lines = [];
        let cur = `${comment[1]}/*`;
        for (const w of words) {
          if (`${cur} ${w}`.length > 68 && cur.trim() !== '/*') { lines.push(cur); cur = `${comment[1]}  `; }
          cur += ` ${w}`;
        }
        lines.push(`${cur} */`);
        if (lines.some((l) => l.length > 72)) break;
        out.push(...lines.slice(0, -1));
        line = lines[lines.length - 1];
        continue;
      }
      let quoted = false;
      let cut = -1;
      for (let k = 0; k < Math.min(line.length, 72); k++) {
        if (line[k] === "'") quoted = !quoted;
        else if (line[k] === ' ' && !quoted && k > indent) cut = k;
      }
      if (cut < 0) break;
      out.push(line.slice(0, cut).replace(/\s+$/, ''));
      line = `${' '.repeat(indent + 4)}${line.slice(cut + 1).trimStart()}`;
    }
    out.push(line);
  }
  return out.join('\n');
}

// A PL/I fixture must read with no lexical error, and every statement must parse.
const PLI_CHECK = `
import { readFileSync } from 'node:fs';
import { readPli, sourceText } from './lib/pli/lex.mjs';
import { parseStatement } from './lib/pli/statements.mjs';
let bad = 0;
for (const f of process.argv.slice(1)) {
  const text = sourceText(readFileSync(f));
  const lines = text.split(/\\r?\\n/);
  const { statements, diags } = readPli(text, { file: f });
  for (const d of diags.filter((x) => x.sev === 'error')) { bad++; console.log(f + ':' + d.line + ' ' + d.kind); }
  lines.forEach((l, k) => {
    if (/^\\S/.test(l) && !/^[*%]PROCESS\\b/i.test(l)) { bad++; console.log(f + ':' + (k + 1) + ' column 1 is outside the PL/I source margins (2,72); start the line in column 2: ' + l.trim()); }
    if (l.length > 72 && l.slice(72).trim()) { bad++; console.log(f + ':' + (k + 1) + ' text past column 72 is outside the PL/I source margins; break the line'); }
  });
  let opened = 0, ended = 0;
  const count = (r) => { if (!r) return; if (['PROCEDURE', 'BEGIN', 'DO', 'SELECT'].includes(r.kind)) opened++; if (r.kind === 'END') ended++; for (const u of r.node?.units || []) count(u); };
  for (const st of statements) {
    const r = parseStatement(st);
    count(r);
    if (r.status !== 'parsed') { bad++; console.log(f + ':' + st.line + ' ' + r.kind + ' does not parse (' + (r.reason || r.status) + '): ' + (lines[st.line - 1] || '').trim()); }
    else if (r.kind === 'PROCEDURE' && !(st.labels || []).length) { bad++; console.log(f + ':' + st.line + ' a PROCEDURE statement needs its entry name as a label, NAME: PROCEDURE ...: ' + (lines[st.line - 1] || '').trim()); }
    else if (r.kind === 'DECLARE' && (r.node.unknownAttributes || []).length) { bad++; console.log(f + ':' + st.line + ' DECLARE gives attributes PL/I does not have (' + r.node.unknownAttributes.join(', ') + '): ' + (lines[st.line - 1] || '').trim()); }
  }
  if (!statements.some((st) => parseStatement(st).kind === 'PROCEDURE')) { bad++; console.log(f + ': no PROCEDURE statement; a fixture is a program, NAME: PROCEDURE OPTIONS(MAIN); ... END NAME;'); }
  if (opened !== ended) { bad++; console.log(f + ': ' + opened + ' PROCEDURE, BEGIN, DO and SELECT statements but ' + ended + ' END statements; every one of them closes with exactly one END'); }
}
process.exit(bad ? 1 : 0);
`;

const MEASURE = `
const [dev, held, syn, script] = process.argv.slice(1);
const { measure } = await import(script);
const out = {};
for (const [name, dir] of [['dev', dev], ['heldout', held]]) {
  const statusMap = {};
  const m = measure(dir, { synthetic: syn.split(','), statusMap, perKind: 25 });
  out[name] = { summary: m.summary, byKind: m.byKind, statusMap, samples: name === 'dev' ? m.samples : undefined };
}
process.stdout.write(JSON.stringify(out));
`;

function measureIn(dir, t) {
  const r = run(process.execPath, ['--input-type=module', '-e', MEASURE, join(t.corpus, 'dev'), join(t.corpus, 'heldout'), t.synthetic.join(','), t.measure], { cwd: dir, timeout: 600000 });
  if (r.code !== 0) return { error: r.out.slice(-4000) };
  return JSON.parse(r.out);
}

function compare(before, after, kinds) {
  const regressions = [];
  let gained = 0;
  for (const split of ['dev', 'heldout']) {
    for (const [k, st] of Object.entries(before[split].statusMap)) {
      const now2 = after[split].statusMap[k];
      if (st === 'parsed' && now2 !== 'parsed') regressions.push(`${split} ${k} parsed -> ${now2}`);
      if (st !== 'parsed' && now2 === 'parsed') gained++;
    }
  }
  const kindDelta = {};
  for (const k of new Set([...Object.keys(before.dev.byKind), ...Object.keys(after.dev.byKind)])) {
    const d = (after.dev.byKind[k]?.parsed || 0) - (before.dev.byKind[k]?.parsed || 0);
    if (d) kindDelta[k] = d;
  }
  const s = (m, split, part) => m[split].summary[part];
  return {
    regressions: regressions.length, regressionSample: regressions.slice(0, 15), gained,
    crashesBefore: s(before, 'dev', 'all').crashes + s(before, 'heldout', 'all').crashes,
    crashesAfter: s(after, 'dev', 'all').crashes + s(after, 'heldout', 'all').crashes,
    devParsed: [s(before, 'dev', 'all').parsed, s(after, 'dev', 'all').parsed],
    devNaturalParsed: [s(before, 'dev', 'natural').parsed, s(after, 'dev', 'natural').parsed],
    devNaturalRate: [s(before, 'dev', 'natural').rate, s(after, 'dev', 'natural').rate],
    heldoutParsed: [s(before, 'heldout', 'all').parsed, s(after, 'heldout', 'all').parsed],
    heldoutRate: [s(before, 'heldout', 'all').rate, s(after, 'heldout', 'all').rate],
    kindDelta,
    targetGain: (kinds || []).reduce((n, k) => n + (kindDelta[k] || 0), 0),
  };
}

function gate(job, files, outDir) {
  const dir = join('/tmp/pli-gate', `${job.id}-${process.pid}`);
  rmSync(dir, { recursive: true, force: true });
  const track = trackOf(job);
  git(track, 'worktree', 'add', '--detach', '--force', dir, track.branch);
  const report = [];
  const result = { ok: false, checks: {} };
  try {
    const before = measureIn(dir, track);
    if (before.error) throw new Error(`baseline measure failed: ${before.error}`);
    writeFileSync(samplesFile(track), JSON.stringify({ head: headSha(track), samples: before.dev.samples }));
    const writable = job.writable || job.target || [];
    const refused = files.filter((f) => !track.allowed.test(f.path) || f.path.includes('..') || (writable.length && !writable.includes(f.path) && !(job.writableDirs || []).some((d) => f.path.startsWith(d))));
    if (refused.length) report.push(`refused: this task may write only ${[...writable, ...(job.writableDirs || [])].join(', ')}; you wrote ${refused.map((f) => f.path).join(', ')}`);
    const kept = files.filter((f) => !refused.includes(f));
    for (const f of kept) { mkdirSync(dirname(join(dir, f.path)), { recursive: true }); writeFileSync(join(dir, f.path), f.body); }
    const missing = (job.target || []).filter((t) => !kept.some((f) => f.path === t));
    result.checks.missingTargets = missing;
    if (missing.length) report.push(`the reply did not contain these files: ${missing.join(', ')}`);
    const syntax = [];
    for (const f of kept.filter((x) => x.path.endsWith('.mjs'))) {
      const r = run(process.execPath, ['--check', f.path], { cwd: dir, timeout: 30000 });
      if (r.code !== 0) syntax.push(`${f.path}:\n${r.out.trim().split('\n').slice(0, 8).join('\n')}`);
    }
    result.checks.syntax = syntax.length === 0;
    // A benchmark pair is run through the bench too; its verdict is recorded for review, not gated,
    // since a miss there is the reader's to fix and not the fixture's.
    if ((job.writableDirs || []).some((d) => d.startsWith('bench/pli/'))) {
      const dirs = [...new Set(kept.map((f) => f.path.split('/').slice(0, 3).join('/')))].filter((d) => existsSync(join(dir, d, 'manifest.json')));
      const b = run(process.execPath, ['--input-type=module', '-e', `import { runCase } from './diag/pli-bench.mjs'; console.log(JSON.stringify(process.argv.slice(1).map((d) => runCase(d))));`, ...dirs], { cwd: dir, timeout: 120000 });
      try { result.checks.bench = JSON.parse(b.out.trim().split('\n').pop()).map((r) => `${r.id}: ${r.verdict}`); } catch { result.checks.bench = [b.out.slice(-300)]; }
    }
    // An HLASM pair is assembled by z390 and scored by the bench; it passes only when both hold.
    const cases = [...new Set(kept.map((f) => /^bench\/cases\/(hlasm-[a-z0-9-]+)\//.exec(f.path)?.[1]).filter(Boolean))];
    if (cases.length) {
      const b = run(process.execPath, ['bench/run.mjs', '--json', '--validate', '--only', cases.join(',')], { cwd: dir, timeout: 300000, env: { ...process.env, Z390: PATHS.Z390_HOME } });
      let res = null;
      try { res = JSON.parse(b.out.slice(b.out.indexOf('{'))); } catch { report.push(`bench did not run:\n${b.out.slice(-1200)}`); }
      if (res) {
        result.checks.bench = res.results.map((r) => `${r.id}: ${r.compiles === true ? 'assembles' : r.compiles}; matched ${r.matched}, missing ${JSON.stringify(r.missing)}, unexpected ${JSON.stringify(r.extra)}`);
        for (const r of res.results) {
          if (r.compiles !== true) report.push(`${r.id} is not valid HLASM by z390: ${r.compiles}`);
          else if (r.missing.length || r.extra.length) report.push(`${r.id} does not score as its manifest declares: missing ${JSON.stringify(r.missing)}, unexpected ${JSON.stringify(r.extra)}`);
        }
        const missingCases = cases.filter((c) => !res.results.some((r) => r.id === c));
        if (missingCases.length) report.push(`no manifest.json for ${missingCases.join(', ')}`);
      }
    }
    // A locator fixture must assemble cleanly under z390 with no macro calls; how the locator grades
    // on it is recorded for review, since a difference there is the locator's to fix.
    const fixtures = kept.filter((f) => /^bench\/hlasm-locate\/[a-z0-9-]+\.asm$/.test(f.path));
    if (fixtures.length) {
      const tmp = mkdtempSync(join(tmpdir(), 'cw-fix-'));
      mkdirSync(join(tmp, 'fix', 'case'), { recursive: true });
      mkdirSync(join(tmp, 'members', 'case'), { recursive: true });
      for (const f of fixtures) writeFileSync(join(tmp, 'fix', 'case', f.path.split('/').pop()), f.body);
      const env = { Z390: PATHS.Z390_HOME };
      const o = run(process.execPath, ['diag/hlasm-oracle.mjs', tmp, join(tmp, 'oracle'), '--split', 'fix', '--force'], { cwd: dir, env, timeout: 300000 });
      const oracles = existsSync(join(tmp, 'oracle')) ? readdirSync(join(tmp, 'oracle')).map((f) => readJson(join(tmp, 'oracle', f))).filter(Boolean) : [];
      if (!oracles.length) report.push(`z390 did not produce an oracle for the fixture:\n${o.out.slice(-1500)}`);
      for (const r of oracles) {
        if (!r.graded) report.push(`${r.file} does not assemble cleanly under z390: ${r.why}; errors ${JSON.stringify((r.errors || []).slice(0, 5))}`);
        else if ((r.macroCalls || []).length) report.push(`${r.file} calls macros at lines ${r.macroCalls.join(', ')}; a locator fixture uses none`);
        else if (r.symbols.filter((y) => y.stable && y.line != null).length < 5) report.push(`${r.file} defines fewer than five labelled symbols z390 can grade`);
      }
      const g = run(process.execPath, ['--input-type=module', '-e', "import { measure } from './diag/hlasm-grade-z390.mjs'; console.log(JSON.stringify(measure(process.argv[1], { oracle: process.argv[2] })))", join(tmp, 'fix'), join(tmp, 'oracle')], { cwd: dir, timeout: 120000 });
      try {
        const m = JSON.parse(g.out.trim().split('\n').pop());
        result.checks.locate = Object.fromEntries(Object.entries(m.byKind).map(([k, v]) => [k, `agree ${v.parsed}, differ ${v.unparsed}, unplaced ${v.unbuilt}`]));
        result.checks.locateSamples = m.samples;
      } catch { result.checks.locate = g.out.slice(-500); }
      rmSync(tmp, { recursive: true, force: true });
    }
    const pliFiles = kept.filter((x) => /\.(pli|inc)$/i.test(x.path));
    if (pliFiles.length) {
      const r = run(process.execPath, ['--input-type=module', '-e', PLI_CHECK, ...pliFiles.map((f) => f.path)], { cwd: dir, timeout: 60000 });
      result.checks.pli = r.out.trim();
      if (r.code !== 0) report.push(`your PL/I source does not read cleanly with lib/pli:\n${r.out.trim().split('\n').slice(0, 30).join('\n')}`);
    }
    if (syntax.length) report.push(`node --check failed:\n${syntax.join('\n')}`);
    const testFiles = readdirSync(join(dir, 'test')).filter((f) => f.startsWith(`${track.name}-`) && f.endsWith('.test.mjs')).map((f) => `test/${f}`);
    const t = run(process.execPath, ['--test', '--test-concurrency=1', ...testFiles], { cwd: dir, timeout: 300000 });
    const num = (re) => Number((re.exec(t.out) || [])[1] || 0);
    result.checks.tests = { pass: num(/ℹ pass (\d+)/), fail: num(/ℹ fail (\d+)/), timedOut: !!t.timedOut };
    if (t.code !== 0 || t.timedOut) {
      const failing = failureLines(t.out, dir);
      report.push(`node --test failed (${result.checks.tests.fail} failing${t.timedOut ? ', timed out' : ''}):\n${failing}`);
    }
    const after = measureIn(dir, track);
    if (after.error) { report.push(`corpus measure failed after your change:\n${after.error}`); result.checks.measure = null; }
    else {
      const cmp = compare(before, after, job.kinds);
      result.checks.measure = cmp;
      if (cmp.regressions) report.push(`${cmp.regressions} corpus statements parsed before your change are not parsed after it:\n${cmp.regressionSample.join('\n')}`);
      if (cmp.crashesAfter > cmp.crashesBefore) report.push(`your parser threw a non-PliSyntax error on ${cmp.crashesAfter - cmp.crashesBefore} corpus statements (a crash, not a refusal)`);
      // A kind the corpus never writes (MODESET, say) cannot gain; its tests are the only evidence.
      const present = (job.kinds || []).some((k) => (before.dev.byKind[k]?.statements || 0) > 0);
      if (present && cmp.targetGain <= 0) report.push(`no corpus statement of kind ${job.kinds.join('/')} parses that did not before (kind deltas ${JSON.stringify(cmp.kindDelta)})`);
      if (!cmp.regressions) {
        const stillFailing = Object.entries(after.dev.samples || {}).filter(([k]) => (job.kinds || []).includes(k));
        for (const [k, list] of stillFailing) report.push(`still not parsed, kind ${k} (first ${Math.min(8, list.length)}):\n${list.slice(0, 8).map((s) => `${s.file}:${s.line} ${s.status} ${s.reason || ''}\n  ${s.text.split('\n')[0].trim()}`).join('\n')}`);
      }
    }
    const blocking = report.filter((r) => !r.startsWith('still not parsed'));
    result.ok = blocking.length === 0;
  } catch (e) {
    report.push(`gate error: ${e.stack || e}`);
  } finally {
    try { git(track, 'worktree', 'remove', '--force', dir); } catch { rmSync(dir, { recursive: true, force: true }); }
  }
  writeFileSync(join(outDir, 'gate.txt'), report.join('\n\n') || 'all gates passed');
  return result;
}

// Groq's free tier admits a request only when its prompt and max_tokens fit the minute's 8K tokens;
// spine's runner clamps max_tokens and waits out 429s. gpt-oss reasons at low, qwen at none.
function draftGroq(job) {
  const model = job.model || 'qwen/qwen3.8-27b';
  const env = {
    ...process.env, SPINE_LLM_LEDGER: '/Users/portll/.spine/llm-runs.jsonl', SPINE_LLM_MAX_TOKENS: String(job.maxTokens || 4000),
    SPINE_LLM_RATE_WAIT_MS: '240000', SPINE_PROMPTS_DIR: PATHS.SPINE_PROMPTS_DIR, SPINE_LLM_TEMPERATURE: '0.2',
    VELD_API_URL: 'http://127.0.0.1:9',
  };
  return runPrompt({ tool: trackOf(job).template, context: context(job), provider: 'groq', model, reasoning: job.reasoning || (model.startsWith('openai/') ? 'low' : 'none'), env });
}

async function draft(job) {
  const host = HOSTS[HOST];
  if (host.provider === 'groq') return draftGroq(job);
  const key = execFileSync('security', ['find-generic-password', '-s', host.key, '-w'], { encoding: 'utf8' }).trim();
  const env = {
    ...process.env,
    SPINE_LLM_URL: host.url, SPINE_LLM_MODEL: 'qwen/qwen3.8-27b', SPINE_LLM_API_KEY: key,
    SPINE_LLM_TIMEOUT_MS: String(75 * 60 * 1000), SPINE_LLM_MAX_TOKENS: String(job.maxTokens || (/^(p-|f\d)/.test(job.id) ? 6000 : 16000)),
    SPINE_PROMPTS_DIR: PATHS.SPINE_PROMPTS_DIR, SPINE_LLM_TEMPERATURE: '0.2',
    // Drafts are not copied into veld: each is 30-60 KB of prompt and reply, and veld chunks every one.
    VELD_API_URL: 'http://127.0.0.1:9',
  };
  return runPrompt({ tool: trackOf(job).template, context: context(job), env, fetchImpl: httpFetch, reasoning: 'none' });
}

function queueRepair(job, reason) {
  const round = (job.round || 0) + 1;
  if (round > MAX_ROUNDS) return null;
  const base = job.repairOf ? job.id.replace(/-r\d+$/, '') : job.id;
  const id = `${base}-r${round}`;
  if (existsSync(join(ROOT, 'bank', `${id}.json`))) return null;
  if (['landed', 'dropped'].includes(stateOf(base).status)) return null;
  const repair = { ...job, id, type: 'repair', round, repairOf: job.id, priority: (job.priority || 0) + 5, estimateMin: Math.max(10, Math.round((job.estimateMin || 15) * 0.8)), reason };
  writeFileSync(join(ROOT, 'bank', `${id}.json`), JSON.stringify(repair, null, 1));
  return id;
}

let lastRefill = 0;
function refillIfLow(status) {
  if (status.readyWallMin >= LOW_BUFFER_MIN || Date.now() - lastRefill < 10 * 60 * 1000) return;
  lastRefill = Date.now();
  // Hill-climbs are not generated: 1 passed in about 97 runs. Benchmark pairs come from the flow rules.
  const shortMin = LOW_BUFFER_MIN + 60 - queue().status.readyWallMin;
  if (shortMin > 0 && existsSync(join(ROOT, 'gen-pairs.mjs'))) {
    const jobs = Math.min(20, Math.ceil((shortMin * status.lanes) / 5));
    const a = run(process.execPath, [join(ROOT, 'gen-pairs.mjs'), '--max', String(jobs)], { cwd: ROOT, timeout: 300000 });
    log(`gen-pairs.mjs --max ${jobs} exit ${a.code}: ${a.out.trim().split('\n').slice(-1)[0].slice(0, 80)}`);
  }
}

// A knowledge-base target: its attempts and gate run inside kbgen.mjs, so it lands or fails here
// with no repair round. veld going away leaves it pending for later.
async function runKb(job) {
  const t0 = Date.now();
  const host = HOSTS[HOST];
  const key = execFileSync('security', ['find-generic-password', '-s', host.key, '-w'], { encoding: 'utf8' }).trim();
  const row = { id: job.id, type: 'kb', round: 0, host: HOST, startedAt: new Date(t0).toISOString() };
  try {
    const { runKbJob } = await import(join(KB, 'kbgen.mjs'));
    const r = await runKbJob(job, { url: host.url, key });
    const sec = Math.round((Date.now() - t0) / 1000);
    // A target the source no longer has (a section renamed or removed) closes the job; gen-kb.mjs
    // writes jobs for the targets that replaced it.
    if (r.gone) { setState(job.id, { status: 'dropped', endedAt: now(), sec, review: { note: r.gone } }); log(`dropped ${job.id}: ${r.gone}`); return true; }
    setState(job.id, { status: r.accepted ? 'landed' : 'failed', endedAt: now(), sec, kb: r });
    appendFileSync(join(ROOT, 'ledger.jsonl'), JSON.stringify({ ...row, sec, status: r.accepted ? 'kb-accepted' : 'kb-rejected', attempts: r.attempts, completionTokens: r.tokens, rows: r.accepted }) + '\n');
    log(`${r.accepted ? 'accepted' : 'rejected'} ${job.id} in ${sec}s after ${r.attempts} attempt(s), ${r.accepted} row(s)`);
  } catch (e) {
    veldUp = { at: 0, ok: false };
    setState(job.id, { status: 'pending', host: null, lastError: String(e.message || e).slice(0, 300) });
    rmSync(join(ROOT, 'state', `${job.id}.lock`), { force: true });
    appendFileSync(join(ROOT, 'ledger.jsonl'), JSON.stringify({ ...row, sec: Math.round((Date.now() - t0) / 1000), status: 'error', error: String(e.message || e).slice(0, 300) }) + '\n');
    log(`error ${job.id}: ${e.message}`);
    await new Promise((r) => setTimeout(r, 30000));
  }
  return true;
}

async function once() {
  if (veldHealthy()) {
    const { flushPending } = await import(join(KB, 'kbgen.mjs'));
    const sent = await flushPending().catch(() => 0);
    if (sent) log(`sent ${sent} queued KB rows to veld`);
  }
  const { ready, status } = queue();
  writeStatus();
  refillIfLow(status);
  const pick = ready.find(({ j }) => mine(j) && take(j));
  if (!pick) return false;
  const job = pick.j;
  const outDir = join(ROOT, 'out', job.id);
  mkdirSync(outDir, { recursive: true });
  writeStatus({ current: job.id });
  log(`start ${job.id} (${job.type || 'draft'}, round ${job.round || 0})`);
  if (job.type === 'kb') return runKb(job);
  const t0 = Date.now();
  let res = null;
  let err = null;
  try { res = await draft(job); } catch (e) { err = e; }
  const sec = Math.round((Date.now() - t0) / 1000);
  const row = { id: job.id, type: job.type || 'draft', round: job.round || 0, host: HOST, startedAt: new Date(t0).toISOString(), sec };
  // A rate limit is Groq's bucket refilling, not the job failing: the job goes back unchanged and the
  // lane waits as long as Groq says, up to half an hour.
  if (err && /rate limited/i.test(String(err.message))) {
    const wait = Math.min(1800, Number((/wait (\d+)s/.exec(String(err.message)) || [])[1] || 300));
    setState(job.id, { status: 'pending', host: null, lastError: String(err.message).slice(0, 200) });
    rmSync(join(ROOT, 'state', `${job.id}.lock`), { force: true });
    log(`rate limited on ${job.id}; waiting ${wait}s`);
    await new Promise((r) => setTimeout(r, wait * 1000));
    return true;
  }
  if (err) {
    writeFileSync(join(outDir, 'error.txt'), String(err.stack || err));
    appendFileSync(join(ROOT, 'ledger.jsonl'), JSON.stringify({ ...row, status: 'error', error: String(err.message || err).slice(0, 300) }) + '\n');
    setState(job.id, { status: 'pending', host: null, lastError: String(err.message || err).slice(0, 300), errors: (stateOf(job.id).errors || 0) + 1 });
    rmSync(join(ROOT, 'state', `${job.id}.lock`), { force: true });
    if ((stateOf(job.id).errors || 0) >= 3) setState(job.id, { status: 'failed' });
    log(`error ${job.id} after ${sec}s: ${err.message}`);
    await new Promise((r) => setTimeout(r, 60000));
    return true;
  }
  writeFileSync(join(outDir, 'reply.md'), res.response || '');
  writeFileSync(join(outDir, 'meta.json'), JSON.stringify({ model: res.model, reasoning: res.reasoning, finish: res.finishReason, truncated: res.truncated, usage: res.usage, sec, warnings: res.warnings }, null, 1));
  // The template asks for an SPDX comment line; in HLASM a comment starts with * in column 1.
  const hlasmComments = (body) => body.replace(/^\/\/ ?(.*)$/gm, (_, t) => `* ${t}`.slice(0, 71));
  const files = extractFiles(res.response || '').map((f) => (/\.(pli|pl1|plx|inc)$/i.test(f.path) ? { ...f, body: withinPliMargins(f.body) } : /\.(asm|mlc|mac)$/i.test(f.path) ? { ...f, body: hlasmComments(f.body) } : f));
  writeFileSync(join(outDir, 'files.json'), JSON.stringify(files, null, 1));
  const g = gate(job, files, outDir);
  writeFileSync(join(outDir, 'eval.json'), JSON.stringify(g, null, 1));
  const verdict = g.ok ? 'passed' : 'failed';
  const repair = g.ok ? null : queueRepair(job, readFileSync(join(outDir, 'gate.txt'), 'utf8').slice(0, 400));
  setState(job.id, { status: verdict, endedAt: now(), sec, repairQueued: repair });
  appendFileSync(join(ROOT, 'ledger.jsonl'), JSON.stringify({
    ...row, status: verdict, finish: res.finishReason, truncated: res.truncated,
    promptTokens: res.usage?.prompt_tokens, completionTokens: res.usage?.completion_tokens,
    files: files.map((f) => f.path), tests: g.checks.tests, measure: g.checks.measure && { ...g.checks.measure, regressionSample: undefined }, repair,
  }) + '\n');
  log(`${verdict} ${job.id} in ${sec}s, ${res.usage?.completion_tokens} tokens${repair ? `, repair ${repair} queued` : ''}`);
  return true;
}

const STOP = join(ROOT, 'STOP');
const RESTART = join(ROOT, 'RESTART');
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  log(`runner up on ${HOST}, pid ${process.pid}`);
  for (;;) {
    if (existsSync(STOP) || existsSync(`${STOP}-${HOST}`)) { log('STOP file present; exiting'); break; }
    if (existsSync(`${RESTART}-${HOST}`)) { rmSync(`${RESTART}-${HOST}`, { force: true }); log('RESTART file present; exiting for the supervisor to restart'); break; }
    let worked = false;
    try { worked = await once(); } catch (e) { log(`loop error: ${e.stack || e}`); }
    if (args.includes('--once')) break;
    if (!worked) await new Promise((r) => setTimeout(r, 60000));
  }
}
