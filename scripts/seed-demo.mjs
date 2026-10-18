#!/usr/bin/env node
// Build the `demo` profile: veld remembering veld.
//
// Everything is written through /api/remember/batch, which is the only path that lets this script
// set created_at. /api/seed stamps its rows at seed time and renders them as the file's own first
// line jammed against its path ("File: tests/x.test.tsimport { describe...") — honest, unreadable,
// and it buries the curated rows under a wall of source. It is still available behind --scan when
// the entity graph needs density.
//
// Rows come from two places: demo/seed.json pins the ones a screenshot shows, and the repository's
// own documentation supplies the rest. Both are already public, so the demo profile discloses
// nothing a reader could not read on GitHub.
//
//   node scripts/seed-demo.mjs              # seed, then report what the profile holds
//   node scripts/seed-demo.mjs --reset      # clear the profile first, so a re-seed is not a pile-up
//   node scripts/seed-demo.mjs --fixture    # pinned rows only, no rows derived from the docs
//   node scripts/seed-demo.mjs --scan       # ALSO call /api/seed for entity-graph density
//   node scripts/seed-demo.mjs --dry        # resolve config and print the plan
//
// Credentials: VELD_API_KEY / VELD_API_URL, else veld's own config.toml. Never passed to a child.

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PROFILE = process.env.VELD_DEMO_USER || 'demo';
const DRY = process.argv.includes('--dry');
const FIXTURE_ONLY = process.argv.includes('--fixture');
const RESET = process.argv.includes('--reset');

function config() {
  const candidates = [
    process.env.VELD_CONFIG,
    `${process.env.HOME}/Library/Application Support/veld/config.toml`,
    `${process.env.XDG_CONFIG_HOME || `${process.env.HOME}/.config`}/veld/config.toml`,
  ].filter(Boolean);
  for (const path of candidates) {
    if (!existsSync(path)) continue;
    const text = readFileSync(path, 'utf8');
    const pick = (k) => text.match(new RegExp(`^\\s*${k}\\s*=\\s*["']?([^"'\\n]+)`, 'm'))?.[1]?.trim();
    return { key: pick('api_key') || '', url: `http://${pick('host') || '127.0.0.1'}:${pick('port') || '3030'}` };
  }
  return { key: '', url: '' };
}
const cfg = config();
const URL_BASE = process.env.VELD_API_URL || cfg.url || 'http://127.0.0.1:3030';
const KEY = process.env.VELD_API_KEY || cfg.key;
if (!KEY) throw new Error('no API key: set VELD_API_KEY or put api_key in veld config.toml');

const call = async (path, body) => {
  const res = await fetch(`${URL_BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'X-API-Key': KEY },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${path} answered ${res.status}: ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : {};
};

const fixture = JSON.parse(readFileSync(`${ROOT}/demo/seed.json`, 'utf8'));

// Ages are relative to seed time, so a freshly seeded demo reads as a session that just happened
// rather than one from last January. Set VELD_DEMO_NOW (the fixture's `anchor` is the suggested
// value) when the point is a byte-identical reproduction instead of a plausible one.
const NOW = new Date(process.env.VELD_DEMO_NOW || Date.now());
const at = (minutesAgo) => new Date(NOW.getTime() - minutesAgo * 60_000).toISOString();

const memories = fixture.memories.map((m) => ({
  content: m.content,
  tags: m.tags ?? [],
  memory_type: m.type ?? 'Context',
  created_at: at(m.minutesAgo ?? 0),
}));

if (RESET && !DRY) {
  const cleared = await call('/api/memories/clear', { user_id: PROFILE, confirm: 'CONFIRM' });
  console.log(`  reset ${PROFILE}: ${JSON.stringify(cleared).slice(0, 120)}`);
}

console.log(`profile ${PROFILE} at ${URL_BASE}`);
console.log(`  fixture: ${memories.length} pinned memories, anchored ${NOW.toISOString()}`);
console.log(`  repo scan: ${FIXTURE_ONLY ? 'skipped (--fixture)' : ROOT}`);
if (DRY) { console.log('dry run: nothing written'); process.exit(0); }

// One call over the whole tree exceeds the server's own request timeout: 25 files take about 16
// seconds, so 400 answer 408. Seeding per subtree keeps each call well inside it and spreads the
// graph across areas instead of whichever files the walk happened to reach first.
const SUBTREES = (process.env.VELD_DEMO_SUBTREES || 'docs,src/handlers,mcp-server').split(',');
const PER_SUBTREE = Number(process.env.VELD_DEMO_MAX_FILES || 25);
const SCAN = process.argv.includes('--scan');

// A heading plus its first sentences reads as a note somebody wrote. The file's raw first line does
// not, which is the whole reason this script does not lean on /api/seed.
function rowsFromDocs() {
  const out = [];
  const docs = resolve(ROOT, 'docs');
  if (!existsSync(docs)) return out;
  const walk = (dir, depth = 0) => {
    if (depth > 2) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = resolve(dir, entry.name);
      if (entry.isDirectory()) { walk(path, depth + 1); continue; }
      if (!entry.name.endsWith('.md')) continue;
      const text = readFileSync(path, 'utf8');
      const rel = path.slice(ROOT.length + 1);
      let heading = null;
      for (const line of text.split('\n')) {
        const h = /^#{1,3}\s+(.+?)\s*$/.exec(line);
        if (h) { heading = h[1].replace(/[`*_]/g, ''); continue; }
        const prose = line.trim();
        if (!heading || prose.length < 80 || prose.startsWith('|') || prose.startsWith('<') || prose.startsWith('-')) continue;
        out.push({
          content: `${heading}. ${prose.replace(/[`*_]/g, '').slice(0, 320)}`,
          tags: ['docs', rel.split('/').slice(0, 2).join('/')],
          type: 'Learning',
        });
        heading = null;
        if (out.length >= Number(process.env.VELD_DEMO_DOC_ROWS || 60)) return;
      }
    }
  };
  walk(docs);
  return out;
}

const docRows = FIXTURE_ONLY ? [] : rowsFromDocs();
console.log(`  docs: ${docRows.length} rows derived from headings`);

// Curated rows are the newest, so they lead the activity feed; doc rows fill the hours behind them.
const allRows = [
  ...memories,
  ...docRows.map((r, i) => ({ ...r, memory_type: r.type, created_at: at(60 + i * 7) })),
].map(({ type, ...row }) => row);

if (SCAN) {
  let created = 0, scanned = 0;
  const failures = [];
  for (const sub of SUBTREES) {
    const path = resolve(ROOT, sub.trim());
    if (!existsSync(path)) { console.log(`  skipped ${sub}: not in this checkout`); continue; }
    try {
      const seeded = await call('/api/seed', { user_id: PROFILE, project_path: path, max_files: PER_SUBTREE });
      created += seeded.memories_created;
      scanned += seeded.files_scanned;
      console.log(`  ${sub}: ${seeded.memories_created} memories from ${seeded.files_scanned} files`);
    } catch (err) {
      failures.push(`${sub}: ${err.message}`);
      console.error(`  ${sub}: FAILED — ${err.message}`);
    }
  }
  console.log(`  scanned ${created} memories from ${scanned} files`);
  if (failures.length) process.exitCode = 5;
}

const batch = await call('/api/remember/batch', { user_id: PROFILE, memories: allRows });
// The response is {created, failed, memory_ids, errors}. An earlier version read keys the server
// never sends and fell back to the INPUT length, so it reported a full write whatever happened.
const created = batch.created ?? 0;
const failed = batch.failed ?? 0;
console.log(`  wrote ${created} memories (${memories.length} pinned + ${docRows.length} from docs)${failed ? `, ${failed} FAILED` : ''}`);
if (failed) {
  for (const e of (batch.errors || []).slice(0, 5)) console.error(`    ${JSON.stringify(e).slice(0, 200)}`);
  process.exitCode = 4;
}
if (created !== allRows.length) {
  console.error(`    ${allRows.length} rows were sent and the server created ${created}`);
  process.exitCode = 4;
}

// Report what the profile actually holds. A seeder that cannot read its own result is a seeder that
// cannot tell you it wrote nothing.
const stats = await fetch(`${URL_BASE}/api/stats?user_id=${encodeURIComponent(PROFILE)}`,
  { headers: { 'X-API-Key': KEY } }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
if (stats) console.log(`  profile now holds: ${JSON.stringify(stats).slice(0, 200)}`);
else console.log('  (no stats endpoint answer; check the TUI)');
