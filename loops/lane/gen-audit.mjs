// Writes audit jobs: batches of corpus statements the reader parses, each with the node it built, for
// Qwen to check against PL/I, pin what is right as tests, and name what is wrong. Coverage counts
// tokens consumed; an audit asks whether they were consumed into the right fields.
//
//   node gen-audit.mjs [--per-batch 12] [--max-jobs 20] [--kinds A,B]
import * as PATHS from '../paths.mjs';
import { writeFileSync, existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = PATHS.LANE_HOME;
const REPO = '/Users/portll/Repositories/Portll/cobolwork-pli-20261001';
const args = process.argv.slice(2);
const flag = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d);
const PER = Number(flag('per-batch', 12));
const MAX = Number(flag('max-jobs', 20));
const ONLY = flag('kinds', null)?.split(',');

// Runs inside a worktree at the branch head: distinct parsed statements per kind, natural corpus only,
// each with a compact node (tokens shown by their text, expression trees kept).
const SAMPLER = `
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { readPli, sourceText } from './lib/pli/lex.mjs';
import { parseStatement } from './lib/pli/statements.mjs';
const [root, synthetic, per] = process.argv.slice(1);
const files = [];
const walk = (d) => { for (const e of readdirSync(d).sort()) { const p = join(d, e); if (statSync(p).isDirectory()) walk(p); else if (/\\.(pli|inc|pcx)$/i.test(e)) files.push(p); } };
walk(root);
const compact = (x, depth = 0) => {
  if (depth > 12 || x == null || typeof x !== 'object') return x;
  if (Array.isArray(x)) return x.map((y) => compact(y, depth + 1));
  if (x.t && 'line' in x && 'col' in x) return x.t === 'lit' ? "'" + x.v + "'" + (x.suffix || '') : x.v ?? x.t;
  const o = {};
  for (const [k, v] of Object.entries(x)) { if (k === 'toks' || k === 'line' || k === 'endLine' || k === 'file') continue; o[k] = compact(v, depth + 1); }
  return o;
};
const out = {};
const seen = new Set();
for (const f of files) {
  const rel = relative(root, f);
  if (rel.split('/')[0] === synthetic) continue;
  const text = sourceText(readFileSync(f));
  const lines = text.split(/\\r?\\n/);
  for (const st of readPli(text, { file: rel }).statements) {
    const r = parseStatement(st);
    if (r.status !== 'parsed') continue;
    const src = lines.slice(st.line - 1, Math.min(st.endLine, st.line + 5)).join('\\n');
    const shape = r.kind + ':' + src.trim().replace(/\\s+/g, ' ').replace(/'[^']*'/g, "'").replace(/\\b\\d+\\b/g, '9').slice(0, 40).toUpperCase();
    if (seen.has(shape)) continue;
    seen.add(shape);
    (out[r.kind] ||= []).push({ file: rel, line: st.line, src, node: compact(r.node) });
  }
}
process.stdout.write(JSON.stringify(out));
`;

const dir = `/tmp/pli-gate/audit-${process.pid}`;
execFileSync('git', ['-C', REPO, 'worktree', 'add', '--detach', '--force', dir, 'feat/pli'], { stdio: 'ignore' });
let samples;
try {
  samples = JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', SAMPLER, join(ROOT, 'corpus', 'dev'), 'jcf608_PacificNationalBank', String(PER)], { cwd: dir, encoding: 'utf8', maxBuffer: 256 << 20 }));
} finally {
  execFileSync('git', ['-C', REPO, 'worktree', 'remove', '--force', dir]);
}

const existing = new Set(readdirSync(join(ROOT, 'bank')).map((f) => f.replace(/\.json$/, '')));
const audited = new Set();
for (const id of existing) {
  const j = JSON.parse(readFileSync(join(ROOT, 'bank', `${id}.json`), 'utf8'));
  for (const k of j.auditKeys || []) audited.add(k);
}

let wrote = 0;
const kinds = Object.entries(samples).filter(([k]) => !ONLY || ONLY.includes(k)).sort((a, b) => b[1].length - a[1].length);
// Round-robin over kinds so a short queue still covers every kind.
for (let round = 0; wrote < MAX; round++) {
  let any = false;
  for (const [kind, list] of kinds) {
    if (wrote >= MAX) break;
    const fresh = list.filter((s) => !audited.has(`${s.file}:${s.line}`));
    const batch = fresh.slice(0, PER);
    if (!batch.length) continue;
    any = true;
    let n = 1;
    while (existing.has(`a-${kind.toLowerCase()}-${String(n).padStart(2, '0')}`)) n++;
    const id = `a-${kind.toLowerCase()}-${String(n).padStart(2, '0')}`;
    const test = `test/pli-audit-${kind.toLowerCase()}-${String(n).padStart(2, '0')}.test.mjs`;
    const job = {
      id, title: `Audit ${batch.length} parsed ${kind} statements from the corpus`, type: 'audit', priority: 35, estimateMin: 8,
      target: [test], kinds: [], auditKeys: batch.map((s) => `${s.file}:${s.line}`),
      spec: `The PL/I reader parses each corpus statement below and builds the node shown. Coverage only proves every token was consumed; your job is to check each node is RIGHT for IBM Enterprise PL/I: the right kind, every name in the right field, expressions grouped by PL/I precedence, options attributed to the right clause, nested units the right statements.\n\nFor each statement:\n- if the node is right, add a test to ${test} that parses the statement's source text and asserts the fields that carry its meaning (names, values, options; for an expression, refs or tree.op);\n- if the node is wrong, add no test for it and list it under Limits as \`WRONG file:line — what is wrong — what it should be\`.\n\nUse exactly this helper in the test file:\n\`\`\`js\nimport { test } from 'node:test';\nimport assert from 'node:assert/strict';\nimport { readPli } from '../lib/pli/lex.mjs';\nimport { parseStatement } from '../lib/pli/statements.mjs';\nconst parse = (src) => parseStatement(readPli(src).statements[0]);\n\`\`\`\nPass the source as a JS string exactly as shown (keep its line breaks with \\n; if it starts in column 1, prefix one space). In the node shown, a token appears as its text, a literal in quotes with its suffix, and 'toks', 'line' and 'file' fields are omitted; in the real node a reference is { t: 'ref', name, path, locators, args }, an expression { t: 'expr', tree, refs }, a number or literal operand { t: 'num'|'lit', tok } where tok is a token object { t, v, line, col }: compare tok.v (and tok.suffix), never the token object itself; an attribute's args are token arrays, compare args.map((t) => t.v).\n\n${batch.map((s, k) => `### ${k + 1}. ${s.file}:${s.line}\n\`\`\`pli\n${s.src}\n\`\`\`\nnode:\n\`\`\`json\n${JSON.stringify(s.node).slice(0, 2500)}\n\`\`\``).join('\n\n')}`,
      context: [],
      maxTokens: 12000,
    };
    writeFileSync(join(ROOT, 'bank', `${id}.json`), JSON.stringify(job, null, 1));
    existing.add(id);
    for (const s of batch) audited.add(`${s.file}:${s.line}`);
    wrote++;
  }
  if (!any) break;
}
console.log(`${wrote} audit jobs; parsed shapes available: ${Object.entries(samples).map(([k, v]) => `${k} ${v.length}`).join(', ')}`);
