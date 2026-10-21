// The main session's side of the lane: list drafts awaiting review, show one, copy its files into
// the feat/pli worktree, and record the verdict in the ledger.
//
//   node review.mjs list
//   node review.mjs show <id>
//   node review.mjs apply <id>
//   node review.mjs verdict <id> landed|dropped --accepted all|part|none --min <correction minutes> [--commit sha] [--note text]
import * as PATHS from '../paths.mjs';
import { readFileSync, writeFileSync, readdirSync, mkdirSync, appendFileSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = PATHS.LANE_HOME;
const REPOS = { pli: '/Users/portll/Repositories/Portll/cobolwork-pli-20261001', ims: '/Users/portll/Repositories/Portll/cobolwork-lang-20261002', db2: '/Users/portll/Repositories/Portll/cobolwork-lang-20261002' };
const ALLOWED = { pli: /^(lib\/pli\/|test\/pli-|diag\/pli-|bench\/pli\/)/, ims: /^(lib\/ims\/|test\/ims-)/, db2: /^(lib\/db2\/|test\/db2-)/ };
const trackOf = (id) => { try { return JSON.parse(readFileSync(join(ROOT, 'bank', `${id}.json`), 'utf8')).track || 'pli'; } catch { return 'pli'; } };
const readJson = (p, d = null) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return d; } };
const stateOf = (id) => readJson(join(ROOT, 'state', `${id}.json`), { status: 'pending' });
const [cmd, id, ...rest] = process.argv.slice(2);
const flag = (n, d = null) => (rest.includes(`--${n}`) ? rest[rest.indexOf(`--${n}`) + 1] : d);

if (cmd === 'list') {
  const ids = readdirSync(join(ROOT, 'state')).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5));
  for (const j of ids.sort()) {
    const s = stateOf(j);
    if (s.status !== 'passed' && !rest.includes('--all')) continue;
    const e = readJson(join(ROOT, 'out', j, 'eval.json'), {});
    const m = e.checks?.measure;
    console.log(`${j.padEnd(26)} ${trackOf(j).padEnd(4)}${s.status.padEnd(8)} ${String(s.sec || '').padStart(5)}s tests ${e.checks?.tests?.pass ?? '-'}/${e.checks?.tests?.fail ?? '-'} gained ${m?.gained ?? '-'} kinds ${JSON.stringify(m?.kindDelta || {})} heldout ${m ? m.heldoutParsed.join('->') : '-'}${e.checks?.bench ? ` bench ${e.checks.bench.join('; ')}` : ''}`);
  }
} else if (cmd === 'show') {
  const out = join(ROOT, 'out', id);
  console.log(readFileSync(join(out, 'gate.txt'), 'utf8'));
  const reply = readFileSync(join(out, 'reply.md'), 'utf8');
  const limits = reply.indexOf('## Limits');
  console.log(limits >= 0 ? reply.slice(limits) : '(no Limits section)');
  for (const f of readJson(join(out, 'files.json'), [])) console.log(`--- ${f.path}: ${f.body.split('\n').length} lines`);
} else if (cmd === 'apply') {
  const track = trackOf(id);
  const repo = REPOS[track];
  for (const f of readJson(join(ROOT, 'out', id, 'files.json'), [])) {
    if (!ALLOWED[track].test(f.path)) { console.log(`skipped ${f.path}`); continue; }
    mkdirSync(dirname(join(repo, f.path)), { recursive: true });
    writeFileSync(join(repo, f.path), f.body);
    console.log(`wrote ${f.path}`);
  }
} else if (cmd === 'verdict') {
  const status = rest[0];
  if (!['landed', 'dropped'].includes(status)) throw new Error('verdict is landed or dropped');
  const s = stateOf(id);
  const review = { at: new Date().toISOString(), accepted: flag('accepted'), correctionMin: Number(flag('min', 0)), commit: flag('commit'), note: flag('note') };
  writeFileSync(join(ROOT, 'state', `${id}.json`), JSON.stringify({ ...s, status, review }, null, 1));
  rmSync(join(ROOT, 'state', `${id}.lock`), { force: true });
  appendFileSync(join(ROOT, 'ledger.jsonl'), JSON.stringify({ id, type: 'review', status, ...review }) + '\n');
  // A landed or dropped draft closes its repair chain: queued repairs of it are dropped too.
  const base = id.replace(/-r\d+$/, '');
  for (const f of readdirSync(join(ROOT, 'bank'))) {
    const other = f.slice(0, -5);
    if (other !== id && other.replace(/-r\d+$/, '') === base && ['pending', 'failed'].includes(stateOf(other).status)) {
      writeFileSync(join(ROOT, 'state', `${other}.json`), JSON.stringify({ ...stateOf(other), status: 'dropped', review: { note: `chain closed by ${id}` } }, null, 1));
    }
  }
  console.log(`${id} ${status}`);
} else {
  console.log('usage: list | show <id> | apply <id> | verdict <id> landed|dropped --accepted all|part|none --min N [--commit sha] [--note text]');
}
