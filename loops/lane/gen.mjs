// Refills the bank with hill-climb jobs: for each statement module that has landed, the corpus
// statements of its kinds that still do not parse, handed back with the module to improve.
//
//   node gen.mjs [--auto] [--max N] [--dry]
//
// --auto is how the runner calls it when the ready buffer drops under four hours. A module gets a
// new job only when no job for it is pending, running or awaiting review.
import * as PATHS from '../paths.mjs';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = PATHS.LANE_HOME;
const REPO = '/Users/portll/Repositories/Portll/cobolwork-pli-20261001';
const BRANCH = 'feat/pli';
const args = process.argv.slice(2);
const MAX = args.includes('--max') ? Number(args[args.indexOf('--max') + 1]) : 6;
const DRY = args.includes('--dry');

export const MODULES = {
  declare: ['DECLARE', 'DECLARE_FRAGMENT'],
  procedure: ['PROCEDURE', 'END', 'ENTRY', 'BEGIN', 'PACKAGE'],
  control: ['IF', 'DO', 'ELSE', 'WHEN', 'OTHERWISE', 'SELECT', 'RETURN', 'GOTO', 'LEAVE', 'ITERATE', 'STOP', 'EXIT'],
  call: ['CALL', 'FETCH', 'RELEASE'],
  assignment: ['ASSIGNMENT'],
  io: ['OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE', 'DELETE', 'LOCATE', 'UNLOCK'],
  stream: ['PUT', 'GET', 'DISPLAY', 'FORMAT', 'FLUSH'],
  conditions: ['ON', 'SIGNAL', 'REVERT', 'RESIGNAL'],
  alloc: ['ALLOCATE', 'FREE'],
  exec: ['EXEC'],
  preprocessor: ['PREPROCESSOR'],
  misc: ['DEFAULT', 'DEFINE', 'WAIT', 'DELAY', 'ATTACH', 'DETACH', 'ASSERT', 'QUALIFY'],
};

const readJson = (p, d = null) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return d; } };
const atHead = (file) => { try { return execFileSync('git', ['-C', REPO, 'show', `${BRANCH}:${file}`], { encoding: 'utf8' }); } catch { return null; } };
const OPEN = new Set(['pending', 'running', 'passed']);

const bank = readdirSync(join(ROOT, 'bank')).filter((f) => f.endsWith('.json')).map((f) => readJson(join(ROOT, 'bank', f))).filter(Boolean);
const stateOf = (id) => readJson(join(ROOT, 'state', `${id}.json`), { status: 'pending' });
// Samples are re-measured at the branch head when the runner's copy predates it.
function headSamples() {
  const head = execFileSync('git', ['-C', REPO, 'rev-parse', BRANCH], { encoding: 'utf8' }).trim();
  const cached = readJson(join(ROOT, 'metrics', 'head-samples.json'), {});
  if (cached.head === head) return cached.samples;
  const dir = `/tmp/pli-gate/gen-${process.pid}`;
  execFileSync('git', ['-C', REPO, 'worktree', 'add', '--detach', '--force', dir, BRANCH]);
  try {
    const out = join(ROOT, 'metrics', `gen-${process.pid}.json`);
    execFileSync(process.execPath, ['diag/pli-measure.mjs', join(ROOT, 'corpus', 'dev'), '--synthetic', 'jcf608_PacificNationalBank', '--per-kind', '25', '--samples', out], { cwd: dir, stdio: 'ignore' });
    const s = readJson(out, { samples: {} }).samples;
    writeFileSync(join(ROOT, 'metrics', 'head-samples.json'), JSON.stringify({ head, samples: s }));
    return s;
  } finally {
    execFileSync('git', ['-C', REPO, 'worktree', 'remove', '--force', dir]);
  }
}
const samples = headSamples();

const busy = (module) => bank.some((j) => (j.target || []).includes(`lib/pli/stmt/${module}.mjs`) && OPEN.has(stateOf(j.id).status));
const landed = (module) => { const body = atHead(`lib/pli/stmt/${module}.mjs`); return body != null && !/export const parsers = \{\};/.test(body); };

const candidates = Object.entries(MODULES)
  .filter(([m]) => landed(m) && !busy(m))
  .map(([m, kinds]) => ({ m, kinds, failing: kinds.flatMap((k) => (samples[k] || []).filter((s) => s.status !== 'parsed').map((s) => ({ ...s, kind: k }))) }))
  .filter((x) => x.failing.length)
  .sort((a, b) => b.failing.length - a.failing.length)
  .slice(0, MAX);

let wrote = 0;
for (const { m, kinds, failing } of candidates) {
  const n = bank.filter((j) => j.id.startsWith(`h-${m}-`) && !/-r\d+$/.test(j.id)).length + 1;
  const id = `h-${m}-${String(n).padStart(2, '0')}`;
  if (existsSync(join(ROOT, 'bank', `${id}.json`))) continue;
  const failingKinds = [...new Set(failing.map((s) => s.kind))];
  const job = {
    id, title: `Hill-climb ${m}: corpus statements of ${failingKinds.join(', ')} that do not parse`, type: 'hillclimb', priority: 50, estimateMin: 16,
    target: [`lib/pli/stmt/${m}.mjs`, `test/pli-${m}.test.mjs`],
    kinds,
    spec: `lib/pli/stmt/${m}.mjs (shown, with its tests) parses PL/I ${kinds.join(', ')} statements. The corpus statements below, of those kinds, still do not parse: each is shown with the reason the reader gave (unparsed: the parser refused or left tokens; unbuilt or unknown inside a nested unit means a statement of another kind is not parsed yet and is not your concern unless the unit is yours).\n\nChange the module so as many of them as possible parse correctly, without making any statement that parses today stop parsing, and without accepting text that is not PL/I. Add one test per new form to test/pli-${m}.test.mjs, keeping every existing test. Reply with both files in full.\n\nRules for parsers in this module: consume every token you accept; call c.fail on what you do not understand; nested statements go through ctx.subStatement and ctx.parseStatement into node.units; expressions through parseExpression and references through parseReference from '../expr.mjs'.`,
    context: [{ file: 'lib/pli/cursor.mjs' }, { file: 'lib/pli/expr.mjs' }, { file: `lib/pli/stmt/${m}.mjs` }, { file: `test/pli-${m}.test.mjs` }, ...failingKinds.slice(0, 4).map((k) => ({ samples: k, n: Math.max(6, Math.floor(24 / failingKinds.length)) }))],
  };
  if (!DRY) writeFileSync(join(ROOT, 'bank', `${id}.json`), JSON.stringify(job, null, 1));
  console.log(`${DRY ? 'would write' : 'wrote'} ${id}: ${failing.length} failing samples over ${failingKinds.join(', ')}`);
  wrote++;
}
console.log(`${wrote} hill-climb jobs`);
