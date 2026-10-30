// veld's lane jobs in spine's queue: pending jobs are filed once, a claim goes through spine so a
// second runner cannot take the same job, statuses are mirrored, spine's own workers skip them, and a
// job whose runner died goes back to pending on a host with veld lanes and no spine lane.
//   SPINE_REPO=/path/to/spine node --test loops/lane/spine-queue.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as PATHS from '../paths.mjs';
import { spineQueue } from './spine-queue.mjs';

const fresh = () => spineQueue({ path: join(mkdtempSync(join(tmpdir(), 'veld-spine-queue-')), 'tasks.db') });
const DEAD = Number(spawnSync('/bin/sh', ['-c', 'echo $$'], { encoding: 'utf8' }).stdout.trim());

test('only pending jobs spine has not seen are filed, each once, in their track plan', async () => {
  const q = await fresh();
  const states = { a: 'pending', b: 'landed', c: 'pending' };
  const jobs = [{ id: 'a', title: 'parser A' }, { id: 'b' }, { id: 'c', track: 'hlasm', priority: 3 }];
  assert.equal(q.sync(jobs, (id) => ({ status: states[id] })), 2);
  assert.equal(q.sync(jobs, (id) => ({ status: states[id] })), 0);
  const { listTasks } = await import(join(PATHS.SPINE_REPO, 'db.mjs'));
  assert.deepEqual(listTasks(q.db, 'lane-veld-pli').map((t) => [t.state.job.key, t.state.job.route, t.goal]), [['a', 'veld', 'parser A']]);
  assert.equal(listTasks(q.db, 'lane-veld-hlasm')[0].state.job.priority, 3);
});

test('a job is claimed once through spine, and spine\'s own workers never take it', async () => {
  const q = await fresh();
  q.sync([{ id: 'a' }], () => ({ status: 'pending' }));
  const { claimNext } = await import(join(PATHS.SPINE_REPO, 'lane', 'bank.mjs'));
  assert.equal(claimNext(q.db, ['lane-veld-pli'], 'spine-worker', { kind: 'local' }), null);
  assert.equal(q.claim('a', 'm4'), true);
  assert.equal(q.claim('a', 'm4-b'), false, 'a second runner loses the claim');
  assert.equal(q.claim('not-in-spine', 'm4'), true, 'a job spine does not track is left to the file lock');
});

test('runner statuses are mirrored: passed completes, dropped in review abandons, failed abandons, pending reopens', async () => {
  const q = await fresh();
  q.sync([{ id: 'a' }, { id: 'b' }, { id: 'c' }], () => ({ status: 'pending' }));
  for (const id of ['a', 'b', 'c']) q.claim(id, 'm4');
  q.mirror('a', { status: 'passed' });
  q.mirror('a', { status: 'dropped', review: { note: 'duplicates lib/pli/x' } });
  q.mirror('b', { status: 'failed', lastError: 'tests fail' });
  q.mirror('c', { status: 'pending', lastError: 'rate limited' });
  const { getTask } = await import(join(PATHS.SPINE_REPO, 'db.mjs'));
  const t = (id) => getTask(q.db, 'lane-veld-pli', id);
  assert.deepEqual([t('1').status, t('1').abandon_reason], ['abandoned', 'dropped in review: duplicates lib/pli/x']);
  assert.deepEqual([t('2').status, t('2').abandon_reason], ['abandoned', 'gate failed: tests fail']);
  assert.deepEqual([t('3').status, t('3').state.veld.lastError], ['pending', 'rate limited']);
});

// A machine whose LaunchAgents run veld lanes and no spine lane, with one job claimed by a runner that
// died and one claimed by a live runner.
async function veldOnlyHost() {
  const root = mkdtempSync(join(tmpdir(), 'veld-only-host-'));
  const home = join(root, 'pli-qwen');
  const agents = join(root, 'agents');
  for (const d of [agents, join(home, 'state')]) mkdirSync(d, { recursive: true });
  writeFileSync(join(agents, 'net.example.veld-m4.plist'), `<plist><dict><key>EnvironmentVariables</key><dict><key>PLI_QWEN_HOME</key><string>${home}</string></dict>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>/r/veld/loops/lane/run.sh</string><string>m4</string></array></dict></plist>`);
  const store = join(root, 'tasks.db');
  const q = await spineQueue({ path: store });
  const { updateTask, getTask } = await import(join(PATHS.SPINE_REPO, 'db.mjs'));
  q.sync([{ id: 'p-dead' }, { id: 'p-live' }], () => ({ status: 'pending' }));
  for (const id of ['p-dead', 'p-live']) {
    q.claim(id, 'm4');
    writeFileSync(join(home, 'state', `${id}.json`), JSON.stringify({ status: 'running', host: 'm4' }));
    writeFileSync(join(home, 'state', `${id}.lock`), '');
  }
  const task = (id) => getTask(q.db, 'lane-veld-pli', q.reindex().get(id).id);
  updateTask(q.db, 'lane-veld-pli', task('p-dead').id, { statePatch: { claim: { ...task('p-dead').state.claim, pid: DEAD } } });
  return {
    root, home, agents, store, q,
    job: (id) => [task(id).status, JSON.parse(readFileSync(join(home, 'state', `${id}.json`), 'utf8')).status, existsSync(join(home, 'state', `${id}.lock`))],
  };
}

test('the sweep returns a job whose runner died, logs it, leaves an in-flight job alone and signals nothing', async () => {
  const h = await veldOnlyHost();
  const log = [];
  assert.deepEqual(h.q.sweep((m) => log.push(m), { agents: h.agents }), ['p-dead']);
  assert.deepEqual(h.job('p-dead'), ['pending', 'pending', false]);
  assert.deepEqual(h.job('p-live'), ['active', 'running', true]);
  assert.deepEqual(log, ['net.example.veld-m4: requeued veld job p-dead; its runner had died']);
  assert.ok(!existsSync(join(h.home, 'RESTART-m4')), 'the sweep wrote a restart signal');
  assert.deepEqual(h.q.sweep((m) => log.push(m), { agents: h.agents }), [], 'a second sweep finds nothing');
});

test('a veld runner starting on a host with no spine lane returns a stranded job with no deploy', async () => {
  const h = await veldOnlyHost();
  writeFileSync(join(h.home, 'STOP'), '');
  const runner = join(dirname(fileURLToPath(import.meta.url)), 'runner.mjs');
  const r = spawnSync(process.execPath, [runner, '--host', 'm4'], {
    env: { PATH: process.env.PATH, HOME: h.root, SPINE_REPO: PATHS.SPINE_REPO, SPINE_TASKS_DB: h.store, SPINE_LAUNCH_AGENTS: h.agents,
      PLI_QWEN_HOME: h.home, VELD_KB_HOME: join(h.root, 'kb'), LANG_QWEN_HOME: join(h.root, 'lang'), SPINE_PROMPTS_DIR: join(h.root, 'prompts') },
    encoding: 'utf8', timeout: 30000,
  });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const log = readFileSync(join(h.home, 'runner-m4.log'), 'utf8');
  assert.match(log, /net\.example\.veld-m4: requeued veld job p-dead; its runner had died/);
  assert.doesNotMatch(log, /p-live|veld sweep failed|has no veld sweep/);
  assert.deepEqual([h.job('p-dead')[0], h.job('p-live')[0]], ['pending', 'active']);
});
