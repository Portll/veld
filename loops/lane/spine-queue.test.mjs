// veld's lane jobs in spine's queue: pending jobs are filed once, a claim goes through spine so a
// second runner cannot take the same job, statuses are mirrored, and spine's own workers skip them.
//   SPINE_REPO=/path/to/spine node --test loops/lane/spine-queue.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as PATHS from '../paths.mjs';
import { spineQueue } from './spine-queue.mjs';

const fresh = () => spineQueue({ path: join(mkdtempSync(join(tmpdir(), 'veld-spine-queue-')), 'tasks.db') });

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
