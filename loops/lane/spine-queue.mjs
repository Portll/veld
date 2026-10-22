// veld's lane jobs in spine's queue. Producers keep writing bank/<id>.json and the runner keeps its
// state files; each pass, every pending job spine has not seen is filed as a task in
// lane-veld-<track> (route 'veld', so spine's own workers leave it alone), a job is claimed through
// spine's conditional update so two runners never take the same one, and each status the runner
// records is mirrored onto its task. Jobs finished before this stay in the files only.
import { join } from 'node:path';
import { hostname } from 'node:os';
import * as PATHS from '../paths.mjs';

const OPEN = new Set(['pending', 'active']);
export const planFor = (job) => `lane-veld-${job.track || 'pli'}`;

/** A queue over the spine store at `path` (the default store when omitted). */
export async function spineQueue({ path, spineRepo = PATHS.SPINE_REPO } = {}) {
  const store = await import(join(spineRepo, 'db.mjs'));
  const lane = await import(join(spineRepo, 'lane', 'bank.mjs'));
  const db = path ? store.open(path) : store.open();
  let index = new Map();

  const reindex = () => {
    index = new Map();
    for (const planId of lane.lanePlans(db).filter((p) => p.startsWith('lane-veld-'))) {
      for (const t of store.listTasks(db, planId)) if (t.state?.job?.key) index.set(t.state.job.key, { planId, id: t.id, status: t.status });
    }
    return index;
  };

  /** Files every pending job not yet in spine; returns how many were filed. */
  const sync = (jobs, stateOf) => {
    reindex();
    let filed = 0;
    for (const job of jobs) {
      if (index.has(job.id) || stateOf(job.id).status !== 'pending') continue;
      const planId = planFor(job);
      lane.ensurePlan(db, planId, `Lane: ${job.track || 'pli'} reader jobs`, { project: 'veld',
        description: `Jobs for the ${job.track || 'pli'} reader in cobolwork, banked by the loop producers and run by their runners; each holds a host slot, and its status here mirrors the runner's.` });
      const id = lane.bankJob(db, planId, { key: job.id, family: `veld-${job.type || 'draft'}`, title: job.title || job.id, route: 'veld', priority: job.priority || 0, round: job.round || 0 });
      if (id) { index.set(job.id, { planId, id, status: 'pending' }); filed++; }
    }
    return filed;
  };

  /** Takes the job's task from pending to active; true for a job spine does not track. */
  const claim = (jobId, host) => {
    const e = index.get(jobId);
    if (!e) return true;
    const won = db.prepare("UPDATE tasks SET status='active', updated_at=? WHERE plan_id=? AND id=? AND status='pending'").run(new Date().toISOString(), e.planId, e.id);
    if (!won.changes) return false;
    store.journalAppend(db, { kind: 'task.status', planId: e.planId, subject: e.id, detail: { from: 'pending', to: 'active', by: `veld ${host}` } });
    store.updateTask(db, e.planId, e.id, { statePatch: { worker: `veld ${host}`, claim: { worker: `veld ${host}`, pid: process.pid, host: hostname() } } });
    e.status = 'active';
    return true;
  };

  /** Records a runner status on the job's task: passed and landed complete it, failed abandons it, and
 *  dropped in review abandons it even after it passed the gate. */
  const mirror = (jobId, state) => {
    const e = index.get(jobId);
    if (!e) return;
    store.updateTask(db, e.planId, e.id, { statePatch: { veld: state } });
    const s = state.status;
    if (s === 'pending' && e.status !== 'pending') {
      db.prepare("UPDATE tasks SET status='pending', updated_at=? WHERE plan_id=? AND id=?").run(new Date().toISOString(), e.planId, e.id);
      e.status = 'pending';
    } else if ((s === 'passed' || s === 'landed') && OPEN.has(e.status)) {
      store.setStatus(db, e.planId, e.id, 'completed', { result: s === 'passed' ? 'gate passed; awaiting review' : 'landed' });
      e.status = 'completed';
    } else if ((s === 'failed' && OPEN.has(e.status)) || (s === 'dropped' && e.status !== 'abandoned')) {
      store.setStatus(db, e.planId, e.id, 'abandoned', { abandonReason: s === 'dropped' ? `dropped in review: ${state.review?.note ?? 'no note'}` : `gate failed${state.lastError ? `: ${String(state.lastError).slice(0, 200)}` : ''}` });
      e.status = 'abandoned';
    }
  };

  return { db, sync, claim, mirror, reindex };
}
