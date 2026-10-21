// Self-evaluation loop state, where veld-desktop and `veld tui` read it: ~/.config/veld/loops/<id>.json
// is the loop's latest snapshot, <id>.events.jsonl every attempt and verdict it has made.
import { mkdirSync, writeFileSync, renameSync, appendFileSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

export const LOOPS_DIR = join(homedir(), '.config', 'veld', 'loops');

export function writeLoop(id, snapshot) {
  mkdirSync(LOOPS_DIR, { recursive: true });
  const path = join(LOOPS_DIR, `${id}.json`);
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, JSON.stringify({ id, updatedAt: new Date().toISOString(), ...snapshot }, null, 1));
  renameSync(tmp, path);
}

export function appendEvent(id, event) {
  mkdirSync(LOOPS_DIR, { recursive: true });
  appendFileSync(join(LOOPS_DIR, `${id}.events.jsonl`), JSON.stringify({ at: new Date().toISOString(), ...event }) + '\n');
}

export function readEvents(id) {
  const path = join(LOOPS_DIR, `${id}.events.jsonl`);
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8').split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}

// Hourly points of a running tally, for the trajectory the views draw.
export function trajectory(events, keyOf) {
  const points = new Map();
  for (const e of events) {
    const hour = e.at.slice(0, 13) + ':00Z';
    const p = points.get(hour) || {};
    const k = keyOf(e);
    if (k) p[k] = (p[k] || 0) + 1;
    points.set(hour, p);
  }
  return [...points.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).slice(-48).map(([at, counts]) => ({ at, ...counts }));
}

export function tally(items, keyOf, top = 8) {
  const m = new Map();
  for (const x of items) { const k = keyOf(x); if (k) m.set(k, (m.get(k) || 0) + 1); }
  return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, top);
}
