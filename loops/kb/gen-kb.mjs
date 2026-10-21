// Writes one runner job per knowledge-base target into the Qwen bank.   node gen-kb.mjs [kbId ...]
import * as PATHS from '../paths.mjs';
import { writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { KBS, writeSnapshot } from './kbgen.mjs';

const BANK = join(PATHS.LANE_HOME, 'bank');
const ids = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(KBS);
for (const kbId of ids) {
  let wrote = 0;
  // A job is keyed to its target, not its position: the source's sections move between runs.
  const jobs = readdirSync(BANK).filter((f) => f.startsWith(`k-${kbId}-`)).map((f) => JSON.parse(readFileSync(join(BANK, f), 'utf8')));
  const covered = new Set(jobs.map((j) => j.target));
  let next = jobs.reduce((m, j) => Math.max(m, Number(j.id.slice(`k-${kbId}-`.length)) || 0), -1) + 1;
  KBS[kbId].targets().forEach((t) => {
    if (covered.has(t.id)) return;
    const id = `k-${kbId}-${String(next++).padStart(3, '0')}`;
    writeFileSync(join(BANK, `${id}.json`), JSON.stringify({ id, type: 'kb', kb: kbId, target: t.id, title: `${kbId}: ${t.id}`, priority: kbId === 'cobolwork-utility' ? 53 : 44, estimateMin: 3 }, null, 1));
    wrote++;
  });
  writeSnapshot(kbId);
  console.log(`${kbId}: ${wrote} jobs`);
}
