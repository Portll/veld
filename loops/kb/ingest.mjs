// Puts knowledge bases' sources into veld and marks each done, which is what lets the runner start
// its targets.   node ingest.mjs [kbId ...]   (all when none named)
import * as PATHS from '../paths.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { KBS, ingest, writeSnapshot } from './kbgen.mjs';
import { health } from './veld.mjs';

const HERE = PATHS.KB_HOME;
const ids = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(KBS);
if (!(await health())) { console.error('veld is not answering /health on 127.0.0.1:3030'); process.exit(1); }
mkdirSync(join(HERE, 'ingested'), { recursive: true });
for (const id of ids) {
  const t0 = Date.now();
  const n = await ingest(id, { onProgress: (k, all) => console.log(`${id}: ${k} of ${all}`) });
  writeFileSync(join(HERE, 'ingested', `${id}.json`), JSON.stringify({ id, passages: n, at: new Date().toISOString(), sec: Math.round((Date.now() - t0) / 1000) }));
  writeSnapshot(id);
  console.log(`${id}: ${n} passages in ${Math.round((Date.now() - t0) / 1000)} s`);
}
