// Drive veld-tui to each screen, then render the captured buffers to PNG.
//
// Two stages on purpose: record.py owns the pty (Node cannot set a pty winsize without node-pty, and
// a 0x0 pty makes ratatui draw nothing), and this file owns the screen replay and rendering.

import { newScreen, feed, toHtml } from './capture.mjs';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const BIN = process.env.VELD_TUI_BIN || '/private/tmp/ext/tmp/veld-tui-shots/release/veld-tui';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUT = `${HERE}/out`;
if (!existsSync(BIN)) throw new Error(`no veld-tui at ${BIN}`);
mkdirSync(OUT, { recursive: true });

// Credentials come from the environment, then from veld's own config.toml — the same file the
// server and the TUI read. Never from a path specific to one machine, and never printed: an API key
// that reaches a child process's argv reaches that child's error messages too.
function fromConfigToml() {
  const candidates = [
    process.env.VELD_CONFIG,
    `${process.env.HOME}/Library/Application Support/veld/config.toml`,
    `${process.env.XDG_CONFIG_HOME || `${process.env.HOME}/.config`}/veld/config.toml`,
  ].filter(Boolean);
  for (const path of candidates) {
    if (!existsSync(path)) continue;
    const text = readFileSync(path, 'utf8');
    const pick = (key) => text.match(new RegExp(`^\\s*${key}\\s*=\\s*["']?([^"'\n]+)`, 'm'))?.[1]?.trim();
    const host = pick('host') || '127.0.0.1';
    const port = pick('port') || '3030';
    return { key: pick('api_key') || '', url: `http://${host}:${port}` };
  }
  return { key: '', url: '' };
}
const cfg = fromConfigToml();
const env = {
  VELD_API_URL: process.env.VELD_API_URL || cfg.url || 'http://127.0.0.1:3030',
  VELD_API_KEY: process.env.VELD_API_KEY || cfg.key,
  VELD_DEV_API_KEY: process.env.VELD_API_KEY || cfg.key,
  VELD_USER_ID: process.env.VELD_USER_ID || 'demo',
};
if (!env.VELD_API_KEY) throw new Error('no API key: set VELD_API_KEY or put api_key in veld config.toml');

// waitMs elapses BEFORE the step's keys are sent; a shot is taken at that instant.
const PROFILE = process.env.SHOT_PROFILE || 'demo';
const REPLAY = process.argv.includes('--replay');

// fetch, not curl: an argv header puts the API key in any error message the child throws, and that
// message goes wherever the caller's output goes. The key stays in this process.
async function profileRow() {
  const res = await fetch(`${env.VELD_API_URL}/api/users`, { headers: { 'X-API-Key': env.VELD_API_KEY } });
  if (!res.ok) throw new Error(`/api/users answered ${res.status}`);
  const users = await res.json();
  const i = users.indexOf(PROFILE);
  if (i < 0) throw new Error(`profile ${PROFILE} is not among the ${users.length} the server lists`);
  console.log(`selecting ${PROFILE}: row ${i} of ${users.length}`);
  return i;
}

// A replay must not need the server: the row count is read back from the recording's own sidecar.
const rowFile = `${OUT}/profile-row.json`;
const idx = REPLAY
  ? JSON.parse(readFileSync(rowFile, 'utf8')).row
  : await profileRow();
if (!REPLAY) writeFileSync(rowFile, JSON.stringify({ profile: PROFILE, row: idx }) + '\n');

const plan = [
  { waitMs: 1300, shot: 'splash' },
  // The splash runs about three seconds and swallows a keypress, so the auth screen needs its own
  // Enter afterwards: a single Enter went to the splash and every later shot was the auth screen.
  { waitMs: 2700, keys: '\r' },
  { waitMs: 1200, shot: 'auth' },
  { waitMs: 100, keys: '\r' },
  { waitMs: 2500, keys: 'j'.repeat(idx) },
  { waitMs: 900, shot: 'profiles' },
  { waitMs: 100, keys: '\r' },
  { waitMs: 3500, shot: 'dashboard' },
  { waitMs: 300, keys: '2' },
  { waitMs: 2200, shot: 'projects-todos' },
  { waitMs: 300, keys: '4' },
  { waitMs: 2400, shot: 'graph-map' },
  { waitMs: 300, keys: '3' },
  { waitMs: 2200, shot: 'activity' },
];

const rec = `${OUT}/stream.jsonl`;
// --replay re-renders the last recording. Rendering changes (fonts, the parser) should not need the
// app driven again, and a re-drive gives different data anyway, so a render fix cannot be compared.
if (REPLAY) {
  if (!existsSync(rec)) throw new Error(`no recording at ${rec} — run without --replay first`);
  console.log(`replaying ${rec}`);
} else {
  const cfg = JSON.stringify({ bin: BIN, out: rec, cols: 150, rows: 42, env, plan: plan.map(({ waitMs, keys }) => ({ waitMs, keys })) });
  const summary = JSON.parse(execFileSync('python3', [`${HERE}/record.py`, cfg], { encoding: 'utf8' }).trim());
  console.log(`recorded ${summary.chunks} chunks, ${summary.bytes} bytes`);
}

const chunks = readFileSync(rec, 'utf8').trim().split('\n').filter(Boolean)
  .map((l) => { const o = JSON.parse(l); return { at: o.at, data: Buffer.from(o.data, 'base64').toString('utf8') }; });

// Shot times are the cumulative plan offsets, matching how record.py paced itself.
let t = 0;
const marks = [];
for (const step of plan) { t += step.waitMs; if (step.shot) marks.push({ name: step.shot, at: t }); }

const INK = 24;  // a frame of nothing is a failed capture, not a screenshot
for (const mark of marks) {
  const s = newScreen();
  for (const c of chunks) if (c.at <= mark.at) feed(s, c.data);
  const ink = s.grid.flat().filter((cell) => cell.ch !== ' ').length;
  const html = `${OUT}/${mark.name}.html`;
  writeFileSync(html, toHtml(s));
  const png = `${OUT}/${mark.name}.png`;
  execFileSync(CHROME, ['--headless', '--disable-gpu', `--screenshot=${png}`, '--window-size=1240,700',
    '--hide-scrollbars', '--force-device-scale-factor=2', `file://${resolve(html)}`], { stdio: 'ignore' });
  console.log(`${mark.name.padEnd(16)} t+${String(mark.at).padStart(5)}ms  ink ${String(ink).padStart(5)} ${ink < INK ? 'BLANK — capture failed' : ''}`);
}
