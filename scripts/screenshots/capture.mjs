// Capture real veld-tui screens. No tmux on this machine, so: run the binary under `script` (which
// gives it a pty, otherwise ratatui refuses to draw), record the ANSI stream with timestamps, then
// replay it into a screen buffer and snapshot that buffer at chosen moments.
//
// The buffer is the point. A raw ANSI dump is a stream of partial redraws; only a parsed screen is
// what a person would have seen.

import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';

const COLS = Number(process.env.SHOT_COLS || 150);
const ROWS = Number(process.env.SHOT_ROWS || 42);

export function newScreen() {
  const blank = () => ({ ch: ' ', fg: null, bg: null, bold: false, dim: false });
  return { grid: Array.from({ length: ROWS }, () => Array.from({ length: COLS }, blank)), x: 0, y: 0, fg: null, bg: null, bold: false, dim: false };
}

// Only the subset ratatui emits: absolute cursor moves, erases, SGR colour, and text.
export function feed(s, dataIn) {
  // A read() boundary can fall inside an escape sequence. Without carrying the tail, the ESC is
  // dropped and the remainder prints as literal text: that is where `2;15;15;20m` came from, drawn
  // across the splash as if it were content.
  const data = (s.pending || '') + dataIn;
  s.pending = '';
  let i = 0;
  const put = (ch) => {
    if (s.y >= 0 && s.y < ROWS && s.x >= 0 && s.x < COLS) {
      s.grid[s.y][s.x] = { ch, fg: s.fg, bg: s.bg, bold: s.bold, dim: s.dim };
    }
    s.x++;
    if (s.x >= COLS) { s.x = 0; s.y++; }
  };
  while (i < data.length) {
    const c = data[i];
    if (c === '\x1b') {
      const rest = data.slice(i);

      // OSC: ESC ] ... (BEL | ESC backslash). The TUI sets its window title this way, and a parser
      // that only knows CSI prints the payload as content — that is where `]0;Veld - Graph` came
      // from, drawn across the bottom row as if the program had written it there.
      if (rest[1] === ']') {
        const end = /\x07|\x1b\\/.exec(rest.slice(2));
        if (!end) { s.pending = rest; return s; }
        i += 2 + end.index + end[0].length;
        continue;
      }
      // Charset selection and keypad mode: two or three bytes, no payload to draw.
      if ('()*+'.includes(rest[1])) {
        if (rest.length < 3) { s.pending = rest; return s; }
        i += 3;
        continue;
      }
      if ('=>78'.includes(rest[1])) { i += 2; continue; }

      const m = /^\x1b\[([0-9;?]*)([A-Za-z])/.exec(rest);
      if (!m) {
        // Incomplete at the tail: hold it for the next chunk. Mid-stream junk is skipped.
        if (/^\x1b(\[[0-9;?]*)?$/.test(rest) || rest.length < 8) { s.pending = rest; return s; }
        i += 1;
        continue;
      }
      const [all, argsRaw, cmd] = m;
      const args = argsRaw.split(';').filter((x) => x !== '').map(Number);
      if (cmd === 'H' || cmd === 'f') { s.y = (args[0] ?? 1) - 1; s.x = (args[1] ?? 1) - 1; }
      else if (cmd === 'J') {
        const mode = args[0] ?? 0;
        for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
          const after = y > s.y || (y === s.y && x >= s.x);
          if (mode === 2 || (mode === 0 && after) || (mode === 1 && !after)) s.grid[y][x] = { ch: ' ', fg: null, bg: null, bold: false, dim: false };
        }
      } else if (cmd === 'K') {
        const mode = args[0] ?? 0;
        for (let x = 0; x < COLS; x++) {
          const after = x >= s.x;
          if (mode === 2 || (mode === 0 && after) || (mode === 1 && !after)) s.grid[s.y][x] = { ch: ' ', fg: null, bg: null, bold: false, dim: false };
        }
      } else if (cmd === 'm') {
        if (args.length === 0) { s.fg = null; s.bg = null; s.bold = false; s.dim = false; }
        for (let k = 0; k < args.length; k++) {
          const a = args[k];
          if (a === 0) { s.fg = null; s.bg = null; s.bold = false; s.dim = false; }
          else if (a === 1) s.bold = true;
          else if (a === 2) s.dim = true;
          else if (a === 22) { s.bold = false; s.dim = false; }
          else if (a === 39) s.fg = null;
          else if (a === 49) s.bg = null;
          else if (a === 38 && args[k + 1] === 2) { s.fg = [args[k + 2], args[k + 3], args[k + 4]]; k += 4; }
          else if (a === 48 && args[k + 1] === 2) { s.bg = [args[k + 2], args[k + 3], args[k + 4]]; k += 4; }
          else if (a >= 30 && a <= 37) s.fg = BASE16[a - 30];
          else if (a >= 40 && a <= 47) s.bg = BASE16[a - 40];
          else if (a >= 90 && a <= 97) s.fg = BASE16[8 + a - 90];
          else if (a >= 100 && a <= 107) s.bg = BASE16[8 + a - 100];
        }
      } else if (cmd === 'A') s.y -= args[0] ?? 1;
      else if (cmd === 'B') s.y += args[0] ?? 1;
      else if (cmd === 'C') s.x += args[0] ?? 1;
      else if (cmd === 'D') s.x -= args[0] ?? 1;
      i += all.length;
      continue;
    }
    if (c === '\n') { s.y++; s.x = 0; i++; continue; }
    if (c === '\r') { s.x = 0; i++; continue; }
    if (c === '\b') { s.x = Math.max(0, s.x - 1); i++; continue; }
    if (c < ' ') { i++; continue; }
    put(c);
    i++;
  }
  return s;
}

const BASE16 = [[0, 0, 0], [205, 49, 49], [13, 188, 121], [229, 229, 16], [36, 114, 200], [188, 63, 188], [17, 168, 205], [229, 229, 229],
  [102, 102, 102], [241, 76, 76], [35, 209, 139], [245, 245, 67], [59, 142, 234], [214, 112, 214], [41, 184, 219], [255, 255, 255]];

const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function toHtml(s, { bg = '#0d0d10', fg = '#c9c9c9', fontPx = 13 } = {}) {
  const rows = s.grid.map((row) => {
    let out = '', run = null, buf = '';
    const flush = () => {
      if (!buf) return;
      const st = [];
      if (run.fg) st.push(`color:rgb(${run.fg.join(',')})`);
      if (run.bg) st.push(`background:rgb(${run.bg.join(',')})`);
      if (run.bold) st.push('font-weight:600');
      if (run.dim) st.push('opacity:.6');
      out += st.length ? `<span style="${st.join(';')}">${esc(buf)}</span>` : esc(buf);
      buf = '';
    };
    for (const cell of row) {
      const key = JSON.stringify([cell.fg, cell.bg, cell.bold, cell.dim]);
      if (run === null || key !== run.key) { flush(); run = { ...cell, key }; }
      buf += cell.ch;
    }
    flush();
    return out;
  });
  return `<!doctype html><html><body style="margin:0;background:${bg}">
<pre style="margin:0;padding:14px;background:${bg};color:${fg};font:${fontPx}px/1.25 'SF Mono','Menlo','Apple Color Emoji',monospace;white-space:pre">${rows.join('\n')}</pre>
</body></html>`;
}

// --- driver ---------------------------------------------------------------
export async function run({ bin, env = {}, script: plan, outDir }) {
  mkdirSync(outDir, { recursive: true });
  const chunks = [];
  const t0 = Date.now();
  const child = spawn('script', ['-q', '/dev/null', bin], {
    env: { ...process.env, ...env, COLUMNS: String(COLS), LINES: String(ROWS), TERM: 'xterm-256color' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (d) => chunks.push({ at: Date.now() - t0, data: d.toString('utf8') }));
  child.stderr.on('data', () => {});

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const marks = [];
  for (const step of plan) {
    await sleep(step.waitMs);
    if (step.shot) marks.push({ name: step.shot, at: Date.now() - t0 });
    if (step.keys) child.stdin.write(step.keys);
  }
  child.stdin.write('q');
  await sleep(400);
  child.kill('SIGTERM');

  const written = [];
  for (const mark of marks) {
    const s = newScreen();
    for (const c of chunks) { if (c.at <= mark.at) feed(s, c.data); }
    const html = `${outDir}/${mark.name}.html`;
    writeFileSync(html, toHtml(s));
    written.push({ name: mark.name, html, at: mark.at });
  }
  writeFileSync(`${outDir}/stream.json`, JSON.stringify({ cols: COLS, rows: ROWS, chunks }, null, 1));
  return written;
}
