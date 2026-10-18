#!/usr/bin/env node
// Draws the veld mark — open grassland, savannah gold over a warm horizon, neuron roots below it —
// and rasterises it. The SVG is generated rather than hand-drawn: a blade is a spine plus a width
// profile that tapers to zero at the tip, and hand-tuned beziers kept near-parallel sides that read
// as cattails instead of grass.
//
//   node scripts/render-logo.mjs            # writes veld_logo.svg and every raster that ships
//   node scripts/render-logo.mjs --svg-only
//
// Rasterising needs Chrome; without it the SVG is still written and the PNG step is reported skipped.

import { writeFileSync, readFileSync, existsSync, mkdtempSync, copyFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SVG_OUT = join(ROOT, 'veld_logo.svg');
const PNG_OUT = join(ROOT, 'veld_logo_nobackground.png');
const CHROME = process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const GOLD = '#E8A33D', AMBER = '#D4892C', DEEP = '#B5651D', EARTH = '#A0845C', HORIZON = '#8B6F47';
const H = 648;

const cube = (p0, c1, c2, p1, t) => {
  const u = 1 - t;
  return [u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p1[0],
    u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p1[1]];
};
const dcube = (p0, c1, c2, p1, t) => {
  const u = 1 - t;
  return [3 * u * u * (c1[0] - p0[0]) + 6 * u * t * (c2[0] - c1[0]) + 3 * t * t * (p1[0] - c2[0]),
    3 * u * u * (c1[1] - p0[1]) + 6 * u * t * (c2[1] - c1[1]) + 3 * t * t * (p1[1] - c2[1])];
};

// `seed` bulges the width near the top so a seed head is part of the blade's own silhouette. Drawn
// as a separate shape it reads as an arrowhead stuck on the stem.
function blade(base, c1, c2, tip, wBase, { taper = 2.1, seed = 0, seedAt = 0.74, spread = 0.1, steps = 72 } = {}) {
  const L = [], R = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const [x, y] = cube(base, c1, c2, tip, t);
    const [dx, dy] = dcube(base, c1, c2, tip, t);
    const len = Math.hypot(dx, dy) || 1;
    const bulge = seed ? seed * Math.exp(-((t - seedAt) ** 2) / (2 * spread ** 2)) * (1 - t) ** 0.6 : 0;
    const w = (wBase * (1 - t) ** taper + bulge) / 2;
    L.push([x + (-dy / len) * w, y + (dx / len) * w]);
    R.push([x - (-dy / len) * w, y - (dx / len) * w]);
  }
  const n = (v) => Math.round(v * 10) / 10;
  return `M${[...L, ...R.reverse()].map(([x, y]) => `${n(x)} ${n(y)}`).join(' L')} Z`;
}

// Tips stay at or below y=160: the mark is scaled 1.3x about (512,548), so anything higher leaves
// the 1024 canvas and the tallest blade loses its point.
const BLADES = [
  { b: [404, H], c1: [352, 540], c2: [250, 430], t: [172, 342], w: 44, f: AMBER, o: 0.88 },
  { b: [438, H], c1: [404, 520], c2: [316, 356], t: [238, 226], w: 52, f: GOLD, o: 1, seed: 56 },
  { b: [474, H], c1: [462, 480], c2: [424, 290], t: [398, 160], w: 58, f: DEEP, o: 1 },
  { b: [512, H], c1: [516, 466], c2: [528, 272], t: [540, 170], w: 64, f: GOLD, o: 1, seed: 66 },
  { b: [552, H], c1: [566, 480], c2: [606, 300], t: [648, 182], w: 56, f: DEEP, o: 0.96 },
  { b: [588, H], c1: [628, 520], c2: [714, 372], t: [790, 254], w: 50, f: GOLD, o: 1, seed: 52 },
  { b: [622, H], c1: [676, 552], c2: [772, 448], t: [856, 372], w: 42, f: AMBER, o: 0.86 },
];

// Roots are brown rather than the white of the basecamp sketch: the mark ships on a transparent
// background, and white roots disappear on a light desktop.
const ROOTS = `  <g stroke="${EARTH}" fill="none" stroke-linecap="round">
    <path d="M438 ${H} L404 726" stroke-width="19" opacity="0.78"/>
    <path d="M404 726 L310 812" stroke-width="12" opacity="0.55"/>
    <path d="M404 726 L410 828" stroke-width="11" opacity="0.48"/>
    <path d="M310 812 L262 864" stroke-width="8" opacity="0.3"/>
    <path d="M512 ${H} L518 754" stroke-width="23" opacity="0.86"/>
    <path d="M518 754 L440 862" stroke-width="15" opacity="0.56"/>
    <path d="M518 754 L598 862" stroke-width="15" opacity="0.56"/>
    <path d="M518 754 L516 896" stroke-width="16" opacity="0.62"/>
    <path d="M518 800 L466 858" stroke-width="8" opacity="0.32"/>
    <path d="M518 800 L572 854" stroke-width="8" opacity="0.32"/>
    <path d="M588 ${H} L634 726" stroke-width="19" opacity="0.78"/>
    <path d="M634 726 L724 812" stroke-width="12" opacity="0.55"/>
    <path d="M634 726 L628 828" stroke-width="11" opacity="0.48"/>
    <path d="M724 812 L772 864" stroke-width="8" opacity="0.3"/>
    <path d="M404 726 L518 754" stroke-width="7" opacity="0.22" stroke-dasharray="15 17"/>
    <path d="M634 726 L518 754" stroke-width="7" opacity="0.22" stroke-dasharray="15 17"/>
  </g>
  <g fill="${EARTH}">
    <circle cx="404" cy="726" r="16" opacity="0.7"/><circle cx="518" cy="754" r="21" opacity="0.82"/>
    <circle cx="634" cy="726" r="16" opacity="0.7"/><circle cx="310" cy="812" r="11" opacity="0.44"/>
    <circle cx="724" cy="812" r="11" opacity="0.44"/><circle cx="440" cy="862" r="10" opacity="0.4"/>
    <circle cx="598" cy="862" r="10" opacity="0.4"/>
  </g>`;

// Blades only, cropped and stretched to the braille dot grid. Same table, so the terminal mark
// and the shipped PNG cannot drift apart.
function bladesSvg(lean = 0, w = 40, h = 24) {
  const paths = BLADES.map((sb) => {
    const tip = [sb.t[0] + lean * (sb.t[0] < 512 ? 0.8 : 1), sb.t[1] + Math.abs(lean) * 0.18];
    const c2 = [sb.c2[0] + lean * 0.5, sb.c2[1]];
    return `  <path d="${blade(sb.b, sb.c1, c2, tip, sb.w, { seed: sb.seed ?? 0 })}" fill="${sb.f}" opacity="${sb.o}"/>`;
  }).join('\n');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="60 40 910 620" preserveAspectRatio="none" fill="none">
  <g transform="translate(512 548) scale(1.3) translate(-512 -548)">
${paths}
  </g>
</svg>
`;
}

const body = [
  ROOTS,
  `  <path d="M206 ${H} H818" stroke="${HORIZON}" stroke-width="11" opacity="0.38" stroke-linecap="round"/>`,
  ...BLADES.map((s) => `  <path d="${blade(s.b, s.c1, s.c2, s.t, s.w, { seed: s.seed ?? 0 })}" fill="${s.f}" opacity="${s.o}"/>`),
].join('\n');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024" fill="none">
  <!-- veld - open grassland. Generated by scripts/render-logo.mjs; edit that, not this file. -->
  <g transform="translate(512 548) scale(1.3) translate(-512 -548)">
${body}
  </g>
</svg>
`;

// A malformed SVG still rasterises: Chrome renders up to the first error and bakes its pink error
// banner into the screenshot, which is how an unbalanced </g> survived a visual check.
function assertWellFormed(text) {
  const tags = [];
  for (const m of text.matchAll(/<(\/?)([a-zA-Z][\w:-]*)([^>]*?)(\/?)>/g)) {
    const [, closing, name, attrs, selfClose] = m;
    if (selfClose || attrs.endsWith('/')) continue;
    if (closing) {
      const open = tags.pop();
      if (open !== name) throw new Error(`SVG is malformed: </${name}> closes <${open ?? 'nothing'}>`);
    } else tags.push(name);
  }
  if (tags.length) throw new Error(`SVG is malformed: unclosed <${tags.join('>, <')}>`);
}

assertWellFormed(svg);
writeFileSync(SVG_OUT, svg);
console.log(`wrote ${SVG_OUT} (${svg.length} bytes, well-formed)`);

// --container writes the app-icon variant: the same mark on a filled rounded square. Grass is airy
// and loses weight when a transparent 1024 is resampled to 32; a container holds contrast at any size,
// which is what a dock, a favicon and the Tauri set need.
export function containerSvg(theme = 'light') {
  const bg = theme === 'dark' ? '#241F19' : '#F5EFE3';
  const ring = theme === 'dark' ? '#3A322A' : '#E2D8C6';
  const horizon = theme === 'dark' ? '#6E5B44' : HORIZON;
  const root = theme === 'dark' ? '#7A684F' : EARTH;
  const body = [
    ROOTS.replaceAll(EARTH, root),
    `  <path d="M206 ${H} H818" stroke="${horizon}" stroke-width="11" opacity="0.45" stroke-linecap="round"/>`,
    ...BLADES.map((sb) => `  <path d="${blade(sb.b, sb.c1, sb.c2, sb.t, sb.w, { seed: sb.seed ?? 0 })}" fill="${sb.f}" opacity="${sb.o}"/>`),
  ].join('\n');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024" fill="none">
  <rect x="64" y="64" width="896" height="896" rx="200" fill="${bg}"/>
  <rect x="64.5" y="64.5" width="895" height="895" rx="199.5" fill="none" stroke="${ring}" stroke-width="3"/>
  <clipPath id="veldIcon"><rect x="64" y="64" width="896" height="896" rx="200"/></clipPath>
  <g clip-path="url(#veldIcon)">
    <g transform="translate(512 560) scale(1.12) translate(-512 -548)">
${body}
    </g>
  </g>
</svg>
`;
}

if (process.argv.includes('--container')) {
  for (const theme of ['light', 'dark']) {
    const svg = containerSvg(theme);
    assertWellFormed(svg);
    const out = join(ROOT, theme === 'light' ? 'veld_icon.svg' : 'veld_icon_dark.svg');
    writeFileSync(out, svg);
    console.log(`wrote ${out}`);
  }
  if (existsSync(CHROME)) {
    const tmpC = mkdtempSync(join(tmpdir(), 'veld-icon-'));
    for (const [theme, file] of [['light', 'veld_icon.png'], ['dark', 'veld_icon_dark.png']]) {
      const src = join(ROOT, theme === 'light' ? 'veld_icon.svg' : 'veld_icon_dark.svg');
      const shot = join(tmpC, `${theme}.png`);
      execFileSync(CHROME, ['--headless', '--disable-gpu', `--screenshot=${shot}`, '--window-size=1024,1024',
        '--default-background-color=00000000', '--hide-scrollbars', `file://${src}`], { stdio: 'ignore' });
      copyFileSync(shot, join(ROOT, file));
      console.log(`wrote ${file}`);
    }
  }
  process.exit(0);
}

// --braille prints the Rust art block for tui/src/logo.rs: the same blades, rasterised into
// braille cells (2 dots wide, 4 tall) at 20 columns by 6 rows.
if (process.argv.includes('--braille')) {
  const { inflateSync } = await import('node:zlib');
  const W = 40, HH = 24, THRESHOLD = 30;
  const DOT = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
  const alphaOf = (file) => {
    const buf = readFileSync(file);
    let p = 8, w = 0, h = 0, depth = 0, colour = 0; const idat = [];
    while (p < buf.length) {
      const len = buf.readUInt32BE(p), type = buf.toString('ascii', p + 4, p + 8);
      const data = buf.subarray(p + 8, p + 8 + len);
      if (type === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); depth = data[8]; colour = data[9]; }
      if (type === 'IDAT') idat.push(data);
      if (type === 'IEND') break;
      p += 12 + len;
    }
    if (depth !== 8 || colour !== 6) throw new Error(`expected 8-bit RGBA, got depth ${depth} colour ${colour}`);
    const raw = inflateSync(Buffer.concat(idat)), bpp = 4, stride = w * bpp;
    const out = Buffer.alloc(h * stride);
    for (let y = 0; y < h; y++) {
      const ft = raw[y * (stride + 1)];
      const line = raw.subarray(y * (stride + 1) + 1, y * (stride + 1) + 1 + stride);
      for (let x = 0; x < stride; x++) {
        const a = x >= bpp ? out[y * stride + x - bpp] : 0;
        const b = y > 0 ? out[(y - 1) * stride + x] : 0;
        const c = y > 0 && x >= bpp ? out[(y - 1) * stride + x - bpp] : 0;
        let v = line[x];
        if (ft === 1) v += a; else if (ft === 2) v += b; else if (ft === 3) v += (a + b) >> 1;
        else if (ft === 4) { const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
          v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c); }
        out[y * stride + x] = v & 0xff;
      }
    }
    return Array.from({ length: h }, (_, y) => Array.from({ length: w }, (_, x) => out[y * stride + x * 4 + 3]));
  };
  if (!existsSync(CHROME)) { console.error(`no Chrome at ${CHROME}`); process.exit(1); }
  const tmpB = mkdtempSync(join(tmpdir(), 'veld-braille-'));
  const frames = [0, 1, 2, 3].map((i) => {
    const lean = 46 * Math.sin((i / 4) * Math.PI * 2);
    const f = join(tmpB, `f${i}.svg`);
    writeFileSync(f, bladesSvg(lean, W, HH));
    const shot = join(tmpB, `f${i}.png`);
    execFileSync(CHROME, ['--headless', '--disable-gpu', `--screenshot=${shot}`, `--window-size=${W},${HH}`,
      '--default-background-color=00000000', '--hide-scrollbars', `file://${f}`], { stdio: 'ignore' });
    const a = alphaOf(shot);
    return Array.from({ length: HH / 4 }, (_, cy) => Array.from({ length: W / 2 }, (_, cx) => {
      let bits = 0;
      for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 2; dx++) {
        if ((a[cy * 4 + dy]?.[cx * 2 + dx] ?? 0) >= THRESHOLD) bits |= DOT[dy][dx];
      }
      return String.fromCharCode(0x2800 + bits);
    }).join(''));
  });
  const rs = (rows) => rows.map((r) => `        "${r}",`).join('\n');
  console.log(`pub const GRASS: &[&str] = &[\n${frames[0].map((r) => `    "${r}",`).join('\n')}\n];\n`);
  console.log('pub const GRASS_FRAMES: &[&[&str]] = &[');
  frames.forEach((f, i) => console.log(`    // Frame ${i}: ${['upright', 'leaning right', 'upright', 'leaning left'][i]}\n    &[\n${rs(f)}\n    ],`));
  console.log('];');
  process.exit(0);
}

if (process.argv.includes('--svg-only')) process.exit(0);
if (!existsSync(CHROME)) {
  console.log(`SKIPPED the PNG: no Chrome at ${CHROME}. Set CHROME_BIN, or render ${SVG_OUT} yourself.`);
  process.exit(0);
}
const tmp = mkdtempSync(join(tmpdir(), 'veld-logo-'));
const shoot = (out, w, h, url) => execFileSync(CHROME, ['--headless', '--disable-gpu',
  `--screenshot=${out}`, `--window-size=${w},${h}`, '--default-background-color=00000000',
  '--hide-scrollbars', '--allow-file-access-from-files', url], { stdio: 'ignore' });

const shot = join(tmp, 'logo.png');
shoot(shot, 1024, 1024, `file://${SVG_OUT}`);

// Every raster the repo ships comes from this one render, so they cannot drift apart.
const sips = (src, dst, size) => execFileSync('sips', ['-z', String(size), String(size), src, '--out', dst], { stdio: 'ignore' });
for (const [rel, size] of [['veld_logo_nobackground.png', 1024], ['assets/logo.png', 1024], ['assets/logo_32x32.png', 32]]) {
  const dst = join(ROOT, rel);
  if (size === 1024) copyFileSync(shot, dst); else sips(shot, dst, size);
  console.log(`wrote ${rel}`);
}

// The TUI mark is deliberately pixel art: render small, then upscale with nearest-neighbour so the
// pixels stay square. A plain resample would just produce a blurry copy of the vector.
const small = join(tmp, 'small.png');
sips(shot, small, 40);
const pixelHtml = join(tmp, 'pixel.html');
writeFileSync(pixelHtml, `<html><body style="margin:0"><img src="file://${small}" `
  + `style="width:1024px;height:1024px;image-rendering:pixelated"></body></html>`);
const pixelShot = join(tmp, 'pixel.png');
shoot(pixelShot, 1024, 1024, `file://${pixelHtml}`);
copyFileSync(pixelShot, join(ROOT, 'tui/veld-pixel.png'));
console.log('wrote tui/veld-pixel.png');
