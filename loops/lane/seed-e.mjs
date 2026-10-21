// Writes the PL/I rule jobs: one module of lib/pli/rules/ each, with its tests.
import * as PATHS from '../paths.mjs';
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(PATHS.LANE_HOME, 'bank');

const CONTRACT = `The contract (lib/pli/rules/index.mjs, shown): a rule module exports RULES, a map from rule id to { sev: 'info'|'low'|'med'|'high'|'crit', evidence: 'coverage'|'construct'|'advisory', cwe: 'CWE-<n>', text: one sentence saying what the finding means }, and check(program) returning findings [{ rule, path, line, detail }] where detail is one sentence naming the construct and its operands (never source text beyond names and keywords). program is { path, statements } and each statement is { labels, conditions, toks, line, endLine, parsed } with parsed = { kind, status, node } from parseStatement. Nested statements (the unit of IF, ELSE, WHEN, OTHERWISE, ON) are in parsed.node.units, each again { kind, status, node } — walk them too. Only statements with parsed.status === 'parsed' carry a node you can rely on; skip the rest.

Node shapes you will meet: DECLARE { items: [{ level, name, dims, attributes: [{ name, args, picture }] }] } with attribute names expanded (BASED, POINTER, ENTRY, VARIABLE, ...); CALL { callee: { t: 'ref', name, path }, name, args }; FETCH { entries: [{ name, set, title }] }; ASSIGNMENT { targets: [ref], op, value: { t: 'expr', refs, tree } }; ON { conditions: [{ name, arg }], snap, system, units }; PREPROCESSOR { directive, members, units }; BEGIN, END; PROCEDURE { name, params, options }.

Tests (test/pli-rules-<module>.test.mjs) build programs with programOf(path, text) from '../lib/pli/rules/index.mjs' and call your module's check directly, with one positive and one near-miss negative per rule, plus a test that every RULES entry has sev, evidence, cwe and text. Write PL/I test sources with every line starting in column 2 or later.`;

const job = (id, module, title, priority, spec, benches) => ({
  id, title, type: 'draft', priority, estimateMin: 10,
  target: [`lib/pli/rules/${module}.mjs`, `test/pli-rules-${module}.test.mjs`], kinds: [],
  spec: `Write lib/pli/rules/${module}.mjs (a stub today) and test/pli-rules-${module}.test.mjs.\n\n${spec}\n\n${CONTRACT}${benches ? `\n\nThe benchmark pair ${benches} in bench/pli/ (if present below) shows the intended positive and negative.` : ''}`,
  context: [{ file: 'lib/pli/rules/index.mjs' }, { file: `lib/pli/rules/${module}.mjs` }, { file: 'lib/sets/opaque.mjs' }],
});

const JOBS = [
  job('e01-rules-based', 'based', 'Rule: BASED storage addressed through a pointer', 58,
    `Rule 'pli-based-storage-addressing' (sev info, evidence coverage, CWE-119): the PL/I counterpart of opaque-pointer-addressing in lib/sets/opaque.mjs (shown). A variable declared BASED (with or without a locator, BASED(P) or BASED(ADDR(X))) is storage addressed through a pointer, so data flow through it is not followed. Report ONE finding per program, at the first BASED declaration, with the number of BASED variables and up to three names in the detail; say in the detail when a locator is ADDR() of a declared variable (an overlay) and when it is a plain pointer (addressing whatever the pointer holds). Also count assignments to a POINTER-declared variable from POINTERADD/PTRADD/POINTERVALUE or arithmetic, and mention the first one in the detail: computed addresses are the case that matters most.`, 'based-storage-from-input'),
  job('e02-rules-entry', 'entry', 'Rule: calls through an entry variable', 58,
    `Rule 'pli-entry-variable-call' (sev info, evidence coverage, CWE-470): a CALL whose callee is a name declared with both ENTRY and VARIABLE (an entry variable), or a member of a structure or array declared ENTRY (a member of a structure is a variable), so the callee is chosen at run time. Report each such CALL with the variable's name, and in the detail the names of the entry constants assigned to it in the program (ASSIGNMENT whose target is the variable and whose value is a single reference), or say none were seen. A CALL of an entry constant (declared ENTRY without VARIABLE, or a PROCEDURE in the program, or undeclared) is not a finding. FETCH of an entry whose TITLE is not a literal is a second rule, 'pli-fetch-title-variable' (sev med, evidence construct, CWE-494): the module loaded is named at run time.`, 'entry-variable-call and fetch-name-from-input'),
  job('e03-rules-preprocessor', 'preprocessor', 'Rule: the source read is not the source compiled', 56,
    `Rule 'pli-preprocessor-in-use' (sev info, evidence coverage, CWE-1164 is wrong — choose the CWE yourself and justify it in Limits): a program whose text the PL/I preprocessor changes before compilation: %IF, %DO, %DCL/%DECLARE, %ACTIVATE, preprocessor assignments, %PROCEDURE. Report ONE finding per program at the first such statement, listing the directives used. %INCLUDE and %XINCLUDE alone do NOT fire this rule (they are read through); %PAGE, %SKIP, %PRINT, %NOPRINT, %NOTE never do. A second rule 'pli-include-unresolved' is NOT yours: do not write it.`, 'preprocessor-conditional-include'),
  job('e04-rules-conditions', 'conditions', 'Rule: errors caught and ignored', 57,
    `Rule 'pli-error-condition-ignored' (sev low, evidence construct, CWE-390): an ON unit for ERROR, ANYCONDITION/ANYCOND, CONVERSION/CONV, ZERODIVIDE/ZDIV, FIXEDOVERFLOW/FOFL, SIZE, STRINGRANGE/STRG, SUBSCRIPTRANGE/SUBRG, KEY, RECORD, TRANSMIT or UNDEFINEDFILE/UNDF that does nothing with the condition: a null unit (ON ERROR;), a BEGIN block whose only statements are END or GO TO (the statements between the ON ... BEGIN statement and its matching END — count nested BEGIN/DO/SELECT/END to find it), or a unit that is only a GO TO. A unit that writes (PUT, DISPLAY, CALL of anything, WRITE), sets a return code (CALL PLIRETC), SIGNALs, RESIGNALs, STOPs or EXITs is not a finding. ON ERROR SYSTEM is not a finding (the system action stands). ENDFILE and ENDPAGE are never findings (a GO TO there is ordinary loop exit). Report each ON statement found, naming the conditions and what the unit does instead.`, 'on-error-swallowed'),
];
for (const j of JOBS) {
  const slugs = j.spec.match(/benchmark pair ([a-z0-9-]+(?: and [a-z0-9-]+)?)/)?.[1]?.split(' and ') || [];
  for (const s of slugs) j.context.push({ file: `bench/pli/${s}/manifest.json` });
}

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
