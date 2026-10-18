// Writes the phase-two jobs: %INCLUDE resolution, the per-file summary the rules read, stress
// fixtures per statement kind, and a second set of benchmark pairs.
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(dirname(fileURLToPath(import.meta.url)), 'bank');
const JOBS = [];

JOBS.push({
  id: 'c04-include', title: '%INCLUDE resolution across a source tree', type: 'draft', priority: 62, estimateMin: 12,
  target: ['lib/pli/include.mjs', 'test/pli-include.test.mjs'], kinds: [],
  spec: `Write lib/pli/include.mjs and its tests. Export expandIncludes(entry, members, { maxDepth = 16 } = {}).
- entry is { path, text }; members is a Map from an upper-case member name to [{ path, text }] (every file in the tree whose base name without extension is that name, in path order). Build it with membersOf(files) where files is [{ path, text }] (export membersOf too).
- Read entry.text with readPli(text, { file: path }) from './lex.mjs'. A statement whose first token is the op '%' followed by the word INCLUDE or XINCLUDE names members: a word (\`%INCLUDE PNBACCT;\`), a quoted literal (\`%INCLUDE 'PNBACCT';\`), library(member) (\`%INCLUDE SYSLIB(PNBACCT);\` — the member is the word in the parentheses), or several separated by commas. Replace the statement with the statements of each member's file, read the same way (each keeps its own file and line, so a finding is reported where the included text sits), recursively.
- XINCLUDE includes a member only the first time it is named in the expansion; INCLUDE every time.
- Choose among several files of one member name the one in the same directory as the including file, else the first in path order; record the choice when there was more than one.
- A member with no file is recorded in unresolved [{ name, file, line }] and its statement kept in place. A member that includes itself directly or through others is recorded in cycles [{ chain: [names] }] and not expanded again. Depth beyond maxDepth is recorded in tooDeep.
- Return { statements, included: [{ name, path, from: { file, line } }], unresolved, cycles, tooDeep, ambiguous: [{ name, chosen, candidates }] }.
- Every statement keeps the shape readPli gives ({ labels, conditions, toks, line, endLine, file }).
Tests use small in-memory files: nested includes, the three member forms, XINCLUDE once, an unresolved member, a cycle, a same-directory preference, and that statements from a member carry the member's file and line.`,
  context: [{ file: 'lib/pli/lex.mjs' }],
});

JOBS.push({
  id: 'd01-summary', title: 'Per-file PL/I summary: procedures, declarations, calls, I/O, SQL and CICS, assignments', type: 'draft', priority: 75, estimateMin: 15,
  target: ['lib/pli/summary.mjs', 'test/pli-summary.test.mjs'], kinds: [],
  spec: `Write lib/pli/summary.mjs and its tests. Export summarize(statements, { file } = {}) taking statements as readPli gives them, parsing each with parseStatement from './statements.mjs', and returning a summary small enough to keep for every file of a large tree (no tokens kept except where named; names upper-case; every entry carries line and file when the statement has one):
{
  procedures: [{ name, params: [names], main: bool (OPTIONS includes MAIN), options: [names], line }],
  declarations: [{ name, level, attributes: [attribute names], based: bool, entryVariable: bool (ENTRY and VARIABLE both present), external: bool, builtin: bool, line }],
  calls: [{ name, args: [ { refs: [names] } ], line }],          // CALL
  fetches: [{ name, title: refs|null, line }],                    // FETCH
  assignments: [{ targets: [names], reads: [names], line }],      // ASSIGNMENT: target path names joined with '.', value refs
  io: [{ verb, file: name|null, into: name|null, from: name|null, set: name|null, key: [names], line }],   // READ WRITE REWRITE DELETE LOCATE OPEN CLOSE GET PUT
  sql: [{ verb, hostVars: [{ host, written }], line }],           // EXEC SQL
  cics: [{ command: [words], options: [{ name, refs: [names] }], line }], // EXEC CICS
  conditions: [{ names: [condition names], system: bool, line }], // ON
  preprocessor: [{ directive, members: [names], line }],
  prefixes: [{ conditions: [names], line }],                      // statements carrying a condition prefix such as (NOSUBSCRIPTRANGE):
  counts: { statements, parsed, unparsed, unknown, unbuilt },
  unparsed: [{ kind, line, reason }]                              // at most 50
}
Statements nested as units (node.units of IF, ELSE, WHEN, OTHERWISE, ON) are summarised as well, with their own lines. The parsed node shapes are those the parsers in lib/pli/stmt/ return (shown); for a kind whose parser is not built yet (status 'unbuilt'), count it and add nothing else. Refs: an expression node is { t: 'expr', tree, toks, refs } with refs the names read; a reference node is { t: 'ref', name, path, args }.
Tests build statements with readPli and check each list on short programs: a MAIN procedure with parameters, a structure, an ENTRY VARIABLE, a CALL with arguments, an assignment, an IF whose unit is a CALL, and counts.`,
  context: [{ file: 'lib/pli/statements.mjs' }, { file: 'lib/pli/expr.mjs' }, { file: 'lib/pli/stmt/declare.mjs' }, { file: 'lib/pli/stmt/procedure.mjs' }, { file: 'lib/pli/stmt/call.mjs' }, { file: 'lib/pli/stmt/assignment.mjs' }],
  maxTokens: 18000,
});

const STRESS = {
  declare: 'DECLARE and DCL: factored lists nested two deep, structures to level 5 with dimensions at several levels, every arithmetic attribute combination, VARYING and VARYINGZ, PICTURE with repetition factors, BASED and DEFINED with POSITION, INITIAL with iteration factors and asterisks, ENTRY with descriptors and RETURNS, FILE attributes with ENVIRONMENT, LIKE, UNION, VALUE, ALIGNED/UNALIGNED, an unnamed (*) member',
  control: 'DO in every form (group, WHILE, UNTIL, both, iterative with TO/BY in both orders, REPEAT, several specs separated by commas with WHILE on one, LOOP, FOREVER), IF with every kind of unit and nested IF ... THEN ... ELSE IF, SELECT with and without a subject, WHEN with several values, OTHERWISE, LEAVE and ITERATE with labels, GO TO and GOTO with subscripted labels, RETURN with and without a value, STOP, EXIT',
  procedure: 'PROCEDURE and PROC with every option (OPTIONS lists, RETURNS, RECURSIVE, REORDER, EXTERNAL with a literal), ENTRY statements with parameters, PACKAGE with EXPORTS and RESERVES, BEGIN with options, END with and without labels, labels on every statement',
  call: 'CALL with no arguments, empty parentheses, nested function references, cross-section asterisks, locator-qualified references, qualified entry names; FETCH with SET and TITLE; RELEASE of a list and of *',
  assignment: 'assignments with several targets, every compound operator, BY NAME, pseudovariables (SUBSTR, UNSPEC, STRING) as targets, locator-qualified and subscripted targets, comparisons and concatenations in the value, repetition-factor literals, bit and hex literals',
  io: 'OPEN with several files and every option in varied order, CLOSE of several files, READ with INTO/SET/IGNORE and KEY/KEYTO/NOLOCK, WRITE with KEYFROM, REWRITE, DELETE with KEY, LOCATE with SET and KEYFROM, UNLOCK',
  stream: 'PUT and GET with FILE and STRING, SKIP and SKIP(n), PAGE, LINE, LIST, EDIT with several data/format pairs and every format item with iteration factors, DATA with and without a list, repetitive specifications (X(I) DO I = 1 TO N); DISPLAY with REPLY; FORMAT statements; FLUSH',
  conditions: 'ON with every condition name and abbreviation, several conditions sharing one unit, SNAP, SYSTEM, units of BEGIN, assignment, GO TO, CALL and PUT; SIGNAL and REVERT of conditions with arguments; RESIGNAL',
  exec: 'EXEC SQL with SELECT INTO with indicator variables (both forms), INSERT with VALUES, UPDATE, DELETE, DECLARE CURSOR, OPEN, FETCH INTO, CLOSE, INCLUDE SQLCA, WHENEVER, COMMIT, PREPARE, EXECUTE USING, CALL; EXEC CICS RECEIVE, SEND MAP, RETURN with TRANSID and COMMAREA, LINK, XCTL, READ, WRITE, HANDLE CONDITION, ASSIGN, GETMAIN',
  preprocessor: '%INCLUDE in all three member forms and with several members, %XINCLUDE, %DCL of preprocessor variables, %assignments, %IF ... %THEN ... %ELSE with %DO ... %END, %ACTIVATE and %DEACTIVATE, %PAGE, %SKIP(2), %NOTE, %PROCEDURE with RETURNS and %RETURN, %REPLACE',
};
for (const [module, forms] of Object.entries(STRESS)) {
  JOBS.push({
    id: `s-${module}`, title: `Stress fixture: unusual but valid ${module} statements`, type: 'draft', priority: 40, estimateMin: 8,
    target: [`bench/pli/stress/${module}.pli`], kinds: [],
    spec: `Write bench/pli/stress/${module}.pli: one PL/I source file of 40 to 80 statements exercising ${forms}. Every statement must be valid IBM Enterprise PL/I; prefer forms a real bank's program would contain, then rarer forms the language allows. Put it inside a procedure (\`STRESS: PROCEDURE OPTIONS(MAIN);\` ... \`END STRESS;\`) with the DCL statements the rest needs. Indent every line at least one column. The automatic gate reads the file with the project's PL/I reader and refuses it if a statement fits no PL/I statement kind or the text has a lexical error, so do not invent statements. Under Limits, list each form you are not sure Enterprise PL/I accepts as UNVERIFIED with the line.`,
    context: [],
  });
}

const FORMAT = `Each case is a directory holding one or more PL/I source files (.pli) and a manifest.json: { "id": "<directory name>", "title": "<one sentence>", "cwe": "CWE-<n>", "expect": [{ "rule": "<rule id>", "file": "<file name>" }] } for the positive; the negative has "expect": [] and "negativeFor": ["<rule id>"] and differs only in the one thing that makes the positive a finding. Ordinary IBM Enterprise PL/I, 30 to 80 lines, every line indented at least one column, no real company, person or credential. The gate refuses a .pli with a lexical error or a statement no PL/I statement kind fits. Reply with every file of both directories under its full path.`;
const pair = (n, slug, title, cwe, rule, pos, neg) => ({
  id: `f${n}-${slug}`, title: `Benchmark pair: ${title}`, type: 'draft', priority: 42, estimateMin: 9,
  target: [], writableDirs: [`bench/pli/${slug}/`, `bench/pli/${slug}-negative/`], kinds: [],
  spec: `Write a benchmark case pair: bench/pli/${slug}/ (positive) and bench/pli/${slug}-negative/ (near-miss negative), for rule ${rule} (${cwe}).\n\nPositive: ${pos}\n\nNegative: ${neg}\n\n${FORMAT}`,
  context: [],
});
JOBS.push(
  pair(13, 'web-input-to-os-command', 'web request data run as a system command', 'CWE-78', 'cics-web-to-os-command',
    'A CICS web program uses EXEC CICS WEB RECEIVE INTO(BODY) and passes a field of it to CALL BPX1SPN (or a USS shell call) as the command.',
    'The same program with the command a literal and the input only logged.'),
  pair(14, 'parm-into-file-title', 'a dataset name from PARM opened as a file', 'CWE-73', 'argv-or-env-to-dynamic-file-path',
    'MAIN procedure parameter PARM CHAR(100) VARYING; a dataset name cut from it with SUBSTR and used as OPEN FILE(OUTF) TITLE(DSN) OUTPUT.',
    'The same program opening FILE(OUTF) with its DD name only.'),
  pair(15, 'terminal-to-queue-name', 'terminal input naming a TS queue', 'CWE-99', 'cics-terminal-to-queue-name',
    'EXEC CICS RECEIVE INTO(INAREA); EXEC CICS WRITEQ TS QUEUE(INAREA.QNAME) FROM(DATA_AREA);',
    'The same program with QUEUE built from EIBTRMID and a constant prefix.'),
  pair(16, 'database-to-xctl-program', 'a program name read from a table and transferred to', 'CWE-470', 'database-to-cics-dynamic-transfer',
    'EXEC SQL SELECT NEXT_PGM INTO :NEXT_PGM FROM ROUTING WHERE TXN = :TXN; EXEC CICS XCTL PROGRAM(NEXT_PGM);',
    'The same program choosing among literal program names with SELECT on the value read.'),
  pair(17, 'record-to-loop-bound', 'a record count field bounding a loop over a fixed array', 'CWE-606', 'file-record-to-loop-bound',
    'READ FILE(IN) INTO(HDR); DO I = 1 TO HDR.COUNT; TOTAL = TOTAL + ITEMS(I); END; with ITEMS dimensioned 100.',
    'The same loop bounded by MIN(HDR.COUNT, HBOUND(ITEMS, 1)).'),
  pair(18, 'terminal-to-sysid', 'terminal input choosing the remote system', 'CWE-15', 'cics-terminal-to-cics-sysid',
    'EXEC CICS RECEIVE INTO(INAREA); EXEC CICS LINK PROGRAM(\'ACCTINQ\') COMMAREA(CA) SYSID(INAREA.SYS);',
    'The same LINK with SYSID(\'PRD1\').'),
  pair(19, 'password-compare-in-program', 'a password checked by the program against a stored copy', 'CWE-256', 'program-checks-stored-password',
    'EXEC SQL SELECT PWD INTO :STORED_PWD FROM USERS WHERE ID = :USERID; IF STORED_PWD = INPUT_PWD THEN ...',
    'The same program calling EXEC CICS VERIFY PASSWORD(INPUT_PWD) USERID(USERID) instead.'),
  pair(20, 'error-text-to-screen', 'Db2 error text shown on the terminal', 'CWE-209', 'system-response-to-screen',
    'After a failed EXEC SQL, MSG_LINE = SQLERRMC; EXEC CICS SEND TEXT FROM(MSG_LINE);',
    'The same program sending a fixed message keyed by SQLCODE through a SELECT of literal texts.'),
);

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
