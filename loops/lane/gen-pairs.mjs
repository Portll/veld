// Writes benchmark-pair jobs from the flow engine's own rules: for each <source>-to-<sink> rule a PL/I
// program can express, a positive where the input reaches the sink and a near-miss where it does not.
//
//   node gen-pairs.mjs [--max N] [--dry]
import { writeFileSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const REPO = '/Users/portll/Repositories/Portll/cobolwork-pli-20261001';
const args = process.argv.slice(2);
const MAX = args.includes('--max') ? Number(args[args.indexOf('--max') + 1]) : 12;
const { RULES } = await import(join(REPO, 'lib/sets/flow.mjs'));

const SOURCE = {
  'argv-or-env': 'a batch program whose MAIN procedure takes the PARM string as its parameter: `P: PROCEDURE (PARM) OPTIONS(MAIN); DCL PARM CHAR(100) VARYING;` — the value comes from SUBSTR(PARM, ...)',
  'cics-terminal': 'a CICS program receiving terminal input: `EXEC CICS RECEIVE INTO(INAREA) LENGTH(INLEN);` with INAREA a level-1 structure and INLEN FIXED BIN(15)',
  'cics-web': 'a CICS web program receiving a request body: `EXEC CICS WEB RECEIVE INTO(BODY) LENGTH(BLEN);` with BODY a structure',
  'file-record': 'a batch program reading a record: `DCL INF FILE RECORD INPUT;`, `ON ENDFILE(INF) EOF = \'1\'B;`, `OPEN FILE(INF) INPUT;`, `READ FILE(INF) INTO(REC);`',
  'database': 'a program reading a column: `EXEC SQL SELECT COL INTO :HV FROM T WHERE K = :KEY;`',
  'jcl-parm': 'a batch program run by a job step: write a third file, RUN.jcl, with `//JOB1     JOB (ACCT),\'BATCH\'` and `//STEP1    EXEC PGM=MAINPGM,PARM=\'...\'`, and MAIN.pli as `MAINPGM: PROCEDURE (PARM) OPTIONS(MAIN); DCL PARM CHAR(100) VARYING;` — the value comes from SUBSTR(PARM, ...)',
  'jcl-instream': 'a batch program reading in-stream data: write a third file, RUN.jcl, with `//JOB1     JOB (ACCT),\'BATCH\'`, `//STEP1    EXEC PGM=MAINPGM` and `//SYSIN    DD *` followed by two data lines and `/*`; MAIN.pli is `MAINPGM: PROCEDURE OPTIONS(MAIN);` with `DCL SYSIN FILE RECORD INPUT;` and `READ FILE(SYSIN) INTO(REC);`',
};

const SINK = {
  'dynamic-sql': 'statement text built with || from the value, then `EXEC SQL PREPARE S1 FROM :SQLTXT;` and `EXEC SQL EXECUTE S1;`',
  'dynamic-program-load': 'a FETCH whose TITLE is the value: `DCL MOD ENTRY;` `FETCH MOD TITLE(MODNAME);` then `CALL MOD;`',
  'cics-dynamic-transfer': '`EXEC CICS LINK PROGRAM(PGMNAME) COMMAREA(CA);` (or XCTL) with PGMNAME the value',
  'dynamic-file-path': '`OPEN FILE(OUTF) TITLE(DSN) OUTPUT;` with `DCL OUTF FILE RECORD OUTPUT;` and DSN the value',
  'queue-name': '`EXEC CICS WRITEQ TS QUEUE(QNAME) FROM(DATA_AREA);` with QNAME the value',
  'subscript': 'the value, converted to FIXED BIN(31), used as the subscript of a table `DCL RATES(12) FIXED DEC(7,4);` in `R = RATES(IX);`',
  'reference-modification': 'the value, converted to FIXED BIN(31), used as the start of `SUBSTR(BUFFER, START, 10)`',
  'storage-length': '`EXEC CICS GETMAIN SET(P) FLENGTH(REQLEN);` with REQLEN FIXED BIN(31) the value',
  'web-response': '`EXEC CICS WEB SEND FROM(REPLY) LENGTH(RLEN);` with REPLY built from the value',
  'http-header': '`EXEC CICS WEB WRITE HTTPHEADER(HNAME) NAMELENGTH(HNL) VALUE(HVAL) VALUELENGTH(HVL);` with HVAL the value',
  'outbound-host': '`EXEC CICS WEB OPEN HOST(HOSTNM) HOSTLENGTH(HL) SESSTOKEN(TOK);` with HOSTNM the value',
  'connection-target': '`EXEC SQL CONNECT TO :LOCN;` with LOCN the value',
  'cics-sysid': "`EXEC CICS LINK PROGRAM('ACCTINQ') COMMAREA(CA) SYSID(SYS);` with SYS the value",
  'message-queue': '`CALL MQPUT(HCONN, HOBJ, MD, PMO, BUFLEN, BUFFER, CC, RC);` with BUFFER holding the value',
  'extrapartition-queue': "`EXEC CICS WRITEQ TD QUEUE('CSSL') FROM(LINE) LENGTH(LL);` with LINE holding the value",
};

const PITFALLS = `PL/I, not COBOL: no END-EXEC (an EXEC statement ends at its semicolon); OPEN FILE(F) INPUT; READ FILE(F) INTO(R) is its own statement; GET and PUT take LIST(...) or EDIT(...)(...); items of a list are separated by commas; a DCL declares simple names, never A.B; PROCEDURE options are OPTIONS(...), RETURNS(...), RECURSIVE, REORDER; every statement ends with a semicolon. Indent every line at least one column.`;

const bank = new Set(readdirSync(join(ROOT, 'bank')).map((f) => f.replace(/\.json$/, '')));
let wrote = 0;
for (const rule of Object.keys(RULES).sort()) {
  if (wrote >= MAX) break;
  const [src, snk] = rule.split('-to-');
  if (!SOURCE[src] || !SINK[snk]) continue;
  const id = `p-${rule}`;
  if (bank.has(id) || existsSync(join(REPO, 'bench/pli', rule))) continue;
  const meta = RULES[rule];
  const job = {
    id, title: `Benchmark pair for ${rule}`, type: 'draft', priority: 46, estimateMin: 4,
    target: [], writableDirs: [`bench/pli/${rule}/`, `bench/pli/${rule}-negative/`], kinds: [], benchRule: rule,
    spec: `Write a benchmark case pair for the PL/I reader: bench/pli/${rule}/ (the positive) and bench/pli/${rule}-negative/ (the near-miss negative), for the flow rule ${rule} (${meta.cwe}, ${meta.sev}): ${meta.text || ''}

The input: ${SOURCE[src]}.
The sink: ${SINK[snk]}.

Positive: the input reaches the sink through one or two assignments (a MOVE-like copy or a concatenation), all in one program.
Negative: the same program, except that the value reaching the sink is a literal or a field the input never reaches, while the input is still read and used elsewhere (for example written with PUT SKIP LIST).

Each directory holds one program, MAIN.pli, plus RUN.jcl when the input says so (30 to 60 lines, ordinary IBM Enterprise PL/I as a bank would write it, no real company, person or credential), and manifest.json: { "id": "<directory name>", "title": "<one sentence>", "cwe": "${meta.cwe}", "expect": [{ "rule": "${rule}", "file": "MAIN.pli" }] } for the positive; the negative has "expect": [] and "negativeFor": ["${rule}"].

${PITFALLS}

The gate reads every statement with the project's PL/I reader and refuses the reply if one does not parse. JCL statements begin with // in columns 1 and 2 and stay within 72 columns. Reply with every file of both directories under its full path.`,
    context: [{ file: 'bench/pli/terminal-input-into-dynamic-sql/MAIN.pli' }],
  };
  if (!args.includes('--dry')) writeFileSync(join(ROOT, 'bank', `${id}.json`), JSON.stringify(job, null, 1));
  console.log(`${args.includes('--dry') ? 'would write' : 'wrote'} ${id}`);
  wrote++;
}
console.log(`${wrote} pair jobs`);
