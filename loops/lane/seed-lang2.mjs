// Writes the IMS model and the IMS and Db2 rule jobs, each waiting on the parsers it reads.
import * as PATHS from '../paths.mjs';
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(PATHS.LANE_HOME, 'bank');

const JOBS = [
  {
    id: 'i03-model', track: 'ims', after: ['i01-dbd', 'i02-psb'], title: 'IMS database and PSB models, with the BYTES= self-consistency check', type: 'draft', priority: 60, estimateMin: 9,
    target: ['lib/ims/model.mjs', 'test/ims-model.test.mjs'], kinds: [],
    spec: `Write lib/ims/model.mjs and test/ims-model.test.mjs. Export modelOf(text) which reads one DBD or PSB source with readIms and parseImsStatement (lib/ims/read.mjs, shown) and returns, from the parsed statements in order:
- for a DBD: { kind: 'DBD', name, access, passwd, segments: [{ name, parent: name|null, bytes, line, fields: [{ name, seq, unique, bytes, start, type, line }] }], datasets: [...], problems: [] }. A FIELD belongs to the SEGM before it.
- for a PSB: { kind: 'PSB', name (PSBGEN PSBNAME), lang, pcbs: [{ label, type, dbd, procopt, keylen, sensegs: [{ name, parent, procopt, line }], line }], problems: [] }. A SENSEG belongs to the PCB before it; a SENSEG without its own PROCOPT takes the PCB's.
- problems, the self-consistency check IMS performs at DBDGEN and PSBGEN, each { line, text }: a FIELD that ends past its segment's BYTES (START + BYTES - 1 > segment bytes, using the maximum when BYTES=(max,min)); two FIELDs of one segment marked SEQ; a SEGM whose PARENT names no earlier segment; a PCB KEYLEN shorter than the longest concatenated key along any SENSEG path (sum of the sequence-field lengths from the root to that segment, when the DBD is not available skip this check and say so); a SENSEG whose PARENT names no earlier SENSEG of the PCB; and statements that did not parse (status not 'parsed'), named with their reason.
The parsed node shapes are those lib/ims/macro/dbd.mjs and lib/ims/macro/psb.mjs return (shown); read them, do not assume. Tests: a DBD with two segments whose fields fit, one whose field overruns BYTES, a PSB with two PCBs, a SENSEG inheriting PROCOPT, an unknown parent.`,
    context: [{ file: 'lib/ims/read.mjs' }, { file: 'lib/ims/macro/dbd.mjs' }, { file: 'lib/ims/macro/psb.mjs' }],
  },
  {
    id: 'i04-rules', track: 'ims', after: ['i03-model'], title: 'IMS rules readable from DBD and PSB alone', type: 'draft', priority: 58, estimateMin: 8,
    target: ['lib/ims/rules.mjs', 'test/ims-rules.test.mjs'], kinds: [],
    spec: `Write lib/ims/rules.mjs and test/ims-rules.test.mjs. Export IMS_RULES (id -> { sev, evidence: 'construct'|'coverage', cwe, text }) and checkIms(files) where files is [{ path, text }] of DBD and PSB sources; use modelOf from './model.mjs' (shown). Findings { rule, path, line, detail }:
- 'ims-pcb-procopt-all' (info, construct, CWE-250): a database PCB granted PROCOPT=A (or a set holding G, I, R and D), which lets the program read, insert, replace and delete every sensitive segment; the detail names the PSB, PCB and DBD. This is the inventory half of the least-privilege rule; the other half compares with the DL/I calls a program makes and is not yours.
- 'ims-senseg-unknown-segment' (low, construct, CWE-1076): a SENSEG naming a segment the PCB's DBD does not define, when the DBD is among the files; the PSB will fail PSBGEN or the DBD in the tree is stale.
- 'ims-dbd-self-inconsistent' (low, construct, CWE-1076): each problem modelOf reports for a DBD or PSB.
Tests with in-memory sources for each rule, positive and near-miss.`,
    context: [{ file: 'lib/ims/model.mjs' }],
  },
  {
    id: 'd09-rules', track: 'db2', after: ['d01-table', 'd03-grant', 'd06-routine'], title: 'Db2 rules: grants to PUBLIC, grant option, exit routines, external routines', type: 'draft', priority: 60, estimateMin: 9,
    target: ['lib/db2/rules.mjs', 'test/db2-rules.test.mjs'], kinds: [],
    spec: `Write lib/db2/rules.mjs and test/db2-rules.test.mjs. Export DB2_RULES (id -> { sev, evidence: 'construct', cwe, text }) and checkDb2(path, text) reading the script with readDb2 and parseDb2Statement (lib/db2/read.mjs, shown) and returning findings { rule, path, line, detail }:
- 'db2-grant-to-public' (high, CWE-732): a GRANT whose grantees include PUBLIC; the detail names the privileges and objects. Raise to crit when a privilege is a write (INSERT, UPDATE, DELETE, ALTER, ALL) or a system or database authority (SYSADM, DBADM, ...).
- 'db2-grant-with-grant-option' (med, CWE-269): WITH GRANT OPTION, naming the grantee and privilege.
- 'db2-system-authority-granted' (med, CWE-250): a grant of SYSADM, SYSCTRL, DBADM, DBCTRL, SECADM, ACCESSCTRL or DATAACCESS to anyone.
- 'db2-exit-routine-declared' (info, CWE-829): CREATE TABLE with EDITPROC or VALIDPROC, and FIELDPROC on a column if the node has it, naming the program that runs inside Db2.
- 'db2-external-routine' (info, CWE-829): CREATE PROCEDURE or FUNCTION with EXTERNAL NAME, naming the load module and its SECURITY and WLM ENVIRONMENT.
The node shapes are those lib/db2/stmt/grant.mjs, table.mjs and routine.mjs return (shown); read them, do not assume. Tests for each rule, positive and near-miss (a GRANT to a named user is not db2-grant-to-public).`,
    context: [{ file: 'lib/db2/read.mjs' }, { file: 'lib/db2/stmt/grant.mjs' }, { file: 'lib/db2/stmt/table.mjs' }, { file: 'lib/db2/stmt/routine.mjs' }],
  },
];

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
