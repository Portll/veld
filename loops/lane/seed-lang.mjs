// Writes the IMS DBD/PSB and Db2 DDL parser jobs (tracks ims and db2) into bank/.
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(dirname(fileURLToPath(import.meta.url)), 'bank');

const IMS_CONTRACT = `How a macro parser plugs in (lib/ims/read.mjs, shown): your module exports \`parsers\`, an object mapping a macro name to a function (st, ops) => node. st is { name (the label or null), operation, field (the operand text), line, endLine }. ops is parseOperands(st.field) from lib/cards.mjs: { positional: [raw strings], keywords: Map(UPPER-CASE keyword -> raw value string) }; a value may be a sublist '(A,B)' or a quoted string. Use sublist(v) and integer(v) from '../operands.mjs' (shown). Refuse anything you do not recognise — an unknown keyword, a value of the wrong form, a missing required keyword — by throwing new ImsSyntax('what is wrong') imported from '../operands.mjs'. Return a node with at least { kind: <macro> } and the fields the task names; keep every keyword you accept in node.keywords as an object of normalised values so nothing is lost.

Tests (test/ims-<name>.test.mjs) build a statement exactly like this, with the macro starting in column 10 and nothing in column 1:
\`\`\`js
import { readIms, parseImsStatement } from '../lib/ims/read.mjs';
const parse = (src) => parseImsStatement(readIms(src).statements[0]);
// parse('         SEGM  NAME=A,PARENT=0,BYTES=8') -> { kind: 'SEGM', status: 'parsed', node: {...} }
\`\`\`
A statement longer than 71 columns must be split: put a non-blank character in column 72 and continue in column 16 of the next line, as the corpus does.`;

const DB2_CONTRACT = `How a statement parser plugs in (lib/db2/read.mjs, shown): your module exports \`parsers\`, an object mapping a statement kind as classify() names it (e.g. 'CREATE TABLE', 'GRANT') to a function (c, st) => node. c is a cursor (lib/db2/cursor.mjs, shown) on the statement's tokens, positioned at the first word (CREATE, GRANT, ...): consume it. The cursor has exactly the methods shown: peek, next, done, rest, drain, slice, isWord, isOp, word, op, expectWord, expectOp, fail, group, items, until; there is no c.toks. Tokens are { t: 'word', v, u } | { t: 'lit', v, prefix } | { t: 'ident', v } (a "delimited name") | { t: 'num', v } | { t: 'op', v }. A name may be a word or an ident and may be qualified, SCHEMA.NAME: read a qualified name as [parts]. The dispatcher refuses the statement if any token is left, so consume everything you accept and call c.fail('what you expected') on anything you do not recognise — never skip tokens you do not understand. Expressions you do not need to understand (a DEFAULT value, a CHECK condition, a view's fullselect, a routine body) may be taken whole with c.group() or c.until(...) and kept as tokens.

Tests (test/db2-<name>.test.mjs) build a statement exactly like this:
\`\`\`js
import { readDb2, parseDb2Statement } from '../lib/db2/read.mjs';
const parse = (sql) => parseDb2Statement(readDb2(sql).statements[0]);
// parse('CREATE TABLE T (C INT);') -> { kind: 'CREATE TABLE', status: 'parsed', node: {...} }
\`\`\``;

const ims = (id, module, title, priority, kinds, spec) => ({
  id, track: 'ims', title, type: 'draft', priority, estimateMin: 8,
  target: [`lib/ims/macro/${module}.mjs`, `test/ims-${module}.test.mjs`], kinds,
  spec: `Write lib/ims/macro/${module}.mjs (an empty stub today) and test/ims-${module}.test.mjs.\n\n${spec}\n\n${IMS_CONTRACT}`,
  context: [{ file: 'lib/ims/read.mjs' }, { file: 'lib/ims/operands.mjs' }, { file: 'lib/cards.mjs' }, ...kinds.map((k) => ({ samples: k, n: Math.max(5, Math.floor(30 / kinds.length)) }))],
});

const db2 = (id, module, title, priority, kinds, spec) => ({
  id, track: 'db2', title, type: 'draft', priority, estimateMin: 8,
  target: [`lib/db2/stmt/${module}.mjs`, `test/db2-${module}.test.mjs`], kinds,
  spec: `Write lib/db2/stmt/${module}.mjs (an empty stub today) and test/db2-${module}.test.mjs.\n\n${spec}\n\n${DB2_CONTRACT}`,
  context: [{ file: 'lib/db2/read.mjs' }, { file: 'lib/db2/cursor.mjs' }, ...kinds.map((k) => ({ samples: k, n: Math.max(5, Math.floor(30 / kinds.length)) }))],
});

const JOBS = [
  ims('i01-dbd', 'dbd', 'IMS DBD macros: DBD, DATASET, AREA, SEGM, FIELD, LCHILD, XDFLD, DBDGEN', 62, ['DBD', 'DATASET', 'SEGM', 'FIELD', 'LCHILD', 'DBDGEN'], `Parsers for the DBD macros. Keywords, as IBM's IMS system utilities reference gives them (accept these, refuse others):
- DBD: NAME (required), ACCESS = a database type or (type, access method): HDAM, HIDAM, HISAM, HSAM, SHISAM, SHSAM, PHDAM, PHIDAM, PSINDEX, INDEX, LOGICAL, DEDB, MSDB, GSAM with VSAM, OSAM, BSAM; RMNAME = (module[,anchors[,rbn[,bytes]]]); PASSWD = YES|NO; VERSION = 'text'; DATXEXIT = YES|NO; EXIT = (...); PSNAME; GSROOT; ENCODING; FPINDEX. Node { kind: 'DBD', name, access: { type, method }, passwd, rmname: [...] }.
- DATASET: DD1, DD2, OVFLW, DEVICE, BLOCK, SIZE, SCAN, FRSPC, RECORD, MODEL, SEARCHA, REL; the statement may carry a label. Node { kind: 'DATASET', label, dd1, dd2, ovflw, device, size }.
- AREA (DEDB): DD1, MODEL, SIZE, UOW, ROOT, REMOTE.
- SEGM: NAME (required), PARENT = 0, a name, or ((name[,SNGL|DBLE])[,(lpname,VIRTUAL|PHYSICAL,dbd)]); BYTES = n or (max,min); FREQ; POINTER (a word or a sublist); RULES = (...); COMPRTN; EXIT; SOURCE = ((seg,DATA|KEY,dbd)[,...]); SSPTR; TYPE (DEDB). Node { kind: 'SEGM', name, parent: name|null (null for 0), bytes: { max, min }, rules, pointer }.
- FIELD: NAME = name or (name,SEQ[,U|M]); BYTES; START; TYPE = C|X|P|Z|H|F|G (default C). Node { kind: 'FIELD', name, seq: bool, unique: bool|null, bytes, start, type }.
- LCHILD: NAME = (segment[,dbd]); POINTER; PAIR; RULES; INDEX; RKSIZE; PTR.
- XDFLD: NAME, SEGMENT, SRCH, SUBSEQ, DDATA, NULLVAL, CONST, EXTRTN.
- DBDGEN: no operands.
Mark as UNVERIFIED any keyword or value you are not sure IMS accepts.`),
  ims('i02-psb', 'psb', 'IMS PSB macros: PCB, SENSEG, SENFLD, PSBGEN', 62, ['PCB', 'SENSEG', 'PSBGEN'], `Parsers for the PSB macros (accept these keywords, refuse others):
- PCB: TYPE = DB|TP|GSAM; DBDNAME (DB and GSAM); PROCOPT = one or more of the letters G I R D A P O N T E K S L H in any order (A means G, I, R and D); KEYLEN; PROCSEQ; PROCSEQD; POS = S|M|SINGLE|MULTIPLE|P|POSITION; PCBNAME; LIST = YES|NO; SB = NO|COND; VIEW = MSDB; NAME (a TP PCB's destination); LTERM; ALTRESP; SAMETRM; MODIFY; EXPRESS; the label is the PCB's name. Node { kind: 'PCB', label, type, dbd, procopt: [letters as written], keylen, pcbname, pos }.
- SENSEG: NAME (required), PARENT = 0|name, PROCOPT, SSPTR, INDICES. Node { kind: 'SENSEG', name, parent: name|null, procopt: [letters]|null }.
- SENFLD: NAME, START, REPL = YES|NO.
- PSBGEN: LANG = COBOL|PL/I|PLI|ASSEM|C|PASCAL|JAVA; PSBNAME (required); IOASIZE; SSASIZE; CMPAT = YES|NO; MAXQ; IOEROPN; OLIC = YES|NO; GSROLBOK; LOCKMAX. Node { kind: 'PSBGEN', psbname, lang }.
Mark as UNVERIFIED any keyword or value you are not sure IMS accepts.`),
  db2('d01-table', 'table', 'Db2 CREATE TABLE', 64, ['CREATE TABLE'], `The parser for kind 'CREATE TABLE': CREATE [GLOBAL TEMPORARY | AUXILIARY] TABLE name ( element, ... ) [option]... and CREATE TABLE name LIKE other.
- A column: name type [length or (precision[,scale])] then, in any order: NOT NULL; WITH DEFAULT [value]; DEFAULT value; FOR BIT|SBCS|MIXED DATA; CCSID ASCII|EBCDIC|UNICODE|n; GENERATED ALWAYS|BY DEFAULT [AS IDENTITY [( options )]]; [CONSTRAINT name] PRIMARY KEY | UNIQUE | REFERENCES table [(cols)] [ON DELETE RESTRICT|CASCADE|SET NULL|NO ACTION] | CHECK (condition). Types: CHAR, CHARACTER, VARCHAR, CHAR VARYING, CHARACTER VARYING, LONG VARCHAR, GRAPHIC, VARGRAPHIC, CLOB, BLOB, DBCLOB, BINARY, VARBINARY, SMALLINT, INT, INTEGER, BIGINT, DECIMAL, DEC, NUMERIC, REAL, FLOAT, DOUBLE [PRECISION], DECFLOAT, DATE, TIME, TIMESTAMP [WITH|WITHOUT TIME ZONE], ROWID, XML.
- A table constraint: [CONSTRAINT name] PRIMARY KEY (cols) | UNIQUE (cols) | FOREIGN KEY (cols) REFERENCES table [(cols)] [ON DELETE ...] | CHECK (condition).
- Options after the list, any order: IN database.tablespace | IN DATABASE name | IN tablespace; EDITPROC name; VALIDPROC name; AUDIT NONE|CHANGES|ALL; OBID n; DATA CAPTURE NONE|CHANGES; WITH RESTRICT ON DROP; CCSID ASCII|EBCDIC|UNICODE; VOLATILE; APPEND YES|NO; PARTITION BY [RANGE] (cols) (PARTITION n ENDING AT (...), ...); ORGANIZE BY HASH ...; and for Db2 LUW: COMPRESS YES|NO, INDEX IN ts, LONG IN ts, NOT LOGGED INITIALLY, DISTRIBUTE BY HASH (cols). Refuse anything else (MySQL's AUTO_INCREMENT or ENGINE= is not Db2).
Node { kind, name: [parts], columns: [{ name, type, length, scale, notNull, default: tokens|true|null, generated, constraints: [...] }], constraints: [{ type, name, columns, references }], options: { in, editproc, validproc, audit, dataCapture, ... } }. EDITPROC and VALIDPROC name exit programs that run inside Db2: keep them as names.`),
  db2('d02-index', 'index', 'Db2 CREATE INDEX', 60, ['CREATE INDEX'], `The parser for kind 'CREATE INDEX': CREATE [UNIQUE [WHERE NOT NULL]] INDEX name ON table ( column [ASC|DESC|RANDOM], ... ) then, any order: INCLUDE (cols); CLUSTER | NOT CLUSTER; USING STOGROUP name [PRIQTY n] [SECQTY n] [ERASE YES|NO] | USING VCAT name; FREEPAGE n; PCTFREE n; GBPCACHE CHANGED|ALL|NONE; BUFFERPOOL name; CLOSE YES|NO; PARTITIONED; PADDED | NOT PADDED; COPY YES|NO; DEFINE YES|NO; COMPRESS YES|NO; PIECESIZE n K|M|G; and for Db2 LUW: ALLOW REVERSE SCANS | DISALLOW REVERSE SCANS, IN tablespace. Refuse anything else. Node { kind, unique, name: [parts], table: [parts], columns: [{ name, order }], options: {...} }.`),
  db2('d03-grant', 'grant', 'Db2 GRANT and REVOKE', 66, ['GRANT'], `Parsers for kinds 'GRANT' and 'REVOKE': GRANT privilege[, privilege]... [ON [object-type] object[, object]...] TO grantee[, grantee]... [WITH GRANT OPTION] and REVOKE ... FROM grantee[, ...] [BY ALL|BY name[, ...]] [RESTRICT | INCLUDING DEPENDENT PRIVILEGES | NOT INCLUDING DEPENDENT PRIVILEGES].
- Privileges: ALL [PRIVILEGES], SELECT, INSERT, UPDATE [(cols)], DELETE, ALTER, INDEX, REFERENCES [(cols)], TRIGGER, EXECUTE, USAGE, BIND, COPY, CREATEIN, ALTERIN, DROPIN, CONTROL, and database and system privileges: DBADM, DBCTRL, DBMAINT, SYSADM, SYSCTRL, SYSOPR, SECADM, ACCESSCTRL, DATAACCESS, BINDADD, BINDAGENT, CREATETAB, CREATETS, CREATEDBA, CREATEDBC, CREATESG, CREATEALIAS, DISPLAYDB, DROP, IMAGCOPY, LOAD, RECOVERDB, REORG, REPAIR, STARTDB, STATS, STOPDB, MONITOR1, MONITOR2, TRACE, ARCHIVE, BSDS, DISPLAY, RECOVER, STOPALL, STOSPACE, CONNECT, CREATE_NOT_FENCED_ROUTINE, IMPLICIT_SCHEMA, LOAD, QUIESCE_CONNECT.
- Object types: TABLE, DATABASE, PLAN, PACKAGE, COLLECTION, SCHEMA, PROCEDURE, FUNCTION, SPECIFIC FUNCTION, SEQUENCE, VARIABLE, TYPE, DISTINCT TYPE, JAR, USE OF STOGROUP|TABLESPACE|BUFFERPOOL (written GRANT USE OF TABLESPACE db.ts TO ...), and no ON at all for system privileges. A name may be qualified; a package is collection.package; a function or procedure may have (parameter types) after it.
- Grantees: a name, PUBLIC, ROLE name, USER name, GROUP name.
Node { kind, privileges: [{ name, columns }], objectType, objects: [[parts]], grantees: [{ type: 'PUBLIC'|'ROLE'|'USER'|'GROUP'|'NAME', name }], withGrantOption: bool }. The security rules read grantees PUBLIC and withGrantOption, so get those exactly right.`),
  db2('d04-view', 'view', 'Db2 CREATE VIEW', 55, ['CREATE VIEW'], `The parser for kind 'CREATE VIEW': CREATE [OR REPLACE] VIEW name [(column, ...)] AS fullselect [WITH [CASCADED | LOCAL] CHECK OPTION]. Keep the fullselect as tokens (it ends at WITH ... CHECK OPTION or at the end of the statement), and list the tables it reads: every name, qualified or not, after FROM or JOIN at any parenthesis depth, up to the next comma, ON, WHERE, GROUP, ORDER, HAVING, UNION, JOIN keyword or closing parenthesis, skipping a correlation name after it. Node { kind, name: [parts], columns: [names]|null, select: tokens, tables: [[parts]], checkOption: 'CASCADED'|'LOCAL'|null }.`),
  db2('d05-storage', 'storage', 'Db2 CREATE TABLESPACE, DATABASE, STOGROUP and BUFFERPOOL', 54, ['CREATE DATABASE', 'CREATE TABLESPACE', 'CREATE STOGROUP', 'CREATE BUFFERPOOL'], `Parsers for kinds 'CREATE DATABASE' (BUFFERPOOL bp, INDEXBP bp, STOGROUP name, CCSID ASCII|EBCDIC|UNICODE, AS WORKFILE [FOR member]; Db2 LUW forms ON path, USING CODESET x TERRITORY y, COLLATE USING x, AUTOMATIC STORAGE YES|NO), 'CREATE TABLESPACE' (name [IN database], USING STOGROUP name [PRIQTY n] [SECQTY n] [ERASE YES|NO] | USING VCAT name, FREEPAGE, PCTFREE, SEGSIZE, NUMPARTS, MAXPARTITIONS, DSSIZE n G, LOCKSIZE ANY|TABLESPACE|TABLE|PAGE|ROW|LOB, LOCKMAX n|SYSTEM, BUFFERPOOL, CLOSE YES|NO, COMPRESS YES|NO, CCSID, MAXROWS, TRACKMOD, LOG YES|NO, MEMBER CLUSTER, GBPCACHE; Db2 LUW: MANAGED BY SYSTEM|DATABASE|AUTOMATIC STORAGE, USING (...), PAGESIZE n K, EXTENTSIZE n, PREFETCHSIZE n), 'CREATE STOGROUP' (VOLUMES (vol, ...) VCAT name, DATACLAS, MGMTCLAS, STORCLAS, KEY LABEL) and 'CREATE BUFFERPOOL' (Db2 LUW: name [IMMEDIATE|DEFERRED] SIZE n [AUTOMATIC] PAGESIZE n K). Refuse anything else. Nodes { kind, name, options: { ... } }.`),
  db2('d06-routine', 'routine', 'Db2 CREATE PROCEDURE, FUNCTION and TRIGGER', 58, ['CREATE PROCEDURE'], `Parsers for kinds 'CREATE PROCEDURE', 'CREATE FUNCTION' and 'CREATE TRIGGER'.
- CREATE [OR REPLACE] PROCEDURE name [( [IN|OUT|INOUT] param type, ... )] option... [body]. Options in any order: LANGUAGE SQL|COBOL|C|PLI|ASSEMBLE|JAVA|REXX; EXTERNAL [NAME name|'name']; PARAMETER STYLE GENERAL [WITH NULLS]|SQL|DB2SQL|JAVA; DYNAMIC RESULT SETS n; MODIFIES SQL DATA | READS SQL DATA | CONTAINS SQL | NO SQL; [NOT] DETERMINISTIC; WLM ENVIRONMENT name|(name,*); ASUTIME LIMIT n|NO LIMIT; STAY RESIDENT YES|NO; PROGRAM TYPE MAIN|SUB; SECURITY DB2|USER|DEFINER; COLLID name|NO COLLID; COMMIT ON RETURN YES|NO; FENCED|NOT FENCED; RUN OPTIONS 'x'; SPECIFIC name; VERSION v; and the native SQL options such as ISOLATION LEVEL, QUALIFIER, PACKAGE OWNER, VALIDATE BIND|RUN, DEBUG MODE. The body, when LANGUAGE SQL, is an SQL statement or BEGIN ... END: take it whole as tokens (the splitter already keeps it in one statement).
- CREATE FUNCTION name ( params ) RETURNS type | RETURNS TABLE (cols) option... [body]; options as above plus CAST FROM, SOURCE name, RETURN expression.
- CREATE TRIGGER name [NO CASCADE] BEFORE|AFTER|INSTEAD OF INSERT|DELETE|UPDATE [OF cols] ON table [REFERENCING OLD AS x NEW AS y ...] FOR EACH ROW|STATEMENT [MODE DB2SQL] [WHEN (condition)] body.
Node { kind, name: [parts], params: [{ mode, name, type }], language, external: name|null, security, wlm, options: {...}, body: tokens|null }. EXTERNAL NAME names a load module that runs inside Db2: keep it exactly.`),
  db2('d07-misc', 'misc', 'Db2 DROP, COMMENT ON, LABEL ON, SET, RENAME, CREATE SEQUENCE, ALIAS, SCHEMA and ROLE', 52, ['DROP TABLE', 'DROP DATABASE', 'DROP VIEW', 'COMMENT ON', 'CREATE SEQUENCE', 'SET'], `Parsers for kinds 'DROP TABLE', 'DROP VIEW', 'DROP INDEX', 'DROP DATABASE', 'DROP TABLESPACE', 'DROP SCHEMA', 'DROP PROCEDURE', 'DROP FUNCTION', 'DROP SEQUENCE', 'DROP ALIAS', 'DROP SYNONYM', 'DROP TRIGGER', 'DROP STOGROUP', 'DROP ROLE' (DROP object name [RESTRICT|CASCADE]; also Db2 LUW's IF EXISTS is not Db2 for z/OS — refuse it), 'COMMENT ON' (COMMENT ON TABLE|COLUMN|INDEX|... name IS 'text', and COMMENT ON table (col IS 'text', ...)), 'LABEL ON' (likewise), 'SET' (SET CURRENT SQLID|SCHEMA|PATH|DEGREE|PACKAGESET|RULES|PRECISION|TIMEZONE ... = value, SET SCHEMA x, SET PATH = ...), 'RENAME' (RENAME [TABLE] a TO b, RENAME INDEX a TO b), 'TRUNCATE' (TRUNCATE [TABLE] t [DROP|REUSE STORAGE] [IGNORE|RESTRICT WHEN DELETE TRIGGERS] [IMMEDIATE]), 'CREATE SEQUENCE' (name [AS type] START WITH n, INCREMENT BY n, MINVALUE n|NO MINVALUE, MAXVALUE n|NO MAXVALUE, CYCLE|NO CYCLE, CACHE n|NO CACHE, ORDER|NO ORDER, in any order), 'CREATE ALIAS' and 'CREATE SYNONYM' (name FOR table), 'CREATE SCHEMA' (name [AUTHORIZATION owner]) and 'CREATE ROLE' (name). Nodes { kind, ...named fields }.`),
  db2('d08-alter', 'alter', 'Db2 ALTER TABLE', 50, ['ALTER TABLE'], `The parser for kind 'ALTER TABLE': ALTER TABLE name clause [clause]..., clauses: ADD [COLUMN] column-definition (as CREATE TABLE's); ADD [CONSTRAINT name] PRIMARY KEY|UNIQUE|FOREIGN KEY|CHECK ...; ALTER [COLUMN] name SET DATA TYPE type | SET DEFAULT v | DROP DEFAULT | SET NOT NULL; DROP [COLUMN] name [RESTRICT|CASCADE]; DROP PRIMARY KEY | DROP UNIQUE name | DROP FOREIGN KEY name | DROP CONSTRAINT name | DROP CHECK name; AUDIT NONE|CHANGES|ALL; DATA CAPTURE NONE|CHANGES; VALIDPROC name|NULL; ACTIVATE|DEACTIVATE ROW|COLUMN ACCESS CONTROL; VOLATILE|NOT VOLATILE; APPEND YES|NO; ADD PARTITION ...; RENAME COLUMN a TO b. Refuse anything else. Node { kind, name: [parts], clauses: [{ action, ... }] }. Take column definitions and constraints whole as tokens if you cannot reuse CREATE TABLE's parser (it is in a separate module; do not import it).`),
];

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
