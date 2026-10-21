// Writes the PL/I benchmark-pair jobs and the builtin table job into bank/.
import * as PATHS from '../paths.mjs';
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(PATHS.LANE_HOME, 'bank');

const FORMAT = `Each case is a directory holding one or more PL/I source files (.pli) and a manifest.json:
{ "id": "<directory name>", "title": "<one sentence saying what the case shows>", "cwe": "CWE-<n>", "expect": [{ "rule": "<rule id>", "file": "<file name>" }] }
for the positive case, and for the near-miss negative the same with "expect": [] and "negativeFor": ["<rule id>"]. The negative differs from the positive in the one thing that makes the positive a finding (a bound check, a constant instead of input, a different field passed) and is otherwise the same program.

The PL/I must be ordinary IBM Enterprise PL/I as a bank's batch or CICS program would be written: \`NAME: PROCEDURE OPTIONS(MAIN);\` (or OPTIONS(MAIN) for a CICS program with EXEC CICS), DCL statements with level-numbered structures and FIXED BIN, FIXED DEC, CHAR and PIC attributes, EXEC SQL / EXEC CICS statements ending in ';', and END NAME;. Indent every line at least one column (column 1 stays empty). Keep each program short (30 to 80 lines). Comments only where a reader would miss what the case shows. No real company, person or credential.

The automatic gate reads each .pli with the project's PL/I reader and refuses a file with a lexical error or a statement no PL/I statement kind fits, so write only standard statements. Reply with every file of both directories, each under its full path.`;

const pair = (n, slug, title, cwe, rule, pos, neg) => ({
  id: `f${String(n).padStart(2, '0')}-${slug}`, title: `Benchmark pair: ${title}`, type: 'draft', priority: 45, estimateMin: 9,
  target: [], writableDirs: [`bench/pli/${slug}/`, `bench/pli/${slug}-negative/`], kinds: [],
  spec: `Write a benchmark case pair for the PL/I reader: bench/pli/${slug}/ (the positive) and bench/pli/${slug}-negative/ (the near-miss negative). The rule it exercises is ${rule} (${cwe}).\n\nPositive: ${pos}\n\nNegative: ${neg}\n\n${FORMAT}`,
  context: [{ file: 'bench/cases/004-web-input-built-into-sql/manifest.json' }],
});

const JOBS = [
  pair(1, 'terminal-input-into-dynamic-sql', 'terminal input built into dynamic SQL', 'CWE-89', 'cics-terminal-to-dynamic-sql',
    'A CICS program receives a customer name with EXEC CICS RECEIVE INTO(INPUT_AREA) LENGTH(LEN), concatenates it into a CHAR(200) VARYING statement text with ||, and runs EXEC SQL PREPARE STMT FROM :SQL_TEXT; then EXEC SQL EXECUTE STMT;',
    'The same program, but the statement text is a constant with a parameter marker and the input is passed as a host variable: EXEC SQL PREPARE STMT FROM :SQL_TEXT (SQL_TEXT built only from literals) and EXECUTE STMT USING :CUST_NAME.'),
  pair(2, 'based-storage-from-input', 'BASED storage addressed through a pointer the input sets', 'CWE-822', 'pli-based-storage-addressing',
    'A batch program reads a record whose first field is an offset, computes a POINTER with ADDR(BUFFER) + the offset (using POINTERADD or PTRADD), and reads a BASED structure through it.',
    'The same program addressing the BASED structure only through ADDR() of a declared buffer, with no input in the pointer.'),
  pair(3, 'entry-variable-call', 'a call through an entry variable chosen from input', 'CWE-470', 'pli-entry-variable-call',
    'DCL HANDLER ENTRY VARIABLE; a SELECT on an input transaction code assigns one of three declared ENTRY constants to HANDLER, and CALL HANDLER(REC); — plus a fallback assigning an entry FETCHed by a name taken from the input record.',
    'The same program calling each of the three ENTRY constants directly in the SELECT branches, with no entry variable.'),
  pair(4, 'fetch-name-from-input', 'a module loaded by a name from input', 'CWE-494', 'argv-or-env-to-dynamic-program-load',
    'PROCEDURE OPTIONS(MAIN) with a PARM string (DCL PARM CHAR(100) VARYING as the main procedure parameter), the module name taken from it with SUBSTR, and FETCH of an ENTRY whose external name is that variable (DCL MOD ENTRY EXTERNAL; and FETCH MOD TITLE(MODNAME);), then CALL MOD.',
    'The same program with FETCH MOD TITLE(\'PAYCALC\') — a literal title.'),
  pair(5, 'preprocessor-conditional-include', 'source that depends on preprocessor decisions', 'CWE-1164', 'pli-preprocessor-in-use',
    'A program with %DCL DEBUG CHAR; %DEBUG = \'YES\'; %IF DEBUG = \'YES\' %THEN %INCLUDE DBGRTN; and a %INCLUDE of a record layout member, plus the included members as separate .pli files in the same directory (DBGRTN.pli and the layout member).',
    'The same program with the record layout written inline and no % statements at all.'),
  pair(6, 'record-into-sql-update', 'a file record written to the database unchecked', 'CWE-20', 'file-record-to-dynamic-sql',
    'A batch program READ FILE(TRANIN) INTO(TRAN_REC), builds UPDATE text from TRAN_REC fields with ||, and PREPAREs and EXECUTEs it.',
    'The same program running a static EXEC SQL UPDATE ACCOUNT SET BALANCE = :NEW_BAL WHERE ACCT_ID = :ACCT_ID; with host variables.'),
  pair(7, 'commarea-length-from-input', 'a COMMAREA length taken from terminal input', 'CWE-130', 'cics-terminal-to-storage-length',
    'A CICS program RECEIVEs a length field and uses it in EXEC CICS GETMAIN SET(P) FLENGTH(REQ_LEN); and then LINK PROGRAM(\'SUBPGM\') COMMAREA(AREA) LENGTH(REQ_LEN);',
    'The same program with REQ_LEN set from STG(AREA) (a constant size) and the input length only compared against it.'),
  pair(8, 'hardcoded-db-password', 'a database password in the source', 'CWE-798', 'hardcoded-credential',
    'DCL DB_PASSWORD CHAR(8) INIT(\'S3cr3tPw\'); used in EXEC SQL CONNECT TO :DB_NAME USER :DB_USER USING :DB_PASSWORD;',
    'The same program reading DB_USER and DB_PASSWORD from a protected file (READ FILE(CREDS) INTO(CRED_REC)) before the CONNECT.'),
  pair(9, 'dli-delete-unqualified', 'an IMS DLET with no segment search argument', 'CWE-284', 'ims-unqualified-destructive-call',
    'A PL/I IMS program: DCL PLITDLI ENTRY; CALL PLITDLI(THREE, GHU_FUNC, DB_PCB, IO_AREA, SSA1); then CALL PLITDLI(TWO_ARGS ... DLET_FUNC, DB_PCB, IO_AREA) with no SSA after a GHN loop.',
    'The same program issuing DLET only right after a qualified GHU with a fully qualified SSA, and checking the PCB status code first.'),
  pair(10, 'substr-index-from-input', 'SUBSTR positioned by an input value with no range check', 'CWE-125', 'cics-terminal-to-reference-modification',
    'A CICS program RECEIVEs START and LEN as character digits, converts them with FIXED BIN(31) assignment, and moves SUBSTR(BUFFER, START, LEN) to the output map field; the procedure has the (NOSTRINGRANGE): prefix.',
    'The same program testing START > 0 & LEN > 0 & START + LEN - 1 <= LENGTH(BUFFER) before the SUBSTR, and no NOSTRINGRANGE prefix.'),
  pair(11, 'subscript-from-input', 'an array subscript from input with range checking off', 'CWE-129', 'cics-terminal-to-subscript',
    'A CICS program with DCL RATES(12) FIXED DEC(7,4); receives MONTH from the terminal and reads RATES(MONTH); the statement carries a (NOSUBSCRIPTRANGE): prefix.',
    'The same program with IF MONTH < 1 | MONTH > 12 THEN ... RETURN before RATES(MONTH).'),
  pair(12, 'on-error-swallowed', 'errors caught and ignored', 'CWE-390', 'pli-error-condition-ignored',
    'A batch program with ON ERROR BEGIN; ON ERROR SYSTEM; GOTO NEXT_RECORD; END; around a record loop that updates a file, so a failed REWRITE is skipped silently; also ON CONVERSION and ON ZERODIVIDE with null BEGIN; END; units.',
    'The same program whose ON ERROR unit writes a message with PUT SKIP LIST, sets a return code with CALL PLIRETC(16), and stops with STOP;.'),
];

JOBS.push({
  id: 'c03-builtins', title: 'Enterprise PL/I builtin functions, subroutines and pseudovariables', type: 'draft', priority: 55, estimateMin: 10,
  target: ['lib/pli/builtins.mjs', 'test/pli-builtins.test.mjs'], kinds: [],
  spec: `Write lib/pli/builtins.mjs: the builtin functions, builtin subroutines and pseudovariables of IBM Enterprise PL/I, so the reader can tell a reference like SUBSTR(S, 1, 2) or ADDR(X) from an array element when the name is declared BUILTIN or not declared at all. Export BUILTINS, a frozen Map from upper-case name to { category, pseudovariable: bool, abbreviation?: string }, with category one of 'string', 'arithmetic', 'mathematical', 'array', 'storage', 'condition', 'date-time', 'conversion', 'precision', 'integer-manipulation', 'input-output', 'multithreading', 'miscellaneous'. Include at least: ABS ACOS ADD ADDR ADDRDATA ALL ALLOCATION ALLOCSIZE ANY ASIN ATAN ATAND ATANH AVAILABLEAREA BINARY BINARYVALUE BIT BOOL BYTE CEIL CENTER CENTERLEFT CENTERRIGHT CHARACTER CHARGRAPHIC CHARVAL COLLATE COMPARE COMPLEX CONJG COPY COS COSD COSH COUNT CS CURRENTSIZE CURRENTSTORAGE DATAFIELD DATE DATETIME DAYS DAYSTODATE DAYSTOSECS DECIMAL DIMENSION DIVIDE EDIT EMPTY ENDFILE ENTRYADDR EPSILON ERF ERFC EXP EXPONENT FIXED FLOAT FLOOR GAMMA GETENV GRAPHIC HANDLE HBOUND HEX HEXIMAGE HIGH HUGE IAND IEOR IMAG INDEX INOT IOR ISIGNED ISLL ISMAIN ISRL IUNSIGNED LBOUND LEFT LENGTH LINENO LOCATION LOG LOG10 LOG2 LOGGAMMA LOW LOWER LOWERCASE MAX MAXEXP MAXLENGTH MEMINDEX MEMSEARCH MEMVERIFY MIN MINEXP MOD MPSTR MULTIPLY NULL OFFSET OFFSETADD OFFSETDIFF OFFSETSUBTRACT OFFSETVALUE OMITTED ONCHAR ONCODE ONCONDCOND ONCONDID ONCOUNT ONFILE ONGSOURCE ONKEY ONLOC ONSOURCE ONSUBCODE ONWCHAR ONWSOURCE ORDINALNAME ORDINALPRED ORDINALSUCC PACKAGENAME PAGENO PLACES PLIASCII PLIEBCDIC PLIRETV POINTER POINTERADD POINTERDIFF POINTERSUBTRACT POINTERVALUE POLY PRECISION PRED PRESENT PROCEDURENAME PROD RADIX RAISE2 RANDOM RANK REAL REM REPEAT REPLACEBY2 REVERSE RIGHT ROUND SAMEKEY SCALE SEARCH SEARCHR SECS SECSTODATE SECSTODAYS SIGN SIGNED SIN SIND SINH SIZE SOURCEFILE SOURCELINE SQRT STATUS STG STORAGE STRING SUBSTR SUBTRACT SUCC SUM SYSNULL TALLY TAN TAND TANH THREADID TIME TINY TRANSLATE TRIM TRUNC TYPE UNALLOCATED UNSIGNED UNSPEC UPPERCASE UTCDATETIME UTCSECS VALID VALIDDATE VARGLIST VARGSIZE VERIFY VERIFYR WCHARVAL WEEKDAY WHIGH WIDECHAR WLOW Y4DATE Y4JULIAN Y4YEAR and the subroutines PLIRETC PLIDUMP PLISRTA PLISRTB PLISRTC PLISRTD PLITEST PLIREST PLICKPT PLICANC; pseudovariables include SUBSTR UNSPEC STRING REAL IMAG ENTRYADDR ONCHAR ONSOURCE ONGSOURCE TYPE. Abbreviations: ADDRDATA? no; known ones are CHAR for CHARACTER, BIN for BINARY, DEC for DECIMAL, DIM for DIMENSION, PTR for POINTER, PREC for PRECISION, STG for STORAGE, CSTG for CURRENTSTORAGE, ALLOCN for ALLOCATION — map each abbreviation to its own entry carrying abbreviation: the full name. Mark any name you are not sure Enterprise PL/I has as UNVERIFIED under Limits, and leave out names you believe are from another dialect. Export isBuiltin(name) too. Tests: lookups, pseudovariables, abbreviations, and that a few non-builtins (PRINTF, STRLEN, CONCAT) are absent.`,
  context: [],
});

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
