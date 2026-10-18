// Writes the HLASM statement-parser jobs (track hlasm, worktree cobolwork-hlasm-20261003, feat/hlasm)
// into bank/. Spec: cobolwork evaluations/spec_2026-10-03_hlasm-track.md.
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(dirname(fileURLToPath(import.meta.url)), 'bank');
const REF = '/Users/portll/.claude/scratch/lang-qwen/ibm-hlasm';
const MACREF = '/Users/portll/.claude/scratch/lang-qwen/ibm-zos-macros';

const CONTRACT = `How a parser plugs in (lib/hlasm/read.mjs): your module exports \`parsers\`, an object mapping an operation name to a function. The dispatcher calls it as parser(st) for an assembler instruction and parser(st, ops) for a system macro. st is { name (the label in column 1, or null), operation (upper case), field (the operand text with continuation cards already joined and remarks removed), line, endLine }. ops is macroOperands(st.field) from lib/hlasm/operands.mjs: { positional: [raw strings, '' for an omitted one], keywords: Map(UPPER-CASE keyword -> raw value string) }. For an assembler instruction split st.field yourself with splitOperands from '../operands.mjs' (it respects quotes, parentheses and L'X attribute references) and read each expression with parseExpression(text), an address with parseAddress(text), from '../expr.mjs': they throw HlasmSyntax on bad input, which is a refusal and fine to let through. NAME (a RegExp for an ordinary symbol) and HlasmSyntax are exported by '../operands.mjs'. Refuse anything you do not recognise - a wrong operand count, an unknown keyword or value, a missing required operand - by throwing new HlasmSyntax('what is wrong'). Return a node { kind: <operation>, ...the fields the task names }. Do not import anything else.

Tests (test/hlasm-<name>.test.mjs) build a statement exactly like this:
\`\`\`js
import { readHlasmStatements, parseHlasmStatement } from '../lib/hlasm/read.mjs';
const parse = (src) => parseHlasmStatement(readHlasmStatements(src).statements[0]);
// parse('LABEL    CSECT') -> { kind: 'CSECT', status: 'parsed', node: {...} }
// parse('         USING *,12') -> { kind: 'USING', status: 'parsed', node: {...} }
// a refusal -> { kind: 'USING', status: 'unparsed', reason: '...' }
\`\`\`
Rules every draft follows:
- Check keyword NAMES against IBM's list; keep keyword VALUES as the raw text written and do not validate them, except where the task names a field to normalise. A value may be a parenthesised sublist such as (2,11), a register in parentheses such as (1), a quoted string, a name or an expression: accept every one of these.
- Positional operands come first. A bare word IBM lists as an option (VL, T, CT, OV, DUMP, STEP and the like) is a positional operand: accept the ones the syntax shows, after the macro's other positional operands.
- Keep the module under 200 lines. Do not write a table of every value IBM lists.
- Every test source line fits in 71 columns. Count them.
- A prototype is only read as one when it is the statement after MACRO. A test of a prototype or a model statement builds the whole definition and picks its statement, for example readHlasmStatements('         MACRO\n&L       MYMAC &A\n&L       MVC   &A,X\n         MEND').statements[1].
Card columns: a label starts in column 1; otherwise column 1 is blank. The operation starts in column 10 and the operands in column 16, as above. Nothing may pass column 71. A statement too long for one card puts a non-blank character in column 72 and continues in column 16 of the next card. Assert on status and on node fields, and include at least one refused statement per operation group.`;

const ctxFiles = (...extra) => [{ file: 'lib/hlasm/read.mjs' }, { file: 'lib/hlasm/operands.mjs' }, { file: 'lib/hlasm/expr.mjs' }, ...extra.map((f) => ({ file: f }))];

const asm = (id, module, title, priority, kinds, spec, ref, extra = {}) => ({
  id, track: 'hlasm', title, type: 'draft', priority, estimateMin: 10,
  target: [`lib/hlasm/asm/${module}.mjs`, `test/hlasm-${module}.test.mjs`], kinds,
  spec: `Write lib/hlasm/asm/${module}.mjs (an empty stub today) and test/hlasm-${module}.test.mjs.\n\n${spec}\n\n${CONTRACT}`,
  context: [...(extra.smallContext ? [{ file: 'lib/hlasm/operands.mjs' }] : ctxFiles()), ...kinds.map((k) => ({ samples: k, n: Math.max(4, Math.floor(24 / kinds.length)) })),
    ...(ref ? [{ reference: `${REF}/${ref}.txt`, title: `IBM HLASM V1R6 Language Reference SC26-4940-09, the ${ref} instructions`, max: extra.refMax || 16000 }] : [])],
  ...extra.job,
});

const JOBS = [
  asm('h01-sections', 'sections', 'HLASM section and linkage instructions: CSECT, DSECT, RSECT, START, COM, LOCTR, ENTRY, EXTRN, WXTRN, AMODE, RMODE, ALIAS, XATTR, CATTR, DXD, CXD', 70,
    ['CSECT', 'DSECT', 'START', 'ENTRY', 'AMODE', 'RMODE', 'RSECT'], `Parsers for these assembler instructions, as the HLASM Language Reference gives them (accept these forms, refuse others):
- CSECT, DSECT, RSECT, COM: no operand. The label (st.name) names the section and may be absent (an unnamed section) for CSECT, RSECT and COM, but a DSECT needs a name. Node { kind, name: st.name or null }.
- START: an optional operand, an absolute expression (the initial location counter value). Node { kind: 'START', name, origin: expression tree or null }.
- LOCTR: a label naming the location counter, required; no operand. Node { kind: 'LOCTR', name }.
- ENTRY, EXTRN, WXTRN: one or more operands, each a name (NAME), separated by commas. EXTRN and WXTRN also accept PART(name[,name]...) as an operand. Node { kind, names: [upper-case names], parts: [names given in PART(...)] }.
- AMODE: one operand: 24, 31, 64, ANY, ANY31 or ANY64. RMODE: one operand: 24, 31, 64 or ANY. The label, when present, names the section. Node { kind, name, mode: the value as written, upper case }.
- ALIAS: one operand, C'string' or X'hex' (the alias for the external symbol named by the label, required). Node { kind: 'ALIAS', name, alias: { type: 'C'|'X', text } }.
- XATTR: one or more keyword operands from ATTRIBUTES(label), LINKAGE(OS|XPLINK), PSECT(name), REFERENCE(DIRECT|INDIRECT|DATA|CODE), SCOPE(SECTION|MODULE|LIBRARY|IMPORT|EXPORT); label required. Node { kind: 'XATTR', name, attributes: { KEYWORD: value } }.
- CATTR: operands each a keyword with or without a parenthesised value from ALIGN(n), DEFLOAD, EXECUTABLE, FILL(n), NOLOAD, NOTEXECUTABLE, NOTREUSABLE, PART(name), PRIORITY(n), READONLY, REFRESH, RENT, REUSABLE, RMODE(24|31|ANY|64); label required. Node { kind: 'CATTR', name, attributes: { KEYWORD: value or true } }.
- DXD: label required, one DC-like operand (type, optional length): take it whole as text. CXD: label optional, no operand. Nodes { kind, name, operand? }.
Mark as UNVERIFIED anything you are not sure HLASM accepts.`, 'sections'),

  asm('h02-symbols', 'symbols', 'HLASM symbol and addressing instructions: EQU, ORG, USING, DROP, LTORG, CNOP, OPSYN, END, PUSH, POP', 72,
    ['EQU', 'USING', 'ORG', 'DROP', 'LTORG', 'END', 'CNOP'], `Parsers for these assembler instructions (accept these forms, refuse others):
- EQU: label required. Operands: value expression (required); then optional length attribute (absolute expression), type attribute (absolute expression, often C'X'), program type, assembler type (one of AR, CR, CR32, CR64, FPR, GR, GR32, GR64, VR). An omitted middle operand is an empty string between commas. Node { kind: 'EQU', name, value: expr, length: expr|null, type: expr|null, programType: expr|null, assemblerType: string|null }.
- ORG: no operand (back to the high-water mark), or an expression, or expression,boundary[,offset] where boundary is a power of 2 from 2 to 4096 written as a decimal number. Node { kind: 'ORG', value: expr|null, boundary: number|null, offset: expr|null }.
- USING: base,register[,register]... (ordinary); (base,end),register[,register]... (with a range); label USING base,register (labeled, st.name is the qualifier); base,relocatable-expression (dependent: the second operand is not a register but an address expression). Registers are absolute expressions (numbers or symbols such as R12). Node { kind: 'USING', label: st.name|null, base: expr, end: expr|null, registers: [expr] }.
- DROP: no operand, or one or more registers or USING labels. Node { kind: 'DROP', operands: [expr] }.
- LTORG: no operand. Node { kind: 'LTORG', name }.
- CNOP: two absolute expressions, byte (0,2,4,...) and boundary (4, 8 or 16): CNOP 0,4 or CNOP 2,8. Node { kind: 'CNOP', byte: expr, boundary: expr }.
- OPSYN: label is the new operation name; no operand (the label's operation is removed) or one operand naming an existing operation. Node { kind: 'OPSYN', name, operation: string|null }.
- END: no operand, or an entry-point expression, and an optional second operand (a language or translator identification in parentheses, kept as text). Node { kind: 'END', entry: expr|null }.
- PUSH and POP: one or more of PRINT, USING, ACONTROL, and for PUSH/POP optionally NOPRINT as the last operand. Node { kind, what: [names] }.
Mark as UNVERIFIED anything you are not sure HLASM accepts.`, 'symbols', { refMax: 20000 }),

  asm('h03-listing', 'listing', 'HLASM listing and output instructions: PRINT, TITLE, EJECT, SPACE, CEJECT, MNOTE, COPY, PUNCH, REPRO, ICTL, ISEQ', 60,
    ['TITLE', 'SPACE', 'PRINT', 'EJECT', 'COPY'], `Parsers for these assembler instructions (accept these forms, refuse others):
- PRINT: one or more of ON, OFF, GEN, NOGEN, DATA, NODATA, MCALL, NOMCALL, MSOURCE, NOMSOURCE, UHEAD, NOUHEAD, NOPRINT (any case). Node { kind: 'PRINT', options: [upper-case names] }.
- TITLE: one operand, a quoted string ('...' with '' for a quote); the label may name the listing. Node { kind: 'TITLE', name, text }.
- EJECT: no operand. SPACE and CEJECT: no operand or one absolute expression. Nodes { kind, value: expr|null }.
- MNOTE: [severity,]'message' where severity is an absolute expression or *. Node { kind: 'MNOTE', severity: expr|'*'|null, text }.
- COPY: one operand, a member name (NAME). Node { kind: 'COPY', member }.
- PUNCH and REPRO: PUNCH takes one quoted string; REPRO takes no operand. Nodes { kind, text? }.
- ICTL: one to three decimal numbers (begin[,end[,continue]]). ISEQ: no operand, or two decimal numbers (left,right). Nodes { kind, values: [numbers] }.
Mark as UNVERIFIED anything you are not sure HLASM accepts.`, null, { smallContext: true, job: { provider: 'groq', model: 'qwen/qwen3.8-27b', maxTokens: 3500 } }),
];

const DFSMS = new Set(['OPEN', 'CLOSE', 'DCB', 'GET', 'PUT', 'READ', 'WRITE', 'RACROUTE']);
const macroRef = (m) => ({ reference: `${MACREF}/${m}.txt`, title: `IBM z/OS 3.1 reference for the ${m} macro (standard form: syntax and parameters)`, ...(DFSMS.has(m) ? {} : { from: '\nSyntax\n' }), max: 9000 });

const macro = (id, module, title, priority, macros, spec, extra = {}) => ({
  id, track: 'hlasm', title, type: 'draft', priority, estimateMin: 12,
  target: [`lib/hlasm/macro/${module}.mjs`, `test/hlasm-${module}.test.mjs`], kinds: macros,
  spec: `Write lib/hlasm/macro/${module}.mjs (an empty stub today) and test/hlasm-${module}.test.mjs: a parser for each of ${macros.join(', ')} in its standard form, as the IBM excerpts below give it. Accept exactly the positional operands and the keywords the IBM syntax lists for the macro, with the value forms it shows (a name, a register in parentheses such as (1), an address expression, a quoted string, a parenthesised list). Refuse any other keyword. The list, execute and modify forms are written with MF=L or MF=(E,addr) or MF=(M,addr): accept MF= with those values and record it. Values are kept as the raw text the operand gives, except where the task names a field to normalise. Keep every keyword you accept in node.keywords as { KEY: raw value } so nothing is lost.

${spec}

${CONTRACT}`,
  context: [{ file: 'lib/hlasm/read.mjs' }, { file: 'lib/hlasm/operands.mjs' }, ...macros.map((k) => ({ samples: k, n: 4 })), ...macros.map(macroRef)],
  ...extra,
});

JOBS.push(
  macro('h10-program', 'program', 'z/OS program-management macros: LINK, XCTL, LOAD, ATTACH', 76, ['LINK', 'XCTL', 'LOAD', 'ATTACH'],
    `Node { kind, ep: the module name when EP= names one (upper case), eploc: the raw EPLOC= value or null (an address holding the name: the name is decided at run time), de: raw DE= value or null, dcb: raw DCB= value or null, param: [raw entries of PARAM=(...)] or null, keywords }. EP=, EPLOC= and DE= are alternatives: refuse a statement that gives more than one of them. The security rules read ep and eploc, so get those exactly right.`),
  macro('h11-linkage', 'linkage', 'z/OS linkage macros: CALL, SAVE, RETURN', 74, ['CALL', 'SAVE', 'RETURN'],
    `- CALL: first positional operand is the entry name or (15) for a register; then an optional parenthesised parameter list; keywords VL, ID=, LINKINST=, PLIST4=, PLIST8=, PARAM64=, MF=. Node { kind: 'CALL', entry: name or null, register: raw text when written as (reg) or null, params: [raw entries], vl: bool, keywords }.
- SAVE: (reg1,reg2) or (reg1) first, then optional T, then an optional identifier (a name, * or a quoted string). Node { kind: 'SAVE', registers: [raw], t: bool, id: raw or null }.
- RETURN: optional (reg1[,reg2]), optional T, then keywords RC= (a number or a register in parentheses) and RETREGS=. Node { kind: 'RETURN', registers: [raw], t: bool, rc: raw or null, keywords }.`),
  macro('h12-storage', 'storage', 'z/OS virtual-storage macros: STORAGE, GETMAIN, FREEMAIN', 72, ['STORAGE', 'GETMAIN', 'FREEMAIN'],
    `- STORAGE: first positional operand OBTAIN or RELEASE (required), then keywords. Node { kind: 'STORAGE', request: 'OBTAIN'|'RELEASE', length: raw LENGTH= or null, address: raw ADDR= or null, subpool: raw SP= or null, key: raw KEY= or null, cond: raw COND= or null, keywords }.
- GETMAIN and FREEMAIN: first positional operand the request type as IBM lists it (R, RU, RC, VU, VC, EU, EC, LU, LC and the others the syntax gives), then keywords. Node { kind, mode: upper-case request type, length: raw LV= or LA= or null, address: raw A= or null, subpool: raw SP= or null, keywords }.`),
  macro('h13-authorization', 'authorization', 'z/OS authorization macros: MODESET, TESTAUTH, RACROUTE', 80, ['MODESET', 'TESTAUTH', 'RACROUTE'],
    `- MODESET: keywords KEY=, MODE=, EXTKEY=, SAVEKEY=, WORKREG= and the others IBM lists. Node { kind: 'MODESET', key: upper-case KEY= value or raw register form, mode: upper-case MODE= value or null, extkey: raw EXTKEY= or null, keywords }. The rules read key=ZERO, mode=SUP and extkey=ZERO: normalise those values to upper case exactly.
- TESTAUTH: keywords FCTN=, STATE=, KEY=, RBLEVEL=, BRANCH= and the others IBM lists. Node { kind: 'TESTAUTH', keywords }.
- RACROUTE: REQUEST= is required: one of AUTH, FASTAUTH, DEFINE, DIRAUTH, EXTRACT, LIST, SIGNON, STAT, TOKENBLD, TOKENMAP, TOKENXTR, VERIFY, VERIFYX, AUDIT; refuse another value. Its other keywords depend on the request and are documented in other chapters: accept any keyword whose name is an ordinary symbol and record it. Node { kind: 'RACROUTE', request, class: raw CLASS= or null, entity: raw ENTITY= or null, attr: raw ATTR= or null, keywords }. List under Limits, as UNVERIFIED, that request-specific keywords are not checked.`),
  macro('h14-operator', 'operator', 'z/OS operator-communication macros: WTO, WTOR', 70, ['WTO', 'WTOR'],
    `- WTO: the message as a quoted string first ('text', or a list of quoted lines in parentheses for a multi-line message), or TEXT=; keywords ROUTCDE=, DESC=, MCSFLAG=, MSGTYP=, CONSID=, CONSNAME=, MF= and the others IBM lists. Node { kind: 'WTO', text: the message text or null, keywords }.
- WTOR: the message first, then positional reply address, reply length and ECB address (all three required in the standard form unless TEXT= is used), then keywords. Node { kind: 'WTOR', text, reply: raw reply address, replyLength: raw, ecb: raw, keywords }. A WTOR reply is operator input to the program: keep reply exactly as written.`),
  macro('h15-datasets', 'datasets', 'z/OS data-set macros: OPEN, CLOSE, DCB (QSAM and BSAM)', 70, ['OPEN', 'CLOSE', 'DCB'],
    `- OPEN and CLOSE: the first positional operand is a parenthesised list of pairs: (dcb[,(options)],dcb[,(options)]...), where options are INPUT, OUTPUT, UPDAT, EXTEND, OUTIN, INOUT, RDBACK, and for CLOSE REREAD, LEAVE, REWIND, FREE, DISP; then keywords MODE=24|31, TYPE=J (OPEN) or TYPE=T (CLOSE), MF=. Node { kind, dcbs: [{ dcb: raw address, options: [upper case] }], keywords }.
- DCB: keywords only, as the QSAM and BSAM DCB syntax lists them (DDNAME=, DSORG=, MACRF=, RECFM=, LRECL=, BLKSIZE=, EODAD=, EXLST=, SYNAD=, BUFNO= and the rest). The label names the DCB. Node { kind: 'DCB', name: st.name, ddname: upper-case DDNAME= or null, dsorg, macrf: raw, recfm, lrecl, blksize, keywords }. DDNAME links the program to a JCL DD statement: get it exactly right.`),
  macro('h16-io', 'io', 'z/OS record macros: GET, PUT (QSAM), READ, WRITE (BSAM)', 66, ['GET', 'PUT', 'READ', 'WRITE'],
    `- GET and PUT (QSAM): dcb address first, then an optional area address (move mode); either may be a register in parentheses. Node { kind, dcb: raw, area: raw or null }.
- READ and WRITE (BSAM/BPAM): decb name, type (SF, SB for READ; SF for WRITE, and the others the syntax lists), dcb address, area address, then optional length ('S' or a value) and further operands as listed; then MF=. Node { kind, decb: raw, type: upper case, dcb: raw, area: raw, length: raw or null, keywords }.`),
  macro('h17-recovery', 'recovery', 'z/OS recovery and dump macros: ESTAE, ESPIE, ABEND, SNAP', 64, ['ESTAE', 'ESPIE', 'ABEND', 'SNAP'],
    `- ESTAE: the exit address or 0 first, then optional CT or OV, then keywords as the ESTAEX syntax below lists them (ESTAE takes the same operands except those IBM marks as ESTAEX only; list those under Limits as UNVERIFIED). Node { kind: 'ESTAE', exit: raw, keywords }.
- ESPIE: first positional SET, RESET or TEST, then the operands that action takes. Node { kind: 'ESPIE', action, operands: [raw], keywords }.
- ABEND: the completion code (decimal, hex or a register), then optional REASON=, then positional DUMP, STEP, and keywords DUMPOPT=, DUMPOPX=, SYSTEM/USER forms as listed. Node { kind: 'ABEND', code: raw, dump: bool, step: bool, keywords }.
- SNAP: keywords DCB=, ID=, SDATA=, PDATA=, STORAGE=, LIST=, TCB= and the others listed (the SNAPX syntax applies). Node { kind: 'SNAP', dcb: raw or null, keywords }.`),
  asm('h03a-listing', 'listing', 'HLASM listing instructions: PRINT, TITLE, EJECT, SPACE, CEJECT', 62, ['TITLE', 'SPACE', 'PRINT', 'EJECT'], `Parsers for these assembler instructions (accept these forms, refuse others):
- PRINT: one or more of ON, OFF, GEN, NOGEN, DATA, NODATA, MCALL, NOMCALL, MSOURCE, NOMSOURCE, UHEAD, NOUHEAD, NOPRINT. Node { kind: 'PRINT', options: [upper-case names] }.
- TITLE: one operand, a quoted string ('...' with '' for a quote); the label may name the listing. Node { kind: 'TITLE', name, text }.
- EJECT: no operand. SPACE and CEJECT: no operand or one absolute expression. Nodes { kind, value: expr|null }.
Keep the module under 80 lines and the test under 60 lines.`, null, { smallContext: true, job: { provider: 'groq', model: 'qwen/qwen3.8-27b', maxTokens: 3500 } }),
  asm('h03b-output', 'output', 'HLASM output instructions: MNOTE, COPY, PUNCH, REPRO, ICTL, ISEQ', 58, ['COPY'], `Parsers for these assembler instructions (accept these forms, refuse others):
- MNOTE: [severity,]'message' where severity is an absolute expression or *. Node { kind: 'MNOTE', severity: expr|'*'|null, text }.
- COPY: one operand, a member name (NAME). Node { kind: 'COPY', member }.
- PUNCH: one quoted string. REPRO: no operand. Nodes { kind, text? }.
- ICTL: one to three decimal numbers (begin[,end[,continue]]). ISEQ: no operand, or two decimal numbers. Nodes { kind, values: [numbers] }.
Keep the module under 80 lines and the test under 60 lines.`, null, { smallContext: true, job: { provider: 'groq', model: 'qwen/qwen3.8-27b', maxTokens: 3500 } }),
);

JOBS.push(
  {
    id: 'h20-model', track: 'hlasm', title: 'HLASM macro prototypes, model statements and substituted open-code statements', type: 'draft', priority: 78, estimateMin: 14,
    target: ['lib/hlasm/model.mjs', 'test/hlasm-model.test.mjs'], kinds: ['MODEL', 'PROTOTYPE', 'SUBSTITUTED'],
    spec: `Write lib/hlasm/model.mjs (an empty stub today) and test/hlasm-model.test.mjs. The dispatcher (lib/hlasm/read.mjs) calls parsers.PROTOTYPE(st) for the statement after MACRO, parsers.MODEL(st) for every other statement inside a macro definition (except MACRO, MEND and conditional assembly, which it handles itself), and parsers.SUBSTITUTED(st) for an open-code statement that holds a variable symbol (&NAME). Nothing here expands a macro; the point is to read each model statement as the statement it is.
- PROTOTYPE: the name field is a variable symbol (&LABEL), or empty; the operation is the macro's name (an ordinary symbol, NAME); the operands are positional parameters (&P) and keyword parameters (&K= with a default value, possibly empty), separated by commas. Node { kind: 'PROTOTYPE', macro, label: '&NAME' or null, positional: ['&P', ...], keywords: { '&K': 'default as written' } }. Refuse an operand that is not a variable symbol or a variable symbol with =.
- MODEL and SUBSTITUTED: replace every variable symbol in the name, operation and operand fields with a placeholder, then parse the result with parseHlasmStatement imported from './read.mjs' (an import cycle that is safe because it is only used inside the function). A variable symbol is & followed by a name, optionally followed by a subscript in parentheses ((1), (&I)) and an optional period that concatenates it to what follows (&A.B reads as one word). &&, inside or outside quotes, is a literal ampersand, not a variable symbol. In the operand field replace each variable symbol (with its subscript and period) by the ordinary symbol VAR1, VAR2, ... (one per occurrence), joined to any characters it was concatenated with; inside a quoted string replace it by the letter X. A name field that is a variable symbol or a sequence symbol (.NAME) is dropped. If the operation field holds a variable symbol, refuse the statement: what it is is decided at expansion. Then: if parseHlasmStatement on the substituted statement (with inMacro: false and prototype: false) returns status 'parsed', return { kind, as: that kind, node: its node }; if it returns 'unparsed' or 'unknown', throw HlasmSyntax with its reason; if 'unbuilt', throw HlasmSyntax('no parser yet for ' + its kind).
${CONTRACT}`,
    context: [{ file: 'lib/hlasm/read.mjs' }, { file: 'lib/hlasm/operands.mjs' }, { file: 'lib/hlasm/expr.mjs' }, { samples: 'MODEL', n: 14 }, { samples: 'PROTOTYPE', n: 6 }, { samples: 'SUBSTITUTED', n: 6 }],
  },
  {
    id: 'h21-exec', track: 'hlasm', title: 'EXEC CICS, EXEC SQL and EXEC DLI in assembler', type: 'draft', priority: 60, estimateMin: 8,
    target: ['lib/hlasm/exec.mjs', 'test/hlasm-exec.test.mjs'], kinds: ['EXEC CICS', 'EXEC SQL'], provider: 'groq', model: 'qwen/qwen3.8-27b', maxTokens: 3500,
    spec: `Write lib/hlasm/exec.mjs (an empty stub today) and test/hlasm-exec.test.mjs: parsers keyed 'EXEC CICS', 'EXEC SQL' and 'EXEC DLI', each called as parser(st) with st.field holding everything after EXEC (for example "CICS SEND MAP('M1') MAPSET('S1') ERASE"). Reuse, do not rewrite, the COBOL-side readers:
- import { cicsOptions, cicsCommand } from '../../precompile-cics.mjs': cicsOptions(text) returns [{ word, arg? }] for the text after CICS (word upper case; arg the text inside the option's parentheses); cicsCommand(options) returns the command's table name (e.g. 'SEND MAP') or null when the table does not know it.
- import { hostVariableRoles } from '../../embedded-sql.mjs': hostVariableRoles(sql) returns the host variables an SQL statement names with their roles.
'EXEC CICS': node { kind: 'EXEC CICS', command: the table name, options: the cicsOptions list }; refuse when cicsCommand gives null. 'EXEC SQL': node { kind: 'EXEC SQL', sql: the text after SQL, hostVariables: hostVariableRoles(sql) }; refuse an empty statement. 'EXEC DLI': node { kind: 'EXEC DLI', options: cicsOptions(text after DLI) }; refuse an empty one. Keep the module under 60 lines and the test under 60 lines.
${CONTRACT}`,
    context: [{ file: 'lib/hlasm/operands.mjs' }, { samples: 'EXEC CICS', n: 6 }, { samples: 'EXEC SQL', n: 2 }],
  },
);

// Second versions of the jobs that failed three rounds, under the rules above.
const V2 = {
  'h01-sections': 'A lone comma as the whole operand field (CSECT , remark) already reaches you as an empty field.',
  'h02-symbols': '',
  'h10-program': '',
  'h11-linkage': 'CALL is written CALL entry,(param,param,...),VL,MF=(E,list): the entry, then the parameter list in parentheses, then VL, then keywords. Accept exactly that order.',
  'h14-operator': 'ROUTCDE=(2,11) and DESC=(6) are sublists: keep them raw. A WTO message continued across cards arrives already joined.',
  'h15-datasets': '',
  'h17-recovery': 'SNAP STORAGE=(start,end) lists pairs of addresses, which may be registers: keep it raw.',
  'h20-model': '',
  'h03b-output': '',
  'h21-exec': '',
};
for (const [id, note] of Object.entries(V2)) {
  const j = JOBS.find((x) => x.id === id);
  const copy = { ...j, id: `${id}-v2`, spec: note ? `${j.spec}\n\nAlso: ${note}` : j.spec };
  if (copy.provider === 'groq') Object.assign(copy, { model: 'openai/gpt-oss-120b', reasoning: 'low', maxTokens: 3500 });
  JOBS.push(copy);
}

// Third versions: the Groq drafts imported a module that does not exist in place of pin-machine, and
// EXEC statements now reach their parser with every word after EXEC.
const PIN = "The test file's imports are exactly: import { test } from 'node:test'; import assert from 'node:assert/strict'; import { readHlasmStatements, parseHlasmStatement } from '../lib/hlasm/read.mjs'; import './pin-machine.mjs'; and nothing else.";
for (const [id, note] of Object.entries({
  'h03b-output': PIN,
  'h21-exec': `${PIN} st.field holds every word after EXEC, blanks included, with continuation cards joined: 'SQL UPDATE DEPT SET MGRNO = :MGR WHERE DEPTNO = :D' or "CICS SEND MAP('M1') MAPSET('S1') ERASE".`,
})) {
  const j = JOBS.find((x) => x.id === id);
  JOBS.push({ ...j, id: `${id}-v3`, spec: `${j.spec}\n\nAlso: ${note}`, model: 'openai/gpt-oss-120b', reasoning: 'low', maxTokens: 3000 });
}

// Benchmark pairs for the HLASM rules: a positive and a near-miss negative, each one HLASM program.
// The gate assembles both with z390 (macros z390 lacks are stubbed) and scores them with the bench.
const PAIR = (rule, sev, cwe, what) => `Write a benchmark pair for the cobolwork rule ${rule} (${sev}, ${cwe}): ${what}

Two case directories, each holding one HLASM program and a manifest.json, written exactly like the example case shown (bench/cases/104-...):
- the positive case: manifest { "id": <its directory name>, "title": <one sentence saying what the program does that the rule reports>, "cwe": "${cwe}", "expect": [{ "rule": "${rule}", "file": <the .asm file name> }] };
- the negative case, a near miss: the same kind of program, changed only so the rule must not report it; manifest { "id", "title" (what it does and why that is not the reported thing), "cwe": "${cwe}", "expect": [], "negativeFor": ["${rule}"] }.
HLASM card rules: a label starts in column 1, the operation in column 10, operands in column 16; nothing past column 71; nothing in columns 72-80. A program is NAME CSECT, then its statements, then END. Use plain register numbers (14,12,12(13)), machine instructions, DC, DS, EQU, USING and z/OS macros. Keep each program under 25 lines. The .asm file is named after its CSECT. The gate assembles each program with z390 (z/OS macros z390 lacks are stubbed, so it checks everything except their operands) and runs the benchmark: the positive must report ${rule} once in its file and nothing else from any rule; the negative must report nothing.`;
const pair = (id, rule, sev, cwe, what, dirs, groq) => ({
  id, track: 'hlasm', title: `Benchmark pair for ${rule}: ${dirs[0]}`, type: 'draft', priority: 56, estimateMin: 6,
  writableDirs: dirs.map((d) => `bench/cases/${d}/`), target: dirs.map((d) => `bench/cases/${d}/manifest.json`),
  spec: `${PAIR(rule, sev, cwe, what)}\n\nThe directories are bench/cases/${dirs[0]}/ (positive) and bench/cases/${dirs[1]}/ (negative).`,
  context: [{ file: 'bench/cases/104-an-assembler-stub-switches-to-key-zero/SETKEY0.asm' }, { file: 'bench/cases/104-an-assembler-stub-switches-to-key-zero/manifest.json' }, { file: 'bench/cases/105-an-assembler-stub-returns-to-problem-state/manifest.json' }],
  ...(groq ? { provider: 'groq', model: 'openai/gpt-oss-120b', reasoning: 'low', maxTokens: 2500 } : {}),
});
JOBS.push(
  pair('h30-mode-sup', 'hlasm-supervisor-state-change', 'crit', 'CWE-250', 'the program switches to supervisor state with MODESET MODE=SUP (positive); the negative only issues MODESET MODE=PROB.', ['hlasm-a-stub-enters-supervisor-state', 'hlasm-a-stub-stays-in-problem-state'], false),
  pair('h31-extkey', 'hlasm-supervisor-state-change', 'crit', 'CWE-250', 'the program sets PSW key zero with MODESET EXTKEY=ZERO (positive); the negative uses MODESET EXTKEY=TCB, its own task\'s key.', ['hlasm-a-stub-takes-key-zero-by-extkey', 'hlasm-a-stub-takes-its-task-key'], true),
  pair('h32-ex', 'hlasm-executes-built-instruction', 'high', 'CWE-119', 'EX runs an MVC whose length comes from a register loaded from the caller\'s parameter (positive); the negative moves the same field with an MVC of fixed length and no EX.', ['hlasm-ex-moves-a-length-from-the-caller', 'hlasm-a-move-of-fixed-length'], false),
  pair('h33-exrl', 'hlasm-executes-built-instruction', 'high', 'CWE-119', 'EXRL runs a CLC whose length is in a register (positive); the negative branches with BRAS to a subroutine and executes nothing.', ['hlasm-exrl-runs-a-built-compare', 'hlasm-a-relative-branch-to-a-subroutine'], true),
  pair('h34-pc', 'hlasm-cross-memory-service', 'high', 'CWE-668', 'the program issues a PC instruction to a program-call number (positive); the negative calls a routine in its own address space with BALR.', ['hlasm-a-program-call-into-another-space', 'hlasm-a-call-within-the-same-space'], false),
  pair('h35-ssar', 'hlasm-cross-memory-service', 'high', 'CWE-668', 'the program changes its secondary address space with SSAR (positive); the negative only sets an access register with SAR.', ['hlasm-ssar-reaches-another-space', 'hlasm-sar-sets-an-access-register'], true),
  pair('h36-racroute', 'hlasm-calls-security-product', 'info', '-', 'the program asks RACF for an authorization check with RACROUTE REQUEST=AUTH (positive); the negative writes an operator message with WTO and calls no security service. The cwe in both manifests is the string "-".', ['hlasm-a-direct-racroute-check', 'hlasm-a-message-without-a-security-call'], true),
  pair('h37-called', 'hlasm-provides-called-module', 'info', '-', 'a COBOL program CALLs the assembler module by name and the assembler CSECT has that name (positive: both files in the case directory, the COBOL in fixed format with an 8-column sequence area convention, IDENTIFICATION DIVISION, PROGRAM-ID, PROCEDURE DIVISION, CALL \'NAME\', GOBACK); the negative: the COBOL program CALLs another name than the assembler CSECT. Each case holds a .cbl and a .asm file; the manifest file entry names the .asm file. The cwe in both manifests is the string "-".', ['hlasm-cobol-calls-the-assembler-module', 'hlasm-cobol-calls-another-module'], false),
  pair('h38-xctl-de', 'hlasm-runtime-module-name', 'med', 'CWE-470', 'XCTL with DE= pointing at a list entry the program filled from its caller (positive); the negative uses XCTL EP= with a fixed name.', ['hlasm-xctl-by-a-list-entry', 'hlasm-xctl-to-a-named-module'], true),
  pair('h39-attach-eploc', 'hlasm-runtime-module-name', 'med', 'CWE-470', 'ATTACH with EPLOC= naming storage the program copied a name into (positive); the negative uses ATTACH EP= with a fixed name.', ['hlasm-attach-by-a-name-in-storage', 'hlasm-attach-of-a-named-module'], false),
);

const LEREF = '/Users/portll/.claude/scratch/lang-qwen/ibm-le';
const TKREF = '/Users/portll/.claude/scratch/lang-qwen/ibm-toolkit';
JOBS.push(
  { ...macro('h40-le', 'le', 'Language Environment macros: CEEENTRY, CEETERM, CEEPPA, CEECAA, CEEDSA', 70, ['CEEENTRY', 'CEETERM', 'CEEPPA', 'CEECAA', 'CEEDSA'],
    `- CEEENTRY: the label names the routine; keywords as IBM lists them (PPA=, AUTO=, MAIN=, NAB=, PARMREG=, BASE=, EXPORT=, PLIST=, and the rest). Node { kind: 'CEEENTRY', name: st.name, main: raw MAIN= or null, ppa: raw PPA= or null, keywords }.
- CEETERM: keywords RC=, MODIFIER= and the rest IBM lists. Node { kind: 'CEETERM', rc: raw RC= or null, keywords }.
- CEEPPA: keywords as IBM lists them (LIBRARY=, PPA2=, EXTPROC=, TSTAMP=, PEP=, INSTOP=, EXITDSA=, OWNEXM=, EPNAME=, VER=, REL=, MOD=, DSA=). Node { kind: 'CEEPPA', name: st.name, keywords }.
- CEECAA and CEEDSA: mapping macros with no operands, or the few keywords IBM shows. Node { kind, keywords }.`),
    context: [{ file: 'lib/hlasm/read.mjs' }, { file: 'lib/hlasm/operands.mjs' }, ...['CEEENTRY', 'CEETERM', 'CEEPPA'].map((k) => ({ samples: k, n: 4 })),
      ...['CEEENTRY', 'CEETERM', 'CEEPPA', 'CEECAA', 'CEEDSA'].map((m) => ({ reference: `${LEREF}/${m}.txt`, title: `IBM z/OS 3.1 Language Environment Programming Guide: the ${m} macro`, max: 6000 }))] },
  { ...macro('h41-structured', 'structured', 'HLASM Toolkit structured programming macros: IF, ELSE, ELSEIF, ENDIF, DO, ENDDO, DOEXIT, ASMLEAVE, ITERATE, SELECT, WHEN, OTHRWISE, ENDSEL', 68, ['IF', 'ELSE', 'ENDIF', 'DO', 'ENDDO'],
    `Parsers for the structured programming macros, as the HLASM Toolkit Feature User's Guide below gives them. A condition is either a condition-code mnemonic in parentheses, (CC=8) or (Z) or (E), or an instruction and its operands in parentheses followed by a mnemonic, (CLI,0(R1),EQ,C'A') or (LTR,R15,R15,NZ), joined by AND, OR, ANDIF and ORIF; keep each condition as raw text and list its parts. Accept the mnemonics IBM lists (H, L, E, Z, O, M, P, NH, NL, NE, NZ, NO, NM, NP and their EQ/NE/GT/LT/GE/LE/Z/NZ forms) and refuse another word in the mnemonic position.
- IF and ELSEIF: one or more conditions. Node { kind, conditions: [raw], joins: ['AND'|'OR'|'ANDIF'|'ORIF'] }.
- ELSE, ENDIF, ENDDO, ENDSEL, OTHRWISE: no operand (a label is allowed). Node { kind }.
- DO: no operand (a simple DO), ONCE, INF, FROM=(reg[,value]) with TO= and BY=, WHILE=condition, UNTIL=condition, or (reg) / (reg,count) for a BCT loop. Node { kind: 'DO', form: 'ONCE'|'INF'|'COUNT'|'FROM'|'WHILE'|'UNTIL'|'SIMPLE', keywords }.
- DOEXIT and ITERATE: a condition, or no operand. ASMLEAVE: an optional label. SELECT: optional CLI,..., or CHI form as IBM shows. WHEN: one or more values or conditions. Node { kind, operands: [raw] }.`),
    context: [{ file: 'lib/hlasm/read.mjs' }, { file: 'lib/hlasm/operands.mjs' }, ...['IF', 'DO', 'ENDDO'].map((k) => ({ samples: k, n: 5 })),
      { reference: `${TKREF}/structured.txt`, title: "IBM HLASM Toolkit Feature User's Guide: structured programming macros", max: 15000 }] },
  { ...JOBS.find((x) => x.id === 'h32-ex'), id: 'h32-ex-v2', spec: `${JOBS.find((x) => x.id === 'h32-ex').spec}\n\nAlso: an SS-format instruction's second operand is D2(B2): write 0(R4) or FIELD, never 0(,R4) or 0(X,B); its first operand is D1(L1,B1): 0(0,R2) for an EX target, OUT(8) or 0(16,R1). z390 refused the last drafts for 0(,1) as a second operand.` },
);

// Locator conformance fixtures: small macro-free programs, one feature each, that z390 assembles
// cleanly. The gate records how the locator grades on each; a difference is the locator's to fix.
const EXAMPLE = ['FIRST    CSECT', '         USING FIRST,15', "A        DC    C'ABC'", 'B        DS    H', "C        DC    PL3'12',Z'123'", 'D        DS    0D', "E        DC    2F'1,2'", 'F        EQU   *-E', '         LTORG', '         END'].join('\n');
const fixture = (id, feature, groq) => ({
  id, track: 'hlasm', title: `Locator fixture: ${feature}`, type: 'draft', priority: 50, estimateMin: 5,
  target: [`bench/hlasm-locate/${id.replace(/^h\d+-/, '')}.asm`],
  spec: `Write bench/hlasm-locate/${id.replace(/^h\d+-/, '')}.asm: one HLASM program of 15 to 40 statements that exercises ${feature}, as an assembler would lay it out.
Rules:
- No macro calls at all (no SAVE, RETURN, WTO, YREGS, DCB or any other macro): use only machine instructions, DC, DS, EQU, ORG, CNOP, LTORG, CSECT, DSECT, USING, DROP and END.
- Start with NAME CSECT and USING NAME,15; finish with END.
- Put a label on every DC, DS and EQU and on several instructions, so the location and length of each can be checked. Labels are 1 to 8 characters, unique, starting with a letter.
- A label starts in column 1, the operation in column 10, the operands in column 16; nothing goes past column 71; columns 72 to 80 stay blank.
- Use plain register numbers (0 to 15).
The gate assembles the program with z390, requires no error and no macro call, and needs at least five labelled symbols. Reply with only the file. A short example of the form:
\`\`\`
${EXAMPLE}
\`\`\``,
  context: [],
  ...(groq ? { provider: 'groq', model: 'openai/gpt-oss-120b', reasoning: 'low', maxTokens: 1800 } : {}),
});
JOBS.push(
  fixture('h50-character-constants', 'character, hexadecimal and binary constants with explicit lengths, duplication factors and several values in one operand (CL8, XL3, 3C, BL2, X\'01,02\')', true),
  fixture('h51-fixed-alignment', 'fixed-point and address constants placed after odd-length fields, so the assembler must align them: H, F, FD, A, AD, Y and V, with and without length modifiers', false),
  fixture('h52-decimal-constants', 'packed and zoned decimal constants with and without explicit lengths and with several values in one operand (P, PL4, Z, ZL5)', true),
  fixture('h53-floating-constants', 'floating-point constants of each length (E, D, L, EH, DH, EB, DB, LB) placed after odd-length fields', false),
  fixture('h54-storage-alignment', 'DS statements with zero duplication used to align and to name a group (DS 0F, DS 0D, DS 0H, DS 0CL80 followed by fields that fill it)', true),
  fixture('h55-org', 'ORG: back to an earlier label to redefine storage, ORG with no operand to return to the high-water mark, and ORG with a boundary operand', false),
  fixture('h56-cnop', 'CNOP with each byte and boundary pair (0,4 2,4 0,8 2,8 4,8 6,8) between instructions and data', true),
  fixture('h57-equ', 'EQU of absolute values, of relocatable labels plus offsets, of *-label lengths, and with explicit length and type operands', false),
  fixture('h58-literal-pools', 'literals of many lengths (=F, =H, =C, =X, =D, =A, =V, =P) used by instructions, gathered by two LTORG statements and by END', false),
  fixture('h59-sections', 'two CSECTs and one DSECT, with the first CSECT resumed after the second, and data defined in each', true),
  fixture('h60-length-expressions', "DS and DC lengths and duplication factors given by expressions over earlier symbols: CL(L'X), XL(END-START), (N)F with N an EQU", false),
  fixture('h61-instruction-formats', 'labelled machine instructions of each common format (RR, RX, RS, SI, SS, RI, RIL, RXY, RSY, SIL) between data, so each length moves the next label', true),
  fixture('h62-dsect-layout', 'a DSECT mapping a record: fields of every constant type, ORG to overlay a field with two smaller ones, and a length EQU at the end', false),
);

// Second fixture batch: the length-expression fixture again, with what z390 refused, and five more
// locator features. HLASM comments start with * in column 1.
const COMMENT = 'An HLASM comment line starts with * in column 1, so begin each .asm file with the line: * SPDX-License-Identifier: AGPL-3.0-or-later (never //).';
for (const [id, feature, note] of [
  ['h63-length-expressions', "DS and DC lengths and duplication factors given by expressions over symbols defined earlier: CL(L'X), XL(B-A) with A and B labels already defined, (N)F with N an EQU defined earlier", 'z390 refused the last draft: every symbol a length or duplication factor uses must be defined above it as an absolute value (an EQU of a number, or the difference of two labels), and every DC needs a nominal value (only DS may omit it).'],
  ['h64-literal-order', 'many literals of different lengths used in one pool, in an order where the assembler must regroup them by alignment (=X\'01\', =F\'1\', =C\'ABC\', =D\'0\', =H\'2\', =FD\'3\'), then LTORG, then labelled data after the pool', ''],
  ['h65-ds-nominal', "DS statements that give a nominal value only to set their length (DS C'ABCDE', DS X'0102', DS P'123', DS CL3'X'), mixed with DC and aligned fields", ''],
  ['h66-equ-attributes', "EQU with explicit length and type operands (EQU *,4,C'F'), EQU of label differences, and labels whose length attributes later DS statements use (DS CL(L'FIELD))", ''],
  ['h67-org-boundary', 'ORG with a boundary and an offset operand (ORG *,8 and ORG *,256,16), and data placed after each', ''],
  ['h68-resumed-dsects', 'two DSECTs, each resumed after the other, with fields of mixed alignment in each part, and a CSECT using both with USING', ''],
]) JOBS.push({ ...fixture(id, feature, false), spec: `${fixture(id, feature, false).spec}\n\n${COMMENT}${note ? `\n\n${note}` : ''}` });

let wrote = 0;
for (const j of JOBS) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
