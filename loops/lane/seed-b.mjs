// Writes the statement-parser and semantics jobs into bank/. Existing job files are left alone.
import * as PATHS from '../paths.mjs';
import { writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const BANK = join(PATHS.LANE_HOME, 'bank');

const CONTRACT = `How a statement parser plugs in. Your module exports \`parsers\`, an object mapping a statement kind (as classify() in lib/pli/statements.mjs names it) to a function (c, stmt, ctx) => node.
- c is a cursor (lib/pli/cursor.mjs) on the statement's tokens after its label and condition prefixes, positioned at the first token, which for a keyword statement is the keyword itself: consume it.
- stmt is { labels: [{ name }], conditions, toks, line, endLine, file }.
- ctx is { parseStatement, subStatement }. A statement nested in this one (the unit after THEN, ELSE, an ON condition, a WHEN list, OTHERWISE) is built with ctx.subStatement(stmt, tokens) from the remaining tokens, parsed with ctx.parseStatement(sub), and the result pushed to node.units. Take those tokens with c.drain(), which returns every remaining token and leaves the cursor at the end.
- Return a node with at least { kind }. The dispatcher refuses the statement if any token is left, so consume everything you accept, and call c.fail('what you expected') on anything you do not understand. Never return a partial node silently.
- Import from a module in lib/pli/stmt/ as: import { cursor, split, PliSyntax } from '../cursor.mjs'; import { parseExpression, parseReference } from '../expr.mjs'. Every expression goes through parseExpression(c, { stopWords, stopOps }), which stops before a depth-0 ',' or unmatched ')' or a stop word/op and returns { t: 'expr', toks, refs }; every reference through parseReference(c), which returns { t: 'ref', name, path, args, toks }. The expression parser is being rewritten in parallel with the same signatures, so rely only on these fields.
- A parenthesised operand: const inner = c.group() gives its tokens and leaves c after the ')'; parse inner with a sub-cursor, const s = cursor(inner), and call s.fail when !s.done() afterwards. c.items() gives a parenthesised list as an array of token arrays split at depth-0 commas.
- Token shapes: { t: 'word', v, u (upper case) } | { t: 'lit', v, suffix } | { t: 'num', v } | { t: 'op', v } with line and col. PL/I has no reserved words: a keyword is recognised by position, never refused as a name.`;

const BASE_CONTEXT = [{ file: 'lib/pli/cursor.mjs' }, { file: 'lib/pli/statements.mjs' }, { file: 'lib/pli/expr.mjs' }];

const job = (id, title, priority, module, kinds, spec, extra = [], est = 18) => ({
  id, title, type: 'draft', priority, estimateMin: est,
  target: [`lib/pli/stmt/${module}.mjs`, `test/pli-${module}.test.mjs`],
  kinds,
  spec: `Write lib/pli/stmt/${module}.mjs (it is an empty stub today) and its tests, test/pli-${module}.test.mjs.\n\n${spec}\n\n${CONTRACT}`,
  context: [...BASE_CONTEXT, { file: `lib/pli/stmt/${module}.mjs` }, ...extra, ...kinds.slice(0, 6).map((k) => ({ samples: k, n: kinds.length > 2 ? 8 : 18 }))],
});

const JOBS = [
  job('b01-declare', 'DECLARE and declaration fragments', 92, 'declare', ['DECLARE', 'DECLARE_FRAGMENT'], `Parsers for kinds DECLARE (keyword DECLARE or DCL) and DECLARE_FRAGMENT (no keyword: an %INCLUDE member holding the middle of a structure starts at a level number, e.g. \`05  CD01_DATA,\` or \`10 BANK_EVERYTHING CHAR(6144), 10 EXT_DATA CHAR(1024)\`).

Syntax: DECLARE item [, item]... ; items are separated by depth-0 commas. item = [level] name [dimension] [attribute]... where:
- level is an unsigned integer number token; absent for a level-less scalar or array.
- name is a word, or '*' for an unnamed structure member; or a factored list \`(A, B, C)\` whose members may carry their own level, dimension and attributes and may nest, with the attributes after the ')' applying to every member: \`DCL (A, B) FIXED BIN(31), (C(10), D) CHAR(1);\` \`DCL (ADDR, LENGTH, SUBSTR) BUILTIN;\`.
- a dimension is a parenthesised list directly after the name: \`A(10)\`, \`A(0:9, -5:5)\`, \`A(*)\`, \`A(N)\`.
- an attribute is a keyword, optionally followed by a parenthesised argument list, or a PICTURE/PIC followed by a string literal. Expand abbreviations to the full name: CHAR→CHARACTER, BIN→BINARY, DEC→DECIMAL, PIC→PICTURE, PTR→POINTER, VAR→VARYING, VARZ→VARYINGZ, NONVAR→NONVARYING, INIT→INITIAL, EXT→EXTERNAL, INT→INTERNAL, AUTO→AUTOMATIC, CTL→CONTROLLED, DEF→DEFINED, POS→POSITION, ENV→ENVIRONMENT, WCHAR→WIDECHAR, COND→CONDITION, PARM→PARAMETER. Known attributes include CHARACTER, BIT, GRAPHIC, WIDECHAR, VARYING, VARYINGZ, NONVARYING, FIXED, FLOAT, BINARY, DECIMAL, PRECISION, SIGNED, UNSIGNED, REAL, COMPLEX, PICTURE, POINTER, OFFSET, HANDLE, AREA, ENTRY, RETURNS, OPTIONS, FILE, STREAM, RECORD, INPUT, OUTPUT, UPDATE, SEQUENTIAL, DIRECT, KEYED, ENVIRONMENT, PRINT, LABEL, FORMAT, CONDITION, BUILTIN, GENERIC, EXTERNAL, INTERNAL, STATIC, AUTOMATIC, BASED, CONTROLLED, DEFINED, POSITION, PARAMETER, ALIGNED, UNALIGNED, INITIAL, VALUE, LIKE, UNION, CONNECTED, NONCONNECTED, ASSIGNABLE, NONASSIGNABLE, NORMAL, ABNORMAL, BYADDR, BYVALUE, OPTIONAL, ORDINAL, TYPE, DIMACROSS, BIGENDIAN, LITTLEENDIAN, HEXADEC, IEEE, NATIVE, NONNATIVE, LIMITED, VARIABLE, IRREDUCIBLE, REDUCIBLE, RESERVED, INONLY, OUTONLY, INOUT, TASK, EVENT, BACKWARDS, BUFFERED, UNBUFFERED, TRANSIENT, EXCLUSIVE. An attribute word you do not know is still accepted with its optional parenthesised arguments, and its name pushed to node.unknownAttributes; never fail on an unknown attribute word.
- LIKE takes a reference (parseReference); BASED, DEFINED and POSITION take a parenthesised expression or reference, or for DEFINED a bare reference; ENTRY's parenthesised list holds parameter descriptors (attribute lists, possibly with levels) separated by commas — keep it as tokens; RETURNS(...) likewise; INITIAL(...) holds items with iteration factors, \`INIT((10)0, (5)(2)'AB', *)\`, and the forms \`INITIAL CALL p(args)\` and \`INITIAL TO(varying) (...)\`: keep INITIAL's argument tokens as args, do not evaluate them.
Node: { kind: 'DECLARE' | 'DECLARE_FRAGMENT', items: [{ level: number|null, name, factored: [names]|null, dims: [token arrays], attributes: [{ name, args: token array|null, picture: string|undefined }], line }], unknownAttributes: [names] }. For a factored list, emit one item per member, each carrying the member's own and the shared attributes, with factored naming the whole list.`, [], 22),
  job('b02-procedure', 'PROCEDURE, ENTRY, PACKAGE, BEGIN and END', 88, 'procedure', ['PROCEDURE', 'END', 'ENTRY', 'BEGIN', 'PACKAGE'], `Parsers for kinds PROCEDURE (keyword PROCEDURE or PROC), ENTRY, PACKAGE, BEGIN and END.
- \`L: PROCEDURE [(params)] [option]...\` where params are names separated by commas (possibly none: \`PROC()\`), and the options in any order are OPTIONS(list), RETURNS(attribute list), RECURSIVE, NONRECURSIVE, REORDER, ORDER, EXTERNAL or EXTERNAL('name') (also EXT), CHARGRAPHIC, NOCHARGRAPHIC, REDUCIBLE, IRREDUCIBLE. OPTIONS' list holds words with optional parenthesised arguments: MAIN, REENTRANT, FETCHABLE, NOEXECOPS, ASSEMBLER/ASM, COBOL, BYVALUE, BYADDR, LINKAGE(SYSTEM), DLLINTERNAL, ...; keep each as { name, args }. Corpus forms: \`PLRINTE06V14: PROCEDURE OPTIONS(MAIN);\`, \`init_processing: PROC;\`, \`EPSBDP1: PROC (CA_PTR) OPTIONS(MAIN,REENTRANT) REORDER;\`.
- \`E: ENTRY [(params)] [RETURNS(...)] [OPTIONS(...)] [EXTERNAL('x')]\`.
- \`P: PACKAGE [EXPORTS(* | names)] [RESERVES(* | names)] [OPTIONS(...)]\`.
- \`[L:] BEGIN [OPTIONS(...)] [ORDER|REORDER] [CHARGRAPHIC|NOCHARGRAPHIC]\`.
- \`END [label]\`.
Node: { kind: 'PROCEDURE'|'ENTRY', name: the last label or null, params: [names], options: [{ name, args }], returns: token array|null, recursive: bool, external: string|true|null, order: 'ORDER'|'REORDER'|null }; { kind: 'PACKAGE', name, exports, reserves, options }; { kind: 'BEGIN', options }; { kind: 'END', label: name|null }.`),
  job('b03-control', 'DO, IF, ELSE, SELECT, WHEN, OTHERWISE and jumps', 90, 'control', ['IF', 'DO', 'ELSE', 'WHEN', 'OTHERWISE', 'SELECT', 'RETURN', 'GOTO', 'LEAVE', 'ITERATE', 'STOP', 'EXIT'], `Parsers for kinds DO, IF, ELSE, SELECT, WHEN, OTHERWISE (keyword OTHERWISE or OTHER), LEAVE, ITERATE, GOTO (keywords GO TO or GOTO), RETURN, STOP and EXIT.
- DO: \`DO;\` (a group); \`DO WHILE(expr) [UNTIL(expr)]\`; \`DO UNTIL(expr) [WHILE(expr)]\`; \`DO LOOP\` and \`DO FOREVER\`; iterative \`DO ref = spec [, spec]...\` where spec is expr [TO expr] [BY expr] or expr [BY expr] [TO expr] or expr REPEAT expr, each optionally followed by WHILE(expr) and/or UNTIL(expr). The control variable is a reference (parseReference: it may be subscripted or qualified), then '='. Expressions in a spec stop at the words TO, BY, REPEAT, WHILE, UNTIL and at a depth-0 ','. A variable may be named WHILE: if the word after DO is followed by '=', it is the control variable.
- IF: \`IF expr THEN unit\`: the condition is parseExpression(c, { stopWords: ['THEN'] }), then THEN, then the unit is every remaining token, parsed as a nested statement (ctx.subStatement and ctx.parseStatement) into node.units. The unit is often \`DO\` (\`IF A THEN DO;\` opens a group), another IF, an assignment, CALL, RETURN, LEAVE, GO TO; it may carry its own label.
- ELSE: \`ELSE unit\` (units as for IF); \`ELSE;\` gives units [].
- SELECT: \`SELECT\` or \`SELECT (expr)\`. WHEN: \`WHEN (expr [, expr]...) unit\`, also \`WHEN ANY (...)\` and \`WHEN ALL (...)\` (UNVERIFIED), units as for IF. OTHERWISE: \`OTHERWISE unit\` or \`OTHERWISE;\`.
- \`LEAVE [label]\`, \`ITERATE [label]\`, \`GO TO label-reference\` or \`GOTO label-reference\` (it may be subscripted, L(I)), \`RETURN [(expr)]\`, \`STOP\`, \`EXIT\`.
Node: { kind: 'DO', form: 'group'|'while'|'iterative'|'loop', control: ref|null, specs: [{ from, to, by, repeat, while, until }], while: expr|null, until: expr|null }; { kind: 'IF', cond: expr, units }; { kind: 'ELSE', units }; { kind: 'SELECT', subject: expr|null }; { kind: 'WHEN', values: [expr], units }; { kind: 'OTHERWISE', units }; { kind: 'LEAVE'|'ITERATE', label }; { kind: 'GOTO', target: ref }; { kind: 'RETURN', value: expr|null }; { kind: 'STOP'|'EXIT' }.`, [], 22),
  job('b04-call', 'CALL, FETCH and RELEASE', 85, 'call', ['CALL', 'FETCH', 'RELEASE'], `Parsers for kinds CALL, FETCH and RELEASE.
- \`CALL entry-reference [(arguments)] [TASK(...)] [EVENT(...)] [PRIORITY(...)]\`: parseReference reads the entry reference and its argument lists; the arguments are the reference's LAST argument list when the reference has any. Each argument is an expression (parse it with a sub-cursor and parseExpression); an empty list \`CALL P();\` has none. Corpus forms: \`CALL init_processing;\`, \`CALL PLITDLI (PARM_CT_3, GU, PCB1, IO_AREA);\`, \`CALL CEEDAYS(DATE_STR, PIC_STR, LILIAN, FC);\`.
- \`FETCH entry-name [SET(ptr)] [TITLE(expr)] [, entry-name ...]\`.
- \`RELEASE entry-name [, entry-name]...\` and \`RELEASE *\`.
Node: { kind: 'CALL', callee: ref, name: callee.path.join('.'), args: [expr], options: [{ name, args }] }; { kind: 'FETCH', entries: [{ name, set: ref|null, title: expr|null }] }; { kind: 'RELEASE', entries: [names] }.`),
  job('b05-assignment', 'Assignment statements', 89, 'assignment', ['ASSIGNMENT'], `The parser for kind ASSIGNMENT. classify() has already decided the statement reads as references followed by an assignment operator.
- \`target [, target]... = expression [, BY NAME]\` and compound \`target op expression\` for += -= *= /= |= &= ||= **=.
- Each target is a reference (parseReference): it may be qualified (A.B.C), subscripted (A(I,J)), located (P->X, H=>Y) or a pseudovariable with arguments (SUBSTR(S,1,2), UNSPEC(X), STRING(S)).
- The value is parseExpression(c); after it, an optional \`, BY NAME\`.
Node: { kind: 'ASSIGNMENT', targets: [ref], op: '='|'+='|..., value: expr, byName: bool }.`),
  job('b06-io-record', 'OPEN, CLOSE and record I/O', 84, 'io', ['OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE', 'DELETE', 'LOCATE', 'UNLOCK'], `Parsers for kinds OPEN, CLOSE, READ, WRITE, REWRITE, DELETE, LOCATE and UNLOCK.
- \`OPEN FILE(ref) [option]... [, FILE(ref) [option]...]...\`: options in any order: STREAM, RECORD, INPUT, OUTPUT, UPDATE, SEQUENTIAL, DIRECT, TRANSIENT, BUFFERED, UNBUFFERED, KEYED, PRINT, BACKWARDS, EXCLUSIVE, TITLE(expr), LINESIZE(expr), PAGESIZE(expr), ENVIRONMENT(...) (keep as tokens; also ENV).
- \`CLOSE FILE(ref) [ENVIRONMENT(...)] [, FILE(ref) ...]...\`, and \`CLOSE FILE(*)\`.
- \`READ FILE(f) INTO(ref)|SET(ptr)|IGNORE(expr) [KEY(expr)|KEYTO(ref)] [EVENT(ref)] [NOLOCK]\`, options in any order.
- \`WRITE FILE(f) FROM(ref) [KEYFROM(expr)|KEYTO(ref)] [EVENT(ref)]\`; \`REWRITE FILE(f) [FROM(ref)] [KEY(expr)] [EVENT(ref)]\`; \`DELETE FILE(f) [KEY(expr)] [EVENT(ref)]\`; \`LOCATE ref FILE(f) [SET(ptr)] [KEYFROM(expr)]\`; \`UNLOCK FILE(f) KEY(expr)\`.
- FILE, INTO, FROM, SET, KEYTO and EVENT take a reference; KEY, KEYFROM, IGNORE, TITLE, LINESIZE, PAGESIZE an expression; parse each inside its parentheses with a sub-cursor and fail if anything is left.
Node: { kind: 'OPEN'|'CLOSE', files: [{ file: ref, options: { NAME: expr|ref|tokens|true } }] }; { kind: 'READ'|'WRITE'|'REWRITE'|'DELETE'|'LOCATE'|'UNLOCK', file: ref, target: ref|null (LOCATE's based variable), options: { INTO, SET, FROM, KEY, KEYTO, KEYFROM, IGNORE, EVENT, NOLOCK } } with only the options present. An unknown option word with parentheses is kept as tokens under its name rather than refused; one without is true.`),
  job('b07-io-stream', 'GET, PUT, DISPLAY, FORMAT and FLUSH', 87, 'stream', ['PUT', 'GET', 'DISPLAY', 'FORMAT', 'FLUSH'], `Parsers for kinds PUT, GET, DISPLAY, FORMAT and FLUSH.
- PUT: options in any order: FILE(ref), STRING(ref), PAGE, LINE(expr), SKIP or SKIP(expr), and one of LIST(data-list), EDIT(data-list)(format-list) — possibly repeated pairs \`EDIT (a)(A) (b)(F(5))\` — or DATA or DATA(data-list). A data-list is expressions separated by commas, and an item may be a repetitive specification \`(X(I) DO I = 1 TO N)\` or nested \`((A(I,J) DO J = 1 TO M) DO I = 1 TO N)\`: keep such an item as tokens.
- A format list is format items separated by commas: A, A(w), B(w), F(w), F(w,d), F(w,d,p), E(w,d), E(w,d,s), P'picture', X(n), SKIP, SKIP(n), PAGE, LINE(n), COLUMN(n) or COL(n), R(label), C(...), G(w), L, and an iteration factor before an item: \`(3)A(5)\`, \`3 A(5)\`, \`(2)(A, X(2))\`. Do not parse format items as expressions: keep each item's tokens.
- GET: FILE(ref), STRING(expr), SKIP[(expr)], COPY[(ref)], and LIST(refs), EDIT(refs)(formats)..., DATA[(refs)].
- \`DISPLAY (expr) [REPLY(ref)] [EVENT(ref)] [ROUTCDE(list)] [DESC(list)]\`.
- \`L: FORMAT (format-list)\`. \`FLUSH FILE(ref)\` and \`FLUSH FILE(*)\`.
Node: { kind: 'PUT'|'GET', file: ref|null, string: ref|expr|null, page, line, skip: expr|true|null, mode: 'LIST'|'EDIT'|'DATA'|null, data: [expr or { tokens }], formats: [[token arrays per item]] }; { kind: 'DISPLAY', value: expr, reply: ref|null }; { kind: 'FORMAT', items: [token arrays] }; { kind: 'FLUSH', file: ref|'*' }. Corpus forms: \`PUT SKIP LIST('text', X);\`, \`PUT FILE(SYSPRINT) EDIT (A, B) (A, F(5));\`, \`PUT SKIP EDIT ('*** ', MSG) (A, A(80));\`.`, [], 22),
  job('b08-conditions', 'ON, SIGNAL, REVERT and RESIGNAL', 80, 'conditions', ['ON', 'SIGNAL', 'REVERT'], `Parsers for kinds ON, SIGNAL, REVERT and RESIGNAL.
- \`ON condition [, condition]... [SNAP] (SYSTEM | unit)\`. A condition is a word with an optional parenthesised argument: ENDFILE(f), ENDPAGE(f), KEY(f), RECORD(f), TRANSMIT(f), UNDEFINEDFILE(f) or UNDF(f), NAME(f), CONDITION(name) or COND(name); and without one: ERROR, FINISH, CONVERSION/CONV, FIXEDOVERFLOW/FOFL, OVERFLOW/OFL, UNDERFLOW/UFL, ZERODIVIDE/ZDIV, SIZE, STRINGRANGE/STRG, STRINGSIZE/STRZ, SUBSCRIPTRANGE/SUBRG, AREA, STORAGE, ATTENTION/ATTN, INVALIDOP, ANYCONDITION/ANYCOND. Keep the condition name as written, upper-cased, with its argument tokens.
- After the conditions: optional SNAP, then either SYSTEM (and nothing else) or the unit: every remaining token parsed as a nested statement via ctx.subStatement and ctx.parseStatement into node.units. The unit is commonly BEGIN (\`ON ERROR BEGIN;\`), an assignment (\`ON ENDFILE(INFILE) EOF = '1'B;\`), GO TO, CALL, or PUT.
- A unit word could collide with a condition-list continuation only through a comma: the conditions end at the first word not preceded by ',' after a complete condition.
- \`SIGNAL condition\`, \`REVERT condition\`, \`RESIGNAL\`.
Node: { kind: 'ON', conditions: [{ name, arg: token array|null }], snap: bool, system: bool, units }; { kind: 'SIGNAL'|'REVERT', condition: { name, arg } }; { kind: 'RESIGNAL' }.`),
  job('b09-alloc', 'ALLOCATE and FREE', 70, 'alloc', ['ALLOCATE', 'FREE'], `Parsers for kinds ALLOCATE (keyword ALLOCATE or ALLOC) and FREE.
- \`ALLOCATE [level] name [dimension] [attribute]... [SET(ref)] [IN(ref)] [, ...]\`: for a CONTROLLED variable a list of names with optional level and attributes (CHAR(n), INITIAL(...), dimensions); for a BASED one \`ALLOCATE X SET(P);\`, \`ALLOC X SET(P) IN(A);\`. Keep attributes as { name, args } like DECLARE's, abbreviations as written.
- \`FREE ref [IN(ref)] [, ref [IN(ref)]]...\`, including \`FREE P->X;\`.
Node: { kind: 'ALLOCATE', items: [{ level, name, dims, attributes, set: ref|null, in: ref|null }] }; { kind: 'FREE', items: [{ ref, in: ref|null }] }.`, [], 15),
  job('b10-exec', 'EXEC SQL, EXEC CICS and EXEC DLI', 80, 'exec', ['EXEC'], `The parser for kind EXEC: \`EXEC SQL statement-text\`, \`EXEC CICS command option...\`, \`EXEC DLI ...\`; the semicolon the splitter used ends it. Fail only when EXEC has no processor word after it.
- Node: { kind: 'EXEC', processor: 'SQL'|'CICS'|'DLI'|other word, toks: tokens after the processor word, sql?: { verb, hostVars: [{ host, written }] }, cics?: { command: [words], options: [{ name, args: token array|null }] } }. Consume every token.
- SQL host variables in PL/I are \`:name\`, \`:a.b\`, \`:name:ind\` and \`:name INDICATOR :ind\`. Use hostVariablesIn from '../../embedded-sql.mjs' (shown below), which takes COBOL-parser tokens, so convert each PL/I token first: word → { t: 'word', u }, op ':' → { t: 'op', v: ':' }, op '.' between a word and a word → { t: 'period', joined: true }, lit → { t: 'lit', v }, num → { t: 'word', u: v }, every other op → { t: 'word', u: v }. Map its result to [{ host, written }]. The verb is the first word: SELECT, INSERT, UPDATE, DELETE, DECLARE, OPEN, FETCH, CLOSE, INCLUDE, WHENEVER, COMMIT, ROLLBACK, CALL, SET, PREPARE, EXECUTE, CONNECT and others.
- CICS: the command is the leading words up to the first word that is followed by '(' or is a known option keyword; options are words each with an optional parenthesised argument (\`EXEC CICS RETURN TRANSID('X') COMMAREA(C) LENGTH(L);\` → command ['RETURN']; \`EXEC CICS SEND MAP('M') MAPSET('S') ERASE;\` → command ['SEND', 'MAP']? — treat MAP as part of the command when the first two words are SEND MAP or RECEIVE MAP, and record MAP('M') as an option too; mark such choices UNVERIFIED). Never fail on an unknown CICS option.`, [{ file: 'lib/embedded-sql.mjs' }]),
  job('b11-preprocessor', 'Preprocessor statements', 83, 'preprocessor', ['PREPROCESSOR'], `The parser for kind PREPROCESSOR: statements whose first token is the op '%'.
- Forms: \`%INCLUDE member;\`, \`%INCLUDE SYSLIB(member);\`, \`%INCLUDE 'member';\`, \`%INCLUDE a, b;\`, \`%XINCLUDE ...\`; \`%DCL\`/\`%DECLARE name [, name] attribute...\` (CHARACTER, FIXED, ENTRY, BUILTIN, ...); \`%name = expression\` (a preprocessor assignment); \`%IF expr %THEN unit\` and \`%ELSE unit\`; \`%DO\`, \`%END\`; \`%ACTIVATE\`/\`%ACT names [RESCAN|NORESCAN]\`, \`%DEACTIVATE\`/\`%DEACT names\`; \`%GO TO label\`/\`%GOTO label\`; \`%PAGE\`, \`%SKIP\` or \`%SKIP(n)\`, \`%PRINT\`, \`%NOPRINT\`, \`%PUSH\`, \`%POP\`, \`%NOTE(message, code)\`; \`%PROCEDURE\`/\`%PROC [(params)] [STATEMENT] [RETURNS(CHARACTER|FIXED)]\` with a label (\`%P: PROC(X) RETURNS(CHAR);\` — note the label precedes the '%' only in some sources: \`%P: PROC\` gives the label after '%', so a word followed by ':' right after '%' is a label); \`%RETURN(expr)\`; \`%REPLACE name BY value\`; \`%SELECT\`, \`%WHEN(expr) unit\`, \`%OTHERWISE unit\`; \`%ITERATE\`, \`%LEAVE\`; \`%INSCAN\`, \`%XINSCAN\`; \`%PROCESS options\`; and \`%;\` (null).
- In \`%IF A = 1 %THEN %DO;\` the condition stops at the op '%' followed by the word THEN: use parseExpression(c, { stopOps: ['%'] }). The unit after %THEN or %ELSE is the remaining tokens: when it begins with '%' parse it recursively with this same parser through ctx.parseStatement(ctx.subStatement(stmt, tokens)); otherwise it is a plain statement, parsed the same way.
- A directive word you do not know: keep its tokens, set directive to the word, and do not fail.
Node: { kind: 'PREPROCESSOR', directive: 'INCLUDE'|'XINCLUDE'|'DECLARE'|'ASSIGN'|'IF'|'ELSE'|'DO'|'END'|'ACTIVATE'|..., label: name|null, members: [{ name, library: name|null }] (INCLUDE), names: [...], value: expr|null, units }.`),
  job('b12-misc', 'DEFAULT, DEFINE, WAIT, DELAY, ATTACH, DETACH, ASSERT and QUALIFY', 66, 'misc', ['DEFAULT'], `Parsers for kinds DEFAULT (keyword DEFAULT or DFT), DEFINE, WAIT, DELAY, ATTACH, DETACH, ASSERT and QUALIFY.
- \`DEFAULT RANGE(letters | letter:letter | *) attribute... [, ...]\`, \`DEFAULT DESCRIPTORS attribute...\`, \`DEFAULT VALUE (attribute...)\`, \`DEFAULT NONE\`, and combinations separated by commas.
- \`DEFINE ALIAS name attribute...\`, \`DEFINE STRUCTURE 1 name, 2 member attribute..., ...\`, \`DEFINE ORDINAL name (value [VALUE(n)], ...) [PRECISION(n)] [SIGNED|UNSIGNED]\`.
- \`WAIT (event [, event]...) [(expr)]\`, \`DELAY (expr)\`, \`ATTACH entry-reference [THREAD(ref)] [ENVIRONMENT(...)]\`, \`DETACH THREAD(ref)\`, \`ASSERT TRUE(expr) | FALSE(expr) | COMPARE(expr, expr [, op]) | UNREACHABLE [TEXT(expr)]\`, \`QUALIFY name\`.
Keep attribute lists as { name, args } and be permissive, but consume every token. Node: { kind, ...fields named after the clauses }.`, [], 15),
];

const SEMANTICS = [
  {
    id: 'c01-picture', title: 'PL/I PICTURE strings: kind, storage, precision and scale', type: 'draft', priority: 65, estimateMin: 18,
    target: ['lib/pli/picture.mjs', 'test/pli-picture.test.mjs'], kinds: [],
    spec: `Write lib/pli/picture.mjs and test/pli-picture.test.mjs. Export analyzePicture(text) where text is the picture string without quotes, e.g. '(5)9V99', 'S(4)9', 'ZZZ9.99', '$$$,$$9.99CR', '9V99E+99', 'X(20)'? — NO: a repetition factor precedes its character, so '(20)X'. Character pictures contain only X, A and 9 with at least one X or A; numeric pictures use 9 Z * Y V S + - $ . , / B K E F(n) T I R CR DB. A repetition factor (n) multiplies the character after it.
Return { kind: 'character'|'numeric', bytes, precision, scale, float, signed, valid, problems: [strings] }:
- bytes is the storage length: every picture character that occupies a position; V, K and F(n) take none; CR and DB take two.
- precision is the count of digit positions: 9, Z, *, Y, T, I, R, and a drifting string of $, + or - or S contributes one fewer digit than its length (the first character of a drifting string is the sign or currency position).
- scale is the number of digit positions after V, minus n for F(n) (F(-n) adds n).
- float is true when E or K appears; signed when S, +, -, CR, DB, T, I or R appears.
- valid false with a problem string when the picture is not one of the two kinds, e.g. mixes X with Z.
List the rules you used under Limits; mark each one you are not sure IBM Enterprise PL/I applies as UNVERIFIED. Tests: at least 20 pictures with exact expected values, including repetition factors, drifting $ and +, CR/DB, V and F(n), floating point with E and K, and character pictures.`,
    context: [],
  },
  {
    id: 'c02-attributes', title: 'PL/I data descriptor from DECLARE attributes, with defaults', type: 'draft', priority: 64, estimateMin: 20,
    target: ['lib/pli/attributes.mjs', 'test/pli-attributes.test.mjs'], kinds: [],
    spec: `Write lib/pli/attributes.mjs and test/pli-attributes.test.mjs. Export describe(attributes, { name } = {}) where attributes is a DECLARE item's attribute list: [{ name, args: token array|null, picture?: string }], names already expanded (CHARACTER, BINARY, DECIMAL, PICTURE, POINTER, VARYING, INITIAL, EXTERNAL, AUTOMATIC, CONTROLLED, DEFINED, POSITION, ...). Token shapes: { t: 'num', v } | { t: 'word', v, u } | { t: 'op', v } | { t: 'lit', v, suffix }.
Return a descriptor { type, length, varying, base, scaleKind, precision, scale, signed, picture, aligned, storage, based, defined, position, external, initial, like, dims: null, entry: { returns, options } | null, problems: [] }:
- type: 'CHARACTER'|'BIT'|'GRAPHIC'|'WIDECHAR'|'ARITHMETIC'|'PICTURE'|'POINTER'|'OFFSET'|'HANDLE'|'AREA'|'ENTRY'|'FILE'|'LABEL'|'FORMAT'|'CONDITION'|'BUILTIN'|'GENERIC'|'ORDINAL'|'TYPE'|'EVENT'|'TASK'|null (null for a structure or when nothing names a type).
- length for CHARACTER/BIT/GRAPHIC/WIDECHAR/AREA from the first argument when it is a plain number; '*' when it is '*'; the argument tokens when it is an expression; default 1 for a string with no length.
- For arithmetic: base 'BINARY'|'DECIMAL', scaleKind 'FIXED'|'FLOAT', precision and scale from (p) or (p,q) after FIXED, FLOAT, BINARY, DECIMAL or PRECISION, signed true unless UNSIGNED. Defaults when only some are given (IBM rules, RULES(IBM)): a missing base is DECIMAL; a missing scale kind is FLOAT; precision FIXED DECIMAL 5,0; FIXED BINARY 15,0; FLOAT DECIMAL 6; FLOAT BINARY 21. A name with no data attributes at all gets nothing (type null) — the implicit I-N rule is the caller's business; say so under Limits.
- aligned: true for ALIGNED, false for UNALIGNED, null when neither is given.
- storage: 'AUTOMATIC'|'STATIC'|'BASED'|'CONTROLLED'|'DEFINED'|'PARAMETER'|null; based is BASED's argument tokens (or true with none), defined is DEFINED's tokens, position POSITION's.
- external: true for EXTERNAL, the literal's value for EXTERNAL('name'), false for INTERNAL, null otherwise. initial: true when INITIAL or VALUE is present. like: LIKE's argument tokens or the reference words.
- Contradictions (CHARACTER and FIXED together, two lengths) push a problem string; never throw.
List every default and rule you applied under Limits, marking those you are unsure of as UNVERIFIED. Tests: at least 20 cases.`,
    context: [],
  },
  {
    id: 't01-lex-edges', title: 'Edge-case tests for the PL/I lexer and classifier, fixing what they find', type: 'draft', priority: 60, estimateMin: 18,
    target: ['test/pli-lex-edge.test.mjs'], kinds: [],
    spec: `Write test/pli-lex-edge.test.mjs: tests of lib/pli/lex.mjs and classify() in lib/pli/statements.mjs (both shown) against PL/I source forms an IBM Enterprise PL/I program can contain. Cover: literal suffixes ('C1'X, '0101'B, 'F'B4, 'abc'A, 'xx'XN, 'xx'GX); numbers (1E5, 1.5E-3, 101B, .5, 12_345 — mark ones you are unsure Enterprise PL/I accepts as UNVERIFIED in the title); a comment containing ';' and quotes; a string containing '/*'; operators written ^=, ¬=, ~=, <>, ¬<, ->, =>, ||, !!; compound assignments; labels with subscripts; a condition prefix list followed by labels; multiple statements on one line; a statement spanning lines; *PROCESS lines; column-1 carriage control with MARGINS(2,72); a sequence area; classify() on assignments to names that are keywords (IF, DO, CALL, END, ON), on \`IF A THEN DO;\`, \`ELSE IF X THEN Y = 1;\`, \`DO I = 1 TO N;\`, \`GO TO L;\` and DECLARE fragments.
If a test you are confident of shows the lexer or classifier is wrong, also fix lib/pli/lex.mjs or lib/pli/statements.mjs and give that file in full; say exactly what you changed and why under Limits. Do not change a behaviour the existing tests in test/pli-lex.test.mjs (shown) assert.`,
    context: [{ file: 'lib/pli/lex.mjs' }, { file: 'lib/pli/statements.mjs' }, { file: 'test/pli-lex.test.mjs' }],
  },
];

let wrote = 0;
for (const j of [...JOBS, ...SEMANTICS]) {
  const p = join(BANK, `${j.id}.json`);
  if (existsSync(p)) continue;
  writeFileSync(p, JSON.stringify(j, null, 1));
  wrote++;
}
console.log(`wrote ${wrote} jobs`);
