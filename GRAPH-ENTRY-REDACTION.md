# Stripping secrets before graph entry

Answers the operator's question of 2026-08-26. Grounded in veld at 0.7.39+15, read this session.

## Why this has to happen on the way IN

When this was written, `delete_entity()` existed at `veld/src/graph_memory.rs:2267` and **was not
exposed as a route** — the entity API was alias, attribute, merge, resolve, add, find, traverse,
stats, with no delete at all. An entity was a one-way door, and that fact drove the whole design.

**That has since been fixed** (`DELETE /api/graph/entity/delete`, see Status). The reasoning still
holds, for a narrower reason: the forget path still does not touch the graph, and the delete route
is a REMEDIATION tool — one known id at a time, after the fact, and only if somebody notices. Entry
remains the only place the decision is made by default.

## Where a secret can become an entity

`handlers/remember.rs` assembles one list and stores it under two names (`:549-550`,
`entities: merged_entities.clone(), tags: merged_entities`). It is seeded at `:475` from
`req.tags.clone()` and then accumulates, in this order:

| # | Source | Author | Risk |
|---|---|---|---|
| 1 | `req.tags` | the writer | low, it is authored |
| 2 | `req.intent_tags`, prefixed `intent:` | the writer | low |
| 3 | ingest hints: JSON keys, CSV columns, code symbols | derived | **a key named `api_key` arrives as an entity** |
| 4 | NER entities | derived | a token classified as an entity |
| 5 | extracted keywords | derived | **high entropy strings read as salient** |
| 6 | cold-start extraction, when the graph is sparse | derived | widest net, fires exactly when the graph is new |
| — | `truncate(MAX_ENTITIES_PER_MEMORY = 50)` | | see ordering below |

Rows 3 to 6 are the exposure. **Nobody chose them**, which is what separates them from row 1.

## The design

**1. Filter the DERIVED list, not the submitted one.** A writer who tags something has authored it
and may legitimately need an odd-looking token. An entity nobody chose has no such claim. Split
`merged_entities` at assembly: authored (rows 1-2) passes; derived (rows 3-6) is filtered.

**2. Filter BEFORE the truncate, not after.** Today `truncate(50)` runs on the assembled list. If a
filter runs after it, a dropped secret has already displaced a legitimate entity, and the record
loses information twice. Order: assemble → filter → truncate.

**3. Precise shapes only. The entropy rule this document originally prescribed was REFUTED by
measurement and is deliberately absent from the implementation.**

The original text here read: *"drop a derived candidate with no whitespace, length ≥ 20, and high
character entropy."* Run against 22,620 entities in 500 live records, that rule matched **415
distinct entities**, and its hits were slugs:
`detection-reachability-lexical-grammatical-semantic`,
`veld-client-5s-timeout-silently-nulls-large-writes`. It would have cost recall to catch nothing,
because the same scan found **zero live secrets** in the corpus. Every credential-shaped entity was
prose *about* a credential.

What shipped instead anchors every rule on a documented vendor prefix, a structural delimiter, or
an explicit assignment: `sk_live_`/`sk_test_`, `AKIA`, `gh[pousr]_`, `github_pat_`, `xox[baprs]-`,
`AIza`, `ya29.`, `SG.`, `npm_`, `glpat-`, `dop_v1_`; PEM headers; the JWT three-segment shape;
`scheme://user:pass@host`; `password=`/`api_key=` **carrying a real value**, with a placeholder
rule so `'password=***'` and `api_key=${API_KEY}` pass through; and `/Users/<name>`, `/home/<name>`,
which are identifying and were the only real hits in the corpus (7).

**4. Log what was dropped, without logging the value.** Record count, source row and reason. A
filter with no output is indistinguishable from a filter nobody wired, which is this repo's most
frequent defect. Never log the dropped string.

**5. Expose `delete_entity` as a route.** DONE — see Status. Filtering only helps writes after the
change, and the store's graphs already contained whatever was minted before it. Without an exit
there was no remediation path for any of it, and no way to act on a report.

## What this does NOT do, stated because a partial control presented as complete is the defect

- **A secret that looks like a word survives every filter.** A passphrase, a project codename used
  as a shared secret, a memorable password. Entropy and patterns both pass it. There is no
  filtering answer to this one.
- **Filtering is not containment.** The only complete control is not sending the body. That is why
  the report-spine rule stands: report bodies go to veld through `redactionCheck`; **raw scanner
  output never does**, because it is the one payload that quotes credentials by design.
- **`redactionCheck` is a caller-side refusal, not a server-side filter.** It matches `field:` and
  `field=` shapes, so a bare token in prose passes it. It is the first gate, not the only one.
- **Nothing here cleans the existing graph.** The delete route now makes cleaning POSSIBLE, but
  nothing has been swept: that needs the classifiers pointed at stored entities, and a route is a
  capability rather than a remediation.

## The cap is not the knob, and this is why

Measured 2026-08-26 over 500 records: **67.2% sit at exactly 50**, none above; median, p75 and p90
are all 50. About **7** of those 50 are authored by the writer (scope tags, slugs); about **43** are
derived. So the cap binds on two-thirds of the store and censors those true counts.

But raising it does not help. `merged_entities` is appended in extraction order and `truncate` keeps
the first 50, so **the cut is by arrival order, not by relevance**. The tail is a mixture: generic
words (`catch`, `difference`, `anything`, `ordinary`) sitting beside real identifiers (`ENOENT`,
`Gradle`, `SCANNER_CHECKS`). A higher cap admits more noise and more signal in the same ratio, and
everything it admits is derived — so it strictly increases the disclosure surface for no precision
gain. Cap left at 50.

The open question, which needs instrumentation rather than argument: **does recall ever match on a
tail entity?** If graph traversal never surfaces one, the tail is pure cost — storage and disclosure
with no retrieval benefit — and the answer is to cut hard rather than filter gently.

## Status

**DONE — commit `0ab568a`.** `src/redaction.rs`, wired into all five entity entry paths
(`handlers/remember.rs` ×3 including the batch `else` branch that previously bypassed everything,
`handlers/ingest.rs`, `zenoh_transport/handlers.rs`). Redaction runs BEFORE the cap at every site.
Counts and classes are logged; values never are. 12 module tests, 1,383 lib tests green.

The gate is `every_capping_path_redacts_before_it_caps`: it walks `src/`, derives the file set
rather than hardcoding it, and asserts a redact call precedes every cap site — so a sixth entry
path fails the day it lands. Verified by breaking it: removing the redact from `ingest.rs` makes it
fail by name.

UUIDs are exposed as `is_non_semantic_id()` and are **not** labelled `redacted:`, because nothing
about a session id is confidential and saying so would put a false claim in the data.

**DONE — the exit exists.** `DELETE /api/graph/entity/delete`, on the protected router so an API
key is required, with `resolve_request_user_id` binding the request to the authenticated tenant and
auditing a cross-tenant attempt — **on a build that has a tenant to bind to, which the stock build
does not. See "The tenant binding is inert as built" below; that sentence was written here on
2026-08-26 and is corrected rather than deleted, because the claim is the finding.** Deletion is by UUID, never by name: a name is user-controlled and
can collide across tenants. A missing entity returns `deleted: false` rather than an error or a
success, because "it is gone now" and "it was never here" are different facts. Every attempt is
audited even when it removes nothing, since an attempt that finds nothing is the shape a probe
makes. The entity NAME is deliberately not echoed into the event stream — this route's first use is
removing a credential-shaped entity, and previewing it would move the disclosure rather than end it.

Three tests, and the load-bearing two read the router source: the route must be MOUNTED (a unit test
on the handler would have passed for the entire time it was unreachable) and it must be mounted on
the protected builder, not the public or probe one. Verified by unmounting it and watching both
fail.

**RULED — UUID entities are ALLOWED** (operator, 2026-08-26). They stay. `is_non_semantic_id()`
remains informational, for reporting or read-time filtering, and must not be wired to a write-time
drop without a new ruling. The hub effect is a known and accepted property, not an open defect.

## The tenant binding is inert as built

Found 2026-08-27, and it corrects the paragraph above rather than adding to it.

`multi-tenant` is **not** a default feature (`default = ["storage-redb"]`). The `AuthenticatedUser`
extension that `resolve_request_user_id` needs is inserted only by `attach_bound_user_to_request`,
which is `#[cfg(feature = "multi-tenant")]`. Without the feature that argument is always `None`,
`resolve_request_user_id` returns whatever `user_id` the caller supplied, and the
`cross_tenant_rejected` audit warning is unreachable code.

Confirmed against the **deployed binary**, not the source: `strings ~/.local/bin/veld` has
`cross_tenant_rejected` and neither `BOUND_USER_MISMATCH` nor `KeyUserBindings`. Probing the live
server agrees — `POST /api/graph/entities/all` with `{"user_id":"alice"}` returns 200, as do
`/api/graph/alice/stats` and `/api/graph/alice/universe`.

So the delete route is authenticated but **not tenant-isolated on a stock build**: any valid API key
may delete any user's entity by naming them. Single-tenant is a legitimate deployment and this is
not a defect in that context. The defect is that the two regimes were indistinguishable from
outside, and that this document asserted the stronger one.

The three tests written for the delete route are unaffected: they assert the route is mounted, is on
the protected builder, and that the handler binds `user_id`. All three remain true. **None of them
could have caught this**, because each asserts a property of the source and the failure is a
property of the build — which is why the fix is not another test of the same kind:

- `print_ready_message` now prints the tenant posture at every start, derived from
  `cfg!(feature = "multi-tenant")`. Verified by running a server on an isolated store, not by
  reading the code: `Tenants: SINGLE-TENANT BUILD — any valid API key may name any user_id`.
- `resolve_request_user_id` gained the test for the branch that actually runs. Its three existing
  tests all passed `Some(&authenticated_user)` or `(None, None)`; **not one covered
  `(Some("bob"), None)`**, the only case a stock binary can reach. A green suite over an unentered
  branch, and the unentered branch was the one serving every request.

**NOT DONE:**

1. ~~Sweep the existing graph.~~ **DONE — see "What the sweep found" below.**
2. **Instrument recall against the entity tail.** The open question from the cap section: if graph
   traversal never matches on a tail entity, the tail is pure cost and should be cut rather than
   filtered. Needs measurement, not argument.

## What the sweep found

`src/bin/graph_sweep.rs` (`graph-sweep`), run 2026-08-27 over **11,041 of 11,013** entities — the
whole population; the store grew by 28 between the stats call and the dump.

It calls `veld::redaction::classify`, the same function the entry path calls. A sweep with its own
patterns would drift from the guard and then certify the guard's blind spot as a clean graph.

| class | n | entity | first seen |
|---|---|---|---|
| `redacted:key` | 1 | `AKIAIOSFODNN7EXAMPLE` | 2026-08-05 |
| `redacted:login` | 1 | `postgres://user:SECRET@host` | 2026-08-06 |
| `redacted:path` | 1 | `cwd=/Users/portll` | 2026-08-05 |

The first is AWS's own published example key; the second says `SECRET` where a password goes. Both
are documentation. The third is real and identifying, and it is the operator's own username in the
operator's own store. All three predate the redaction work, so they are genuine store contents
rather than this exercise's own tailings.

**The 500-record sample's "zero live secrets" now holds over the population.** 599 entities are
UUID-named (ruled allowed 2026-08-26; counted, never planned for deletion).

The tool never deletes and never holds a key: it reads a dump and writes a plan of uuids, and
feeding that plan to the delete route stays a human act. `--graph-total` is mandatory, which is the
point of the tool as much as the classifying is — `entities/all` silently defaults to `limit: 100`,
so a dump taken without one covers 0.9% of this graph and looks exactly like a full one. Run that
way it reports **0 flagged**, prints "no credential-shaped entity in the 100 READ. The other 10,913
are unexamined, and this is not a statement about them", and exits 2. Every headline number degrades
gracefully under sampling except zero.
