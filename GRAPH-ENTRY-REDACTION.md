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
auditing a cross-tenant attempt. Deletion is by UUID, never by name: a name is user-controlled and
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

**NOT DONE:**

1. **Sweep the existing graph.** Now unblocked by the delete route. Needs the classifiers pointed at
   stored entities. The corpus scan says the yield would be small — zero live secrets, 7 home paths
   — but "small" measures what these rules can see, not what is there.
2. **Instrument recall against the entity tail.** The open question from the cap section: if graph
   traversal never matches on a tail entity, the tail is pure cost and should be cut rather than
   filtered. Needs measurement, not argument.
