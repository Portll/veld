# Stripping secrets before graph entry

Answers the operator's question of 2026-08-26. Grounded in veld at 0.7.39+15, read this session.

## Why this has to happen on the way IN

`delete_entity()` exists at `veld/src/graph_memory.rs:2267` and **is not exposed as a route**. The
entity API is alias, attribute, merge, resolve, add, find, traverse, stats. There is no delete. And
the forget path does not touch the graph, so removing a memory does not remove the entities it
minted.

**An entity is a one-way door.** Every other control is a preference; this one is the constraint.

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

**3. Two filters, because they fail differently.**
- *Pattern*: `sk_live`/`sk_test`, `ghp_`/`github_pat_`, `AKIA`, `xox[baprs]-`, PEM headers, JWT
  shape. Catches known credential formats; blind to anything unlisted.
- *Shape*: drop a derived candidate with no whitespace, length ≥ 20, and high character entropy.
  Catches unlisted tokens; blind to anything that looks like a phrase.
Neither subsumes the other, and both are cheap.

**4. Log what was dropped, without logging the value.** Record count, source row and reason. A
filter with no output is indistinguishable from a filter nobody wired, which is this repo's most
frequent defect. Never log the dropped string.

**5. Expose `delete_entity` as a route.** Filtering only helps writes after the change. There are 20
users of a live store whose graphs already contain whatever was minted before it. Without an exit
there is no remediation path for anything already in there, and no way to act on a report.

## What this does NOT do, stated because a partial control presented as complete is the defect

- **A secret that looks like a word survives every filter.** A passphrase, a project codename used
  as a shared secret, a memorable password. Entropy and patterns both pass it. There is no
  filtering answer to this one.
- **Filtering is not containment.** The only complete control is not sending the body. That is why
  the report-spine rule stands: report bodies go to veld through `redactionCheck`; **raw scanner
  output never does**, because it is the one payload that quotes credentials by design.
- **`redactionCheck` is a caller-side refusal, not a server-side filter.** It matches `field:` and
  `field=` shapes, so a bare token in prose passes it. It is the first gate, not the only one.
- **Nothing here cleans the existing graph.** That needs the delete route plus a sweep, and the
  sweep needs a way to find candidates, which is the same two filters pointed at stored entities.

## Order to build

1. Expose `delete_entity` as a route. Without an exit, everything else is a one-way improvement to
   a one-way door.
2. Split authored from derived at assembly; filter derived; filter before truncate.
3. Pattern filter, then shape filter, then the drop log.
4. Sweep the existing graph with the same two filters and the new delete route.
