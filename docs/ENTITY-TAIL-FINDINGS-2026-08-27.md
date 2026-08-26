# Does recall ever match on a tail entity?

Answers the question `GRAPH-ENTRY-REDACTION.md` left open on 2026-08-26, which it said needed
instrumentation rather than argument:

> **does recall ever match on a tail entity?** If graph traversal never surfaces one, the tail is
> pure cost — storage and disclosure with no retrieval benefit — and the answer is to cut hard
> rather than filter gently.

Measured 2026-08-27 by `probe.py` against the live store: 4,929 memories, 11,013 entities,
397 of 400 sampled memories carry 26+ tags. Probe and raw output: `bench/entity-tail/probe.py`, `bench/entity-tail/results-2026-08-27.json`.

## The answer: cut the tail hard is REFUTED, and position was the wrong axis

| bucket | index | median mentions | on 100+ memories | near-unique | no entity record |
|---|---|---|---|---|---|
| head | 0-6 | **1,977** | 73.1% | 2.2% | 454 of 2,779 |
| mid | 7-24 | 115 | 52.0% | 6.0% | 1,828 of 7,146 |
| tail | 25+ | **106** | 50.9% | 3.4% | 1,998 of 8,459 |

The tail is not the worthless part. It is **nineteen times more selective than the head** on the
median. The authored head — `hook`, `scope:claude-code`, `automated`, `session-close` — sits on
around 2,000 of 4,929 memories each and cannot pick one out of the corpus at all.

That does not make the head a defect. Those tags are scope markers and exist to FILTER, so appearing
everywhere is their design. It makes the *comparison* the point:

**`truncate(50)` cuts by arrival order, and arrival order is uncorrelated with worth.** On the 67.2%
of records that hit the cap, it keeps head tags carrying almost no information and discards whatever
arrived after position 50 — material that, by these medians, is likelier to be distinctive than what
it kept. The knob was never the cap size. **Ordering by selectivity before truncating strictly
dominates ordering by arrival**, at the same cap, with no increase in disclosure surface. That is
the actionable result and it needs no new signal: `mention_count` already exists on every entity.

The tail is still a mixture, as the original document said: half of it sits on 100+ memories and
only 3.4% is near-unique. Selectivity ordering does not make the tail good, it stops the cut from
being blind.

## Two findings that were not the question

**Semantic recall on an entity string is ~0.** 0.05 / 0.017 / 0.033 hit@20 across the three buckets.
Querying an entity's own text almost never retrieves the memory it was extracted from. Entity text
is not functioning as a semantic retrieval handle in any bucket, so the entity namespace is
currently earning its keep — if at all — through tag lookup and traversal, not through similarity.
This is consistent with `minilm.rs` truncating at 256 tokens in silence, but it does not demonstrate
that; the two would need separating.

**About a quarter of tail tags have no entity record at all** (1,998 of 8,459; 1,828 of 7,146 in
mid; 454 of 2,779 in head). `remember.rs` stores one list under two names — `entities` and `tags`
are the same `merged_entities` — so a tag with no entity is an asymmetry that should not exist.
Either the graph applies a cut the tag list does not, or entity creation fails silently for some
shapes. Not chased here. Filed rather than guessed at.

## What this does NOT measure, stated because the gap is the whole caveat

**Whether real queries ever match a tail entity.** They cannot be observed: the recall response
returns a `score` and no attribution, so nothing says WHICH tag or entity caused a match. That
instrumentation does not exist and this probe did not add it — it measures REACHABILITY, the upper
bound on usefulness. A low number here would have been conclusive. These numbers are not low, so
they bound the answer without settling it.

**A correction to this probe's own first result, kept because it is the more useful half.** The
first run reported tag-lookup hit@20 of head 26.7% / mid 45% / tail 28.3% and read as though the
head were hardest to retrieve. It is not. The head's misses were `hook` and `scope:claude-code`:
asking for 20 results from a tag that thousands of memories share will almost never return the one
you started from. That is the denominator, not reachability. **Hit@N conflates "not indexed" with
"indexed but carrying no information" — opposite problems with the same score.** The selectivity
columns exist because of that error, and the hit@N columns are retained beside them so the
conflation stays visible rather than being quietly replaced by the metric that survived it.
