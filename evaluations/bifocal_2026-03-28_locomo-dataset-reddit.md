# Bifocal+ 3-Agent + Source Audit Cross-Verification: LOCOMO Dataset Errors

**Method:** Three Claude models (Haiku, Sonnet, Opus) independently read the full 1280-line benchmark and performed exhaustive edge-walk verification. A fourth independent source-only audit reviewed the Rust fixture and scoring code directly without consulting the 3-agent report first. Results compared for consensus.

**Result:** 44 unique errors across 16/20 queries (80%). All 15 original findings confirmed unanimously. 23 new errors from agent verification + 6 new from source audit.

---

## Original 15 Errors — All Confirmed, Zero Refutations

**Unanimous (13/15):**

- **GT-1 (HIGH):** Query 5 "What decisions last week?" missing Memory[58] — 4th Decision-type memory in session 4 omitted from expected set
- **GT-2 (CRITICAL):** Query 9 "What bugs during debugging phase?" expects only [30,31] but session 3 has 10 bug memories. System returning ANY 5 valid bugs gets R@5=0.0. Highest-impact error.
- **GT-3 (MEDIUM):** Query 10 "Why change DB AND timeline?" missing Memory[49] (second timeline-impact memory)
- **GT-5 (LOW):** Query 13 "How testing changed?" missing Memory[4]. Author's own comment mentions it but omits from expected.
- **QT-1 (HIGH):** Query 2 "What database?" classified single_hop but expects 2 memories — contradicts definition
- **QT-2 (MEDIUM):** Query 9 is temporal_multi_hop, not pure temporal
- **ABS-1 (MEDIUM):** Query 9 missing Memory[48] (session-4 bugs) from absence list
- **ABS-2 (MEDIUM):** Query 6 "second meeting" missing session-1 memories from absence list
- **SCORE-1 (HIGH):** Comment says 40/30/20/10 weights, code uses 30/20/15/20/15
- **CARD-1 (MEDIUM):** Expected set sizes: single_hop mean 1.2 vs temporal 2.2 — systematic R@5 bias
- **TAG-1 (LOW):** Person-name tags inconsistent across corpus
- **TAG-2 (LOW):** Redundant "decision" tag on Memory[2,3] creates double-boosting
- **TYPE-1 (LOW):** Memory[5] milestones typed Task, better as Decision

**Confirmed with nuance (2/15):**

- **GT-4 (MEDIUM):** Query 12 "decisions reversed?" missing Memory[18]. Haiku: "context, not direct answer." Sonnet: "internal inconsistency — if [18] needed for [46], then [47] needed for [48]. Either both originals or neither." Opus: "semantically incomplete without original."
- **TYPE-2 (LOW):** Memory[38] should be Error not Learning. Haiku initially REFUTED, then confirmed after cross-checking session-3 peers.

---

## New Errors — Found by Verification Agents

### Found by all 3 agents

**NEW-ABSENCE-SCOPE (MEDIUM):** count_absence_violations checks all 10 returned results but MRR/R@5/P@5 evaluate top-5. Absence violation at rank 8 penalized equally to rank 1. Asymmetric scoring.

**NEW-Q6-NARROW (MEDIUM):** Query 6 "What happened during second meeting?" expects [15,17,18] — only 3 of 15 valid session-2 memories. Same class as GT-2. System returning [16,19,20,22,25] (all correct) gets MRR=0.0.

**NEW-Q16-STALE (MEDIUM):** Query 16 "biggest risks?" expects only [11] (4-week-old session-1 risk register). Memory[55] (tech debt, session 4) has more current risks. Session-1 risk "CRDT complexity" already resolved. System surfacing [55] at rank 1 gets MRR=0.0.

**NEW-SEPARATOR-WIDTH (LOW):** print_report line 1096 uses 78-char separator, line 1110 uses 88. Cosmetic.

### Found by 2 of 3 agents

**NEW-Q15-MEMORY59 (MEDIUM):** Query 15 "Summarize the project" missing Memory[59] (lessons learned, importance 0.85). Canonical project synthesis. (Sonnet, Opus)

**NEW-Q19-MEMORY51 (LOW):** Query 19 "patterns in decision-making?" missing Memory[51] which says "decisions made too quickly without enough evaluation." Direct pattern description. (Sonnet, Opus)

**NEW-Q5-ABSENCE-GAP (MEDIUM):** Query 5 "decisions last week" absence [0,2,3] only guards session-1. Memory[15] (importance 0.90, session 2) and Memory[18] (importance 0.95, session 2) completely unguarded. Sonnet rates HIGH. (Sonnet only found, but compelling)

**NEW-TYPE-MEMORY44 (LOW):** Memory[44] typed Task but describes completed implementation with measured results ("~2ms latency"). Better as Observation. (Sonnet, Opus)

**NEW-TYPE-MEMORY9 (MEDIUM):** Memory[9] (infra plan) typed Task but Memory[22] (observability, same "Finalized X" pattern) typed Decision. Inconsistent boundary. (Opus, Haiku round 2)

**NEW-MRR-GAMBLING (MEDIUM):** Composite weights MRR at 30%. System putting 1 correct at rank-1 + 4 garbage gets composite contribution 0.34. System with 3 correct at ranks 2-4 gets 0.27. Rewards rank-1 gambling. (Opus, Sonnet)

**NEW-EMPTY-SET (LOW):** recall_at_k returns 1.0 for empty relevant set, reciprocal_rank returns 0.0, precision_at_k returns 0.0. Inconsistent semantics. Latent. (Sonnet only)

**NEW-RECALL-CEILING (LOW):** If GT-2 fixed (expected expanded to 10), R@5 can never exceed 0.5 for that query. Fix interacts with metric design. (Sonnet only)

### Found by 1 of 3 agents only

**NEW-CORPUS-INCONSISTENCY (LOW, Sonnet):** Memory[46] says "moving analytics BACK to PostgreSQL" but no prior memory establishes analytics ever left PostgreSQL. Memory[18] only moves document storage.

**NEW-MEMORY21-TYPE (LOW, Sonnet):** Memory[21] typed Conversation but content says "Net assessment: keeping Rust because..." — this IS a decision. Conversation type with "decision" tag creates opposite problem from TAG-2.

**NEW-MEMORY53-SINGLETON (LOW, Opus):** Memory[53] is the ONLY Discovery-type memory in the corpus. Makes Discovery untestable as a type category.

**NEW-Q11-AMBIGUITY (LOW, Opus):** "Who proposed the architecture that caused the most bugs?" — CRDT decision (Memory[15]) arguably caused more bugs (30,34,38,40 = 4) than microservice split (Memory[48] lists 3). Contestable premise.

**NEW-MEMORY48-DECISION (LOW, Opus):** Memory[48] typed Error but contains "Decided to revert to the monolith." Explicit decision in Error-type content.

**NEW-CORRELATED-MEMORIES (MEDIUM, Opus):** Memory[46] in 3 expected sets, Memory[48] in 3, Memory[18] in 3. Pipeline changes on these create triple-counted effects.

**NEW-CI-BOUNDS (HIGH, Opus):** N=5 per category, MRR std dev ~0.4 gives 95% CI = +/-0.35. Category-level MRR differences under 0.35 are statistically indistinguishable from noise.

**NEW-TAG-REVERSED-REVERT (LOW, Sonnet round 1):** Memory[46] tags "reversed", Memory[48] tags "revert". Stemming may not unify.

**NEW-LATENCY-TRUNCATION (LOW, Opus):** Latency averaging casts f32 to u64, losing fractional ms. Informational.

---

## New Errors — Independent Source Audit

A separate source-only pass read the Rust benchmark file directly without consulting the 3-agent report. It confirmed 10 existing findings from the code and added 6 new source-grounded issues:

**SRC-Q2-CURRENT-STATE (HIGH):** The database query expected set [3,18] omits Memory[46], which explicitly moves analytics and audit logs back to PostgreSQL. A system returning both MongoDB and PostgreSQL evidence for the current architecture can be scored as wrong. (Lines 818-820 vs 628-634)

**SRC-Q11-PREMISE-ERROR (CRITICAL):** The query "Who proposed the architecture that caused the most bugs?" bakes in a contestable premise. The expected answer hard-codes the microservice split [47,48], but the corpus contains a larger debugging-phase bug cluster (10 bugs in session 3) unrelated to that architecture. A model that infers a different architecture as the dominant bug source is penalized despite the corpus not cleanly supporting the fixture premise. (Lines 885-889 vs 867-872 vs 654-656)

**SRC-PRECISION-K-FIXED-DENOM (MEDIUM):** precision_at_k always divides by k, even if fewer than k results are returned. This is defensible but undocumented — changes how sparse result sets are interpreted. (Lines 976-982)

**SRC-INDEX-BRITTLENESS (MEDIUM):** The benchmark relies on positional memory indices as the fixture contract. Any insertion or reorder in the corpus invalidates expected sets across the entire test. Mixes content truth with physical ordering. (Lines 72-78 vs 818-945)

**SRC-OPEN-DOMAIN-EXACTNESS (MEDIUM):** The open-domain block describes "broad, context-dependent queries" but still grades against narrow exact index sets. Turns synthesis prompts into brittle exact-match tests and under-rewards semantically valid alternative summaries. (Lines 915-945)

**SRC-TYPE-SUMMARY-NO-VARIANCE (LOW):** Type-level reporting averages MRR/Recall/Precision but exposes no variance, spread, or per-type instability signal. Category summaries can look stable while hiding catastrophic outliers. (Lines 1015-1041)

---

## Per-Query Error Map

### Query 0: "What programming language did we choose for the backend?"

    Type: single_hop | Expected: [2] | Absence: [7, 10, 44]
    Status: CLEAN — no errors found

### Query 1: "Who is the project lead?"

    Type: single_hop | Expected: [1] | Absence: [10, 29, 44]
    Status: CLEAN — no errors found

### Query 2: "What database are we using?"

    Type: single_hop | Expected: [3, 18] | Absence: [10, 14, 27]

    QT-1 (HIGH, 3/3): Classified single_hop but expects 2 memories.
        Should be multi_hop.
    SRC-Q2-CURRENT-STATE (HIGH, source audit): Expected [3,18] omits
        Memory[46] (analytics back to PostgreSQL). Current architecture
        requires synthesizing 3 memories. System returning correct
        current state can be scored as wrong.
    GT-6 (MEDIUM, 2/3): Same Memory[46] omission noted by agents.
    ABS-3 (LOW, Sonnet): Memory[19] (Redis) likely retrieval noise
        but neither expected nor absent.

### Query 3: "What was the first bug we encountered?"

    Type: single_hop | Expected: [30] | Absence: [0, 5, 14]
    Status: CLEAN — no errors found

### Query 4: "What testing framework did we pick?"

    Type: single_hop | Expected: [4] | Absence: [0, 10, 29]
    Status: CLEAN — no errors found

### Query 5: "What decisions did we make last week?"

    Type: temporal | Expected: [46, 49, 52] | Absence: [0, 2, 3]

    GT-1 (HIGH, 3/3): Missing Memory[58] (client SDK decision).
        Session 4 has 4 Decision-type memories, expected lists 3.
    NEW-ABS-5 (HIGH, Sonnet): Session 2/3 Decision memories unguarded.
        Memory[15] (importance 0.90) and Memory[18] (0.95) could
        surface for "last week" with no penalty.
    NEW-MEMORY48-DECISION (LOW, Opus): Memory[48] typed Error but
        contains "Decided to revert to the monolith."

### Query 6: "What happened during the second meeting?"

    Type: temporal | Expected: [15, 17, 18] | Absence: [30, 45, 48]

    NEW-Q6-NARROW (MEDIUM, 3/3): Expected is 3 of 15 valid session-2
        memories. System returning [16,19,20,22,25] gets MRR=0.0.
    ABS-2 (MEDIUM, 3/3): All 15 session-1 memories unguarded.
    GT-8 (MEDIUM, 2/3): Memory[24] (timeline impact, importance 0.85)
        missing from expected.

### Query 7: "When did we switch from PostgreSQL to MongoDB?"

    Type: temporal | Expected: [18] | Absence: [0, 5, 30]

    NEW-Q7-MEMORY24 (LOW, Sonnet round 1): Memory[24] directly
        references switch and timing impact. Could be in expected.

### Query 8: "What was the most recent architecture change?"

    Type: temporal | Expected: [46, 48] | Absence: [6, 12, 3]

    NEW-Q8-MEMORY47 (LOW, Sonnet round 1): Memory[47] (the proposal
        that [48] reverts) not in expected or absence.
    NEW-Q8-MEMORY18 (LOW, Sonnet round 1): Memory[18] (session-2
        architecture) not in absence despite being 3 weeks old.

### Query 9: "What bugs did we find during the debugging phase?"

    Type: temporal | Expected: [30, 31] | Absence: [0, 5, 14]

    GT-2 (CRITICAL, 3/3 + source audit): Session 3 has 10 bug
        memories. Expected lists 2. System returning ANY 5 valid
        bugs gets R@5=0.0. HIGHEST-IMPACT ERROR IN THE DATASET.
    QT-2 (MEDIUM, 3/3): Actually temporal_multi_hop. Temporal
        constraint is trivial; real difficulty is recall breadth.
    ABS-1 (MEDIUM, 3/3): Memory[48] (session-4 bugs) should be
        in absence list.
    NEW-RECALL-CEILING (LOW, Sonnet): If fixed to 10 expected,
        R@5 can never exceed 0.5. Fix interacts with metric.

### Query 10: "Why did we change the database AND what was the impact on the timeline?"

    Type: multi_hop | Expected: [18, 24] | Absence: [7, 10, 14]

    GT-3 (MEDIUM, 3/3): Memory[49] ("alpha pushed to month 3 due
        to MongoDB switch delay") missing from expected.
    NEW-CROSS-Q (MEDIUM, 2/3): Memory[49] is in Query 15 expected
        but NOT here — cross-query inconsistency.

### Query 11: "Who proposed the architecture that caused the most bugs?"

    Type: multi_hop | Expected: [47, 48] | Absence: [0, 5, 10]

    SRC-Q11-PREMISE-ERROR (CRITICAL, source audit): Query encodes
        a contestable premise. Expected hard-codes microservice split
        [47,48] but corpus has a larger bug cluster (10 session-3
        bugs) from other architecture decisions. CRDT decision
        (Memory[15]) arguably caused 4 bugs (30,34,38,40) vs
        microservice split listing 3 in Memory[48]. A model
        inferring a different dominant bug source is penalized.
    NEW-Q11-AMBIGUITY (LOW, Opus): Same issue framed as ambiguity.

### Query 12: "What decisions were reversed and why?"

    Type: multi_hop | Expected: [46, 48] | Absence: [0, 5, 14]

    GT-4 (MEDIUM, 3/3 with nuance): Memory[18] (original decision
        reversed by [46]) missing. "Why" incomplete without original.
    GT-9 (MEDIUM, Sonnet): Internal inconsistency — if [18] needed
        for [46], then [47] needed for [48]. Either both or neither.
    NEW-TAG-REVERSED-REVERT (LOW, Sonnet): "reversed" vs "revert"
        tag inconsistency between Memory[46] and [48].

### Query 13: "How did the testing strategy change based on the bugs we found?"

    Type: multi_hop | Expected: [35, 43] | Absence: [0, 10, 29]

    GT-5 (LOW, 3/3 + source audit): Memory[4] (original testing
        strategy) missing. Author's own comment on line 901 mentions
        [4] but omits from expected — self-contradicting.

### Query 14: "What trade-offs did we discuss about the backend language choice?"

    Type: multi_hop | Expected: [2, 21] | Absence: [7, 10, 44]
    Status: CLEAN — no errors found

### Query 15: "Summarize the project so far"

    Type: open_domain | Expected: [0, 45, 49] | Absence: [39, 42]

    NEW-Q15-MEMORY59 (MEDIUM, 2/3 + source audit): Memory[59]
        (lessons learned, importance 0.85) is a canonical project
        synthesis. Missing from expected.
    SRC-OPEN-DOMAIN-EXACTNESS (MEDIUM, source audit): This broad
        synthesis prompt is graded against a narrow 3-item exact
        set. Under-rewards valid alternative summaries.

### Query 16: "What are the biggest risks?"

    Type: open_domain | Expected: [11] | Absence: [14, 29]

    NEW-Q16-STALE (MEDIUM, 3/3 + source audit): Expected [11] is
        4-week-old session-1 risk register. Memory[55] (tech debt,
        session 4) more current. Session-1 "CRDT complexity" already
        resolved. System surfacing [55] at rank 1 gets MRR=0.0.
    SRC-OPEN-DOMAIN-EXACTNESS (MEDIUM, source audit): Single-item
        expected set makes this functionally a single_hop query
        inside the open_domain category.

### Query 17: "What should we focus on next?"

    Type: open_domain | Expected: [54, 55] | Absence: [0, 10]
    Status: CLEAN — no errors found

### Query 18: "What went well and what didn't?"

    Type: open_domain | Expected: [50, 51] | Absence: [3, 12]
    Status: CLEAN — no errors found

### Query 19: "What patterns do you see in our decision-making?"

    Type: open_domain | Expected: [52, 59] | Absence: [3, 9]

    NEW-Q19-MEMORY51 (LOW, 2/3 + source audit): Memory[51] says
        "decisions made too quickly without enough evaluation."
        Direct pattern description missing from expected.
    SRC-OPEN-DOMAIN-EXACTNESS (MEDIUM, source audit): Broad
        pattern-recognition prompt graded against 2-item exact set.

---

## Corpus-Level Errors

    TYPE-1 (LOW, 3/3): Memory[5] milestones typed Task, better Decision
    TYPE-2 (LOW, 3/3): Memory[38] optimization typed Learning, should
        be Error (matches session-3 peers)
    NEW-TYPE-MEMORY9 (MEDIUM, Opus): Memory[9] typed Task but
        Memory[22] (same pattern) typed Decision
    NEW-TYPE-MEMORY44 (LOW, 2/3): Memory[44] typed Task but
        describes completed implementation with measurements
    NEW-MEMORY21-TYPE (LOW, Sonnet): Memory[21] typed Conversation
        but content reconfirms Rust decision
    NEW-MEMORY53-SINGLETON (LOW, Opus): Memory[53] only Discovery
        type in corpus. Untestable category.
    TAG-1 (LOW, 3/3): Person-name tags inconsistent
    TAG-2 (LOW, 3/3): Redundant "decision" tag on Memory[2,3]
    NEW-CORPUS-INCONSISTENCY (LOW, Sonnet): Memory[46] says "BACK to
        PostgreSQL" but analytics never left PostgreSQL in corpus
    NEW-MEMORY48-DECISION (LOW, Opus): Error-typed but contains
        explicit decision
    NEW-TAG-REVERSED-REVERT (LOW, Sonnet): "reversed" vs "revert"
        tag inconsistency

---

## Scoring and Structural Errors

    SCORE-1 (HIGH, 3/3 + source audit): Comment says 40/30/20/10
        weights. Code uses 30/20/15/20/15. Five components vs four
        documented.
    NEW-CI-BOUNDS (HIGH, Opus): N=5/category, 95% CI = +/-0.35.
        Category MRR differences under 0.35 are noise.
    CARD-1 (MEDIUM, 3/3): Expected set sizes single_hop 1.2 vs
        temporal 2.2. R@5 systematically penalizes temporal.
    NEW-ABSENCE-SCOPE (MEDIUM, 3/3): Absence violations checked on
        all 10 results; positive metrics use top-5. Asymmetric.
    NEW-MRR-GAMBLING (MEDIUM, 2/3): 1 correct at rank-1 + garbage
        scores higher than 3 correct at ranks 2-4.
    NEW-CORRELATED-MEMORIES (MEDIUM, Opus): Memory[46,48,18] each
        in 3 expected sets. Pipeline changes triple-counted.
    SRC-PRECISION-K-FIXED-DENOM (MEDIUM, source audit): precision_at_k
        always divides by k even if fewer results returned.
        Defensible but undocumented.
    SRC-INDEX-BRITTLENESS (MEDIUM, source audit): Benchmark relies on
        positional memory indices. Any corpus insertion or reorder
        invalidates all expected sets.
    SRC-OPEN-DOMAIN-EXACTNESS (MEDIUM, source audit): Broad
        "context-dependent" prompts graded as narrow exact-match
        fixtures. Under-rewards valid alternative answers.
    NEW-EMPTY-SET (LOW, Sonnet): Inconsistent empty-set handling
        across scoring functions. Latent.
    NEW-RECALL-CEILING (LOW, Sonnet): Fixing GT-2 makes R@5<=0.5
        structurally. Fix interacts with metric.
    NEW-SEPARATOR-WIDTH (LOW, 2/3): 78 vs 88 char separator.
    NEW-LATENCY-TRUNCATION (LOW, Opus): f32 to u64 cast loses
        fractional ms.
    SRC-TYPE-SUMMARY-NO-VARIANCE (LOW, source audit): Type-level
        reporting publishes means without spread. Hides outliers.

---

## Agent Comparison

    Agent          | Total | Known | New  | Unique Strength
    ---------------|-------|-------|------|----------------------------
    Haiku          |  19   |  15   |   4  | Fast validator, confirms all
    Sonnet         |  21+  |  15   |  21  | Semantic subtlety, narrative
    Opus           |  34   |  15   |  19  | Structural/systemic, stats
    Source audit   |  16   |  10   |   6  | Benchmark-contract review

**What each missed:**

Haiku missed: Memory[59] for Q15, Memory[51] for Q19, Q5 absence gap, Memory[44] type, Memory[21] contradiction, corpus inconsistency, MRR gambling, statistical bounds, R@5 ceiling

Sonnet missed: Memory[9] type inconsistency, Memory[53] singleton, Q11 ambiguity, Memory[48] decision-in-Error, correlated memory effects

Opus missed: Memory[46] corpus inconsistency, Memory[21] contradiction, Q5 absence gap (Sonnet's ABS-5), R@5 ceiling, empty-set handling

Source audit missed: CI significance calculation, Memory[44]/[9]/[53] type issues, tag-vocabulary mismatches

---

## Summary

    Total unique errors:            44
    Original evaluation found:      15
    New from agent verification:    23
    New from source audit:           6
    Queries affected:               16/20 (80%)
    Clean queries:                  Q0, Q1, Q3, Q4, Q14, Q17, Q18

    CRITICAL:  2 (GT-2 unanimous, SRC-Q11-PREMISE-ERROR source audit)
    HIGH:      7
    MEDIUM:   17
    LOW:      18
