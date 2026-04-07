# Bifocal+ Multi-Agent Audit of a LOCOMO Benchmark Dataset: 44 Findings, Then a Defense Attorney Tore Half of Them Apart

## Background

We built a 60-memory, 20-query LOCOMO-style retrieval benchmark for [veld](https://github.com/pnavez/veld), a persistent memory system for AI agents. The benchmark tests retrieval across four query types (single-hop, temporal, multi-hop, open-domain) with MRR, Recall@5, Recall@10, Precision@5, and absence violation metrics.

We then ran a Bifocal+ edge-walk evaluation to find errors in the dataset itself. Three Claude models (Haiku, Sonnet, Opus) independently audited every line. A fourth independent source-only pass reviewed the Rust fixture code without seeing the other reports. That produced 44 findings across 16 of 20 queries.

Then we ran an adversarial defense pass: an Opus agent argued FOR the benchmark author's choices on every single question, scored each defense and criticism 0-10, and delivered a verdict. The defense overturned or downgraded roughly 10 findings, leaving 34 actionable.

This post covers all of it.

---

## Method

**Phase 1 -- Initial Bifocal+ evaluation.** One Opus agent read the full 1280-line benchmark and produced 15 error findings with HAZOP/FMEA analysis.

**Phase 2 -- 3-agent cross-verification.** Haiku, Sonnet, and Opus each independently re-read the dataset and the initial evaluation. Every finding was tagged CONFIRMED, REFUTED, or NEW.

**Phase 3 -- Independent source audit.** A separate pass read only the Rust source, not the prior reports. Confirmed 10 existing findings, added 6 new.

**Phase 4 -- Adversarial defense.** Four parallel Opus agents acted as defense attorneys for queries 0-4, 5-9, 10-14, and 15-19. Each argued the benchmark author's design was correct, scored both sides, and delivered verdicts.

---

## The Scorecard: Defense vs Criticism on Every Query

    Q   Defense  Crit  Verdict           Key Defense Argument
    --  -------  ----  ----------------  ------------------------------------
    0   10       0     DEFENSE WINS      Clean. No criticism exists.
    1   10       0     DEFENSE WINS      Clean. No criticism exists.
    2    6       6     SPLIT             single_hop = reasoning pattern, not
                                         cardinality. Memory[46] is analytics
                                         workload, not core database.
    3   10       0     DEFENSE WINS      Clean. No criticism exists.
    4   10       0     DEFENSE WINS      Clean. No criticism exists.
    5    6       5     SPLIT             Expected is editorial curation of
                                         most important decisions. But
                                         absence gap for session 2/3 is real.
    6    8       3     DEFENSE WINS      Expected = 3 highest-importance
                                         session-2 memories. Expanding to 15
                                         caps R@5 at 0.33, destroys the
                                         benchmark's discriminative power.
    7    9       1     DEFENSE WINS      Query asks "when." Memory[18] IS the
                                         event. Memory[24] is impact.
    8    7       2     DEFENSE WINS      Memory[47] is a proposal, not a
                                         change. Absence list is adequate.
    9    2       9     CRITICISM WINS    Indefensible. "What bugs" is open
                                         plural. 10 valid answers, 2
                                         expected. Perfect retrieval scores 0.
    10   6       5     SPLIT             Memory[49] conflates two causes.
                                         Author tests clean chain [18->24].
    11   7       4     DEFENSE WINS      Corpus has exactly ONE explicit
                                         architecture->bugs causal chain.
                                         CRDT bugs are implementation bugs.
    12   8       3     DEFENSE WINS      Reversal memories self-contain both
                                         WHAT was reversed and WHY. Originals
                                         are context, not answers.
    13   5       5     SPLIT             Author tests for the CHANGE, not the
                                         baseline. Comment mentions [4] but
                                         code excludes it. Thin distinction.
    14  10       0     DEFENSE WINS      Clean. No criticism exists.
    15   6       5     SPLIT             Expected = structural skeleton
                                         (origin, status, trajectory).
                                         Memory[59] is reflection. Minor gap.
    16   8       3     DEFENSE WINS      Memory[11] is the ONLY explicit risk
                                         assessment. Memory[55] is tech debt,
                                         not risk. 3/4 risks still active.
    17   9       0     DEFENSE WINS      Clean. No criticism exists.
    18  10       0     DEFENSE WINS      Clean. Perfect 1:1 mapping.
    19   7       3     DEFENSE WINS      Expected captures synthesized
                                         insights. Memory[51] is raw
                                         complaint, not pattern analysis.

    Result              Count  Queries
    ------------------  -----  --------------------------
    DEFENSE WINS           12  Q0,Q1,Q3,Q4,Q6,Q7,Q8,Q11,
                               Q12,Q14,Q17,Q18
    SPLIT DECISION          6  Q2,Q5,Q10,Q13,Q15,Q19
    CRITICISM WINS          2  Q9

    Defense avg:  7.6 / 10
    Criticism avg: 2.7 / 10

---

## Findings That Survived the Defense

### CRITICAL (1)

**GT-2: Query 9 "What bugs did we find during the debugging phase?"**
Expected [30,31] but session 3 has 10 Error-type bug memories. The query says "what bugs" (open-ended plural). A system returning [32,33,34,36,37] -- all valid debugging-phase bugs -- scores R@5=0.0, MRR=0.0, P@5=0.0.

Defense scored itself 2/10. The author's comment says "two earliest bugs" but the query text does not say "first" or "earliest." A perfect retrieval system gets zero on every metric. This is the single most damaging error in the dataset. All three agents plus the source audit confirmed it.

### HIGH (5)

**GT-1: Query 5 "What decisions last week?" missing Memory[58].**
Session 4 has 4 Decision-type memories [46,49,52,58]. Expected lists 3. Defense argues Memory[58] (importance 0.75) is a lower-priority tactical decision, and IR benchmarks routinely use incomplete relevance pools. Split -- the gap is real but the editorial argument has merit.

**QT-1: Query 2 "What database?" classified single_hop but expects 2 memories.**
Defense argues single_hop classifies the reasoning pattern (direct fact lookup), not the answer cardinality. Like "What countries border France?" being single-hop despite multiple answers. Split -- genuinely debatable, but violates the benchmark's own apparent convention.

**SCORE-1: Composite formula comment/code weight mismatch.**
Comment on line 1126 says "40% MRR + 30% Recall@5 + 20% Precision@5 + 10% absence compliance." Code on line 1132 uses 30/20/15/20/15 with five components. Defense did not contest. Factual error.

**SRC-Q2-CURRENT-STATE: Query 2 expected [3,18] omits Memory[46].**
Memory[46] moves analytics back to PostgreSQL. Defense argues this is analytics workload routing, not the core "what database" answer. Split.

**NEW-CI-BOUNDS: N=5 per category gives 95% CI of +/-0.35 on MRR.**
Category-level differences under 0.35 are statistically indistinguishable from noise. Opus was the only agent to compute this. Defense did not contest.

### MEDIUM (11)

**GT-3: Query 10 missing Memory[49].** Memory[49] names "MongoDB switch delay" but also cites microservices. Defense: author tests the clean causal chain [18->24]. Split.

**QT-2: Query 9 is temporal_multi_hop.** Temporal constraint is trivial; real difficulty is recall breadth. Confirmed alongside GT-2.

**ABS-1: Query 9 missing Memory[48] in absence.** Session-4 bugs should not count for session-3 query. Defense did not contest.

**ABS-2: Query 6 missing session-1 memories in absence.** Defense: session-1 is temporally adjacent, session-3/4 contamination is the primary failure mode. Partially defended.

**CARD-1: Expected set size asymmetry.** Single_hop mean 1.2, temporal 2.2. Systematically penalizes temporal R@5. Confirmed.

**NEW-ABSENCE-SCOPE: Absence violations checked on all 10 results; positive metrics use top-5.** Violation at rank 8 penalized same as rank 1. All three agents found this.

**NEW-Q5-ABSENCE-GAP: Query 5 absence only guards session-1.** Memory[15] (importance 0.90, session 2) completely unguarded for a "last week" query. Defense conceded the absence gap.

**NEW-MRR-GAMBLING: Composite rewards rank-1 gambling.** 1 correct at rank-1 plus 4 garbage (contribution 0.34) beats 3 correct at ranks 2-4 (contribution 0.27). Confirmed.

**NEW-CORRELATED-MEMORIES: Memory[46,48,18] each in 3 expected sets.** Pipeline changes on these create triple-counted effects. Opus only.

**SRC-PRECISION-K-FIXED-DENOM: P@k divides by k even if fewer results returned.** Defensible convention but undocumented.

**SRC-INDEX-BRITTLENESS: Benchmark uses positional indices as fixture contract.** Any corpus insertion invalidates all expected sets.

---

## Findings Overturned by the Defense

These were in the original 44 but the defense successfully argued them down:

**SRC-Q11-PREMISE-ERROR: Downgraded from CRITICAL to LOW.**
The criticism said Query 11 "Who proposed the architecture that caused the most bugs?" bakes in a contestable premise because CRDT bugs outnumber microservice bugs. The defense demolished this: the corpus has exactly ONE memory (Memory[48]) that explicitly says an architecture "exposed several bugs." The CRDT bugs in session 3 are attributed to implementation issues (missing bounds check, off-by-one error), not to the architectural decision to build a custom CRDT. No memory says "the CRDT algorithm choice caused bugs." The benchmark follows the explicit causal chain the corpus provides. Scoring this CRITICAL was the single most overblown finding in the analysis.

**NEW-Q6-NARROW: Withdrawn.**
The criticism said Query 6 expected [15,17,18] is only 3 of 15 valid session-2 memories. The defense: these are the three HIGHEST-IMPORTANCE memories in session 2 (0.90, 0.85, 0.95). A system that retrieves five session-2 memories but misses all three of the most important ones genuinely has a retrieval problem. Expanding to 15 expected would cap R@5 at 0.33 and destroy discriminative power. This is how TREC and MS MARCO work -- you test for the hardest targets, not all valid answers.

**GT-4: Downgraded from MEDIUM to LOW.**
The criticism said Query 12 needs Memory[18] (original MongoDB decision) for completeness. The defense: Memory[46] says "Reversed part of the MongoDB decision... MongoDB is great for CRDT operations but its aggregation pipeline is too complex for the analytics queries we need." Both the reversal and its reason are self-contained. The original decision is context, not the answer.

**NEW-Q16-STALE: Downgraded from MEDIUM to LOW.**
The criticism said Memory[11] (risk assessment) is stale. The defense: 3 of 4 risks listed in Memory[11] are still active (Rust hiring, WebSocket scaling below 10K threshold, SOC2 timeline uncertain). Only "CRDT complexity" has been resolved. Memory[55] (tech debt) is categorically different from risk -- debt is known deferred work, risk is uncertainty about future outcomes.

**NEW-Q19-MEMORY51: Withdrawn.**
The criticism said Memory[51] ("decisions made too quickly") is a missing pattern description. The defense: Memory[51] is a raw observation; Memory[52] (RFC process adopted) and Memory[59] (distilled lessons) are the synthesized pattern insights. The benchmark rewards retrieval of higher-order abstractions over lower-order complaints. Correct design choice.

**SRC-OPEN-DOMAIN-EXACTNESS: Withdrawn.**
The criticism said broad synthesis prompts should not be graded against narrow exact sets. The defense: every IR benchmark in existence (MS MARCO, BEIR, Natural Questions, TREC) uses fixed relevant sets. The alternative -- human judgment per run -- is an evaluation study, not a benchmark. This criticism reflects a methodological misunderstanding, not a dataset error.

---

## Findings Nobody Contested

These survived both the multi-agent audit AND the defense without challenge:

    SCORE-1: Comment says 40/30/20/10, code does 30/20/15/20/15
    ABS-1: Query 9 absence list missing Memory[48]
    NEW-CI-BOUNDS: N=5/category, CI +/-0.35 -- categories are noise
    NEW-ABSENCE-SCOPE: Absence checks all 10 results vs top-5 metrics
    CARD-1: Expected set size asymmetry biases R@5 against temporal
    SRC-INDEX-BRITTLENESS: Positional indices as fixture contract
    TAG-1: Person-name tags inconsistent across corpus
    TAG-2: Redundant "decision" tag on Memory[2,3]
    TYPE-1: Memory[5] typed Task, should be Decision
    TYPE-2: Memory[38] typed Learning, should be Error

---

## Corpus-Level Errors (all LOW, none contested)

    TYPE-1: Memory[5] milestones typed Task, better Decision
    TYPE-2: Memory[38] optimization typed Learning, should be Error
    NEW-TYPE-MEMORY9: Memory[9] typed Task but Memory[22] (same
        "Finalized X" pattern) typed Decision
    NEW-TYPE-MEMORY44: Memory[44] typed Task but describes completed
        implementation with measurements
    NEW-MEMORY21-TYPE: Memory[21] typed Conversation but content
        reconfirms Rust decision
    NEW-MEMORY53-SINGLETON: Memory[53] only Discovery type in corpus
    TAG-1: Person-name tags inconsistent
    TAG-2: Redundant "decision" tag on Memory[2,3]
    NEW-CORPUS-INCONSISTENCY: Memory[46] says "BACK to PostgreSQL"
        but analytics never left PostgreSQL in corpus
    NEW-MEMORY48-DECISION: Error-typed but contains explicit decision
    NEW-TAG-REVERSED-REVERT: "reversed" vs "revert" tag inconsistency

---

## Scoring and Structural Errors

    SCORE-1 (HIGH): Comment says 40/30/20/10 weights. Code uses
        30/20/15/20/15. Five components vs four documented.
    NEW-CI-BOUNDS (HIGH): N=5/category, 95% CI = +/-0.35.
        Category MRR differences under 0.35 are noise.
    CARD-1 (MEDIUM): Expected set sizes single_hop 1.2 vs
        temporal 2.2. R@5 systematically penalizes temporal.
    NEW-ABSENCE-SCOPE (MEDIUM): Absence violations checked on
        all 10 results; positive metrics use top-5.
    NEW-MRR-GAMBLING (MEDIUM): 1 correct at rank-1 + garbage
        scores higher than 3 correct at ranks 2-4.
    NEW-CORRELATED-MEMORIES (MEDIUM): Memory[46,48,18] each
        in 3 expected sets. Pipeline changes triple-counted.
    SRC-PRECISION-K-FIXED-DENOM (MEDIUM): P@k divides by k
        even if fewer results returned. Undocumented.
    SRC-INDEX-BRITTLENESS (MEDIUM): Positional memory indices
        as fixture contract. Any corpus edit breaks everything.
    NEW-EMPTY-SET (LOW): Inconsistent empty-set handling across
        scoring functions. Latent.
    NEW-RECALL-CEILING (LOW): Fixing GT-2 makes R@5 <= 0.5
        structurally. Fix interacts with metric.
    NEW-SEPARATOR-WIDTH (LOW): 78 vs 88 char separator. Cosmetic.
    NEW-LATENCY-TRUNCATION (LOW): f32 to u64 cast. Informational.
    SRC-TYPE-SUMMARY-NO-VARIANCE (LOW): Category means hide outliers.

---

## Agent Performance Comparison

    Agent          Total  Known  New  Unique Strength
    -------------  -----  -----  ---  ----------------------------
    Haiku            19     15    4   Fast validator, confirms all
    Sonnet           21+    15   21   Semantic subtlety, narrative
    Opus             34     15   19   Structural/systemic, stats
    Source audit     16     10    6   Benchmark-contract review
    Defense pass     20      -    -   Overturned 10 of 44 findings

**What each missed:**

Haiku: Memory[59] for Q15, Q5 absence gap, Memory[21] contradiction, MRR gambling, statistical bounds, R@5 ceiling (9 misses)

Sonnet: Memory[9] type inconsistency, Memory[53] singleton, Q11 ambiguity, correlated memory effects (4 misses)

Opus: Memory[46] corpus inconsistency, Memory[21] contradiction, Q5 absence gap, R@5 ceiling, empty-set handling (5 misses)

Source audit: CI significance, Memory type issues, tag mismatches (3 misses)

Defense: conceded GT-2 at 2/10 self-score (1 concession on the biggest finding)

---

## Final Numbers

    Original findings:              44
    Overturned by defense:          10
    Remaining actionable:           34
    Queries affected:               12/20 (60%)
    Clean queries:                  Q0,Q1,Q3,Q4,Q7,Q8,Q14,Q17,Q18

    After defense:
        CRITICAL:   1 (GT-2 only)
        HIGH:       5
        MEDIUM:    11
        LOW:       17

The consensus was right on the structural issues: GT-2 is genuinely broken, the composite formula has a factual error, absence checking is asymmetric, and N=5 per category is statistically insufficient. But the consensus significantly overreached on content completeness: the Q6, Q11, Q12, Q16, and Q19 criticisms applied a "return all valid answers" standard that no IR benchmark uses and that would actively degrade discriminative power if adopted.

The single biggest correction from the defense: SRC-Q11-PREMISE-ERROR dropping from CRITICAL to LOW. The corpus encodes exactly one explicit architecture-to-bugs causal chain. The benchmark follows it. That is correct.
