# veld v0.5.0 LOCOMO Benchmark

**Date:** 2026-03-28 | **Branch:** dev | **Hardware:** Apple Silicon, local inference

## Results

| Category | N | MRR | R@5 | R@10 | P@5 | Abs.Viol |
|---|---|---|---|---|---|---|
| Single-hop | 5 | 0.900 | 1.000 | 1.000 | 0.240 | 0 |
| Temporal | 5 | 0.650 | 0.733 | 0.900 | 0.320 | 5 |
| Multi-hop | 5 | 0.900 | 0.800 | 0.900 | 0.320 | 0 |
| Open-domain | 5 | 0.900 | 0.833 | 0.900 | 0.320 | 0 |
| **Overall** | **20** | **0.837** | **0.842** | **0.925** | **0.300** | **5/55** |

**Composite: 75.5%** (MRR=83.8% R@5=84.2% R@10=92.5% P@5=30.0% AbsCompl=90.9%)

## Per-Query Detail

| # | Type | Query | MRR | Expected | Retrieved (top-5) |
|---|---|---|---|---|---|
| Q0 | single_hop | What programming language did we choose for the backend? | 1.00 | [2] | [2, ...] |
| Q1 | single_hop | Who is the project lead? | 1.00 | [1] | [1, ...] |
| Q2 | single_hop | What database are we using? | 0.50 | [3, 18] | [16, 3, 18, ...] |
| Q3 | single_hop | What was the first bug we encountered? | 1.00 | [30] | [30, ...] |
| Q4 | single_hop | What testing framework did we pick? | 1.00 | [4] | [4, ...] |
| Q5 | temporal | What decisions did we make last week? | 0.50 | [46, 49, 52] | [..., 52, ...] |
| Q6 | temporal | What happened during the second meeting? | 1.00 | [15, 17, 18] | [15, 17, 18, ...] |
| Q7 | temporal | When did we switch from PostgreSQL to MongoDB? | 1.00 | [18] | [18, ...] |
| Q8 | temporal | What was the most recent architecture change? | 0.50 | [46, 48] | [..., 46, ...] |
| Q9 | temporal | What bugs did we find during the debugging phase? | 0.25 | [30, 31] | [..., 30, ..., 31] |
| Q10 | multi_hop | Why did we change the database AND timeline impact? | 0.50 | [18, 24] | [..., 18, ...] |
| Q11 | multi_hop | Who proposed the architecture that caused the most bugs? | 1.00 | [47, 48] | [47, 48, ...] |
| Q12 | multi_hop | What decisions were reversed and why? | 1.00 | [46, 48] | [46, ...] |
| Q13 | multi_hop | How did the testing strategy change based on bugs? | 1.00 | [35, 43] | [35, ...] |
| Q14 | multi_hop | What trade-offs about the backend language choice? | 1.00 | [2, 21] | [21, 2, ...] |
| Q15 | open_domain | Summarize the project so far | 1.00 | [0, 45, 49] | [0, 45, ...] |
| Q16 | open_domain | What are the biggest risks? | 1.00 | [11] | [11, ...] |
| Q17 | open_domain | What should we focus on next? | 1.00 | [54, 55] | [54, ..., 55] |
| Q18 | open_domain | What went well and what didn't? | 1.00 | [50, 51] | [51, ...] |
| Q19 | open_domain | What patterns in our decision-making? | 0.50 | [52, 59] | [..., 52, ...] |

## Latency

| Metric | Value |
|---|---|
| Mean | 47ms |
| p50 | 49ms |
| p99 | 98ms |

## v0.4 -> v0.5 Changes

1. **Content-fallback entity overlap** (Layer 4.525): When tag/entity overlap is 0 and query has no temporal intent, check if 2+ query stems appear in memory content (+0.05/match, cap 0.20). Fixed Q1 "project lead" and Q3 "first bug".

2. **Ordinal session resolution before category error demotion** (Layer 5.9): Wavelet-detected sessions resolve "second meeting" -> session 2 before applying the 50% demotion. Injects up to 5 session members into candidate pool. Fixed Q6 "second meeting" (was total MISS, now MRR=1.00).

3. **Punctuation stripping in ordinal extraction**: `extract_ordinal_session_ref` now strips trailing punctuation before matching session nouns. "meeting?" -> "meeting". Root cause of Q6 failure.

## Comparison

| System | Score | Local? |
|---|---|---|
| EverMemOS | ~92% | No |
| MemMachine v0.2 | 91% | No |
| Letta/MemGPT | ~83% | No |
| **veld v0.5** | **75.5%** | **Yes** |
| Zep/Graphiti | 75% | No |
| Mem0 | 67% | No |
| OpenAI Memory | 53% | No |

Scores are not directly comparable (different metrics and dataset sizes). See methodology notes in evaluations/bifocal_2026-03-28_locomo-final-reddit.md.
