---
id: 0009
title: Index text is derived from content, never stored as content
status: accepted
date: 2026-10-19
---

# 0009 — Index text is derived from content, never stored as content

## Context

- The LoCoMo gate at baseline `1b85a9b5` (n=100, `RECALL_DIAG_K=100`) has 41 cases with no gold
  memory in the top 10. The miss taxonomy (`evaluations/recall/locomo-gate-miss-taxonomy.json`)
  labels 9 of them `anaphora` (the gold turn is a reply such as "I've had them for 3 years" that
  names neither subject nor object) and 8 `date_in_metadata` (the date the question asks about
  exists only in `created_at`). Both are candidate-generation misses: the turn never enters the
  pool, so no scorer can rank it.
- veld indexed the raw utterance. The embedder and the BM25 index saw the turn alone, with no date
  and nothing from the turn it answered.
- `Memory` already carries `EpisodeContext { episode_id, sequence_number, preceding_memory_id }`
  for writers that thread a conversation. Many writers, the recall harness included, do not.

## Decision

The text an index receives is **derived** at index time by `src/memory/index_text.rs` from the
record's content, its `created_at` and the turn that preceded it in the same conversation. It is
never written into `content`, into any facet, or into any other persisted field. A re-index
recomputes it from the same inputs.

- The embedding gets `[YYYY-MM-DD] {turn}\n(previous: {prior turn})`.
- The BM25 document gets the date, ISO and spelled out, plus the prior turn's entities, not its
  text: entities are what a question about the turn uses; the prior turn's full text would make
  every turn a lexical match for the one before it.
- The prior turn is `preceding_memory_id` when the writer sent one; otherwise the latest
  conversation turn sharing the conversation key (`episode_id`, else a `conv-*` tag) within the
  two-hour session gap, ordered by sequence number where both turns carry one. When the fallback
  resolves a prior, its id is written into `preceding_memory_id` so the deferred embed path and a
  later re-index derive the same text. That is threading metadata, not content.
- Every index derives from the stored record through `index_text::for_stored`: the direct BM25
  write, the intent log's BM25 projection (live, on replay, and on a replication follower, through
  the tenant's `Bm25DocumentSource`), the in-memory backfill and the deferred embed. The successor
  index that neighbour expansion reads is recorded from the same `preceding_memory_id`, so a
  follower applying the leader's snapshots builds the leader's rows.
- Linked records are contextualised: `ExperienceType::Conversation` turns, and any record its
  writer threaded into an episode (`episode_id` or `preceding_memory_id`), such as the Claude Code
  hook's task and error memories and the prompts `proactive_context` ingests. Every other record
  keeps the enriched searchable text it had. (Amended 2026-10-21: the first version contextualised
  conversation turns only, which no production writer produces.)
- The behaviour is behind the `turn_context` dial (`None` = on, `Some(false)` = off,
  `VELD_DIALS_TURN_CONTEXT`), so a gate run can measure it both ways.
- Prior-turn source is counted in `veld_turn_context_total{source}`.

## Consequences

- No `Memory` schema change. On-disk records are unchanged and the persisted-layout test stands.
- What the embedder is fed changes for linked records, so a pinned model tape would miss;
  none is pinned today (`tests/recall/model-tape.json` is absent).
- Memories embedded before this decision carry a bare-content embedding. Re-embedding them during
  consolidation, keyed on an index-text version that includes the prior turn's id, is WP 1.7.
- Each conversation write pays one prior-turn lookup (a map read, or a store `get` by id). The
  gate report's p50 is the measurement.
- Content-hash idempotency still keys on raw content: the same utterance stored twice is still one
  record, whatever its context.
