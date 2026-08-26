# Keep the whole length

Operator instruction, 2026-08-27: *"keep whole length, don't truncate."*

This records what was actually truncating, what it cost, what changed, and the three things that
are still true and were not fixed.

## The finding: the configured window was never the real one

`EmbeddingConfig::max_length` is `256`. `all-MiniLM-L6-v2` ships `tokenizer.json` containing:

```json
"truncation": {"max_length": 128, "strategy": "LongestFirst", "stride": 0}
```

`MiniLMEmbedder` called `tokenizer.encode(text, true)` and let that stand. So the tokenizer cut at
**128** before `max_length` was ever consulted, `256` never bound, and the `take(max_length)` below
it was dead code. Two ceilings, one of them written down and the other one in force.

Downstream, `chunking.rs` sized chunks at 800 characters, commented *"~200 tokens \* 4 chars/token
... leave headroom for tokenizer differences"* — headroom against a limit that was half what it
believed, using a density this corpus does not have.

## What it cost, measured

`src/bin/chunk_audit.rs`, over 300 memories (1,278 chunks), with the real `chunk_text` and the real
tokenizer:

| against | chunks over the window | records affected | tokens dropped |
|---|---|---|---|
| the configured 256 | 58 (4.5%) | 44 (14.7%) | 1,170 |
| **the real 128** | **1,153 (90.2%)** | **258 (86.0%)** | **82,300** |

Chunk lengths: median 195, p90 240, p99 280, max 412 tokens. The densest chunks run **2.5-3.0
chars/token** — paths, JSON, frontmatter, SHAs — against the assumed 4.0.

`ChunkResult::coverage_ratio()` returned a hardcoded `1.0` throughout, with the comment "With
chunking, we cover everything".

**The module built to stop content being silently dropped was dropping content from 86% of
records**, and the metric that existed to notice could not fall below 1.0.

### The first run of the audit was wrong in the corpus's favour, and the tell was in the output

It reported *"0 over the window — every chunk fits"*. The tokenizer it loaded to measure with had
the same 128-token truncation configured, so every chunk came back at exactly 128 tokens and none
exceeded 256.

Median, p90, p99 and max were all `128`. **That is not a distribution, it is a ruler with one mark
on it** — and it is the only reason the false clean got caught. The audit now clears the
tokenizer's truncation before measuring and refuses to report a result where every chunk has an
identical length, because a clamp in the instrument and a property of the text produce the same
number and only one of them is worth reporting.

## What changed

**1. veld owns its window.** `LazyModel::new` clears the shipped truncation with
`with_truncation(None)` and logs what it cleared. A downloaded JSON no longer imposes a ceiling
behind the configured one.

**2. `encode` embeds the whole input.** `tokens.iter().take(max_length)` is replaced by windowed
mean pooling: the sequence is split into windows, each is run, and the pooling numerator and
denominator accumulate across all of them before a single divide. Mean pooling decomposes across
windows exactly, so **text that fits in one window embeds bit-identically to before** — only
content that was previously being truncated changes, which is what keeps the existing vector index
usable.

It is not equivalent to a longer-context model, and the code says so: windows past the first begin
without a real `[CLS]` and their position embeddings restart, so cross-window attention does not
happen. It is an honest average of the whole text rather than an exact reading of it. That is the
difference between approximate and absent.

A 64-window ceiling **refuses** rather than truncating, because an unbounded loop over
caller-supplied text is a denial of service and the one thing this function must not do is return
an embedding of part of its input while reporting it as the whole.

**3. The batch path was fixed too, not just the one that was looked at.** `encode_batch` builds one
fixed `[batch, max_length]` tensor and had the identical `take`. A batch containing over-length text
now routes to the windowed path. Otherwise batched and unbatched callers would silently disagree
about the same input, with only one of them tested.

**4. `coverage_ratio()` measures.** It returns covered bytes over original bytes, where covered is
the union of the spans the chunker actually emitted. The chunk loop *can* leave a gap — it breaks on
`start >= end` and force-advances when the overlap arithmetic stalls — and the old constant could
not have reported either.

## The tests, and why they are these tests

The property that makes "keep the whole length" true is that the windows **tile** the token
sequence. `window_spans` is extracted from the inference loop so that can be checked with no model,
no tokenizer and no ONNX runtime — which matters, because this machine has no MiniLM model files at
all, and a test that only runs where the model exists is a test that does not run here.

`windows_tile_the_sequence_exactly_leaving_nothing_out` reconstructs coverage by visiting every
index, across 6 window sizes × 14 sequence lengths, and asserts **dropped and duplicated
separately**: a token in no window is the original bug, a token in two windows is the bug a naive
overlap fix introduces, and only one of those is the direction that lies to you. They do not share
an assertion.

Verified by reintroducing the defect: replacing the window range with `0..1` fails
`windows_tile_the_sequence_exactly_leaving_nothing_out` by name, with *"1 token(s) in NO window —
this is the silent truncation the windowing exists to remove"*, and also fails
`text_within_the_window_still_takes_exactly_one_pass`. Restored, 1,391 lib tests green.

The `coverage_ratio` test previously asserted `== 1.0` against a function that returned `1.0`
unconditionally — the metric and its only test agreeing by construction. It now also asserts the
value can **fall**, on a constructed partial result.

## Three things still true, stated so they are not mistaken for fixed

**1. There is no MiniLM model on this machine.** `find` turns up the cross-encoder and nothing else,
so the running server is not using this code path at all — it is on the hash-bigram fallback in the
same file. That is consistent with the independent measurement earlier the same day that semantic
recall on an entity's own text is ~0 in every positional bucket. **The truncation fix is correct and
currently unexercised here**, and the fallback embedder is the more urgent finding for retrieval
quality. Not chased in this change.

**2. `ner.rs` still truncates the same way**, at four sites. Neural NER is off by default on this
deployment (`VELD_NEURAL_NER` unset), so it is not currently dropping anything, but it is the same
defect in a file that was not part of this instruction.

**3. `chunk_size` is still 800 characters against a measured 2.5-3.0 chars/token.** With windowed
pooling this is a granularity cost rather than a correctness one: an over-length chunk is blurred
across two windows instead of cut in half. It is left alone deliberately — changing it moves every
chunk boundary in the store and invalidates the vector index, which is an operator's decision, not
a side effect of a comment turning out to be wrong.

## Reproducing

```sh
# 0x1E-delimited memory bodies; NOT committed — an audit fixture that quotes the store
# discloses the store.
cargo run --bin chunk-audit -- --corpus /tmp/corpus.txt \
    --tokenizer ~/…/minilm-l6/tokenizer.json --max-length 256
```

Exits 1 when any chunk exceeds the window, so it can gate rather than merely report. It is
operator-run and **not wired into CI**, because CI here has no model files and a test that silently
skips is worth less than a command someone runs.
