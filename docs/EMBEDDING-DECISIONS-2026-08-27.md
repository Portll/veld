# Four questions about the embedding path

Operator, 2026-08-27: *"add miniml, should ner be on? [and item] 3, should chunk size even be used?
and is it 'correct' or should it be something else? can the index be rewritten?"*

Answers, in the order the evidence forced rather than the order asked, because (4) turned out to
gate (3) and (1) turned out to be blocked by a bug.

---

## (1) MiniLM: installed — and the reason it never had been is a bug, not an omission

Both the model and the ONNX runtime were **absent**. Not stale, not partial: nothing had ever been
installed, so every vector in the store was produced by the hash-bigram fallback in `minilm.rs` —
384 floats of the right shape carrying no semantic content. That independently explains the
measurement from earlier the same day that semantic recall on an entity's own text is ~0 in every
positional bucket.

It had never been installed because **`extract_onnx_runtime` could not succeed on macOS or Linux**:

```rust
if name.ends_with(lib_name) || name.contains(lib_name) {   // lib_name = "libonnxruntime.dylib"
    entry.unpack(&dest_path)?;
    return Ok(());
}
```

The release tarball ships `libonnxruntime.dylib` as a **symlink** to `libonnxruntime.1.23.2.dylib`.
The versioned real file matches neither predicate — it does not end with, nor contain, the string
`libonnxruntime.dylib`. So the only entry that could ever match was the symlink; `unpack` faithfully
recreated a link whose target had not been extracted; the function returned `Ok` before reaching the
real library. The result was a **broken symlink at exactly the path `get_onnx_runtime_path()`
probes**, so `is_onnx_runtime_downloaded()` said false, the caller bailed with "Failed to extract
ONNX Runtime", and the archive was deleted on the way out — making every retry re-download and
re-fail identically. Linux has the same shape (`libonnxruntime.so` → `libonnxruntime.so.1.23.2`).

Fixed by requiring a regular file and matching the library STEM, so a versioned name qualifies.
Verified by deleting the broken link and re-running: runtime and model both land, and a server
started on an isolated store logs `MiniLM-L6-v2 model loaded successfully` and embeds.

Fetching is now `model-fetch --minilm`, not `veld init`. `init` is the only path that downloaded
models and it also writes config and generates an API key, which on a box with a running server and
a key in the keychain is not a setup step but a rewrite of live configuration to obtain a file.
Downloads still go through `downloader`, so URLs stay pinned to a HuggingFace commit and every file
is SHA-256 verified.

**And veld's own `tokenizer.json`, now on disk, contains `truncation: {"max_length": 128}`** —
confirming yesterday's finding against the actual file rather than a copy borrowed from another
project.

---

## (4) Yes, the index can be rewritten — but not by the route named `rebuild`

Two routes exist and they do different things:

| route | what it does |
|---|---|
| `POST /api/index/rebuild` | **skips every already-indexed memory.** A gap-filler. On a healthy store it does nothing and reports success. |
| `POST /api/index/reembed` | clears each memory's stored embedding, persists that, and re-indexes — a true re-embed. |

`rebuild_index` opens with `if indexed_ids.contains(memory_id) { skipped += 1 }`. It is resumable
repair wearing the name of a rebuild, and calling it to pick up a new embedder would return
`success: true` having changed nothing.

`reembed_all` is the one that works, and it is already routed.

### This is now REQUIRED, not optional, and the clock is running

Installing the model changed the live server's behaviour without restarting it. The store holds
**14,534 vectors** produced by the hash fallback. The running process lazily loads the model on
first use, so from the moment the files appeared, new writes get MiniLM vectors while every existing
vector stays hash-bigram — **in one index**. Cosine similarity between the two families is noise.

The live index also independently reports `degradation_percent: 100.0`, `needs_rebuild: true`
(14,533 incremental inserts against a 10,000 threshold), so it was overdue regardless.

Worse, the running binary is the **Aug 5 build**: it has neither the cleared tokenizer truncation nor
the windowed pooling, so the MiniLM vectors it is producing right now are computed from each input's
first 128 tokens. Restarting onto the current build is what makes a re-embed worth doing once.

**Operator actions, in order** — both touch a live LaunchAgent-supervised service, so neither was
taken here:

```sh
cargo build --release --bin veld && cp target/release/veld ~/.local/bin/veld
launchctl kickstart -k gui/$(id -u)/com.portll.commitwork-monitor-veld   # or the ≡ menu

curl -X POST http://127.0.0.1:3030/api/index/reembed \
     -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"user_id":"portll"}'      # 4,929 memories; not instant
```

---

## (3) Chunking should stay. 800 was wrong. It is now 480, and 400 is defensible

**Should chunk_size be used at all — yes.** The index maps one memory to *many* vectors, which is
what lets a long memory be found by its later content. Without it a 23,680-character memory is one
blurry point. Windowed pooling in `minilm.rs` removed the data LOSS but not the reason chunking
exists.

**Is 800 correct — no, and now it can be changed, because (4) says a re-embed is happening anyway.**
That was the entire objection when this was deferred yesterday: changing chunk_size moves every
boundary and invalidates the index. The index is being invalidated regardless.

`chunk-audit --sweep`, 300 real memories, real tokenizer, counting chunks that exceed the 256-token
window:

| chunk_size | chunks | over window | max tokens |
|---|---|---|---|
| 400 | 2,431 | **0** | 248 |
| 480 | 2,046 | 2 (0.1%) | 269 |
| 640 | 1,564 | 10 | 322 |
| **800** (old) | 1,278 | **58 (4.5%)** | 412 |
| 1000 | 1,043 | 373 | 440 |

The old comment reasoned "~200 tokens × 4 chars/token … leave headroom for tokenizer differences".
**The headroom ran the wrong way.** Dense text has FEWER characters per token, not more, so the
assumption fails toward chunks that are too big. At the extreme this corpus reaches under 2
chars/token — the two chunks still over the window at 480 are both raw JSON session records.

400 is the only value with zero overflow, at 1.9× the vectors. **480 costs 1.6× and leaves 0.1%**,
and since an over-length chunk is now pooled rather than truncated, that 0.1% is blur and not loss.
That trade is why it is 480. The table is in the code so the next person can choose differently from
the same numbers instead of re-deriving them.

Note what this does NOT claim: that 480 retrieves better than 800. It claims 480 fits the encoder
and 800 did not. Whether smaller chunks improve or hurt recall on this corpus is a separate
measurement that needs the re-embed to have happened first.

---

## (2) NER: no, not yet — and the reason is a defect with a fix, not a preference

`VELD_NEURAL_NER` stays unset. Three reasons, in descending weight:

**1. `ner.rs` has the identical silent truncation, at four sites, with `max_length: 128` — and for
NER it is strictly worse than it was for the embedder.** A truncated embedding is a blurred vector.
A truncated NER pass means entities past token ~128 are **never extracted at all**, on a corpus
where 54% of memories exceed 800 characters. Windowed pooling has no analogue here: you cannot
average entity spans, you have to run each window and concatenate the lists with adjusted offsets.
That is real work and it was not in scope for "add minilm".

**2. It would be a sixth derived source feeding a cap that already discards the good material.**
`merged_entities` truncates at 50 by ARRIVAL order on 67.2% of records, and yesterday's measurement
found the discarded tail is 19× more selective than the authored head it keeps. Adding a source
before fixing the ordering makes the cap bind harder and displaces more of what is worth keeping.

**3. General-purpose NER is a poor fit for this corpus.** TinyBERT fine-tuned on newswire entity
types has little to say about file paths, SHAs, lockfile names and rule ids, which is most of what
is in here. That is the weakest of the three reasons and the only one that is a judgement rather
than a measurement, so it is listed last.

**Turn it on when:** (a) `ner.rs` windows its input the way `minilm.rs` now does, and (b) entity
truncation orders by `mention_count` before it cuts. Both are measurable, so this is a condition and
not a veto.

---

## What was NOT done

- **The re-embed itself.** It needs a restart of a live LaunchAgent-supervised service holding the
  operator's memory store. Commands above.
- **`ner.rs` windowing.** Named in (2) as the condition, not implemented.
- **Selectivity-ordered entity truncation.** Named in (2) and in
  `docs/ENTITY-TAIL-FINDINGS-2026-08-27.md`, still open.
- **Whether 480 retrieves better than 800.** Unmeasured, and stated as unmeasured above.
