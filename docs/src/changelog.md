<!-- GENERATED FILE — do not edit by hand.
     Source: git log + tags
     Generator: docs/generators/src/bin/gen-changelog.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-changelog`) -->

# Changelog

Generated from git history. Sections are conventional-commit prefixes; hashes link to the commit on origin.

## Unreleased

### acl

- acl: the ACL store takes a key-value handle, not a database (2ca8b046)

### b

- B: long_term_memory is a trait object, and the trait stopped lying about its width (d8a0a5a2)

### backup

- backup: the five backup operations work on redb, built on the migration dump (e5cc043a)
- backup: the engine handle is captured at construction, and get_db() is gone (cb492712)

### bench

- bench(qa): mem0 ingest prints a line per session (fe364d77)
- bench(qa): offline tests for budgets, Wilson intervals, retries and the mem0 adapter (d92f4f6b)
- bench(qa): mem0 OSS as the memory under test, 2.2.1 and 1.0.11 from PyPI (1f6708d5)
- bench(qa): LM Studio readers, retried errors, context budgets and Wilson intervals (c8236fd6)
- bench(spaces): memory growth per space, measured with the physical footprint (91ac0607)
- bench(spaces): embedding_spaces measured at fc38cc63 and at bedbe286 (13320b8b)
- bench(qa): OpenAI-compatible readers and judges, and a pinned scoring clock (ff9cf46c)
- bench: end-to-end LoCoMo answer accuracy (J), and LongMemEval with Claude as reader and judge (e141db49)
- bench(longmemeval): say why the judge counts any 'yes': the official evaluator does (19ecf266)
- bench(pinned): 174 of 321 questions were unlocatable under the old labels, not 173 (180c0ce4)
- bench(pinned): gold sessions come from LoCoMo's evidence, and the population says what it is (ac08958d)
- bench: the HTTP engine comparison can run storage-bound on the offline hash embedder (b7301af8)
- bench(spaces): per-space search, footprint, write and drain harness (abca8bec)
- bench: a store-size latency lane that can see O(N)-per-request paths (0556bf29)
- bench: durable pinned recall harness + gold labels (locomo-mc10) (#36) (6cc29f2c)

### build

- build: the local profile drops LTO and compiles incrementally, so an edit installs in 22 s (8e2acbf0)
- build: the install loop takes two minutes, not sixteen, and CI stops compiling the graph twice (5c4413ec)
- build: dev and test builds carry line tables for veld and no debug info for dependencies (28316395)
- build: the release profile unwinds, so cargo test --release can link and a panic stays on its task (031e13cb)
- build: RocksDB is the default engine; both engines build by default and CI tests a redb-only build (470584a2)
- build: RocksDB is optional behind legacy-rocksdb; the default build runs on redb only (886d4b40)
- build(windows-service): harden -Configure race + add shawl/WinSW backends (1cb2d2bf)

### chore

- chore(integrations): @veld/vercel-ai ships its licence and lockfile, and the crewai extra stops below 1.16 (21d44b77)
- chore(tests): tier 2 routes the rocksdb-only leg and the slow recall lane by changed paths (f369b83b)
- chore(gate): a verdict carried across a rebase reruns the cheap legs only, and a busy machine is not red (a2b210b9)
- chore(deps): cargo update across the workspace and its sub-crates, clearing the reachable advisories (7335afb8)
- chore(tests): a_declared_file_path_is_recalled_by_its_file_name is known red until #79 (66e37c28)
- chore(tests): tier 2 runs CI's test legs: nextest, the doctests, a bench build (5c03ce88)
- chore(ci): bump the actions group across 1 directory with 7 updates (f458da5b)
- chore(deps): update maturin requirement from <2.0,>=1.5 to >=1.15.0,<2.0 (c936bcd8)
- chore(deps): update openai-agents requirement from >=0.0.7 to >=0.24.0 (6dcc767a)
- chore(deps): update pydantic requirement from >=2.0.0 to >=2.14.1 (caf2219f)
- chore(deps): move tantivy from 0.25 to 0.26.2 (7b92de3b)
- chore(deps): move the telemetry feature to opentelemetry 0.32 (db6193e2)
- chore(deps): move rmcp from 1.8 to 2.2 (0d4111b4)
- chore(deps): ratatui 0.30 in tui, status-tui and status-widget (65aa86af)
- chore(deps): update backoff requirement from >=2.2.0 to >=2.2.1 (37eff38d)
- chore(tests): run tests locally in tiers, through a sandboxing runner (6b2a33f6)
- chore(deps): bump criterion from 0.5.1 to 0.8.2 (f90e5b11)
- chore(deps): bump tower-http from 0.6.11 to 0.7.1 (3b6c8d82)
- chore(deps): bump vitest from 4.1.12 to 5.0.4 in /mcp-server (5b72bc12)
- chore(deps): bump @types/node from 25.9.9 to 26.6.5 in /mcp-server (31ca67b8)
- chore(deps): bump @vitest/coverage-v8 in /mcp-server (fa5db2bb)
- chore: stop tracking locomo_results.json (57e009df)
- chore(deps): vitest 4.1.11 in mcp-server (c6e247be)
- chore(deps): patch rustls, quinn-proto and git2 in the client crates (4fe1897a)
- chore(deps): bump git2 from 0.19.0 to 0.20.4 in /tui (01aafe1f)
- chore(deps): bump open from 10.2.0 to 11.0.4 in /veld-telegram (81d95ea4)
- chore(deps): update huggingface-hub requirement from >=0.20.0 to >=2.2.0 (758caeac)
- chore(deps): Dependabot version updates for actions, Cargo, npm and pip (6301e775)
- chore(storage): the branch's RocksDB types are imported once, not spelled per use (c752c345)
- chore(repo): stop publishing third-party correspondence, another app's audit, binaries and logs (4609ea34)
- chore: Cargo.lock records the crate at 0.8.3, as Cargo.toml on main declares (8552015c)
- chore(loops): bring the loop producers in from ~/.claude/scratch as found (bb02ccac)
- chore: gitignore shodh runtime-data directories (#21) (39335537)
- chore(bench): exclude bench output from git + funnel it to a temp/output dir (a7bfeb26)
- chore: bump version to 0.7.28 (772503c9)
- chore(memory): delete orphan src/memory/maintenance.rs (undeclared dead code; FIX-R1 expiry now ported into live run_maintenance) (6e4a7951)
- chore: bump version 0.7.20 -> 0.7.21 (60c7a64e)
- chore(clippy): resolve all lib warnings (fixes + targeted allows) (69862477)
- chore(lib): apply cargo fix — drop unused import + mark unused binding (3700aa50)
- chore(docs): move veld-vs-shodh-comparison.html to ../veld-docs/ (e8b5e58f)
- chore: remove duplicate rlm_refiner.rs; fix comparison HTML; update generated docs (23b5c775)
- chore(gitignore): exclude entire evaluations/ directory (f9d887d1)
- chore(gitignore): exclude benchmark result snapshots from git (d6cfedb8)
- chore(locomo): warm-veld result snapshot (MRR 0.917) (b9d2b920)
- chore(carry): include llm_reranker.rs to unblock compile (8579177e)

### ci

- ci(nextest): the engine-parity binary gets thirty minutes per test, not fifteen (bff68abf)
- ci: run tests with nextest, and compile Argon2 optimised in test builds (27ed007d)
- ci: scripts/ci-local.sh is what green means before a push that starts CI (468bdea6)
- ci(recall): fetch a 100-deep rank list per case so a miss can be told from a low rank (0a841a9b)
- ci: the test suite runs on every push and pull request again (42e7041a)
- ci(recall): give the held-out suite the job maximum of 360 minutes (06415c9a)
- ci(recall): test the diff renderer and let a dispatch run the held-out suite (0e092619)
- ci(recall): let a dispatch run every layer rung (e433c434)
- ci: clippy denies the correctness lints on both builds (7034e1a8)
- ci(recall): compare the engines' scores and leave their latencies out (6f94ec8e)
- ci(recall): cache the cross-encoder model so every repeat ranks with the same one (2e252be8)
- ci(recall): the recall gate measures veld's shipped NER, the rule-based one (b20256a1)
- ci(upstream): CI fails when the upstream watcher has not succeeded for 48 hours (7c9f6e9b)
- ci(upstream): a daily watcher appends new upstream commits and release tags to the ledger (4023c72e)
- ci(clippy): the clippy step fails when clippy does; three vacuous test asserts check something (1fa2b8fc)
- CI: tests get the ONNX runtime and MiniLM, and run on tags and dispatch (19e5bb5a)
- CI: tests build without debug info, and a failing test fails the job (15315362)
- CI: the Linux release build runs on release tags and dispatch, not every push (ab24cf0f)
- CI: macOS and Windows builds on release tags and manual dispatch, not every push; cancel superseded runs (58b969eb)
- ci(recall): add evidence-recall + latency regression guard (#22) (c8d328c5)
- ci: fix invalid YAML in version-check (heredoc dedented out of block scalar) (02a40442)
- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (30ca3578)
- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (aff3e120)

### cli

- cli: the MCP and hook clients match the API, and --help stops printing secrets (bac4f35c)

### compression

- compression: the semantic pass is additive, and a destroyed record says so instead of returning its summary (f621ce2e)
- compression: the default path was lossy, and it labelled itself lossless (b2785b75)

### config

- config: request rocksdb, because redb is unfinished and every request for it is served rocksdb (d8b6128d)

### cross-encoder

- cross-encoder: serialise downloads per cache directory so the pre-warm and the first scorer cannot race on one .part file (13300c83)
- cross-encoder: serialise downloads per cache directory so the pre-warm and the first scorer cannot race on one .part file (94c57bdc)
- cross-encoder: pin ms-marco-MiniLM-L-6-v2 and record every reranker's licence (2844f3bf)
- cross-encoder: pre-warm only downloads, on one worker, and never builds an ONNX session (93cd9d85)
- cross-encoder: pad a batch to its longest pair, not to 512 tokens (e7ac41d8)
- cross-encoder: the exit segfault was a detached pre-warm thread, not rocksdb (d350c5bc)

### deps

- deps: clear rmcp, lopdf and pyo3 RustSec advisories (4730852d)
- deps: clear rustls and sqlx RustSec advisories (0e8e2116)
- deps: hold zenoh at 1.8.0 so twox-hash 1.6.3 and lz4_flex 0.10.0 leave the lock (fafac7a3)
- deps: cargo update across the three lockfiles — 26 advisories drop to 9, and the 9 that remain need manifest bumps (#43) (29bbe86c)

### dials

- dials: rerank_model selects the cross-encoder at runtime (13c6fc2e)

### docs

- docs(claude): operating rules for token efficiency: one task per session, hand off at about 150k, no waiting on verdicts in a session (14e4801d)
- docs(memory): owner_user_id does not cover the Python bindings (5b83b194)
- docs(upstream): ledger rows for the PPR graph leg port (6de4908b)
- docs(python): CrewAI and Vercel AI SDK beside LangChain and LlamaIndex (7d4f92c6)
- docs(examples): CrewAI and Vercel AI SDK examples (82c315bc)
- docs(tests): tier 1 runs a module's tests and the handler suite as separate calls (14279b41)
- docs: the container image is ghcr.io/portll/veld, not varunveld/veld (e7425a95)
- docs(upstream): ledger rows for the FLAT fusion port (#320, in #325) and the boost conversion (#174) (09e93455)
- docs(http-api): aliases are listed as aliases (5ff5f4d7)
- docs(gen-schemas): lenient JSON bodies and named path params (76238d4d)
- docs(adr 0007): soak days 2 to 6 are clean on the replica (0052bbe3)
- docs: the generated schema reference pages, and type names link from table cells (bf619556)
- docs: a generated schema reference for every HTTP route (ddf45ce3)
- docs(tls): state when status-core reads VELD_CA_CERT (ce92feb9)
- docs(tls): the network-access page states what the listener refuses, reloads and allows; the restart test names tls_url (63881adf)
- docs: reaching veld from another machine, and how each client trusts it (4b8af672)
- docs(claude): the push gate, its verdict command and its recorded override (3ad02a9a)
- docs(adr 0007): soak day 1 is clean; the 16 GiB drop is side-data compaction, not loss (9087b20c)
- docs(intent-log): segments, their backup and restore, and VELD_INTENT_LOG_SEGMENT_MB (2148b579)
- docs(adr 0007): the redb live soak log starts 2026-10-21T13:29Z on the :3031 replica (8d039c9a)
- docs(progress): the embedding-spaces branch, built and unmeasured (de6fb3a2)
- docs: say what the LoCoMo figures measure, and that the gate's questions are in them (b8644b8a)
- docs: withdraw the MRR 0.925 and 0.917 claims and cite the LoCoMo run (c061d1ab)
- docs(constants): state the directional boosts' approach to 1.0 exactly (e77e3d92)
- docs: the arctic-embed-xs embedder, its licence and how to run it as a gate arm (a63c9a49)
- docs(architecture): live replication page — routes, follower loop, coalescing, limits, measured rates (166df555)
- docs(graph): retract a measurement error in the hygiene module comment (41ea0ab2)
- docs(upstream): give every W0 row an outcome and name its veld commits (66527fcb)
- docs(upstream): ledger rows for W5 name their veld commits (287603c2)
- docs(upstream): the pr-133 row records its two ports (59c3b750)
- docs(upstream): ledger rows for W1 name their veld commits (2bc9e659)
- docs(decisions): ADR 0008 sets out what taking upstream's local IPC transport would mean (b99efebf)
- docs(upstream): the #325 row names the rebased #321 commit (04084a1d)
- docs(upstream): the #325 row records the hebbian default from #321 (ef3e2ec3)
- docs(upstream): #462's repeat-stable graph ordering is ported in part (e352465c)
- docs(upstream): the PMI gate (#336) belongs to the graph wave (5a8f0c8c)
- docs(upstream): triage the upstream commits filed under no wave (96f875cb)
- docs(upstream): latency budgets from a live clone, gate status, and #399 re-read (0eb73e38)
- docs(upstream): veld tracks shodh-memory: the plan, ADR 0007 and a ledger of every upstream commit (9de91307)
- docs: parity status ledger at 2026-09-16 — landed, in flight, and findings that are not port bugs (3ef9e7e7)
- docs(adr): 0006 — backend parity and on-demand migration (8768654e)
- docs(recall): four documented entrypoints that do not exist, and a gate nothing reads (975504a7)
- docs: the evaluation engine is spine, not sleight (8528ac0f)
- docs: handoff for the remaining parity work and the attribution defect (8cda75c1)
- docs: dependency-ordered execution prompt for the rest of redb parity (0207a859)
- docs(encryption): confidentiality boundary + production sign-off (H4) (c1a36fa7)
- docs(sidecar): gen-cli-ref + versioned subdirs + embedded-docs release + CLAUDE.md sections (f86217e8)
- docs(sidecar): gen-changelog + lychee CI + smoke-test + consolidation accuracy (0fb3019f)
- docs(refresh): post-dials regeneration + dials + instance-registry pages (1063818f)
- docs(reference): regenerate config.md (92 env vars) (fd1979ab)
- docs(guides): add relational-backends operator guide (3f7e174a)
- docs(comparison): correct and extend veld-vs-shodh-comparison.html (07003352)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (4dfb215a)
- docs(sidecar): polish generators, content, and plugin set (debe86c3)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (ade3bd54)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (6944852d)
- docs: Phase A+B mdBook skeleton with all reference stubs (3fd082f2)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (07893df4)
- docs(facts): SECURITY.md + decision record 0005 (f6a2fb3e)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (0d8a0249)
- docs(sidecar): polish generators, content, and plugin set (0032e784)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (77ba4923)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (a2f9337a)
- docs: Phase A+B mdBook skeleton with all reference stubs (c36f4099)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (9264abbb)

### embeddings

- embeddings: snowflake-arctic-embed-xs as a primary embedder selected by VELD_EMBEDDER (9775c404)
- embeddings: nomic over HTTP is a primary, chosen deliberately, and the store knows which (f2de3014)
- embeddings: the corpus marker recorded a dimension, so a model swap at 384 was invisible (ebe7f853)
- embeddings: stop degrading silently, and stop inventing entity positions (5c9e6908)
- embeddings: the configured window was 256, the tokenizer cut at 128, and the module built to stop that was feeding it (70a5c8ea)

### facets

- facets: a stored Place can be read back (4cbffcc3)
- facets: serde(default) covers JSON, not the bincode store (b388bd98)

### facts

- facts: a purge reason is externally tagged on disk, and tagged-layout records still decode (a946567f)
- facts: a purged fact reads back purged (9fbf9434)

### feat

- feat(storage): read memory records in a zstd value envelope on both engines (13f074ea)
- feat(crewai): raise_on_error makes a failed recall raise instead of returning no memories (ef8e8890)
- feat(replication): a follower names the unreplicated data it serves (8e7cabf1)
- feat(replication): a follower reports how long since it reached its leader (8cbe77d7)
- feat(replication): the leader keeps a view of its followers (8c61a513)
- feat(replication): a follower is a read-only replica (e0fd066d)
- feat(recall): the Layer 5 graph-strength term is an ablation family, off by default (2df3880f)
- feat(integrations): Vercel AI SDK tools and prompt memory, @veld/vercel-ai (f6de8cc7)
- feat(recall): the default graph leg walks personalized PageRank and ranks by relevance (034b0212)
- feat(integrations): CrewAI memory backed by veld (8317d338)
- feat(recall): Layer 0.6's untimed entity fallback is a signal that can be switched off, for a paired run (598849f0)
- feat(recall): a graph_candidates_only dial, where graph hits join the pool without a score of their own (4e885867)
- feat(recall): Layer 4 fuses by calibrated max (upstream FLAT, #320); reciprocal rank stays behind the fusion dial (5d818908)
- feat(tags): the derived budget defaults to 50, cutting nothing; the shape filter ships alone (79967302)
- feat(tags): a record keeps twelve derived tags, the rarest, and none that cannot name a thing (d117e36a)
- feat(intent-log): the journal purge keeps every Forget a replication follower has not read yet (94baabe2)
- feat(intent-log): erase forgotten memories from sealed segments after a configured window (c8668382)
- feat(tui): Spine tab (7): open, install or update from GitHub releases, uninstall (d352eb87)
- feat(keys): a request over the TLS listener counts as private transport (21061e66)
- feat(server): start the TLS listener beside the plain one and advertise it (0aa74e54)
- feat(tls): a native TLS listener and its VELD_TLS_* settings (3d2babcf)
- feat(tui): every request trusts VELD_CA_CERT, and a bad bundle stops the dashboard before it opens (2e3c027d)
- feat(status-core): carry the registry's tls_url for display (218b24a1)
- feat(status-core): every client trusts VELD_CA_CERT, and a bad bundle is an error (41be2aee)
- feat(replication): trust VELD_CA_CERT roots and present a client certificate to the leader (b1b1013d)
- feat(gap): incremental graph mirror, paced gap worker and scoped analysis (537aa151)
- feat(graph): log the uuids of entity and relationship writes (90657c90)
- feat(loops): queue the PL/I lane's jobs in spine (0ae31a82)
- feat(intent-log): live journals rotate at 256 MiB, and backup, restore, replication and the CLI read segments (2025cdd6)
- feat(intent-log): a log rotates its head into sealed segments and reads across them (31f7ff35)
- feat(http): GET /api/memory/by-external-id reads a record back by the caller's key (a05fb2d0)
- feat(hooks): the memory hook scopes its surfacing and its writes to the session's repository (5ce19a45)
- feat(scope): memories carry the repository they were written in, and recall can be held to it (32f5ad5f)
- feat(spaces): a follower with another embedder converges, then searches its own space (e2b621a3)
- feat(spaces): every space but the primary holds codes, not a second copy of its vectors (16a5e535)
- feat(spaces): the query once per space, every space's leg beside the rest of recall (53b40743)
- feat(spaces): writes queue for batched embedding; no request waits on it (84cf17db)
- feat(recall): blend the cross-encoder by query shape, and weigh query-term coverage (db166749)
- feat(spaces): a change of embedder is a resumable background drain (7a4e345e)
- feat(readme): screenshots from the demo profile (833fb07f)
- feat(spaces): per-space fusion weights, calibration and attribution (ceef3920)
- feat(monitor): server uptime from the registry, per-tenant replication lag, tray lag line (dc68b5f3)
- feat(spaces): the space crosses every boundary a vector does (fa24549f)
- feat(spaces): alignment as a bridge between spaces, fitted from the corpus (244b8987)
- feat(cli): veld restart [--port N] restarts a server the way the monitor does (87a3a49c)
- feat(spaces): one index per embedding space (6d776e1f)
- feat(spaces): caches, chunking and a registry keyed by space (f9e0d776)
- feat(spaces): every stored vector names its embedding space (dea6fdda)
- feat(demo): a publishable demo profile and a screenshot harness (93a53c0a)
- feat(brand): savannah grass mark / grass sheaf (cec9e0d7)
- feat(brand): savannah grass mark / grass sheaf (7ce7edde)
- feat(graph): key synapse strengthening on its source (45cbbcca)
- feat(memories): exact tag filter on listing, and a filtered count (e8139b15)
- feat(monitor): add a Security tab for keys, recovery and logins (eb775c8d)
- feat(admin): administer logins and the keyring over HTTP behind 2FA step-up (5e4f6093)
- feat(keys): unlock the keyring with Touch ID or a passkey (b2ea038c)
- feat(keys): add a keyring with named recovery codes, device factors and rotation (b593e1fe)
- feat(memory): an embedding backlog that drains itself, and a fallback embedder host (9ab92250)
- feat(cli): add reseal-journal to seal what a passphrase-only tenant wrote in plaintext (bda247e2)
- feat(recall): resolve declared file paths from a query, keep identifiers whole, trust NER less on code (730e5c8e)
- feat(graph): typed entities declared by the writer, and a gate on generic derived ones (58ab5b1a)
- feat(graph): census of tags that sit on exactly the same records (7fdea715)
- feat(graph): hygiene shelf on every entity row, and mentions per day or week (8740cca7)
- feat(graph): classify entity names onto cast, vocabulary and noise shelves (3cf07a21)
- feat(ui): read-only /api/ui/* contracts for the integrated GUI (e76725b2)
- feat(memory): record each stored tag's source, and log every recall (ca2a7ae1)
- feat(upsert): carry episode, agent and writer provenance on upsert (2976c591)
- feat(api): let /api/recall run one rung of the layer ladder (4eff1288)
- feat(recall): replay a pinned model tape in the gate, and refuse to compare outputs from two sources (7d964609)
- feat(recall-harness): record every model output of a gate run, so a later run can replay it (e31e5111)
- feat(recall-harness): refuse to gate a run reranked differently from its baseline (fab07f23)
- feat(recall): name the cases that moved against a per-case gate baseline (832caf61)
- feat(recall-harness): pair every ablation delta against its reference, with a bootstrap interval (0a6a716a)
- feat(recall-harness): report which entity labels carry the graph, and the top hubs by name (dc886367)
- feat(migrate): carry sealed records and the keystore between engines (714637ec)
- feat(server): background writes run on a task tracker, and shutdown waits for them (4dd628e4)
- feat(tui): add a Loops view of the self-evaluation loops (18e12402)
- feat(upstream): the ledger check fails on a port commit that does not name its upstream source (c02f40d1)
- feat(storage): the semantic merge pass and its tombstone GC run on redb, as one implementation for both engines (56455a35)
- feat(recall): the recall gate runs: recall-eval, a weekly workflow per storage engine, and an engines-equal check (99001be8)
- feat(storage): record-level encryption (encryption-v2) on redb (cbd7ace2)
- feat(upstream): a translator that turns an upstream commit into a veld-shaped patch and reports what applies (796257b2)
- feat(embeddings): embed over-long text in pieces, and serve the embedder on its own llama-server (95fedbb3)
- feat(gap): gap analysis runs in the background and serves its kept result (7d63718d)
- feat(memory): record who wrote a memory, and let a reader see it (40d581ef)
- feat(api): add /api/spine/dimensions, retain /api/sleight/dimensions as alias (9d18b4d2)
- feat(temporal): intent-verb expansion + word-boundary precision + tense routing (#37) (4c08a1e8)
- feat(temporal): extraction patterns — explicit dates, "in N", pronoun-entity fallback (coverage 65%->~85%) (#35) (b900b081)
- feat(recall): surface B4 derived temporal answer on /api/recall (eval-path fix) (#34) (dea3267a)
- feat(temporal): answer-derivation surface — return derived date for temporal questions (B4) (#32) (9f32543c)
- feat(dials): runtime disable-signals dial for per-signal recall ablation (#31) (04c87eb4)
- feat(temporal): query-time event-anchor resolution for anchor-relative dates (#27) (5bb70a82)
- feat(temporal): anchor-relative date resolution (the <weekday> before/after <date>) (#26) (4f842300)
- feat(memory): semantic merge via tombstone+audit, lazy remap (flag-gated, default off) (#24) (a0bf40a6)
- feat(recall): shadow learned-weight injection + feedback update plumbing (live OFF) (#23) (6b0e14dd)
- feat(graph): path-credit reinforcement + bounded spontaneous recovery (#20) (f4300811)
- feat(gc): inbound-ref index covers SemanticFact + TemporalFact citations (Fold 3) (#19) (4e521dbb)
- feat(dials): expose recall ranking as runtime sliders (bd11fa85)
- feat(temporal): resolve before/after <named-event> via second hop (write-time, cycle-safe) (#15) (5a9ecc94)
- feat(temporal): inferential After/Before/Fuzzy bounds (since/after/before/around) (#14) (69040bd1)
- feat(memory): R3a — persist learned ranker weights (no behavior change) (17f65d71)
- feat(memory): wave 1 substrate — prefetch, reverse index, LTR shadow capture (ab8da515)
- feat(temporal): validate temporal facts on event-time via bound algebra (#9) (7db03695)
- feat(crypto): bind primary records to their identity (anti-relocation AAD) (7de27fcf)
- feat(intent-log): encrypt journal frame payloads at rest (da93205a)
- feat(crypto): generalized AAD identity context — bind TOTP + slow-store secrets (620161ed)
- feat(slow_store): per-column encryption toggles (payload / graph / embeddings) (b0acecdc)
- feat(storage): operator override to accept a restored (older) keystore (f6fdd57f)
- feat(keyctl): recovery-code recovery flow — recovery codes are now usable (54832ef3)
- feat(crypto): multi-epoch record read — data-key rotation no longer orphans data (fe654d15)
- feat(crypto): harden slow-store + TOTP secrets onto the v2 keystore (53ada71a)
- feat(keyctl): veld-keyctl rotation CLI (H5) (d45d807b)
- feat(keystore): XChaCha20-Poly1305 record encryption (H3) — removes nonce ceiling (8f0130c5)
- feat(keystore): integrity MAC + rollback guard (H2) (2e3ff7ab)
- feat(oblivious,pir): encryption-v2 P6 + P7 — oblivious-access seam + 2-server PIR (08aa0803)
- feat(keystore,storage): encryption-v2 P5 — pluggable KMS unseal provider (081514a1)
- feat(keystore): encryption-v2 P4 — passphrase rotation, recovery code, DEK rotation (9d2d1dcb)
- feat(storage): encryption-v2 P3(b) — finish exact-match index blinding + rebuild-index (9c0ad1bb)
- feat(keystore,storage): encryption-v2 P3(a) — index blinding (tag + entity) (2e04b08d)
- feat(storage): encryption-v2 P2(b) — record-level encryption wiring (record-only) (144400cf)
- feat(keystore): encryption-v2 P2(a) — record envelope primitive (8d94e474)
- feat(keystore): encryption-v2 P1 — envelope keystore crypto core (e94ef7c1)
- feat(monitor): custom frameless title bar with baseline-aligned controls (5e2bdc1f)
- feat(monitor): directory browser for first-run path inputs (d9c3ec7f)
- feat(cross-encoder): autodetect output head + input layout from ONNX session; swap BGE to apache-2.0 O4 export (174976cc)
- feat(handlers): LenientJson extractor for /api/remember (cc7e2f59)
- feat(cli): veld stop --port — stop a running server by its port (7ddf1949)
- feat(slow-store): encrypt the memory payload at rest (parity with the primary store) (a63a5c50)
- feat(alignment): v2 engine — payload hardening, conditioning, whitened-Procrustes, CSLS, perf-flag config (145e9ee3)
- feat(cross-encoder): per-model rerank_top_k + VELD_RERANK_DEPTH override + CPU-O3 BGE export (73fdff93)
- feat(monitor): first-run setup panel (seed a project + check Claude hooks) (2f8233b7)
- feat(dials): make max_active_shadows + working_memory_threshold runtime dials (b8a6ff9a)
- feat(monitor): multi-instance discovery + tabs (consumes the registry) (55d1f8d8)
- feat(wiki): live edit-back watcher — optional, default-off [W-003/004 live layer] (2b1152df)
- feat(memory): file_answer + anti-autophagy, merge/split with char-safe ranges [A5/W-022/J3/W-021] (2a98a2f4)
- feat(wiki): edit-back — Obsidian edits flow back to the canonical store [W-003/004] (e6e05633)
- feat(wiki): authed POST /api/wiki/render endpoint [W-018] (8ee9370c)
- feat(wiki): WikiProjection foundation — wiki render as an intent-log projection [W-007/008/013/014/016] (f8b887a8)
- feat(c3.1): surface injection_suspected for non-trusted-source content [W-006] (8336a6a3)
- feat(c2,c4,c5): store↔log durability — reconcile-on-startup, Lsn ordering, bulk-forget journaling (8f46b2f0)
- feat(hooks): statusline discovers live instances from the registry (113b29f0)
- feat(monitor): expose the graph viewer through Tauri (812bfa9a)
- feat(graph-view): add manual Refresh button to the server graph viewer (5b1c40dd)
- feat(storage): implement redb primary/graph/audit stores (baecf4cc)
- feat(c3): surface source-trust at recall (injection boundary, W-006) (e68b34d3)
- feat(c1): per-user write-path serialization lock (W-005) (50cbaf61)
- feat(p4): iterative re-cueing (pattern completion) + clippy sweep (927c1883)
- feat(r3): populate EngramBinding conjunctive binding + surface reliability (60a6592b)
- feat(r6): bi-temporal as-of point-in-time queries (722ebc2b)
- feat(discovery): server-authored instance registry (replaces port-guessing) (38ac57d6)
- feat(hooks): statusline accepts URL list as CLI arg; wire dual-server in settings (4a8aedb7)
- feat(hooks): wire Claude session → veld so the monitor sees live sessions (fa595a00)
- feat(monitor): polish dials UI — ranges, tooltips, descriptions (7df15680)
- feat(monitor): Dials panel in Tauri tray monitor (get_dials/set_dials commands + status-core helpers + frontend) (a0b28695)
- feat(dials): persist runtime dials via KeyValueStore (rocksdb kv CF accessor + load-on-ctor/persist-on-set) (0a9a3aa3)
- feat(storage): implement real redb KeyValueStore (was a stub) + round-trip/scan/persist test (966a5988)
- feat(tui): add Dials tab (view + nav); documents reconsolidation/retrieval dials and how to set them (5a31c417)
- feat(dials): MCP get_dials/set_dials tools (rmcp Rust + TS); client get/put (b13c21da)
- feat(dials): GET/PUT /api/dials handlers (protected, per-user, sanitized) (6fe0a98f)
- feat(dials): per-query retrieval overrides (refiner/rerank/candidate/rrf) via attached dials handle (28124f56)
- feat(reconsolidation): Layers 2 & 3 re-encode — entity enrich + embedding nudge (dial-gated, single re-index) (56819752)
- feat(reconsolidation): Layer 1 re-encode — strengthen graph edges to retrieval-context entities (dial-gated) (7c95724e)
- feat(reconsolidation): run_reconsolidation_expiry Layer 0 (drain+decay), gated; called from run_maintenance (aaf0b535)
- feat(reconsolidation): accumulate retrieval_context across labile window; window length from dial (6440cfd6)
- feat(dials): seed CognitiveDials from VELD_* env at construction (7e1b17b6)
- feat(dials): wire CognitiveDials runtime state onto MemorySystem (33b34e9d)
- feat(dials): add CognitiveDials runtime tuning struct (inert by default) (d41f7a3b)
- feat(e6): full TCM context_drift — graded temporal-context contiguity (bc07ee59)
- feat(cross-encoder): default to mxbai-xsmall + add jina-tiny/turbo MiniLM updates (a25b79e9)
- feat(m4): return query-level entity coverage + weighted-available FOK (92468291)
- feat(m3.1): cross-embedder agreement signal for query-level FOK (06b3282c)
- feat(cross-encoder): register mxbai-rerank-{xsmall,base}-v1 small-model options (cf83b218)
- feat(cross-encoder): registry-driven catalogue + BGE-reranker-v2-m3 default (d0c77634)
- feat(m3): return query-level feeling-of-knowing (FOK) on recall (ac3ad32b)
- feat(cross-encoder): pluggable trait + ensemble support (4386011a)
- feat(m5): wire gap_topology persistence + scope-filtered query-in-gap (feca30f8)
- feat(w4): MssqlRelationalStore backend via tiberius (gated behind mssql feature) (32516bfc)
- feat(w7): dataset row CDC events (7fb9e35b)
- feat(w7): dataset MCP tools (rmcp Rust + TypeScript) (415affd2)
- feat(w7): tabular ingest mode — CSV/JSON -> relational dataset (838f8177)
- feat(w4): wire a configured relational backend at startup (cutover activation) (05a313f8)
- feat(storage): ErasedRelationalStore — type-erase any backend to Error=BoxError (a53e0482)
- feat(w4): memories projection auto-uses the configured relational store (fa78330b)
- feat(w4): SupabaseRelationalStore — thin Postgres wrapper (gated) (8d76038f)
- feat(w4): PostgresRelationalStore backend (gated behind postgres feature) (2ccf6ff7)
- feat(w4): wire the memories projection through the RelationalStore trait (gated) (c9c1e2a2)
- feat(w4): BlockingRelationalStore — sync->async bridge for the projection cutover (e96c2f54)
- feat(w4): extend RelationalSlowStoreAdapter to the memories projection table (b7828288)
- feat(w6): /api/query/plan + /api/query/execute planner HTTP surface (cb488739)
- feat(w6): RealGraphQuerier over entity/episode topology (4811c4b6)
- feat(w6): real VamanaProvider for MultiUserMemoryManager via live writer (0827fc24)
- feat(w6): RealRelationalQuerier + RealVectorQuerier + StubGraphQuerier (no HTTP) (b7f60d60)
- feat(sleep-time): V1 stage 2 + V2 stage 1 — orchestrator, worker, handlers, R27/R42/R43/R46/R48/R66 (89b0ad77)
- feat(vamana): public memory_id_for(vector_id) reverse accessor (7ccacdef)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (6bc1758d)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (c964fa4a)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (1ec77f42)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (af44de12)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (7f5318d4)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (86abcbe9)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (859e7e08)
- feat(facts): destructive purge_facts under FACT_PURGE_LOCKS + reaper (b2feaf75)
- feat(facts): facts_preview_purge — bucketed, audit-logged, read-only (8924802a)
- feat(facts): purged_at field + is_active filter + fact_narratives MCP tool (63153cd0)
- feat(w4): RelationalSlowStoreAdapter — parallel path proving trait against slow-store DDL (a9a188ee)
- feat(w7): RelationalDatasetStore + LinkStore + HTTP CRUD (29fc74f2)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (dd71162c)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (c0448404)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (ef637cd7)
- feat(w6): query planner data model + selectivity planner + executor skeleton (foundation) (290e2c9d)
- feat(w7): Dataset schema + DatasetStore trait + in-memory impl (foundation) (8c6933fd)
- feat(w4): RelationalStore trait + SqliteRelationalStore (foundation) (ef131b5f)
- feat(vamana): multi-embedding-kind projections + persist memory_id map via CheckpointStore side data (a8d579f1)
- feat(bm25): unify retrieval-side BM25Index with projection-owned Arc<BM25Index> (c77231bb)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (400e23d1)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (7f75c694)
- feat(intent-log): BM25 as third real TypedProjection (1296d977)
- feat(intent-log): Vamana as second real TypedProjection (17d6665d)
- feat(intent-log): SQLite as first real TypedProjection — journaled writes + replay (ce6bbea9)
- feat(user-auth): admin promote/demote + persistent throttle + session revocation (f558df99)
- feat(memory): wire AgentSession into writer paths with detection helper (72997a67)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (6ed867d9)
- feat: wire ACL into reads, LLM into consolidation, schema-constrained extraction (#3) (7fa4ee1c)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (2d8a0427)
- feat(alignment): corpus expansion tooling (MDN/cpython/Rust/React/TS) + multi-lambda Ridge sweep (af74d511)
- feat(alignment): Procrustes + Ridge fitters, fit/eval bins, retrieval integration (#2) (090166d5)
- feat(hooks): write .veld-agent-session.<pid> marker on SessionStart (5160687a)
- feat: agentic-memory feature parity — bi-temporal, self-paging, HNSW, ranker, connectors, LLM hook, ACL (#1) (e56dd8d4)

### feedback

- feedback: second secondary store off Arc<DB>, and a discarded flush error (d2a7ab1d)

### files

- files: third secondary store off Arc<DB>, first user of ColumnStore (569f5118)

### fix

- fix(replication): journal the leader's reinforce, retire and stamp rewrites (eee7e295)
- fix(remember): store and journal a remember in one blocking task (937fbf90)
- fix(replication): journal the leader's re-embed and compression rewrites (7fbe7604)
- fix(upsert): store and journal an upsert in one blocking task (68328171)
- fix(replication): the graph leg strengthens no edge while the instance follows a leader (963aa08f)
- fix(spaces): a re-embed keeps the replaced vector when its slot names its space (ce87fa6c)
- fix(graph): a repeat source that moves nothing is not written back (51d68d07)
- fix(recall): every graph read on the recall path is recursive (da4b7a79)
- fix(recall): fusion credits the graph leg's graph strength, not its relevance score (c6cc2edb)
- fix(integrations): the adapters include untrusted memories as labelled, escaped data by default; strict mode withholds them (6cbdf5ee)
- fix(replication): a follower erases forgotten memories from its own journal (c6df78b8)
- fix(recall): the shipped fusion is reciprocal rank; FLAT stays behind the dial (72f8f303)
- fix(vercel-ai): recalled memories go in the user turn, escaped and labelled by trust, and untrusted ones are withheld by default (5add6a7a)
- fix(crewai): recall carries veld's trust fields, withholds untrusted memories and stays in the adapter's tags (705b5897)
- fix(replication): keys never reach Debug output (58d9b8f1)
- fix(replication): the leader records a follower's lag after its reply (19bf8be2)
- fix(replication): no writer but the applier writes on a follower (ff0d314f)
- fix(crewai): warn before sending the API key over plain HTTP to another host (c0dcaa73)
- fix(crewai): scope, category and source tags are percent-encoded, so veld's redactor leaves them intact (eb0aa42c)
- fix(query): a causal query asks for Triggers edges, the type causal cues are written as (accdede5)
- fix(graph): the ontological penalty spares AssociatedWith like the other generic relations (3f14aa5e)
- fix(upstream): a ledger commit exists only if origin can reach it (f26ae2b2)
- fix(replication): a follower journals a refitted snapshot as it stored it (6328e22f)
- fix(vector-index): saves of one index run one at a time, and shutdown waits on a slow flush instead of racing it (85ec4788)
- fix(auth): the startup banner names the key validation actually checks (c3cee501)
- fix(recall): sum an entity match's word weights in a fixed order (6dddd99f)
- fix(recall): drop Layer 5.7's score-gap pruning, which cut whatever sat below a boosted few (3993f24b)
- fix(recall): Layer 5 signals scale the fused score instead of adding to it (upstream #174) (a51a8e91)
- fix(recall): every Layer 4 boost is a bounded share of the candidate's own fused score (upstream #174) (3a5ab5d7)
- fix(recall): an entity word match is worth two top fused scores at full specificity (a0ac3963)
- fix(recall): an entity word match weighs by how few candidates carry the word (70206530)
- fix(recall): post-fusion boosts are shares of the fused score, and topic dedup waits for recency intent (3dc0b29e)
- fix(recall): the layer ladder measures what it names, and the tier scan stops re-adding index rejects (3e6aaf9d)
- fix(replication): a follower indexes snapshots whose vector its index cannot hold (b8da4358)
- fix(recall-harness): ingest stores each memory's temporal facts, as the remember handler does (0bb8cae2)
- fix(recall): a declared file path the query names outranks a shared word (65c93365)
- fix(recall): a recall that names no user runs as the tenant that owns the store (968f64ad)
- fix(embeddings): encode_batch gives each text the vector encode gives it alone (cb1c47ac)
- fix(upstream): point veld commit ids at the history rewritten on 2026-10-28 (3d6c896f)
- fix(memory): text is cut on char boundaries, so multi-byte content stops panicking workers (0b0f0eaa)
- fix(recall): rank on the unified score before the reranker, and expose that score (670a0667)
- fix(memory): the write gate absorbs only exact copies; near-exact ones are kept, tagged and de-ranked (9c0a1fec)
- fix(recall): tag recall leaves out NER annotations and context unless asked (66966ad5)
- fix(recall): a declared path keeps its fused rank when no cross-encoder can score it (23faa91f)
- fix(spaces): chunks are embedded one at a time again, as on main (a0b58468)
- fix(intent-log): a cut below an erased LSN keeps the numbering, and followers and restores are tested over erased frames (7b3798d2)
- fix(mcp): a recall question is no longer stored as a memory of its own (bd27bc33)
- fix(replication): a boot-time follow whose leader reports this instance id stops following (80b24a1f)
- fix(tls): a certificate change is seen by device and inode on Unix, not only mtime and size (07a2d550)
- fix(tls): cap one peer address at 16 handshakes and 64 connections (7019cb0d)
- fix(cli): a blank or padded VELD_TLS_* value no longer stops veld server (21b8a29e)
- fix(server): decide the local-dev rate-limit default after .env loads (8c420fe3)
- fix(tls): offer only http/1.1 by ALPN (4d105ff7)
- fix(replication): the follow route knows this process by its instance id, so its TLS URL cannot follow itself (82928caa)
- fix(docs): escape the requested path in the embedded docs 404 page (d1be61cb)
- fix: one slow-store open per tenant; sleep-time evidence and apply phases off the async workers (f608c898)
- fix(loops): the lane plans the queue creates say what they hold (bb627f09)
- fix(prefetch): the temporal query searches the day its context was taken at (b0ca1a88)
- fix(spaces): a backlog drain counts the rows its batched embed clears (80da49e3)
- fix(memory): a bulk forget counts each memory once, not each tier copy (d36e1695)
- fix(loops): reach LM Studio hosts through spine, holding one of the host's slots (b2b25385)
- fix(intent-log): a failed rotation backs off, and a segment past the head is set aside (16c7c555)
- fix(shutdown): log when the shutdown sequence returns and when the runtime stops (0b810e6b)
- fix(memory): maintenance decay, fact keywords and the prefetch tick stop depending on run order (1a0df4b9)
- fix: open tenants, writers and slow stores off the async workers (bd22ae53)
- fix(audit,bm25): unique audit row keys under a coarse clock; one BM25 indexing thread (973ae001)
- fix(spaces): a re-embed keeps the vector it replaces; CI's layout, format and test failures (22c48c1f)
- fix(handlers): open graphs off the async workers (57526bdb)
- fix(zenoh): remember keeps its external id, and a failed recall is an error (70dbc491)
- fix(journal): the embedding-backlog drain journals what it rewrites (951cb8ad)
- fix(http): a timed-out request's 408 says the work may still complete (26c8f9ce)
- fix(journal): every write path journals, embeddings included, and only text projections open (6f75ea2a)
- fix(compression): definition extraction slices on char boundaries (8f34157b)
- fix(handlers): the last six tenant opens in ingest, MIF and seed leave the async workers (b7dac921)
- fix(recall): question-frame words leave query-term coverage, and short acronyms keep the reranker (aa2f4a25)
- fix(hooks): the Claude Code memory hook works as designed, stores less noise, and fails quietly (0c0bcfb9)
- fix(shutdown): close tenant writes before the flush, so exit follows it (3f4d1955)
- fix(zenoh): compile the transport against today's handlers, and clippy it in CI (efcb3501)
- fix(memory): a background embedding no longer writes back a stale copy of its memory (9176be36)
- fix(recall): tag recall answers with the newest memories the writer tagged, without vectors (8af55e89)
- fix(replication): the leader lists only tenants it can feed (1a516ba8)
- fix(storage): a full disk under a redb store latches it hard, and later writes answer 503 (bbc123b1)
- fix(handlers): open tenants off the async workers in the remaining handlers (3555e511)
- fix(vectors): a Vamana save is atomic and runs outside the index locks; unchanged content is not reindexed (7f9133a0)
- fix(security): bind query and path user ids to the key's tenant; context status behind auth (dfe58c15)
- fix(memory): the write gate stops deleting acknowledged memories on similarity alone (4970c1a1)
- fix(dials): a global cross-encoder share keeps decomposed queries at their own floor (7f66e9df)
- fix(recall): tier hits on the decomposed scale, ordinal pins that hold, filters before the cut (80a5c81c)
- fix(checkpoint store): cut a torn tail on open, roll a failed write back to the last sync (6994d3af)
- fix(monitor): send an admin key for replication status to loopback servers only (d446d116)
- fix(memory): anchored memories keep the anchor importance floor (fb6bf926)
- fix(state): the startup reconcile restores a memory the store lost, not forgets it (49b52daf)
- fix(graph): a repeat from one source moves no directional strength either (7b30b94f)
- fix(replication): the feed ends at the head it read under the writer lock (b89c661b)
- fix(storage): every runtime RocksDB write goes through the write latch, keyed by its database (56a1773d)
- fix(health): a tenant still opening does not fail process readiness (acd61d91)
- fix(server): open tenants off the async workers so probes answer during init (96329956)
- fix(intent-log): cut a failed write back to a frame boundary and torn tails on open (16128205)
- fix(state): the instance registry and the follower's cursors are not tenants (6c4341c3)
- fix(replication): stop the follower at shutdown; report pass progress and lag (ea372fe7)
- fix(intent-log,replication): journal the anchored flag as a new SetAnchored frame (490d1ef9)
- fix(restart): start a launchd job again with kickstart instead of relaunching it (6ed3d15e)
- fix(replication): the follower leaves BM25 to the journal's projection (b608e245)
- fix(replication,shutdown): per-tenant isolation, rollback detection, safe follow; bounded exit (4497c7d0)
- fix(replication): decode legacy memory snapshots with the store's decoder (c92cd53c)
- fix(demo): delete the tracked TUI debug log (1d0cf8c8)
- fix(demo): untrack the TUI debug log (4ad79249)
- fix(embeddings): pin the default reranker and verify downloads (4ea9ca36)
- fix(tests): CI's two red tests on main, left by the key-management merge (de4c3aa0)
- fix(tests): the RocksDB-backed tests build only with the rocksdb feature (27238134)
- fix(memory): record the embedding backlog on the upsert path too (ee2eb593)
- fix(slow-store): seal payloads under the store keystore when it is the only one (33d13037)
- fix(ui): count recalls from the recall log, not from access_count (f933eac9)
- fix(bm25): keep an encrypted tenant's lexical index in memory (d5019cef)
- fix(storage): migrate v0.7 field-encrypted content into the record envelope (d44e6729)
- fix(recall-harness): leave the cross-encoder pre-warm pair out of a recording's table (34c44668)
- fix(recall-harness): time the gate's recalls with no diagnostic recall between them (5aae873e)
- fix(benchmarks): count an unanswered judge call as an abstention, not option 0 (a4a2917c)
- fix(user-auth): refuse to start when the configured keystore cannot seal TOTP secrets (60f864cb)
- fix(keystore): seal the journal under the store keystore when it is the only one (880bfafb)
- fix(migrate): a plaintext store migrated into rocksdb stays plaintext with a passphrase set (1f62f819)
- fix(todos): the flat todo aliases work, todo responses drop the embedding, and memories link back to the todos citing them (11fb9b32)
- fix(todos): todo search reads the embeddings stored on the todos, and todos link the memories and dependencies they cite (3ce0c919)
- fix(facts): extraction re-reads the whole eligible corpus, so an age-gated memory is no longer lost to the watermark (acdbe151)
- fix(memory): a content duplicate merges its enrichment, and every fact ingest goes through one arbitration policy (b5c9f0cc)
- fix(facts): a plain declarative sentence becomes a fact candidate on every non-operational type (ef1c048f)
- fix(facts): a contradicted fact is superseded instead of living beside its correction, and a session promotion reaches storage (7989b06a)
- fix(facts): extraction is gated by experience type and sentence shape, and support counts distinct evidence (90ab56a1)
- fix(facts): protocol output that reaches a memory verbatim never becomes a fact (657ebb88)
- fix(api): an unknown memory type, todo field or filter value is a 400, not a silent default (b2fcb75a)
- fix(facts): the extractor no longer mints "X relates to Y" facts, and drops operational noise (17fe6771)
- fix(query-parsing): a query cannot close its line in the LLM parser's prompt (d40c6096)
- fix(files): a codebase scan or index reads only a project directory (8352c1f1)
- fix(storage): a read error in a legacy-DB migration stops it before the old DB is renamed aside (f08d1865)
- fix: graph handlers keep the graph lock off the runtime, and fact updates reindex (f2c4653c)
- fix(mcp): the recall_by_tags tool the server's help lists is registered (bc43958c)
- fix(integrations): an insecure remote integration URL is refused by default, and the guard now covers Linear and GitHub (8b81a32a)
- fix(audit): audit writes run on the task tracker, so a drain waits for them (9c90e033)
- fix(storage): say which engine is the default and what redb backups capture (34780786)
- fix(recall): switch off the Layer-5 hebbian term by default (cddb7b7c)
- fix(backup): close the tenant's stores before a restore replaces their files (4afe07bc)
- fix(vector-index): small indexes are searched exactly, and Vamana is built with its pruning rule (ce32a46d)
- fix(rerank): the cross-encoder honours VELD_ONNX_THREADS like the embedders (0681709e)
- fix(search): BM25 cutoff ties are ordered by content and RRF fusion keeps first-appearance order (d920211c)
- fix(recall): graph-leg candidates and fused-score ties are ordered by content, not hash order (81026bf2)
- fix(graph): equal-strength edges and fuzzy name matches resolve by name, not by uuid or hash order (a69ace81)
- fix(backup): a RocksDB restore no longer deletes the tenant's data, and waits for its store to close (17f5a15a)
- fix(boot-guard): a build without RocksDB reads an empty RocksDB store as empty (309eb346)
- fix(graph): deleting an entity deletes the relationships it is an endpoint of (4a75d6e5)
- fix(gap): the slow-store mirror syncs only edges between entities (77185327)
- fix(relevance): bound the graph walk /api/relevant runs per named entity (b1d9a948)
- fix(chunking): chunk_text always advances; startup orphan repair persists progress every 500 (3a31f4f8)
- fix(webhooks): an empty secret no longer verifies forged signatures, and malformed headers are errors (6becd818)
- fix(mif): a Custom relation was renamed by its own backup, silently (50f7ac2f)
- fix(decay): a rate learned per edge, and an L1 window that pruned a third of returns (fec94be7)
- fix: four defects shodh-memory had already fixed, and one that helped hide them (ead4a2d9)
- fix(durability): LSN-consistent backup cut, both-direction reconcile, honest init errors (649b2163)
- fix(alignment): live-secondary autofit + canonical pair_id + wire aligned secondary into recall (#38) (2c29bb3c)
- fix(memory): compute secondary embedding on live ingest so dual-embedder index populates (#33) (39b1c1e0)
- fix(benchmarks): make locomo_mc10_eval OOTB-correct on mac/windows (438a9869)
- fix(cross-encoder): fail soft when ONNX Runtime is unavailable (4a4a4a7b)
- fix(temporal): resolve N-months/years-ago to MonthYear/Year; drop entity leak from event_stems (#13) (4b801b58)
- fix(monitor): authenticate graph preview (14542324)
- fix(monitor): authenticate graph preview (bd1f5bf0)
- fix(storage): surface decrypt failures in forget/decay passes (veld) (6bbfdba4)
- fix(2fa): fail closed on TOTP encryption — never store a 2FA seed in plaintext (6d82f984)
- fix(storage): blind episode_id in the episode_seq index (d29a331d)
- fix(keystore): reject KDF-downgraded Argon2 params on unseal (9a949b0e)
- fix(keyctl): disarm rotate-dek data-loss footgun (4942217e)
- fix(crypto): wire operator signals to the keystore + fail closed on partial config (965c1175)
- fix(keystore): bound Argon2 params from untrusted keystore (audit/breakers) (7c619181)
- fix(keystore): atomic keystore persistence (H1) — temp+fsync+rename+backup (1eac7bb4)
- fix(keystore): zeroize intermediate key-material buffers (breakers finding) (1e36f818)
- fix(security): port shodh-memory PR #285 hardening to veld siblings (3d64fc72)
- fix(monitor): minimize to tray on close; start visible; drop strict CSP (d577ac17)
- fix(slow-store): log unreadable rows instead of silently dropping them (f89c3fd8)
- fix(slow-store): run retention cleanup deletes in one transaction (d8edfde9)
- fix(slow-store): purge entities/edges dropped from the graph on sync (8f124d1b)
- fix(monitor): self-heal dials + add graph reload (resilient to server flaps) (c6aa515d)
- fix(monitor): open graph viewer in-window (iframe), not a crashing OS window (76474351)
- fix(relational-bridge): bound the synchronous recv so a hung backend can't wedge a worker (c5730fff)
- fix(slow-store): edge sync aborted on multi-relation entity pairs (558f3b3f)
- fix(mcp-server): make the server typecheckable and tsc-clean (54cf76eb)
- fix(tests): initialize SemanticFact purge fields in adaptive_memory_tests (840b3dd4)
- fix(relational): bridge_block_on returns Result instead of panicking (95c7554c)
- fix(datasets): dialect-correct relational dataset layer for Postgres and SQL Server (1e612e67)
- fix(m5): clippy explicit_auto_deref at gap-persist call sites (45542ee4)
- fix(w4): dialect-correct memories DDL/upsert for Postgres/Supabase/MSSQL (d967737b)
- fix(cross-encoder): track HF main + pre-warm on startup (b51cb7af)
- fix(tests): complete BoxError migration + facts/visibility in test harness (046b892c)
- fix(build): unblock lib + bin compile (40 errors) (2e181e3f)
- fix(retrieval): restore rlm_refiner.rs accidentally dropped in HTML commit (2c1efc80)
- fix(relational): switch type-erased Error from anyhow::Error to BoxError (StdError-compatible) (3978e48f)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (61f252ae)
- fix(facts): add purged_at/purge_reason to inline-test SemanticFact constructions (3621e71a)
- fix(w4): close pluggability gaps — Row::from_owned, RelationalBackend::Custom, docstring example, Cargo dupe (725f71a4)
- fix(memory): finish AgentSession writer-read — marker file + writer paths + e2e test (2c39fa8e)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (296bf083)
- fix: brand whitelist + argon2 semaphore + 503-on-disabled-auth + recovery-code charset audit (31f45a1d)
- fix(alignment): paths-relative-to-CWD in TOML specs; add pairs.vec cache (623a4863)
- fix(intent-log): repair conflict-resolution build breaks (a1b67b80)

### graph

- graph: clock-injected decay and the causal walk the recall harness needs (3c93ffb3)
- graph: a read-only recall walks edges without strengthening them, and the hub cap prunes the same edges every ingest (67c54972)
- graph: the last store off Arc<DB> — nine families onto ColumnStore (6d1ee23d)
- graph: delete_entity left its entity_episodes index rows behind (02f19f46)
- graph: an entity's merge scope is a property of its type, not of its name (3e4e98ce)
- graph: the entity namespace had five ways in and none out, and UUID entities are ruled to stay (5a649325)

### handlers

- handlers: typed replies for A/B tests, health probes, context status, webhooks, backlog drain, reembed, file_answer, dials schema (cb9735b4)
- handlers: typed replies for the graph, entity and MIF routes (4bc2caa8)
- handlers: typed replies for forget, bulk, clear, compress, decompress, merge, split, lineage confirm and reject, wiki render (dc546dc3)

### handoff

- handoff: the remaining storage work, dependency-ordered (b013b4aa)
- handoff: the suite segfaults on exit intermittently, and it predates this work (afcf5e4d)
- handoff: the traits were never the bottleneck — the stores outside them are (7d1fdcf2)
- handoff: replace the estimates with the executable measurement (4687cb91)
- handoff: R-6a/b is a prerequisite for R-3, not a follow-on (8f1ec923)

### health

- health: report tenants with data on each backend, measured at boot (a8cc5b41)

### ingest

- ingest: type causal lineage verbs as Triggers so the causal walk has edges to follow (fc7a5389)

### lib

- lib: declare recall_harness, which acb4ccd committed but never referenced (2fc7c76c)

### licence

- licence: veld is published under an evaluation licence (700b6a2d)
- licence: BUSL-1.1 → AGPL-3.0-or-later, and the Apache-2.0 conditions the fork never met (dfc37c61)

### list

- list: offset paging, a lifted cap, and a preview that says it is one (482c0c8b)

### mcp

- mcp: remember reads the handler's success field, not a stored field that never arrives (6224a50a)
- mcp: the original tools decode what the server sends (088eebff)
- mcp: expand VELD_* from the environment, dev key only as fallback (0ae4dbc4)

### metadata

- metadata: discovery terms chosen from measured adoption, not from what the code calls itself (6c09205b)

### minilm

- minilm: the runtime extractor could only ever match a symlink, so nothing had ever been installed (0226ad24)

### misc

- Merge perf/idle-onnx-20261030 (6e9fb999) into train F (f3ffde59)
- Merge perf/tantivy-unstored-20261030 (84577e07) into train F (552c6b2d)
- Merge perf/tantivy-commit-20261030 (1c017617) into train F (7e1d250a)
- Merge feat/value-envelope-read-20261030 (13f074ea) into train F (aecc4344)
- Merge fix/journal-rewrites-20261030 (eee7e295) into train F (b4f0a375)
- Merge fix/journal-gaps-20261030 (937fbf90) into train F (a3fdb549)
- Merge schema/p1-typed-replies (cb9735b4) into train C2 (cf321d21)
- Merge land/failover-spaces-20261029 (4b124ee3) into train C2 (59f35f9f)
- Merge land/integrations-crewai-vercel-20261029 (6cbdf5ee) into train C2 (6bed4729)
- Merge perf/shutdown-tenant-close (cad080c9) into train C2 (df963a63)
- Merge feat/graph-leg-20261028 (51d68d07) onto main 2300c637 (63807ee7)
- Merge feat/fusion-flat-20261028 (72f8f303) into the train C spaces head (2300c637)
- Merge origin/main bb35da55 (train B) into feat/embedding-spaces (1fc9bf31)
- Merge pull request #104 from Portll/docs/te9-operating-rules-20261029 (bb35da55)
- Merge feat/embedding-spaces (364a99ba) into the replica-failover work (4b124ee3)
- Merge exp/scoresort-after-ce-20261029 (#98, 6ec06c9e) (50910ff3)
- Merge feat/ci-rebase-check (rebase credit, busy exit 75) (42c2c1b9)
- Merge local main a08afcd8 (ADR 0007 soak days 2-6) (edf0fe7d)
- Merge feat/locomo-qa-mem0-20261028 (348e786a)
- Merge chore/cargo-update-20261029 (70cb5339)
- Merge perf/read-scans-20261024 (6cc2d68e)
- Merge fix/auth-banner-precedence (c614847f)
- Merge fix/shutdown-index-save-20261029 (8b734071)
- Merge fix/lean-tag-recall-20261024 (#100) (816381b2)
- Merge fix/derived-tag-budget-20261024 (#88) (30cedafb)
- Merge fix/mcp-query-echo-20261024 (#85) (ea6447c0)
- Merge fix/write-gate-exact-only-20261024 (#96) (62de3e41)
- Merge fix/declared-path-rank-20261022 (#79) (a7f38092)
- Merge fix/replica-vector-parity (3814c930)
- Merge schema/p0-foundations (#94) (fcaf1f9f)
- Merge remote-tracking branch 'origin/main' into feat/embedding-spaces (364a99ba)
- Merge remote-tracking branch 'origin/main' (a08afcd8)
- Merge remote-tracking branch 'origin/main' into land/hotfix-20261029 (2c436b4f)
- Merge origin/main 351f02a1 into feat/embedding-spaces (c6ae0874)
- Merge branch 'fix/ledger-remap-20261028' into land/hotfix-20261029 (7c189e78)
- Merge remote-tracking branch 'origin/main' into feat/native-tls (351f02a1)
- Merge pull request #89 from Portll/docs/schema-reference (cb582e5f)
- Merge remote-tracking branch 'origin/main' into feat/native-tls (c6f52b7e)
- Merge pull request #87 from Portll/hotfix/mcp-remember-success (6398bbdc)
- Merge remote-tracking branch 'origin/main' into feat/native-tls (b48af168)
- Merge remote-tracking branch 'origin/main' into land/deps-sleeptime-20261024 (d7eb836a)
- Merge pull request #86 from Portll/hotfix/mcp-response-shapes (1728f765)
- Merge remote-tracking branch 'origin/main' into feat/native-tls (d1669d63)
- Merge remote-tracking branch 'origin/main' into land/deps-sleeptime-20261024 (21606791)
- Merge branch 'feat/native-tls-clients' into feat/native-tls (eb79e6fb)
- Merge branch 'feat/ci-local-gate' (b1f901aa)
- Merge remote-tracking branch 'origin/main' into land/deps-sleeptime-20261024 (e0e7f58d)
- Merge remote-tracking branch 'origin/main' into feat/embedding-spaces (9e544991)
- Merge remote-tracking branch 'origin/main' (915ce728)
- Merge branch 'chore/dependency-migrations' (a7312d21)
- Merge branch 'fix/slow-store-lock-and-sleep-time' (4cc8a513)
- Merge branch 'chore/ratatui-0.30' (43880ea0)
- Merge pull request #55 from Portll/fix/upstream-ports-20261020 (7e297771)
- Merge pull request #68 from Portll/dependabot/cargo/tui/git2-0.20.4 (2fc19faf)
- Merge pull request #72 from Portll/dependabot/cargo/criterion-0.8.2 (2a42e82e)
- Merge pull request #70 from Portll/dependabot/cargo/tower-http-0.7.1 (6f33ad2c)
- Merge pull request #64 from Portll/dependabot/npm_and_yarn/veld-telegram/open-11.0.4 (a1be708f)
- Merge pull request #61 from Portll/dependabot/npm_and_yarn/mcp-server/vitest-5.0.4 (8904bf3b)
- Merge pull request #58 from Portll/dependabot/npm_and_yarn/mcp-server/types/node-26.6.5 (8d56ad08)
- Merge pull request #57 from Portll/dependabot/npm_and_yarn/mcp-server/vitest/coverage-v8-5.0.4 (305eb3ef)
- Merge pull request #66 from Portll/dependabot/pip/backoff-gte-2.2.1 (de39c36d)
- Merge pull request #63 from Portll/dependabot/pip/huggingface-hub-gte-2.2.0 (80fea34b)
- Merge pull request #67 from Portll/dependabot/pip/maturin-gte-1.15.0-and-lt-2.0 (909ac4ad)
- Merge pull request #62 from Portll/dependabot/pip/openai-agents-gte-0.23.1 (a23b300f)
- Merge pull request #59 from Portll/dependabot/pip/pydantic-gte-2.14.1 (a847c431)
- Merge pull request #60 from Portll/dependabot/github_actions/actions-3c4779d133 (e7164d56)
- Merge local main 9087b20c (bench qa readers, soak log day 1) into land/parity-20261022 (b1893897)
- Merge remote-tracking branch 'origin/main' into land/parity-20261022 (4a9199a2)
- merge veld main into feat/gap-analysis-scale (aecac4a3)
- Merge remote-tracking branch 'origin/main' (403da9b8)
- Merge remote-tracking branch 'origin/main' into land/parity-20261022 (825ceb42)
- Merge origin/main into feat/embedding-spaces (3ab3877d)
- Merge origin/main into feat/embedding-spaces (f5b4724b)
- Merge remote-tracking branch 'origin/main' into land/parity-20261022 (742c40b0)
- Every similarity bar counts its comparisons and its fires (42c3f20a)
- A derived keyword between two event names takes no causal edge (c751f8a8)
- Merge origin/main into feat/embedding-spaces (2fd32f19)
- Merge branch 'main' into feat/intent-log-segments (e2fdfa72)
- Merge branch 'fix/parity-determinism-20261022' (bf785144)
- Merge remote-tracking branch 'origin/main' (6f6c4c72)
- Synchronous writes read interference cosines on MiniLM's scale too (7167c09e)
- Merge remote-tracking branch 'origin/main' (368cd7c5)
- A keyword overlapping a co-mentioned name takes no causal edge (90572ea9)
- Merge branch 'main' into fix/zenoh-contract (62e46635)
- Merge branch 'main' into fix/journal-and-projections (045b9eb7)
- Merge main into feat/embedding-spaces (a17c684f)
- Memory competition compares raw cosines with its pool-relative floor (2e91d580)
- The write gate's absorption bar is read on MiniLM's scale, once (64e274f2)
- Entity concept merge reads name cosines on MiniLM's name scale (3497b2d6)
- Similarity bars read each embedder's cosine on MiniLM's scale (e8e71f01)
- Merge main into feat/embedding-spaces (b7737892)
- The recall harness ingests with the tags remember stores (f744e186)
- Merge branch 'main' into perf/follower-batch-store (0dd30ef5)
- Merge branch 'main' into perf/follower-batch-store (4b7d075f)
- Merge branch 'main' into perf/follower-batch-store (8a9ce618)
- Merge branch 'main' into perf/follower-batch-store (a7a43032)
- Merge branch 'main' into land/recall-20261021 (023e1f67)
- Merge branch 'main' into land/recall-20261021 (764f49c7)
- Merge branch 'fix/shutdown-drain' (36a4d5d3)
- Merge branch 'fix/zenoh-build' (a2f392ff)
- Merge branch 'chore/security-lockfiles' (92731ca6)
- Merge branch 'main' into perf/write-path-20261018 (e2efb3ca)
- Merge remote-tracking branch 'origin/main' into release/push-20261021 (8e70a4cd)
- Merge branch 'main' into perf/write-path-20261018 (778e6458)
- Merge branch 'feat/embedder-arctic-xs' into land/recall-20261021 (af980380)
- Merge branch 'perf/reranker-speed' into land/recall-20261021 (256b7231)
- An explicit rerank_count reaches the cross-encoder at any max_results (67ace586)
- Merge branch 'feat/rerank-depth' into land/recall-20261021 (1daad6dc)
- Merge branch 'feat/reranker-parity-dial' into land/recall-20261021 (50a4d372)
- Merge branch 'feat/date-window-leg' into land/recall-20261021 (fae35cff)
- Merge branch 'feat/recall-query-side' into land/recall-20261021 (5319d8fb)
- Merge branch 'main' into perf/write-path-20261018 (00d0875c)
- Merge branch 'release/push-20261020' (08d14045)
- Merge remote-tracking branch 'origin/feat/reranker-parity-dial' into perf/reranker-speed (115e8038)
- Merge remote-tracking branch 'origin/feat/reranker-parity-dial' into feat/rerank-depth (d71a874e)
- Selecting a reranker that is not loaded yet no longer deadlocks (f246cf85)
- Merge remote-tracking branch 'origin/main' into release/push-20261020 (9b1a30b4)
- Merge pull request #53 from Portll/chore/public-repo-hygiene (2ba8ff10)
- Merge remote-tracking branch 'origin/main' (f65b8d1a)
- Merge branch 'main' into perf/write-path-20261018 (2c04f8d7)
- Merge remote-tracking branch 'origin/feat/reranker-parity-dial' into perf/reranker-speed (2be62762)
- Merge remote-tracking branch 'origin/feat/reranker-parity-dial' into feat/rerank-depth (c552cb39)
- Cross-encoder downloads have a deadline, so a stalled one cannot hold every scorer (f281d300)
- Memory competition measures a pair against its own pool, not against MiniLM's scale (22a74b23)
- Merge remote-tracking branch 'origin/main' into perf/write-path-20261018 (5a15649e)
- Merge branch 'main' into perf/write-path-20261018 (3a1d47a5)
- Recall gate: nomic-embed-text-v1.5 as an embedder arm (f4b8f7f2)
- The cross-encoder uses the cores it has, batches pairs of similar length, opens a second session under load and remembers pair scores (14450b8f)
- recall gate: an embedder dispatch input that selects, fetches and caches the arm's model (0c5d3cd7)
- Merge main into perf/write-path-20261018 (37fe881c)
- The cross-encoder can be given a deeper pool, a larger budget and a heavier blend (63d4ac8b)
- recall harness: record retrieval dial overrides and refuse a mismatched diff (b6fbc2ae)
- recall gate: rerank_model and rerank_count dispatch inputs (5a159540)
- Merge pull request #51 from Portll/feat/measure-diag-k (59ac2f84)
- Merge remote-tracking branch 'origin/main' into feat/measure-diag-k (93ac2c9f)
- slow store: rustfmt the begin_sync test (634bb332)
- Merge remote-tracking branch 'origin/main' into feat/measure-diag-k (6cbf2a3d)
- slow store: one mirror rebuild at a time, and the TTL counts from its start (d5f56581)
- Merge pull request #50 from Portll/fix/pin-reranker (53c4c515)
- slow store: one mirror rebuild at a time, and the TTL counts from its start (22891f79)
- Merge pull request #49 from Portll/fix/ci-tests (4c79f666)
- Merge pull request #48 from Portll/licence/evaluation (bc1544d6)
- Merge pull request #47 from Portll/wave/w0-land (67f7b343)
- Merge wave/w0-measurement: the recall gate, per-case baseline, model tapes and the W0 ledger outcomes (29ad6498)
- Merge pull request #46 from Portll/fix/hub-cap-test-race (d9cf8450)
- Merge pull request #45 from Portll/feat/key-management (02610059)
- Merge origin/main into feat/key-management (1a6848d6)
- Merge pull request #44 from Portll/feat/typed-entities (5cb1dbd1)
- recall harness: the ablation arms measure the thing they name, or say they cannot (c9daf38d)
- cargo fmt over the files the ablation port touches (40fd188a)
- session detect: resolve the branch of a repository with no commits yet (fdc43a1e)
- recall harness: seven tests that measure a veld difference are ignored, with the finding as the reason (72b7d5e9)
- recall harness: declare the eight remaining modules (77af9838)
- shared stores: boot the rocksdb arm through with_store and delete the legacy constructors (7e82fbdd)
- shared stores: prove the rocksdb with_store constructors read what the legacy ones wrote (a505d5c8)
- LoCoMo 10x runners and the visual model page (4f615403)
- Stop tracking .claude/settings.json; add a generic settings.json.example (33f57e9a)
- Benches compile again; a 14-day point on the L2 decay curve; CLAUDE.md architecture notes (4baeeb86)
- Ignore the cargo fmt commit in blame (8b1aec83)
- cargo fmt: format the tree with CI's rustfmt (1.9.0, Rust 1.98.1) (8413f048)
- sleep_time: the four stores and the orchestrator build from a ColumnStore (12630ad9)
- storage tests: the decoder-counter tests take one lock, because the metric they assert on is process-wide (b479464c)
- mif export: a store read that fails fails the export (85385557)
- expose index coverage, and mount the probe routes the harness never had (8bf00105)
- reconcile stored against indexed, and delete a test that proved nothing (2b094af6)
- recall harness: a report that cannot say what embedded it says unknown, and unknown is refused (744794c5)
- recall harness: a report names the embedder that ran it, and a diff across embedders is refused (3eefd144)
- get memory: the response says whether its content is the document (e032961a)
- path_guard: one user-id rule, checked at the request and at the filesystem (3152d8cd)
- index-insert loss is counted too, and the counter is proven to move (6e99d975)
- recall harness: 45 errors to 5, and a silent-loss guard veld did not have (7ea996e1)
- harness APIs: three of five ported, and two that are not 15-line ports (29f0849b)
- recall capture: the two modules, four wired stages, and a corrected gap (2ba48db5)
- sleep_time: the last four stores off Arc<DB> (cda736db)
- recall harness: record the measured gap for the eight gated modules (5c189815)
- ablation switch: the retrieval pipeline can be run one rung at a time (66c49e48)
- redb boots: the last construction refusal is wired, and the request is honoured (3e65e4a9)
- recall harness: import upstream's measurement layer, with its corpora (acb4ccda)
- auth seam + four fixes shodh-memory already merged (505b610a)
- R-5: MemorySystem constructs over the trait, and redb's arm is wired (94c2b328)
- facts, lineage, learning-history, temporal-facts: four default-keyspace stores off Arc<DB> (9125f8b2)
- R-5 (first half): the shared bundle and backup engine stop assuming their engine (59ea8ec2)
- Merge remote-tracking branch 'origin/main' into HEAD (dd1ab932)
- secondary stores: with_store constructors, and ensure_cf closes a masked divergence (820a7c0d)
- context_blocks: the first secondary store off Arc<DB> — and onto the LIGHTER trait (cd20c1d3)
- R-6c: verify an imported target against its dump, both directions (1dcd605e)
- R-6b: portable import for redb, and cf:None meant two keyspaces (bf41c1a9)
- R-6a: portable export for rocksdb, with a format that refuses a partial read (b83c0120)
- forget_all: erase facts through the store, not through db() (b1d54b63)
- storage handoff: A's batch API must be DB-scoped, because 4 call sites already batch across column families (0b3726e8)
- storage handoff: the order is A then B, and my own recommendation was the wrong way round (f9bd416e)
- R-1 storage: the TODO under-scoped the work fourfold, and the count was never the hard part (567717d8)
- a backup that omitted a store said success, and a spread that went nowhere (e8786575)
- entity cap + NER: the cap kept the wrong fifty, and NER never saw past token 128 (4c2d630f)
- llm hosts: carry the exposure facts — which hosts bind every interface, which fetch models on demand, and that runners are off by default (53f06d5a)
- llm hosts: carry the declaration's contextNote — the KV-cache guardrail ollama's removal cost, recorded where both repos can see it (c167344d)
- llm hosts: ollama out, and the two defaults that disagreed with each other are now one declaration (540602cf)
- onnx extractor: checking shodh-memory found the bug fixed there first, and my fix wrong in the way theirs documents (34fdf3a8)
- entity tail: the tail is nineteen times more selective than the head, so cut-the-tail was backwards (7abbed2d)
- graph sweep: the population says zero live secrets, and the tenant binding I claimed yesterday is not compiled (ee76ecd6)
- graph-entry doc: the design prescribed a rule the implementation refuses, and said so nowhere (68df1bd5)
- graph entry: an entity is a one-way door, so redaction has to happen on the way in (03180be6)
- Merge remote-tracking branch 'origin/main' (70e424e3)
- Merge branch 'feat/encryption-v2' (9695c944)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (e4c28c14)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (54d33a7b)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (154bdf15)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (bab8d339)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (90ca735c)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (f3f9445e)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (885f3a9c)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (cc32b335)
- Merge branch 'main' into reconcile/unify-mainline (ce5f89df)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (f603d163)
- Merge origin/main into local main: facts + sleep-time + relational + intent-log (d657e95e)

### monitor

- monitor: working Tauri GUI — IPC, tray, X-quit, diagnostics (#4) (ce3d98e2)

### nomic

- nomic: unblock concurrent inference + dynamic seq-len + split metrics + hook debounce (#6) (0366bf4f)

### obs

- obs: write-stage latency histograms and per-tenant storage checks in doctor (48e3eadd)

### observability

- observability: a wrapper that renamed its model, and a warning that was 26% of the log (05874cc4)

### packaging

- packaging: add cross-platform service installation scripts (89a5f01f)
- packaging: add cross-platform service installation scripts (514d7edf)

### perf

- perf(embeddings): unload idle ONNX models behind VELD_MODEL_IDLE_UNLOAD_SECS (6e9fb999)
- perf(bm25): stop storing memory content in the Tantivy index (84577e07)
- perf(bm25): commit the Tantivy index off the write path (231e6c46)
- perf(graph): bound the PPR walk by edges decoded, and name each tie peer once per walk (a8e81bb8)
- perf(shutdown): tenants close in parallel inside the shutdown sequence (cad080c9)
- perf(graph): the graph leg decodes at most 5000 episodes per recall (9ba00b7e)
- perf(health, crud): per-tenant index health and the unfiltered count read maintained totals (e5820f80)
- perf(replication): a follower page's snapshots are one store write (8125b8d5)
- perf(graph): the full decay pass reads and decays the graph in chunks (3544bd5d)
- perf(replication): the follower journals a page in one append batch (414bb5a9)
- perf(write path): upsert and batch extraction off the async workers; strict, memoised entity lookups for edges (b1b046d2)
- perf(graph): an episode's edges and a typed pair are index reads, not family scans (32991d19)
- perf(write path): no full-store scans per request, one BM25 writer, every write journaled (e33a1f3a)
- perf(recall, graph): read each candidate once, bounded graph expansion, interval-based fact decay (28a20e70)
- perf(intent log): deferred slow-store apply, single-pass replay, id-map snapshots on a cadence (225475b0)
- perf(embeddings): one embedder, session pool and cross-encoder per model spec, process-wide (aa9078c5)
- perf(relevance, parsing): no full-store scan on /api/relevant, compiled regexes, bounded trace upkeep (eb12a15f)
- perf(slow store): rebuild the graph mirror off the projection's connection, and make it configurable (71eb3585)
- perf(streaming): extract off the runtime, on the deferred write path, per-session locks (6626dd61)
- perf(keystore): cache unsealed keystores and lock per path, not per process (6fc7cd43)
- perf(replication): coalesce pages, one BM25 commit per page; raise the file limit (e09e1423)
- perf: score each rerank pair at its own length, not padded together (1c249ec0)
- perf(gap): find open triads and diamonds in memory; count orbit links in one pass (f0d74adf)
- perf(recall): batch + de-duplicate cross-encoder reranking (14.5s -> ~1s target) (#28) (f0e9453e)
- perf(feedback): batch pending-bound enforcement off the per-insert recall hot path (#16) (e79c2345)
- perf(slow-store): amortize the live-arm checkpoint fsync over a batch (0dc27c4d)
- perf(slow-store): add a read-connection pool so reads don't serialize behind the writer (af9c2dc9)
- perf(slow-store): batch entity-name and embedding lookups into one query (fa9c7b6c)
- perf(slow-store): collapse find_star_gaps inter-spoke check to one query (793779f2)
- perf(retrieval): reuse stored embeddings in bi-encoder rerank (89f8ea89)
- perf(retrieval): reuse stored embeddings in bi-encoder rerank (5337c2e8)

### projections

- projections: decode snapshots through the store's legacy arms (0fa7829c)

### prospective

- prospective: fourth secondary store off Arc<DB>, both cross-family batches preserved (9e53f261)

### recall

- recall: the unified-score sort runs after the cross-encoder, not before it (6ec06c9e)
- recall: the gate baseline is re-recorded at 90572ea9 (31a7b654)
- recall: the other three LGBTQ questions behind the short-acronym change leave the hold-out (294ceada)
- recall: cases someone has examined leave the hold-out (7a9c9498)
- recall: a held-out LoCoMo suite without the gate's conversation, and intervals for one run (a2d6827a)
- recall: embed the recall query and its sub-queries on the query side (f9d3c2d9)
- recall: embed the recall query and its sub-queries on the query side (3922483b)
- recall: date window candidate leg for queries that name a day or month-year (339a292d)
- recall: baseline 0a841a9 from a 100-deep diagnostic run, and a miss taxonomy script (69852383)
- recall: a filter-only recall returns every match instead of letting competition drop them (3d977954)
- recall: a read-only recall writes nothing, and the harness's flag is finally read by something (bad93a6e)
- recall: the tags filter reaches the query (68f98a5a)
- recall: a failure is not an empty result set (3b764f96)

### redaction

- redaction: credential shapes get a class label before the graph, and the lexicon is measured not imagined (9b5c414f)

### redb

- redb: BackupStore and portable export — 57/69 to 66/69 (1d8fe03f)
- redb: 29/33 to 31/33, and the last two refuse instead of panicking (09bc095e)
- redb: seven more PrimaryMemoryStore methods — 22/33 to 29/33 (1fdd41b1)
- redb: six PrimaryMemoryStore methods, matching rocksdb's semantics not just its signatures (66c59222)

### refactor

- refactor(loops): code runs from the repo, data stays where it is (f1905965)
- refactor: the user auth store reads and writes through ColumnStore (0816f58d)
- refactor: de-metaphor earth/roots identifiers to literal names (#25) (947dbebf)
- refactor(recall): compute temporal boost via bound overlap (align with gate) (#18) (06446fe1)
- refactor(graph): single idempotent reinforcement coordinator for edge.strength (#17) (cb337f2a)
- refactor(temporal): resolve relative query dates to TemporalBound (F6) (#12) (3ee3e8ee)
- refactor(temporal): remove dead EventType::Recurring phantom variant (#10) (e2a51cf3)
- refactor(user-auth): rename field_encryptor -> totp_encryptor + drop TOTP hex detour (03960511)
- refactor(crypto): remove orphaned v1 FieldEncryptor (f38011b1)
- refactor(refiner)!: rename RLM->LLM refiner; consolidate duplicate module (e3a3de3e)

### release

- release: 0.8.0 — Data panel, restart with launch recipes, live replication (559e97af)

### remember

- remember: a caller may not assert its own trustworthiness (3662e656)

### restore

- restore: deleting the graph before knowing the replacement copied is not a restore (201efc73)

### retrieval

- retrieval: memory-tier trust scales the graph evidence, not the weight mix (38723b69)
- retrieval: every truncating sort has a total order, and a tag search has an order at all (585b11ef)
- retrieval: the engine reads through the trait, and the trait grew the two methods it proved missing (1e479dcb)

### security

- security: confine per-user paths to their base directory, and stop printing the full API key (#40) (636f7d72)

### seed

- seed: rustfmt the demo-profile hunk (8d5fe8c7)

### storage

- storage: the two parity tests c76548ec left failing pass (105ce190)
- storage: redb answers as RocksDB does, proven by test, and both engines are benchmarked alike (c76548ec)
- storage: the write latch's RocksDB recovery path compiles out of the redb-only build (1c7a093c)
- storage: import rocksdb::Cache once, and the memory ratchet falls to 67 (cff53892)
- storage: the write latch is per store, and the public body names a store by digest (a37431fb)
- storage: a NoSpace write latch is detected, bounded-retried, and reported instead of served as healthy (14c14048)
- storage: an undecodable value stays undecodable; lazy migration writes only under its own key (936d9ae6)
- storage: veld export / import / migrate between rocksdb and redb, verified (c053bbd5)
- storage: the probe opens a copied store through a read-only helper, not rocksdb:: directly (946bf259)
- storage: restore swaps by rename-aside, and SPANN keeps inserts after load (c9953577)
- storage: recover records persisted with a pre-40d581e AgentSession (5248a6ed)
- storage: an undecodable record is named, not counted (2c33f98f)
- storage: a ratchet so rocksdb:: outside src/storage/ can only fall (1f22c9da)
- storage: refuse to boot an empty backend while the other holds the data (8fce4557)
- storage: the full-consumption guard belongs to the two tiny candidates, not the full-memory ones (b9abc1d8)
- storage: a legacy decoder that read half a record was calling it a match (501656eb)
- storage: ColumnStore — the multi-family shape the five secondary stores need (35ef5e3a)
- storage: an executable coverage matrix — every trait method CALLED on both backends (1530877c)
- storage: PrimaryMemoryStore gains the 21 methods MemorySystem calls (3563fa9c)
- storage: a DB-scoped batch API, so supports_transactional_batch means something (42c4a1ce)

### style

- style(recall): elide admitted_graph_hit_terms' lifetime, as clippy::needless_lifetimes asks (b3bb7ae7)
- style: cargo fmt --all (d493d4c7)
- style: cargo fmt --all (e78ba075)
- style: cargo fmt over the files main left unformatted (f0c366d4)

### test

- test(bm25): read-your-writes under the background committer, recall after a write (1c017617)
- test(recall): a user-less recall on an owned system runs the owner's fact layers (216566b8)
- test(replication): the follower refusals and a read-route sweep (47ce9f4f)
- test(mcp): every tool's response type deserialises a sample of its handler's schema (f2d48324)
- test(recall): the declared path ranks first with the reranker and with reranking off (55dc1751)
- test(tls): name the plain-router test for what it checks (d82d1113)
- test(graph): the queued-prune test reads the clock the prune decides on (93ecbbad)
- test(gap): mirror parity tests and a writer-latency bench (d5b1f08b)
- test(ner): the many-entities budget times a warm extractor, not the first call in a process (1ac7abf2)
- test(http): a by-external-id miss answers 404 with the MEMORY_NOT_FOUND body (1c4fbe35)
- test(recall): the declared-path ranking waits for the target's embedding (d895c8af)
- test(ratchet): memory's rocksdb count is 52 under runtime-only counting (2ff41b62)
- test(storage-migration): the rocksdb-to-redb dry run needs a redb build (e2ed4a96)
- test(status-core,cli): name started_at in the DiscoveredInstance fixtures (5a5b2b2f)
- test(graph): reinforce the hebbian persistence tests from distinct sources (ce4e8129)
- test(graph): wait for remember's background work before reading the capped hub (e39d18f4)
- test(graph): name the frequency fixture tuple clippy flagged as type_complexity (b0585886)
- test(facts): measure what fact distillation yields on the recall corpora (faae24e4)
- test(stress): the two graph-integrity asserts in the brutal stress tests can fail (686a39a2)
- test(storage): lower the memory ratchet to the 68 rocksdb uses left (93dbf0ec)
- test(recall): record the LoCoMo gate baseline at ef3e2ec3 (cd56cfef)
- test(recall): an ignored test ingests the gate corpus twice and compares every rank list (8c88b17f)
- test(storage): snapshot the layout of every bincode-persisted root, not only Memory (a82c624a)
- test(backup): backup and point-in-time restore leave the same store on both engines (de6ff201)
- test(auth): every protected route refuses a request without an API key (e5772975)
- test(storage): pin the position of every variant a bincode record stores (d6f52df2)
- test(storage): the trait coverage matrix fails when redb lacks a method RocksDB has (1c19e915)
- test: four more memory stores run on both engines (1a6c323c)
- test: temporal facts run on both engines, not only rocksdb (a319fb8e)
- test(graph-memory): 34 integration tests that had never run, and 4 recovered without widening the API (ef608451)
- test(temporal): anchor-relative dual-path contract + q53/q63/duration regression guards (#30) (7cad3c78)
- test(temporal): integration test for event-time gate validation (#11) (a9199f95)
- test(wiki): end-to-end render / edit-back / forget demo (a0418a6f)
- test(reconsolidation): Layer-0 expiry gate + drain unit tests (passing) (0decc701)
- test(stress): gate brutal_stress_tests behind a stress feature (a2d78a43)

### tests

- tests: both ratchets count runtime code, and the encrypted dry run starts in plaintext (4656be7f)
- tests: every bincode-persisted root is walked for a type bincode cannot decode (88fed03a)
- tests: the benchmark's latency bound moves to its own ignored release-only test (10bf9b3a)
- tests: BUG-007 asserts scaling, and benchmark latency is the minimum of warm runs (88472509)
- tests(webhooks): the handler's empty-secret and malformed-signature paths are exercised end to end (eab2f846)
- tests: the persisted layout of Memory is pinned, and changing it fails (557aaa6f)
- tests: a source file no mod declaration reaches fails the suite (58af788c)
- tests: three of veld's four red tests were wrong about the code, not the code about them (8c2f4098)

### todos

- todos: fifth secondary store off Arc<DB> — the largest, and the last (6d388c75)

### types

- types: one name per wire shape (0d7b9438)

### vamana

- vamana: an id map is only valid for the index it was built against (55436f4a)
- vamana: the test that was wrong was hiding a defect that is not (bc6dfbf8)

### versions

- Versions: mcp-server/package.json and pyproject.toml to 0.7.39, matching Cargo.toml (2342c715)

## v0.8.2

### perf

- perf(replication): coalesce pages, one BM25 commit per page; raise the file limit (edd9b341)

## v0.8.1

### fix

- fix(replication): decode legacy memory snapshots with the store's decoder (b38d86b2)

## v0.8.0

### acl

- acl: the ACL store takes a key-value handle, not a database (068c6d07)

### b

- B: long_term_memory is a trait object, and the trait stopped lying about its width (8b365f1e)

### backup

- backup: the five backup operations work on redb, built on the migration dump (3441eab0)
- backup: the engine handle is captured at construction, and get_db() is gone (0b38e532)

### bench

- bench: durable pinned recall harness + gold labels (locomo-mc10) (#36) (d92017d3)

### build

- build: dev and test builds carry line tables for veld and no debug info for dependencies (13d84ce9)
- build: the release profile unwinds, so cargo test --release can link and a panic stays on its task (ad49321a)
- build: RocksDB is the default engine; both engines build by default and CI tests a redb-only build (9b667cfe)
- build: RocksDB is optional behind legacy-rocksdb; the default build runs on redb only (b6d3b0b3)
- build(windows-service): harden -Configure race + add shawl/WinSW backends (9d2afcaf)

### chore

- chore(loops): bring the loop producers in from ~/.claude/scratch as found (84fd0b37)
- chore: gitignore shodh runtime-data directories (#21) (dd722d67)
- chore(bench): exclude bench output from git + funnel it to a temp/output dir (481efe1a)
- chore: bump version to 0.7.28 (f77a2c87)
- chore(memory): delete orphan src/memory/maintenance.rs (undeclared dead code; FIX-R1 expiry now ported into live run_maintenance) (ae785704)
- chore: bump version 0.7.20 -> 0.7.21 (e0cfbd5f)
- chore(clippy): resolve all lib warnings (fixes + targeted allows) (32c82633)
- chore(lib): apply cargo fix — drop unused import + mark unused binding (e59b6b2f)
- chore(docs): move veld-vs-shodh-comparison.html to ../veld-docs/ (e285a431)
- chore: remove duplicate rlm_refiner.rs; fix comparison HTML; update generated docs (7f59e813)
- chore(gitignore): exclude entire evaluations/ directory (cc20ca03)
- chore(gitignore): exclude benchmark result snapshots from git (298c4ad5)
- chore(locomo): warm-veld result snapshot (MRR 0.917) (dbbdcd33)
- chore(carry): include llm_reranker.rs to unblock compile (63a75628)

### ci

- ci: the test suite runs on every push and pull request again (dddab160)
- ci(recall): give the held-out suite the job maximum of 360 minutes (80f18ff9)
- ci(recall): test the diff renderer and let a dispatch run the held-out suite (9e5c5e85)
- ci(recall): let a dispatch run every layer rung (adb99e27)
- ci: clippy denies the correctness lints on both builds (431c3575)
- ci(recall): compare the engines' scores and leave their latencies out (bc974cd7)
- ci(recall): cache the cross-encoder model so every repeat ranks with the same one (38ad5c5a)
- ci(recall): the recall gate measures veld's shipped NER, the rule-based one (6e769253)
- ci(upstream): CI fails when the upstream watcher has not succeeded for 48 hours (247501f1)
- ci(upstream): a daily watcher appends new upstream commits and release tags to the ledger (a22acbe3)
- ci(clippy): the clippy step fails when clippy does; three vacuous test asserts check something (df967de5)
- CI: tests get the ONNX runtime and MiniLM, and run on tags and dispatch (1110c5c9)
- CI: tests build without debug info, and a failing test fails the job (87a56fc9)
- CI: the Linux release build runs on release tags and dispatch, not every push (f49ca2e4)
- CI: macOS and Windows builds on release tags and manual dispatch, not every push; cancel superseded runs (6fe10988)
- ci(recall): add evidence-recall + latency regression guard (#22) (1d519476)
- ci: fix invalid YAML in version-check (heredoc dedented out of block scalar) (3ccec28d)
- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (945b045d)
- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (977468e8)
- ci: install NSIS via choco before makensis-action (3e7560f6)

### cli

- cli: the MCP and hook clients match the API, and --help stops printing secrets (cc3b3bc3)

### compression

- compression: the semantic pass is additive, and a destroyed record says so instead of returning its summary (b7fff76c)
- compression: the default path was lossy, and it labelled itself lossless (401c833e)

### config

- config: request rocksdb, because redb is unfinished and every request for it is served rocksdb (c11e9b7b)

### cross-encoder

- cross-encoder: pre-warm only downloads, on one worker, and never builds an ONNX session (0e298906)
- cross-encoder: pad a batch to its longest pair, not to 512 tokens (7b95c74b)
- cross-encoder: the exit segfault was a detached pre-warm thread, not rocksdb (79a80157)

### deps

- deps: clear rmcp, lopdf and pyo3 RustSec advisories (f15e6840)
- deps: clear rustls and sqlx RustSec advisories (cd7b23a0)
- deps: hold zenoh at 1.8.0 so twox-hash 1.6.3 and lz4_flex 0.10.0 leave the lock (f781e82d)
- deps: cargo update across the three lockfiles — 26 advisories drop to 9, and the 9 that remain need manifest bumps (#43) (408b8d77)

### docs

- docs(graph): retract a measurement error in the hygiene module comment (5658f916)
- docs(upstream): give every W0 row an outcome and name its veld commits (173aca18)
- docs(upstream): ledger rows for W5 name their veld commits (e56dc34c)
- docs(upstream): the pr-133 row records its two ports (e9848c75)
- docs(upstream): ledger rows for W1 name their veld commits (9cb8e30d)
- docs(decisions): ADR 0008 sets out what taking upstream's local IPC transport would mean (fca95f11)
- docs(upstream): the #325 row names the rebased #321 commit (4c701461)
- docs(upstream): the #325 row records the hebbian default from #321 (2f1f8239)
- docs(upstream): #462's repeat-stable graph ordering is ported in part (8dd40794)
- docs(upstream): the PMI gate (#336) belongs to the graph wave (2a7146ac)
- docs(upstream): triage the upstream commits filed under no wave (7640c886)
- docs(upstream): latency budgets from a live clone, gate status, and #399 re-read (5dd17e67)
- docs(upstream): veld tracks shodh-memory: the plan, ADR 0007 and a ledger of every upstream commit (65ed6e86)
- docs: parity status ledger at 2026-09-16 — landed, in flight, and findings that are not port bugs (bca037cd)
- docs(adr): 0006 — backend parity and on-demand migration (90fde9a8)
- docs(recall): four documented entrypoints that do not exist, and a gate nothing reads (c7f70702)
- docs: the evaluation engine is spine, not sleight (bb54d0e6)
- docs: handoff for the remaining parity work and the attribution defect (f9077efe)
- docs: dependency-ordered execution prompt for the rest of redb parity (6de3769a)
- docs(encryption): confidentiality boundary + production sign-off (H4) (c06e22cb)
- docs(sidecar): gen-cli-ref + versioned subdirs + embedded-docs release + CLAUDE.md sections (b93c160c)
- docs(sidecar): gen-changelog + lychee CI + smoke-test + consolidation accuracy (e981c9ce)
- docs(refresh): post-dials regeneration + dials + instance-registry pages (e4bff85d)
- docs(reference): regenerate config.md (92 env vars) (30b08696)
- docs(guides): add relational-backends operator guide (26771da3)
- docs(comparison): correct and extend veld-vs-shodh-comparison.html (e8a158b8)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (8429d8a2)
- docs(sidecar): polish generators, content, and plugin set (86f5b336)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (a395d995)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (345d3d83)
- docs: Phase A+B mdBook skeleton with all reference stubs (3d5eea75)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (794de654)
- docs(facts): SECURITY.md + decision record 0005 (967d31f4)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (dfb0181c)
- docs(sidecar): polish generators, content, and plugin set (68a3ffa6)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (c81d8b23)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (6b6cfa85)
- docs: Phase A+B mdBook skeleton with all reference stubs (a182628b)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (42d2f5a3)

### embeddings

- embeddings: nomic over HTTP is a primary, chosen deliberately, and the store knows which (235ec430)
- embeddings: the corpus marker recorded a dimension, so a model swap at 384 was invisible (de04f1db)
- embeddings: stop degrading silently, and stop inventing entity positions (78a530d5)
- embeddings: the configured window was 256, the tokenizer cut at 128, and the module built to stop that was feeding it (62d04ad7)

### facets

- facets: a stored Place can be read back (492a1c75)
- facets: serde(default) covers JSON, not the bincode store (88f298c0)

### facts

- facts: a purge reason is externally tagged on disk, and tagged-layout records still decode (7a9d3f5c)
- facts: a purged fact reads back purged (c300d1ce)

### feat

- feat(demo): a publishable demo profile and a screenshot harness (80f1c61f)
- feat(brand): savannah grass mark / grass sheaf (6c354820)
- feat(graph): key synapse strengthening on its source (424e95c1)
- feat(memories): exact tag filter on listing, and a filtered count (f4ae2e28)
- feat(monitor): add a Security tab for keys, recovery and logins (667edd37)
- feat(admin): administer logins and the keyring over HTTP behind 2FA step-up (ba2bb9b3)
- feat(keys): unlock the keyring with Touch ID or a passkey (8a4c9201)
- feat(keys): add a keyring with named recovery codes, device factors and rotation (2f5c0e60)
- feat(memory): an embedding backlog that drains itself, and a fallback embedder host (b85d1a9d)
- feat(cli): add reseal-journal to seal what a passphrase-only tenant wrote in plaintext (55780079)
- feat(recall): resolve declared file paths from a query, keep identifiers whole, trust NER less on code (6ce99ff3)
- feat(graph): typed entities declared by the writer, and a gate on generic derived ones (c46072a3)
- feat(graph): census of tags that sit on exactly the same records (4299406a)
- feat(graph): hygiene shelf on every entity row, and mentions per day or week (b046e2f6)
- feat(graph): classify entity names onto cast, vocabulary and noise shelves (4712c0ef)
- feat(ui): read-only /api/ui/* contracts for the integrated GUI (13c2783b)
- feat(memory): record each stored tag's source, and log every recall (9dba2b51)
- feat(upsert): carry episode, agent and writer provenance on upsert (8cceb6c1)
- feat(api): let /api/recall run one rung of the layer ladder (3e2d6eb8)
- feat(recall): replay a pinned model tape in the gate, and refuse to compare outputs from two sources (0f884b32)
- feat(recall-harness): record every model output of a gate run, so a later run can replay it (6ab967ec)
- feat(recall-harness): refuse to gate a run reranked differently from its baseline (7f8a5380)
- feat(recall): name the cases that moved against a per-case gate baseline (c1e5bc99)
- feat(recall-harness): pair every ablation delta against its reference, with a bootstrap interval (171723fd)
- feat(recall-harness): report which entity labels carry the graph, and the top hubs by name (773b66c0)
- feat(migrate): carry sealed records and the keystore between engines (cffb2cb1)
- feat(server): background writes run on a task tracker, and shutdown waits for them (08c84a77)
- feat(tui): add a Loops view of the self-evaluation loops (4099bf8d)
- feat(upstream): the ledger check fails on a port commit that does not name its upstream source (3d1c0e30)
- feat(storage): the semantic merge pass and its tombstone GC run on redb, as one implementation for both engines (4d06b690)
- feat(recall): the recall gate runs: recall-eval, a weekly workflow per storage engine, and an engines-equal check (e2502e73)
- feat(storage): record-level encryption (encryption-v2) on redb (bb872ca1)
- feat(upstream): a translator that turns an upstream commit into a veld-shaped patch and reports what applies (395c3b2a)
- feat(embeddings): embed over-long text in pieces, and serve the embedder on its own llama-server (43040971)
- feat(gap): gap analysis runs in the background and serves its kept result (bdaa1263)
- feat(memory): record who wrote a memory, and let a reader see it (f8d9ae21)
- feat(api): add /api/spine/dimensions, retain /api/sleight/dimensions as alias (01a4ea65)
- feat(temporal): intent-verb expansion + word-boundary precision + tense routing (#37) (c4854e74)
- feat(temporal): extraction patterns — explicit dates, "in N", pronoun-entity fallback (coverage 65%->~85%) (#35) (55aa5435)
- feat(recall): surface B4 derived temporal answer on /api/recall (eval-path fix) (#34) (1e9e7e4c)
- feat(temporal): answer-derivation surface — return derived date for temporal questions (B4) (#32) (9968ccc1)
- feat(dials): runtime disable-signals dial for per-signal recall ablation (#31) (dc510b39)
- feat(temporal): query-time event-anchor resolution for anchor-relative dates (#27) (d360ab22)
- feat(temporal): anchor-relative date resolution (the <weekday> before/after <date>) (#26) (c0233958)
- feat(memory): semantic merge via tombstone+audit, lazy remap (flag-gated, default off) (#24) (929d1118)
- feat(recall): shadow learned-weight injection + feedback update plumbing (live OFF) (#23) (af8c0ad7)
- feat(graph): path-credit reinforcement + bounded spontaneous recovery (#20) (13434f4f)
- feat(gc): inbound-ref index covers SemanticFact + TemporalFact citations (Fold 3) (#19) (88a0b3f4)
- feat(dials): expose recall ranking as runtime sliders (3988571a)
- feat(temporal): resolve before/after <named-event> via second hop (write-time, cycle-safe) (#15) (da66d2ae)
- feat(temporal): inferential After/Before/Fuzzy bounds (since/after/before/around) (#14) (a7b8a506)
- feat(memory): R3a — persist learned ranker weights (no behavior change) (e8364dad)
- feat(memory): wave 1 substrate — prefetch, reverse index, LTR shadow capture (d05e4b04)
- feat(temporal): validate temporal facts on event-time via bound algebra (#9) (c49ceca2)
- feat(crypto): bind primary records to their identity (anti-relocation AAD) (3c4b81a0)
- feat(intent-log): encrypt journal frame payloads at rest (f04d8fb5)
- feat(crypto): generalized AAD identity context — bind TOTP + slow-store secrets (204be106)
- feat(slow_store): per-column encryption toggles (payload / graph / embeddings) (673e090e)
- feat(storage): operator override to accept a restored (older) keystore (2fd6f31f)
- feat(keyctl): recovery-code recovery flow — recovery codes are now usable (e7ab3589)
- feat(crypto): multi-epoch record read — data-key rotation no longer orphans data (2e149cd5)
- feat(crypto): harden slow-store + TOTP secrets onto the v2 keystore (e66b845c)
- feat(keyctl): veld-keyctl rotation CLI (H5) (415a9ef6)
- feat(keystore): XChaCha20-Poly1305 record encryption (H3) — removes nonce ceiling (ea8bbe1b)
- feat(keystore): integrity MAC + rollback guard (H2) (41fca775)
- feat(oblivious,pir): encryption-v2 P6 + P7 — oblivious-access seam + 2-server PIR (55580bbd)
- feat(keystore,storage): encryption-v2 P5 — pluggable KMS unseal provider (7e5d35a8)
- feat(keystore): encryption-v2 P4 — passphrase rotation, recovery code, DEK rotation (fe281b52)
- feat(storage): encryption-v2 P3(b) — finish exact-match index blinding + rebuild-index (e14717d5)
- feat(keystore,storage): encryption-v2 P3(a) — index blinding (tag + entity) (b4ffdcd7)
- feat(storage): encryption-v2 P2(b) — record-level encryption wiring (record-only) (3b2f4621)
- feat(keystore): encryption-v2 P2(a) — record envelope primitive (fcbdd0ea)
- feat(keystore): encryption-v2 P1 — envelope keystore crypto core (f0b44b15)
- feat(monitor): custom frameless title bar with baseline-aligned controls (af200005)
- feat(monitor): directory browser for first-run path inputs (13d93a97)
- feat(cross-encoder): autodetect output head + input layout from ONNX session; swap BGE to apache-2.0 O4 export (5ef600b4)
- feat(handlers): LenientJson extractor for /api/remember (78c417ae)
- feat(cli): veld stop --port — stop a running server by its port (8f4296c0)
- feat(slow-store): encrypt the memory payload at rest (parity with the primary store) (2b1f1c7b)
- feat(alignment): v2 engine — payload hardening, conditioning, whitened-Procrustes, CSLS, perf-flag config (3e415ece)
- feat(cross-encoder): per-model rerank_top_k + VELD_RERANK_DEPTH override + CPU-O3 BGE export (85565c54)
- feat(monitor): first-run setup panel (seed a project + check Claude hooks) (d510a2bd)
- feat(dials): make max_active_shadows + working_memory_threshold runtime dials (92144c9c)
- feat(monitor): multi-instance discovery + tabs (consumes the registry) (ce9f3dae)
- feat(wiki): live edit-back watcher — optional, default-off [W-003/004 live layer] (745a6e94)
- feat(memory): file_answer + anti-autophagy, merge/split with char-safe ranges [A5/W-022/J3/W-021] (19b6e72c)
- feat(wiki): edit-back — Obsidian edits flow back to the canonical store [W-003/004] (648aec66)
- feat(wiki): authed POST /api/wiki/render endpoint [W-018] (0352e9b0)
- feat(wiki): WikiProjection foundation — wiki render as an intent-log projection [W-007/008/013/014/016] (aa2bcc1c)
- feat(c3.1): surface injection_suspected for non-trusted-source content [W-006] (33a8ed26)
- feat(c2,c4,c5): store↔log durability — reconcile-on-startup, Lsn ordering, bulk-forget journaling (300a04ca)
- feat(hooks): statusline discovers live instances from the registry (f7a80297)
- feat(monitor): expose the graph viewer through Tauri (c257a9dc)
- feat(graph-view): add manual Refresh button to the server graph viewer (4f6e915b)
- feat(storage): implement redb primary/graph/audit stores (7cf04117)
- feat(c3): surface source-trust at recall (injection boundary, W-006) (2f65c361)
- feat(c1): per-user write-path serialization lock (W-005) (c930d590)
- feat(p4): iterative re-cueing (pattern completion) + clippy sweep (ff92e750)
- feat(r3): populate EngramBinding conjunctive binding + surface reliability (4e69c1bf)
- feat(r6): bi-temporal as-of point-in-time queries (06f6f47e)
- feat(discovery): server-authored instance registry (replaces port-guessing) (4149d82a)
- feat(hooks): statusline accepts URL list as CLI arg; wire dual-server in settings (a42e5004)
- feat(hooks): wire Claude session → veld so the monitor sees live sessions (d717c102)
- feat(monitor): polish dials UI — ranges, tooltips, descriptions (c3f3025d)
- feat(monitor): Dials panel in Tauri tray monitor (get_dials/set_dials commands + status-core helpers + frontend) (4595987d)
- feat(dials): persist runtime dials via KeyValueStore (rocksdb kv CF accessor + load-on-ctor/persist-on-set) (d45b1064)
- feat(storage): implement real redb KeyValueStore (was a stub) + round-trip/scan/persist test (f4263445)
- feat(tui): add Dials tab (view + nav); documents reconsolidation/retrieval dials and how to set them (1af50401)
- feat(dials): MCP get_dials/set_dials tools (rmcp Rust + TS); client get/put (7fdf8ea4)
- feat(dials): GET/PUT /api/dials handlers (protected, per-user, sanitized) (8fdb6951)
- feat(dials): per-query retrieval overrides (refiner/rerank/candidate/rrf) via attached dials handle (46e4f753)
- feat(reconsolidation): Layers 2 & 3 re-encode — entity enrich + embedding nudge (dial-gated, single re-index) (4d4e05ac)
- feat(reconsolidation): Layer 1 re-encode — strengthen graph edges to retrieval-context entities (dial-gated) (a3c48951)
- feat(reconsolidation): run_reconsolidation_expiry Layer 0 (drain+decay), gated; called from run_maintenance (7bf7151f)
- feat(reconsolidation): accumulate retrieval_context across labile window; window length from dial (f746a3d7)
- feat(dials): seed CognitiveDials from VELD_* env at construction (7b7aa8c1)
- feat(dials): wire CognitiveDials runtime state onto MemorySystem (f3432808)
- feat(dials): add CognitiveDials runtime tuning struct (inert by default) (c0c78268)
- feat(e6): full TCM context_drift — graded temporal-context contiguity (dd666318)
- feat(cross-encoder): default to mxbai-xsmall + add jina-tiny/turbo MiniLM updates (ee5c01f1)
- feat(m4): return query-level entity coverage + weighted-available FOK (7d265105)
- feat(m3.1): cross-embedder agreement signal for query-level FOK (5b66dcbb)
- feat(cross-encoder): register mxbai-rerank-{xsmall,base}-v1 small-model options (7530bfb7)
- feat(cross-encoder): registry-driven catalogue + BGE-reranker-v2-m3 default (58f7416d)
- feat(m3): return query-level feeling-of-knowing (FOK) on recall (3ed3aa82)
- feat(cross-encoder): pluggable trait + ensemble support (d0db7520)
- feat(m5): wire gap_topology persistence + scope-filtered query-in-gap (91d65a7c)
- feat(w4): MssqlRelationalStore backend via tiberius (gated behind mssql feature) (8cf09688)
- feat(w7): dataset row CDC events (ca722e11)
- feat(w7): dataset MCP tools (rmcp Rust + TypeScript) (ba9cf877)
- feat(w7): tabular ingest mode — CSV/JSON -> relational dataset (a69740e4)
- feat(w4): wire a configured relational backend at startup (cutover activation) (97f59e84)
- feat(storage): ErasedRelationalStore — type-erase any backend to Error=BoxError (0ba55ee7)
- feat(w4): memories projection auto-uses the configured relational store (30f7ea2f)
- feat(w4): SupabaseRelationalStore — thin Postgres wrapper (gated) (c4362f18)
- feat(w4): PostgresRelationalStore backend (gated behind postgres feature) (f606b952)
- feat(w4): wire the memories projection through the RelationalStore trait (gated) (26c6ba52)
- feat(w4): BlockingRelationalStore — sync->async bridge for the projection cutover (1eb92bbe)
- feat(w4): extend RelationalSlowStoreAdapter to the memories projection table (94237eb3)
- feat(w6): /api/query/plan + /api/query/execute planner HTTP surface (47609e60)
- feat(w6): RealGraphQuerier over entity/episode topology (6d053dfe)
- feat(w6): real VamanaProvider for MultiUserMemoryManager via live writer (2412f089)
- feat(w6): RealRelationalQuerier + RealVectorQuerier + StubGraphQuerier (no HTTP) (1d5101fa)
- feat(sleep-time): V1 stage 2 + V2 stage 1 — orchestrator, worker, handlers, R27/R42/R43/R46/R48/R66 (02700269)
- feat(vamana): public memory_id_for(vector_id) reverse accessor (c2492be4)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (eb197297)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (0b2b8467)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (edf32be3)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (ead8b00f)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (42e050ce)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (f2ff6ffc)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (54f3a57d)
- feat(facts): destructive purge_facts under FACT_PURGE_LOCKS + reaper (5ce880f7)
- feat(facts): facts_preview_purge — bucketed, audit-logged, read-only (41b6408d)
- feat(facts): purged_at field + is_active filter + fact_narratives MCP tool (2b92620c)
- feat(w4): RelationalSlowStoreAdapter — parallel path proving trait against slow-store DDL (0cea1500)
- feat(w7): RelationalDatasetStore + LinkStore + HTTP CRUD (7ee315c1)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (bb6cb1fb)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (75df4606)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (40f22dd2)
- feat(w6): query planner data model + selectivity planner + executor skeleton (foundation) (65424e53)
- feat(w7): Dataset schema + DatasetStore trait + in-memory impl (foundation) (3de2ba9e)
- feat(w4): RelationalStore trait + SqliteRelationalStore (foundation) (e01ec5ce)
- feat(vamana): multi-embedding-kind projections + persist memory_id map via CheckpointStore side data (f6a3d4d3)
- feat(bm25): unify retrieval-side BM25Index with projection-owned Arc<BM25Index> (ed12dad9)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (3484cc9c)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (650aed4d)
- feat(intent-log): BM25 as third real TypedProjection (5ed09e03)
- feat(intent-log): Vamana as second real TypedProjection (05424c2f)
- feat(intent-log): SQLite as first real TypedProjection — journaled writes + replay (e5c1de33)
- feat(user-auth): admin promote/demote + persistent throttle + session revocation (d8b367c0)
- feat(memory): wire AgentSession into writer paths with detection helper (3f7405c0)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (127e9147)
- feat: wire ACL into reads, LLM into consolidation, schema-constrained extraction (#3) (3b2359b1)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (e7ad23ce)
- feat(alignment): corpus expansion tooling (MDN/cpython/Rust/React/TS) + multi-lambda Ridge sweep (68e959b7)
- feat(alignment): Procrustes + Ridge fitters, fit/eval bins, retrieval integration (#2) (1b0863dc)
- feat(hooks): write .veld-agent-session.<pid> marker on SessionStart (52dd1559)
- feat: agentic-memory feature parity — bi-temporal, self-paging, HNSW, ranker, connectors, LLM hook, ACL (#1) (e3739415)
- feat(user-auth): password + TOTP + recovery codes behind VELD_USER_AUTH_ENABLED (Phase C) (d2ea6669)
- feat(backup): include intent log + cf_user_auth; add --max-lsn PITR (2a350614)
- feat(intent-log): explicit format_version header + per-record schema_version (f7f9bfc3)
- feat(intent-log): Prometheus metrics + tracing spans (observability) (8f670a2a)
- feat(tui): git tab — branches, worktrees, recent commits (Phase A) (0bfb5683)
- feat(facets): AgentSession — worktree/branch/agent identity (8b82fda4)
- feat(scripts): veld-worktree-spawn — sibling-worktree-per-agent automation (d2d71595)
- feat(intent-log): JournaledWriter — log-first writer-side primitive (e2fbaa94)
- feat(intent-log): typed IntentPayload with bincode encode/decode (3173a160)
- feat(intent-log): Projection trait + replay driver + checkpoint store (eff7987d)

### feedback

- feedback: second secondary store off Arc<DB>, and a discarded flush error (ee8bb08e)

### files

- files: third secondary store off Arc<DB>, first user of ColumnStore (da47bfef)

### fix

- fix(demo): delete the tracked TUI debug log (4bc3de87)
- fix(demo): untrack the TUI debug log (57a76143)
- fix(embeddings): pin the default reranker and verify downloads (8d703d10)
- fix(tests): CI's two red tests on main, left by the key-management merge (65921528)
- fix(tests): the RocksDB-backed tests build only with the rocksdb feature (69cadafa)
- fix(memory): record the embedding backlog on the upsert path too (0d33ad70)
- fix(slow-store): seal payloads under the store keystore when it is the only one (1a14cfa6)
- fix(ui): count recalls from the recall log, not from access_count (e25f16d5)
- fix(bm25): keep an encrypted tenant's lexical index in memory (1d8526b5)
- fix(storage): migrate v0.7 field-encrypted content into the record envelope (4d922439)
- fix(recall-harness): leave the cross-encoder pre-warm pair out of a recording's table (40dea81d)
- fix(recall-harness): time the gate's recalls with no diagnostic recall between them (5acf877e)
- fix(benchmarks): count an unanswered judge call as an abstention, not option 0 (8c639a7e)
- fix(user-auth): refuse to start when the configured keystore cannot seal TOTP secrets (58da5560)
- fix(keystore): seal the journal under the store keystore when it is the only one (a71571d0)
- fix(migrate): a plaintext store migrated into rocksdb stays plaintext with a passphrase set (ef4a6c31)
- fix(todos): the flat todo aliases work, todo responses drop the embedding, and memories link back to the todos citing them (a5b6d018)
- fix(todos): todo search reads the embeddings stored on the todos, and todos link the memories and dependencies they cite (0a36e88d)
- fix(facts): extraction re-reads the whole eligible corpus, so an age-gated memory is no longer lost to the watermark (b44dc780)
- fix(memory): a content duplicate merges its enrichment, and every fact ingest goes through one arbitration policy (24ceaf4e)
- fix(facts): a plain declarative sentence becomes a fact candidate on every non-operational type (6573927c)
- fix(facts): a contradicted fact is superseded instead of living beside its correction, and a session promotion reaches storage (8415afab)
- fix(facts): extraction is gated by experience type and sentence shape, and support counts distinct evidence (2cb14cbb)
- fix(facts): protocol output that reaches a memory verbatim never becomes a fact (288a2d2a)
- fix(api): an unknown memory type, todo field or filter value is a 400, not a silent default (7dfa0892)
- fix(facts): the extractor no longer mints "X relates to Y" facts, and drops operational noise (3a2741dc)
- fix(query-parsing): a query cannot close its line in the LLM parser's prompt (980b722c)
- fix(files): a codebase scan or index reads only a project directory (97800bd7)
- fix(storage): a read error in a legacy-DB migration stops it before the old DB is renamed aside (aef47086)
- fix: graph handlers keep the graph lock off the runtime, and fact updates reindex (af09e173)
- fix(mcp): the recall_by_tags tool the server's help lists is registered (137858d9)
- fix(integrations): an insecure remote integration URL is refused by default, and the guard now covers Linear and GitHub (a3e4178b)
- fix(audit): audit writes run on the task tracker, so a drain waits for them (7a2e987a)
- fix(storage): say which engine is the default and what redb backups capture (98c77388)
- fix(recall): switch off the Layer-5 hebbian term by default (003a0645)
- fix(backup): close the tenant's stores before a restore replaces their files (bdb7f947)
- fix(vector-index): small indexes are searched exactly, and Vamana is built with its pruning rule (9a6b2609)
- fix(rerank): the cross-encoder honours VELD_ONNX_THREADS like the embedders (3b1f6eeb)
- fix(search): BM25 cutoff ties are ordered by content and RRF fusion keeps first-appearance order (0592199a)
- fix(recall): graph-leg candidates and fused-score ties are ordered by content, not hash order (8d332409)
- fix(graph): equal-strength edges and fuzzy name matches resolve by name, not by uuid or hash order (103e60c9)
- fix(backup): a RocksDB restore no longer deletes the tenant's data, and waits for its store to close (f44e1490)
- fix(boot-guard): a build without RocksDB reads an empty RocksDB store as empty (f25a17fd)
- fix(graph): deleting an entity deletes the relationships it is an endpoint of (0aa4aa57)
- fix(gap): the slow-store mirror syncs only edges between entities (4ace039b)
- fix(relevance): bound the graph walk /api/relevant runs per named entity (b2a47508)
- fix(chunking): chunk_text always advances; startup orphan repair persists progress every 500 (47ac00fe)
- fix(webhooks): an empty secret no longer verifies forged signatures, and malformed headers are errors (4b548c51)
- fix(mif): a Custom relation was renamed by its own backup, silently (f7e5b746)
- fix(decay): a rate learned per edge, and an L1 window that pruned a third of returns (5468917f)
- fix: four defects shodh-memory had already fixed, and one that helped hide them (643dbd5a)
- fix(durability): LSN-consistent backup cut, both-direction reconcile, honest init errors (8d431d8b)
- fix(alignment): live-secondary autofit + canonical pair_id + wire aligned secondary into recall (#38) (1a91627a)
- fix(memory): compute secondary embedding on live ingest so dual-embedder index populates (#33) (66a8c138)
- fix(benchmarks): make locomo_mc10_eval OOTB-correct on mac/windows (26722a07)
- fix(cross-encoder): fail soft when ONNX Runtime is unavailable (223f19ec)
- fix(temporal): resolve N-months/years-ago to MonthYear/Year; drop entity leak from event_stems (#13) (f768cf71)
- fix(monitor): authenticate graph preview (4d1fd7e5)
- fix(monitor): authenticate graph preview (b1899fff)
- fix(storage): surface decrypt failures in forget/decay passes (veld) (13406a55)
- fix(2fa): fail closed on TOTP encryption — never store a 2FA seed in plaintext (4e1b5d0a)
- fix(storage): blind episode_id in the episode_seq index (3b91284e)
- fix(keystore): reject KDF-downgraded Argon2 params on unseal (dd9eb6ef)
- fix(keyctl): disarm rotate-dek data-loss footgun (2f8870d7)
- fix(crypto): wire operator signals to the keystore + fail closed on partial config (1a6c20f8)
- fix(keystore): bound Argon2 params from untrusted keystore (audit/breakers) (ecbb9a6d)
- fix(keystore): atomic keystore persistence (H1) — temp+fsync+rename+backup (f28e6d45)
- fix(keystore): zeroize intermediate key-material buffers (breakers finding) (e2b3b336)
- fix(security): port shodh-memory PR #285 hardening to veld siblings (8ef863bc)
- fix(monitor): minimize to tray on close; start visible; drop strict CSP (7af1bc91)
- fix(slow-store): log unreadable rows instead of silently dropping them (33cbc872)
- fix(slow-store): run retention cleanup deletes in one transaction (8a3a5a81)
- fix(slow-store): purge entities/edges dropped from the graph on sync (dfcc822a)
- fix(monitor): self-heal dials + add graph reload (resilient to server flaps) (26620604)
- fix(monitor): open graph viewer in-window (iframe), not a crashing OS window (57fc4c32)
- fix(relational-bridge): bound the synchronous recv so a hung backend can't wedge a worker (48196654)
- fix(slow-store): edge sync aborted on multi-relation entity pairs (c264474a)
- fix(mcp-server): make the server typecheckable and tsc-clean (c1fda011)
- fix(tests): initialize SemanticFact purge fields in adaptive_memory_tests (84a1e28a)
- fix(relational): bridge_block_on returns Result instead of panicking (1b57c9b5)
- fix(datasets): dialect-correct relational dataset layer for Postgres and SQL Server (259c3e8b)
- fix(m5): clippy explicit_auto_deref at gap-persist call sites (8e557ab6)
- fix(w4): dialect-correct memories DDL/upsert for Postgres/Supabase/MSSQL (c849ca54)
- fix(cross-encoder): track HF main + pre-warm on startup (878979b7)
- fix(tests): complete BoxError migration + facts/visibility in test harness (12a11f01)
- fix(build): unblock lib + bin compile (40 errors) (4b5f5125)
- fix(retrieval): restore rlm_refiner.rs accidentally dropped in HTML commit (dbd9ede6)
- fix(relational): switch type-erased Error from anyhow::Error to BoxError (StdError-compatible) (ad01e889)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (f0d92861)
- fix(facts): add purged_at/purge_reason to inline-test SemanticFact constructions (9d1ea279)
- fix(w4): close pluggability gaps — Row::from_owned, RelationalBackend::Custom, docstring example, Cargo dupe (8cbb3305)
- fix(memory): finish AgentSession writer-read — marker file + writer paths + e2e test (1e190e00)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (9d2b777e)
- fix: brand whitelist + argon2 semaphore + 503-on-disabled-auth + recovery-code charset audit (c68771fc)
- fix(alignment): paths-relative-to-CWD in TOML specs; add pairs.vec cache (619550f1)
- fix(intent-log): repair conflict-resolution build breaks (a7f73dc1)
- fix(user-auth): session middleware sets Extension<AuthenticatedUser> (f3a71b07)

### graph

- graph: clock-injected decay and the causal walk the recall harness needs (a1525934)
- graph: a read-only recall walks edges without strengthening them, and the hub cap prunes the same edges every ingest (afc83727)
- graph: the last store off Arc<DB> — nine families onto ColumnStore (00a80471)
- graph: delete_entity left its entity_episodes index rows behind (847a09f8)
- graph: an entity's merge scope is a property of its type, not of its name (1538d759)
- graph: the entity namespace had five ways in and none out, and UUID entities are ruled to stay (c7b79186)

### handoff

- handoff: the remaining storage work, dependency-ordered (5eaa32db)
- handoff: the suite segfaults on exit intermittently, and it predates this work (29774221)
- handoff: the traits were never the bottleneck — the stores outside them are (4cc0ac2c)
- handoff: replace the estimates with the executable measurement (3bbc7c18)
- handoff: R-6a/b is a prerequisite for R-3, not a follow-on (8a09da4d)

### health

- health: report tenants with data on each backend, measured at boot (0defd53a)

### ingest

- ingest: type causal lineage verbs as Triggers so the causal walk has edges to follow (433ad8a0)

### lib

- lib: declare recall_harness, which 4d5d44d committed but never referenced (f5934798)

### licence

- licence: veld is published under an evaluation licence (aa2ad197)
- licence: BUSL-1.1 → AGPL-3.0-or-later, and the Apache-2.0 conditions the fork never met (d987919c)

### list

- list: offset paging, a lifted cap, and a preview that says it is one (a4287ed8)

### mcp

- mcp: expand VELD_* from the environment, dev key only as fallback (5dcc0592)

### metadata

- metadata: discovery terms chosen from measured adoption, not from what the code calls itself (76512eec)

### minilm

- minilm: the runtime extractor could only ever match a symlink, so nothing had ever been installed (b5b414d1)

### misc

- slow store: one mirror rebuild at a time, and the TTL counts from its start (d9a362f9)
- Merge pull request #50 from Portll/fix/pin-reranker (10ffd49a)
- Merge pull request #49 from Portll/fix/ci-tests (94a62d9a)
- Merge pull request #48 from Portll/licence/evaluation (fa398834)
- Merge pull request #47 from Portll/wave/w0-land (a65fefa8)
- Merge wave/w0-measurement: the recall gate, per-case baseline, model tapes and the W0 ledger outcomes (0f1b152f)
- Merge pull request #46 from Portll/fix/hub-cap-test-race (e816e311)
- Merge pull request #45 from Portll/feat/key-management (3b845d95)
- Merge origin/main into feat/key-management (63a60f93)
- Merge pull request #44 from Portll/feat/typed-entities (8191c4f8)
- recall harness: the ablation arms measure the thing they name, or say they cannot (feb4a42a)
- cargo fmt over the files the ablation port touches (2ec298bb)
- session detect: resolve the branch of a repository with no commits yet (12efd7c8)
- recall harness: seven tests that measure a veld difference are ignored, with the finding as the reason (7f442f7e)
- recall harness: declare the eight remaining modules (116ae69f)
- shared stores: boot the rocksdb arm through with_store and delete the legacy constructors (e8d84201)
- shared stores: prove the rocksdb with_store constructors read what the legacy ones wrote (2bdf4035)
- LoCoMo 10x runners and the visual model page (7b6ec16c)
- Stop tracking .claude/settings.json; add a generic settings.json.example (30c8a489)
- Benches compile again; a 14-day point on the L2 decay curve; CLAUDE.md architecture notes (8f29b779)
- Ignore the cargo fmt commit in blame (473cb3f3)
- cargo fmt: format the tree with CI's rustfmt (1.9.0, Rust 1.98.1) (0069820d)
- sleep_time: the four stores and the orchestrator build from a ColumnStore (d015be38)
- storage tests: the decoder-counter tests take one lock, because the metric they assert on is process-wide (f4fd752f)
- mif export: a store read that fails fails the export (4abe8e7b)
- expose index coverage, and mount the probe routes the harness never had (ca6b2f9d)
- reconcile stored against indexed, and delete a test that proved nothing (2f0648d5)
- recall harness: a report that cannot say what embedded it says unknown, and unknown is refused (1c4c2ba8)
- recall harness: a report names the embedder that ran it, and a diff across embedders is refused (117cbe32)
- get memory: the response says whether its content is the document (1398c68d)
- path_guard: one user-id rule, checked at the request and at the filesystem (4d50116c)
- index-insert loss is counted too, and the counter is proven to move (5c57be78)
- recall harness: 45 errors to 5, and a silent-loss guard veld did not have (e39ce42c)
- harness APIs: three of five ported, and two that are not 15-line ports (fa72733a)
- recall capture: the two modules, four wired stages, and a corrected gap (3c8a3028)
- sleep_time: the last four stores off Arc<DB> (b40ba6ab)
- recall harness: record the measured gap for the eight gated modules (a9e48af8)
- ablation switch: the retrieval pipeline can be run one rung at a time (7caedd3d)
- redb boots: the last construction refusal is wired, and the request is honoured (e1cd6c95)
- recall harness: import upstream's measurement layer, with its corpora (4d5d44d9)
- auth seam + four fixes shodh-memory already merged (ede90e40)
- R-5: MemorySystem constructs over the trait, and redb's arm is wired (5acb70a1)
- facts, lineage, learning-history, temporal-facts: four default-keyspace stores off Arc<DB> (d5004b55)
- R-5 (first half): the shared bundle and backup engine stop assuming their engine (ba08f303)
- Merge remote-tracking branch 'origin/main' into HEAD (999761ec)
- secondary stores: with_store constructors, and ensure_cf closes a masked divergence (dc8bd28a)
- context_blocks: the first secondary store off Arc<DB> — and onto the LIGHTER trait (3ddf5df1)
- R-6c: verify an imported target against its dump, both directions (c1932db9)
- R-6b: portable import for redb, and cf:None meant two keyspaces (c874dce0)
- R-6a: portable export for rocksdb, with a format that refuses a partial read (51bf2e26)
- forget_all: erase facts through the store, not through db() (f1bcd39e)
- storage handoff: A's batch API must be DB-scoped, because 4 call sites already batch across column families (6323192d)
- storage handoff: the order is A then B, and my own recommendation was the wrong way round (77d5eb4a)
- R-1 storage: the TODO under-scoped the work fourfold, and the count was never the hard part (139d96aa)
- a backup that omitted a store said success, and a spread that went nowhere (5cf5a64c)
- entity cap + NER: the cap kept the wrong fifty, and NER never saw past token 128 (1c8a85a8)
- llm hosts: carry the exposure facts — which hosts bind every interface, which fetch models on demand, and that runners are off by default (59fbd85f)
- llm hosts: carry the declaration's contextNote — the KV-cache guardrail ollama's removal cost, recorded where both repos can see it (95a4e5e1)
- llm hosts: ollama out, and the two defaults that disagreed with each other are now one declaration (2a507bf1)
- onnx extractor: checking shodh-memory found the bug fixed there first, and my fix wrong in the way theirs documents (ee166804)
- entity tail: the tail is nineteen times more selective than the head, so cut-the-tail was backwards (82666c53)
- graph sweep: the population says zero live secrets, and the tenant binding I claimed yesterday is not compiled (be74b8da)
- graph-entry doc: the design prescribed a rule the implementation refuses, and said so nowhere (f20913a7)
- graph entry: an entity is a one-way door, so redaction has to happen on the way in (7a4507cc)
- Merge remote-tracking branch 'origin/main' (c0516a24)
- Merge branch 'feat/encryption-v2' (97de98e5)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (52c34599)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (a73f974a)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (4168bd76)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (c3b3bcc0)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (76b4f7a4)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (06c356d2)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (920aea4f)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (7406fdd6)
- Merge branch 'main' into reconcile/unify-mainline (a1175cc8)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (e7e7ec03)
- Merge origin/main into local main: facts + sleep-time + relational + intent-log (efc8a2b9)

### monitor

- monitor: working Tauri GUI — IPC, tray, X-quit, diagnostics (#4) (2c7fbd5f)

### nomic

- nomic: unblock concurrent inference + dynamic seq-len + split metrics + hook debounce (#6) (7c5b05f6)

### observability

- observability: a wrapper that renamed its model, and a warning that was 26% of the log (bcc2b19f)

### packaging

- packaging: add cross-platform service installation scripts (5fe17aa1)
- packaging: add cross-platform service installation scripts (fce5e337)

### perf

- perf: score each rerank pair at its own length, not padded together (92a21c8a)
- perf(gap): find open triads and diamonds in memory; count orbit links in one pass (fe689f8e)
- perf(recall): batch + de-duplicate cross-encoder reranking (14.5s -> ~1s target) (#28) (f74f9c93)
- perf(feedback): batch pending-bound enforcement off the per-insert recall hot path (#16) (865b79a3)
- perf(slow-store): amortize the live-arm checkpoint fsync over a batch (1a1ca4ad)
- perf(slow-store): add a read-connection pool so reads don't serialize behind the writer (7c4c9757)
- perf(slow-store): batch entity-name and embedding lookups into one query (46de43b4)
- perf(slow-store): collapse find_star_gaps inter-spoke check to one query (1bed97bb)
- perf(retrieval): reuse stored embeddings in bi-encoder rerank (d04f7224)
- perf(retrieval): reuse stored embeddings in bi-encoder rerank (22bb6ae5)

### projections

- projections: decode snapshots through the store's legacy arms (8d7de2ed)

### prospective

- prospective: fourth secondary store off Arc<DB>, both cross-family batches preserved (c4223e07)

### recall

- recall: a filter-only recall returns every match instead of letting competition drop them (0721c8cd)
- recall: a read-only recall writes nothing, and the harness's flag is finally read by something (2c3f1a59)
- recall: the tags filter reaches the query (29108a10)
- recall: a failure is not an empty result set (cee2f134)

### redaction

- redaction: credential shapes get a class label before the graph, and the lexicon is measured not imagined (ef9a5ea5)

### redb

- redb: BackupStore and portable export — 57/69 to 66/69 (1ee74700)
- redb: 29/33 to 31/33, and the last two refuse instead of panicking (724d08bf)
- redb: seven more PrimaryMemoryStore methods — 22/33 to 29/33 (7eb39ce2)
- redb: six PrimaryMemoryStore methods, matching rocksdb's semantics not just its signatures (efd299f4)

### refactor

- refactor(loops): code runs from the repo, data stays where it is (67d41d37)
- refactor: the user auth store reads and writes through ColumnStore (090447d2)
- refactor: de-metaphor earth/roots identifiers to literal names (#25) (0a812f87)
- refactor(recall): compute temporal boost via bound overlap (align with gate) (#18) (c61e1329)
- refactor(graph): single idempotent reinforcement coordinator for edge.strength (#17) (ce3ec0db)
- refactor(temporal): resolve relative query dates to TemporalBound (F6) (#12) (7c5fa41c)
- refactor(temporal): remove dead EventType::Recurring phantom variant (#10) (915fe123)
- refactor(user-auth): rename field_encryptor -> totp_encryptor + drop TOTP hex detour (8b8aaa97)
- refactor(crypto): remove orphaned v1 FieldEncryptor (b934c324)
- refactor(refiner)!: rename RLM->LLM refiner; consolidate duplicate module (3f68e712)

### release

- release: 0.8.0 — Data panel, restart with launch recipes, live replication (5fbc9405)

### remember

- remember: a caller may not assert its own trustworthiness (3600381d)

### restore

- restore: deleting the graph before knowing the replacement copied is not a restore (80b6ed5a)

### retrieval

- retrieval: memory-tier trust scales the graph evidence, not the weight mix (daddfcc4)
- retrieval: every truncating sort has a total order, and a tag search has an order at all (7aa8336c)
- retrieval: the engine reads through the trait, and the trait grew the two methods it proved missing (00f46001)

### security

- security: confine per-user paths to their base directory, and stop printing the full API key (#40) (49a15194)

### storage

- storage: import rocksdb::Cache once, and the memory ratchet falls to 67 (d311d34b)
- storage: the write latch is per store, and the public body names a store by digest (e4193ad7)
- storage: a NoSpace write latch is detected, bounded-retried, and reported instead of served as healthy (c9f2c24f)
- storage: an undecodable value stays undecodable; lazy migration writes only under its own key (fde22bb4)
- storage: veld export / import / migrate between rocksdb and redb, verified (01e50e87)
- storage: the probe opens a copied store through a read-only helper, not rocksdb:: directly (ae0197a9)
- storage: restore swaps by rename-aside, and SPANN keeps inserts after load (0e8df63e)
- storage: recover records persisted with a pre-f8d9ae2 AgentSession (187fa3a3)
- storage: an undecodable record is named, not counted (0e7d1840)
- storage: a ratchet so rocksdb:: outside src/storage/ can only fall (379c130b)
- storage: refuse to boot an empty backend while the other holds the data (6896b0dc)
- storage: the full-consumption guard belongs to the two tiny candidates, not the full-memory ones (6b8065c9)
- storage: a legacy decoder that read half a record was calling it a match (3a818d42)
- storage: ColumnStore — the multi-family shape the five secondary stores need (a77805b0)
- storage: an executable coverage matrix — every trait method CALLED on both backends (69fe1ebe)
- storage: PrimaryMemoryStore gains the 21 methods MemorySystem calls (0d1ec38e)
- storage: a DB-scoped batch API, so supports_transactional_batch means something (bd4ef572)

### style

- style: cargo fmt over the files main left unformatted (0eb54408)

### test

- test(graph): wait for remember's background work before reading the capped hub (593d9e81)
- test(graph): name the frequency fixture tuple clippy flagged as type_complexity (36eb1517)
- test(facts): measure what fact distillation yields on the recall corpora (73c8f79a)
- test(stress): the two graph-integrity asserts in the brutal stress tests can fail (15e1d82f)
- test(storage): lower the memory ratchet to the 68 rocksdb uses left (c4164c7c)
- test(recall): record the LoCoMo gate baseline at 2f1f8239 (796e27b1)
- test(recall): an ignored test ingests the gate corpus twice and compares every rank list (9d2be1bc)
- test(storage): snapshot the layout of every bincode-persisted root, not only Memory (25e25174)
- test(backup): backup and point-in-time restore leave the same store on both engines (d420b3f6)
- test(auth): every protected route refuses a request without an API key (33acf7d2)
- test(storage): pin the position of every variant a bincode record stores (ead61862)
- test(storage): the trait coverage matrix fails when redb lacks a method RocksDB has (bd54343d)
- test: four more memory stores run on both engines (18078014)
- test: temporal facts run on both engines, not only rocksdb (b188daf1)
- test(graph-memory): 34 integration tests that had never run, and 4 recovered without widening the API (86b7a446)
- test(temporal): anchor-relative dual-path contract + q53/q63/duration regression guards (#30) (2adf2a1c)
- test(temporal): integration test for event-time gate validation (#11) (b9cb2979)
- test(wiki): end-to-end render / edit-back / forget demo (aceb95cc)
- test(reconsolidation): Layer-0 expiry gate + drain unit tests (passing) (fbda2021)
- test(stress): gate brutal_stress_tests behind a stress feature (fe053fc5)

### tests

- tests: every bincode-persisted root is walked for a type bincode cannot decode (6b27a538)
- tests: the benchmark's latency bound moves to its own ignored release-only test (41406580)
- tests: BUG-007 asserts scaling, and benchmark latency is the minimum of warm runs (75a63711)
- tests(webhooks): the handler's empty-secret and malformed-signature paths are exercised end to end (1e91bf97)
- tests: the persisted layout of Memory is pinned, and changing it fails (4cb90969)
- tests: a source file no mod declaration reaches fails the suite (283a034c)
- tests: three of veld's four red tests were wrong about the code, not the code about them (875acd4e)

### todos

- todos: fifth secondary store off Arc<DB> — the largest, and the last (b0f6e258)

### vamana

- vamana: an id map is only valid for the index it was built against (e5973654)
- vamana: the test that was wrong was hiding a defect that is not (66396f0c)

### versions

- Versions: mcp-server/package.json and pyproject.toml to 0.7.39, matching Cargo.toml (e29b4e8b)

## v0.7.18

### ci

- ci: install NSIS via choco before makensis-action (07343231)

### feat

- feat(embeddings): alignment scaffold + collect bin (Phase 1) (fac25c5d)

## v0.7.17

### feat

- feat(intent-log): durable checksummed append-only log (W5 scaffold) (b2eada2a)

### fix

- fix(security): TLS posture warning + keyed-store eviction sweeper (e09d7468)
- fix(security): structural circuit breaker on the public router (669a58e5)

### misc

- Fix veld hook commit decode error + add schema regression tests (346d5fee)

## v0.7.14

### build

- build(release): autobump pre-commit hook + finish v0.7.10 packaging (baa78bf4)

### feat

- feat(memory): add Orientation to the facets re-export (61867691)
- feat(memory): geospatial enrichment — LocalFrame, GeoFix, Orientation (37a0a481)
- feat(python): memory-to-dict reads via resolved_* (W3 readers) (2858e897)
- feat(retrieval+storage): spatial sort + RocksDB indices via resolved_* (W3 readers) (a1c6ae58)
- feat(query): Query::matches reads via resolved_* accessors (W3 readers) (feafb771)

### fix

- fix(security): three W2 hardening items (rate-limit clamp, audit, fixed-buffer compare) (73b853c0)

### misc

- Fix CI release + memory hook auth + add bump-version tests (e373a5cb)
- Merge branch 'w3/reader-switch' (7b517ccd)

## v0.7.8

First tagged release tracked by this changelog. For pre-v0.7.8 history see `git log`.
