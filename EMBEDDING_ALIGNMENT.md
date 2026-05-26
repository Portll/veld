# Embedding Alignment Proposal

**Author:** john@portll.net
**Date:** 2026-05-27
**Status:** Proposal — pre-implementation
**Scope:** `src/embeddings/`, `src/memory/retrieval.rs`, new `evaluations/alignment/`

---

## Summary

`CompetitiveEmbedder` currently holds a primary and an optional secondary embedder and forbids cross-space vector math by design — the doc comment says so, and the retrieval layer merges only at the score level via max-score union across two independent Vamana indices. That is the correct default. This proposal adds a controlled exit from that default: an `Alignment` abstraction that learns a projection from the secondary embedding space into the primary space, so that *with an explicit alignment installed* veld can do cross-space cosine, true score-space fusion in a shared geometry, and consistent placement of Claude-produced artifacts in a single "task space."

The alignment is trained once on a modest paired corpus (~30–50K text pairs spanning the working domains), persisted alongside the secondary Vamana index, version-pinned to the embedder model identifiers, and validated by a held-out retrieval evaluation. The default method is orthogonal Procrustes — closed-form, seconds to fit, geometry-preserving, requires no hyperparameter tuning. Ridge-regularised linear and a small MLP are provided as fall-backs for cases where the two spaces are not isometric.

---

## Motivation

Three concrete capabilities are blocked today by the no-mixing rule:

1. **Same-space fusion at retrieval.** The current max-score union treats the secondary index as a parallel oracle; it cannot do reciprocal-rank fusion in a shared metric, weighted averages of nearest-neighbour distances, or centroid queries that pool evidence from both models.
2. **Cross-index migration without re-embedding.** The migration path in `retrieval.rs` already opportunistically reuses a secondary embedding *when its length matches the new primary target* (the "Prefer an existing secondary embedding already at the target length" branch). With a learned alignment, the same vector reuse becomes available even when dimensions differ — a one-off projection during migration instead of a full re-encode of the corpus.
3. **Regular task space for Claude artifacts.** A project produced by Claude (a new component, schema, ticket, query, analysis) currently lands wherever the *primary* embedder happens to place it. If the secondary embedder is more discriminative on, say, code structure, but lives in its own latent space, the system cannot exploit that. With an alignment, every Claude artifact can be projected to a canonical point in primary-space *informed by both lenses*, making "find related projects," "is this a duplicate of something we built before," and "cluster Claude's outputs into task families" tractable as ordinary vector queries.

Alignment is the principled exception to the no-mixing rule, not a relaxation of it.

---

## The `Alignment` abstraction

A new module `src/embeddings/alignment.rs`:

```text
pub trait Alignment: Send + Sync {
    /// Project a secondary-space vector into primary space.
    fn project(&self, secondary: &[f32]) -> Vec<f32>;

    /// Batch projection (default: sequential).
    fn project_batch(&self, secondary: &[&[f32]]) -> Vec<Vec<f32>> { ... }

    /// Dimensions: secondary input → primary output.
    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;

    /// Identifier of the (primary_model, secondary_model) pair this alignment
    /// was trained on. Used as a guard at load time.
    fn pair_id(&self) -> &str;

    /// Persist/load to a versioned file format.
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self> where Self: Sized;
}
```

Concrete implementations:

- `IdentityAlignment` — no-op, used when both embedders are the same model or when alignment is intentionally disabled. Asserts `in_dim == out_dim`.
- `OrthogonalProcrustesAlignment` — closed-form, default. Holds a `d_out × d_in` rotation matrix `R` with `R^T R = I` (restricted to a partial isometry when dims differ). Projection is one matrix-vector multiply.
- `LinearRidgeAlignment` — least squares with L2 regularisation. Used when the two spaces are not isometric (rare for normalised sentence embedders, common when one side is a CLIP-style model or has a different pooling head).
- `MlpAlignment` — small (one hidden layer) MLP, gated behind a feature flag, only fit when the linear methods miss a target retrieval threshold on the held-out set.

`CompetitiveEmbedder` gains one new field:

```text
alignment: Option<Arc<dyn Alignment>>
```

and one new method:

```text
pub fn encode_aligned(&self, text: &str) -> Result<Option<Vec<f32>>>;
```

which encodes with the secondary, then projects to primary space, returning `None` if either the secondary embedder or alignment is absent. The existing `Embedder` trait impl is unchanged — alignment is strictly additive.

**Invariant:** any code path that produces a vector intended to live in *primary space* must either come from the primary embedder directly or from `encode_aligned`. Raw secondary vectors are never used in primary-space operations. This invariant is enforced by giving `Alignment::project` a distinct return type wrapper (`PrimarySpaceVector`) in the type-strict variant; or, in the simpler variant, by routing all cross-space operations through one module so the rule can be code-reviewed in one place.

---

## Training methods

**Default: orthogonal Procrustes.**

Given paired matrices `A ∈ R^{n × d_p}` (primary embeddings) and `B ∈ R^{n × d_s}` (secondary embeddings), solve

```text
R* = argmin_R ||A − B R||_F     subject to R^T R = I
```

Closed-form solution via SVD of `B^T A`. Fits in seconds on 50K pairs. Requires no hyperparameters. Preserves angles and norms — appropriate because both veld embedders already L2-normalise their outputs. When `d_s ≠ d_p`, restrict to a partial isometry (the top `min(d_s, d_p)` singular vectors) and either zero-pad or zero-truncate the output, with a documented one-time loss in expressivity.

**Fallback 1: ridge regression.** When the two spaces are not isometric — typically when one of the embedders is asymmetric (Nomic with its query prefix) or uses different pooling — Procrustes residuals will be visibly worse than a free linear map. Ridge with cross-validated regularisation gives the unconstrained optimum without overfitting.

**Fallback 2: MLP (one hidden layer, width ~2× output dim, GELU, dropout 0.1).** Only used if Procrustes and ridge both fall below the retrieval target. Adds nonlinearity at the cost of more data, more training time, and worse interpretability. Feature-flagged.

**Rule of method selection:** start with Procrustes. If held-out cosine of paired vectors is below 0.85, try ridge. If ridge is below 0.85, try MLP. If MLP is below 0.80, the two embedders likely encode genuinely different things and a shared space is not the right abstraction — fall back to score-level fusion only.

---

## Dataset design

The training set is paired text: each row is one piece of text that both embedders encode, yielding `(primary_vec, secondary_vec)`. The signal is *identity of the input*. Quality matters more than quantity at this scale; Procrustes converges on 5–10K pairs and saturates by ~30K for typical sentence embedders.

**Target size: 40,000 paired texts.** This is the "modest but efficient" target.

**Composition (per domain, ~5K each):**

- **Web development (5K).** React/Vue/Svelte component snippets, HTML/CSS fragments, REST and GraphQL endpoint definitions, frontend bug reports, accessibility notes, browser-compatibility entries, build-config snippets (Vite, Webpack), CSS rule clusters.
- **Project management (5K).** Issue and ticket descriptions (Linear/Jira/Asana style), milestones, sprint goals, retro notes, dependency-blocker statements, RFC abstracts, design-doc summaries, release-note entries, status-update paragraphs.
- **Database design and management (5K).** Table schemas (DDL), ER-diagram captions, migration descriptions, index-strategy notes, query plans, lock-and-isolation discussions, replication setup, sharding rationales, NoSQL document shapes, vector-index configs.
- **Programming (5K).** Function docstrings, commit messages, code-review comments, stack-trace summaries, refactor notes, test cases (unit, integration, property-based), error-message clusters, configuration snippets, build-system rules.
- **Analytics (5K).** SQL queries (OLTP and warehouse styles), dashboard widget specs, metric definitions, KPI summaries, dimensional-model fragments, exploratory-analysis narratives, statistical-method captions, A/B-test write-ups, anomaly-detection notes.

**Gaps the user did not name that I am adding (~3K each, 15K total):**

- **DevOps and infrastructure.** CI/CD pipeline configs, deployment scripts, Terraform/Pulumi fragments, monitoring alerts, runbook entries, incident reports, SLO and error-budget statements. *Why:* you cannot do project-management retrieval cleanly without ops artifacts being co-embedded; "the build is broken" lives at the seam.
- **Documentation and specifications.** README sections, ADRs, RFC bodies, API reference paragraphs, changelog entries, onboarding guides. *Why:* docs are how projects describe themselves. Without good doc coverage, the alignment will systematically misplace anything written *about* code as opposed to code itself.
- **Security and review.** Threat-model entries, dependency-audit notes, secret-handling rules, code-review findings, vulnerability descriptions, license-compatibility notes. *Why:* security artifacts are linguistically distinct from the code they protect; without explicit coverage they collapse onto unrelated neighbours.
- **Testing strategy.** Test plans, coverage notes, fixture definitions, mock and stub rationales, flake reports, performance-regression entries. *Why:* test text often surfaces as the nearest neighbour to production code, and you want that signal clean.
- **AI/Claude-loop artifacts.** Prompt definitions, system messages, tool schemas, evaluation-set entries, prompt-regression notes, model-output critiques. *Why directly relevant to the goal:* if Claude is the producer, the things Claude consumes (prompts, schemas, evals) need their own cluster in the task space, otherwise Claude's outputs will pile up against its own inputs and retrieval will degrade.

**Where the pairs come from:**

The dataset is deliberately built from *code and public sources only* — no email, no chat threads, no document stores, no personal correspondence. MCP stores (Gmail, Drive, etc.) remain connected and connectable for *runtime* memory operations, but are explicitly excluded from the alignment training corpus. This keeps the published alignment shareable, defensible under license, and free of any tenant's private data.

1. **User-owned code repositories.** `Portll/*` repos including veld and sleight — source files, commit messages, code review comments, ADRs, READMEs, and design docs already under the user's license. This is the highest-signal in-scope source. Estimate: 8–12K pairs available immediately from veld/sleight alone.
2. **Public domain-specific sources.** Permissively licensed code (GitHub public repos under MIT/Apache/BSD with attribution preserved), Stack Overflow questions and accepted answers (CC-BY-SA, attribution required), public dataset cards (Hugging Face, Kaggle), API documentation from major frameworks, RFC and W3C specs.
3. **Synthetic generation.** Claude generates paired text for under-covered cells using a balanced template ("write a [kind of artifact] about [topic] in [domain]"). Used to fill gaps in the public sources, especially for DevOps, security review, and AI-loop artifacts. Capped at ~25% of any one domain's pairs to avoid alignment collapsing onto Claude's own stylistic priors.
4. **Friends-contributed corpora.** A documented contribution path for collaborators to donate paired-text corpora from their own working environments (with the same code-and-public-sources constraint). Contributors submit a JSONL file of `(text, domain, license, source)` rows; veld encodes them locally with both embedders and merges them into the global pairs file. This is the single highest-leverage source for breadth — five friends contributing 2K rows each doubles the available signal in domains where the maintainer has thin coverage. Contribution guide ships in `evaluations/alignment/CONTRIBUTING.md`.
5. **Snapshot of Claude's prior outputs.** Code, commits, and documents Claude has produced inside the user's repos (already covered by source 1, but called out separately because of its disproportionate weight for the regular-task-space goal). These give the alignment a direct calibration to "what Claude tends to write," which is the population the regular task space most needs to handle well.

All sources are de-duplicated by content hash before encoding, then encoded once per side and stored as a Parquet/Arrow file of `(text_hash, domain, source, license, primary_vec, secondary_vec)`. Re-running alignment on a new model pair never requires re-collecting the text — only re-encoding. The pairs file itself is intended to be publishable (subject to per-row license filters), so that the alignment ships as a generic, helpful default that any veld install can use without running its own fit.

---

## Evaluation

Three complementary metrics, all computed on a held-out 10% split, stratified by domain:

1. **Paired cosine.** For each held-out pair, cosine between `project(secondary)` and `primary`. Target ≥ 0.85 mean per domain, ≥ 0.80 worst-domain.
2. **Cross-space retrieval recall@k.** For each held-out text `t`, encode `t` with the secondary, project to primary, query the primary Vamana index. Measure how often the correct memory ID is in the top-k (k = 1, 5, 10). Target: within 5 percentage points of the same-space (primary→primary) baseline.
3. **Task-space coherence.** Cluster Claude-produced artifacts in the aligned space (HDBSCAN or k-means with silhouette); compare cluster purity against a human-labelled "task type" axis. Target: silhouette ≥ 0.3 with task-type as ground truth. This is the metric that most directly answers "does the regular task space actually work?"

Evaluation lives under `evaluations/alignment/`, mirroring the existing `evaluations/` layout in the repo. The harness runs end-to-end in CI on a small fixed sample (≤ 500 pairs) so regressions are caught before a full re-fit.

---

## Veld integration

**File layout (additions only):**

```text
src/embeddings/
    alignment.rs              # trait, IdentityAlignment, persistence
    alignment_procrustes.rs   # default fitter
    alignment_ridge.rs        # fallback fitter
    alignment_mlp.rs          # gated fitter

evaluations/alignment/
    fit.rs                    # CLI: train an alignment from a pairs file
    eval.rs                   # CLI: evaluate against held-out
    pairs/                    # source-text corpora, gitignored beyond a README
    fitted/                   # persisted .alignment files

src/memory/retrieval.rs       # add aligned cross-search path, gated by Option<Alignment>
```

**Storage:** alignment is a *global*, install-wide artifact, not a per-tenant one. It is resolved at startup from the first path that exists:

1. `$VELD_ALIGNMENT_PATH` (operator override).
2. `~/.cache/veld/alignments/<primary_pair_id>__<secondary_pair_id>.bin` (auto-fitted or downloaded local cache).
3. Bundled default shipped with the release under `assets/alignments/` for known embedder pairs.

The per-storage Vamana directories remain untouched — `primary_vector_index/` and `secondary_vector_index/` are unchanged. The alignment file's header records `(primary_pair_id, secondary_pair_id, method, in_dim, out_dim, fit_date, eval_summary)`. Loading refuses to adopt an alignment whose pair IDs do not match the current embedders, exactly mirroring the existing dimension-guard pattern in `rebuild_secondary_from_rocksdb`.

**Retrieval changes:** add `search_aligned(query_text, limit)` that encodes the query once with the secondary, projects, and queries the primary Vamana index. The existing `search_ids_secondary` is unchanged; the existing max-score union is unchanged. The new path is additive — it gives callers a third option, not a replacement.

**Gating:** all aligned operations require `embedder.alignment().is_some()`. Without an alignment present, the existing behaviour is byte-identical to today. There is no behaviour change for installs that do not run the fit.

---

## The regular task space

The user's framing — "placing projects made by Claude into a regular task space" — is the most demanding of the three motivating goals and worth treating separately.

A regular task space is a single primary-space region where every Claude-produced artifact lands at a position that reflects *what the artifact is for*, not which model encoded it or what surface form it took. The same React component should land in roughly the same place whether it was described in prose, written as TSX, or scaffolded from a Storybook entry. The way you achieve this:

1. **Encode every artifact through both lenses.** Primary captures semantic meaning, secondary captures structural/syntactic signal that the primary may flatten. Both go through `encode_dual`.
2. **Project the secondary into primary space via the alignment.** Now both views live in one coordinate system.
3. **Fuse to a single canonical position.** Weighted average with a small learned weight `α ∈ [0, 1]` chosen per domain — e.g. for code-heavy domains the secondary may carry more signal, so `α` is closer to 0.5; for prose-heavy domains, closer to 0.2. The weight itself is a hyperparameter found by maximising cluster purity on a small labelled set during evaluation.
4. **Anchor with task-type prototypes.** Maintain a small set of "prototype" vectors — one per task type (React component, SQL migration, Linear ticket, ADR, runbook, eval, etc.) — computed as the centroid of a curated example set. New artifacts are placed *and* their nearest prototype is recorded as a metadata tag. This is what makes the space "regular": every point has a nearby named landmark.
5. **Reuse prototypes for retrieval queries.** When Claude needs to find related work for a new project, it queries the aligned space *and* compares against prototypes — the dual signal disambiguates between "similar in meaning" and "similar in task type."

The prototype set is the bridge from a continuous vector space to a discrete task ontology. It is small enough to maintain by hand (~30–50 prototypes covering the named domains and gaps), recomputable on demand from the curated example set, and version-pinned alongside the alignment.

---

## Risks and invariants

**Invariants the implementation must hold:**

- Raw secondary vectors never enter primary-space operations.
- Loading an alignment whose pair IDs do not match the current embedders is a hard error, never a silent fallback.
- The existing `Embedder` trait surface is unchanged. All new APIs are additive.
- The no-alignment install path is byte-identical to today.

**Risks and mitigations:**

- *Domain skew.* If the paired corpus is web-dev-heavy, alignment will be tighter on web-dev than on analytics. *Mitigation:* the 5K-per-domain quota is enforced at corpus-build time, not "best effort"; under-supplied domains are filled with synthetic pairs.
- *Embedder drift.* A new release of Nomic or MiniLM invalidates the alignment. *Mitigation:* pair IDs include the model version; the loader refuses mismatched alignments and prompts a re-fit. Re-fitting Procrustes on a stored pairs file is a single CLI invocation.
- *Asymmetric models.* Nomic's query prefix changes the secondary vector for queries vs. documents. *Mitigation:* fit two alignments — `alignment_doc.bin` and `alignment_query.bin` — and route through whichever side the call originated from.
- *Procrustes assumes L2-normalised inputs.* Both veld embedders normalise today, but a future embedder might not. *Mitigation:* `OrthogonalProcrustesAlignment::fit` asserts norms ∈ `[0.99, 1.01]` and refuses non-normalised inputs.
- *Catastrophic mismatch.* If the two embedders genuinely encode different things (e.g. one is text-only, one is multimodal), no alignment will be good. *Mitigation:* the evaluation gate (paired cosine ≥ 0.80 worst-domain) refuses to install a bad alignment. The fallback is unchanged max-score union — no regression.

---

## Phased plan

**Phase 1 — Scaffolding (1–2 days).** Add `Alignment` trait, `IdentityAlignment`, persistence format, dimension and pair-ID guards. No behaviour change for any caller. Unit tests cover save/load round-trips and guard failures.

**Phase 2 — Corpus assembly (2–3 days).** Build the 40K paired-text file. Pull from the user's own repos (veld, sleight) and the connected stores first; supplement with public sources; backfill with synthetic generation. De-duplicate by content hash; record `(text_hash, domain, source)` per row. Encode once with each embedder and write the pairs Parquet.

**Phase 3 — Procrustes fit and evaluation (1 day).** Implement `OrthogonalProcrustesAlignment::fit`, fit on the 90% split, evaluate on the 10%. If targets hit, persist and integrate.

**Phase 4 — Retrieval integration (1–2 days).** Add `search_aligned` path in `retrieval.rs`. Add fused-position writes for Claude artifacts (the regular task space). Wire prototype set and nearest-prototype tagging.

**Phase 5 — Fallbacks (1 day).** Implement ridge if Phase 3 misses a target; MLP only if ridge also misses. Likely unnecessary for Nomic↔MiniLM; held in reserve for future model swaps.

**Phase 6 — Embedder-onboarding hook (1 day).** A `cargo run --bin alignment-refit` CLI that re-encodes the pairs file, refits Procrustes, validates against the held-out set, and atomically swaps the alignment file. *Auto-triggered* on embedder onboarding: when veld starts and detects a new `(primary_pair_id, secondary_pair_id)` combination for which no alignment exists, it runs the fit against the bundled pairs file before the secondary index is allowed to serve cross-space queries. Failure to meet evaluation targets leaves the alignment slot empty and falls back to the existing max-score union — no behaviour regression. Manual invocation remains available for quarterly drift checks and for fitting against an updated pairs file.

Total elapsed work: roughly 6–10 focused days, of which Phase 2 (corpus assembly) is the single largest item and the one most worth front-loading.

---

## Resolved decisions

1. **Training data scope: code and public sources only.** MCP stores (Gmail, Drive, etc.) remain connected and connectable for runtime memory operations but are excluded from the alignment training corpus. Gmail in particular is unlikely to add useful signal for the task-space goal — domain-relevant code, docs, and tickets dominate. The alignment is therefore publishable as a generic, helpful default that any veld install can adopt without running its own fit. Where domain coverage is thin, the project solicits friends-contributed corpora rather than reaching into private data.
2. **One global alignment, auto-refit on new embedders.** veld ships and maintains a single global alignment artifact, fitted from the published pairs file. When a new embedder model is brought online (either side of the pair), the embedder-onboarding hook in Phase 6 detects the unknown `pair_id` combination, runs the fit, validates against held-out targets, and installs the new alignment atomically. Per-user alignments are explicitly out of scope until concrete evidence of per-user drift emerges.
3. **Prototype set governance: hand-curated, file-per-prototype, regenerable.** Prototypes live under `evaluations/alignment/prototypes/` as one JSON file per prototype with `name`, `description`, `example_ids`, and a recomputed `centroid_vec` field. They are edited by hand and recomputed from `example_ids` on demand. Versioning piggy-backs on git.
4. **MCP surface: experimental feature branch with documentation.** Tools such as `recall_aligned` and `find_related_projects` will be added to the MCP server, but exclusively on a documented experimental feature branch — gated behind a feature flag, marked `[experimental]` in the tool description, and explicitly tagged in release notes. Promotion from experimental to stable requires the retrieval evaluation to show a clean win over the existing `recall` for two consecutive quarterly evaluations against a frozen test set. Branch name: `feat/alignment-mcp-experimental`.
