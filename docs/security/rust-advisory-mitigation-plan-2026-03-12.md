# Rust Advisory Mitigation Plan (2026-03-12)

## Scope
This plan covers active `cargo audit` findings in `shodh-memory` after dependency updates:
- `RUSTSEC-2026-0002` (`lru 0.12.5`, unsound)
- `RUSTSEC-2025-0141` (`bincode 1.3.3` and `bincode 2.0.1`, unmaintained)
- `RUSTSEC-2024-0436` (`paste 1.0.15`, unmaintained)

## Current State (verified)
- Direct `lru` dependency is upgraded to `0.16.3` in `Cargo.toml`.
- `lru 0.12.5` remains only as a transitive dependency through `tantivy 0.25.0`.
- `bincode` is used in two forms:
  - `bincode` v2 for current storage format.
  - `bincode1` alias for legacy decode fallback and migrations.
- Legacy decode fallbacks are concentrated in `src/memory/storage.rs` and include multiple `bincode1::deserialize` paths.

## Decision
`bincode` cannot be removed safely in one patch without risking inability to read legacy persisted memory records. Migration must be staged.

## Phased Plan

### Phase 1: Safety Gates (short-term)
1. Add a dedicated integration test fixture set for legacy encoded payloads currently accepted by `deserialize_with_fallback`.
2. Record baseline migration behavior:
   - Which fallback branch decoded each fixture.
   - Whether output rewrites to current format successfully.
3. Add a metric counter for fallback usage by branch label in production paths.

Exit criteria:
- Legacy fixture tests pass reliably.
- Fallback branch usage can be observed in logs/metrics.

### Phase 2: Write-Path Isolation (short-term)
1. Ensure all writes continue using current `bincode` v2 or MessagePack path only.
2. Explicitly ban new writes via `bincode1` by lint/search checks in CI.

Exit criteria:
- No code path writes with `bincode1`.
- Existing write tests still pass.

### Phase 3: Reader Migration (medium-term)
1. Introduce an alternate codec abstraction behind `StorageCodec` for current-format reads/writes.
2. Keep `bincode1` only in a quarantined `legacy_decode` module.
3. Rewrite data to canonical current format immediately after successful legacy decode (already partially present), and test this end-to-end.

Exit criteria:
- Legacy support is isolated to one module.
- New codec can be swapped without touching memory business logic.

### Phase 4: Legacy Sunset (medium/long-term)
1. After observed fallback usage reaches near-zero for a defined window, gate legacy decode behind feature flag.
2. Disable by default in new deployments.
3. Remove `bincode1` dependency in a major version release.

Exit criteria:
- No active fallback usage in telemetry window.
- Removal PR passes full regression and migration tests.

## Tantivy/LRU Advisory Track
- `tantivy 0.25.0` currently pulls `lru 0.12.5`.
- Upstream crate index currently lists `tantivy 0.25.0` as latest.
- Practical options:
  1. Track upstream release for patched `lru` adoption.
  2. Evaluate maintained fork only if security policy requires immediate closure.
  3. Temporarily accept transitive risk with documented rationale and periodic recheck.

## Immediate Follow-ups
1. Add legacy fixture integration tests for `deserialize_with_fallback`.
2. Add fallback-branch metrics in `src/memory/storage.rs`.
3. Open an issue to monitor `tantivy`/`lru` transitive advisory closure.
