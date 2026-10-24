<!-- GENERATED FILE — do not edit by hand.
     Source: src/**/*.rs (env::var calls)
     Generator: docs/generators/src/bin/gen-config-ref.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-config-ref`) -->

# Configuration Reference

Veld is configured via environment variables. The generator scanned `src/**/*.rs` and found **122** distinct `VELD_*` variables.

| Variable | Default | First seen in |
|---|---|---|
| `VELD_ACTIVATION_DECAY` | — | `src/config.rs` |
| `VELD_AGENT_ID` | — | `src/memory/session_detect.rs` |
| `VELD_ALIGNMENT_AUTOFIT` | — | `src/memory/alignment_onboarding.rs` |
| `VELD_ALIGNMENT_PAIRS` | — | `src/memory/alignment_onboarding.rs` |
| `VELD_ALIGNMENT_PATH` | — | `src/embeddings/alignment.rs` |
| `VELD_ALIGN_WHITEN` | — | `src/embeddings/alignment_config.rs` |
| `VELD_ALLOW_INSECURE_REMOTE_EMBEDDER` | — | `src/embeddings/http_embedder.rs` |
| `VELD_ALLOW_KEYSTORE_ROLLBACK` | — | `src/memory/storage.rs` |
| `VELD_ALLOW_UNSIGNED_WEBHOOKS` | — | `src/handlers/integrations.rs` |
| `VELD_API_KEY` | — | `src/auth.rs` |
| `VELD_API_KEYS` | — | `src/auth.rs` |
| `VELD_AUDIT_MAX_ENTRIES` | — | `src/config.rs` |
| `VELD_AUDIT_RETENTION_DAYS` | — | `src/config.rs` |
| `VELD_AUTO_COMPRESS` | — | `src/config.rs` |
| `VELD_BACKUP_ENABLED` | — | `src/config.rs` |
| `VELD_BACKUP_INTERVAL` | — | `src/config.rs` |
| `VELD_BACKUP_MAX_COUNT` | — | `src/config.rs` |
| `VELD_COLLECTIVE_STORE_DIR` | — | `src/config.rs` |
| `VELD_CORS_CREDENTIALS` | — | `src/config.rs` |
| `VELD_CORS_HEADERS` | — | `src/config.rs` |
| `VELD_CORS_MAX_AGE` | — | `src/config.rs` |
| `VELD_CORS_METHODS` | — | `src/config.rs` |
| `VELD_CORS_ORIGINS` | — | `src/config.rs` |
| `VELD_CORS_WARN` | — | `src/config.rs` |
| `VELD_CROSS_ENCODER` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_CROSS_ENCODERS` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_DERIVED_TAG_BUDGET` | — | `src/config.rs` |
| `VELD_DEV_API_KEY` | — | `src/auth.rs` |
| `VELD_DIALS_DISABLE_SIGNALS` | — | `src/memory/dials.rs` |
| `VELD_DOCS_DIR` | — | `src/handlers/docs.rs` |
| `VELD_EMBEDDER` | — | `src/memory/mod.rs` |
| `VELD_EMBEDDING_API_KEY` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_MODEL` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_TASK_PREFIXES` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_TIMEOUT_MS` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_URL` | — | `src/memory/mod.rs` |
| `VELD_EMBED_TIMEOUT_MS` | — | `src/embeddings/minilm.rs` |
| `VELD_ENFORCE_HTTPS` | — | `src/integrations/mod.rs` |
| `VELD_ENV` | — | `src/config.rs` |
| `VELD_EVAL_NOW` | — | `src/memory/ablation.rs` |
| `VELD_GAP_TIMEOUT` | — | `src/config.rs` |
| `VELD_GAP_WORKER_CHUNK_ROWS` | — | `src/config.rs` |
| `VELD_GAP_WORKER_CPU_PCT` | — | `src/config.rs` |
| `VELD_GAP_WORKER_NICE` | — | `src/config.rs` |
| `VELD_GRAPH_MIRROR_SYNC_SECS` | — | `src/config.rs` |
| `VELD_HIDE_DEV_KEY` | — | `src/auth.rs` |
| `VELD_HOST` | — | `src/config.rs` |
| `VELD_INSTANCE_LABEL` | — | `src/instance_registry.rs` |
| `VELD_INTENT_LOG_SEGMENT_MB` | — | `src/config.rs` |
| `VELD_KEYSTORE_PATH` | — | `src/keystore.rs` |
| `VELD_KMS_WRAP_KEY` | — | `src/keystore.rs` |
| `VELD_LAZY_CROSS_ENCODER` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_LAZY_LOAD` | — | `src/embeddings/minilm.rs` |
| `VELD_LIVE_EMBEDDER` | — | `src/memory/mod.rs` |
| `VELD_LLM_API_KEY` | — | `src/memory/llm.rs` |
| `VELD_LLM_ENDPOINT` | — | `src/memory/llm.rs` |
| `VELD_LLM_HOST` | `_\| decl.default.clone(` | `src/query_parsing/llm_parser.rs` |
| `VELD_LLM_MODEL` | `_\|` | `src/memory/llm.rs` |
| `VELD_LLM_REFINER_API_KEY` | — | `src/memory/llm_refiner.rs` |
| `VELD_LLM_REFINER_ENDPOINT` | — | `src/memory/llm_refiner.rs` |
| `VELD_LLM_REFINER_MODEL` | — | `src/memory/llm_refiner.rs` |
| `VELD_LLM_URL` | `_\| {` | `src/query_parsing/llm_parser.rs` |
| `VELD_LOG_PERIODIC_SCALES` | — | `src/config.rs` |
| `VELD_MAINTENANCE_INTERVAL` | — | `src/config.rs` |
| `VELD_MASTER_PASSPHRASE` | — | `src/memory/storage.rs` |
| `VELD_MAX_CONCURRENT` | — | `src/config.rs` |
| `VELD_MAX_ENTITIES` | — | `src/config.rs` |
| `VELD_MAX_USERS` | — | `src/config.rs` |
| `VELD_MEMORY_PATH` | — | `src/config.rs` |
| `VELD_METRICS_PUBLIC` | — | `src/config.rs` |
| `VELD_MODEL_PATH` | — | `src/embeddings/minilm.rs` |
| `VELD_MSSQL_ADO_STRING` | — | `src/config.rs` |
| `VELD_MULTI_TENANT` | — | `src/config.rs` |
| `VELD_NER_CONFIDENCE` | — | `src/embeddings/ner.rs` |
| `VELD_NER_MODEL_PATH` | — | `src/embeddings/ner.rs` |
| `VELD_NOMIC_DIM` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_EMBED_TIMEOUT_MS` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_MODEL_PATH` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_POOL_SIZE` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_USE_QUANTIZED` | — | `src/embeddings/nomic.rs` |
| `VELD_OFFLINE` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_ONNX_THREADS` | — | `src/embeddings/minilm.rs` |
| `VELD_PACKAGE_DIR` | — | `src/embeddings/minilm.rs` |
| `VELD_PORT` | — | `src/config.rs` |
| `VELD_POSTGRES_URL` | — | `src/config.rs` |
| `VELD_PROBE_IDS` | — | `src/memory/storage.rs` |
| `VELD_PROBE_STORE` | — | `src/memory/storage.rs` |
| `VELD_PUBLIC_RATE_LIMIT` | — | `src/config.rs` |
| `VELD_QUERY_NER` | — | `src/handlers/state.rs` |
| `VELD_RATE_BURST` | — | `src/config.rs` |
| `VELD_RATE_LIMIT` | — | `src/config.rs` |
| `VELD_RECALL_READONLY` | — | `src/recall_harness/runner.rs` |
| `VELD_RELATIONAL_BACKEND` | — | `src/config.rs` |
| `VELD_RELATIONAL_SQLITE_PATH` | — | `src/config.rs` |
| `VELD_REQUEST_TIMEOUT` | — | `src/config.rs` |
| `VELD_RERANK_DEPTH` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_SLEEP_TIME_PROFILE` | — | `src/server.rs` |
| `VELD_STORAGE_BACKEND` | — | `src/config.rs` |
| `VELD_SUPABASE_DB_PASSWORD` | — | `src/config.rs` |
| `VELD_SUPABASE_PROJECT_REF` | — | `src/config.rs` |
| `VELD_TLS_ACK` | — | `src/server.rs` |
| `VELD_USE_QUANTIZED_MODEL` | — | `src/embeddings/minilm.rs` |
| `VELD_WIKI_DIR` | — | `src/wiki/mod.rs` |
| `VELD_WIKI_RENDER` | — | `src/wiki/mod.rs` |
| `VELD_WIKI_WATCH` | — | `src/wiki/watcher.rs` |
| `VELD_WRITE_BASELINE` | — | `src/recall_harness/runner.rs` |
| `VELD_WRITE_LATCH_COOLDOWN_SECS` | — | `src/storage/write_latch.rs` |
| `VELD_WRITE_LATCH_MAX_ATTEMPTS` | — | `src/storage/write_latch.rs` |
| `VELD_WRITE_LATCH_MIN_FREE_BYTES` | — | `src/storage/write_latch.rs` |
| `VELD_WRITE_LATCH_RESTART` | — | `src/storage/write_latch.rs` |
| `VELD_WRITE_MODE` | — | `src/memory/storage.rs` |
| `VELD_ZENOH_API_KEY` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_AUTO_TOPICS` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_CONNECT` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_DIMENSION` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_ENABLED` | — | `src/memory/mod.rs` |
| `VELD_ZENOH_EMBED_MODEL` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_PREFIX` | `_\|` | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_TIMEOUT_MS` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_ENABLED` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_LISTEN` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_MODE` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_PREFIX` | `_\|` | `src/zenoh_transport/config.rs` |

---

*Defaults shown above are best-effort extractions from `.unwrap_or(...)` chains. For full semantics, consult the source file listed.*
