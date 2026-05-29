<!-- GENERATED FILE — do not edit by hand.
     Source: src/**/*.rs (env::var calls)
     Generator: docs/generators/src/bin/gen-config-ref.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-config-ref`) -->

# Configuration Reference

Veld is configured via environment variables. The generator scanned `src/**/*.rs` and found **79** distinct `VELD_*` variables.

| Variable | Default | First seen in |
|---|---|---|
| `VELD_ACTIVATION_DECAY` | — | `src/config.rs` |
| `VELD_ADMIN_KEY` | — | `src/handlers/users.rs` |
| `VELD_ALIGNMENT_AUTOFIT` | — | `src/memory/alignment_onboarding.rs` |
| `VELD_ALIGNMENT_PAIRS` | — | `src/memory/alignment_onboarding.rs` |
| `VELD_ALIGNMENT_PATH` | — | `src/embeddings/alignment.rs` |
| `VELD_ALLOW_UNSIGNED_WEBHOOKS` | — | `src/handlers/integrations.rs` |
| `VELD_API_KEY` | — | `src/auth.rs` |
| `VELD_API_KEYS` | — | `src/auth.rs` |
| `VELD_AUDIT_MAX_ENTRIES` | — | `src/config.rs` |
| `VELD_AUDIT_RETENTION_DAYS` | — | `src/config.rs` |
| `VELD_BACKUP_ENABLED` | — | `src/config.rs` |
| `VELD_BACKUP_INTERVAL` | — | `src/config.rs` |
| `VELD_BACKUP_MAX_COUNT` | — | `src/config.rs` |
| `VELD_COLLECTIVE_STORE_DIR` | — | `src/auth.rs` |
| `VELD_CORS_CREDENTIALS` | — | `src/config.rs` |
| `VELD_CORS_HEADERS` | — | `src/config.rs` |
| `VELD_CORS_MAX_AGE` | — | `src/config.rs` |
| `VELD_CORS_METHODS` | — | `src/config.rs` |
| `VELD_CORS_ORIGINS` | — | `src/config.rs` |
| `VELD_CORS_WARN` | — | `src/config.rs` |
| `VELD_DEV_API_KEY` | — | `src/auth.rs` |
| `VELD_EMBEDDING_API_KEY` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_MODEL` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_TIMEOUT_MS` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBEDDING_API_URL` | — | `src/embeddings/http_embedder.rs` |
| `VELD_EMBED_TIMEOUT_MS` | — | `src/embeddings/minilm.rs` |
| `VELD_ENCRYPTION_KEY` | — | `src/encryption.rs` |
| `VELD_ENFORCE_HTTPS` | — | `src/integrations/mod.rs` |
| `VELD_ENV` | — | `src/auth.rs` |
| `VELD_HIDE_DEV_KEY` | — | `src/auth.rs` |
| `VELD_HOST` | — | `src/config.rs` |
| `VELD_LAZY_LOAD` | — | `src/embeddings/minilm.rs` |
| `VELD_LLM_API_KEY` | — | `src/memory/llm.rs` |
| `VELD_LLM_API_TYPE` | — | `src/query_parsing/llm_parser.rs` |
| `VELD_LLM_ENDPOINT` | — | `src/memory/llm.rs` |
| `VELD_LLM_MODEL` | — | `src/memory/llm.rs` |
| `VELD_LLM_URL` | — | `src/query_parsing/llm_parser.rs` |
| `VELD_LOG_PERIODIC_SCALES` | — | `src/config.rs` |
| `VELD_MAINTENANCE_INTERVAL` | — | `src/config.rs` |
| `VELD_MAX_CONCURRENT` | — | `src/config.rs` |
| `VELD_MAX_ENTITIES` | — | `src/config.rs` |
| `VELD_MAX_USERS` | — | `src/config.rs` |
| `VELD_MEMORY_PATH` | — | `src/config.rs` |
| `VELD_METRICS_PUBLIC` | — | `src/config.rs` |
| `VELD_MODEL_PATH` | — | `src/embeddings/minilm.rs` |
| `VELD_MULTI_TENANT` | — | `src/auth.rs` |
| `VELD_NER_CONFIDENCE` | — | `src/embeddings/ner.rs` |
| `VELD_NER_MODEL_PATH` | — | `src/embeddings/ner.rs` |
| `VELD_NOMIC_DIM` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_EMBED_TIMEOUT_MS` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_MODEL_PATH` | — | `src/embeddings/nomic.rs` |
| `VELD_NOMIC_USE_QUANTIZED` | — | `src/embeddings/nomic.rs` |
| `VELD_OFFLINE` | — | `src/embeddings/cross_encoder.rs` |
| `VELD_ONNX_THREADS` | — | `src/embeddings/minilm.rs` |
| `VELD_PACKAGE_DIR` | — | `src/embeddings/minilm.rs` |
| `VELD_PORT` | — | `src/cli.rs` |
| `VELD_PUBLIC_RATE_LIMIT` | — | `src/config.rs` |
| `VELD_RATE_BURST` | — | `src/config.rs` |
| `VELD_RATE_LIMIT` | — | `src/config.rs` |
| `VELD_REQUEST_TIMEOUT` | — | `src/config.rs` |
| `VELD_RLM_API_KEY` | — | `src/memory/llm_reranker.rs` |
| `VELD_RLM_ENDPOINT` | — | `src/bin/rlm_eval.rs` |
| `VELD_RLM_MODEL` | — | `src/bin/rlm_eval.rs` |
| `VELD_STORAGE_BACKEND` | — | `src/config.rs` |
| `VELD_TLS_ACK` | — | `src/server.rs` |
| `VELD_USE_QUANTIZED_MODEL` | — | `src/embeddings/minilm.rs` |
| `VELD_WRITE_MODE` | — | `src/memory/storage.rs` |
| `VELD_ZENOH_API_KEY` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_AUTO_TOPICS` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_CONNECT` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_DIMENSION` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_ENABLED` | — | `src/memory/mod.rs` |
| `VELD_ZENOH_EMBED_MODEL` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_PREFIX` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_EMBED_TIMEOUT_MS` | — | `src/embeddings/zenoh_embedder.rs` |
| `VELD_ZENOH_ENABLED` | — | `src/server.rs` |
| `VELD_ZENOH_LISTEN` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_MODE` | — | `src/zenoh_transport/config.rs` |
| `VELD_ZENOH_PREFIX` | `_\|` | `src/server.rs` |

---

*Defaults shown above are best-effort extractions from `.unwrap_or(...)` chains. For full semantics, consult the source file listed.*
