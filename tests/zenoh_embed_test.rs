//! Integration tests for the Zenoh embedding layer.
//!
//! These tests exercise pure functions only — no live Zenoh mesh required.
//! Gate all test code behind `#[cfg(feature = "zenoh")]` so they compile
//! cleanly on configurations that omit the `zenoh` feature.
//!
//! Run with:
//!   cargo test --test zenoh_embed_test --features zenoh

#[cfg(feature = "zenoh")]
mod zenoh_embed_tests {
    use shodh_memory::embeddings::zenoh_embedder::{validate_embeddings, ZenohEmbedderConfig};

    // -------------------------------------------------------------------------
    // ZenohEmbedderConfig::from_env defaults
    // -------------------------------------------------------------------------

    /// When no relevant env vars are set, `from_env` must return the documented
    /// defaults so that a zero-config deployment works out-of-the-box.
    #[test]
    fn test_config_from_env_defaults() {
        // Remove variables that would shadow defaults.
        std::env::remove_var("SHODH_ZENOH_EMBED_PREFIX");
        std::env::remove_var("SHODH_ZENOH_EMBED_MODEL");
        std::env::remove_var("SHODH_ZENOH_EMBED_DIMENSION");
        std::env::remove_var("SHODH_ZENOH_EMBED_TIMEOUT_MS");
        std::env::remove_var("SHODH_ZENOH_CONNECT");
        std::env::remove_var("SHODH_ZENOH_API_KEY");

        let config = ZenohEmbedderConfig::from_env();
        assert_eq!(config.prefix, "shodh");
        assert_eq!(config.target_model, "minilm-l6-v2");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.connect_endpoints.is_empty());
        assert!(config.api_key.is_none());
    }

    /// Env vars must override every default.
    #[test]
    fn test_config_from_env_overrides() {
        std::env::set_var("SHODH_ZENOH_EMBED_PREFIX", "mybot");
        std::env::set_var("SHODH_ZENOH_EMBED_MODEL", "bge-small-en");
        std::env::set_var("SHODH_ZENOH_EMBED_DIMENSION", "768");
        std::env::set_var("SHODH_ZENOH_EMBED_TIMEOUT_MS", "2000");
        std::env::set_var("SHODH_ZENOH_CONNECT", "tcp/10.0.0.1:7447,tcp/10.0.0.2:7447");
        std::env::set_var("SHODH_ZENOH_API_KEY", "secret");

        let config = ZenohEmbedderConfig::from_env();

        std::env::remove_var("SHODH_ZENOH_EMBED_PREFIX");
        std::env::remove_var("SHODH_ZENOH_EMBED_MODEL");
        std::env::remove_var("SHODH_ZENOH_EMBED_DIMENSION");
        std::env::remove_var("SHODH_ZENOH_EMBED_TIMEOUT_MS");
        std::env::remove_var("SHODH_ZENOH_CONNECT");
        std::env::remove_var("SHODH_ZENOH_API_KEY");

        assert_eq!(config.prefix, "mybot");
        assert_eq!(config.target_model, "bge-small-en");
        assert_eq!(config.dimension, 768);
        assert_eq!(config.timeout_ms, 2000);
        assert_eq!(
            config.connect_endpoints,
            vec!["tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447"]
        );
        assert_eq!(config.api_key.as_deref(), Some("secret"));
    }

    // -------------------------------------------------------------------------
    // validate_embeddings
    // -------------------------------------------------------------------------

    /// All embeddings match expected dimension — must succeed.
    #[test]
    fn test_validate_embeddings_all_match() {
        let embeddings = vec![vec![0.0_f32; 384], vec![1.0_f32; 384]];
        assert!(validate_embeddings(&embeddings, 384).is_ok());
    }

    /// Empty slice — trivially valid (no embeddings to reject).
    #[test]
    fn test_validate_embeddings_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        assert!(validate_embeddings(&embeddings, 384).is_ok());
    }

    /// First embedding has the wrong dimension — must return an error.
    #[test]
    fn test_validate_embeddings_first_wrong() {
        let embeddings = vec![vec![0.0_f32; 256], vec![0.0_f32; 384]];
        let err = validate_embeddings(&embeddings, 384).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("256"),
            "error should mention the actual dim: {msg}"
        );
        assert!(
            msg.contains("index 0"),
            "error should mention index 0: {msg}"
        );
        assert!(
            msg.contains("384"),
            "error should mention the expected dim: {msg}"
        );
    }

    /// Second embedding is wrong; first is correct — error must name index 1.
    #[test]
    fn test_validate_embeddings_second_wrong() {
        let embeddings = vec![vec![0.0_f32; 384], vec![0.0_f32; 128]];
        let err = validate_embeddings(&embeddings, 384).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("index 1"),
            "error should mention index 1: {msg}"
        );
    }

    /// Single embedding with a dimension mismatch.
    #[test]
    fn test_validate_embeddings_single_wrong() {
        let embeddings = vec![vec![0.0_f32; 512]];
        assert!(validate_embeddings(&embeddings, 384).is_err());
    }

    /// Dimension of zero — edge case; a non-empty embedding always fails.
    #[test]
    fn test_validate_embeddings_zero_expected_dim() {
        let embeddings = vec![vec![1.0_f32; 1]];
        assert!(validate_embeddings(&embeddings, 0).is_err());
    }

    /// Expected dimension of zero with an actually-empty embedding succeeds.
    #[test]
    fn test_validate_embeddings_zero_dim_empty_vec() {
        let embeddings = vec![vec![]];
        assert!(validate_embeddings(&embeddings, 0).is_ok());
    }
}
