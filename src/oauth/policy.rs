//! OAuth security-policy gate.
//!
//! Currently a single predicate: is this process running in a
//! production-like environment? Used by `TokenStore::new` to hard-refuse
//! startup when `VELD_ENCRYPTION_KEY` is unset in production — the
//! `FieldEncryptor` dev-fallback (plaintext-with-warning) is acceptable
//! for unit tests but never for tokens.

/// True iff `VELD_ENV` names a production-class environment. The
/// predicate is liberal on the "production-strict" side so that an
/// unconfigured environment is the safe default for an unrecognised
/// value (the alternative is silently accepting plaintext fallback).
///
/// | `VELD_ENV` value           | Result        |
/// |----------------------------|---------------|
/// | unset / empty / `dev` / `test` / `ci` | `false` |
/// | `production` / `prod` / `staging`     | `true`  |
/// | anything else              | `true` (with warn-log) |
pub fn is_production() -> bool {
    match std::env::var("VELD_ENV").as_deref() {
        Ok("production") | Ok("prod") | Ok("staging") => true,
        Ok("dev") | Ok("test") | Ok("ci") | Ok("") | Err(_) => false,
        Ok(other) => {
            tracing::warn!(
                env = other,
                "Unknown VELD_ENV value — treating as production for safety"
            );
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;

    /// Tests in this module mutate `VELD_ENV` — serialize them so they
    /// don't race each other.
    static GUARD: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(value: Option<&str>, f: F) {
        let _g = GUARD.lock().unwrap();
        let saved = env::var("VELD_ENV").ok();
        match value {
            Some(v) => env::set_var("VELD_ENV", v),
            None => env::remove_var("VELD_ENV"),
        }
        f();
        match saved {
            Some(v) => env::set_var("VELD_ENV", v),
            None => env::remove_var("VELD_ENV"),
        }
    }

    #[test]
    fn unset_is_not_production() {
        with_env(None, || assert!(!is_production()));
    }

    #[test]
    fn dev_values_are_not_production() {
        for v in ["dev", "test", "ci", ""] {
            with_env(Some(v), || {
                assert!(!is_production(), "VELD_ENV={v} should be non-production")
            });
        }
    }

    #[test]
    fn known_prod_values_are_production() {
        for v in ["production", "prod", "staging"] {
            with_env(Some(v), || {
                assert!(is_production(), "VELD_ENV={v} should be production")
            });
        }
    }

    #[test]
    fn unknown_is_production() {
        with_env(Some("smoke"), || assert!(is_production()));
    }
}
