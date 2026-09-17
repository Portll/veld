//! Single-component path guards for identifiers that come from callers.
//!
//! Every per-user directory in this codebase is `base.join(user_id)` where `user_id` originates
//! outside the process (HTTP request, CLI argument, MCP call). A value like `../other-user` or an
//! absolute path escapes the base directory at every one of those joins. This module is the one
//! chokepoint: validate the identifier as a SINGLE, plain path component before it ever reaches a
//! `join`, and reject rather than normalize — silently rewriting `../x` into something safe would
//! hide the attempt and change behavior invisibly.

/// Accepts a non-empty string of at most 128 bytes drawn from `[A-Za-z0-9._@-]`, that is not `.`
/// or `..` and does not START with a dot (a leading dot names hidden files and the special
/// entries). Returns the same slice on success so call sites stay borrow-friendly.
pub fn sanitize_component<'a>(value: &'a str, what: &str) -> anyhow::Result<&'a str> {
    // Deliberately a delegation, not a second vocabulary: two validators drift. The API layer
    // (`validation::validate_user_id`) accepted unicode alphanumerics that this guard refused,
    // so a tenant could be created whose directory could never be — and this guard did not
    // reject `a..b` or a trailing dot, which the API layer did. One rule, checked at both the
    // request and the filesystem. Ported from shodh-memory b0ba1c5 (#518), the upstream port
    // of veld's own 01bcf9b.
    crate::validation::validate_user_id(value)
        .map_err(|e| anyhow::anyhow!("invalid {what}: {e}"))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::sanitize_component;

    /// Everything the API layer accepts, the filesystem guard accepts — including the
    /// unicode alphanumerics the old inline charset refused.
    #[test]
    fn accepts_everything_the_api_layer_accepts() {
        for ok in [
            "all",
            "user-1",
            "a.b",
            "X_9",
            "someone@example.com",
            "名前",
            "müller",
        ] {
            assert!(sanitize_component(ok, "user id").is_ok(), "{ok}");
            assert!(
                crate::validation::validate_user_id(ok).is_ok(),
                "{ok}: the API layer disagrees"
            );
        }
    }

    #[test]
    fn rejects_everything_that_could_leave_the_directory() {
        for bad in [
            "",
            ".",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "/etc",
            "a\0b",
            ".hidden",
            "..%2f",
            "a b",
            "a..b",
            "trailing.",
            &"x".repeat(129),
        ] {
            assert!(sanitize_component(bad, "user id").is_err(), "{bad:?}");
        }
    }
}
