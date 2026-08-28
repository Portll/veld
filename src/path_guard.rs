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
    let ok_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '@' | '-');
    if value.is_empty()
        || value.len() > 128
        || value.starts_with('.')
        || !value.chars().all(ok_char)
    {
        anyhow::bail!(
            "invalid {what}: must be a single plain path component ([A-Za-z0-9._@-], not starting with '.', <=128 bytes)"
        );
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::sanitize_component;

    #[test]
    fn accepts_ordinary_identifiers() {
        for ok in ["all", "user-1", "a.b", "X_9", "someone@example.com"] {
            assert!(sanitize_component(ok, "user id").is_ok(), "{ok}");
        }
    }

    #[test]
    fn rejects_everything_that_could_leave_the_directory() {
        for bad in [
            "", ".", "..", "../x", "a/b", "a\\b", "/etc", "a\0b", ".hidden",
            "..%2f", "a b", "名前", &"x".repeat(129),
        ] {
            assert!(sanitize_component(bad, "user id").is_err(), "{bad:?}");
        }
    }
}
