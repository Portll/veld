//! External integrations for syncing data sources to Veld memory
//!
//! Supports:
//! - Linear: Issue tracking webhooks and bulk sync
//! - GitHub: PR/Issue webhooks and bulk sync

pub mod github;
pub mod linear;

pub use github::{GitHubSyncRequest, GitHubSyncResponse, GitHubWebhook, GitHubWebhookPayload};
pub use linear::{LinearSyncRequest, LinearSyncResponse, LinearWebhook, LinearWebhookPayload};

/// Returns `true` when `url` should be flagged as using an insecure transport
/// (HTTP over a non-localhost host).
///
/// Limitation: userinfo URLs (`user:pass@host`) are handled by splitting on the
/// first `@` and treating the suffix as the authority. A URL like
/// `http://127.0.0.1@remote.example.com/` would be (correctly) flagged insecure
/// because `remote.example.com` is the actual host. An unusual percent-encoded
/// `@` inside the userinfo section could cause mis-extraction, but the function
/// errs toward "insecure" in ambiguous cases — safe.
pub fn is_insecure_remote_url(url: &str) -> bool {
    if !url.starts_with("http://") {
        return false;
    }
    let rest = &url["http://".len()..];
    let authority = rest.split('/').next().unwrap_or(rest);
    // Strip userinfo — take the part AFTER the first '@' as the true host+port.
    let host_and_port = authority
        .split_once('@')
        .map_or(authority, |(_, after)| after);
    let host = if host_and_port.starts_with('[') {
        // IPv6 bracket notation: `[::1]` or `[::1]:port`
        host_and_port
            .trim_start_matches('[')
            .split(']')
            .next()
            .unwrap_or(host_and_port)
    } else {
        host_and_port
            .rsplit_once(':')
            .map_or(host_and_port, |(h, _)| h)
    };
    !matches!(host, "127.0.0.1" | "localhost" | "::1" | "0.0.0.0")
}

/// Resolve an operator-supplied API URL override for an integration.
///
/// Reads `env_var`; if unset or empty, returns `default`. An `http://` override
/// for a non-localhost host would send the integration's token in cleartext, so
/// it is rejected and `default` is used, unless `VELD_ENFORCE_HTTPS` is `false`
/// or `0`. Any other value enforces, so a typo fails safe. Localhost `http://`
/// (proxies, test servers) is always allowed.
pub fn resolve_api_url_override(env_var: &str, default: &str) -> String {
    let candidate = std::env::var(env_var).unwrap_or_default();
    if candidate.is_empty() {
        return default.to_string();
    }
    if is_insecure_remote_url(&candidate) {
        let enforce = std::env::var("VELD_ENFORCE_HTTPS")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("false") || v == "0")
            })
            .unwrap_or(true);
        if enforce {
            tracing::error!(
                "{env_var} override rejected: insecure HTTP for a non-localhost host; \
                 using the default ({default}). Set VELD_ENFORCE_HTTPS=false to allow it."
            );
            return default.to_string();
        }
        tracing::warn!(
            "{env_var} override uses insecure HTTP for a non-localhost host \
             (VELD_ENFORCE_HTTPS=false); the token travels in cleartext"
        );
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_insecure_remote_url() {
        assert!(is_insecure_remote_url("http://remote.example.com/api"));
        assert!(is_insecure_remote_url("http://remote.example.com:8080/"));
        // userinfo with remote host → still insecure
        assert!(is_insecure_remote_url(
            "http://user:pass@remote.example.com/"
        ));
        // localhost variants → not insecure
        assert!(!is_insecure_remote_url("http://localhost:3030"));
        assert!(!is_insecure_remote_url("http://127.0.0.1:3030"));
        assert!(!is_insecure_remote_url("http://0.0.0.0:3030"));
        // https → never flagged (not our concern)
        assert!(!is_insecure_remote_url("https://remote.example.com/api"));
        // non-http schemes → false
        assert!(!is_insecure_remote_url("ftp://remote.example.com/"));
    }
}

/// `VELD_ENFORCE_HTTPS` and the URL overrides are process-wide; every test that
/// sets them holds this lock.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn with_override_env<T>(
    enforce: Option<&str>,
    vars: &[(&str, &str)],
    f: impl FnOnce() -> T,
) -> T {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match enforce {
        Some(v) => std::env::set_var("VELD_ENFORCE_HTTPS", v),
        None => std::env::remove_var("VELD_ENFORCE_HTTPS"),
    }
    for (name, value) in vars {
        std::env::set_var(name, value);
    }
    let out = f();
    std::env::remove_var("VELD_ENFORCE_HTTPS");
    for (name, _) in vars {
        std::env::remove_var(name);
    }
    out
}

#[cfg(test)]
mod https_default_tests {
    use super::{resolve_api_url_override, with_override_env};

    const DEFAULT_URL: &str = "https://api.example.com/v1";
    const VAR: &str = "VELD_TEST_INTEGRATION_URL";

    fn resolve(enforce: Option<&str>, url: &str) -> String {
        with_override_env(enforce, &[(VAR, url)], || {
            resolve_api_url_override(VAR, DEFAULT_URL)
        })
    }

    #[test]
    fn insecure_remote_override_is_rejected_by_default() {
        assert_eq!(resolve(None, "http://evil.example.com/v1"), DEFAULT_URL);
    }

    #[test]
    fn explicit_opt_out_allows_the_insecure_override() {
        for v in ["false", "0", "FALSE", " false "] {
            assert_eq!(
                resolve(Some(v), "http://legacy.internal.example.com/v1"),
                "http://legacy.internal.example.com/v1",
                "opt-out value {v:?} ignored"
            );
        }
    }

    #[test]
    fn any_other_value_still_enforces() {
        for v in ["true", "1", "yes", "nope", ""] {
            assert_eq!(
                resolve(Some(v), "http://evil.example.com/v1"),
                DEFAULT_URL,
                "value {v:?} must not disable enforcement"
            );
        }
    }

    #[test]
    fn localhost_http_is_always_allowed() {
        for url in [
            "http://localhost:8787/v1",
            "http://127.0.0.1:3000/v1",
            "http://[::1]:3000/v1",
        ] {
            assert_eq!(resolve(None, url), url);
        }
    }

    #[test]
    fn https_and_unset_overrides_resolve_as_before() {
        assert_eq!(
            resolve(None, "https://mirror.example.com/v1"),
            "https://mirror.example.com/v1"
        );
        let unset = with_override_env(None, &[], || resolve_api_url_override(VAR, DEFAULT_URL));
        assert_eq!(unset, DEFAULT_URL);
    }
}
