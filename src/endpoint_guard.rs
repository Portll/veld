//! Transport guard for the CLI's `--api-url`.
//!
//! The API key travels in an `x-api-key` header on every request this binary makes. Over `http://`
//! to a remote host that header is readable by anything on the path. The MCP server has guarded
//! this since it shipped — `shouldWarnInsecureApiUrl` in mcp-server/security-utils.ts, byte
//! identical in the shodh-memory twin — and the Rust CLI, which is the client an operator actually
//! runs, guarded nothing.
//!
//! LOOPBACK IS THE DEFAULT AND STAYS EXEMPT. Every `--api-url` default in this binary is
//! `http://127.0.0.1:3030`; plaintext to a socket on the same machine crosses no network, and
//! demanding TLS there would mean shipping a certificate for localhost or making the ordinary case
//! fail. So the rule is about REMOTE plaintext, not about plaintext.
//!
//! REFUSES rather than warns, which is the one place this deliberately diverges from the MCP
//! server. A warning on stderr is invisible in a hook, a TUI, or anything that redirects — and the
//! failure it warns about is silent by construction, since a leaked key produces no error at the
//! time it leaks. The escape hatch is the env var the MCP server already defines, so an operator
//! who has a reason (an SSH tunnel, a service mesh terminating TLS elsewhere) keeps one switch to
//! find rather than two.

/// Hosts for which plaintext HTTP crosses no network.
fn is_loopback_host(host: &str) -> bool {
    // Strip an IPv6 literal's brackets before parsing: `[::1]` is how a URL spells it.
    let h = host.trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match h.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        // A name we cannot parse is NOT assumed local. `localhost.example.com` resolves off-box,
        // and a suffix match on "localhost" would have accepted it.
        Err(_) => false,
    }
}

/// `Err(message)` when the URL would send the API key in cleartext across a network.
pub fn check_api_url(url: &str, allow_http: Option<&str>) -> Result<(), String> {
    let opted_out = matches!(allow_http, Some(v) if v.eq_ignore_ascii_case("true") || v == "1");

    let rest = match url.split_once("://") {
        Some((scheme, rest)) => {
            if scheme.eq_ignore_ascii_case("https") {
                return Ok(());
            }
            if !scheme.eq_ignore_ascii_case("http") {
                // Not a transport this guard reasons about; let the HTTP client reject it with its
                // own error rather than inventing one here.
                return Ok(());
            }
            rest
        }
        None => return Ok(()), // not a URL we can parse — the client will say so more precisely
    };

    // host[:port] up to the first `/`, `?` or `#`; userinfo stripped if present.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = if let Some(end) = authority.rfind(']') {
        &authority[..=end] // IPv6 literal, port (if any) follows the bracket
    } else {
        authority.split(':').next().unwrap_or(authority)
    };

    if is_loopback_host(host) {
        return Ok(());
    }
    if opted_out {
        return Ok(());
    }
    Err(format!(
        "refusing to send the API key in cleartext: --api-url is {url}, which is plain http:// to \
         the remote host `{host}`. Every request this command makes carries the key in an \
         x-api-key header, and over http:// anything on the path can read it.\n\n  \
         Use an https:// URL, or point --api-url at loopback (127.0.0.1 / ::1 / localhost) and \
         tunnel to the server.\n  \
         If TLS genuinely terminates elsewhere — a service mesh, an SSH tunnel you trust — set \
         VELD_ALLOW_HTTP=true to opt out. That is the same switch the MCP server uses."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_is_always_fine() {
        assert!(check_api_url("https://veld.example.com/api", None).is_ok());
    }

    #[test]
    fn loopback_over_http_is_fine_because_it_crosses_no_network() {
        for u in [
            "http://127.0.0.1:3030",
            "http://localhost:3030/api",
            "http://[::1]:3030",
            "http://127.5.5.5:80",
        ] {
            assert!(check_api_url(u, None).is_ok(), "{u} should be allowed");
        }
    }

    #[test]
    fn remote_http_is_refused_and_the_message_names_the_host() {
        let e = check_api_url("http://veld.example.com:3030/api", None).unwrap_err();
        assert!(
            e.contains("veld.example.com"),
            "the error must name the host: {e}"
        );
        assert!(e.contains("VELD_ALLOW_HTTP"), "and the escape hatch: {e}");
    }

    #[test]
    fn the_opt_out_works_and_only_for_the_documented_values() {
        assert!(check_api_url("http://remote.example.com", Some("true")).is_ok());
        assert!(check_api_url("http://remote.example.com", Some("1")).is_ok());
        assert!(check_api_url("http://remote.example.com", Some("TRUE")).is_ok());
        assert!(check_api_url("http://remote.example.com", Some("yes")).is_err());
        assert!(check_api_url("http://remote.example.com", Some("")).is_err());
    }

    // The bug a suffix match would have shipped: a hostname ENDING in "localhost" is not local.
    #[test]
    fn a_hostname_that_merely_ends_in_localhost_is_remote() {
        assert!(check_api_url("http://evil-localhost.example.com", None).is_err());
        assert!(check_api_url("http://localhost.example.com", None).is_err());
    }

    // userinfo must not be mistaken for the host — `http://127.0.0.1@evil.com` is a request to
    // evil.com, and reading the host as the part before `@` would have called it loopback.
    #[test]
    fn userinfo_does_not_disguise_a_remote_host() {
        assert!(check_api_url("http://127.0.0.1@evil.example.com/api", None).is_err());
    }
}
