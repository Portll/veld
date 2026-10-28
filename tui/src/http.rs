use reqwest::{Certificate, Client};
use std::sync::OnceLock;

pub const ENV_CA_CERT: &str = "VELD_CA_CERT";

static CLIENT: OnceLock<Client> = OnceLock::new();

/// Builds the one client every request goes through: the built-in webpki roots plus
/// any in [`ENV_CA_CERT`]. A bundle that cannot be read or holds no certificate is an
/// error naming the variable and the path, never skipped.
pub fn init() -> Result<(), String> {
    let client = build(std::env::var(ENV_CA_CERT).ok().as_deref())?;
    let _ = CLIENT.set(client);
    Ok(())
}

pub fn client() -> Client {
    CLIENT.get_or_init(Client::new).clone()
}

fn build(ca_bundle: Option<&str>) -> Result<Client, String> {
    let builder = Client::builder();
    let Some(path) = ca_bundle.map(str::trim).filter(|p| !p.is_empty()) else {
        return builder.build().map_err(|e| e.to_string());
    };
    let refusal = |reason: String| format!("{ENV_CA_CERT}={path}: {reason}");
    let pem = std::fs::read(path).map_err(|e| refusal(format!("cannot read the file: {e}")))?;
    let certs = Certificate::from_pem_bundle(&pem)
        .map_err(|e| refusal(format!("not a PEM certificate bundle: {e}")))?;
    if certs.is_empty() {
        return Err(refusal("no PEM certificate in the file".into()));
    }
    certs
        .into_iter()
        .fold(builder, |b, cert| b.add_root_certificate(cert))
        .build()
        .map_err(|e| refusal(format!("refused by the TLS client: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!(
            "{}/../tests/fixtures/tls/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn no_bundle_builds_a_plain_client() {
        assert!(build(None).is_ok());
        assert!(build(Some("  ")).is_ok());
    }

    #[test]
    fn a_bundle_of_certificates_builds() {
        assert!(build(Some(&fixture("test-ca-bundle.pem"))).is_ok());
    }

    #[test]
    fn an_unreadable_bundle_names_the_variable_and_the_path() {
        let missing = fixture("no-such-bundle.pem");
        let err = build(Some(&missing)).unwrap_err();
        assert!(
            err.starts_with(&format!("{ENV_CA_CERT}={missing}: cannot read")),
            "{err}"
        );
    }

    #[test]
    fn a_corrupt_or_empty_bundle_is_refused() {
        let corrupt = fixture("corrupt-cert.pem");
        let err = build(Some(&corrupt)).unwrap_err();
        assert!(
            err.starts_with(&format!("{ENV_CA_CERT}={corrupt}:")),
            "{err}"
        );

        let not_pem = format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
        let err = build(Some(&not_pem)).unwrap_err();
        assert!(err.contains("no PEM certificate"), "{err}");
    }
}
