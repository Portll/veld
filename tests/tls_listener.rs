//! The native TLS listener (`src/tls.rs`), served in-process on port 0: https over
//! HTTP/1.1, client certificates, certificate reload, handshakes off the accept loop,
//! and the refusals at load. Certificates are minted per test with rcgen.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Method, Request, StatusCode},
    routing::get,
    Extension, Router,
};
use http_body_util::BodyExt;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose,
};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

use veld::config::{ServerConfig, TlsListenerConfig};
use veld::handlers::{build_protected_routes, MultiUserMemoryManager};
use veld::tls::{ArrivedOverTls, CertReloader, TlsListener, TlsSetup};

struct Ca {
    issuer: CertifiedIssuer<'static, KeyPair>,
}

struct Leaf {
    cert_pem: String,
    cert_der: Vec<u8>,
    key_pem: String,
}

impl Ca {
    fn new(name: &str) -> Self {
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.distinguished_name.push(DnType::CommonName, name);
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        Self {
            issuer: CertifiedIssuer::self_signed(params, KeyPair::generate().unwrap()).unwrap(),
        }
    }

    fn leaf(&self, names: &[&str], usage: ExtendedKeyUsagePurpose, serial: u64) -> Leaf {
        let mut params =
            CertificateParams::new(names.iter().map(|n| n.to_string()).collect::<Vec<_>>())
                .unwrap();
        params.serial_number = Some(serial.into());
        params.extended_key_usages = vec![usage];
        let key = KeyPair::generate().unwrap();
        let cert = params.signed_by(&key, &*self.issuer).unwrap();
        Leaf {
            cert_pem: cert.pem(),
            cert_der: cert.der().to_vec(),
            key_pem: key.serialize_pem(),
        }
    }

    fn server_leaf(&self, serial: u64) -> Leaf {
        self.leaf(
            &["localhost", "127.0.0.1"],
            ExtendedKeyUsagePurpose::ServerAuth,
            serial,
        )
    }

    fn client_leaf(&self) -> Leaf {
        self.leaf(&["veld-follower"], ExtendedKeyUsagePurpose::ClientAuth, 7)
    }
}

fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    path
}

/// A router whose `/health` says how the request arrived and from where.
fn app() -> Router {
    Router::new().route(
        "/health",
        get(
            |tls: Option<Extension<ArrivedOverTls>>, ConnectInfo(peer): ConnectInfo<SocketAddr>| async move {
                format!(
                    "{} {}",
                    if tls.is_some() { "tls" } else { "plain" },
                    peer.ip()
                )
            },
        ),
    )
}

struct Served {
    addr: SocketAddr,
    certs: Arc<CertReloader>,
    stop: CancellationToken,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl Served {
    fn url(&self) -> String {
        format!("https://127.0.0.1:{}/health", self.addr.port())
    }

    async fn stop(self) {
        self.stop.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("the TLS server stops on its shutdown signal")
            .unwrap()
            .unwrap();
    }
}

async fn serve(cert: &Path, key: &Path, client_ca: Option<&Path>) -> Served {
    serve_router(cert, key, client_ca, app()).await
}

async fn serve_router(cert: &Path, key: &Path, client_ca: Option<&Path>, router: Router) -> Served {
    let setup = TlsSetup::load(&TlsListenerConfig {
        addr: "127.0.0.1:0".parse().unwrap(),
        cert: cert.to_path_buf(),
        key: key.to_path_buf(),
        client_ca: client_ca.map(Path::to_path_buf),
    })
    .unwrap();
    let listener = TlsListener::bind(setup.addr, Arc::clone(&setup.config))
        .await
        .unwrap();
    let addr = listener.addr();
    let stop = CancellationToken::new();
    let task = tokio::spawn(veld::tls::serve(
        listener,
        router,
        stop.clone().cancelled_owned(),
    ));
    Served {
        addr,
        certs: setup.certs,
        stop,
        task,
    }
}

/// A fresh client, so no pooled connection carries an earlier handshake's certificate.
fn client(ca_pem: &str, identity: Option<&Leaf>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder()
        .use_rustls_tls()
        .tls_built_in_root_certs(false)
        .add_root_certificate(reqwest::Certificate::from_pem(ca_pem.as_bytes()).unwrap())
        .tls_info(true)
        .no_proxy()
        .timeout(Duration::from_secs(10));
    if let Some(leaf) = identity {
        let pem = format!("{}{}", leaf.cert_pem, leaf.key_pem);
        builder = builder.identity(reqwest::Identity::from_pem(pem.as_bytes()).unwrap());
    }
    builder.build().unwrap()
}

/// The body of `GET url`, and the leaf certificate the server presented.
async fn get_health(client: &reqwest::Client, url: &str) -> reqwest::Result<(String, Vec<u8>)> {
    let response = client.get(url).send().await?.error_for_status()?;
    let peer = response
        .extensions()
        .get::<reqwest::tls::TlsInfo>()
        .and_then(|info| info.peer_certificate())
        .expect("tls_info(true) records the server's certificate")
        .to_vec();
    Ok((response.text().await?, peer))
}

#[tokio::test]
async fn https_serves_the_router_with_peer_address_and_tls_mark() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let leaf = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);

    let served = serve(&cert, &key, None).await;
    let (body, peer) = get_health(&client(&ca.issuer.pem(), None), &served.url())
        .await
        .unwrap();
    assert_eq!(body, "tls 127.0.0.1");
    assert_eq!(peer, leaf.cert_der);
    served.stop().await;
}

#[tokio::test]
async fn the_plain_listener_does_not_mark_requests_tls() {
    let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = tcp.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            tcp,
            app().into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
    });
    let body = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(format!("http://{addr}/health"))
        .header("ArrivedOverTls", "true")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, "plain 127.0.0.1");
    server.abort();
}

#[tokio::test]
async fn alpn_negotiates_http1_even_when_the_client_offers_h2_first() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let leaf = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);
    let served = serve(&cert, &key, None).await;

    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(rustls::pki_types::CertificateDer::from(
            ca.issuer.der().to_vec(),
        ))
        .unwrap();
    let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let tcp = tokio::net::TcpStream::connect(served.addr).await.unwrap();
    let mut tls = tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(
            rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap(),
            tcp,
        )
        .await
        .unwrap();
    assert_eq!(tls.get_ref().1.alpn_protocol(), Some(&b"http/1.1"[..]));

    tls.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut reply = Vec::new();
    // What was read stays in `reply` whether the close carries close_notify or not.
    let _ = tokio::time::timeout(Duration::from_secs(5), tls.read_to_end(&mut reply))
        .await
        .expect("the server answers and closes");
    let reply = String::from_utf8_lossy(&reply);
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.ends_with("tls 127.0.0.1"), "{reply}");
    served.stop().await;
}

#[tokio::test]
async fn a_client_ca_requires_a_certificate_from_that_ca() {
    let dir = TempDir::new().unwrap();
    let server_ca = Ca::new("veld server CA");
    let client_ca = Ca::new("veld client CA");
    let stranger_ca = Ca::new("someone else's CA");
    let leaf = server_ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);
    let client_ca_path = write(dir.path(), "client-ca.pem", &client_ca.issuer.pem());

    let served = serve(&cert, &key, Some(&client_ca_path)).await;
    let roots = server_ca.issuer.pem();

    assert!(
        get_health(&client(&roots, None), &served.url())
            .await
            .is_err(),
        "a client without a certificate is refused"
    );
    assert!(
        get_health(
            &client(&roots, Some(&stranger_ca.client_leaf())),
            &served.url()
        )
        .await
        .is_err(),
        "a certificate from another CA is refused"
    );
    let (body, _) = get_health(
        &client(&roots, Some(&client_ca.client_leaf())),
        &served.url(),
    )
    .await
    .unwrap();
    assert_eq!(body, "tls 127.0.0.1");
    served.stop().await;
}

#[tokio::test]
async fn a_reload_serves_the_new_certificate_and_a_bad_one_changes_nothing() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let first = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &first.cert_pem);
    let key = write(dir.path(), "key.pem", &first.key_pem);
    let served = serve(&cert, &key, None).await;
    let roots = ca.issuer.pem();

    let (_, peer) = get_health(&client(&roots, None), &served.url())
        .await
        .unwrap();
    assert_eq!(peer, first.cert_der);

    let second = ca.server_leaf(2);
    write(dir.path(), "chain.pem", &second.cert_pem);
    write(dir.path(), "key.pem", &second.key_pem);
    served.certs.reload_now().unwrap();
    assert_eq!(served.certs.current().cert[0].as_ref(), second.cert_der);
    let (_, peer) = get_health(&client(&roots, None), &served.url())
        .await
        .unwrap();
    assert_eq!(peer, second.cert_der, "a new connection gets the new leaf");

    // A key that does not match the certificate: the reload refuses, naming the key.
    write(dir.path(), "key.pem", &ca.server_leaf(3).key_pem);
    let err = format!("{:#}", served.certs.reload_now().unwrap_err());
    assert!(err.contains(&key.display().to_string()), "{err}");
    assert!(err.contains("does not match"), "{err}");
    let (_, peer) = get_health(&client(&roots, None), &served.url())
        .await
        .unwrap();
    assert_eq!(
        peer, second.cert_der,
        "the previous certificate stays in service"
    );
    served.stop().await;
}

#[tokio::test]
async fn a_silent_client_does_not_stall_other_handshakes() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let leaf = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);
    let served = serve(&cert, &key, None).await;

    let silent: Vec<_> =
        futures::future::join_all((0..4).map(|_| tokio::net::TcpStream::connect(served.addr)))
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect();
    let (body, _) = tokio::time::timeout(
        Duration::from_secs(5),
        get_health(&client(&ca.issuer.pem(), None), &served.url()),
    )
    .await
    .expect("a handshake completes while others are pending, well inside their timeout")
    .unwrap();
    assert_eq!(body, "tls 127.0.0.1");
    drop(silent);
    served.stop().await;
}

const API_KEY: &str = "tls-listener-test-key";
const ADMIN_KEY: &str = "tls-listener-test-admin-key";

/// `uri` on `router` with both keys, and the reply's status and JSON body.
async fn admin_call(
    router: &Router,
    method: Method,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("x-api-key", API_KEY)
        .header(veld::handlers::admin::ADMIN_API_KEY_HEADER, ADMIN_KEY)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn following_this_instance_through_its_tls_url_is_refused() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let leaf = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);
    let ca_path = write(dir.path(), "ca.pem", &ca.issuer.pem());
    std::env::set_var("VELD_API_KEYS", API_KEY);
    std::env::set_var(veld::handlers::admin::ADMIN_API_KEY_ENV, ADMIN_KEY);
    std::env::set_var(veld::replication::ENV_CA_CERT, &ca_path);
    std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");

    let storage = dir.path().join("data");
    std::fs::create_dir_all(&storage).unwrap();
    let config = ServerConfig {
        storage_path: storage.clone(),
        backup_enabled: false,
        ..ServerConfig::default()
    };
    let (plain_host, plain_port) = (config.host.clone(), config.port);
    let manager = Arc::new(MultiUserMemoryManager::new(storage, config).unwrap());
    let router = build_protected_routes(manager, false)
        .layer(axum::middleware::from_fn(veld::auth::auth_middleware));
    let served = serve_router(&cert, &key, None, router.clone()).await;

    let (status, body) =
        admin_call(&router, Method::GET, "/api/admin/replication/status", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["instance_id"],
        veld::replication::instance_id().to_string()
    );

    let source = format!("https://127.0.0.1:{}", served.addr.port());
    assert!(
        !veld::replication::is_same_instance(&source, &plain_host, plain_port),
        "the address check alone does not know the TLS port"
    );
    let (status, body) = admin_call(
        &router,
        Method::POST,
        "/api/admin/replication/follow",
        Some(serde_json::json!({
            "source": source,
            "admin_key": ADMIN_KEY,
            "api_key": API_KEY,
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"], "would_loop", "{body}");
    assert_eq!(body["detail"], format!("{source} is this instance"));
    assert!(veld::replication::status().target.is_none());
    served.stop().await;
}

fn load(dir: &Path, cert: &Path, key: &Path, client_ca: Option<&Path>) -> String {
    let err = TlsSetup::load(&TlsListenerConfig {
        addr: "127.0.0.1:0".parse().unwrap(),
        cert: cert.to_path_buf(),
        key: key.to_path_buf(),
        client_ca: client_ca.map(Path::to_path_buf),
    })
    .err()
    .unwrap_or_else(|| panic!("load in {} should refuse", dir.display()));
    format!("{err:#}")
}

#[test]
fn loading_refuses_files_that_do_not_serve_naming_the_file() {
    let dir = TempDir::new().unwrap();
    let ca = Ca::new("veld test CA");
    let leaf = ca.server_leaf(1);
    let cert = write(dir.path(), "chain.pem", &leaf.cert_pem);
    let key = write(dir.path(), "key.pem", &leaf.key_pem);
    let other_key = write(dir.path(), "other-key.pem", &ca.server_leaf(2).key_pem);
    let garbage = write(dir.path(), "garbage.pem", "not a certificate\n");
    let missing = dir.path().join("missing.pem");

    let err = load(dir.path(), &cert, &other_key, None);
    assert!(err.contains(&other_key.display().to_string()), "{err}");
    assert!(err.contains("does not match the certificate"), "{err}");

    let err = load(dir.path(), &garbage, &key, None);
    assert!(err.contains(&garbage.display().to_string()), "{err}");
    assert!(err.contains("no PEM certificate"), "{err}");

    let err = load(dir.path(), &cert, &missing, None);
    assert!(err.contains(&missing.display().to_string()), "{err}");

    let err = load(dir.path(), &cert, &cert, None);
    assert!(
        err.contains("no PKCS#8, RSA or SEC1 private key"),
        "a certificate is not a key: {err}"
    );

    let err = load(dir.path(), &cert, &key, Some(&garbage));
    assert!(err.contains("TLS client CA"), "{err}");
    assert!(err.contains(&garbage.display().to_string()), "{err}");
}

#[test]
fn a_half_configured_listener_is_refused_by_name() {
    let vars = [("VELD_TLS_KEY", "/etc/veld/key.pem")];
    let settings = veld::config::TlsSettings::from_lookup(|name| {
        vars.iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.to_string())
    });
    let err = settings
        .validate("127.0.0.1", 3030)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("VELD_TLS_CERT, VELD_TLS_PORT missing"),
        "{err}"
    );
}
