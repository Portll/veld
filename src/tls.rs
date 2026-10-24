//! Native TLS: an optional second listener serving the HTTP API over rustls.
//!
//! The plain listener is unchanged and keeps every local client on http over loopback.
//! This one starts when `VELD_TLS_CERT`, `VELD_TLS_KEY` and `VELD_TLS_PORT` are all set
//! (see [`crate::config::TlsSettings`]). Handshakes run off the accept loop, bounded in
//! number and time, so a slow or hostile client cannot stall other connections, and one
//! peer address is capped in the handshakes and connections it may hold. The
//! certificate reloads when its files change and on SIGHUP, without a restart.
//!
//! Crypto is rustls on ring, named explicitly: the tree carries no aws-lc-rs and the
//! process-default provider is never consulted.

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use axum::serve::ListenerExt;
use chrono::{DateTime, NaiveDateTime, Utc};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::{self, PemObject};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::{ClientHello, ResolvesServerCert, WebPkiClientVerifier};
use rustls::sign::CertifiedKey;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Semaphore};
use tokio_rustls::server::TlsStream;
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::TlsListenerConfig;

/// How long a client gets to finish the TLS handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Handshakes in progress at once. At the limit the listener stops accepting, and the
/// kernel's backlog holds new connections until a handshake ends.
pub const MAX_CONCURRENT_HANDSHAKES: usize = 256;

/// Handshakes one peer IP address may have in progress at once, so one host cannot take
/// every slot of [`MAX_CONCURRENT_HANDSHAKES`]. A connection over it is closed at accept.
pub const MAX_HANDSHAKES_PER_PEER: usize = 16;

/// TLS connections, handshaking or established, one peer IP address may hold at once.
/// A connection over it is closed at accept.
pub const MAX_CONNECTIONS_PER_PEER: usize = 64;

/// How often the certificate and key files are checked for a change.
pub const RELOAD_POLL_INTERVAL: Duration = Duration::from_secs(30);

static PROVIDER: LazyLock<Arc<CryptoProvider>> =
    LazyLock::new(|| Arc::new(rustls::crypto::ring::default_provider()));

/// A request extension present only on requests that arrived over the TLS listener.
/// It is added by a layer on that listener's copy of the router, so a client cannot
/// set it on the plain listener.
#[derive(Debug, Clone, Copy)]
pub struct ArrivedOverTls;

/// A TLS listener's files loaded and checked, ready for [`TlsListener::bind`].
pub struct TlsSetup {
    pub addr: SocketAddr,
    pub certs: Arc<CertReloader>,
    pub config: Arc<rustls::ServerConfig>,
    /// Whether a client must present a certificate from the configured CA.
    pub client_auth: bool,
}

impl TlsSetup {
    /// Load the certificate, key and client CA. Fails, naming the file, when one does
    /// not parse or the key does not match the certificate.
    pub fn load(listener: &TlsListenerConfig) -> Result<Self> {
        let certs = Arc::new(CertReloader::load(&listener.cert, &listener.key)?);
        let config = server_config(Arc::clone(&certs), listener.client_ca.as_deref())?;
        info!(
            cert = %listener.cert.display(),
            not_after = %expiry(&certs.current()),
            client_ca = %listener
                .client_ca
                .as_deref()
                .map_or_else(|| "none".to_string(), |p| p.display().to_string()),
            "TLS certificate loaded"
        );
        Ok(Self {
            addr: listener.addr,
            certs,
            config: Arc::new(config),
            client_auth: listener.client_ca.is_some(),
        })
    }
}

fn server_config(
    certs: Arc<CertReloader>,
    client_ca: Option<&Path>,
) -> Result<rustls::ServerConfig> {
    let builder = rustls::ServerConfig::builder_with_provider(Arc::clone(&PROVIDER))
        .with_safe_default_protocol_versions()
        .context("the ring provider offers no default TLS version")?;
    let builder = match client_ca {
        None => builder.with_no_client_auth(),
        Some(path) => {
            let mut roots = rustls::RootCertStore::empty();
            for cert in read_chain(path, "TLS client CA")? {
                roots.add(cert).with_context(|| {
                    format!(
                        "TLS client CA {}: not a usable CA certificate",
                        path.display()
                    )
                })?;
            }
            let verifier =
                WebPkiClientVerifier::builder_with_provider(Arc::new(roots), Arc::clone(&PROVIDER))
                    .build()
                    .with_context(|| format!("TLS client CA {}", path.display()))?;
            builder.with_client_cert_verifier(verifier)
        }
    };
    let mut config = builder.with_cert_resolver(certs);
    // Not h2: browser WebSockets over h2 need RFC 8441, and both the `?api_key=`
    // WebSocket fallback and passkey `relying_party()` read HTTP/1.1's `Host` header.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(config)
}

fn read_chain(path: &Path, what: &str) -> Result<Vec<CertificateDer<'static>>> {
    let chain = CertificateDer::pem_file_iter(path)
        .and_then(|certs| certs.collect::<Result<Vec<_>, _>>())
        .with_context(|| format!("{what} {}", path.display()))?;
    if chain.is_empty() {
        anyhow::bail!("{what} {}: no PEM certificate in the file", path.display());
    }
    Ok(chain)
}

fn read_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    PrivateKeyDer::from_pem_file(path)
        .map_err(|e| match e {
            pem::Error::NoItemsFound => {
                anyhow::anyhow!("no PKCS#8, RSA or SEC1 private key in the file")
            }
            e => anyhow::Error::new(e),
        })
        .with_context(|| format!("TLS key {}", path.display()))
}

fn load_certified_key(cert: &Path, key: &Path) -> Result<CertifiedKey> {
    let chain = read_chain(cert, "TLS certificate")?;
    let signing_key = PROVIDER
        .key_provider
        .load_private_key(read_key(key)?)
        .with_context(|| format!("TLS key {}: not a key rustls can sign with", key.display()))?;
    let certified = CertifiedKey::new(chain, signing_key);
    certified.keys_match().with_context(|| {
        format!(
            "TLS key {} does not match the certificate {}",
            key.display(),
            cert.display()
        )
    })?;
    Ok(certified)
}

type FileStamp = Option<(SystemTime, u64)>;

fn stamps(cert: &Path, key: &Path) -> (FileStamp, FileStamp) {
    let stamp = |path: &Path| {
        let meta = std::fs::metadata(path).ok()?;
        Some((meta.modified().ok()?, meta.len()))
    };
    (stamp(cert), stamp(key))
}

/// Serves the current certificate to every handshake and swaps in a new one when the
/// files change. A reload that fails leaves the previous certificate in service.
pub struct CertReloader {
    cert: PathBuf,
    key: PathBuf,
    current: ArcSwap<CertifiedKey>,
    seen: parking_lot::Mutex<(FileStamp, FileStamp)>,
}

impl std::fmt::Debug for CertReloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertReloader")
            .field("cert", &self.cert)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl ResolvesServerCert for CertReloader {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.current.load_full())
    }
}

impl CertReloader {
    pub fn load(cert: &Path, key: &Path) -> Result<Self> {
        let seen = stamps(cert, key);
        let certified = load_certified_key(cert, key)?;
        Ok(Self {
            cert: cert.to_path_buf(),
            key: key.to_path_buf(),
            current: ArcSwap::from_pointee(certified),
            seen: parking_lot::Mutex::new(seen),
        })
    }

    /// The certificate and key handshakes are served now.
    pub fn current(&self) -> Arc<CertifiedKey> {
        self.current.load_full()
    }

    /// Read both files again and serve them if they parse and match.
    pub fn reload_now(&self) -> Result<()> {
        // Recorded before parsing, so a broken file is reported once per change rather
        // than on every poll, and a write that lands mid-read is seen by the next poll.
        *self.seen.lock() = stamps(&self.cert, &self.key);
        let certified = load_certified_key(&self.cert, &self.key)?;
        let not_after = expiry(&certified);
        self.current.store(Arc::new(certified));
        info!(cert = %self.cert.display(), not_after = %not_after, "TLS certificate reloaded");
        Ok(())
    }

    fn reload_logged(&self) {
        if let Err(e) = self.reload_now() {
            error!("TLS certificate reload failed; still serving the previous certificate: {e:#}");
        }
    }

    fn reload_if_changed(&self) {
        if stamps(&self.cert, &self.key) != *self.seen.lock() {
            self.reload_logged();
        }
    }
}

/// Reload `certs` when its files change and on SIGHUP, until `stop` is cancelled.
pub fn spawn_reloader(certs: Arc<CertReloader>, stop: CancellationToken) {
    let mut hangups = Hangups::new();
    tokio::spawn(async move {
        let mut poll = tokio::time::interval(RELOAD_POLL_INTERVAL);
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        poll.tick().await;
        loop {
            let forced = tokio::select! {
                _ = stop.cancelled() => return,
                _ = poll.tick() => false,
                _ = hangups.recv() => true,
            };
            let certs = Arc::clone(&certs);
            let reload = tokio::task::spawn_blocking(move || {
                if forced {
                    info!("SIGHUP: reloading the TLS certificate");
                    certs.reload_logged();
                } else {
                    certs.reload_if_changed();
                }
            });
            if let Err(e) = reload.await {
                error!("TLS certificate reload panicked: {e}");
            }
        }
    });
}

#[cfg(unix)]
struct Hangups(Option<tokio::signal::unix::Signal>);

#[cfg(unix)]
impl Hangups {
    fn new() -> Self {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()) {
            Ok(signal) => Self(Some(signal)),
            Err(e) => {
                warn!("SIGHUP handler unavailable ({e}); the TLS certificate reloads only when its files change");
                Self(None)
            }
        }
    }

    async fn recv(&mut self) {
        if let Some(signal) = self.0.as_mut() {
            if signal.recv().await.is_some() {
                return;
            }
            self.0 = None;
        }
        std::future::pending().await
    }
}

#[cfg(not(unix))]
struct Hangups;

#[cfg(not(unix))]
impl Hangups {
    fn new() -> Self {
        Self
    }

    async fn recv(&mut self) {
        std::future::pending().await
    }
}

fn expiry(certified: &CertifiedKey) -> String {
    certified
        .cert
        .first()
        .and_then(|leaf| not_after(leaf))
        .map_or_else(|| "unknown".to_string(), |t| t.to_rfc3339())
}

/// The certificate's validity notAfter (RFC 5280 4.1.2.5), read from its DER.
fn not_after(cert: &[u8]) -> Option<DateTime<Utc>> {
    let (certificate, _) = der_element(cert, 0x30)?;
    let (tbs, _) = der_element(certificate, 0x30)?;
    let tbs = der_element(tbs, 0xA0).map_or(tbs, |(_, rest)| rest);
    let (_, tbs) = der_element(tbs, 0x02)?;
    let (_, tbs) = der_element(tbs, 0x30)?;
    let (_, tbs) = der_element(tbs, 0x30)?;
    let (validity, _) = der_element(tbs, 0x30)?;
    let (_, rest) = der_time(validity)?;
    Some(der_time(rest)?.0)
}

/// The content of the DER element at the head of `input` if it has `tag`, and what
/// follows it.
fn der_element(input: &[u8], tag: u8) -> Option<(&[u8], &[u8])> {
    let (&found, rest) = input.split_first()?;
    if found != tag {
        return None;
    }
    let (&first, rest) = rest.split_first()?;
    let (len, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let width = usize::from(first & 0x7f);
        if width == 0 || width > 4 || rest.len() < width {
            return None;
        }
        let (len, rest) = rest.split_at(width);
        (
            len.iter()
                .fold(0usize, |acc, b| (acc << 8) | usize::from(*b)),
            rest,
        )
    };
    (rest.len() >= len).then(|| rest.split_at(len))
}

/// A UTCTime or GeneralizedTime at the head of `input`, and what follows it.
fn der_time(input: &[u8]) -> Option<(DateTime<Utc>, &[u8])> {
    let tag = *input.first()?;
    let (value, rest) = der_element(input, tag)?;
    let text = std::str::from_utf8(value).ok()?.strip_suffix('Z')?;
    let full = match tag {
        0x17 => {
            let yy: u32 = text.get(..2)?.parse().ok()?;
            let century = if yy >= 50 { "19" } else { "20" };
            format!("{century}{text}")
        }
        0x18 => text.to_string(),
        _ => return None,
    };
    let time = NaiveDateTime::parse_from_str(&full, "%Y%m%d%H%M%S").ok()?;
    Some((time.and_utc(), rest))
}

/// Slots held per peer IP address, up to a cap. An address's entry is removed when its
/// count returns to zero, so the map holds only addresses with something open.
#[derive(Debug)]
struct PeerSlots {
    cap: usize,
    held: parking_lot::Mutex<HashMap<IpAddr, usize>>,
}

impl PeerSlots {
    fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            cap,
            held: parking_lot::Mutex::new(HashMap::new()),
        })
    }

    /// A slot for `ip`, or `None` when it already holds `cap`.
    fn try_acquire(self: &Arc<Self>, ip: IpAddr) -> Option<PeerSlot> {
        let mut held = self.held.lock();
        let count = held.get(&ip).copied().unwrap_or(0);
        if count >= self.cap {
            return None;
        }
        held.insert(ip, count + 1);
        Some(PeerSlot {
            slots: Arc::clone(self),
            ip,
        })
    }
}

/// One slot of a [`PeerSlots`], given back on drop.
#[derive(Debug)]
struct PeerSlot {
    slots: Arc<PeerSlots>,
    ip: IpAddr,
}

impl Drop for PeerSlot {
    fn drop(&mut self) {
        let mut held = self.slots.held.lock();
        if let Some(count) = held.get_mut(&self.ip) {
            *count -= 1;
            if *count == 0 {
                held.remove(&self.ip);
            }
        }
    }
}

/// A TLS connection from the listener. It holds its peer's slot of
/// [`MAX_CONNECTIONS_PER_PEER`] until it is dropped.
#[derive(Debug)]
pub struct TlsConnection {
    stream: TlsStream<TcpStream>,
    _slot: PeerSlot,
}

impl AsyncRead for TlsConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().stream).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().stream).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_shutdown(cx)
    }
}

/// The TLS listener: accepts TCP, runs each handshake on its own task, and hands
/// finished streams to `axum::serve`.
pub struct TlsListener {
    addr: SocketAddr,
    ready: mpsc::Receiver<(TlsConnection, SocketAddr)>,
    accepting: tokio::task::JoinHandle<()>,
}

impl TlsListener {
    pub async fn bind(addr: SocketAddr, config: Arc<rustls::ServerConfig>) -> Result<Self> {
        let tcp = TcpListener::bind(addr)
            .await
            .with_context(|| format!("Failed to bind TLS listener on {addr}"))?;
        let addr = tcp
            .local_addr()
            .with_context(|| format!("TLS listener on {addr} has no local address"))?;
        let (handoff, ready) = mpsc::channel(MAX_CONCURRENT_HANDSHAKES);
        let accepting = tokio::spawn(accept_loop(tcp, TlsAcceptor::from(config), handoff));
        Ok(Self {
            addr,
            ready,
            accepting,
        })
    }

    /// The address bound, with the port the OS chose when asked for port 0.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        self.accepting.abort();
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsConnection;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.ready.recv().await {
            Some(conn) => conn,
            None => {
                error!(addr = %self.addr, "TLS accept loop ended; no new TLS connections are taken");
                std::future::pending().await
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Ok(self.addr)
    }
}

async fn accept_loop(
    tcp: TcpListener,
    acceptor: TlsAcceptor,
    handoff: mpsc::Sender<(TlsConnection, SocketAddr)>,
) {
    let handshakes = Arc::new(Semaphore::new(MAX_CONCURRENT_HANDSHAKES));
    let peer_handshakes = PeerSlots::new(MAX_HANDSHAKES_PER_PEER);
    let peer_connections = PeerSlots::new(MAX_CONNECTIONS_PER_PEER);
    loop {
        let Ok(permit) = Arc::clone(&handshakes).acquire_owned().await else {
            return;
        };
        let (stream, peer) = match tcp.accept().await {
            Ok(accepted) => accepted,
            Err(e) => {
                accept_error(e).await;
                continue;
            }
        };
        let ip = peer.ip().to_canonical();
        let Some(connection) = peer_connections.try_acquire(ip) else {
            debug!(%peer, "TLS connection closed: the peer holds {MAX_CONNECTIONS_PER_PEER} connections");
            continue;
        };
        let Some(handshake) = peer_handshakes.try_acquire(ip) else {
            debug!(%peer, "TLS connection closed: the peer has {MAX_HANDSHAKES_PER_PEER} handshakes in progress");
            continue;
        };
        let acceptor = acceptor.clone();
        let handoff = handoff.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _handshake = handshake;
            // Debug, not warn: scanners and port probes fail handshakes all day.
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
                Ok(Ok(stream)) => {
                    let connection = TlsConnection {
                        stream,
                        _slot: connection,
                    };
                    let _ = handoff.send((connection, peer)).await;
                }
                Ok(Err(e)) => debug!(%peer, "TLS handshake failed: {e}"),
                Err(_) => debug!(
                    %peer,
                    "TLS handshake timed out after {}s",
                    HANDSHAKE_TIMEOUT.as_secs()
                ),
            }
        });
    }
}

/// As `axum::serve` treats accept errors: a connection-level error is the peer's and
/// is skipped; anything else (EMFILE) is logged and waited out.
async fn accept_error(e: std::io::Error) {
    use std::io::ErrorKind::{ConnectionAborted, ConnectionRefused, ConnectionReset};
    if matches!(
        e.kind(),
        ConnectionRefused | ConnectionAborted | ConnectionReset
    ) {
        return;
    }
    error!("TLS listener accept error: {e}");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

/// Serve `app` on `listener` until `shutdown` completes, marking every request
/// [`ArrivedOverTls`].
pub async fn serve<F>(listener: TlsListener, app: axum::Router, shutdown: F) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = app.layer(axum::Extension(ArrivedOverTls));
    // The tap does nothing. axum implements `Connected` for `SocketAddr` over a
    // `TapIo` of any listener and over no other custom one, so it is what keeps
    // `ConnectInfo<SocketAddr>` (rate limiting, admin audit) working here.
    axum::serve(
        listener.tap_io(|_| {}),
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cert_expiring(year: i32) -> Vec<u8> {
        let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()]).unwrap();
        params.not_after = rcgen::date_time_ymd(year, 5, 17);
        let key = rcgen::KeyPair::generate().unwrap();
        params.self_signed(&key).unwrap().der().to_vec()
    }

    #[test]
    fn not_after_reads_utc_and_generalized_times() {
        assert_eq!(
            not_after(&cert_expiring(2031)).unwrap().to_rfc3339(),
            "2031-05-17T00:00:00+00:00"
        );
        assert_eq!(
            not_after(&cert_expiring(2051)).unwrap().to_rfc3339(),
            "2051-05-17T00:00:00+00:00",
            "years from 2050 are GeneralizedTime"
        );
    }

    #[test]
    fn one_peer_at_its_handshake_cap_leaves_room_for_another() {
        let slots = PeerSlots::new(MAX_HANDSHAKES_PER_PEER);
        let busy: IpAddr = "192.0.2.1".parse().unwrap();
        let other: IpAddr = "2001:db8::2".parse().unwrap();
        let held: Vec<_> = (0..MAX_HANDSHAKES_PER_PEER)
            .map(|_| slots.try_acquire(busy).expect("under the cap"))
            .collect();
        assert!(slots.try_acquire(busy).is_none(), "over the cap");
        assert!(
            slots.try_acquire(other).is_some(),
            "another peer has its own"
        );
        drop(held);
        assert!(slots.try_acquire(busy).is_some());
    }

    #[test]
    fn a_peer_slot_is_given_back_on_drop_and_an_idle_peer_leaves_the_map() {
        let slots = PeerSlots::new(2);
        let a: IpAddr = "192.0.2.1".parse().unwrap();
        let b: IpAddr = "192.0.2.2".parse().unwrap();
        let first = slots.try_acquire(a).unwrap();
        let second = slots.try_acquire(a).unwrap();
        assert!(slots.try_acquire(a).is_none());
        let on_b = slots.try_acquire(b).unwrap();
        assert_eq!(slots.held.lock().get(&a), Some(&2));

        drop(first);
        assert_eq!(slots.held.lock().get(&a), Some(&1));
        let third = slots.try_acquire(a).expect("a dropped slot is given back");
        drop((second, third));
        assert!(!slots.held.lock().contains_key(&a), "a peer at zero leaves");
        assert_eq!(slots.held.lock().get(&b), Some(&1));
        drop(on_b);
        assert!(slots.held.lock().is_empty());

        let none = PeerSlots::new(0);
        assert!(none.try_acquire(a).is_none());
        assert!(none.held.lock().is_empty(), "a refusal leaves no entry");
    }

    #[test]
    fn not_after_refuses_what_is_not_a_certificate() {
        assert_eq!(not_after(b""), None);
        assert_eq!(not_after(&[0x30, 0x82, 0xff]), None);
        let cert = cert_expiring(2031);
        assert_eq!(not_after(&cert[..cert.len() / 2]), None);
    }
}
