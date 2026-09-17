//! `LenientJson<T>` — a drop-in replacement for [`axum::Json`] that survives
//! transport-level encoding noise in the request body.
//!
//! Background: `axum::Json<T>` insists on the body being valid UTF-8 because
//! it goes through `serde_json::from_slice` which rejects non-UTF-8 bytes
//! with `invalid unicode code point at line N column M`. That makes the
//! handler reject the entire request when a client (especially Windows
//! shells like Git Bash, PowerShell with CP1252 default, or `curl -d` from
//! a terminal that re-encodes copy-pasted text) sends an otherwise valid
//! JSON document containing a single byte such as `0x97` (CP1252 em-dash)
//! that doesn't decode as UTF-8.
//!
//! `LenientJson<T>` recovers from that case:
//!
//! 1. Read the body as raw bytes.
//! 2. If the bytes are already valid UTF-8, parse directly (zero copy).
//! 3. Otherwise, run `String::from_utf8_lossy`, which substitutes each
//!    invalid byte sequence with U+FFFD (REPLACEMENT CHARACTER) and then
//!    parse the resulting `String`.
//!
//! Outcome: malformed bytes get one replacement char per invalid sequence,
//! but the document structure (braces, commas, quotes, ASCII text, valid
//! UTF-8 multibyte runs) is preserved and the request succeeds.
//!
//! Lossy substitution is fine for human-readable JSON content (the cost is
//! one `\u{fffd}` per garbled char) and is strictly an upgrade over the
//! status quo of rejecting the entire request.

use axum::{
    body::Bytes,
    extract::{rejection::BytesRejection, FromRequest, Request},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;
use std::borrow::Cow;

/// `Json<T>`-shaped extractor that lossy-decodes the body to UTF-8 before
/// running `serde_json`. See module docs for rationale.
pub struct LenientJson<T>(pub T);

impl<T, S> FromRequest<S> for LenientJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = LenientJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        if !is_json_content_type(&req) {
            return Err(LenientJsonRejection::WrongContentType);
        }
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(LenientJsonRejection::Bytes)?;
        let text: Cow<'_, str> = match std::str::from_utf8(&bytes) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(String::from_utf8_lossy(&bytes).into_owned()),
        };
        let value = serde_json::from_str::<T>(&text).map_err(LenientJsonRejection::Json)?;
        Ok(LenientJson(value))
    }
}

fn is_json_content_type(req: &Request) -> bool {
    // Mirror axum::Json's rule: accept missing Content-Type (some clients
    // omit it) and anything whose mime essence is application/json or its
    // common +json variants. Anything else is rejected up-front so
    // form-encoded or multipart bodies don't accidentally fall through.
    let Some(ct) = req.headers().get(header::CONTENT_TYPE) else {
        return true;
    };
    let Ok(s) = ct.to_str() else { return false };
    let essence = s
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    essence == "application/json" || essence.ends_with("+json")
}

#[derive(Debug)]
pub enum LenientJsonRejection {
    WrongContentType,
    Bytes(BytesRejection),
    Json(serde_json::Error),
}

impl IntoResponse for LenientJsonRejection {
    fn into_response(self) -> Response {
        match self {
            Self::WrongContentType => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Expected request with Content-Type: application/json",
            )
                .into_response(),
            Self::Bytes(rej) => rej.into_response(),
            Self::Json(err) => (
                StatusCode::BAD_REQUEST,
                format!("Failed to parse the request body as JSON: {err}"),
            )
                .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, PartialEq)]
    struct Probe {
        content: String,
        n: u32,
    }

    fn build_req(bytes: Vec<u8>) -> Request<Body> {
        Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(bytes))
            .unwrap()
    }

    #[tokio::test]
    async fn parses_valid_utf8_json() {
        let req = build_req(br#"{"content":"hello","n":1}"#.to_vec());
        let LenientJson(v) = LenientJson::<Probe>::from_request(req, &()).await.unwrap();
        assert_eq!(
            v,
            Probe {
                content: "hello".into(),
                n: 1
            }
        );
    }

    #[tokio::test]
    async fn recovers_from_cp1252_em_dash() {
        // {"content":"a\x97b","n":2} — 0x97 is em-dash in CP1252 (invalid UTF-8).
        let mut bytes = br#"{"content":"a"#.to_vec();
        bytes.push(0x97);
        bytes.extend_from_slice(br#"b","n":2}"#);
        let req = build_req(bytes);
        let LenientJson(v) = LenientJson::<Probe>::from_request(req, &()).await.unwrap();
        // The invalid byte becomes U+FFFD; structure survives.
        assert_eq!(v.n, 2);
        assert!(v.content.contains('a') && v.content.contains('b'));
    }

    #[tokio::test]
    async fn rejects_structurally_invalid_json() {
        let req = build_req(br#"{"content":"unclosed,"n":3"#.to_vec());
        let r = LenientJson::<Probe>::from_request(req, &()).await;
        assert!(matches!(r, Err(LenientJsonRejection::Json(_))));
    }

    #[tokio::test]
    async fn rejects_wrong_content_type() {
        let req = Request::builder()
            .header(header::CONTENT_TYPE, "text/plain")
            .body(Body::from(br#"{"content":"x","n":4}"#.to_vec()))
            .unwrap();
        let r = LenientJson::<Probe>::from_request(req, &()).await;
        assert!(matches!(r, Err(LenientJsonRejection::WrongContentType)));
    }
}
