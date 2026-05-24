//! SSE subscriber that appends to the snapshot's activity tail.
//!
//! Reconnects with exponential backoff (capped at 30s). All transient errors are
//! traced rather than surfaced; the consumer learns about reachability from the
//! main HTTP probe in `client.rs`.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use parking_lot::RwLock;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use tracing::{debug, warn};

use crate::dto::MemoryEventDto;
use crate::snapshot::{ActivityEntry, StatusSnapshot};

const BACKOFF_INITIAL: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);

pub(crate) async fn run_sse_loop(
    http: Client,
    base: String,
    api_key: String,
    user_id: String,
    snapshot: Arc<RwLock<StatusSnapshot>>,
) {
    let url = format!("{}/api/events?user_id={}", base, user_id);
    let mut backoff = BACKOFF_INITIAL;

    loop {
        let request = http.get(&url).header("X-API-Key", &api_key);
        let mut source = match EventSource::new(request) {
            Ok(s) => s,
            Err(err) => {
                warn!(?err, "could not start SSE event source");
                tokio::time::sleep(backoff).await;
                backoff = next_backoff(backoff);
                continue;
            }
        };

        while let Some(event) = source.next().await {
            match event {
                Ok(Event::Open) => {
                    debug!(url, "SSE connection opened");
                    backoff = BACKOFF_INITIAL;
                }
                Ok(Event::Message(msg)) => {
                    if let Ok(dto) = serde_json::from_str::<MemoryEventDto>(&msg.data) {
                        let entry = ActivityEntry {
                            event_type: dto.event_type,
                            timestamp: dto.timestamp,
                            user_id: dto.user_id,
                            memory_type: dto.memory_type,
                            preview: dto.content_preview,
                        };
                        let mut guard = snapshot.write();
                        guard.push_activity(entry);
                    }
                }
                Err(err) => {
                    debug!(?err, "SSE stream ended");
                    break;
                }
            }
        }

        source.close();
        tokio::time::sleep(backoff).await;
        backoff = next_backoff(backoff);
    }
}

fn next_backoff(current: Duration) -> Duration {
    let doubled = current.saturating_mul(2);
    if doubled > BACKOFF_MAX {
        BACKOFF_MAX
    } else {
        doubled
    }
}
