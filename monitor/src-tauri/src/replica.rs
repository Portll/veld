//! The tray's replication line: every instance in the fleet that follows a leader,
//! reduced from `GET /api/admin/replication/status` to how far its tenants are behind.

use serde_json::Value;

/// One following instance's tenants, reduced to what the tray says about them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReplicaSummary {
    /// Tenants whose lag is known and above zero.
    pub behind: usize,
    pub max_lag: u64,
    /// Tenants retrying after a failed page.
    pub retrying: usize,
    pub diverged: usize,
    /// The follower could not read its leader's status at the last poll.
    pub leader_unreachable: bool,
    /// No poll has finished: the first pass over every tenant is still running.
    pub first_pass: bool,
}

impl ReplicaSummary {
    fn merge(mut self, other: &ReplicaSummary) -> ReplicaSummary {
        self.behind += other.behind;
        self.max_lag = self.max_lag.max(other.max_lag);
        self.retrying += other.retrying;
        self.diverged += other.diverged;
        self.leader_unreachable |= other.leader_unreachable;
        self.first_pass |= other.first_pass;
        self
    }
}

fn present(value: &Value, key: &str) -> bool {
    value.get(key).is_some_and(|v| !v.is_null())
}

/// Leader frames a tenant has yet to apply: the server's own `lag` when it reports
/// one, else `leader_head_lsn - 1 - applied_lsn`. None while either is unknown.
pub fn tenant_lag(progress: &Value) -> Option<u64> {
    if let Some(lag) = progress.get("lag").and_then(Value::as_u64) {
        return Some(lag);
    }
    let head = progress.get("leader_head_lsn")?.as_u64()?;
    let applied = progress.get("applied_lsn")?.as_u64()?;
    Some(head.saturating_sub(1).saturating_sub(applied))
}

/// The replication status of one instance, reduced; None when it follows nothing.
pub fn summarize(status: &Value) -> Option<ReplicaSummary> {
    let following = status.get("following").filter(|f| f.is_object())?;
    let mut summary = ReplicaSummary {
        leader_unreachable: present(following, "last_error"),
        first_pass: following.get("polls").and_then(Value::as_u64) == Some(0),
        ..ReplicaSummary::default()
    };
    for progress in following
        .get("users")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|users| users.values())
    {
        // A diverged tenant is no longer synced, so its lag means nothing.
        if present(progress, "diverged") {
            summary.diverged += 1;
            continue;
        }
        if present(progress, "last_error") {
            summary.retrying += 1;
        }
        if let Some(lag) = tenant_lag(progress).filter(|lag| *lag > 0) {
            summary.behind += 1;
            summary.max_lag = summary.max_lag.max(lag);
        }
    }
    Some(summary)
}

/// One line over every following instance (label, summary); None when none follows.
pub fn tray_line(replicas: &[(String, ReplicaSummary)]) -> Option<String> {
    let (name, total) = match replicas {
        [] => return None,
        [(label, summary)] => (format!("Replica {label}"), summary.clone()),
        many => (
            format!("{} replicas", many.len()),
            many.iter()
                .fold(ReplicaSummary::default(), |acc, (_, s)| acc.merge(s)),
        ),
    };
    let mut parts: Vec<String> = Vec::new();
    if total.leader_unreachable {
        parts.push("leader unreachable".into());
    }
    if total.behind > 0 {
        let tenants = if total.behind == 1 { "tenant" } else { "tenants" };
        parts.push(format!(
            "{} {tenants} behind, max lag {} frames",
            total.behind,
            crate::tray::thousands(total.max_lag)
        ));
    } else if total.first_pass {
        parts.push("first pass".into());
    }
    if total.retrying > 0 {
        parts.push(format!("{} retrying", total.retrying));
    }
    if total.diverged > 0 {
        parts.push(format!("{} diverged", total.diverged));
    }
    if parts.is_empty() {
        parts.push("in sync".into());
    }
    Some(format!("{name}: {}", parts.join(" · ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn status(polls: u64, users: Value) -> Value {
        json!({
            "base_url": "http://127.0.0.1:3031",
            "users": [],
            "following": {
                "target": { "source": "http://127.0.0.1:3030", "since": "2026-10-19T14:43:05Z" },
                "generation": 1, "polls": polls, "pages": 37,
                "last_poll": null, "last_applied": "2026-10-19T16:10:56Z", "last_error": null,
                "users": users,
            },
        })
    }

    #[test]
    fn a_leader_has_no_line() {
        let leader = json!({ "base_url": "http://127.0.0.1:3030", "users": [], "following": null });
        assert_eq!(summarize(&leader), None);
        assert_eq!(tray_line(&[]), None);
    }

    #[test]
    fn lag_prefers_the_servers_figure_and_is_unknown_without_both_lsns() {
        assert_eq!(tenant_lag(&json!({ "lag": 7, "leader_head_lsn": 100, "applied_lsn": 10 })), Some(7));
        assert_eq!(tenant_lag(&json!({ "leader_head_lsn": 41219, "applied_lsn": 18499 })), Some(22719));
        assert_eq!(tenant_lag(&json!({ "leader_head_lsn": 11, "applied_lsn": 10 })), Some(0));
        assert_eq!(tenant_lag(&json!({ "leader_head_lsn": null, "applied_lsn": 1475 })), None);
        assert_eq!(tenant_lag(&json!({ "leader_head_lsn": 5, "applied_lsn": null })), None);
    }

    #[test]
    fn a_first_pass_reports_the_tenants_behind_and_ignores_unknown_lag() {
        let s = status(0, json!({
            "portll": { "applied_lsn": 18499, "leader_head_lsn": 41219, "diverged": null, "last_error": null },
            "claude-code": { "applied_lsn": 1475, "leader_head_lsn": null, "diverged": null, "last_error": null },
            "admin": { "applied_lsn": null, "leader_head_lsn": null },
        }));
        let summary = summarize(&s).expect("following");
        assert_eq!(summary.behind, 1);
        assert_eq!(summary.max_lag, 22719);
        assert!(summary.first_pass);
        assert_eq!(
            tray_line(&[("redb-replica".into(), summary)]).as_deref(),
            Some("Replica redb-replica: 1 tenant behind, max lag 22,719 frames")
        );
    }

    #[test]
    fn caught_up_reads_in_sync_and_trouble_is_named() {
        let synced = summarize(&status(4, json!({
            "a": { "applied_lsn": 9, "leader_head_lsn": 10 },
        })))
        .expect("following");
        assert_eq!(tray_line(&[("r".into(), synced)]).as_deref(), Some("Replica r: in sync"));

        let troubled = summarize(&status(4, json!({
            "a": { "applied_lsn": 2, "leader_head_lsn": 10, "last_error": "page failed", "failures": 3 },
            "b": { "applied_lsn": 50, "leader_head_lsn": 10, "diverged": "leader log moved backwards" },
        })))
        .expect("following");
        assert_eq!(
            tray_line(&[("r".into(), troubled)]).as_deref(),
            Some("Replica r: 1 tenant behind, max lag 7 frames · 1 retrying · 1 diverged")
        );
    }

    #[test]
    fn several_followers_share_one_line() {
        let a = ReplicaSummary { behind: 2, max_lag: 40, ..Default::default() };
        let b = ReplicaSummary { behind: 1, max_lag: 900, leader_unreachable: true, ..Default::default() };
        assert_eq!(
            tray_line(&[("a".into(), a), ("b".into(), b)]).as_deref(),
            Some("2 replicas: leader unreachable · 3 tenants behind, max lag 900 frames")
        );
    }
}
