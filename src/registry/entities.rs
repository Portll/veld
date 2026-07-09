//! Registry entity types: the GitHub account → org → workspace → project → folder
//! graph the Veld Manager curates, plus per-key ACL grants. Values are JSON blobs
//! (like `KeyBinding` / `Project` elsewhere in the codebase).
//!
//! IMPORTANT (R2a reconciliation): the registry `scope` is the **flat veld
//! `user_id`** (dot-encoded `{boundary}.{org}.{repo}`, `validate_user_id`-valid) —
//! NOT `ScopePath`, which is an orthogonal *intra-tenant* memory-
//! attribution dimension (workspace/project/folder WITHIN one user's Earth). The
//! registry "workspace" groups whole tenants (`user_id`s); `ScopePath`'s
//! "workspace" partitions memories inside a single tenant. They coexist under
//! different names and never alias.

use serde::{Deserialize, Serialize};

/// Trust boundary of a scope/workspace. Stored per entity and enforced at link
/// write-time (single-boundary workspaces + edges) and re-checked at recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Boundary {
    Internal,
    External,
}

impl Boundary {
    /// The `Portll` org ⇒ internal; every other org ⇒ external.
    pub fn for_org(org: &str) -> Self {
        if org.eq_ignore_ascii_case("Portll") {
            Boundary::Internal
        } else {
            Boundary::External
        }
    }

    /// The 3-letter tag used as the leading segment of a derived scope id.
    pub fn tag(self) -> &'static str {
        match self {
            Boundary::Internal => "int",
            Boundary::External => "ext",
        }
    }
}

/// A GitHub account (a gh login such as `Portll` or `gourmetfire`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub login: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A GitHub org under an account (`Portll`, `Gourmet-Fire`, …).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Org {
    pub id: String,
    pub account_id: String,
    pub login: String,
    pub boundary: Boundary,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A cross-tenant workspace: a single-boundary grouping of projects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub org_id: String,
    pub name: String,
    pub boundary: Boundary,
    #[serde(default)]
    pub order: i64,
    #[serde(default)]
    pub pinned: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A project = a repo = a veld tenant scope (a flat `user_id`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub org_id: String,
    pub repo: String,
    pub boundary: Boundary,
    /// The flat veld `user_id`, e.g. `int.portll.veld` (see [`derive_scope`]).
    pub scope: String,
    #[serde(default)]
    pub order: i64,
    #[serde(default)]
    pub pinned: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A local folder / git worktree that maps to a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Folder {
    pub id: String,
    pub project_id: String,
    pub host: String,
    pub abs_path: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// An ACL grant: which `principal` may see which `scope`, with what role.
///
/// The `principal` is the tenant's authoritative veld `user_id` — the identity
/// veld's auth layer resolves for a caller (see
/// [`crate::handlers::utils::resolve_request_user_id`]). It is NOT a `key_hash`:
/// veld never exposes an API-key hash to a handler, so keying grants by
/// `key_hash` (as R3 originally did) could never be matched against a live
/// caller. Grants are keyed by `user_id` throughout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grant {
    /// The authorized tenant `user_id`.
    pub principal: String,
    pub scope: String,
    pub role: String,
    pub granted_at: chrono::DateTime<chrono::Utc>,
}

/// Derive the flat veld tenant `user_id` for a repo under an org:
/// `{boundary_tag}.{org}.{repo}`, lowercased with each segment sanitized to
/// `[a-z0-9_-]` (dots appear ONLY as the two separators), so the result is a
/// 3-segment id that passes `validate_user_id` and can never collide with a
/// `sys.*` special scope.
pub fn derive_scope(boundary: Boundary, org: &str, repo: &str) -> String {
    let seg = |s: &str| -> String {
        let cleaned: String = s
            .to_ascii_lowercase()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        // collapse a leading/trailing '-' so segments never start/end oddly
        cleaned.trim_matches('-').to_string()
    };
    format!("{}.{}.{}", boundary.tag(), seg(org), seg(repo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_derivation_is_user_id_valid_shape() {
        assert_eq!(
            derive_scope(Boundary::Internal, "Portll", "veld"),
            "int.portll.veld"
        );
        assert_eq!(
            derive_scope(Boundary::External, "Gourmet-Fire", "antidote-dev"),
            "ext.gourmet-fire.antidote-dev"
        );
        // exactly 3 dot-separated segments; leading segment ∈ {int, ext}
        let s = derive_scope(Boundary::External, "gourmetfire", "fond-blanc");
        assert_eq!(s.split('.').count(), 3);
        assert!(s.starts_with("ext."));
    }

    #[test]
    fn boundary_from_org() {
        assert_eq!(Boundary::for_org("Portll"), Boundary::Internal);
        assert_eq!(Boundary::for_org("portll"), Boundary::Internal);
        assert_eq!(Boundary::for_org("Gourmet-Fire"), Boundary::External);
    }
}
