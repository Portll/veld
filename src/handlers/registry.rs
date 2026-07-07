//! Registry HTTP endpoints for the account → org → workspace → project → folder
//! graph, in two tiers:
//!
//! - **Admin CRUD + ACL** (`/api/admin/registry/*`) — gated by veld's EXISTING
//!   admin-key mechanism ([`crate::handlers::admin::check_admin_key`] —
//!   `X-Admin-API-Key` header checked constant-time against `VELD_ADMIN_API_KEY`).
//! - **Tenant reads** (`/api/registry/tree`, `/api/registry/resolve`) — NOT
//!   admin-gated. They resolve the caller's authoritative `user_id` via veld's
//!   own [`crate::handlers::utils::resolve_request_user_id`] and ACL-filter the
//!   result to that tenant's [`authorized_scopes`]. Their protection is the
//!   tenant auth layer plus the ACL filter, never an admin key.
//!
//! This module does NOT introduce a new auth system.
//!
//! ## Atomicity + uniqueness
//!
//! Every multi-key mutation (entity + secondary indexes + reverse edges) is
//! applied through a SINGLE [`RegistryStore::write_batch`] so it commits
//! all-or-nothing. Uniqueness pre-checks (`scope2proj`, `wsname`) that must be
//! race-free hold [`REGISTRY_LOCK`] across the check-then-write, because RocksDB
//! has no unique index.
//!
//! ## Boundary invariants
//!
//! - **I1** (ws↔project link): a workspace may only link a project of the SAME
//!   [`Boundary`]. A cross-boundary link is rejected `409 boundary_violation`.
//! - **I6** (ACL grant): a single `principal` (tenant `user_id`) may not mix
//!   `int.*` and `ext.*` scope grants — the leading boundary tag of every grant
//!   a principal holds must be consistent. A mixing grant is rejected
//!   `409 boundary_violation`.
//!
//! Boundaries are always derived (`Boundary::for_org`) or inherited from the
//! parent org — never taken raw from an untrusted body for orgs/projects. A
//! workspace body MAY carry an explicit boundary, but it must equal the org's.

use std::collections::BTreeSet;

use axum::{
    body::Bytes,
    extract::{Extension, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;

use crate::auth::AuthenticatedUser;
use crate::handlers::admin::check_admin_key;
use crate::handlers::router::AppState;
use crate::handlers::utils::resolve_request_user_id;
use crate::registry::{
    derive_scope, Account, Boundary, Folder, Grant, Org, Project, RegistryStore, Workspace, CF_ACL,
    CF_ENTITIES, CF_INDEX, CF_LINKS, REGISTRY_LOCK,
};

// ===========================================================================
// Error plumbing
// ===========================================================================

/// A typed handler error that maps 1:1 to the documented `(status, {"error"})`
/// response shapes. Kept separate from `Response` so the pure logic functions
/// stay unit-testable without constructing HTTP responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
}

impl ApiError {
    fn bad_request() -> Self {
        Self { status: StatusCode::BAD_REQUEST, code: "bad_request" }
    }
    fn not_found() -> Self {
        Self { status: StatusCode::NOT_FOUND, code: "not_found" }
    }
    fn conflict() -> Self {
        Self { status: StatusCode::CONFLICT, code: "conflict" }
    }
    fn boundary_violation() -> Self {
        Self { status: StatusCode::CONFLICT, code: "boundary_violation" }
    }
    fn internal() -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, code: "internal_error" }
    }
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.code }))).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ===========================================================================
// Store helpers (shared by the logic functions)
// ===========================================================================

fn to_bytes<T: Serialize>(value: &T) -> ApiResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|_| ApiError::internal())
}

fn load_entity<T: DeserializeOwned>(
    store: &RegistryStore,
    cf: &str,
    key: &str,
) -> ApiResult<Option<T>> {
    match store.get(cf, key).map_err(|_| ApiError::internal())? {
        Some(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).map_err(|_| ApiError::internal())?,
        )),
        None => Ok(None),
    }
}

/// Trim and reject empty required string fields as `400 bad_request`.
fn require(value: &str) -> ApiResult<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(ApiError::bad_request())
    } else {
        Ok(trimmed.to_string())
    }
}

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ===========================================================================
// Request bodies
// ===========================================================================

#[derive(Debug, Deserialize)]
struct CreateAccountRequest {
    login: String,
}

#[derive(Debug, Deserialize)]
struct CreateOrgRequest {
    account_id: String,
    login: String,
}

#[derive(Debug, Deserialize)]
struct CreateWorkspaceRequest {
    org_id: String,
    name: String,
    #[serde(default)]
    boundary: Option<Boundary>,
}

#[derive(Debug, Deserialize)]
struct CreateProjectRequest {
    org_id: String,
    repo: String,
}

#[derive(Debug, Deserialize)]
struct CreateFolderRequest {
    project_id: String,
    host: String,
    abs_path: String,
}

#[derive(Debug, Deserialize)]
struct LinkWsProjectRequest {
    workspace_id: String,
    project_id: String,
}

#[derive(Debug, Deserialize)]
struct GrantRequest {
    /// The tenant `user_id` the grant authorizes.
    principal: String,
    scope: String,
    role: String,
}

#[derive(Debug, Deserialize)]
struct RevokeRequest {
    /// The tenant `user_id` whose grant is revoked.
    principal: String,
    scope: String,
}

/// Body for the tenant read endpoints. `user_id` is only consulted when the API
/// key is not tenant-bound (single-tenant / dev mode); under multi-tenant auth
/// the middleware injects it and [`resolve_request_user_id`] enforces the match.
#[derive(Debug, Default, Deserialize)]
struct TreeRequest {
    #[serde(default)]
    user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResolveRequest {
    git_remote: String,
    #[serde(default)]
    user_id: Option<String>,
}

// ===========================================================================
// Key builders (follow the schema documented in registry/store.rs)
// ===========================================================================

fn k_acct(id: &str) -> String {
    format!("acct:{id}")
}
fn k_org(id: &str) -> String {
    format!("org:{id}")
}
fn k_ws(id: &str) -> String {
    format!("ws:{id}")
}
fn k_proj(id: &str) -> String {
    format!("proj:{id}")
}
fn k_folder(id: &str) -> String {
    format!("folder:{id}")
}
fn k_wsname(org_id: &str, name: &str) -> String {
    format!("wsname:{org_id}:{name}")
}
fn k_scope2proj(scope: &str) -> String {
    format!("scope2proj:{scope}")
}
fn k_remote2scope(org_login: &str, repo: &str) -> String {
    format!("remote2scope:{org_login}/{repo}")
}
fn k_org_proj(org_id: &str, proj_id: &str) -> String {
    format!("org_proj:{org_id}:{proj_id}")
}
fn k_ws_proj(ws_id: &str, proj_id: &str) -> String {
    format!("ws_proj:{ws_id}:{proj_id}")
}
fn k_proj_ws(proj_id: &str, ws_id: &str) -> String {
    format!("proj_ws:{proj_id}:{ws_id}")
}
fn k_folder_proj(folder_id: &str, proj_id: &str) -> String {
    format!("folder_proj:{folder_id}:{proj_id}")
}
fn k_proj_folder(proj_id: &str, folder_id: &str) -> String {
    format!("proj_folder:{proj_id}:{folder_id}")
}
fn k_grant(principal: &str, scope: &str) -> String {
    format!("grant:{principal}:{scope}")
}
fn k_scopegrant(scope: &str, principal: &str) -> String {
    format!("scopegrant:{scope}:{principal}")
}

/// Leading boundary tag (`int` / `ext`) of a derived scope id.
fn scope_tag(scope: &str) -> &str {
    scope.split('.').next().unwrap_or("")
}

// ===========================================================================
// Core logic (pure over &RegistryStore — directly unit-testable)
// ===========================================================================

fn create_account_logic(store: &RegistryStore, login: &str) -> ApiResult<Account> {
    let login = require(login)?;
    let account = Account { id: new_id(), login, created_at: now() };
    let value = to_bytes(&account)?;
    store
        .write_batch(&[(CF_ENTITIES, k_acct(&account.id), value)], &[])
        .map_err(|_| ApiError::internal())?;
    Ok(account)
}

fn create_org_logic(store: &RegistryStore, account_id: &str, login: &str) -> ApiResult<Org> {
    let account_id = require(account_id)?;
    let login = require(login)?;

    // Hold the lock across the parent load → write_batch sequence.
    let _guard = REGISTRY_LOCK.lock();

    // Parent must exist.
    let _account: Account = load_entity(store, CF_ENTITIES, &k_acct(&account_id))?
        .ok_or_else(ApiError::not_found)?;

    // Boundary is DERIVED from the org login — never taken from the body.
    let boundary = Boundary::for_org(&login);
    let org = Org { id: new_id(), account_id, login, boundary, created_at: now() };
    let value = to_bytes(&org)?;
    store
        .write_batch(&[(CF_ENTITIES, k_org(&org.id), value)], &[])
        .map_err(|_| ApiError::internal())?;
    Ok(org)
}

fn create_workspace_logic(
    store: &RegistryStore,
    org_id: &str,
    name: &str,
    boundary: Option<Boundary>,
) -> ApiResult<Workspace> {
    let org_id = require(org_id)?;
    let name = require(name)?;

    let org: Org = load_entity(store, CF_ENTITIES, &k_org(&org_id))?
        .ok_or_else(ApiError::not_found)?;

    // A workspace inherits its org's boundary. An explicit body boundary is
    // allowed but MUST equal the org's, else it is a boundary violation.
    let boundary = match boundary {
        None => org.boundary,
        Some(b) if b == org.boundary => b,
        Some(_) => return Err(ApiError::boundary_violation()),
    };

    // Uniqueness of workspace name within the org: hold REGISTRY_LOCK across
    // the check-then-write so two concurrent creates cannot both pass.
    let _guard = REGISTRY_LOCK.lock();
    let name_key = k_wsname(&org_id, &name);
    if store.exists(CF_INDEX, &name_key).map_err(|_| ApiError::internal())? {
        return Err(ApiError::conflict());
    }

    let ws = Workspace {
        id: new_id(),
        org_id,
        name,
        boundary,
        order: 0,
        pinned: false,
        created_at: now(),
    };
    let entity = to_bytes(&ws)?;
    store
        .write_batch(
            &[
                (CF_ENTITIES, k_ws(&ws.id), entity),
                (CF_INDEX, name_key, ws.id.clone().into_bytes()),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(ws)
}

fn create_project_logic(store: &RegistryStore, org_id: &str, repo: &str) -> ApiResult<Project> {
    let org_id = require(org_id)?;
    let repo = require(repo)?;

    let org: Org = load_entity(store, CF_ENTITIES, &k_org(&org_id))?
        .ok_or_else(ApiError::not_found)?;

    // Boundary + scope are DERIVED from the org — never from the body.
    let boundary = org.boundary;
    let scope = derive_scope(boundary, &org.login, &repo);

    // Scope uniqueness: hold REGISTRY_LOCK across the check-then-write.
    let _guard = REGISTRY_LOCK.lock();
    let scope_key = k_scope2proj(&scope);
    if store.exists(CF_INDEX, &scope_key).map_err(|_| ApiError::internal())? {
        return Err(ApiError::conflict());
    }

    let project = Project {
        id: new_id(),
        org_id: org_id.clone(),
        repo: repo.clone(),
        boundary,
        scope: scope.clone(),
        order: 0,
        pinned: false,
        created_at: now(),
    };
    let entity = to_bytes(&project)?;
    store
        .write_batch(
            &[
                (CF_ENTITIES, k_proj(&project.id), entity),
                (CF_INDEX, scope_key, project.id.clone().into_bytes()),
                (
                    CF_INDEX,
                    k_remote2scope(&org.login, &repo),
                    scope.into_bytes(),
                ),
                (
                    CF_INDEX,
                    k_org_proj(&org_id, &project.id),
                    project.id.clone().into_bytes(),
                ),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(project)
}

fn create_folder_logic(
    store: &RegistryStore,
    project_id: &str,
    host: &str,
    abs_path: &str,
) -> ApiResult<Folder> {
    let project_id = require(project_id)?;
    let host = require(host)?;
    let abs_path = require(abs_path)?;

    // Hold the lock across the parent load → write_batch sequence.
    let _guard = REGISTRY_LOCK.lock();

    let _project: Project = load_entity(store, CF_ENTITIES, &k_proj(&project_id))?
        .ok_or_else(ApiError::not_found)?;

    let folder = Folder { id: new_id(), project_id: project_id.clone(), host, abs_path, created_at: now() };
    let entity = to_bytes(&folder)?;
    // Entity + both reverse edges in one batch so a scan for a project's
    // folders and vice-versa is always consistent with the entity.
    store
        .write_batch(
            &[
                (CF_ENTITIES, k_folder(&folder.id), entity),
                (
                    CF_LINKS,
                    k_folder_proj(&folder.id, &project_id),
                    project_id.clone().into_bytes(),
                ),
                (
                    CF_LINKS,
                    k_proj_folder(&project_id, &folder.id),
                    folder.id.clone().into_bytes(),
                ),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(folder)
}

/// INVARIANT I1: a workspace may only link a project of the SAME boundary.
fn link_ws_project_logic(
    store: &RegistryStore,
    workspace_id: &str,
    project_id: &str,
) -> ApiResult<serde_json::Value> {
    let workspace_id = require(workspace_id)?;
    let project_id = require(project_id)?;

    // Hold the lock across the load → write_batch sequence.
    let _guard = REGISTRY_LOCK.lock();

    let ws: Workspace = load_entity(store, CF_ENTITIES, &k_ws(&workspace_id))?
        .ok_or_else(ApiError::not_found)?;
    let project: Project = load_entity(store, CF_ENTITIES, &k_proj(&project_id))?
        .ok_or_else(ApiError::not_found)?;

    if ws.boundary != project.boundary {
        return Err(ApiError::boundary_violation());
    }

    // Both edges in ONE write batch.
    store
        .write_batch(
            &[
                (
                    CF_LINKS,
                    k_ws_proj(&workspace_id, &project_id),
                    project_id.clone().into_bytes(),
                ),
                (
                    CF_LINKS,
                    k_proj_ws(&project_id, &workspace_id),
                    workspace_id.clone().into_bytes(),
                ),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;

    Ok(json!({
        "workspace_id": workspace_id,
        "project_id": project_id,
        "boundary": ws.boundary,
        "linked": true,
    }))
}

/// INVARIANT I6: a single `principal` (tenant `user_id`) may not mix `int.*`
/// and `ext.*` grants.
fn grant_logic(
    store: &RegistryStore,
    principal: &str,
    scope: &str,
    role: &str,
) -> ApiResult<Grant> {
    let principal = require(principal)?;
    let scope = require(scope)?;
    let role = require(role)?;

    // The scope must carry a recognised leading boundary tag.
    let tag = scope_tag(&scope);
    if tag != "int" && tag != "ext" {
        return Err(ApiError::bad_request());
    }

    // Hold the lock across the consistency scan + write so a concurrent grant
    // of the opposite boundary cannot slip between check and commit.
    let _guard = REGISTRY_LOCK.lock();

    // I6 own-scope boundary: if the principal is itself a registered project
    // scope, `authorized_scopes` unions that own scope into the caller's
    // authorized set. Granting a scope of the opposite boundary would then
    // yield a mixed int/ext authorized set, so the principal's own tag must
    // match the grant's tag.
    if store
        .exists(CF_INDEX, &k_scope2proj(&principal))
        .map_err(|_| ApiError::internal())?
        && scope_tag(&principal) != tag
    {
        return Err(ApiError::boundary_violation());
    }

    let grant_prefix = format!("grant:{principal}:");
    for (existing_key, _) in store.scan_prefix(CF_ACL, &grant_prefix) {
        let existing_scope = existing_key.strip_prefix(&grant_prefix).unwrap_or("");
        if scope_tag(existing_scope) != tag {
            return Err(ApiError::boundary_violation());
        }
    }

    let grant = Grant { principal: principal.clone(), scope: scope.clone(), role, granted_at: now() };
    let value = to_bytes(&grant)?;
    store
        .write_batch(
            &[
                (CF_ACL, k_grant(&principal, &scope), value.clone()),
                (CF_ACL, k_scopegrant(&scope, &principal), value),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(grant)
}

fn revoke_logic(store: &RegistryStore, principal: &str, scope: &str) -> ApiResult<()> {
    let principal = require(principal)?;
    let scope = require(scope)?;

    let grant_key = k_grant(&principal, &scope);
    if !store.exists(CF_ACL, &grant_key).map_err(|_| ApiError::internal())? {
        return Err(ApiError::not_found());
    }

    // Delete both ACL keys in one batch.
    store
        .write_batch(
            &[],
            &[
                (CF_ACL, grant_key),
                (CF_ACL, k_scopegrant(&scope, &principal)),
            ],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(())
}

fn list_logic<T: DeserializeOwned>(
    store: &RegistryStore,
    cf: &str,
    prefix: &str,
) -> ApiResult<Vec<T>> {
    let mut out = Vec::new();
    for (_, bytes) in store.scan_prefix(cf, prefix) {
        let entity: T = serde_json::from_slice(&bytes).map_err(|_| ApiError::internal())?;
        out.push(entity);
    }
    Ok(out)
}

/// Delete an entity plus every index key and reverse edge it owns, in ONE
/// write batch. `kind` is the plural URL segment.
fn delete_logic(store: &RegistryStore, kind: &str, id: &str) -> ApiResult<()> {
    let id = require(id)?;

    // Hold the lock across the load/scan → write_batch sequence so an orphan
    // check cannot race a concurrent create of a child entity.
    let _guard = REGISTRY_LOCK.lock();

    let mut deletes: Vec<(&str, String)> = Vec::new();

    match kind {
        "accounts" => {
            let _acct: Account = load_entity(store, CF_ENTITIES, &k_acct(&id))?
                .ok_or_else(ApiError::not_found)?;
            // Reject if any org still points at this account.
            if list_logic::<Org>(store, CF_ENTITIES, "org:")?
                .iter()
                .any(|o| o.account_id == id)
            {
                return Err(ApiError::conflict());
            }
            deletes.push((CF_ENTITIES, k_acct(&id)));
        }
        "orgs" => {
            let _org: Org = load_entity(store, CF_ENTITIES, &k_org(&id))?
                .ok_or_else(ApiError::not_found)?;
            // Reject if the org still owns any project or workspace.
            if !store
                .scan_prefix(CF_INDEX, &format!("org_proj:{id}:"))
                .is_empty()
                || !store
                    .scan_prefix(CF_INDEX, &format!("wsname:{id}:"))
                    .is_empty()
            {
                return Err(ApiError::conflict());
            }
            deletes.push((CF_ENTITIES, k_org(&id)));
        }
        "workspaces" => {
            let ws: Workspace = load_entity(store, CF_ENTITIES, &k_ws(&id))?
                .ok_or_else(ApiError::not_found)?;
            deletes.push((CF_ENTITIES, k_ws(&id)));
            deletes.push((CF_INDEX, k_wsname(&ws.org_id, &ws.name)));
            // Reverse edges: every project this workspace links.
            for (edge_key, _) in store.scan_prefix(CF_LINKS, &format!("ws_proj:{id}:")) {
                if let Some(proj_id) = edge_key.rsplit(':').next() {
                    deletes.push((CF_LINKS, k_proj_ws(proj_id, &id)));
                }
                deletes.push((CF_LINKS, edge_key));
            }
        }
        "projects" => {
            let project: Project = load_entity(store, CF_ENTITIES, &k_proj(&id))?
                .ok_or_else(ApiError::not_found)?;
            deletes.push((CF_ENTITIES, k_proj(&id)));
            deletes.push((CF_INDEX, k_scope2proj(&project.scope)));
            deletes.push((CF_INDEX, k_org_proj(&project.org_id, &id)));
            // remote2scope is keyed by org LOGIN, which the project doesn't
            // store — resolve it via the parent org when still present.
            if let Some(org) =
                load_entity::<Org>(store, CF_ENTITIES, &k_org(&project.org_id))?
            {
                deletes.push((CF_INDEX, k_remote2scope(&org.login, &project.repo)));
            }
            // Reverse edges: workspace links.
            for (edge_key, _) in store.scan_prefix(CF_LINKS, &format!("proj_ws:{id}:")) {
                if let Some(ws_id) = edge_key.rsplit(':').next() {
                    deletes.push((CF_LINKS, k_ws_proj(ws_id, &id)));
                }
                deletes.push((CF_LINKS, edge_key));
            }
            // Reverse edges: folder links. Also delete the child FOLDER ENTITY
            // itself — the edges alone would otherwise orphan the `folder:` row.
            for (edge_key, _) in store.scan_prefix(CF_LINKS, &format!("proj_folder:{id}:")) {
                if let Some(folder_id) = edge_key.rsplit(':').next() {
                    deletes.push((CF_ENTITIES, k_folder(folder_id)));
                    deletes.push((CF_LINKS, k_folder_proj(folder_id, &id)));
                }
                deletes.push((CF_LINKS, edge_key));
            }
            // ACL grants naming this project's scope as TARGET
            // (`scopegrant:{scope}:{principal}` + mirror `grant:{principal}:{scope}`).
            // Without this a later project that re-derives the same scope would
            // inherit stale grants — a scope-reuse privilege hole.
            let scopegrant_prefix = format!("scopegrant:{}:", project.scope);
            for (key, _) in store.scan_prefix(CF_ACL, &scopegrant_prefix) {
                if let Some(principal) = key.strip_prefix(&scopegrant_prefix) {
                    deletes.push((CF_ACL, k_scopegrant(&project.scope, principal)));
                    deletes.push((CF_ACL, k_grant(principal, &project.scope)));
                }
            }
            // ACL grants where this project's scope is the PRINCIPAL
            // (`grant:{scope}:{target}` + mirror `scopegrant:{target}:{scope}`).
            let grant_prefix = format!("grant:{}:", project.scope);
            for (key, _) in store.scan_prefix(CF_ACL, &grant_prefix) {
                if let Some(target) = key.strip_prefix(&grant_prefix) {
                    deletes.push((CF_ACL, k_grant(&project.scope, target)));
                    deletes.push((CF_ACL, k_scopegrant(target, &project.scope)));
                }
            }
        }
        "folders" => {
            let folder: Folder = load_entity(store, CF_ENTITIES, &k_folder(&id))?
                .ok_or_else(ApiError::not_found)?;
            deletes.push((CF_ENTITIES, k_folder(&id)));
            deletes.push((CF_LINKS, k_folder_proj(&id, &folder.project_id)));
            deletes.push((CF_LINKS, k_proj_folder(&folder.project_id, &id)));
        }
        _ => return Err(ApiError::bad_request()),
    }

    store
        .write_batch(&[], &deletes)
        .map_err(|_| ApiError::internal())?;
    Ok(())
}

// ===========================================================================
// R4 — tenant read logic (ACL-filtered to the calling tenant's `user_id`)
// ===========================================================================

/// Resolve the complete set of scopes the tenant `caller_user_id` is authorized
/// to read.
///
/// Contract:
/// - The caller's OWN scope is included **iff** `caller_user_id` is itself a
///   registered project scope (looked up through the `scope2proj:` index). A
///   tenant is always authorized for their own scope; a `user_id` that is not a
///   registered project contributes no own-scope entry.
/// - PLUS every scope `S` for which an ACL grant `grant:{caller_user_id}:{S}`
///   exists (prefix scan on [`CF_ACL`]).
///
/// The returned set is exactly the anti-leak allow-list: no scope outside it may
/// ever be surfaced to this caller by a tenant endpoint.
fn authorized_scopes(store: &RegistryStore, caller_user_id: &str) -> BTreeSet<String> {
    let mut scopes = BTreeSet::new();

    // Own scope — only if it is a registered project scope.
    if store
        .exists(CF_INDEX, &k_scope2proj(caller_user_id))
        .unwrap_or(false)
    {
        scopes.insert(caller_user_id.to_string());
    }

    // Granted scopes.
    let prefix = format!("grant:{caller_user_id}:");
    for (key, _) in store.scan_prefix(CF_ACL, &prefix) {
        if let Some(scope) = key.strip_prefix(&prefix) {
            if !scope.is_empty() {
                scopes.insert(scope.to_string());
            }
        }
    }

    scopes
}

/// Build the registry graph pruned to `caller_user_id`'s [`authorized_scopes`].
///
/// Emitted as filtered entity lists (not a nested tree) plus the authorized set:
/// ```json
/// { "accounts": [...], "orgs": [...], "workspaces": [...],
///   "projects": [...], "folders": [...], "authorized_scopes": [...] }
/// ```
/// Pruning is bottom-up so no ancestor can leak a scope the caller may not see:
/// - a **project** is included iff its `scope` ∈ authorized;
/// - a **folder** iff its parent project is included;
/// - a **workspace** iff it links ≥1 included project (via `ws_proj:` edges);
/// - an **org** iff it owns ≥1 included project or included workspace;
/// - an **account** iff it owns ≥1 included org.
fn registry_tree_logic(
    store: &RegistryStore,
    caller_user_id: &str,
) -> ApiResult<serde_json::Value> {
    let authorized = authorized_scopes(store, caller_user_id);

    // Projects: authorized scopes only.
    let projects: Vec<Project> = list_logic::<Project>(store, CF_ENTITIES, "proj:")?
        .into_iter()
        .filter(|p| authorized.contains(&p.scope))
        .collect();
    let project_ids: BTreeSet<String> = projects.iter().map(|p| p.id.clone()).collect();

    // Folders of included projects.
    let folders: Vec<Folder> = list_logic::<Folder>(store, CF_ENTITIES, "folder:")?
        .into_iter()
        .filter(|f| project_ids.contains(&f.project_id))
        .collect();

    // Workspaces linking ≥1 included project.
    let workspaces: Vec<Workspace> = list_logic::<Workspace>(store, CF_ENTITIES, "ws:")?
        .into_iter()
        .filter(|ws| {
            store
                .scan_prefix(CF_LINKS, &format!("ws_proj:{}:", ws.id))
                .into_iter()
                .any(|(key, _)| {
                    key.rsplit(':')
                        .next()
                        .map(|proj_id| project_ids.contains(proj_id))
                        .unwrap_or(false)
                })
        })
        .collect();

    // Orgs owning an included project or workspace.
    let owning_org_ids: BTreeSet<String> = projects
        .iter()
        .map(|p| p.org_id.clone())
        .chain(workspaces.iter().map(|w| w.org_id.clone()))
        .collect();
    let orgs: Vec<Org> = list_logic::<Org>(store, CF_ENTITIES, "org:")?
        .into_iter()
        .filter(|o| owning_org_ids.contains(&o.id))
        .collect();

    // Accounts owning an included org.
    let owning_account_ids: BTreeSet<String> =
        orgs.iter().map(|o| o.account_id.clone()).collect();
    let accounts: Vec<Account> = list_logic::<Account>(store, CF_ENTITIES, "acct:")?
        .into_iter()
        .filter(|a| owning_account_ids.contains(&a.id))
        .collect();

    Ok(json!({
        "accounts": accounts,
        "orgs": orgs,
        "workspaces": workspaces,
        "projects": projects,
        "folders": folders,
        "authorized_scopes": authorized.into_iter().collect::<Vec<_>>(),
    }))
}

/// Extract the `owner/repo` slug from a git remote in any common form:
/// `https://host/owner/repo(.git)`, `ssh://…`, `git@host:owner/repo(.git)`, or
/// a bare `owner/repo`. Returns `None` when no two-segment slug can be isolated.
fn remote_slug(git_remote: &str) -> Option<String> {
    let trimmed = git_remote.trim().trim_end_matches('/');
    // URL forms carry a scheme; SCP form carries `user@host:`; bare is as-is.
    let tail = if let Some(idx) = trimmed.find("://") {
        let after_scheme = &trimmed[idx + 3..];
        after_scheme
            .split_once('/')
            .map(|(_host, rest)| rest)
            .unwrap_or(after_scheme)
    } else if let Some((_userhost, rest)) = trimmed.split_once('@') {
        rest.split_once(':').map(|(_host, r)| r).unwrap_or(rest)
    } else {
        trimmed
    };
    let tail = tail.strip_suffix(".git").unwrap_or(tail);
    let parts: Vec<&str> = tail.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    Some(format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]))
}

/// Resolve a git remote to its project scope via the `remote2scope:` index,
/// falling back to reconstructing `{org.login}/{repo}` per project (case-
/// insensitively) when the exact index key does not hit. `None` = unknown remote.
fn resolve_remote_to_scope(
    store: &RegistryStore,
    git_remote: &str,
) -> ApiResult<Option<String>> {
    let Some(slug) = remote_slug(git_remote) else {
        return Ok(None);
    };

    // Primary: exact `remote2scope:{org_login}/{repo}` hit.
    if let Some(bytes) = store
        .get(CF_INDEX, &format!("remote2scope:{slug}"))
        .map_err(|_| ApiError::internal())?
    {
        return Ok(Some(String::from_utf8_lossy(&bytes).to_string()));
    }

    // Fallback: match a project's stored remote, case-insensitively.
    let slug_lower = slug.to_ascii_lowercase();
    for project in list_logic::<Project>(store, CF_ENTITIES, "proj:")? {
        if let Some(org) = load_entity::<Org>(store, CF_ENTITIES, &k_org(&project.org_id))? {
            let candidate = format!("{}/{}", org.login, project.repo).to_ascii_lowercase();
            if candidate == slug_lower {
                return Ok(Some(project.scope));
            }
        }
    }

    Ok(None)
}

/// `{ scope, authorized }` for a git remote, from the caller's viewpoint.
fn registry_resolve_logic(
    store: &RegistryStore,
    caller_user_id: &str,
    git_remote: &str,
) -> ApiResult<serde_json::Value> {
    let git_remote = require(git_remote)?;
    let scope = resolve_remote_to_scope(store, &git_remote)?;
    let authorized = scope
        .as_ref()
        .map(|s| authorized_scopes(store, caller_user_id).contains(s))
        .unwrap_or(false);
    // Do NOT disclose the resolved scope to an unauthorized caller.
    let disclosed = if authorized { scope } else { None };
    Ok(json!({ "scope": disclosed, "authorized": authorized }))
}

// ===========================================================================
// HTTP handlers (admin-gated thin wrappers over the logic functions)
// ===========================================================================

/// The `Err` payload is a fully-formed `Response` (a `400 bad_request`) that
/// the caller returns verbatim; the large `Err` variant is intentional.
#[allow(clippy::result_large_err)]
fn parse_body<T: DeserializeOwned>(body: &Bytes) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(|_| ApiError::bad_request().into_response())
}

fn created<T: Serialize>(entity: T) -> Response {
    (StatusCode::CREATED, Json(entity)).into_response()
}

fn ok_json<T: Serialize>(value: T) -> Response {
    (StatusCode::OK, Json(value)).into_response()
}

macro_rules! gate {
    ($headers:expr) => {
        if let Err(resp) = check_admin_key(&$headers) {
            return resp;
        }
    };
}

pub async fn create_account(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: CreateAccountRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match create_account_logic(state.registry_store(), &req.login) {
        Ok(entity) => created(entity),
        Err(e) => e.into_response(),
    }
}

pub async fn create_org(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: CreateOrgRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match create_org_logic(state.registry_store(), &req.account_id, &req.login) {
        Ok(entity) => created(entity),
        Err(e) => e.into_response(),
    }
}

pub async fn create_workspace(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: CreateWorkspaceRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match create_workspace_logic(state.registry_store(), &req.org_id, &req.name, req.boundary) {
        Ok(entity) => created(entity),
        Err(e) => e.into_response(),
    }
}

pub async fn create_project(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: CreateProjectRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match create_project_logic(state.registry_store(), &req.org_id, &req.repo) {
        Ok(entity) => created(entity),
        Err(e) => e.into_response(),
    }
}

pub async fn create_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: CreateFolderRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match create_folder_logic(state.registry_store(), &req.project_id, &req.host, &req.abs_path) {
        Ok(entity) => created(entity),
        Err(e) => e.into_response(),
    }
}

pub async fn link_ws_project(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: LinkWsProjectRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match link_ws_project_logic(state.registry_store(), &req.workspace_id, &req.project_id) {
        Ok(value) => created(value),
        Err(e) => e.into_response(),
    }
}

pub async fn acl_grant(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: GrantRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match grant_logic(state.registry_store(), &req.principal, &req.scope, &req.role) {
        Ok(grant) => created(grant),
        Err(e) => e.into_response(),
    }
}

pub async fn acl_revoke(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gate!(headers);
    let req: RevokeRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    match revoke_logic(state.registry_store(), &req.principal, &req.scope) {
        Ok(()) => ok_json(json!({ "revoked": true })),
        Err(e) => e.into_response(),
    }
}

pub async fn list_accounts(State(state): State<AppState>, headers: HeaderMap) -> Response {
    gate!(headers);
    match list_logic::<Account>(state.registry_store(), CF_ENTITIES, "acct:") {
        Ok(list) => ok_json(list),
        Err(e) => e.into_response(),
    }
}

pub async fn list_orgs(State(state): State<AppState>, headers: HeaderMap) -> Response {
    gate!(headers);
    match list_logic::<Org>(state.registry_store(), CF_ENTITIES, "org:") {
        Ok(list) => ok_json(list),
        Err(e) => e.into_response(),
    }
}

pub async fn list_workspaces(State(state): State<AppState>, headers: HeaderMap) -> Response {
    gate!(headers);
    match list_logic::<Workspace>(state.registry_store(), CF_ENTITIES, "ws:") {
        Ok(list) => ok_json(list),
        Err(e) => e.into_response(),
    }
}

pub async fn list_projects(State(state): State<AppState>, headers: HeaderMap) -> Response {
    gate!(headers);
    match list_logic::<Project>(state.registry_store(), CF_ENTITIES, "proj:") {
        Ok(list) => ok_json(list),
        Err(e) => e.into_response(),
    }
}

pub async fn list_folders(State(state): State<AppState>, headers: HeaderMap) -> Response {
    gate!(headers);
    match list_logic::<Folder>(state.registry_store(), CF_ENTITIES, "folder:") {
        Ok(list) => ok_json(list),
        Err(e) => e.into_response(),
    }
}

pub async fn delete_entity(
    State(state): State<AppState>,
    Path((kind, id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    gate!(headers);
    match delete_logic(state.registry_store(), &kind, &id) {
        Ok(()) => ok_json(json!({ "deleted": true })),
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// R4 — tenant HTTP handlers (NOT admin-gated). The caller's authoritative
// `user_id` is resolved via veld's own `resolve_request_user_id`, which fails
// closed (400 INVALID_INPUT) when neither a tenant binding nor a body `user_id`
// is present — so under multi-tenant auth an unauthenticated caller is rejected.
// ===========================================================================

/// `POST /api/registry/tree` — the registry graph pruned to the caller's
/// authorized scopes. Body `{}` (or an optional `user_id` for non-bound keys).
pub async fn registry_tree(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    body: Bytes,
) -> Response {
    let req: TreeRequest = if body.is_empty() {
        TreeRequest::default()
    } else {
        match parse_body(&body) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    let caller = match resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|extension| &extension.0),
    ) {
        Ok(id) => id,
        Err(e) => return e.into_response(),
    };

    match registry_tree_logic(state.registry_store(), &caller) {
        Ok(value) => ok_json(value),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/registry/resolve` — `{ git_remote }` → `{ scope, authorized }`
/// where `authorized = scope ∈ authorized_scopes(caller)`.
pub async fn registry_resolve(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    body: Bytes,
) -> Response {
    let req: ResolveRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let caller = match resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|extension| &extension.0),
    ) {
        Ok(id) => id,
        Err(e) => return e.into_response(),
    };

    match registry_resolve_logic(state.registry_store(), &caller, &req.git_remote) {
        Ok(value) => ok_json(value),
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// Tests — RegistryStore + logic level (temp RocksDB), covering the headline
// invariants. HTTP wiring is exercised implicitly through the same logic
// functions the handlers call.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn temp_store() -> (RegistryStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let mut cfs = vec![rocksdb::ColumnFamilyDescriptor::new(
            "default",
            rocksdb::Options::default(),
        )];
        cfs.extend(RegistryStore::cf_descriptors());
        let db = rocksdb::DB::open_cf_descriptors(&opts, dir.path(), cfs).unwrap();
        (RegistryStore::new(Arc::new(db)), dir)
    }

    /// Build account → external org → workspace + project fixtures.
    fn seed_external(store: &RegistryStore) -> (Org, Workspace, Project) {
        let acct = create_account_logic(store, "gourmetfire").unwrap();
        let org = create_org_logic(store, &acct.id, "Gourmet-Fire").unwrap();
        assert_eq!(org.boundary, Boundary::External);
        let ws = create_workspace_logic(store, &org.id, "clients", None).unwrap();
        let project = create_project_logic(store, &org.id, "antidote-dev").unwrap();
        (org, ws, project)
    }

    #[test]
    fn cross_boundary_ws_project_link_rejected_409() {
        let (store, _dir) = temp_store();
        // External workspace.
        let (_org, ext_ws, _ext_proj) = seed_external(&store);
        // Internal org + project.
        let acct = create_account_logic(&store, "portll-owner").unwrap();
        let int_org = create_org_logic(&store, &acct.id, "Portll").unwrap();
        assert_eq!(int_org.boundary, Boundary::Internal);
        let int_proj = create_project_logic(&store, &int_org.id, "veld").unwrap();
        assert_eq!(int_proj.scope, "int.portll.veld");

        // Linking an external workspace to an internal project must be rejected.
        let err = link_ws_project_logic(&store, &ext_ws.id, &int_proj.id).unwrap_err();
        assert_eq!(err, ApiError::boundary_violation());
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "boundary_violation");
    }

    #[test]
    fn same_boundary_ws_project_link_succeeds() {
        let (store, _dir) = temp_store();
        let (_org, ws, project) = seed_external(&store);
        let link = link_ws_project_logic(&store, &ws.id, &project.id).unwrap();
        assert_eq!(link["linked"], json!(true));
        // Both edges persisted.
        assert!(store.exists(CF_LINKS, &k_ws_proj(&ws.id, &project.id)).unwrap());
        assert!(store.exists(CF_LINKS, &k_proj_ws(&project.id, &ws.id)).unwrap());
    }

    #[test]
    fn duplicate_scope_rejected_409() {
        let (store, _dir) = temp_store();
        let (org, _ws, _project) = seed_external(&store);
        // Same org + repo derives the same scope → conflict.
        let err = create_project_logic(&store, &org.id, "antidote-dev").unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "conflict");
    }

    #[test]
    fn duplicate_workspace_name_rejected_409() {
        let (store, _dir) = temp_store();
        let (org, _ws, _project) = seed_external(&store);
        let err = create_workspace_logic(&store, &org.id, "clients", None).unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "conflict");
    }

    #[test]
    fn workspace_explicit_mismatched_boundary_rejected() {
        let (store, _dir) = temp_store();
        let (org, _ws, _project) = seed_external(&store);
        // org is External; asking for an Internal workspace under it violates.
        let err =
            create_workspace_logic(&store, &org.id, "misboundaried", Some(Boundary::Internal))
                .unwrap_err();
        assert_eq!(err.code, "boundary_violation");
    }

    #[test]
    fn create_list_delete_roundtrip() {
        let (store, _dir) = temp_store();
        let (org, ws, project) = seed_external(&store);
        let folder =
            create_folder_logic(&store, &project.id, "macbook", "/Users/x/repo").unwrap();

        // list
        assert_eq!(list_logic::<Account>(&store, CF_ENTITIES, "acct:").unwrap().len(), 1);
        assert_eq!(list_logic::<Org>(&store, CF_ENTITIES, "org:").unwrap().len(), 1);
        assert_eq!(list_logic::<Workspace>(&store, CF_ENTITIES, "ws:").unwrap().len(), 1);
        assert_eq!(list_logic::<Project>(&store, CF_ENTITIES, "proj:").unwrap().len(), 1);
        assert_eq!(list_logic::<Folder>(&store, CF_ENTITIES, "folder:").unwrap().len(), 1);

        // link so the project carries a reverse edge to clean on delete
        link_ws_project_logic(&store, &ws.id, &project.id).unwrap();

        // delete project → entity, indexes and reverse edges all gone
        delete_logic(&store, "projects", &project.id).unwrap();
        assert!(!store.exists(CF_ENTITIES, &k_proj(&project.id)).unwrap());
        assert!(!store.exists(CF_INDEX, &k_scope2proj(&project.scope)).unwrap());
        assert!(!store.exists(CF_INDEX, &k_org_proj(&org.id, &project.id)).unwrap());
        assert!(!store.exists(CF_INDEX, &k_remote2scope(&org.login, &project.repo)).unwrap());
        assert!(!store.exists(CF_LINKS, &k_ws_proj(&ws.id, &project.id)).unwrap());
        assert!(!store.exists(CF_LINKS, &k_proj_ws(&project.id, &ws.id)).unwrap());
        assert!(!store.exists(CF_LINKS, &k_proj_folder(&project.id, &folder.id)).unwrap());
        assert_eq!(list_logic::<Project>(&store, CF_ENTITIES, "proj:").unwrap().len(), 0);

        // The child folder ENTITY is cascade-deleted with its project (F2c) —
        // no longer orphaned, so it is already gone here.
        assert!(!store.exists(CF_ENTITIES, &k_folder(&folder.id)).unwrap());
        assert_eq!(list_logic::<Folder>(&store, CF_ENTITIES, "folder:").unwrap().len(), 0);

        // deleting a missing entity → not_found
        let err = delete_logic(&store, "projects", &project.id).unwrap_err();
        assert_eq!(err.code, "not_found");
    }

    #[test]
    fn acl_grant_and_revoke_roundtrip() {
        // The principal is a tenant `user_id` (a scope id), NOT a key_hash.
        let (store, _dir) = temp_store();
        let principal = "ext.acme.client-a";
        let scope = "ext.gourmet-fire.antidote-dev";
        let grant = grant_logic(&store, principal, scope, "reader").unwrap();
        assert_eq!(grant.principal, principal);
        assert_eq!(grant.role, "reader");
        assert!(store.exists(CF_ACL, &k_grant(principal, scope)).unwrap());
        assert!(store.exists(CF_ACL, &k_scopegrant(scope, principal)).unwrap());

        revoke_logic(&store, principal, scope).unwrap();
        assert!(!store.exists(CF_ACL, &k_grant(principal, scope)).unwrap());
        assert!(!store.exists(CF_ACL, &k_scopegrant(scope, principal)).unwrap());

        // revoking a missing grant → not_found
        let err = revoke_logic(&store, principal, scope).unwrap_err();
        assert_eq!(err.code, "not_found");
    }

    #[test]
    fn acl_grant_mixed_boundary_rejected_409() {
        let (store, _dir) = temp_store();
        // A single principal (tenant user_id) may not hold int.* and ext.* grants.
        let principal = "int.portll.veld";
        grant_logic(&store, principal, "int.portll.veld", "admin").unwrap();
        let err = grant_logic(&store, principal, "ext.gourmet-fire.antidote-dev", "reader")
            .unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "boundary_violation");
    }

    #[test]
    fn acl_grant_bad_scope_tag_rejected_400() {
        let (store, _dir) = temp_store();
        let err = grant_logic(&store, "ext.acme.client-a", "bogus.scope.here", "reader")
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    // ── R4: authorized_scopes + tenant tree/resolve ──

    #[test]
    fn authorized_scopes_returns_own_and_granted_only() {
        let (store, _dir) = temp_store();
        let (_org, _ws, project) = seed_external(&store);
        let own = project.scope.clone(); // ext.gourmet-fire.antidote-dev

        // A second project → a second scope the caller does NOT own.
        let other = create_project_logic(&store, &_org.id, "fond-blanc").unwrap();

        // Caller = own scope: authorized for own, not the other.
        let scopes = authorized_scopes(&store, &own);
        assert!(scopes.contains(&own));
        assert!(!scopes.contains(&other.scope));

        // Grant the other scope to the caller principal.
        grant_logic(&store, &own, &other.scope, "reader").unwrap();
        let scopes = authorized_scopes(&store, &own);
        assert!(scopes.contains(&own));
        assert!(scopes.contains(&other.scope));

        // A principal that is neither a registered scope nor granted → empty.
        assert!(authorized_scopes(&store, "ext.nobody.nothing").is_empty());
    }

    #[test]
    fn tree_prunes_to_authorized_scopes() {
        let (store, _dir) = temp_store();
        let acct = create_account_logic(&store, "gourmetfire").unwrap();
        let org = create_org_logic(&store, &acct.id, "Gourmet-Fire").unwrap();
        let ws = create_workspace_logic(&store, &org.id, "clients", None).unwrap();
        let proj_a = create_project_logic(&store, &org.id, "antidote-dev").unwrap();
        let proj_b = create_project_logic(&store, &org.id, "fond-blanc").unwrap();
        link_ws_project_logic(&store, &ws.id, &proj_a.id).unwrap();

        let scope_ids = |v: &serde_json::Value| -> Vec<String> {
            v["projects"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["scope"].as_str().unwrap().to_string())
                .collect()
        };

        // Principal A owns proj_a's scope → sees A, never B.
        let tree_a = registry_tree_logic(&store, &proj_a.scope).unwrap();
        let a_scopes = scope_ids(&tree_a);
        assert!(a_scopes.contains(&proj_a.scope));
        assert!(!a_scopes.contains(&proj_b.scope));
        // Included project pulls in its account/org/workspace ancestors.
        assert_eq!(tree_a["accounts"].as_array().unwrap().len(), 1);
        assert_eq!(tree_a["orgs"].as_array().unwrap().len(), 1);
        assert_eq!(tree_a["workspaces"].as_array().unwrap().len(), 1);

        // Principal B owns proj_b's scope → sees B, never A. proj_a granted to A
        // stays invisible to B (structural anti-leak).
        let tree_b = registry_tree_logic(&store, &proj_b.scope).unwrap();
        let b_scopes = scope_ids(&tree_b);
        assert!(b_scopes.contains(&proj_b.scope));
        assert!(!b_scopes.contains(&proj_a.scope));
        // proj_b is not linked to any workspace → no workspace surfaces for B.
        assert_eq!(tree_b["workspaces"].as_array().unwrap().len(), 0);

        // Grant proj_b's scope to A → A now sees both.
        grant_logic(&store, &proj_a.scope, &proj_b.scope, "reader").unwrap();
        let tree_a2 = registry_tree_logic(&store, &proj_a.scope).unwrap();
        let a2 = scope_ids(&tree_a2);
        assert!(a2.contains(&proj_a.scope));
        assert!(a2.contains(&proj_b.scope));
    }

    #[test]
    fn resolve_returns_scope_and_authorized() {
        let (store, _dir) = temp_store();
        let (org, _ws, project) = seed_external(&store); // Gourmet-Fire/antidote-dev
        let remote = format!("{}/{}", org.login, project.repo);

        // Own scope → authorized true, scope resolved.
        let own = registry_resolve_logic(&store, &project.scope, &remote).unwrap();
        assert_eq!(own["scope"], json!(project.scope));
        assert_eq!(own["authorized"], json!(true));

        // Different (un-granted) tenant → scope withheld (null), authorized false.
        let other = registry_resolve_logic(&store, "ext.someone.else", &remote).unwrap();
        assert_eq!(other["scope"], json!(null));
        assert_eq!(other["authorized"], json!(false));

        // Granting the scope flips authorized to true for that principal.
        grant_logic(&store, "ext.someone.else", &project.scope, "reader").unwrap();
        let granted = registry_resolve_logic(&store, "ext.someone.else", &remote).unwrap();
        assert_eq!(granted["authorized"], json!(true));

        // Full URL + .git form resolves to the same scope.
        let via_url = registry_resolve_logic(
            &store,
            &project.scope,
            "https://github.com/Gourmet-Fire/antidote-dev.git",
        )
        .unwrap();
        assert_eq!(via_url["scope"], json!(project.scope));
        assert_eq!(via_url["authorized"], json!(true));

        // Unknown remote → {null, false}.
        let unknown = registry_resolve_logic(&store, &project.scope, "no/such-repo").unwrap();
        assert_eq!(unknown["scope"], json!(null));
        assert_eq!(unknown["authorized"], json!(false));
    }

    // ── F1: grant must honour the principal's OWN-SCOPE boundary ──

    #[test]
    fn grant_own_scope_opposite_boundary_rejected_409() {
        let (store, _dir) = temp_store();
        // An internal project whose scope will be used as the principal.
        let acct = create_account_logic(&store, "portll-owner").unwrap();
        let int_org = create_org_logic(&store, &acct.id, "Portll").unwrap();
        let int_proj = create_project_logic(&store, &int_org.id, "veld").unwrap();
        assert_eq!(int_proj.scope, "int.portll.veld");

        // Granting an ext.* scope to a principal that is itself an int.* project
        // scope would mix boundaries in authorized_scopes → boundary_violation.
        let err = grant_logic(&store, &int_proj.scope, "ext.a.b", "reader").unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "boundary_violation");

        // A same-boundary grant to the same principal is allowed.
        grant_logic(&store, &int_proj.scope, "int.c.d", "reader").unwrap();
        assert!(authorized_scopes(&store, &int_proj.scope).contains("int.c.d"));
    }

    // ── F2: deletes must not orphan children / stale grants ──

    #[test]
    fn delete_account_with_child_org_rejected_409() {
        let (store, _dir) = temp_store();
        let (org, _ws, _project) = seed_external(&store);
        let err = delete_logic(&store, "accounts", &org.account_id).unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "conflict");
    }

    #[test]
    fn delete_org_with_child_project_rejected_409() {
        let (store, _dir) = temp_store();
        // seed_external gives the org a workspace AND a project.
        let (org, _ws, _project) = seed_external(&store);
        let err = delete_logic(&store, "orgs", &org.id).unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "conflict");
    }

    #[test]
    fn delete_project_removes_folder_entity_and_inbound_grant() {
        let (store, _dir) = temp_store();
        let (_org, _ws, project) = seed_external(&store);
        let folder =
            create_folder_logic(&store, &project.id, "macbook", "/Users/x/repo").unwrap();

        // An external principal is granted read on this project's scope.
        let principal = "ext.acme.client-a";
        grant_logic(&store, principal, &project.scope, "reader").unwrap();
        assert!(authorized_scopes(&store, principal).contains(&project.scope));

        delete_logic(&store, "projects", &project.id).unwrap();

        // Folder entity itself gone (not just its edges).
        assert!(!store.exists(CF_ENTITIES, &k_folder(&folder.id)).unwrap());
        assert_eq!(list_logic::<Folder>(&store, CF_ENTITIES, "folder:").unwrap().len(), 0);

        // Inbound grant gone → the scope-reuse privilege hole is closed: a later
        // project re-deriving this scope cannot inherit the stale grant.
        assert!(!authorized_scopes(&store, principal).contains(&project.scope));
        assert!(!store.exists(CF_ACL, &k_grant(principal, &project.scope)).unwrap());
        assert!(!store.exists(CF_ACL, &k_scopegrant(&project.scope, principal)).unwrap());
    }

    // ── F3: resolve must not disclose scope to an unauthorized caller ──

    #[test]
    fn resolve_withholds_scope_from_unauthorized_caller() {
        let (store, _dir) = temp_store();
        let (org, _ws, project) = seed_external(&store);
        let remote = format!("{}/{}", org.login, project.repo);

        // Unauthorized caller: scope must be null, authorized false.
        let out = registry_resolve_logic(&store, "ext.someone.else", &remote).unwrap();
        assert_eq!(out["scope"], json!(null));
        assert_eq!(out["authorized"], json!(false));

        // Authorized caller (owner) still sees the scope.
        let owner = registry_resolve_logic(&store, &project.scope, &remote).unwrap();
        assert_eq!(owner["scope"], json!(project.scope));
        assert_eq!(owner["authorized"], json!(true));
    }

    // ── F4: guarded mutation paths serialize without self-deadlock ──

    #[test]
    fn concurrent_guarded_creates_do_not_deadlock() {
        use std::thread;
        let (store, _dir) = temp_store();
        let store = Arc::new(store);
        let acct = create_account_logic(&store, "gourmetfire").unwrap();
        let org = create_org_logic(&store, &acct.id, "Gourmet-Fire").unwrap();
        let project = create_project_logic(&store, &org.id, "antidote-dev").unwrap();

        let mut handles = Vec::new();
        for i in 0..8 {
            let s = Arc::clone(&store);
            let acct_id = acct.id.clone();
            let proj_id = project.id.clone();
            handles.push(thread::spawn(move || {
                // Both paths are newly REGISTRY_LOCK-guarded (F4); hammering them
                // concurrently must serialize cleanly with no reentrant deadlock.
                create_org_logic(&s, &acct_id, &format!("Org-{i}")).unwrap();
                create_folder_logic(&s, &proj_id, "host", &format!("/p/{i}")).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // 1 seed org + 8 concurrent orgs; 8 concurrent folders — all committed.
        assert_eq!(list_logic::<Org>(&store, CF_ENTITIES, "org:").unwrap().len(), 9);
        assert_eq!(list_logic::<Folder>(&store, CF_ENTITIES, "folder:").unwrap().len(), 8);
    }

    #[test]
    fn create_org_unknown_account_404() {
        let (store, _dir) = temp_store();
        let err = create_org_logic(&store, "no-such-account", "Portll").unwrap_err();
        assert_eq!(err.code, "not_found");
    }

    #[test]
    fn empty_required_field_400() {
        let (store, _dir) = temp_store();
        let err = create_account_logic(&store, "   ").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }
}
