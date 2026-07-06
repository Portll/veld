//! Admin CRUD + ACL + boundary-invariant HTTP endpoints for the registry graph
//! (account → org → workspace → project → folder, plus per-key ACL grants).
//!
//! Every endpoint lives under `/api/admin/registry` and is gated by veld's
//! EXISTING admin-key mechanism ([`crate::handlers::admin::check_admin_key`] —
//! `X-Admin-API-Key` header checked constant-time against `VELD_ADMIN_API_KEY`).
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
//! - **I6** (ACL grant): a single `key_hash` may not mix `int.*` and `ext.*`
//!   scope grants — the leading boundary tag of every grant a principal holds
//!   must be consistent. A mixing grant is rejected `409 boundary_violation`.
//!
//! Boundaries are always derived (`Boundary::for_org`) or inherited from the
//! parent org — never taken raw from an untrusted body for orgs/projects. A
//! workspace body MAY carry an explicit boundary, but it must equal the org's.

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;

use crate::handlers::admin::check_admin_key;
use crate::handlers::router::AppState;
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
    key_hash: String,
    scope: String,
    role: String,
}

#[derive(Debug, Deserialize)]
struct RevokeRequest {
    key_hash: String,
    scope: String,
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
fn k_grant(key_hash: &str, scope: &str) -> String {
    format!("grant:{key_hash}:{scope}")
}
fn k_scopegrant(scope: &str, key_hash: &str) -> String {
    format!("scopegrant:{scope}:{key_hash}")
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

/// INVARIANT I6: a single `key_hash` may not mix `int.*` and `ext.*` grants.
fn grant_logic(
    store: &RegistryStore,
    key_hash: &str,
    scope: &str,
    role: &str,
) -> ApiResult<Grant> {
    let key_hash = require(key_hash)?;
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

    let grant_prefix = format!("grant:{key_hash}:");
    for (existing_key, _) in store.scan_prefix(CF_ACL, &grant_prefix) {
        let existing_scope = existing_key.strip_prefix(&grant_prefix).unwrap_or("");
        if scope_tag(existing_scope) != tag {
            return Err(ApiError::boundary_violation());
        }
    }

    let grant = Grant { key_hash: key_hash.clone(), scope: scope.clone(), role, granted_at: now() };
    let value = to_bytes(&grant)?;
    store
        .write_batch(
            &[
                (CF_ACL, k_grant(&key_hash, &scope), value.clone()),
                (CF_ACL, k_scopegrant(&scope, &key_hash), value),
            ],
            &[],
        )
        .map_err(|_| ApiError::internal())?;
    Ok(grant)
}

fn revoke_logic(store: &RegistryStore, key_hash: &str, scope: &str) -> ApiResult<()> {
    let key_hash = require(key_hash)?;
    let scope = require(scope)?;

    let grant_key = k_grant(&key_hash, &scope);
    if !store.exists(CF_ACL, &grant_key).map_err(|_| ApiError::internal())? {
        return Err(ApiError::not_found());
    }

    // Delete both ACL keys in one batch.
    store
        .write_batch(
            &[],
            &[
                (CF_ACL, grant_key),
                (CF_ACL, k_scopegrant(&scope, &key_hash)),
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
    let mut deletes: Vec<(&str, String)> = Vec::new();

    match kind {
        "accounts" => {
            let _acct: Account = load_entity(store, CF_ENTITIES, &k_acct(&id))?
                .ok_or_else(ApiError::not_found)?;
            deletes.push((CF_ENTITIES, k_acct(&id)));
        }
        "orgs" => {
            let _org: Org = load_entity(store, CF_ENTITIES, &k_org(&id))?
                .ok_or_else(ApiError::not_found)?;
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
            // Reverse edges: folder links.
            for (edge_key, _) in store.scan_prefix(CF_LINKS, &format!("proj_folder:{id}:")) {
                if let Some(folder_id) = edge_key.rsplit(':').next() {
                    deletes.push((CF_LINKS, k_folder_proj(folder_id, &id)));
                }
                deletes.push((CF_LINKS, edge_key));
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
    match grant_logic(state.registry_store(), &req.key_hash, &req.scope, &req.role) {
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
    match revoke_logic(state.registry_store(), &req.key_hash, &req.scope) {
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

        // delete folder
        delete_logic(&store, "folders", &folder.id).unwrap();
        assert!(!store.exists(CF_ENTITIES, &k_folder(&folder.id)).unwrap());

        // deleting a missing entity → not_found
        let err = delete_logic(&store, "projects", &project.id).unwrap_err();
        assert_eq!(err.code, "not_found");
    }

    #[test]
    fn acl_grant_and_revoke_roundtrip() {
        let (store, _dir) = temp_store();
        let grant = grant_logic(&store, "abc123", "ext.gourmet-fire.antidote-dev", "reader")
            .unwrap();
        assert_eq!(grant.role, "reader");
        assert!(store.exists(CF_ACL, &k_grant("abc123", "ext.gourmet-fire.antidote-dev")).unwrap());
        assert!(store.exists(CF_ACL, &k_scopegrant("ext.gourmet-fire.antidote-dev", "abc123")).unwrap());

        revoke_logic(&store, "abc123", "ext.gourmet-fire.antidote-dev").unwrap();
        assert!(!store.exists(CF_ACL, &k_grant("abc123", "ext.gourmet-fire.antidote-dev")).unwrap());
        assert!(!store.exists(CF_ACL, &k_scopegrant("ext.gourmet-fire.antidote-dev", "abc123")).unwrap());

        // revoking a missing grant → not_found
        let err = revoke_logic(&store, "abc123", "ext.gourmet-fire.antidote-dev").unwrap_err();
        assert_eq!(err.code, "not_found");
    }

    #[test]
    fn acl_grant_mixed_boundary_rejected_409() {
        let (store, _dir) = temp_store();
        grant_logic(&store, "keyX", "int.portll.veld", "admin").unwrap();
        // Same principal now asks for an ext.* scope → I6 violation.
        let err = grant_logic(&store, "keyX", "ext.gourmet-fire.antidote-dev", "reader")
            .unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "boundary_violation");
    }

    #[test]
    fn acl_grant_bad_scope_tag_rejected_400() {
        let (store, _dir) = temp_store();
        let err = grant_logic(&store, "keyY", "bogus.scope.here", "reader").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
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
