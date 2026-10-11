//! Knowledge Graph Handlers
//!
//! Handlers for advanced knowledge graph operations including traversal,
//! entity management, and memory universe visualization.

use axum::{
    extract::{Extension, Path, State},
    response::Json,
};
use serde::Deserialize;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::entity_frequency::{frequency, Bucket, Frequency, Window};
use crate::entity_hygiene::{classify, Shelf};
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, GraphTraversal, MemoryUniverse};
use crate::memory::{Experience, MemoryId};
use crate::tag_cores::{stamped_cores, CoreThresholds, StampedCores};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// Runs a graph operation on the blocking pool: the graph's `parking_lot` lock and its store
/// reads must not stall a runtime worker.
async fn blocking<T, F>(op: F) -> Result<T, AppError>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(op)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("graph task failed: {e}")))?
        .map_err(AppError::Internal)
}

/// GET /api/graph/{user_id}/stats - Get graph statistics for a user
pub async fn get_graph_stats(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphStats>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Request to find an entity
#[derive(Debug, Deserialize)]
pub struct FindEntityRequest {
    pub user_id: String,
    pub entity_name: String,
}

/// POST /api/graph/entity/find - Find an entity by name
pub async fn find_entity(
    State(state): State<AppState>,
    Json(req): Json<FindEntityRequest>,
) -> Result<Json<Option<EntityNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name;
    let entity = blocking(move || graph.read().find_entity_by_name(&entity_name)).await?;

    Ok(Json(entity))
}

/// Request to traverse graph
#[derive(Debug, Deserialize)]
pub struct TraverseGraphRequest {
    pub user_id: String,
    pub entity_name: String,
    pub max_depth: Option<usize>,
}

/// POST /api/graph/traverse - Traverse graph from an entity
pub async fn traverse_graph(
    State(state): State<AppState>,
    Json(req): Json<TraverseGraphRequest>,
) -> Result<Json<GraphTraversal>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name.clone();
    let max_depth = req.max_depth.unwrap_or(2);
    let traversal = blocking(move || {
        let graph_guard = graph.read();
        match graph_guard.find_entity_by_name(&entity_name)? {
            Some(entity) => graph_guard
                .traverse_from_entity(&entity.uuid, max_depth)
                .map(Some),
            None => Ok(None),
        }
    })
    .await?
    .ok_or_else(|| AppError::MemoryNotFound(format!("Entity not found: {}", req.entity_name)))?;

    Ok(Json(traversal))
}

/// Request to get an episode
#[derive(Debug, Deserialize)]
pub struct GetEpisodeRequest {
    pub user_id: String,
    pub episode_uuid: String,
}

/// POST /api/graph/episode/get - Get an episodic node by UUID
pub async fn get_episode(
    State(state): State<AppState>,
    Json(req): Json<GetEpisodeRequest>,
) -> Result<Json<Option<EpisodicNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let episode_uuid =
        uuid::Uuid::parse_str(&req.episode_uuid).map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let episode = blocking(move || graph.read().get_episode(&episode_uuid)).await?;

    Ok(Json(episode))
}

/// Request to get all entities
#[derive(Debug, Deserialize)]
pub struct GetAllEntitiesRequest {
    pub user_id: String,
    pub limit: Option<usize>,
    /// Keep only entities whose lexical shelf matches, applied before `limit`.
    #[serde(default)]
    pub shelf: Option<Shelf>,
    /// Join each entity's tag-source provenance (one scan of the tag_source store).
    #[serde(default)]
    pub with_sources: bool,
    /// `name_embedding` is 768 floats per row; a client that draws names does not need it.
    #[serde(default = "default_true")]
    pub include_embeddings: bool,
}

fn default_true() -> bool {
    true
}

/// Per-tag provenance folded over every memory that stored the tag. Absent when no row names
/// the tag: records written before the store existed are undetermined, not caller-authored.
#[derive(Debug, Default, serde::Serialize)]
struct EntityProvenance {
    memories: usize,
    redacted: usize,
    sources: std::collections::BTreeMap<&'static str, usize>,
}

fn fold_provenance(
    rows: &std::collections::HashMap<MemoryId, crate::memory::tag_sources::TagSources>,
) -> std::collections::HashMap<String, EntityProvenance> {
    let mut out: std::collections::HashMap<String, EntityProvenance> =
        std::collections::HashMap::new();
    for sources in rows.values() {
        for tag in &sources.tags {
            let entry = out.entry(tag.tag.clone()).or_default();
            entry.memories += 1;
            if tag.redacted {
                entry.redacted += 1;
            }
            *entry.sources.entry(tag.source.as_str()).or_default() += 1;
        }
    }
    out
}

/// POST /api/graph/entities/all - Get all entities
pub async fn get_all_entities(
    State(state): State<AppState>,
    Json(req): Json<GetAllEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entities = blocking(move || graph.read().get_all_entities()).await?;

    let provenance = if req.with_sources {
        let memory = state
            .get_user_memory(&req.user_id)
            .map_err(AppError::Internal)?;
        let rows = blocking(move || memory.read().tag_source_store().all()).await?;
        Some(fold_provenance(&rows))
    } else {
        None
    };

    let total = entities.len();
    let limit = req.limit.unwrap_or(100);
    let mut rows = Vec::new();
    let mut shelf_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for mut entity in entities {
        let hygiene = classify(&entity.name, entity.is_proper_noun);
        *shelf_counts.entry(shelf_name(hygiene.shelf)).or_default() += 1;
        if req.shelf.is_some_and(|s| s != hygiene.shelf) {
            continue;
        }
        if rows.len() >= limit {
            continue;
        }
        if !req.include_embeddings {
            entity.name_embedding = None;
        }
        let mut row = serde_json::to_value(&entity).map_err(|e| AppError::Internal(e.into()))?;
        row["hygiene"] = serde_json::to_value(hygiene).map_err(|e| AppError::Internal(e.into()))?;
        if let Some(provenance) = &provenance {
            row["provenance"] = match provenance.get(&entity.name) {
                Some(p) => serde_json::to_value(p).map_err(|e| AppError::Internal(e.into()))?,
                None => serde_json::Value::Null,
            };
        }
        rows.push(row);
    }
    let count = rows.len();

    Ok(Json(serde_json::json!({
        "entities": rows,
        "count": count,
        "total": total,
        "shelves": shelf_counts,
        "witness": "graph.entities",
    })))
}

fn shelf_name(shelf: Shelf) -> &'static str {
    match shelf {
        Shelf::Cast => "cast",
        Shelf::Vocabulary => "vocabulary",
        Shelf::Noise => "noise",
    }
}

#[derive(Debug, Deserialize)]
pub struct EntityFrequencyRequest {
    pub user_id: String,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub until: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub bucket: Bucket,
    pub limit: Option<usize>,
    #[serde(default)]
    pub shelf: Option<Shelf>,
}

/// POST /api/graph/entities/frequency - Entity mentions per day or week over a window.
///
/// The window defaults to the 28 days ending now and is echoed back, so two readers can tell
/// whether they measured the same thing.
pub async fn get_entity_frequency(
    State(state): State<AppState>,
    Json(req): Json<EntityFrequencyRequest>,
) -> Result<Json<Frequency>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let until = req.until.unwrap_or_else(chrono::Utc::now);
    let since = req.since.unwrap_or(until - chrono::Duration::days(28));
    if since >= until {
        return Err(AppError::InvalidInput {
            field: "since".to_string(),
            reason: "since must be before until".to_string(),
        });
    }
    let limit = req.limit.unwrap_or(50);
    if limit == 0 {
        return Err(AppError::InvalidInput {
            field: "limit".to_string(),
            reason: "limit must be at least 1".to_string(),
        });
    }
    let window = Window {
        since,
        until,
        bucket: req.bucket,
        limit,
        shelf: req.shelf,
    };

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let result = blocking(move || {
        let graph = graph.read();
        let (episodes, undecodable) = graph.get_all_episodes()?;
        let names: std::collections::HashMap<_, _> = graph
            .get_all_entities()?
            .into_iter()
            .map(|e| (e.uuid, (e.name, e.is_proper_noun)))
            .collect();
        Ok(frequency(&episodes, undecodable, &names, &window))
    })
    .await?;

    Ok(Json(result))
}

/// GET /api/graph/{user_id}/universe - Get Memory Universe visualization
pub async fn get_memory_universe(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<MemoryUniverse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let universe = blocking(move || graph.read().get_universe()).await?;

    Ok(Json(universe))
}

/// Request body for `DELETE /api/graph/entity`.
#[derive(Debug, Deserialize)]
pub struct DeleteEntityRequest {
    /// Owning tenant. Checked against the authenticated binding by
    /// `resolve_request_user_id`, which rejects and audits a mismatch.
    pub user_id: String,
    /// The entity's UUID. Deletion is by id, never by name: a name can collide
    /// across tenants and is user-controlled, an id cannot be guessed into.
    pub entity_id: String,
}

/// Delete one entity from a tenant's graph.
///
/// WHY THIS ROUTE EXISTS. `graph_memory::delete_entity` has been implemented since the graph was
/// built and was reachable from nothing. Every path that WRITES an entity was routed; the one that
/// removes one was not. That made the entity namespace a one-way door: the forget path does not
/// touch the graph either, so nothing that reached it could be removed through the API at all.
/// Redaction (`crate::redaction`) stops new credential-shaped entities entering, but it cannot act
/// on anything already stored, and without this route there was no remediation path for a report.
///
/// AUTH. Mounted on the protected router, so an API key is required before the handler is entered.
/// Inside it, `resolve_request_user_id` binds the request to the authenticated tenant and refuses a
/// cross-tenant `user_id` with an audited warning. The graph is resolved per user, so a caller
/// cannot reach another tenant's entity even with a valid id.
///
/// HONEST RETURN. `delete_entity` returns false when the entity did not exist. That is reported as
/// `deleted: false`, never as an error and never as a success: "it is gone now" and "it was never
/// here" are different facts, and collapsing them is how a delete that silently did nothing gets
/// read as a delete that worked.
pub async fn delete_entity(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<DeleteEntityRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&req.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let uuid = uuid::Uuid::parse_str(req.entity_id.trim()).map_err(|_| AppError::InvalidInput {
        field: "entity_id".to_string(),
        reason: "must be a UUID".to_string(),
    })?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    let deleted = blocking(move || graph.write().delete_entity(&uuid)).await?;

    // Audited whether or not it removed anything. A deletion attempt is a security-relevant act
    // even when it is a no-op, and an attempt that found nothing is the shape a probe makes.
    info!(
        audit = "entity_deleted",
        user_id = %user_id,
        entity_id = %uuid,
        deleted,
        "entity delete requested"
    );

    if deleted {
        state.emit_event(MemoryEvent {
            event_type: "ENTITY_DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: user_id.clone(),
            memory_id: Some(uuid.to_string()),
            // The entity NAME is deliberately not previewed. This route's first use is removing a
            // credential-shaped entity, and echoing the name into the event stream would move the
            // disclosure rather than end it.
            content_preview: None,
            memory_type: Some("graph".to_string()),
            importance: None,
            count: Some(1),
            entities: None,
            results: None,
        });
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted": deleted,
        "entity_id": uuid.to_string(),
        "user_id": user_id,
    })))
}

/// DELETE /api/graph/{user_id}/clear - Clear all graph data for a user
pub async fn clear_user_graph(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    let (entities, relationships, episodes) = blocking(move || graph.write().clear_all()).await?;

    info!(
        "Cleared graph for user {}: {} entities, {} relationships, {} episodes",
        user_id, entities, relationships, episodes
    );

    state.emit_event(MemoryEvent {
        event_type: "GRAPH_CLEAR".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(format!("{}/{}/{}", entities, relationships, episodes)),
        content_preview: Some(format!(
            "Cleared {} entities, {} relationships, {} episodes",
            entities, relationships, episodes
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities + relationships + episodes),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "cleared": {
            "entities": entities,
            "relationships": relationships,
            "episodes": episodes
        }
    })))
}

/// POST /api/graph/{user_id}/rebuild - Rebuild graph from all existing memories
pub async fn rebuild_user_graph(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    // First, clear existing graph data
    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    if let Err(e) = blocking(move || graph.write().clear_all()).await {
        tracing::debug!("Graph clear before rebuild failed for {}: {:?}", user_id, e);
    }

    // Get all memories for this user
    let memory_sys = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memories: Vec<(MemoryId, Experience)> = {
        let memory_guard = memory_sys.read();
        memory_guard
            .get_all_memories()
            .map_err(AppError::Internal)?
            .into_iter()
            .map(|m| (m.id.clone(), m.experience.clone()))
            .collect()
    };

    let total_memories = memories.len();
    let mut processed = 0;

    // Re-process each memory through entity extraction
    for (memory_id, experience) in memories {
        if let Err(e) = state.process_experience_into_graph(&user_id, &experience, &memory_id) {
            tracing::debug!("Failed to process memory {}: {}", memory_id.0, e);
        } else {
            processed += 1;
        }
    }

    // Get final stats
    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;
    let entities_created = stats.entity_count;
    let relationships_created = stats.relationship_count;

    info!(
        "Rebuilt graph for user {}: processed {}/{} memories, created {} entities, {} relationships",
        user_id, processed, total_memories, entities_created, relationships_created
    );

    state.emit_event(MemoryEvent {
        event_type: "GRAPH_REBUILD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "Rebuilt: {} memories -> {} entities, {} relationships",
            processed, entities_created, relationships_created
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities_created + relationships_created),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "processed_memories": processed,
        "total_memories": total_memories,
        "entities_created": entities_created,
        "relationships_created": relationships_created
    })))
}

/// Request to invalidate a relationship
#[derive(Debug, Deserialize)]
pub struct InvalidateRelationshipRequest {
    pub user_id: String,
    pub relationship_uuid: String,
}

/// POST /api/graph/relationship/invalidate - Invalidate a relationship edge
pub async fn invalidate_relationship(
    State(state): State<AppState>,
    Json(req): Json<InvalidateRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let rel_uuid =
        uuid::Uuid::parse_str(&req.relationship_uuid).map_err(|_| AppError::InvalidInput {
            field: "relationship_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    blocking(move || graph.write().invalidate_relationship(&rel_uuid)).await?;

    state.emit_event(MemoryEvent {
        event_type: "EDGE_INVALIDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(req.relationship_uuid.clone()),
        content_preview: Some("Relationship invalidated".to_string()),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Relationship invalidated"
    })))
}

#[cfg(test)]
mod delete_entity_route_tests {
    use std::fs;
    use std::path::Path;

    fn router_src() -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/router.rs");
        fs::read_to_string(p).expect("router.rs must be readable")
    }

    const ROUTE: &str = "/api/graph/entity/delete";

    /// The defect this route fixes is a function nobody could reach: `delete_entity` existed in
    /// `graph_memory` since the graph was built and was routed from nothing. A unit test on the
    /// handler would have passed the whole time it was unreachable, so the assertion that matters
    /// is that it is MOUNTED.
    #[test]
    fn the_delete_route_is_mounted() {
        let src = router_src();
        assert!(
            src.contains(ROUTE),
            "{ROUTE} is not mounted — the handler is unreachable, which is the defect it exists \
             to fix"
        );
        assert!(
            src.contains(&format!("\"{ROUTE}\", delete(graph::delete_entity)")),
            "{ROUTE} must be mounted as DELETE on graph::delete_entity"
        );
    }

    /// AUTH, asserted structurally rather than trusted. The protected builder is where API-key
    /// middleware is applied by the caller; the public and probe builders are not. A delete route
    /// on either of those would be unauthenticated entity removal.
    #[test]
    fn the_delete_route_is_behind_auth_not_on_a_public_router() {
        let src = router_src();
        let protected_at = src
            .find("pub fn build_protected_routes")
            .expect("build_protected_routes must exist");
        let route_at = src.find(ROUTE).expect("route must be mounted");
        assert!(
            route_at > protected_at,
            "{ROUTE} is declared before build_protected_routes, so it sits on a public or probe \
             router and entity deletion would be unauthenticated"
        );

        // And it must not appear in either unauthenticated builder's body.
        let public_body = {
            let a = src
                .find("pub fn build_public_routes")
                .expect("build_public_routes must exist");
            &src[a..protected_at]
        };
        assert!(
            !public_body.contains(ROUTE),
            "{ROUTE} appears inside build_public_routes"
        );
        let probe_body = {
            let a = src
                .find("pub fn build_probe_routes")
                .expect("build_probe_routes must exist");
            let b = src
                .find("pub fn build_public_routes")
                .expect("build_public_routes must exist");
            &src[a..b]
        };
        assert!(
            !probe_body.contains(ROUTE),
            "{ROUTE} appears inside build_probe_routes"
        );
    }

    /// The handler must bind the request to the authenticated tenant. Without this, a valid API key
    /// for tenant A could delete tenant B's entity by supplying B's user_id.
    #[test]
    fn the_handler_resolves_the_user_against_the_authenticated_binding() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/graph.rs");
        let src = fs::read_to_string(p).expect("graph.rs must be readable");
        let at = src
            .find("pub async fn delete_entity")
            .expect("handler must exist");
        let body = &src[at..at + 1200.min(src.len() - at)];
        assert!(
            body.contains("resolve_request_user_id"),
            "delete_entity does not bind user_id to the authenticated tenant"
        );
        assert!(
            body.contains("authenticated_user"),
            "delete_entity does not take the authenticated user extension"
        );
    }
}

#[cfg(test)]
mod entity_frequency_route_tests {
    use std::fs;
    use std::path::Path;

    const ROUTE: &str = "/api/graph/entities/frequency";

    fn router_src() -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/router.rs");
        fs::read_to_string(p).expect("router.rs must be readable")
    }

    #[test]
    fn the_frequency_route_is_mounted_as_post_behind_auth() {
        let src = router_src();
        let protected_at = src
            .find("pub fn build_protected_routes")
            .expect("build_protected_routes must exist");
        let route_at = src.find(ROUTE).expect("route must be mounted");
        assert!(
            route_at > protected_at,
            "{ROUTE} sits on a public or probe router"
        );
        let after = &src[route_at..route_at + 200];
        assert!(
            after.contains("post(graph::get_entity_frequency)"),
            "{ROUTE} must be mounted as POST on graph::get_entity_frequency"
        );
    }

    #[test]
    fn a_request_with_no_window_fields_deserialises_to_defaults() {
        let req: super::EntityFrequencyRequest =
            serde_json::from_str(r#"{"user_id":"u"}"#).unwrap();
        assert!(req.since.is_none() && req.until.is_none());
        assert_eq!(req.bucket, super::Bucket::Day);
        assert!(req.shelf.is_none());
        let req: super::EntityFrequencyRequest =
            serde_json::from_str(r#"{"user_id":"u","bucket":"week","shelf":"noise","limit":3}"#)
                .unwrap();
        assert_eq!(req.bucket, super::Bucket::Week);
        assert_eq!(req.shelf, Some(super::Shelf::Noise));
        assert_eq!(req.limit, Some(3));
    }

    #[test]
    fn entities_all_request_defaults_keep_the_old_shape() {
        let req: super::GetAllEntitiesRequest =
            serde_json::from_str(r#"{"user_id":"u","limit":5}"#).unwrap();
        assert!(
            req.include_embeddings,
            "embeddings stay on by default for existing readers"
        );
        assert!(!req.with_sources);
        assert!(req.shelf.is_none());
    }
}

#[derive(Debug, Deserialize)]
pub struct StampedCoresRequest {
    pub user_id: String,
    pub min_records: Option<usize>,
    pub min_tags: Option<usize>,
    pub limit: Option<usize>,
}

/// POST /api/graph/tags/stamped-cores - Tags that sit on exactly the same records.
///
/// Read-only over the tag-source store. The census is a signal about producers, not about
/// words: a core of ordinary vocabulary shared by hundreds of records is a hook stamping
/// boilerplate, and the fix belongs at that hook.
pub async fn get_stamped_cores(
    State(state): State<AppState>,
    Json(req): Json<StampedCoresRequest>,
) -> Result<Json<StampedCores>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let defaults = CoreThresholds::default();
    let thresholds = CoreThresholds {
        min_records: req.min_records.unwrap_or(defaults.min_records),
        min_tags: req.min_tags.unwrap_or(defaults.min_tags),
        limit: req.limit.unwrap_or(defaults.limit),
    };
    for (field, value) in [
        ("min_records", thresholds.min_records),
        ("min_tags", thresholds.min_tags),
        ("limit", thresholds.limit),
    ] {
        if value == 0 {
            return Err(AppError::InvalidInput {
                field: field.to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
    }
    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let rows = blocking(move || memory.read().tag_source_store().all()).await?;
    Ok(Json(stamped_cores(&rows, thresholds)))
}

#[cfg(test)]
mod stamped_cores_route_tests {
    use std::fs;
    use std::path::Path;

    const ROUTE: &str = "/api/graph/tags/stamped-cores";

    #[test]
    fn the_stamped_cores_route_is_mounted_as_post_behind_auth() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/router.rs");
        let src = fs::read_to_string(p).expect("router.rs must be readable");
        let protected_at = src
            .find("pub fn build_protected_routes")
            .expect("build_protected_routes must exist");
        let route_at = src.find(ROUTE).expect("route must be mounted");
        assert!(
            route_at > protected_at,
            "{ROUTE} sits on a public or probe router"
        );
        assert!(src[route_at..route_at + 200].contains("post(graph::get_stamped_cores)"));
    }

    #[test]
    fn thresholds_default_when_omitted() {
        let req: super::StampedCoresRequest = serde_json::from_str(r#"{"user_id":"u"}"#).unwrap();
        assert!(req.min_records.is_none() && req.min_tags.is_none() && req.limit.is_none());
    }
}
