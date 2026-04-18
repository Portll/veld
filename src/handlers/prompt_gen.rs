//! Prompt Generation Handler
//!
//! Assembles a complete prompt context from all veld subsystems in a single call.
//! This is the "end-to-end" endpoint that sleight and other consumers use to get
//! a fully formed context package — memories, entities, facts, todos, reminders,
//! and graph relationships — ready to inject into an LLM prompt.
//!
//! Also provides entity resolution and canonicalization for "exact" entity lookup.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::utils::strip_system_noise;
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory::EntityNode;
use crate::memory::{Query as MemoryQuery, RetrievalMode, TodoStatus};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

// ─── Request ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct PromptGenRequest {
    pub user_id: String,

    /// The goal, question, or task the prompt is being assembled for.
    pub goal: String,

    /// Optional current context (e.g., the user's latest message or file content).
    /// Used to bias retrieval toward relevant memories.
    #[serde(default)]
    pub context: Option<String>,

    /// Maximum memories to include (default: 10)
    #[serde(default = "default_max_memories")]
    pub max_memories: usize,

    /// Maximum facts to include (default: 10)
    #[serde(default = "default_max_facts")]
    pub max_facts: usize,

    /// Maximum entities to include (default: 15)
    #[serde(default = "default_max_entities")]
    pub max_entities: usize,

    /// Maximum todos to include (default: 5)
    #[serde(default = "default_max_todos")]
    pub max_todos: usize,

    /// Include entity graph neighborhood (default: true)
    #[serde(default = "default_true")]
    pub include_graph: bool,

    /// Include due reminders (default: true)
    #[serde(default = "default_true")]
    pub include_reminders: bool,

    /// Include active todos (default: true)
    #[serde(default = "default_true")]
    pub include_todos: bool,

    /// Include extracted facts (default: true)
    #[serde(default = "default_true")]
    pub include_facts: bool,

    /// Specific entity names to resolve and include (exact match with alias resolution)
    #[serde(default)]
    pub resolve_entities: Vec<String>,

    /// Memory types to filter for (empty = all types)
    #[serde(default)]
    pub memory_types: Vec<String>,

    /// Retrieval mode: semantic, associative, temporal, hybrid (default: hybrid)
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_max_memories() -> usize { 10 }
fn default_max_facts() -> usize { 10 }
fn default_max_entities() -> usize { 15 }
fn default_max_todos() -> usize { 5 }
fn default_true() -> bool { true }
fn default_mode() -> String { "hybrid".to_string() }

// ─── Response ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct PromptGenResponse {
    /// Assembled prompt context as structured sections
    pub sections: Vec<PromptSection>,

    /// The full assembled prompt text (all sections concatenated)
    pub prompt: String,

    /// Resolved entities with their structured details
    pub entities: Vec<ResolvedEntity>,

    /// Active relationships between resolved entities
    pub relationships: Vec<EntityRelationship>,

    /// Metadata about what was assembled
    pub meta: PromptMeta,
}

#[derive(Serialize)]
pub struct PromptSection {
    pub name: String,
    pub content: String,
    pub item_count: usize,
}

#[derive(Serialize, Clone)]
pub struct ResolvedEntity {
    pub id: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub entity_type: String,
    pub attributes: HashMap<String, String>,
    pub mention_count: usize,
    pub salience: f32,
    pub is_proper_noun: bool,
    pub related_facts: Vec<String>,
}

#[derive(Serialize)]
pub struct EntityRelationship {
    pub from: String,
    pub to: String,
    pub relation_type: String,
    pub strength: f32,
    pub context: String,
}

#[derive(Serialize)]
pub struct PromptMeta {
    pub memories_retrieved: usize,
    pub facts_retrieved: usize,
    pub entities_resolved: usize,
    pub todos_included: usize,
    pub reminders_included: usize,
    pub latency_ms: f64,
}

// ─── Handler ────────────────────────────────────────────────────────────────

/// POST /api/prompt/gen — Assemble a complete prompt from all veld subsystems.
///
/// This is the single-call endpoint for end-to-end context assembly. It:
/// 1. Retrieves relevant memories using the goal + context
/// 2. Extracts and resolves entities from memories and the knowledge graph
/// 3. Looks up semantic facts for resolved entities
/// 4. Fetches active todos and due reminders
/// 5. Assembles everything into a structured prompt with sections
pub async fn prompt_gen(
    State(state): State<AppState>,
    Json(req): Json<PromptGenRequest>,
) -> Result<Json<PromptGenResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.goal, false).map_validation_err("goal")?;
    validation::validate_max_results(req.max_memories).map_validation_err("max_memories")?;
    validation::validate_max_results(req.max_facts).map_validation_err("max_facts")?;
    validation::validate_max_results(req.max_entities).map_validation_err("max_entities")?;
    validation::validate_max_results(req.max_todos).map_validation_err("max_todos")?;
    validation::validate_entities(&req.resolve_entities).map_validation_err("resolve_entities")?;

    let op_start = std::time::Instant::now();

    let memory_system = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let todo_store = state.todo_store.clone();
    let prospective_store = state.prospective_store.clone();

    let user_id = req.user_id.clone();
    let goal = strip_system_noise(&req.goal);
    let context = req.context.as_deref().map(strip_system_noise);
    let max_memories = req.max_memories;
    let max_facts = req.max_facts;
    let max_entities = req.max_entities;
    let max_todos = req.max_todos;
    let include_graph = req.include_graph;
    let include_reminders = req.include_reminders;
    let include_todos = req.include_todos;
    let include_facts = req.include_facts;
    let resolve_entities = req.resolve_entities.clone();
    let mode_str = req.mode.clone();

    let result = tokio::task::spawn_blocking(move || {
        let memory_guard = memory_system.read();
        let graph_guard = graph_memory.read();

        let mut sections = Vec::new();
        let mut all_entities: Vec<ResolvedEntity> = Vec::new();
        let mut relationships: Vec<EntityRelationship> = Vec::new();

        // ── 1. Retrieve relevant memories ──────────────────────────

        let search_text = match &context {
            Some(ctx) => format!("{}\n\n{}", goal, ctx),
            None => goal.clone(),
        };

        let retrieval_mode = match mode_str.as_str() {
            "semantic" | "similarity" => RetrievalMode::Similarity,
            "associative" => RetrievalMode::Associative,
            "temporal" => RetrievalMode::Temporal,
            "causal" => RetrievalMode::Causal,
            "spreading_activation" | "graph" => RetrievalMode::SpreadingActivation,
            _ => RetrievalMode::Hybrid,
        };

        let query = MemoryQuery {
            query_text: Some(search_text.clone()),
            max_results: max_memories,
            retrieval_mode,
            ..Default::default()
        };

        let memories = memory_guard
            .recall(&query)
            .unwrap_or_default();

        if !memories.is_empty() {
            let mut memory_lines = Vec::new();
            for (i, mem) in memories.iter().enumerate() {
                let age = chrono::Utc::now()
                    .signed_duration_since(mem.created_at)
                    .num_days();
                let type_label = format!("{:?}", mem.experience.experience_type);
                memory_lines.push(format!(
                    "{}. [{}] ({}d ago, importance: {:.2}) {}",
                    i + 1,
                    type_label,
                    age,
                    mem.importance(),
                    mem.experience.content
                ));
            }

            sections.push(PromptSection {
                name: "Relevant Memories".to_string(),
                content: memory_lines.join("\n"),
                item_count: memories.len(),
            });
        }

        // ── 2. Resolve entities ────────────────────────────────────

        // Collect entities from explicitly requested names
        for name in &resolve_entities {
            if let Some(entity) = resolve_entity_by_name(&graph_guard, name) {
                if all_entities.len() < max_entities {
                    all_entities.push(entity);
                }
            }
        }

        // Collect entities from retrieved memories via the graph.
        // We look up each memory's associated entity episodes through the
        // entity graph, finding entities that appear in those episodes.
        if include_graph {
            let mut seen_uuids: std::collections::HashSet<String> = all_entities
                .iter()
                .map(|e| e.id.clone())
                .collect();

            for mem in &memories {
                // Try to find entities mentioned in this memory by searching the
                // graph for the memory content. Use find_entity_by_name on words
                // that could be entity names extracted from the memory content.
                // This is a lightweight approach — full NER extraction would
                // require the NER extractor which lives outside GraphMemory.
                let content_words: Vec<&str> = mem.experience.content
                    .split_whitespace()
                    .filter(|w| w.len() > 2 && w.chars().next().is_some_and(|c| c.is_uppercase()))
                    .collect();

                for word in content_words {
                    if all_entities.len() >= max_entities {
                        break;
                    }
                    if let Ok(Some(node)) = graph_guard.find_entity_by_name(word) {
                        let uuid_str = node.uuid.to_string();
                        if seen_uuids.contains(&uuid_str) {
                            continue;
                        }
                        let resolved = entity_node_to_resolved(&node);
                        seen_uuids.insert(uuid_str);
                        all_entities.push(resolved);
                    }
                }
            }
        }

        // ── 3. Look up facts for resolved entities ─────────────────

        if include_facts && !all_entities.is_empty() {
            let mut fact_lines = Vec::new();

            for entity in &mut all_entities {
                // Use search_facts with entity name as query
                let entity_facts = memory_guard
                    .search_facts(&user_id, &entity.canonical_name, 3)
                    .unwrap_or_default();

                for fact in &entity_facts {
                    let line = format!("- {} (confidence: {:.2})", fact.fact, fact.confidence);
                    entity.related_facts.push(fact.fact.clone());
                    fact_lines.push(line);
                }
            }

            // Also search facts by goal query
            let query_facts = memory_guard
                .search_facts(&user_id, &goal, max_facts)
                .unwrap_or_default();

            for fact in &query_facts {
                let line = format!("- {} (confidence: {:.2})", fact.fact, fact.confidence);
                if !fact_lines.contains(&line) {
                    fact_lines.push(line);
                }
            }

            if !fact_lines.is_empty() {
                fact_lines.truncate(max_facts);
                sections.push(PromptSection {
                    name: "Known Facts".to_string(),
                    content: fact_lines.join("\n"),
                    item_count: fact_lines.len(),
                });
            }
        }

        // ── 4. Build entity section ────────────────────────────────

        if !all_entities.is_empty() {
            let mut entity_lines = Vec::new();
            for entity in &all_entities {
                let mut line = format!(
                    "- **{}** [{}]",
                    entity.canonical_name, entity.entity_type
                );
                if !entity.aliases.is_empty() {
                    line.push_str(&format!(" (aka: {})", entity.aliases.join(", ")));
                }
                if !entity.attributes.is_empty() {
                    let attrs: Vec<String> = entity
                        .attributes
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();
                    line.push_str(&format!(" — {}", attrs.join(", ")));
                }
                if entity.mention_count > 1 {
                    line.push_str(&format!(" (mentioned {}×)", entity.mention_count));
                }
                entity_lines.push(line);
            }

            sections.push(PromptSection {
                name: "Known Entities".to_string(),
                content: entity_lines.join("\n"),
                item_count: all_entities.len(),
            });
        }

        // ── 5. Build relationship section ──────────────────────────

        if include_graph && all_entities.len() >= 2 {
            let entity_uuids: Vec<_> = all_entities
                .iter()
                .filter_map(|e| uuid::Uuid::parse_str(&e.id).ok())
                .collect();

            for (i, uuid_a) in entity_uuids.iter().enumerate() {
                for uuid_b in entity_uuids.iter().skip(i + 1) {
                    if let Ok(Some(edge)) = graph_guard.find_relationship_between(uuid_a, uuid_b) {
                        if edge.strength > 0.2 {
                            let from_name = all_entities
                                .iter()
                                .find(|e| e.id == uuid_a.to_string())
                                .map(|e| e.canonical_name.clone())
                                .unwrap_or_default();
                            let to_name = all_entities
                                .iter()
                                .find(|e| e.id == uuid_b.to_string())
                                .map(|e| e.canonical_name.clone())
                                .unwrap_or_default();

                            relationships.push(EntityRelationship {
                                from: from_name,
                                to: to_name,
                                relation_type: format!("{:?}", edge.relation_type),
                                strength: edge.strength,
                                context: edge.context.clone(),
                            });
                        }
                    }
                }
            }

            if !relationships.is_empty() {
                let rel_lines: Vec<String> = relationships
                    .iter()
                    .map(|r| {
                        format!(
                            "- {} ↔ {} ({}, strength: {:.2})",
                            r.from, r.to, r.relation_type, r.strength
                        )
                    })
                    .collect();

                sections.push(PromptSection {
                    name: "Entity Relationships".to_string(),
                    content: rel_lines.join("\n"),
                    item_count: relationships.len(),
                });
            }
        }

        // ── 6. Fetch active todos ──────────────────────────────────

        if include_todos {
            let active_statuses = [TodoStatus::Todo, TodoStatus::InProgress, TodoStatus::Blocked];
            let todos = todo_store
                .list_todos_for_user(&user_id, Some(&active_statuses))
                .unwrap_or_default();

            if !todos.is_empty() {
                let todo_lines: Vec<String> = todos
                    .iter()
                    .take(max_todos)
                    .map(|t| {
                        let priority = format!("{:?}", t.priority).to_lowercase();
                        let project = t
                            .project_prefix
                            .as_deref()
                            .map(|p| format!(" [{}]", p))
                            .unwrap_or_default();
                        format!("- [{}]{} {}", priority, project, t.content)
                    })
                    .collect();

                sections.push(PromptSection {
                    name: "Active Todos".to_string(),
                    content: todo_lines.join("\n"),
                    item_count: todo_lines.len(),
                });
            }
        }

        // ── 7. Fetch due reminders ─────────────────────────────────

        if include_reminders {
            let reminders = prospective_store
                .get_due_tasks(&user_id)
                .unwrap_or_default();

            if !reminders.is_empty() {
                let reminder_lines: Vec<String> = reminders
                    .iter()
                    .take(5)
                    .map(|r| format!("- ⏰ {}", r.content))
                    .collect();

                sections.push(PromptSection {
                    name: "Due Reminders".to_string(),
                    content: reminder_lines.join("\n"),
                    item_count: reminder_lines.len(),
                });
            }
        }

        // ── 8. Assemble full prompt text ───────────────────────────

        let mut prompt_parts = Vec::new();
        for section in &sections {
            prompt_parts.push(format!("## {}\n\n{}", section.name, section.content));
        }
        let prompt = prompt_parts.join("\n\n---\n\n");

        let meta = PromptMeta {
            memories_retrieved: memories.len(),
            facts_retrieved: sections
                .iter()
                .find(|s| s.name == "Known Facts")
                .map(|s| s.item_count)
                .unwrap_or(0),
            entities_resolved: all_entities.len(),
            todos_included: sections
                .iter()
                .find(|s| s.name == "Active Todos")
                .map(|s| s.item_count)
                .unwrap_or(0),
            reminders_included: sections
                .iter()
                .find(|s| s.name == "Due Reminders")
                .map(|s| s.item_count)
                .unwrap_or(0),
            latency_ms: op_start.elapsed().as_secs_f64() * 1000.0,
        };

        PromptGenResponse {
            sections,
            prompt,
            entities: all_entities,
            relationships,
            meta,
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Prompt gen task panicked: {e}")))?;

    info!(
        user_id = %req.user_id,
        memories = result.meta.memories_retrieved,
        entities = result.meta.entities_resolved,
        facts = result.meta.facts_retrieved,
        latency_ms = format!("{:.1}", result.meta.latency_ms),
        "prompt_gen assembled"
    );

    Ok(Json(result))
}

// ─── Entity Resolution ──────────────────────────────────────────────────────

/// POST /api/entity/resolve — Resolve an entity name to its canonical form,
/// with alias matching, fuzzy lookup, and structured attributes.
pub async fn resolve_entity(
    State(state): State<AppState>,
    Json(req): Json<EntityResolveRequest>,
) -> Result<Json<EntityResolveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entities(&req.names).map_validation_err("names")?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let names = req.names.clone();

    let result = tokio::task::spawn_blocking(move || {
        let graph_guard = graph_memory.read();
        let mut resolved = Vec::new();
        let mut unresolved = Vec::new();

        for name in &names {
            if let Some(entity) = resolve_entity_by_name(&graph_guard, name) {
                resolved.push(entity);
            } else {
                unresolved.push(name.clone());
            }
        }

        EntityResolveResponse {
            resolved,
            unresolved,
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Entity resolve task panicked: {e}")))?;

    Ok(Json(result))
}

#[derive(Deserialize)]
pub struct EntityResolveRequest {
    pub user_id: String,
    pub names: Vec<String>,
}

#[derive(Serialize)]
pub struct EntityResolveResponse {
    pub resolved: Vec<ResolvedEntity>,
    pub unresolved: Vec<String>,
}

/// POST /api/entity/set-attribute — Set a structured attribute on an entity.
/// This is how you make entities "exact" — store DOB, email, coordinates, etc.
///
/// Implemented by reading the entity, setting the attribute, and writing it back
/// via `add_entity` (which merges by name). This increments `mention_count` as a
/// side effect of the merge path — acceptable for attribute writes on active entities.
pub async fn set_entity_attribute(
    State(state): State<AppState>,
    Json(req): Json<SetAttributeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.entity_name).map_validation_err("entity_name")?;
    validation::validate_content(&req.key, false).map_validation_err("key")?;
    validation::validate_content(&req.value, false).map_validation_err("value")?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name.clone();
    let key = req.key.clone();
    let value = req.value.clone();

    tokio::task::spawn_blocking(move || {
        let graph_guard = graph_memory.read();

        // Find entity by name (handles exact, case-insensitive, stemmed, substring)
        let node = graph_guard
            .find_entity_by_name(&entity_name)
            .map_err(|e| AppError::Internal(e))?
            .ok_or_else(|| {
                AppError::InvalidInput {
                    field: "entity_name".to_string(),
                    reason: format!("Entity not found: {}", entity_name),
                }
            })?;

        // Re-add with the attribute set. add_entity merges into existing entities
        // matched by name, preserving created_at and canonical name while updating
        // the stored data. The attribute we set here will be part of the merged entity.
        let mut updated = node.clone();
        updated.attributes.insert(key.clone(), value.clone());
        graph_guard.add_entity(updated)
            .map_err(|e| AppError::Internal(e))?;

        Ok::<_, AppError>(serde_json::json!({
            "entity": node.name,
            "attribute": key,
            "value": value,
            "status": "set"
        }))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Set attribute task panicked: {e}")))?
    .map(Json)
}

#[derive(Deserialize)]
pub struct SetAttributeRequest {
    pub user_id: String,
    pub entity_name: String,
    pub key: String,
    pub value: String,
}

/// POST /api/entity/merge — Merge two entities into one canonical entity.
/// All edges from the source are redirected to the target, then the source is deleted.
pub async fn merge_entities(
    State(state): State<AppState>,
    Json(req): Json<MergeEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.source_name).map_validation_err("source_name")?;
    validation::validate_entity(&req.target_name).map_validation_err("target_name")?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let source_name = req.source_name.clone();
    let target_name = req.target_name.clone();

    let result = tokio::task::spawn_blocking(move || {
        let graph_guard = graph_memory.read();

        let source = graph_guard
            .find_entity_by_name(&source_name)
            .map_err(|e| AppError::Internal(e))?
            .ok_or_else(|| {
                AppError::InvalidInput {
                    field: "source_name".to_string(),
                    reason: format!("Source entity not found: {}", source_name),
                }
            })?;

        let target = graph_guard
            .find_entity_by_name(&target_name)
            .map_err(|e| AppError::Internal(e))?
            .ok_or_else(|| {
                AppError::InvalidInput {
                    field: "target_name".to_string(),
                    reason: format!("Target entity not found: {}", target_name),
                }
            })?;

        if source.uuid == target.uuid {
            return Err(AppError::InvalidInput {
                field: "source_name".to_string(),
                reason: "Source and target are the same entity".to_string(),
            });
        }

        // Manually redirect relationships from source to target.
        // Get all edges involving the source entity and create corresponding
        // edges pointing to/from the target instead.
        let source_edges = graph_guard
            .get_entity_relationships(&source.uuid)
            .map_err(|e| AppError::Internal(e))?;

        let mut edges_moved: usize = 0;
        for edge in &source_edges {
            // Skip self-referential edges that would become after redirect
            let new_from = if edge.from_entity == source.uuid { target.uuid } else { edge.from_entity };
            let new_to = if edge.to_entity == source.uuid { target.uuid } else { edge.to_entity };
            if new_from == new_to {
                continue;
            }

            // Check if an equivalent relationship already exists between these entities
            if graph_guard
                .find_relationship_between(&new_from, &new_to)
                .map_err(|e| AppError::Internal(e))?
                .is_some()
            {
                // Relationship already exists, skip to avoid duplicates
                continue;
            }

            let mut redirected = edge.clone();
            redirected.uuid = uuid::Uuid::new_v4();
            redirected.from_entity = new_from;
            redirected.to_entity = new_to;
            graph_guard
                .add_relationship(redirected)
                .map_err(|e| AppError::Internal(e))?;
            edges_moved += 1;
        }

        // Update target: merge attributes from source (target wins on conflict),
        // and record source name as an alias.
        let mut target_updated = target.clone();
        if target.name != source.name {
            target_updated
                .attributes
                .entry("aliases".to_string())
                .and_modify(|v: &mut String| {
                    if !v.contains(&source.name) {
                        v.push_str(&format!(", {}", source.name));
                    }
                })
                .or_insert_with(|| source.name.clone());
        }
        for (k, v) in &source.attributes {
            if k != "merged_into" {
                target_updated.attributes.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        graph_guard
            .add_entity(target_updated)
            .map_err(|e| AppError::Internal(e))?;

        // Delete the source entity now that all edges have been redirected to target.
        graph_guard
            .delete_entity(&source.uuid)
            .map_err(|e| AppError::Internal(e))?;

        Ok::<_, AppError>(serde_json::json!({
            "source": source_name,
            "target": target_name,
            "edges_moved": edges_moved,
            "status": "merged"
        }))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Merge task panicked: {e}")))?;

    result.map(Json)
}

#[derive(Deserialize)]
pub struct MergeEntitiesRequest {
    pub user_id: String,
    pub source_name: String,
    pub target_name: String,
}

/// POST /api/entity/alias — Add an alias for an entity.
/// The alias is stored in the entity's `aliases` attribute and is used by
/// `find_entity_by_name`'s fuzzy/substring matching to surface the canonical entity.
/// For exact O(1) alias lookup, resubmit the alias as a new memory mention so the
/// entity extraction pipeline can index it in the name graph.
pub async fn add_entity_alias(
    State(state): State<AppState>,
    Json(req): Json<AddAliasRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.entity_name).map_validation_err("entity_name")?;
    validation::validate_entity(&req.alias).map_validation_err("alias")?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name.clone();
    let alias = req.alias.clone();

    let result = tokio::task::spawn_blocking(move || {
        let graph_guard = graph_memory.read();

        let node = graph_guard
            .find_entity_by_name(&entity_name)
            .map_err(|e| AppError::Internal(e))?
            .ok_or_else(|| {
                AppError::InvalidInput {
                    field: "entity_name".to_string(),
                    reason: format!("Entity not found: {}", entity_name),
                }
            })?;

        // Add alias to attributes and re-add via add_entity to persist
        let mut updated = node.clone();
        updated
            .attributes
            .entry("aliases".to_string())
            .and_modify(|v: &mut String| {
                if !v.contains(&alias) {
                    v.push_str(&format!(", {}", alias));
                }
            })
            .or_insert_with(|| alias.clone());

        graph_guard
            .add_entity(updated)
            .map_err(|e| AppError::Internal(e))?;

        Ok::<_, AppError>(serde_json::json!({
            "entity": entity_name,
            "alias": alias,
            "status": "added"
        }))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Add alias task panicked: {e}")))?;

    result.map(Json)
}

#[derive(Deserialize)]
pub struct AddAliasRequest {
    pub user_id: String,
    pub entity_name: String,
    pub alias: String,
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Resolve an entity name through GraphMemory's built-in multi-strategy lookup:
/// 1. Exact match (O(1))
/// 2. Case-insensitive match (O(1))
/// 3. Stemmed match (O(1))
/// 4. Substring match
/// 5. Word-level match
fn resolve_entity_by_name(
    graph: &crate::graph_memory::GraphMemory,
    name: &str,
) -> Option<ResolvedEntity> {
    let node = graph.find_entity_by_name(name).ok()??;
    Some(entity_node_to_resolved(&node))
}

/// Convert a raw EntityNode into a ResolvedEntity with aliases.
fn entity_node_to_resolved(node: &EntityNode) -> ResolvedEntity {
    let aliases_str = node
        .attributes
        .get("aliases")
        .cloned()
        .unwrap_or_default();
    let aliases: Vec<String> = if aliases_str.is_empty() {
        Vec::new()
    } else {
        aliases_str
            .split(", ")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && *s != node.name)
            .collect()
    };

    let entity_type = node
        .labels
        .first()
        .map(|l| format!("{:?}", l))
        .unwrap_or_else(|| "Unknown".to_string());

    ResolvedEntity {
        id: node.uuid.to_string(),
        canonical_name: node.name.clone(),
        aliases,
        entity_type,
        attributes: node.attributes.clone(),
        mention_count: node.mention_count,
        salience: node.salience,
        is_proper_noun: node.is_proper_noun,
        related_facts: Vec::new(), // filled by caller
    }
}
