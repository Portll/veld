//! MCP tool implementations for Veld.
//!
//! All `#[tool]` method implementations live here. The tool names and parameter
//! names are part of the MCP contract and MUST NOT change.

use anyhow::Result;
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ErrorCode},
    tool, tool_router, ErrorData as McpError,
};
use std::borrow::Cow;
use std::sync::Arc;

use super::types::*;
use super::VeldMcpServer;

#[tool_router]
impl VeldMcpServer {
    pub(crate) fn create(api_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: Arc::new(super::client::AsyncApiClient::new(api_url, api_key, user_id)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a memory for future recall. Use this to remember important information, decisions, user preferences, project context, or anything you want to recall later."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RememberResponse> = self
            .client
            .post(
                "/api/remember",
                &RememberRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    memory_type: params.memory_type,
                    tags: params.tags,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Stored memory: {} ({})",
                resp.id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Search memories using semantic, associative, or hybrid retrieval. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'hybrid' (combines both)."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RecallResponse> = self
            .client
            .post(
                "/api/recall",
                &RecallRequest {
                    user_id: self.client.user_id.clone(),
                    query: params.query,
                    limit: params.limit,
                    mode: params.mode,
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Found {} memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "**[{}]** {} (similarity: {:.0}%)\n{}\n\n",
                        mem.memory_type,
                        &mem.id[..8.min(mem.id.len())],
                        mem.similarity * 100.0,
                        mem.content
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "REQUIRED: Call this to surface relevant memories based on current context. Enables automatic memory surfacing and implicit feedback learning."
    )]
    async fn proactive_context(
        &self,
        Parameters(params): Parameters<ProactiveContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ProactiveContextResponse> = self
            .client
            .post(
                "/api/proactive_context",
                &ProactiveContextRequest {
                    user_id: self.client.user_id.clone(),
                    context: params.context,
                    max_results: params.max_results.unwrap_or(5),
                    auto_ingest: params.auto_ingest.unwrap_or(true),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Surfaced {} relevant memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "- [{}%] **{}**: {}\n",
                        (mem.relevance_score * 100.0) as u32,
                        mem.memory_type,
                        mem.content.chars().take(200).collect::<String>()
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // AGENT SELF-PAGING TOOLS - Tier + Pin Control
    // =========================================================================

    #[tool(
        description = "Pin a memory so it resists automatic decay and skips compression. Use for critical facts that must persist in working context (Letta-style core-memory anchor)."
    )]
    async fn pin_memory(
        &self,
        Parameters(params): Parameters<PinMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<AnchorResponse> = self
            .client
            .post(
                "/api/anchor",
                &AnchorRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    anchor: true,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Pinned memory {} (anchored={})",
                resp.memory_id, resp.anchored
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Unpin a previously pinned memory, allowing automatic decay and compression to apply again."
    )]
    async fn unpin_memory(
        &self,
        Parameters(params): Parameters<PinMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<AnchorResponse> = self
            .client
            .post(
                "/api/anchor",
                &AnchorRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    anchor: false,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Unpinned memory {} (anchored={})",
                resp.memory_id, resp.anchored
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Promote a memory back into the hot working tier. Use when an archived memory needs to be brought into immediate focus for the current task."
    )]
    async fn promote_to_hot(
        &self,
        Parameters(params): Parameters<PinMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<TierMoveResponse> = self
            .client
            .post(
                "/api/memory/tier",
                &TierMoveRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    target_tier: "working".to_string(),
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Promoted memory {} from {} to {}",
                resp.memory_id, resp.previous_tier, resp.current_tier
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Push a memory down to cold archival storage. Use when a memory is no longer relevant to the current task but should be preserved for possible future recall."
    )]
    async fn demote_to_cold(
        &self,
        Parameters(params): Parameters<PinMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<TierMoveResponse> = self
            .client
            .post(
                "/api/memory/tier",
                &TierMoveRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    target_tier: "archive".to_string(),
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Demoted memory {} from {} to {}",
                resp.memory_id, resp.previous_tier, resp.current_tier
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Give explicit feedback on memories that were surfaced by a recent recall. outcome: 'helpful' boosts (memories that helped solve the task), 'misleading' suppresses (memories that wasted time or were wrong), 'neutral' is a mild access bump. Feeds the closed-loop ranker — future recalls re-rank based on accumulated momentum."
    )]
    async fn reinforce_memories(
        &self,
        Parameters(params): Parameters<ReinforceParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ReinforceResponse> = self
            .client
            .post(
                "/api/reinforce",
                &ReinforceRequest {
                    user_id: self.client.user_id.clone(),
                    ids: params.memory_ids,
                    outcome: params.outcome,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Reinforced {} memories — {} boosts, {} decays, {} associations strengthened",
                resp.memories_processed,
                resp.importance_boosts,
                resp.importance_decays,
                resp.associations_strengthened
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Move a memory to a specific tier. target_tier accepts: 'working', 'session', 'longterm', 'archive'. Prefer the named helpers (promote_to_hot, demote_to_cold) for the common cases."
    )]
    async fn move_to_tier(
        &self,
        Parameters(params): Parameters<MoveTierParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<TierMoveResponse> = self
            .client
            .post(
                "/api/memory/tier",
                &TierMoveRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    target_tier: params.target_tier,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Moved memory {} from {} to {}",
                resp.memory_id, resp.previous_tier, resp.current_tier
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // LINEAGE TOOLS - Causal Memory Tracking
    // =========================================================================

    #[tool(
        description = "Trace the causal lineage of a memory. Find what caused it (backward), what it led to (forward), or both. Useful for understanding 'why' something happened."
    )]
    async fn lineage_trace(
        &self,
        Parameters(params): Parameters<LineageTraceParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageTraceResponse> = self
            .client
            .post(
                "/api/lineage/trace",
                &LineageTraceRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    direction: params.direction.unwrap_or_else(|| "backward".to_string()),
                    max_depth: params.max_depth.unwrap_or(10),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Lineage Trace** ({})\n\nRoot: {}\nDepth: {}\n\n",
                    resp.direction, resp.root, resp.depth
                );

                if resp.edges.is_empty() {
                    output.push_str("No causal connections found.\n");
                } else {
                    output.push_str("**Causal Chain:**\n");
                    for edge in &resp.edges {
                        let confidence = (edge.confidence * 100.0) as u32;
                        let source_icon = match edge.source.as_str() {
                            "Confirmed" => "✓",
                            "Explicit" => "⚡",
                            _ => "?",
                        };
                        output.push_str(&format!(
                            "  {} --[{} {}% {}]--> {}\n",
                            &edge.from[..8.min(edge.from.len())],
                            edge.relation,
                            confidence,
                            source_icon,
                            &edge.to[..8.min(edge.to.len())]
                        ));
                    }

                    output.push_str(&format!("\n**Path:** {}\n", resp.path.join(" → ")));
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Confirm an inferred causal relationship between memories. This improves the system's confidence and learning."
    )]
    async fn lineage_confirm(
        &self,
        Parameters(params): Parameters<LineageConfirmParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageConfirmResponse> = self
            .client
            .post(
                "/api/lineage/confirm",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✓ Confirmed edge: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Reject an incorrectly inferred causal relationship. This helps the system learn better inference patterns."
    )]
    async fn lineage_reject(
        &self,
        Parameters(params): Parameters<LineageRejectParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageRejectResponse> = self
            .client
            .post(
                "/api/lineage/reject",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✗ Rejected edge: {}",
                resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Create an explicit causal link between two memories. Relations: Caused (Error→Todo), ResolvedBy (Todo→Learning), InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo."
    )]
    async fn lineage_link(
        &self,
        Parameters(params): Parameters<LineageLinkParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageAddResponse> = self
            .client
            .post(
                "/api/lineage/link",
                &LineageAddEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    from_memory_id: params.from_memory_id,
                    to_memory_id: params.to_memory_id,
                    relation: params.relation,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "⚡ Created link: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Get statistics about the causal lineage graph - edge counts, relation types, confidence distribution."
    )]
    async fn lineage_stats(
        &self,
        Parameters(_params): Parameters<LineageStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageStatsResponse> = self
            .client
            .post(
                "/api/lineage/stats",
                &LineageStatsRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = "**Lineage Graph Statistics**\n\n".to_string();
                output.push_str(&format!("**Edges:** {}\n", resp.total_edges));
                output.push_str(&format!("  ✓ Confirmed: {}\n", resp.confirmed_edges));
                output.push_str(&format!("  ? Inferred: {}\n", resp.inferred_edges));
                output.push_str(&format!("  ⚡ Explicit: {}\n", resp.explicit_edges));
                output.push_str(&format!(
                    "Average Confidence: {:.1}%\n\n",
                    resp.avg_confidence * 100.0
                ));
                output.push_str(&format!(
                    "**Branches:** {} total, {} active\n\n",
                    resp.total_branches, resp.active_branches
                ));

                if !resp.edges_by_relation.is_empty() {
                    output.push_str("**By Relation Type:**\n");
                    let mut relations: Vec<_> = resp.edges_by_relation.iter().collect();
                    relations.sort_by(|a, b| b.1.cmp(a.1));
                    for (relation, count) in relations {
                        output.push_str(&format!("  {}: {}\n", relation, count));
                    }
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Detect structural gaps in the knowledge graph. Finds missing connections (open triads), parallel paths not reconciled (diamonds), hub-spoke disconnects (stars), and cluster silos (orbits). Returns raw structural data."
    )]
    async fn gap_analyze(&self) -> Result<CallToolResult, McpError> {
        let result: Result<GapAnalyzeResponse> = self
            .client
            .post(
                "/api/gap/analyze",
                &GapAnalyzeRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Structural Gap Analysis** ({} gaps, {}ms)\n\n",
                    resp.gaps.len(),
                    resp.duration_ms
                );

                for (gap_type, count) in &resp.type_counts {
                    output.push_str(&format!("- {}: {}\n", gap_type, count));
                }
                output.push('\n');

                for gap in &resp.gaps {
                    output.push_str(&format!(
                        "### [{}] {} (confidence: {:.0}%, impact: {:.0}%)\n  Entities: {}\n\n",
                        gap.gap_type,
                        gap.id,
                        gap.confidence * 100.0,
                        gap.impact_score * 100.0,
                        gap.entity_names.join(", ")
                    ));
                }

                if resp.gaps.is_empty() {
                    output.push_str("No structural gaps detected.\n");
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Run Mapper topological analysis on the knowledge graph. Produces a simplified graph that reveals branches, loops, flares, and connected components — topological structure invisible to standard clustering. Different filter functions reveal different aspects of the knowledge topology."
    )]
    async fn gap_mapper(
        &self,
        Parameters(params): Parameters<GapMapperParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<GapMapperResponse> = self
            .client
            .post(
                "/api/gap/mapper",
                &GapMapperRequest {
                    user_id: self.client.user_id.clone(),
                    filter: params.filter,
                    num_intervals: params.num_intervals,
                    overlap: params.overlap,
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Mapper Analysis** (filter: {}, {} entities, {}ms)\n\n",
                    resp.filter, resp.stats.entity_count, resp.stats.duration_ms
                );

                output.push_str(&format!(
                    "**Topology:** {} components, {} loops, {} flares, {} branches\n",
                    resp.num_components, resp.num_loops, resp.flare_count, resp.branch_count
                ));
                output.push_str(&format!(
                    "**Graph:** {} clusters, {} edges\n\n",
                    resp.stats.cluster_count, resp.stats.edge_count
                ));

                if !resp.nodes.is_empty() {
                    output.push_str("**Clusters:**\n");
                    for node in &resp.nodes {
                        let members_preview: Vec<&str> = node
                            .member_names
                            .iter()
                            .take(5)
                            .map(|s| s.as_str())
                            .collect();
                        let suffix = if node.size > 5 {
                            format!(" +{} more", node.size - 5)
                        } else {
                            String::new()
                        };
                        output.push_str(&format!(
                            "  [{}] size={}, filter={:.3}: {}{}\n",
                            node.id,
                            node.size,
                            node.avg_filter_value,
                            members_preview.join(", "),
                            suffix
                        ));
                    }
                }

                if resp.nodes.is_empty() {
                    output.push_str("No clusters found. The knowledge graph may be too sparse for Mapper analysis.\n");
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // W7 DATASET TOOLS - relational tabular storage
    // =========================================================================

    #[tool(
        description = "Create a relational dataset (table) for tabular data. `schema` is a JSON object { name, columns: [{ name, ty, nullable }], primary_key: [..] }, where ty is one of Bool|I64|F64|Text|Bytes|Timestamp|Json. Requires a configured relational backend."
    )]
    async fn dataset_create(
        &self,
        Parameters(params): Parameters<DatasetCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<DatasetCreateResponse> = self
            .client
            .post(
                "/api/datasets",
                &DatasetCreateRequest {
                    user_id: self.client.user_id.clone(),
                    schema: params.schema,
                },
            )
            .await;
        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Created dataset '{}' (table {})",
                resp.dataset.name, resp.dataset.table
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(description = "List the datasets you own, with row counts.")]
    async fn dataset_list(
        &self,
        Parameters(_params): Parameters<DatasetListParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/datasets?user_id={}",
            url_encode_component(&self.client.user_id)
        );
        let result: Result<DatasetListResponse> = self.client.get(&endpoint).await;
        match result {
            Ok(resp) => {
                let mut output = format!("{} dataset(s):\n", resp.total);
                for d in resp.datasets {
                    output.push_str(&format!("- {} ({} rows)\n", d.name, d.row_count));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Run a parametrised SELECT over a dataset. Optional `where_clause` is a SQL fragment (no WHERE keyword) with `?` placeholders bound by `params` in order. Returns columns + rows (first 50 shown)."
    )]
    async fn dataset_query(
        &self,
        Parameters(params): Parameters<DatasetQueryParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/datasets/{}/query", url_encode_component(&params.name));
        let result: Result<DatasetQueryResponse> = self
            .client
            .post(
                &endpoint,
                &DatasetQueryRequest {
                    user_id: self.client.user_id.clone(),
                    where_clause: params.where_clause,
                    params: params.params,
                },
            )
            .await;
        match result {
            Ok(resp) => {
                let mut output = format!("{} | {} row(s)\n", resp.columns.join(", "), resp.rows.len());
                for row in resp.rows.iter().take(50) {
                    let cells: Vec<String> = row.iter().map(render_query_cell).collect();
                    output.push_str(&cells.join(", "));
                    output.push('\n');
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Link a dataset row to a knowledge-graph entity or a memory. `row_pk` is { columns: { <pk_col>: <value> } }, `kind` is 'entity' or 'memory', `target_id` is the entity/memory id."
    )]
    async fn dataset_link(
        &self,
        Parameters(params): Parameters<DatasetLinkParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/datasets/{}/link", url_encode_component(&params.name));
        let result: Result<DatasetLinkResponse> = self
            .client
            .post(
                &endpoint,
                &DatasetLinkRequest {
                    user_id: self.client.user_id.clone(),
                    row_pk: params.row_pk,
                    kind: params.kind,
                    target_id: params.target_id,
                },
            )
            .await;
        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(
                if resp.linked {
                    "Linked.".to_string()
                } else {
                    "Link not created.".to_string()
                },
            )])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }
}

/// Percent-encode a single URL path/query component (RFC 3986 unreserved
/// set passes through; everything else escapes), so dataset names and user
/// ids with reserved characters don't corrupt the request URL.
fn url_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Render a query result cell for the text view: strings unquoted, other
/// JSON values via their compact form.
fn render_query_cell(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "NULL".to_string(),
        other => other.to_string(),
    }
}
