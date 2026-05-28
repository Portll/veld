<!-- GENERATED FILE — do not edit by hand.
     Source: src/handlers/router.rs
     Generator: docs/generators/src/bin/gen-http-api.rs
     Regenerate: cd docs/generators && cargo run --bin gen-http-api -->

# HTTP API

Veld exposes **202** HTTP routes. Every route (except `/health/*` probes) requires API-key authentication via the `X-API-Key` header.

> Warning: 1 route(s) used a non-literal path string and were skipped by the generator. Inspect [src/handlers/router.rs](https://github.com/Portll/veld/blob/main/src/handlers/router.rs) directly for those.

Base URL: `http://127.0.0.1:3030` (default; configurable via `VELD_BIND_ADDR`).

## /ab

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/ab/summary` | `ab_testing::get_ab_summary` |
| `POST` | `/api/ab/tests` | `ab_testing::create_ab_test` |
| `GET` | `/api/ab/tests` | `ab_testing::list_ab_tests` |
| `DELETE` | `/api/ab/tests/{test_id}` | `ab_testing::delete_ab_test` |
| `GET` | `/api/ab/tests/{test_id}` | `ab_testing::get_ab_test` |
| `GET` | `/api/ab/tests/{test_id}/analyze` | `ab_testing::analyze_ab_test` |
| `POST` | `/api/ab/tests/{test_id}/click` | `ab_testing::record_ab_click` |
| `POST` | `/api/ab/tests/{test_id}/complete` | `ab_testing::complete_ab_test` |
| `POST` | `/api/ab/tests/{test_id}/feedback` | `ab_testing::record_ab_feedback` |
| `POST` | `/api/ab/tests/{test_id}/impression` | `ab_testing::record_ab_impression` |
| `POST` | `/api/ab/tests/{test_id}/pause` | `ab_testing::pause_ab_test` |
| `POST` | `/api/ab/tests/{test_id}/resume` | `ab_testing::resume_ab_test` |
| `POST` | `/api/ab/tests/{test_id}/start` | `ab_testing::start_ab_test` |

## /admin

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/admin/reset-rate-limit` | `admin::reset_rate_limit` |

## /anchor

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/anchor` | `crud::anchor_memory` |

## /backup

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/backup/create` | `consolidation::create_backup` |
| `POST` | `/api/backup/list` | `consolidation::list_backups` |
| `POST` | `/api/backup/purge` | `consolidation::purge_backups` |
| `POST` | `/api/backup/restore` | `consolidation::restore_backup` |
| `POST` | `/api/backup/verify` | `consolidation::verify_backup` |

## /backups

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/backups` | `consolidation::list_backups` |
| `POST` | `/api/backups/purge` | `consolidation::purge_backups` |

## /batch_remember

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/batch_remember` | `remember::batch_remember` |

## /brain

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/brain/{user_id}` | `visualization::get_brain_state` |

## /consolidate

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/consolidate` | `consolidation::consolidate_memories` |

## /consolidation

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/consolidation/events` | `consolidation::get_consolidation_events` |
| `POST` | `/api/consolidation/report` | `consolidation::get_consolidation_report` |
| `POST` | `/api/consolidation/sleep` | `consolidation::sleep_phase_consolidation` |

## /context

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/context` | `recall::proactive_context` |
| `GET` | `/api/context/blocks` | `context_blocks::list_context_blocks` |
| `DELETE` | `/api/context/blocks/{key}` | `context_blocks::delete_context_block` |
| `PUT` | `/api/context/blocks/{key}` | `context_blocks::set_context_block` |
| `GET` | `/api/context/blocks/{key}` | `context_blocks::get_context_block` |
| `GET` | `/api/context/monitor` | `webhooks::context_monitor_ws` |
| `GET` | `/api/context/sse` | `webhooks::context_status_sse` |
| `GET` | `/api/context/status` | `health::get_context_status` |
| `POST` | `/api/context/status` | `health::update_context_status` |

## /context_status

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/context_status` | `health::get_context_status` |
| `POST` | `/api/context_status` | `health::update_context_status` |

## /context_summary

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/context_summary` | `recall::context_summary` |

## /entity

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/entity/alias` | `prompt_gen::add_entity_alias` |
| `POST` | `/api/entity/attribute` | `prompt_gen::set_entity_attribute` |
| `POST` | `/api/entity/merge` | `prompt_gen::merge_entities` |
| `POST` | `/api/entity/resolve` | `prompt_gen::resolve_entity` |

## /events

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/events` | `webhooks::memory_events_sse` |
| `GET` | `/api/events/sse` | `webhooks::memory_events_sse` |

## /export

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/export/mif` | `mif::export_mif` |

## /external

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/external/dimensions` | `external_dimensions::push_dimensions` |

## /facts

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/facts/by-entity` | `facts::facts_by_entity` |
| `POST` | `/api/facts/list` | `facts::list_facts` |
| `POST` | `/api/facts/search` | `facts::search_facts` |
| `POST` | `/api/facts/stats` | `facts::get_facts_stats` |
| `POST` | `/api/facts/temporal` | `facts::list_temporal_facts` |
| `POST` | `/api/facts/temporal/search` | `facts::search_temporal_facts` |

## /files

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/files/stats` | `files::get_file_stats` |

## /forget

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/forget` | `crud::forget_by_id` |
| `POST` | `/api/forget/age` | `crud::forget_by_age` |
| `POST` | `/api/forget/date` | `crud::forget_by_date` |
| `POST` | `/api/forget/importance` | `crud::forget_by_importance` |
| `POST` | `/api/forget/pattern` | `crud::forget_by_pattern` |
| `POST` | `/api/forget/tags` | `crud::forget_by_tags` |
| `DELETE` | `/api/forget/{memory_id}` | `crud::delete_memory` |

## /gap

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/gap/analyze` | `gap_analysis::analyze_gaps` |
| `POST` | `/api/gap/mapper` | `gap_analysis::mapper_analysis` |
| `POST` | `/api/gap/persistence` | `gap_analysis::persistence_analysis` |
| `POST` | `/api/gap/stats` | `gap_analysis::gap_stats` |
| `POST` | `/api/gap/voronoi` | `gap_analysis::voronoi_analysis` |

## /graph

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/graph/data/{user_id}` | `visualization::get_graph_data` |
| `POST` | `/api/graph/entities/all` | `graph::get_all_entities` |
| `POST` | `/api/graph/entity/add` | `mif::add_entity` |
| `POST` | `/api/graph/entity/find` | `graph::find_entity` |
| `POST` | `/api/graph/episode/get` | `graph::get_episode` |
| `POST` | `/api/graph/relationship/add` | `mif::add_relationship` |
| `POST` | `/api/graph/relationship/invalidate` | `graph::invalidate_relationship` |
| `POST` | `/api/graph/traverse` | `graph::traverse_graph` |
| `DELETE` | `/api/graph/{user_id}/clear` | `graph::clear_user_graph` |
| `POST` | `/api/graph/{user_id}/rebuild` | `graph::rebuild_user_graph` |
| `GET` | `/api/graph/{user_id}/stats` | `graph::get_graph_stats` |
| `GET` | `/api/graph/{user_id}/universe` | `graph::get_memory_universe` |
| `GET` | `/graph/view` | `visualization::graph_view` |

## /health

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/health/index` | `health::health_index_user` |
| `GET` | `/api/health/ready` | `health::health_ready_user` |
| `GET` | `/health` | `health::health` |
| `GET` | `/health/index` | `health::health_index` |
| `GET` | `/health/live` | `health::health_live` |
| `GET` | `/health/ready` | `health::health_ready` |

## /import

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/import/mif` | `mif::import_mif` |

## /index

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/index/rebuild` | `consolidation::rebuild_index` |
| `POST` | `/api/index/reembed` | `consolidation::reembed_all` |
| `POST` | `/api/index/repair` | `consolidation::repair_vector_index` |
| `POST` | `/api/index/verify` | `consolidation::verify_index_integrity` |

## /ingest

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/ingest` | `ingest::ingest` |

## /lineage

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/lineage/branch` | `lineage::lineage_create_branch` |
| `POST` | `/api/lineage/branches` | `lineage::lineage_list_branches` |
| `POST` | `/api/lineage/confirm` | `lineage::lineage_confirm_edge` |
| `POST` | `/api/lineage/edges` | `lineage::lineage_list_edges` |
| `POST` | `/api/lineage/link` | `lineage::lineage_add_edge` |
| `POST` | `/api/lineage/reject` | `lineage::lineage_reject_edge` |
| `POST` | `/api/lineage/stats` | `lineage::lineage_stats` |
| `POST` | `/api/lineage/trace` | `lineage::lineage_trace` |

## /list

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/list/{user_id}` | `crud::list_memories` |

## /memories

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/memories` | `crud::list_memories_get` |
| `POST` | `/api/memories` | `crud::list_memories_post` |
| `POST` | `/api/memories/bulk` | `crud::bulk_delete_memories` |
| `POST` | `/api/memories/clear` | `crud::clear_all_memories` |
| `GET` | `/api/memories/{memory_id}` | `crud::get_memory` |

## /memory

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/memory/compress` | `compression::compress_memory` |
| `POST` | `/api/memory/decompress` | `compression::decompress_memory` |
| `POST` | `/api/memory/tier` | `crud::move_memory_tier` |
| `DELETE` | `/api/memory/{memory_id}` | `crud::delete_memory` |
| `PUT` | `/api/memory/{memory_id}` | `crud::update_memory` |
| `GET` | `/api/memory/{memory_id}` | `crud::get_memory` |
| `GET` | `/api/memory/{memory_id}/health` | `crud::get_memory_health` |

## /metrics

| Method | Path | Handler |
|---|---|---|
| `GET` | `/metrics` | `health::metrics_endpoint` |

## /mif

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/mif/adapters` | `mif::list_adapters` |

## /proactive_context

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/proactive_context` | `recall::proactive_context` |

## /projects

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/projects` | `todos::create_project` |
| `POST` | `/api/projects/add` | `todos::create_project` |
| `POST` | `/api/projects/list` | `todos::list_projects` |
| `DELETE` | `/api/projects/{project_id}` | `todos::delete_project` |
| `GET` | `/api/projects/{project_id}` | `todos::get_project` |
| `POST` | `/api/projects/{project_id}/delete` | `todos::delete_project` |
| `POST` | `/api/projects/{project_id}/files` | `files::list_project_files` |
| `POST` | `/api/projects/{project_id}/files/search` | `files::search_project_files` |
| `POST` | `/api/projects/{project_id}/index` | `files::index_project_codebase` |
| `POST` | `/api/projects/{project_id}/scan` | `files::scan_project_codebase` |
| `POST` | `/api/projects/{project_id}/update` | `todos::update_project` |

## /prompt

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/prompt/gen` | `prompt_gen::prompt_gen` |
| `POST` | `/api/prompt/generate` | `prompt_gen::prompt_gen` |

## /recall

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/recall` | `recall::recall` |
| `POST` | `/api/recall/by-tags` | `recall::recall_by_tags` |
| `POST` | `/api/recall/date` | `recall::recall_by_date` |
| `POST` | `/api/recall/tags` | `recall::recall_by_tags` |
| `POST` | `/api/recall/tracked` | `recall::recall_tracked` |

## /reinforce

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/reinforce` | `recall::reinforce_feedback` |

## /relevant

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/relevant` | `recall::surface_relevant` |

## /remember

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/remember` | `remember::remember` |
| `POST` | `/api/remember/batch` | `remember::batch_remember` |

## /remind

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/remind` | `todos::create_reminder` |

## /reminders

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/reminders` | `todos::list_reminders` |
| `POST` | `/api/reminders/check` | `todos::check_context_reminders` |
| `POST` | `/api/reminders/context` | `todos::check_context_reminders` |
| `POST` | `/api/reminders/due` | `todos::get_due_reminders` |
| `POST` | `/api/reminders/set` | `todos::create_reminder` |
| `POST` | `/api/reminders/{reminder_id}/delete` | `todos::delete_reminder` |
| `POST` | `/api/reminders/{reminder_id}/dismiss` | `todos::dismiss_reminder` |

## /search

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/search/advanced` | `search::advanced_search` |
| `POST` | `/api/search/multimodal` | `search::multimodal_search` |
| `POST` | `/api/search/robotics` | `search::robotics_search` |

## /seed

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/seed` | `seed::seed_project` |

## /sessions

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/sessions` | `sessions::list_sessions` |
| `POST` | `/api/sessions/end` | `sessions::end_session` |
| `GET` | `/api/sessions/stats` | `sessions::get_session_stats` |
| `GET` | `/api/sessions/{session_id}` | `sessions::get_session` |

## /sleight

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/sleight/dimensions` | `external_dimensions::push_dimensions` |

## /stats

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/stats` | `users::get_stats_query` |

## /storage

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/storage/cleanup` | `consolidation::cleanup_corrupted` |
| `POST` | `/api/storage/migrate` | `consolidation::migrate_legacy` |
| `GET` | `/api/storage/stats` | `compression::get_storage_stats` |
| `POST` | `/api/storage/uncompressed` | `mif::get_uncompressed_old` |

## /stream

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/stream` | `webhooks::streaming_memory_ws` |

## /sync

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/sync/github` | `integrations::github_sync` |
| `POST` | `/api/sync/linear` | `integrations::linear_sync` |

## /todos

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/todos` | `todos::list_todos` |
| `POST` | `/api/todos/add` | `todos::create_todo` |
| `POST` | `/api/todos/complete` | `todos::complete_todo` |
| `POST` | `/api/todos/delete` | `todos::delete_todo` |
| `POST` | `/api/todos/due` | `todos::list_due_todos` |
| `POST` | `/api/todos/list` | `todos::list_todos` |
| `POST` | `/api/todos/ready` | `todos::list_ready_todos` |
| `POST` | `/api/todos/reorder` | `todos::reorder_todo` |
| `POST` | `/api/todos/stats` | `todos::get_todo_stats` |
| `POST` | `/api/todos/update` | `todos::update_todo` |
| `DELETE` | `/api/todos/{todo_id}` | `todos::delete_todo` |
| `GET` | `/api/todos/{todo_id}` | `todos::get_todo` |
| `POST` | `/api/todos/{todo_id}/comments` | `todos::add_todo_comment` |
| `GET` | `/api/todos/{todo_id}/comments` | `todos::list_todo_comments` |
| `DELETE` | `/api/todos/{todo_id}/comments/{comment_id}` | `todos::delete_todo_comment` |
| `PUT` | `/api/todos/{todo_id}/comments/{comment_id}` | `todos::update_todo_comment` |
| `POST` | `/api/todos/{todo_id}/comments/{comment_id}/update` | `todos::update_todo_comment` |
| `POST` | `/api/todos/{todo_id}/complete` | `todos::complete_todo` |
| `GET` | `/api/todos/{todo_id}/dependency_chain` | `todos::dependency_chain` |
| `GET` | `/api/todos/{todo_id}/dependents` | `todos::list_dependents` |
| `POST` | `/api/todos/{todo_id}/reorder` | `todos::reorder_todo` |
| `GET` | `/api/todos/{todo_id}/subtasks` | `todos::list_subtasks` |
| `POST` | `/api/todos/{todo_id}/update` | `todos::update_todo` |

## /upsert

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/upsert` | `remember::upsert_memory` |

## /user_auth

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/user_auth/2fa/confirm` | `user_auth::confirm_2fa` |
| `POST` | `/api/user_auth/2fa/enroll` | `user_auth::enroll_2fa` |
| `POST` | `/api/user_auth/login` | `user_auth::login` |
| `POST` | `/api/user_auth/logout` | `user_auth::logout` |
| `POST` | `/api/user_auth/recover` | `user_auth::recover` |
| `POST` | `/api/user_auth/register` | `user_auth::register` |

## /users

| Method | Path | Handler |
|---|---|---|
| `GET` | `/api/users` | `users::list_users` |
| `DELETE` | `/api/users/{user_id}` | `users::delete_user` |
| `GET` | `/api/users/{user_id}/stats` | `users::get_user_stats` |

## /visualization

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/visualization/build` | `visualization::build_visualization` |
| `GET` | `/api/visualization/{user_id}/dot` | `visualization::get_visualization_dot` |
| `GET` | `/api/visualization/{user_id}/stats` | `visualization::get_visualization_stats` |

## /webhook

| Method | Path | Handler |
|---|---|---|
| `POST` | `/webhook/github` | `integrations::github_webhook` |
| `POST` | `/webhook/linear` | `integrations::linear_webhook` |

## /wintermute

| Method | Path | Handler |
|---|---|---|
| `POST` | `/api/wintermute/dimensions` | `external_dimensions::push_dimensions` |

---

*Handlers live in `src/handlers/*.rs`. For request/response shapes, see the corresponding handler source or run `veld serve` and call `OPTIONS /api/...` (where supported).*
