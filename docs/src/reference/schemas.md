<!-- GENERATED FILE — do not edit by hand.
     Source: src/handlers/router.rs and src/handlers/*.rs
     Generator: docs/generators/src/bin/gen-schemas.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-schemas`) -->

# HTTP API schemas

The request and response shape of every HTTP route, read from the handler signatures and the types they name: **273** routes over **570** types. `schemas.json` beside this page holds the same facts for tools and tests.

**How to read a type.** *Required* means the field has no `Option` and no serde default, so a request without it is rejected and a response always carries it. A field marked *omitted when empty* is left out of a response when it has no value. Enum variants are listed in wire form. `DateTime<Utc>` is an RFC 3339 string, `Uuid` and `MemoryId` are UUID strings, and `Value` is any JSON. A response shown as *opaque* is built by hand in the handler; read the handler for it.

> 1 route registration(s) use a non-literal path or an unrecognised method chain and are not listed; see src/handlers/router.rs.

## Routes

### /ab

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/ab/summary` | none | [`AbSummaryResponse`](#absummaryresponse) |
| `GET` | `/api/ab/tests` | none | [`AbTestListResponse`](#abtestlistresponse) |
| `POST` | `/api/ab/tests` | body [`CreateABTestRequest`](#createabtestrequest) | [`ABTestResponse`](#abtestresponse) |
| `DELETE` | `/api/ab/tests/{test_id}` | path `{test_id: String}` | [`ABTestResponse`](#abtestresponse) |
| `GET` | `/api/ab/tests/{test_id}` | path `{test_id: String}` | [`AbTestDetailResponse`](#abtestdetailresponse) |
| `GET` | `/api/ab/tests/{test_id}/analyze` | path `{test_id: String}` | [`AbTestAnalysisResponse`](#abtestanalysisresponse) |
| `POST` | `/api/ab/tests/{test_id}/click` | body [`RecordClickRequest`](#recordclickrequest); path `{test_id: String}` | [`AbRecordResponse`](#abrecordresponse) |
| `POST` | `/api/ab/tests/{test_id}/complete` | path `{test_id: String}` | [`AbTestCompleteResponse`](#abtestcompleteresponse) |
| `POST` | `/api/ab/tests/{test_id}/feedback` | body [`RecordFeedbackRequest`](#recordfeedbackrequest); path `{test_id: String}` | [`AbRecordResponse`](#abrecordresponse) |
| `POST` | `/api/ab/tests/{test_id}/impression` | body [`RecordImpressionRequest`](#recordimpressionrequest); path `{test_id: String}` | [`AbImpressionResponse`](#abimpressionresponse) |
| `POST` | `/api/ab/tests/{test_id}/pause` | path `{test_id: String}` | [`ABTestResponse`](#abtestresponse) |
| `POST` | `/api/ab/tests/{test_id}/resume` | path `{test_id: String}` | [`ABTestResponse`](#abtestresponse) |
| `POST` | `/api/ab/tests/{test_id}/start` | path `{test_id: String}` | [`ABTestResponse`](#abtestresponse) |

### /admin

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/admin/replication/follow` | none | `opaque` |
| `GET` | `/api/admin/replication/records` | query [`RecordsQuery`](#recordsquery) | `opaque` |
| `GET` | `/api/admin/replication/status` | none | `opaque` |
| `POST` | `/api/admin/replication/unfollow` | none | `opaque` |
| `POST` | `/api/admin/reset-rate-limit` | none | `opaque` |
| `POST` | `/api/admin/storage/export` | none | `opaque` |
| `POST` | `/api/admin/storage/import` | none | `opaque` |
| `POST` | `/api/admin/storage/migrate` | none | `opaque` |

### /anchor

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/anchor` | body [`AnchorRequest`](#anchorrequest) | [`AnchorResponse`](#anchorresponse) |

### /backup

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/backup/create` | body [`CreateBackupRequest`](#createbackuprequest) | [`BackupResponse`](#backupresponse) |
| `POST` | `/api/backup/list` | body [`ListBackupsRequest`](#listbackupsrequest) | [`ListBackupsResponse`](#listbackupsresponse) |
| `POST` | `/api/backup/purge` | body [`PurgeBackupsRequest`](#purgebackupsrequest) | [`PurgeBackupsResponse`](#purgebackupsresponse) |
| `POST` | `/api/backup/restore` | body [`RestoreBackupRequest`](#restorebackuprequest) | [`RestoreBackupResponse`](#restorebackupresponse) |
| `POST` | `/api/backup/verify` | body [`VerifyBackupRequest`](#verifybackuprequest) | [`VerifyBackupResponse`](#verifybackupresponse) |

### /backups

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/backups` | body [`ListBackupsRequest`](#listbackupsrequest) | [`ListBackupsResponse`](#listbackupsresponse) |
| `POST` | `/api/backups/purge` | body [`PurgeBackupsRequest`](#purgebackupsrequest) | [`PurgeBackupsResponse`](#purgebackupsresponse) |

### /batch_remember

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/batch_remember` | body [`BatchRememberRequest`](#batchrememberrequest) | [`BatchRememberResponse`](#batchrememberresponse) |

### /brain

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/brain/{user_id}` | path `{user_id: String}` | [`BrainStateResponse`](#brainstateresponse) |

### /consolidate

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/consolidate` | body [`ConsolidateRequest`](#consolidaterequest) | [`ConsolidateResponse`](#consolidateresponse) |

### /consolidation

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/consolidation/events` | query [`ConsolidationEventsRequest`](#consolidationeventsrequest) | Vec&lt;memory::[`ConsolidationEvent`](#consolidationevent)&gt; |
| `POST` | `/api/consolidation/report` | body [`ConsolidationReportRequest`](#consolidationreportrequest) | memory::[`ConsolidationReport`](#consolidationreport) |
| `POST` | `/api/consolidation/sleep` | body [`SleepPhaseRequest`](#sleepphaserequest) | [`SleepPhaseResponse`](#sleepphaseresponse) |

### /context

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/context` | body [`ProactiveContextRequest`](#proactivecontextrequest) | [`ProactiveContextResponse`](#proactivecontextresponse) |
| `GET` | `/api/context/blocks` | query [`BlockQuery`](#blockquery) | [`BlockListResponse`](#blocklistresponse) |
| `DELETE` | `/api/context/blocks/{key}` | query [`BlockQuery`](#blockquery); path `{key: String}` | [`BlockDeleteResponse`](#blockdeleteresponse) |
| `GET` | `/api/context/blocks/{key}` | query [`BlockQuery`](#blockquery); path `{key: String}` | [`BlockResponse`](#blockresponse) |
| `PUT` | `/api/context/blocks/{key}` | body [`SetBlockRequest`](#setblockrequest); path `{key: String}` | [`BlockResponse`](#blockresponse) |
| `GET` | `/api/context/monitor` | none | `opaque` |
| `GET` | `/api/context/sse` | none | `Sse<impl futures::Stream<Item = Result<Event, Infallible>>>` |
| `GET` | `/api/context/status` | none | Vec&lt;[`ContextStatus`](#contextstatus)&gt; |
| `POST` | `/api/context/status` | body [`ContextStatusRequest`](#contextstatusrequest) | [`ContextStatusUpdateResponse`](#contextstatusupdateresponse) |

### /context_status

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/context_status` | none | Vec&lt;[`ContextStatus`](#contextstatus)&gt; |
| `POST` | `/api/context_status` | body [`ContextStatusRequest`](#contextstatusrequest) | [`ContextStatusUpdateResponse`](#contextstatusupdateresponse) |

### /context_summary

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/context_summary` | body [`ContextSummaryRequest`](#contextsummaryrequest) | [`ContextSummaryResponse`](#contextsummaryresponse) |

### /datasets

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/datasets` | query [`DatasetUserQuery`](#datasetuserquery) | [`ListDatasetsResponse`](#listdatasetsresponse) |
| `POST` | `/api/datasets` | body [`CreateDatasetRequest`](#createdatasetrequest) | [`CreateDatasetResponse`](#createdatasetresponse) |
| `DELETE` | `/api/datasets/{name}` | query [`DatasetUserQuery`](#datasetuserquery); path `{name: String}` | [`DropDatasetResponse`](#dropdatasetresponse) |
| `GET` | `/api/datasets/{name}` | query [`DatasetUserQuery`](#datasetuserquery); path `{name: String}` | [`DatasetMetadataResponse`](#datasetmetadataresponse) |
| `POST` | `/api/datasets/{name}/link` | body [`LinkRequest`](#linkrequest); path `{name: String}` | [`LinkResponse`](#linkresponse) |
| `POST` | `/api/datasets/{name}/query` | body [`QueryRequest`](#queryrequest); path `{name: String}` | `opaque` |
| `POST` | `/api/datasets/{name}/rows` | body [`InsertRowsRequest`](#insertrowsrequest); path `{name: String}` | [`InsertRowsResponse`](#insertrowsresponse) |

### /dials

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/dials` | query [`DialsQuery`](#dialsquery) | [`CognitiveDials`](#cognitivedials) |
| `PUT` | `/api/dials` | body [`SetDialsRequest`](#setdialsrequest) | [`CognitiveDials`](#cognitivedials) |
| `GET` | `/api/dials/schema` | none | [`DialsSchemaResponse`](#dialsschemaresponse) |

### /embeddings

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/embeddings/backlog` | query [`EmbeddingBacklogQuery`](#embeddingbacklogquery) | crate::memory::embedding_backlog::[`BacklogReport`](#backlogreport) |
| `POST` | `/api/embeddings/backlog/drain` | body [`DrainEmbeddingBacklogRequest`](#drainembeddingbacklogrequest) | [`DrainBacklogResponse`](#drainbacklogresponse) |

### /entity

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/entity/alias` | body [`AddAliasRequest`](#addaliasrequest) | [`AddAliasResponse`](#addaliasresponse) |
| `POST` | `/api/entity/attribute` | body [`SetAttributeRequest`](#setattributerequest) | [`SetAttributeResponse`](#setattributeresponse) |
| `POST` | `/api/entity/merge` | body [`MergeEntitiesRequest`](#mergeentitiesrequest) | [`MergeEntitiesResponse`](#mergeentitiesresponse) |
| `POST` | `/api/entity/resolve` | body [`EntityResolveRequest`](#entityresolverequest) | [`EntityResolveResponse`](#entityresolveresponse) |

### /events

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/events` | query [`SseQuery`](#ssequery) | `Sse<impl futures::Stream<Item = Result<Event, Infallible>>>` |
| `GET` | `/api/events/sse` | query [`SseQuery`](#ssequery) | `Sse<impl futures::Stream<Item = Result<Event, Infallible>>>` |

### /export

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/export/mif` | body [`MifExportRequest`](#mifexportrequest) | [`ExportMifResponse`](#exportmifresponse) |

### /external

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/external/dimensions` | body [`SleightDimensionsRequest`](#sleightdimensionsrequest) | [`SleightDimensionsResponse`](#sleightdimensionsresponse) |

### /facts

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/facts/by-entity` | body [`FactsByEntityRequest`](#factsbyentityrequest) | [`FactsResponse`](#factsresponse) |
| `POST` | `/api/facts/list` | body [`FactsListRequest`](#factslistrequest) | [`FactsResponse`](#factsresponse) |
| `POST` | `/api/facts/narratives` | body [`FactNarrativesRequest`](#factnarrativesrequest) | [`FactNarrativesResponse`](#factnarrativesresponse) |
| `POST` | `/api/facts/preview-purge` | body [`FactsPreviewPurgeRequest`](#factspreviewpurgerequest) | [`FactsPreviewPurgeResponse`](#factspreviewpurgeresponse) |
| `POST` | `/api/facts/purge` | body [`FactsPurgeRequest`](#factspurgerequest) | [`FactsPurgeResponse`](#factspurgeresponse) |
| `POST` | `/api/facts/search` | body [`FactsSearchRequest`](#factssearchrequest) | [`FactsResponse`](#factsresponse) |
| `POST` | `/api/facts/stats` | body [`FactsListRequest`](#factslistrequest) | memory::[`FactStats`](#factstats) |
| `POST` | `/api/facts/temporal` | body [`TemporalFactsRequest`](#temporalfactsrequest) | [`TemporalFactsResponse`](#temporalfactsresponse) |
| `POST` | `/api/facts/temporal/search` | body [`TemporalFactsSearchRequest`](#temporalfactssearchrequest) | [`TemporalFactsResponse`](#temporalfactsresponse) |

### /file_answer

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/file_answer` | body [`FileAnswerRequest`](#fileanswerrequest) | [`FileAnswerResponse`](#fileanswerresponse) |

### /files

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/files/stats` | query [`TodoQuery`](#todoquery) | [`FileStatsResponse`](#filestatsresponse) |

### /forget

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/forget` | body [`ForgetByIdRequest`](#forgetbyidrequest) | [`DeleteMemoryResponse`](#deletememoryresponse) |
| `POST` | `/api/forget/age` | body [`ForgetByAgeRequest`](#forgetbyagerequest) | [`ForgetByCriteriaResponse`](#forgetbycriteriaresponse) |
| `POST` | `/api/forget/date` | body [`ForgetByDateRequest`](#forgetbydaterequest) | [`ForgetByDateResponse`](#forgetbydateresponse) |
| `POST` | `/api/forget/importance` | body [`ForgetByImportanceRequest`](#forgetbyimportancerequest) | [`ForgetByCriteriaResponse`](#forgetbycriteriaresponse) |
| `POST` | `/api/forget/pattern` | body [`ForgetByPatternRequest`](#forgetbypatternrequest) | [`ForgetByPatternResponse`](#forgetbypatternresponse) |
| `POST` | `/api/forget/tags` | body [`ForgetByTagsRequest`](#forgetbytagsrequest) | [`ForgetByTagsResponse`](#forgetbytagsresponse) |
| `DELETE` | `/api/forget/{memory_id}` | query `HashMap<String, String>`; path `{memory_id: String}` | [`DeleteMemoryResponse`](#deletememoryresponse) |

### /gap

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/gap/analyze` | body [`GapAnalysisRequest`](#gapanalysisrequest) | [`GapAnalysisResponse`](#gapanalysisresponse) |
| `POST` | `/api/gap/mapper` | body [`MapperRequest`](#mapperrequest) | [`MapperResponse`](#mapperresponse) |
| `POST` | `/api/gap/persistence` | body [`PersistenceRequest`](#persistencerequest) | [`PersistenceResponse`](#persistenceresponse) |
| `POST` | `/api/gap/stats` | body [`GapStatsRequest`](#gapstatsrequest) | [`GapStatsResponse`](#gapstatsresponse) |
| `POST` | `/api/gap/voronoi` | body [`VoronoiAnalysisRequest`](#voronoianalysisrequest) | [`VoronoiAnalysisResponse`](#voronoianalysisresponse) |

### /graph

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/graph/data/{user_id}` | path `{user_id: String}` | [`GraphDataResponse`](#graphdataresponse) |
| `POST` | `/api/graph/entities/all` | body [`GetAllEntitiesRequest`](#getallentitiesrequest) | [`GraphEntitiesResponse`](#graphentitiesresponse) |
| `POST` | `/api/graph/entities/frequency` | body [`EntityFrequencyRequest`](#entityfrequencyrequest) | [`Frequency`](#frequency) |
| `POST` | `/api/graph/entity/add` | body [`AddEntityRequest`](#addentityrequest) | [`AddEntityResponse`](#addentityresponse) |
| `DELETE` | `/api/graph/entity/delete` | body [`DeleteEntityRequest`](#deleteentityrequest) | [`DeleteEntityResponse`](#deleteentityresponse) |
| `POST` | `/api/graph/entity/find` | body [`FindEntityRequest`](#findentityrequest) | Option&lt;[`EntityNode`](#entitynode)&gt; |
| `POST` | `/api/graph/episode/get` | body [`GetEpisodeRequest`](#getepisoderequest) | Option&lt;[`EpisodicNode`](#episodicnode)&gt; |
| `POST` | `/api/graph/relationship/add` | body [`AddRelationshipRequest`](#addrelationshiprequest) | [`AddRelationshipResponse`](#addrelationshipresponse) |
| `POST` | `/api/graph/relationship/invalidate` | body [`InvalidateRelationshipRequest`](#invalidaterelationshiprequest) | [`InvalidateRelationshipResponse`](#invalidaterelationshipresponse) |
| `POST` | `/api/graph/tags/stamped-cores` | body [`StampedCoresRequest`](#stampedcoresrequest) | [`StampedCores`](#stampedcores) |
| `POST` | `/api/graph/traverse` | body [`TraverseGraphRequest`](#traversegraphrequest) | [`GraphTraversal`](#graphtraversal) |
| `DELETE` | `/api/graph/{user_id}/clear` | path `{user_id: String}` | [`ClearGraphResponse`](#cleargraphresponse) |
| `POST` | `/api/graph/{user_id}/rebuild` | path `{user_id: String}` | [`RebuildGraphResponse`](#rebuildgraphresponse) |
| `GET` | `/api/graph/{user_id}/stats` | path `{user_id: String}` | [`GraphStats`](#graphstats) |
| `GET` | `/api/graph/{user_id}/universe` | path `{user_id: String}` | [`MemoryUniverse`](#memoryuniverse) |
| `GET` | `/graph/view` | query [`GraphViewParams`](#graphviewparams) | `opaque` |

### /health

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/health/index` | query `HashMap<String, String>` | [`UserIndexHealthResponse`](#userindexhealthresponse) |
| `GET` | `/api/health/ready` | query `HashMap<String, String>` | [`UserReadyResponse`](#userreadyresponse) |
| `GET` | `/health` | none | [`HealthResponse`](#healthresponse) |
| `GET` | `/health/index` | none | [`IndexHealthResponse`](#indexhealthresponse) |
| `GET` | `/health/live` | none | [`LiveResponse`](#liveresponse) |
| `GET` | `/health/ready` | none | [`ReadyResponse`](#readyresponse) |

### /import

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/import/mif` | body [`MifImportRequest`](#mifimportrequest) | [`MifImportResponse`](#mifimportresponse) |

### /index

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/index/rebuild` | body [`RebuildIndexRequest`](#rebuildindexrequest) | [`RebuildIndexResponse`](#rebuildindexresponse) |
| `POST` | `/api/index/reembed` | body [`RebuildIndexRequest`](#rebuildindexrequest) | [`ReembedResponse`](#reembedresponse) |
| `POST` | `/api/index/repair` | body [`RepairIndexRequest`](#repairindexrequest) | [`RepairIndexResponse`](#repairindexresponse) |
| `POST` | `/api/index/verify` | body [`VerifyIndexRequest`](#verifyindexrequest) | memory::[`IndexIntegrityReport`](#indexintegrityreport) |

### /ingest

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/ingest` | body [`IngestRequest`](#ingestrequest) | [`IngestResponse`](#ingestresponse) |

### /lineage

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/lineage/branch` | body [`LineageCreateBranchRequest`](#lineagecreatebranchrequest) | [`LineageBranch`](#lineagebranch) |
| `POST` | `/api/lineage/branches` | body [`LineageListRequest`](#lineagelistrequest) | [`LineageBranchesResponse`](#lineagebranchesresponse) |
| `POST` | `/api/lineage/confirm` | body [`LineageEdgeRequest`](#lineageedgerequest) | [`LineageConfirmResponse`](#lineageconfirmresponse) |
| `POST` | `/api/lineage/edges` | body [`LineageListRequest`](#lineagelistrequest) | [`LineageEdgesResponse`](#lineageedgesresponse) |
| `POST` | `/api/lineage/link` | body [`LineageAddEdgeRequest`](#lineageaddedgerequest) | [`LineageEdge`](#lineageedge) |
| `POST` | `/api/lineage/reject` | body [`LineageEdgeRequest`](#lineageedgerequest) | [`LineageRejectResponse`](#lineagerejectresponse) |
| `POST` | `/api/lineage/stats` | body [`LineageListRequest`](#lineagelistrequest) | [`LineageStats`](#lineagestats) |
| `POST` | `/api/lineage/trace` | body [`LineageTraceRequest`](#lineagetracerequest) | [`LineageTraceResponse`](#lineagetraceresponse) |

### /list

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/list/{user_id}` | query [`ListQuery`](#listquery); path `{user_id: String}` | [`ListResponse`](#listresponse) |

### /memories

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/memories` | query [`ListMemoriesQuery`](#listmemoriesquery) | [`ListResponse`](#listresponse) |
| `POST` | `/api/memories` | body [`ListMemoriesRequest`](#listmemoriesrequest) | [`ListResponse`](#listresponse) |
| `POST` | `/api/memories/bulk` | body [`BulkDeleteRequest`](#bulkdeleterequest) | [`BulkDeleteResponse`](#bulkdeleteresponse) |
| `POST` | `/api/memories/clear` | body [`ClearAllRequest`](#clearallrequest) | [`ClearAllResponse`](#clearallresponse) |
| `GET` | `/api/memories/count` | query [`CountMemoriesQuery`](#countmemoriesquery) | [`CountResponse`](#countresponse) |
| `GET` | `/api/memories/{memory_id}` | query `HashMap<String, String>`; path `{memory_id: String}` | [`MemoryWithHierarchy`](#memorywithhierarchy) |

### /memory

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/memory/by-external-id` | query `HashMap<String, String>` | [`ExternalIdRecord`](#externalidrecord) |
| `POST` | `/api/memory/compress` | body [`CompressMemoryRequest`](#compressmemoryrequest) | [`CompressMemoryResponse`](#compressmemoryresponse) |
| `POST` | `/api/memory/decompress` | body [`DecompressMemoryRequest`](#decompressmemoryrequest) | [`DecompressMemoryResponse`](#decompressmemoryresponse) |
| `POST` | `/api/memory/merge` | body [`MergeRequest`](#mergerequest) | [`MergeMemoriesResponse`](#mergememoriesresponse) |
| `POST` | `/api/memory/split` | body [`SplitRequest`](#splitrequest) | [`SplitMemoryResponse`](#splitmemoryresponse) |
| `POST` | `/api/memory/tier` | body [`TierMoveRequest`](#tiermoverequest) | [`TierMoveResponse`](#tiermoveresponse) |
| `DELETE` | `/api/memory/{memory_id}` | query `HashMap<String, String>`; path `{memory_id: String}` | [`DeleteMemoryResponse`](#deletememoryresponse) |
| `GET` | `/api/memory/{memory_id}` | query `HashMap<String, String>`; path `{memory_id: String}` | [`MemoryWithHierarchy`](#memorywithhierarchy) |
| `PUT` | `/api/memory/{memory_id}` | body [`UpdateMemoryRequest`](#updatememoryrequest); path `{memory_id: String}` | [`UpdateMemoryResponse`](#updatememoryresponse) |
| `GET` | `/api/memory/{memory_id}/health` | query `HashMap<String, String>`; path `{memory_id: String}` | [`MemoryHealthResponse`](#memoryhealthresponse) |

### /metrics

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/metrics` | none | `String` |

### /mif

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/mif/adapters` | none | [`MifAdaptersResponse`](#mifadaptersresponse) |

### /proactive_context

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/proactive_context` | body [`ProactiveContextRequest`](#proactivecontextrequest) | [`ProactiveContextResponse`](#proactivecontextresponse) |

### /projects

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/projects` | body [`CreateProjectRequest`](#createprojectrequest) | [`ProjectResponse`](#projectresponse) |
| `POST` | `/api/projects/add` | body [`CreateProjectRequest`](#createprojectrequest) | [`ProjectResponse`](#projectresponse) |
| `POST` | `/api/projects/list` | body [`ListProjectsRequest`](#listprojectsrequest) | [`ProjectListResponse`](#projectlistresponse) |
| `DELETE` | `/api/projects/{project_id}` | body [`DeleteProjectRequest`](#deleteprojectrequest); path `{project_id: String}` | [`ProjectResponse`](#projectresponse) |
| `GET` | `/api/projects/{project_id}` | query [`TodoQuery`](#todoquery); path `{project_id: String}` | [`ProjectResponse`](#projectresponse) |
| `POST` | `/api/projects/{project_id}/delete` | body [`DeleteProjectRequest`](#deleteprojectrequest); path `{project_id: String}` | [`ProjectResponse`](#projectresponse) |
| `POST` | `/api/projects/{project_id}/files` | body [`ListFilesRequest`](#listfilesrequest); path `{project_id: String}` | [`FileListResponse`](#filelistresponse) |
| `POST` | `/api/projects/{project_id}/files/search` | body [`SearchFilesRequest`](#searchfilesrequest); path `{project_id: String}` | [`FileListResponse`](#filelistresponse) |
| `POST` | `/api/projects/{project_id}/index` | body [`IndexCodebaseRequest`](#indexcodebaserequest); path `{project_id: String}` | [`IndexResponse`](#indexresponse) |
| `POST` | `/api/projects/{project_id}/scan` | body [`IndexCodebaseRequest`](#indexcodebaserequest); path `{project_id: String}` | [`ScanResponse`](#scanresponse) |
| `POST` | `/api/projects/{project_id}/update` | body [`UpdateProjectRequest`](#updateprojectrequest); path `{project_id: String}` | [`ProjectResponse`](#projectresponse) |

### /prompt

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/prompt/gen` | body [`PromptGenRequest`](#promptgenrequest) | [`PromptGenResponse`](#promptgenresponse) |
| `POST` | `/api/prompt/generate` | body [`PromptGenRequest`](#promptgenrequest) | [`PromptGenResponse`](#promptgenresponse) |

### /query

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/query/execute` | body [`Query`](#query) | [`ExecuteResponse`](#executeresponse) |
| `POST` | `/api/query/plan` | body [`Query`](#query) | [`PlanResponse`](#planresponse) |

### /recall

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/recall` | body [`RecallRequest`](#recallrequest) | [`RecallResponse`](#recallresponse) |
| `POST` | `/api/recall/by-tags` | body [`RecallByTagsRequest`](#recallbytagsrequest) | [`RetrieveResponse`](#retrieveresponse) |
| `POST` | `/api/recall/date` | body [`RecallByDateRequest`](#recallbydaterequest) | [`RetrieveResponse`](#retrieveresponse) |
| `POST` | `/api/recall/tags` | body [`RecallByTagsRequest`](#recallbytagsrequest) | [`RetrieveResponse`](#retrieveresponse) |
| `POST` | `/api/recall/tracked` | body [`TrackedRetrieveRequest`](#trackedretrieverequest) | [`TrackedRetrieveResponse`](#trackedretrieveresponse) |

### /reinforce

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/reinforce` | body [`ReinforceFeedbackRequest`](#reinforcefeedbackrequest) | [`ReinforceFeedbackResponse`](#reinforcefeedbackresponse) |

### /relevant

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/relevant` | body relevance::[`RelevanceRequest`](#relevancerequest) | relevance::[`RelevanceResponse`](#relevanceresponse) |

### /remember

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/remember` | body [`RememberRequest`](#rememberrequest) | [`RememberResponse`](#rememberresponse) |
| `POST` | `/api/remember/batch` | body [`BatchRememberRequest`](#batchrememberrequest) | [`BatchRememberResponse`](#batchrememberresponse) |

### /remind

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/remind` | body [`CreateReminderRequest`](#createreminderrequest) | [`CreateReminderResponse`](#createreminderresponse) |

### /reminders

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/reminders` | body [`ListRemindersRequest`](#listremindersrequest) | [`ListRemindersResponse`](#listremindersresponse) |
| `POST` | `/api/reminders/check` | body [`CheckContextRemindersRequest`](#checkcontextremindersrequest) | [`DueRemindersResponse`](#dueremindersresponse) |
| `POST` | `/api/reminders/context` | body [`CheckContextRemindersRequest`](#checkcontextremindersrequest) | [`DueRemindersResponse`](#dueremindersresponse) |
| `POST` | `/api/reminders/due` | body [`GetDueRemindersRequest`](#getdueremindersrequest) | [`DueRemindersResponse`](#dueremindersresponse) |
| `POST` | `/api/reminders/set` | body [`CreateReminderRequest`](#createreminderrequest) | [`CreateReminderResponse`](#createreminderresponse) |
| `POST` | `/api/reminders/{reminder_id}/delete` | query [`DeleteReminderQuery`](#deletereminderquery); path `{reminder_id: String}` | [`ReminderActionResponse`](#reminderactionresponse) |
| `POST` | `/api/reminders/{reminder_id}/dismiss` | body [`DismissReminderRequest`](#dismissreminderrequest); path `{reminder_id: String}` | [`ReminderActionResponse`](#reminderactionresponse) |

### /search

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/search/advanced` | body [`AdvancedSearchRequest`](#advancedsearchrequest) | [`RetrieveResponse`](#retrieveresponse) |
| `POST` | `/api/search/multimodal` | body [`MultiModalSearchRequest`](#multimodalsearchrequest) | [`RetrieveResponse`](#retrieveresponse) |
| `POST` | `/api/search/robotics` | body [`RoboticsSearchRequest`](#roboticssearchrequest) | [`RetrieveResponse`](#retrieveresponse) |

### /seed

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/seed` | body [`SeedRequest`](#seedrequest) | [`SeedResponse`](#seedresponse) |

### /sessions

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/sessions` | body [`ListSessionsRequest`](#listsessionsrequest) | [`ListSessionsResponse`](#listsessionsresponse) |
| `POST` | `/api/sessions/end` | body [`EndSessionRequest`](#endsessionrequest) | [`EndSessionResponse`](#endsessionresponse) |
| `GET` | `/api/sessions/stats` | none | [`SessionStoreStatsResponse`](#sessionstorestatsresponse) |
| `GET` | `/api/sessions/{session_id}` | query [`GetSessionRequest`](#getsessionrequest); path `{session_id: String}` | [`GetSessionResponse`](#getsessionresponse) |

### /sleep_time

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/sleep_time/enqueue` | body [`EnqueueRequest`](#enqueuerequest) | `opaque` |
| `POST` | `/api/sleep_time/lock_block` | body [`LockBlockRequest`](#lockblockrequest) | `opaque` |
| `GET` | `/api/sleep_time/status` | none | `opaque` |
| `GET` | `/api/sleep_time/status/{user_id}` | path `{user_id: String}` | `opaque` |

### /sleight

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/sleight/dimensions` | body [`SleightDimensionsRequest`](#sleightdimensionsrequest) | [`SleightDimensionsResponse`](#sleightdimensionsresponse) |

### /spine

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/spine/dimensions` | body [`SleightDimensionsRequest`](#sleightdimensionsrequest) | [`SleightDimensionsResponse`](#sleightdimensionsresponse) |

### /stats

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/stats` | query [`StatsQuery`](#statsquery) | [`MemoryStats`](#memorystats) |

### /storage

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/storage/cleanup` | body [`CleanupCorruptedRequest`](#cleanupcorruptedrequest) | [`CleanupCorruptedResponse`](#cleanupcorruptedresponse) |
| `POST` | `/api/storage/migrate` | body [`MigrateLegacyRequest`](#migratelegacyrequest) | [`MigrateLegacyResponse`](#migratelegacyresponse) |
| `GET` | `/api/storage/stats` | query [`StorageStatsRequest`](#storagestatsrequest) | memory::storage::[`StorageStats`](#storagestats) |
| `POST` | `/api/storage/uncompressed` | body [`GetUncompressedRequest`](#getuncompressedrequest) | [`RetrieveResponse`](#retrieveresponse) |

### /stream

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/stream` | none | `opaque` |

### /sync

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/sync/github` | body integrations::github::[`GitHubSyncRequest`](#githubsyncrequest) | integrations::github::[`GitHubSyncResponse`](#githubsyncresponse) |
| `POST` | `/api/sync/linear` | body integrations::linear::[`LinearSyncRequest`](#linearsyncrequest) | integrations::linear::[`LinearSyncResponse`](#linearsyncresponse) |

### /todos

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/todos` | body [`ListTodosRequest`](#listtodosrequest) | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/add` | body [`CreateTodoRequest`](#createtodorequest) | [`TodoResponse`](#todoresponse) |
| `POST` | `/api/todos/complete` | body [`FlatTodoRequest`](#flattodorequest) | [`TodoCompleteResponse`](#todocompleteresponse) |
| `POST` | `/api/todos/delete` | body [`FlatTodoRequest`](#flattodorequest) | [`TodoResponse`](#todoresponse) |
| `POST` | `/api/todos/due` | body [`DueTodosRequest`](#duetodosrequest) | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/list` | body [`ListTodosRequest`](#listtodosrequest) | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/ready` | body [`ListTodosRequest`](#listtodosrequest) | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/reorder` | body [`ReorderTodoRequest`](#reordertodorequest) | [`TodoResponse`](#todoresponse) |
| `POST` | `/api/todos/stats` | body [`TodoStatsRequest`](#todostatsrequest) | [`TodoStatsResponse`](#todostatsresponse) |
| `POST` | `/api/todos/update` | body [`UpdateTodoRequest`](#updatetodorequest) | [`TodoResponse`](#todoresponse) |
| `DELETE` | `/api/todos/{todo_id}` | query [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`TodoResponse`](#todoresponse) |
| `GET` | `/api/todos/{todo_id}` | query [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`TodoResponse`](#todoresponse) |
| `GET` | `/api/todos/{todo_id}/comments` | query [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`CommentListResponse`](#commentlistresponse) |
| `POST` | `/api/todos/{todo_id}/comments` | body [`AddCommentRequest`](#addcommentrequest); path `{todo_id: String}` | [`CommentResponse`](#commentresponse) |
| `DELETE` | `/api/todos/{todo_id}/comments/{comment_id}` | query [`TodoQuery`](#todoquery); path `{todo_id: String, comment_id: String}` | [`CommentResponse`](#commentresponse) |
| `PUT` | `/api/todos/{todo_id}/comments/{comment_id}` | body [`UpdateCommentRequest`](#updatecommentrequest); path `{todo_id: String, comment_id: String}` | [`CommentResponse`](#commentresponse) |
| `POST` | `/api/todos/{todo_id}/comments/{comment_id}/update` | body [`UpdateCommentRequest`](#updatecommentrequest); path `{todo_id: String, comment_id: String}` | [`CommentResponse`](#commentresponse) |
| `POST` | `/api/todos/{todo_id}/complete` | body [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`TodoCompleteResponse`](#todocompleteresponse) |
| `GET` | `/api/todos/{todo_id}/dependency_chain` | query [`DependencyChainQuery`](#dependencychainquery); path `{todo_id: String}` | [`TodoListResponse`](#todolistresponse) |
| `GET` | `/api/todos/{todo_id}/dependents` | query [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/{todo_id}/reorder` | body [`ReorderTodoRequest`](#reordertodorequest); path `{todo_id: String}` | [`TodoResponse`](#todoresponse) |
| `GET` | `/api/todos/{todo_id}/subtasks` | query [`TodoQuery`](#todoquery); path `{todo_id: String}` | [`TodoListResponse`](#todolistresponse) |
| `POST` | `/api/todos/{todo_id}/update` | body [`UpdateTodoRequest`](#updatetodorequest); path `{todo_id: String}` | [`TodoResponse`](#todoresponse) |

### /ui

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/ui/knows` | query [`KnowsQuery`](#knowsquery) | [`UiEnvelope`](#uienvelope)&lt;[`Knows`](#knows)&gt; |
| `GET` | `/api/ui/recall-log` | query [`RecallLogQuery`](#recalllogquery) | [`UiEnvelope`](#uienvelope)&lt;[`Field`](#field)&lt;[`RecallLogReport`](#recalllogreport)&gt;&gt; |
| `GET` | `/api/ui/tags` | query [`TagsQuery`](#tagsquery) | [`UiEnvelope`](#uienvelope)&lt;[`Field`](#field)&lt;[`TagsReport`](#tagsreport)&gt;&gt; |

### /upsert

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/upsert` | body [`UpsertRequest`](#upsertrequest) | [`UpsertResponse`](#upsertresponse) |

### /user_auth

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/user_auth/2fa/confirm` | body [`Confirm2faRequest`](#confirm2farequest) | `opaque` |
| `POST` | `/api/user_auth/2fa/enroll` | none | `opaque` |
| `GET` | `/api/user_auth/admin/audit` | query [`AuditQuery`](#auditquery) | `opaque` |
| `POST` | `/api/user_auth/admin/demote` | body [`UsernameRequest`](#usernamerequest) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/factors` | body [`EnrolBody`](#enrolbody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/init` | body [`InitBody`](#initbody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/passkey` | body [`PasskeyStartBody`](#passkeystartbody) | `opaque` |
| `GET` | `/api/user_auth/admin/keys/passkey/{ticket}` | none | `opaque` |
| `POST` | `/api/user_auth/admin/keys/recovery_codes` | body [`IssueBody`](#issuebody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/revoke` | body [`RevokeBody`](#revokebody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/rotate_data_keys` | body [`UnlockOnlyBody`](#unlockonlybody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/rotate_master_keys` | body [`InitBody`](#initbody) | `opaque` |
| `POST` | `/api/user_auth/admin/keys/rotate_passphrase` | body [`RotatePassphraseBody`](#rotatepassphrasebody) | `opaque` |
| `GET` | `/api/user_auth/admin/keys/status` | none | `opaque` |
| `POST` | `/api/user_auth/admin/promote` | body [`UsernameRequest`](#usernamerequest) | `opaque` |
| `GET` | `/api/user_auth/admin/sessions` | query [`SessionsQuery`](#sessionsquery) | `opaque` |
| `POST` | `/api/user_auth/admin/sessions/revoke` | body [`RevokeSessionBody`](#revokesessionbody) | `opaque` |
| `GET` | `/api/user_auth/admin/users` | none | `opaque` |
| `POST` | `/api/user_auth/admin/users/disable` | body [`UsernameBody`](#usernamebody) | `opaque` |
| `POST` | `/api/user_auth/admin/users/enable` | body [`UsernameBody`](#usernamebody) | `opaque` |
| `POST` | `/api/user_auth/admin/users/reset_2fa` | body [`UsernameBody`](#usernamebody) | `opaque` |
| `POST` | `/api/user_auth/admin/users/reset_password` | body [`UsernameBody`](#usernamebody) | `opaque` |
| `POST` | `/api/user_auth/admin/users/unlock` | body [`UsernameBody`](#usernamebody) | `opaque` |
| `GET` | `/api/user_auth/keys/passkey/{ticket}` | none | `opaque` |
| `POST` | `/api/user_auth/keys/passkey/{ticket}/result` | body [`PasskeyResultBody`](#passkeyresultbody) | `opaque` |
| `POST` | `/api/user_auth/login` | body [`LoginRequest`](#loginrequest) | `opaque` |
| `POST` | `/api/user_auth/logout` | none | `opaque` |
| `GET` | `/api/user_auth/me` | none | `opaque` |
| `POST` | `/api/user_auth/password` | body [`ChangePasswordBody`](#changepasswordbody) | `opaque` |
| `POST` | `/api/user_auth/recover` | body [`RecoverRequest`](#recoverrequest) | `opaque` |
| `POST` | `/api/user_auth/recovery_codes` | none | `opaque` |
| `POST` | `/api/user_auth/register` | body [`RegisterRequest`](#registerrequest) | `opaque` |
| `POST` | `/api/user_auth/sessions/revoke` | body [`OwnSessionBody`](#ownsessionbody) | `opaque` |
| `POST` | `/api/user_auth/sessions/revoke_all` | body [`UsernameRequest`](#usernamerequest) | `opaque` |
| `POST` | `/api/user_auth/sessions/revoke_mine` | none | `opaque` |
| `ANY` | `/api/user_auth/{*path}` | none | `opaque` |

### /users

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/api/users` | none | `Vec<String>` |
| `DELETE` | `/api/users/{user_id}` | path `{user_id: String}` | [`DeleteUserResponse`](#deleteuserresponse) |
| `GET` | `/api/users/{user_id}/stats` | path `{user_id: String}` | [`MemoryStats`](#memorystats) |

### /visualization

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/visualization/build` | body [`BuildVisualizationRequest`](#buildvisualizationrequest) | `VisualizationStats` |
| `GET` | `/api/visualization/{user_id}/dot` | path `{user_id: String}` | `String` |
| `GET` | `/api/visualization/{user_id}/stats` | path `{user_id: String}` | `VisualizationStats` |

### /webhook

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/webhook/github` | none | [`WebhookResponse`](#webhookresponse) |
| `POST` | `/webhook/linear` | none | [`WebhookResponse`](#webhookresponse) |

### /wiki

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/wiki/render` | query `HashMap<String, String>` | [`WikiRenderResponse`](#wikirenderresponse) |

### /wintermute

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/wintermute/dimensions` | body [`SleightDimensionsRequest`](#sleightdimensionsrequest) | [`SleightDimensionsResponse`](#sleightdimensionsresponse) |

## Types

### ABTestManagerSummary

Defined in `src/ab_testing.rs`. Summary of test manager state

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_active` | `usize` | yes |  |
| `draft` | `usize` | yes |  |
| `running` | `usize` | yes |  |
| `paused` | `usize` | yes |  |
| `completed` | `usize` | yes |  |
| `archived` | `usize` | yes |  |

### ABTestResponse

Defined in `src/handlers/ab_testing.rs`. Response for A/B test operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `test_id` | `Option<String>` | no |  |
| `message` | `String` | yes |  |

### AbImpressionResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `POST /api/ab/tests/{test_id}/impression`: `variant` on success (`null` when the user has no assignment), else `message`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `variant` | `Option<Option<String>>` | no | omitted when empty; `Control` or `Treatment`. |
| `message` | `Option<String>` | no | omitted when empty |

### AbRecordResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `POST /api/ab/tests/{test_id}/click` and `.../feedback`: `message` only on failure.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `Option<String>` | no | omitted when empty |

### AbSummaryResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `GET /api/ab/summary`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `summary` | ab_testing::[`ABTestManagerSummary`](#abtestmanagersummary) | yes |  |
| `expired_tests` | `Vec<String>` | yes | Tests this call archived for running past `max_duration_hours`. |

### AbTestAnalysis

Defined in `src/handlers/ab_testing.rs`. The statistics of one test, as `analyze` and `complete` report them.

| Field | Type | Required | Notes |
|---|---|---|---|
| `is_significant` | `bool` | yes |  |
| `confidence_level` | `f64` | yes |  |
| `chi_squared` | `f64` | yes |  |
| `p_value` | `f64` | yes |  |
| `winner` | `Option<String>` | no | `Control` or `Treatment`; absent while neither arm is ahead significantly. |
| `relative_improvement` | `f64` | yes |  |
| `control_ctr` | `f64` | yes |  |
| `treatment_ctr` | `f64` | yes |  |
| `confidence_interval` | `(f64, f64)` | yes |  |
| `recommendations` | `Vec<String>` | yes |  |

### AbTestAnalysisResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `GET /api/ab/tests/{test_id}/analyze`: `test_id` and `analysis` on success, else `message`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `test_id` | `Option<String>` | no | omitted when empty |
| `analysis` | Option&lt;[`AbTestAnalysis`](#abtestanalysis)&gt; | no | omitted when empty |
| `message` | `Option<String>` | no | omitted when empty |

### AbTestCompleteResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `POST /api/ab/tests/{test_id}/complete`: `test_id` and `results` on success, else `message`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `test_id` | `Option<String>` | no | omitted when empty |
| `results` | Option&lt;[`AbTestAnalysis`](#abtestanalysis)&gt; | no | omitted when empty |
| `message` | `Option<String>` | no | omitted when empty |

### AbTestDetail

Defined in `src/handlers/ab_testing.rs`. A test as `GET /api/ab/tests/{test_id}` reports it.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `description` | `String` | yes |  |
| `status` | `String` | yes | `Draft`, `Running`, `Paused`, `Completed` or `Archived`. |
| `traffic_split` | `f32` | yes |  |
| `min_impressions` | `u64` | yes |  |
| `max_duration_hours` | `Option<u64>` | no |  |
| `control_weights` | relevance::[`LearnedSignalWeights`](#learnedsignalweights) | yes |  |
| `treatment_weights` | relevance::[`LearnedSignalWeights`](#learnedsignalweights) | yes |  |
| `control_metrics` | [`AbVariantMetrics`](#abvariantmetrics) | yes |  |
| `treatment_metrics` | [`AbVariantMetrics`](#abvariantmetrics) | yes |  |
| `created_at` | `String` | yes | RFC 3339. |
| `started_at` | `Option<String>` | no | RFC 3339. |
| `completed_at` | `Option<String>` | no | RFC 3339. |

### AbTestDetailResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `GET /api/ab/tests/{test_id}`: `test` when it exists, else `message`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `test` | Option&lt;[`AbTestDetail`](#abtestdetail)&gt; | no | omitted when empty |
| `message` | `Option<String>` | no | omitted when empty |

### AbTestListItem

Defined in `src/handlers/ab_testing.rs`. One test of a `GET /api/ab/tests` reply.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `description` | `String` | yes |  |
| `status` | `String` | yes | `Draft`, `Running`, `Paused`, `Completed` or `Archived`. |
| `traffic_split` | `f32` | yes |  |
| `control_impressions` | `u64` | yes |  |
| `treatment_impressions` | `u64` | yes |  |
| `created_at` | `String` | yes | RFC 3339. |

### AbTestListResponse

Defined in `src/handlers/ab_testing.rs`. Reply of `GET /api/ab/tests`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `tests` | Vec&lt;[`AbTestListItem`](#abtestlistitem)&gt; | yes |  |
| `summary` | ab_testing::[`ABTestManagerSummary`](#abtestmanagersummary) | yes |  |

### AbVariantMetrics

Defined in `src/handlers/ab_testing.rs`. One arm's counters, with its click-through rate.

| Field | Type | Required | Notes |
|---|---|---|---|
| `impressions` | `u64` | yes |  |
| `clicks` | `u64` | yes |  |
| `ctr` | `f64` | yes | `clicks / impressions`, 0 when there were no impressions. |
| `positive_feedback` | `u64` | yes |  |
| `negative_feedback` | `u64` | yes |  |

### AddAliasRequest

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity_name` | `String` | yes |  |
| `alias` | `String` | yes |  |

### AddAliasResponse

Defined in `src/handlers/prompt_gen.rs`. Reply of `POST /api/entity/alias`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | `String` | yes | The canonical entity name the alias now resolves to. |
| `alias` | `String` | yes |  |
| `status` | `String` | yes | Always `added`. |

### AddCommentRequest

Defined in `src/handlers/todos.rs`. Request to add a comment to a todo

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `author` | `Option<String>` | no | default when absent |
| `comment_type` | `Option<String>` | no | default when absent |

### AddEntityRequest

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `label` | `String` | yes |  |
| `attributes` | `Option<HashMap<String, String>>` | no |  |

### AddEntityResponse

Defined in `src/handlers/mif.rs`. Reply of `POST /api/graph/entity/add`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `entity` | `String` | yes | The entity name, as sent. |
| `label` | `String` | yes | The label applied, as its variant name. |

### AddRelationshipRequest

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `from_entity` | `String` | yes |  |
| `to_entity` | `String` | yes |  |
| `relation_type` | `String` | yes |  |
| `strength` | `Option<f32>` | no |  |

### AddRelationshipResponse

Defined in `src/handlers/mif.rs`. Reply of `POST /api/graph/relationship/add`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `from` | `String` | yes |  |
| `to` | `String` | yes |  |
| `type` | `String` | yes | The relation type applied, as its variant name. |
| `strength` | `f32` | yes |  |

### AdvancedSearchRequest

Defined in `src/handlers/search.rs`. Request for advanced search

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity_name` | `Option<String>` | no |  |
| `start_date` | `Option<String>` | no |  |
| `end_date` | `Option<String>` | no |  |
| `min_importance` | `Option<f32>` | no | score |
| `max_importance` | `Option<f32>` | no | score |

### AgentRef

Defined in `src/memory/facets.rs`. A reference to an agent (person, organization, self) in an engram.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_id` | `Option<Uuid>` | no | Link to the graph Person/Organization entity, when known. |
| `name` | `String` | yes | Human-readable name. |
| `role` | [`AgentRole`](#agentrole) | yes | Role this agent plays in the engram. |

### AgentRole

Defined in `src/memory/facets.rs`. How an agent participates in an engram. Self / Author / Subject / Audience / Mentioned are dissociable in episodic memory (distinct social-brain mPFC/TPJ functions) and must not be collapsed.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `self_agent` | none | The self — this agent/assistant. mPFC self-referential marker. |
| `author` | none | Produced or asserted the content. |
| `subject` | none | What the content is about — the neutral default. |
| `audience` | none | To whom the content was addressed. |
| `mentioned` | none | A third party merely mentioned. |

### AgentSession

Defined in `src/memory/facets.rs`. Records which agent was operating from which git worktree/branch at the moment a memory was written.

| Field | Type | Required | Notes |
|---|---|---|---|
| `worktree_path` | `Option<PathBuf>` | no | default when absent; Absolute path of the active worktree this agent was running in. |
| `branch` | `Option<String>` | no | default when absent; Git branch checked out in that worktree. |
| `agent_id` | `Option<String>` | no | default when absent; Chat-brand identity of the agent — e.g. `"Claude"`, `"Copilot"`, `"Cursor"`. Deliberately the *agentic-chat* brand rather than the launcher binary (`claude-code` / `claude-cli` / Claude Desktop all collapse to `"Claude"`) so the same conversation identity reads the same across surfaces. Populated by auto-detecting the parent process at session start: when the parent is a known chat launcher (`code` with a Claude/Copilot extension active, the Claude binaries, `cursor`, etc.) the brand follows. The binary / launcher slug is the fallback when detection is ambiguous. |
| `vscode_window_id` | `Option<String>` | no | default when absent; VS Code window id when the agent is running inside a VS Code window. Lets the planned viewer disambiguate concurrent windows on the same worktree. |
| `started_at` | `Option<DateTime<Utc>>` | no | default when absent; When this agent session began — distinct from the engram's `encoded_at`. Multiple memories recorded in one session share this timestamp, which is what makes per-session filtering possible. |
| `parent_repo` | `Option<PathBuf>` | no | default when absent; Path of the main worktree (the original clone) when this session is running in a sibling/linked worktree. Anchors a child worktree back to its parent so a branch-aware tool can group siblings. |
| `declared` | [`DeclaredWriter`](#declaredwriter) | no | default when absent; What the CALLER said about itself — see [`DeclaredWriter`]. |
| `verified` | [`VerifiedWriter`](#verifiedwriter) | no | default when absent; What the SERVER established — see [`VerifiedWriter`]. |

### AnchorRequest

Defined in `src/handlers/crud.rs`. Request to anchor or unanchor a memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memory_id` | `String` | yes |  |
| `anchor` | `bool` | no | default when absent; true to anchor, false to unanchor |

### AnchorResponse

Defined in `src/handlers/crud.rs`. Response for anchor operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `memory_id` | `String` | yes |  |
| `anchored` | `bool` | yes |  |

### AssociationCard

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `from` | `String` | yes |  |
| `to` | `String` | yes |  |
| `relation` | `String` | yes |  |
| `strength` | `f32` | yes |  |
| `activation_count` | `u32` | yes |  |

### AssociationChange

Defined in `src/memory/introspection.rs`. Change in an association/edge

| Field | Type | Required | Notes |
|---|---|---|---|
| `from_memory_id` | `String` | yes |  |
| `to_memory_id` | `String` | yes |  |
| `strength_before` | `Option<f32>` | no |  |
| `strength_after` | `f32` | yes |  |
| `co_activations` | `Option<u32>` | no |  |
| `reason` | `String` | yes |  |
| `timestamp` | `DateTime<Utc>` | yes |  |

### AuditQuery

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `limit` | `Option<usize>` | no |  |

### BackendCensus

Defined in `src/storage/boot_guard.rs`. Tenants holding at least one memory, per backend, measured at boot.

| Field | Type | Required | Notes |
|---|---|---|---|
| `rocksdb` | `usize` | yes |  |
| `redb` | `Option<usize>` | no | `None` when this build cannot read redb, which is unknown rather than zero. |
| `measured_at` | `chrono::DateTime<chrono::Utc>` | yes |  |

### BacklogReport

Defined in `src/memory/embedding_backlog.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `witness` | `&'static str` | yes |  |
| `pending` | `usize` | yes |  |
| `oldest_enqueued_at` | `Option<DateTime<Utc>>` | no |  |
| `max_attempts` | `u32` | yes |  |
| `by_reason` | `BTreeMap<String, usize>` | yes |  |
| `last_error` | `Option<String>` | no |  |

### BackupMetadata

Defined in `src/backup.rs`. Backup metadata for tracking and verification

| Field | Type | Required | Notes |
|---|---|---|---|
| `backup_id` | `u32` | yes | Unique backup ID |
| `created_at` | `DateTime<Utc>` | yes | Timestamp when backup was created |
| `user_id` | `String` | yes | User ID (if single-user backup) or "all" for full backup |
| `backup_type` | [`BackupType`](#backuptype) | yes | Backup type: "full" or "incremental" |
| `size_bytes` | `u64` | yes | Size in bytes (compressed) |
| `checksum` | `String` | yes | SHA-256 checksum for integrity verification |
| `memory_count` | `usize` | yes | Number of memories included in backup |
| `sequence_number` | `u64` | yes | RocksDB sequence number (for PITR) |
| `secondary_stores` | `Vec<String>` | no | default when absent; Secondary stores included in this backup |
| `secondary_size_bytes` | `u64` | no | default when absent; Total size of secondary store backups in bytes |
| `intent_log_files` | Vec&lt;[`IntentLogFileMeta`](#intentlogfilemeta)&gt; | no | default when absent; Intent-log files captured under `intent_log/` in the archive (W5). Includes the journal log + the checkpoint store + any other files found in the configured `intent_log/` source directory. |
| `journal_cut_lsn` | `Option<u64>` | no | default when absent; Highest LSN in the main journal at the moment the intent log was captured — the backup's consistency cut. The journal is captured *before* the primary DB is checkpointed, and every primary write precedes its journal frame, so the primary snapshot holds every memory written up to this LSN unless it was deleted in between; anything the primary holds beyond it is re-journalled by the C2 reconcile on first access after restore, and the restore marks the tenant so that reconcile forgets, rather than restores, a memory deleted between the capture and the checkpoint. `None` for backups taken before this field existed, or when no journal file was present. |
| `column_families` | `Vec<String>` | no | default when absent; Column families that the main memories DB actually contained at backup time. `cf_user_auth` appears here when Phase C has been initialised; otherwise it is omitted. Older backups written before this field existed deserialise with an empty vector. |

### BackupResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `backup` | Option&lt;crate::backup::[`BackupMetadata`](#backupmetadata)&gt; | no |  |
| `message` | `String` | yes |  |

### BackupType

Defined in `src/backup.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Full` | none |  |
| `Incremental` | none |  |

### BatchErrorItem

Defined in `src/handlers/remember.rs`. Error detail for batch item

| Field | Type | Required | Notes |
|---|---|---|---|
| `index` | `usize` | yes |  |
| `error` | `String` | yes |  |

### BatchMemoryItem

Defined in `src/handlers/remember.rs`. Single item in batch remember

| Field | Type | Required | Notes |
|---|---|---|---|
| `content` | `String` | yes |  |
| `tags` | `Vec<String>` | no | default when absent |
| `memory_type` | `Option<String>` | no | default when absent |
| `created_at` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent |
| `emotional_valence` | `Option<f32>` | no | default when absent |
| `emotional_arousal` | `Option<f32>` | no | default when absent |
| `emotion` | `Option<String>` | no | default when absent |
| `source_type` | `Option<String>` | no | default when absent |
| `credibility` | `Option<f32>` | no | default when absent |
| `episode_id` | `Option<String>` | no | default when absent |
| `sequence_number` | `Option<u32>` | no | default when absent |
| `preceding_memory_id` | `Option<String>` | no | default when absent |
| `parent_id` | `Option<String>` | no | default when absent; Parent memory ID for hierarchical organization |
| `writer` | Option&lt;crate::memory::facets::[`DeclaredWriter`](#declaredwriter)&gt; | no | default when absent; Per-item writer identity — see [`RememberRequest::writer`]. Declared per ITEM rather than per batch: a batch is a transport convenience, and nothing guarantees every item in one came from the same session or the same model. |
| `entities` | Vec&lt;[`DeclaredEntityInput`](#declaredentityinput)&gt; | no | default when absent; Per-item declared entities — see [`RememberRequest::entities`]. An invalid declaration fails its item, as an unknown memory type does. |
| `repo` | `Option<String>` | no | default when absent; Per-item repository — see [`RememberRequest::repo`]. An invalid one fails its item. |

### BatchRememberOptions

Defined in `src/handlers/remember.rs`. Options for batch remember

| Field | Type | Required | Notes |
|---|---|---|---|
| `extract_entities` | `bool` | no | default when absent |
| `create_edges` | `bool` | no | default when absent |

### BatchRememberRequest

Defined in `src/handlers/remember.rs`. Batch remember request

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memories` | Vec&lt;[`BatchMemoryItem`](#batchmemoryitem)&gt; | yes |  |
| `options` | [`BatchRememberOptions`](#batchrememberoptions) | no | default when absent |

### BatchRememberResponse

Defined in `src/handlers/remember.rs`. Batch remember response

| Field | Type | Required | Notes |
|---|---|---|---|
| `created` | `usize` | yes |  |
| `failed` | `usize` | yes |  |
| `memory_ids` | `Vec<String>` | yes |  |
| `errors` | Vec&lt;[`BatchErrorItem`](#batcherroritem)&gt; | yes |  |

### BettiCurveSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `epsilon` | `f32` | yes |  |
| `beta_0` | `usize` | yes |  |
| `beta_1` | `usize` | yes |  |
| `beta_2` | `usize` | yes |  |
| `simplex_count` | `usize` | yes |  |

### Binding

Defined in `src/handlers/ui.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `tenant-key` | none |  |
| `ui-user-env` | none |  |

### BlockDeleteResponse

Defined in `src/handlers/context_blocks.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `deleted` | `bool` | yes |  |

### BlockListResponse

Defined in `src/handlers/context_blocks.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `blocks` | Vec&lt;[`BlockResponse`](#blockresponse)&gt; | yes |  |
| `total` | `usize` | yes |  |

### BlockQuery

Defined in `src/handlers/context_blocks.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### BlockResponse

Defined in `src/handlers/context_blocks.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `key` | `String` | yes |  |
| `content` | `String` | yes |  |
| `max_tokens` | `usize` | yes |  |
| `updated_at` | `String` | yes |  |
| `version` | `u32` | yes |  |

### BoundaryCause

Defined in `src/memory/facets.rs`. Why a new event boundary fired at this engram.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `goal_change` | none | A new goal opened. |
| `prediction_error` | none | An expectation was violated — Event Segmentation Theory's trigger. |
| `location_change` | none | WHERE shifted significantly. |
| `agent_change` | none | WHO shifted. |
| `temporal_gap` | none | A long idle gap. |

### BrainStateResponse

Defined in `src/handlers/visualization.rs`. Brain state response with memories organized by tier

| Field | Type | Required | Notes |
|---|---|---|---|
| `working_memory` | Vec&lt;[`MemoryNeuron`](#memoryneuron)&gt; | yes |  |
| `session_memory` | Vec&lt;[`MemoryNeuron`](#memoryneuron)&gt; | yes |  |
| `longterm_memory` | Vec&lt;[`MemoryNeuron`](#memoryneuron)&gt; | yes |  |
| `stats` | [`BrainStats`](#brainstats) | yes |  |

### BrainStats

Defined in `src/handlers/visualization.rs`. Brain statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_memories` | `usize` | yes |  |
| `working_count` | `usize` | yes |  |
| `session_count` | `usize` | yes |  |
| `longterm_count` | `usize` | yes |  |
| `avg_activation` | `f32` | yes |  |
| `avg_importance` | `f32` | yes | score |

### Bucket

Defined in `src/entity_frequency.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `day` | none |  |
| `week` | none |  |

### BucketCounts

Defined in `src/entity_frequency.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `start` | `DateTime<Utc>` | yes |  |
| `counts` | `BTreeMap<String, usize>` | yes |  |

### BuildVisualizationRequest

Defined in `src/handlers/visualization.rs`. Request to build visualization

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### BulkDeleteRequest

Defined in `src/handlers/crud.rs`. Bulk delete memories by filters

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `tags` | `Option<Vec<String>>` | no | Delete memories matching ANY of these tags |
| `memory_type` | `Option<String>` | no | Delete memories of this type |
| `created_after` | `Option<chrono::DateTime<chrono::Utc>>` | no | Delete memories created after this timestamp |
| `created_before` | `Option<chrono::DateTime<chrono::Utc>>` | no | Delete memories created before this timestamp |

### BulkDeleteResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/memories/bulk`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted_count` | `usize` | yes |  |

### CausalFactLink

Defined in `src/memory/compression.rs`. A heuristic-detected causal relationship between two facts.

| Field | Type | Required | Notes |
|---|---|---|---|
| `from_fact_id` | `String` | yes |  |
| `to_fact_id` | `String` | yes |  |
| `from_fact` | `String` | yes | The source fact statement (snapshot at detection time) |
| `to_fact` | `String` | yes | The target fact statement (snapshot at detection time) |
| `relation` | `String` | yes | Relationship type |

### CausalKind

Defined in `src/memory/facets.rs`. Kinds of causal relation between memories.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `caused` | none | Strictly caused. |
| `triggered` | none | Triggered without strictly causing. |
| `enabled` | none | Made possible. |
| `prevented` | none | Prevented. |
| `motivated` | none | Provided motivation. |

### CausalLink

Defined in `src/memory/facets.rs`. A typed causal link between this engram and another memory.

| Field | Type | Required | Notes |
|---|---|---|---|
| `other` | [`MemoryId`](#memoryid) | yes | The other memory. |
| `relation` | [`CausalKind`](#causalkind) | yes | The kind of causal relation. |
| `confidence` | `f32` | yes | score; Confidence in this link, 0.0–1.0. |
| `inferred` | `bool` | yes | Was this link observed or inferred? Inferred links are weaker evidence at retrieval — the source-monitoring discipline applied to Veld's own inferences. |

### CausalRelation

Defined in `src/memory/lineage.rs`. Causal relationship types between memories

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Caused` | none | Error/Bug caused a Todo to be created |
| `ResolvedBy` | none | Todo was resolved by a Learning/Fix |
| `InformedBy` | none | Decision was informed by a Learning/Discovery |
| `SupersededBy` | none | Old decision/pattern was superseded by new one |
| `TriggeredBy` | none | Discovery/Learning triggered a Todo |
| `BranchedFrom` | none | Memory branched from another (project pivot) |
| `RelatedTo` | none | Generic relation when type is unclear |

### ChangePasswordBody

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `current_password` | `String` | yes |  |
| `new_password` | `String` | yes |  |

### ChangeType

Defined in `src/memory/types.rs`. Type of change made to a memory

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `created` | none | Memory was created |
| `content_updated` | none | Content was updated |
| `status_changed` | none | Status/state changed (e.g., Linear issue status) |
| `tags_updated` | none | Tags/entities were modified |
| `importance_adjusted` | none | Importance was adjusted |

### CheckContextRemindersRequest

Defined in `src/handlers/todos.rs`. Request to check context-triggered reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `context` | `String` | yes |  |
| `mark_triggered` | `bool` | no | default when absent |

### Classification

Defined in `src/entity_hygiene.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `shelf` | [`Shelf`](#shelf) | yes |  |
| `reason` | [`Reason`](#reason) | yes |  |
| `proper_noun_hint` | `bool` | yes | veld's NER flag, carried for display. It decided nothing above. |

### CleanupCorruptedRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### CleanupCorruptedResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted_count` | `usize` | yes |  |

### ClearAllRequest

Defined in `src/handlers/crud.rs`. Clear ALL memories for a user (GDPR compliance)

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `confirm` | `String` | yes | Safety confirmation - must be "CONFIRM" to proceed |

### ClearAllResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/memories/clear`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted_count` | `usize` | yes |  |
| `message` | `String` | yes |  |

### ClearGraphResponse

Defined in `src/handlers/graph.rs`. Reply of `DELETE /api/graph/{user_id}/clear`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `cleared` | [`ClearedGraphCounts`](#clearedgraphcounts) | yes |  |

### ClearedGraphCounts

Defined in `src/handlers/graph.rs`. What `DELETE /api/graph/{user_id}/clear` removed.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entities` | `usize` | yes |  |
| `relationships` | `usize` | yes |  |
| `episodes` | `usize` | yes |  |

### Cluster

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `usize` | yes |  |
| `tags` | `Vec<String>` | yes |  |
| `memories` | `usize` | yes |  |
| `producer` | [`Producer`](#producer) | yes |  |

### CoTag

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `tag` | `String` | yes |  |
| `memories` | `usize` | yes |  |

### CodeContext

Defined in `src/memory/types.rs`. Code-specific context

| Field | Type | Required | Notes |
|---|---|---|---|
| `current_file` | `Option<String>` | no | Current file being edited |
| `current_scope` | `Option<String>` | no | Current function/class |
| `related_files` | `Vec<String>` | yes | Related files (imports, dependencies) |
| `recent_edits` | `Vec<String>` | yes | Recently edited functions |
| `call_stack` | `Vec<String>` | yes | Call stack context |
| `git_branch` | `Option<String>` | no | Git branch |
| `recent_commits` | `Vec<String>` | yes | Recent commits |
| `patterns` | `Vec<String>` | yes | Code patterns detected |

### CognitiveDials

Defined in `src/memory/dials.rs`. Runtime-tunable cognitive parameters for a single user's [`crate::memory::MemorySystem`].

| Field | Type | Required | Notes |
|---|---|---|---|
| `reconsolidation_enabled` | `bool` | no | default when absent; Master switch for the FIX-R1 reconsolidation expiry + re-encode block in `run_maintenance`. When `false`, shadows are neither expired nor re-encoded (pre-feature behavior). |
| `reconsolidation_window_secs` | `i64` | no | default when absent; Labile-window length in seconds (used when creating/extending shadows on recall). |
| `re_encode_graph_strengthen` | `bool` | no | default when absent; Layer 1: strengthen Hebbian graph edges between the memory and the entities found in its accumulated `retrieval_context` at window close. |
| `graph_boost_scale` | `f32` | no | default when absent; Scale applied to the Layer-1 edge boost (× a function of retrieval count). |
| `re_encode_entity_enrich` | `bool` | no | default when absent; Layer 2: merge genuinely-new `retrieval_context` entities into the memory's `entity_refs` and persist (re-index) via `update_memory`. |
| `re_encode_embedding_nudge` | `bool` | no | default when absent; Layer 3: blend the memory's embedding toward the `retrieval_context` embedding and persist (re-index). Heaviest layer; bounded by the active-shadow cap. |
| `embedding_blend` | `f32` | no | default when absent; Blend factor for Layer 3 (`new = normalize((1-blend)·mem + blend·ctx)`). |
| `max_active_shadows` | `usize` | no | default when absent; Max simultaneous labile shadows before the oldest (expired-first) are evicted on the recall path. Caps memory + per-cycle re-encode work. |
| `working_memory_threshold` | `u32` | no | default when absent; Consecutive-retrieval count at/above which a shadow is treated as *working memory* and kept labile (NOT drained at window close). Lower ⇒ drains sooner. |
| `refiner_mode` | Option&lt;[`RefinerMode`](#refinermode)&gt; | no | default when absent; Override the post-fusion refiner stage. `None` keeps the engine default. |
| `use_reranking` | `Option<bool>` | no | default when absent; Override whether the cross-encoder rerank runs. `None` keeps the engine default. |
| `use_mmr` | `Option<bool>` | no | default when absent; Override whether Layer 5.4 MMR diversification runs. `None` keeps the engine default ([`crate::constants::MMR_ENABLED_DEFAULT`], off). |
| `candidate_count` | `Option<usize>` | no | default when absent; Override the initial candidate-pool size. `None` keeps the engine default. |
| `rerank_count` | `Option<usize>` | no | default when absent; Override the rerank top-k. `None` keeps the engine default. |
| `rerank_pool` | `Option<usize>` | no | default when absent; Fused candidates kept for the cross-encoder layer (recall Layer 4's truncation). `None` keeps 30, or 50 for a decomposed query; the pool never drops below `rerank_count`. Raising `rerank_count` alone cannot reach gold that the truncation already cut. |
| `rerank_model` | `Option<String>` | no | default when absent; Which cross-encoder reranks, by registry id (`crate::embeddings::cross_encoder::lookup`). `None` keeps the process-wide choice (`VELD_CROSS_ENCODER`, else mxbai-rerank-xsmall-v1). Set to `ms-marco-MiniLM-L-6-v2` for the recall gate's parity arm against upstream. [`CognitiveDials::sanitized`] drops an id the registry does not know, so what `GET /api/dials` returns is what reranks. |
| `rrf_weights` | `Option<(f32, f32, f32)>` | no | default when absent; Override the RRF fusion weights `(bm25, vector, graph)`. `None` keeps the engine/learned weights. |
| `date_window_leg` | `Option<bool>` | no | default when absent; Whether recall adds the date window candidate leg: when a query names a calendar day or month-year, memories whose `created_at` falls in that window enter RRF fusion as their own candidate list (recall.rs Layer 4). `None` and `Some(true)` run it; `Some(false)` leaves the window to the temporal boost only. |
| `rrf_k_outer` | `Option<f32>` | no | default when absent; RRF k for the outer graph+hybrid fusion (recall.rs Layer 4). Default 20. |
| `rrf_k_decomp` | `Option<f32>` | no | default when absent; RRF k for the decomposed sub-query merge. Default 20. |
| `rrf_k_inner` | `Option<f32>` | no | default when absent; RRF k for the inner BM25+vector fusion (hybrid_search). Default 45. |
| `secondary_rrf_ratio` | `Option<f32>` | no | default when absent; Weight ratio of the secondary (768d) vector index signal. Default 0.5. |
| `bm25_min_score` | `Option<f32>` | no | default when absent; score; BM25 minimum-score floor for lexical candidates. Default 0.1. |
| `onto_rerank_boost` | `Option<f32>` | no | default when absent; Ontological rerank boost per entity-label match. Default 0.15. |
| `onto_rerank_max` | `Option<f32>` | no | default when absent; Ontological rerank boost cap. Default 0.40. |
| `onto_rerank_penalty` | `Option<f32>` | no | default when absent; Ontological rerank mismatch penalty (negative). Default -0.08. |
| `ce_blend` | `Option<f32>` | no | default when absent; Cross-encoder share of the Layer 5.3 blend for every query shape at once, except that Abstract (decomposed) never drops below its own share. `None` lets the per-shape dials and `constants::CE_BLEND_*` decide. See `ce_share`. |
| `temporal_recall_boost` | `Option<f32>` | no | default when absent; Temporal-fact source-memory boost. Default 0.20. |
| `ce_blend_identifier` | `Option<f32>` | no | default when absent; Cross-encoder share for one query shape; each overrides its `constants::CE_BLEND_*` default and is itself overridden by `ce_blend`. See `ce_share`. |
| `ce_blend_ordinal` | `Option<f32>` | no | default when absent |
| `ce_blend_temporal` | `Option<f32>` | no | default when absent |
| `ce_blend_people` | `Option<f32>` | no | default when absent |
| `ce_blend_keyword` | `Option<f32>` | no | default when absent |
| `ce_blend_exploratory` | `Option<f32>` | no | default when absent |
| `ce_blend_attribute` | `Option<f32>` | no | default when absent |
| `ce_blend_relational` | `Option<f32>` | no | default when absent |
| `ce_blend_abstract` | `Option<f32>` | no | default when absent |
| `query_coverage_weight` | `Option<f32>` | no | default when absent; score; Layer 5.36 query-term coverage weight; `0` disables. Default `constants::QUERY_COVERAGE_WEIGHT`. |
| `people_boost_max` | `Option<f32>` | no | default when absent; Layer 5.37 person-density boost ceiling for people questions; `0` disables. Default `constants::PEOPLE_BOOST_MAX`. |
| `use_prf` | `Option<bool>` | no | default when absent; Pseudo-relevance feedback leg. Default `constants::PRF_ENABLED_DEFAULT`. |
| `prf_docs` | `Option<usize>` | no | default when absent; Top first-pass documents PRF reads terms from. Default `constants::PRF_DOCS`. |
| `prf_terms` | `Option<usize>` | no | default when absent; Expansion terms PRF appends. Default `constants::PRF_TERMS`. |
| `prf_weight` | `Option<f32>` | no | default when absent; score; PRF leg weight as a fraction of the hybrid weight. Default `constants::PRF_WEIGHT`. |
| `mmr_lambda_exploratory` | `Option<f32>` | no | default when absent; MMR diversity λ for exploratory queries. Default 0.6. |
| `mmr_lambda_abstract` | `Option<f32>` | no | default when absent; MMR diversity λ for abstract queries. Default 0.5. |
| `mmr_lambda_relationship` | `Option<f32>` | no | default when absent; MMR diversity λ for relationship queries. Default 0.7. |
| `density_linguistic_w` | `Option<f32>` | no | default when absent; Linguistic weight in the density-adaptive fusion mix. Default 0.15. |
| `updated_at` | `Option<DateTime<Utc>>` | no | default when absent; Last time these dials were set (set by the writer; informational). |
| `disabled_signals` | `HashSet<String>` | no | default when absent; Set of signal IDs that are disabled at runtime. An empty set (the default) means every signal runs exactly as compiled — **no behaviour change**.  Populated from `VELD_DIALS_DISABLE_SIGNALS` (comma-separated IDs) at startup, or via the `PUT /api/dials` HTTP endpoint at runtime. |

### ColumnDef

Defined in `src/datasets/schema.rs`. Definition of a single column within a [`DatasetSchema`].

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `ty` | [`ColumnType`](#columntype) | yes |  |
| `nullable` | `bool` | yes |  |

### ColumnType

Defined in `src/datasets/schema.rs`. Storage-engine-independent column type. The DDL rendering layer maps each variant to the closest native type in SQLite or Postgres.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Bool` | none |  |
| `I64` | none |  |
| `F64` | none |  |
| `Text` | none |  |
| `Bytes` | none |  |
| `Timestamp` | none |  |
| `Json` | none |  |

### CommentListResponse

Defined in `src/handlers/todos.rs`. Response for listing comments

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `count` | `usize` | yes |  |
| `comments` | Vec&lt;[`TodoComment`](#todocomment)&gt; | yes |  |
| `formatted` | `String` | yes |  |

### CommentResponse

Defined in `src/handlers/todos.rs`. Response for comment operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `comment` | Option&lt;[`TodoComment`](#todocomment)&gt; | no |  |
| `formatted` | `String` | yes |  |

### CompetitionMode

Defined in `src/memory/types.rs`. Retrieval-time resolution policy for entity-overlapping candidates.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `coexist` | none | Keep all candidates and only annotate conflicts where needed. |
| `resolve_newest` | none | Keep the most recent competing memory. |
| `resolve_strongest` | none | Keep the strongest competing memory. |
| `surface_both` | none | Keep both competitors but mark them for downstream inspection. |

### CompressMemoryRequest

Defined in `src/handlers/compression.rs`. Request for compressing a memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `id` | `String` | yes |  |

### CompressMemoryResponse

Defined in `src/handlers/compression.rs`. Reply of `POST /api/memory/compress`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `String` | yes |  |

### ConceptRelationship

Defined in `src/memory/types.rs`. Relationship between concepts

| Field | Type | Required | Notes |
|---|---|---|---|
| `from` | `String` | yes |  |
| `to` | `String` | yes |  |
| `relationship_type` | [`RelationshipType`](#relationshiptype) | yes |  |
| `strength` | `f32` | yes |  |

### Confirm2faRequest

Defined in `src/handlers/user_auth.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `totp` | `String` | yes |  |

### ConsolidateRequest

Defined in `src/handlers/types.rs`. Request to trigger consolidation

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `min_support` | `usize` | no | default when absent; Minimum supporting memories required for fact extraction |
| `min_age_days` | `i64` | no | default when absent; Minimum age in days for memories to be considered for consolidation |

### ConsolidateResponse

Defined in `src/handlers/types.rs`. Response from consolidation

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_analyzed` | `usize` | yes |  |
| `facts_extracted` | `usize` | yes |  |
| `facts_reinforced` | `usize` | yes |  |
| `fact_ids` | `Vec<String>` | yes |  |
| `memories_replayed` | `usize` | yes |  |
| `edges_strengthened` | `usize` | yes |  |
| `entity_edges_strengthened` | `usize` | yes |  |
| `memories_decayed` | `usize` | yes |  |
| `warnings` | `Vec<String>` | yes | omitted when empty |

### ConsolidationEvent

Defined in `src/memory/introspection.rs`. Types of consolidation events that can occur

Enum, internally tagged: the payload's fields plus `"type": <variant>`.

| Variant | Payload | Notes |
|---|---|---|
| `memory_strengthened` | `{ memory_id: String, content_preview: String, activation_before: f32, activation_after: f32, reason: StrengtheningReason, timestamp: DateTime<Utc> }` | Memory activation was strengthened (accessed/boosted) |
| `memory_decayed` | `{ memory_id: String, content_preview: String, activation_before: f32, activation_after: f32, at_risk: bool, timestamp: DateTime<Utc> }` | Memory activation decayed during maintenance |
| `edge_formed` | `{ from_memory_id: String, to_memory_id: String, initial_strength: f32, reason: EdgeFormationReason, timestamp: DateTime<Utc> }` | Hebbian edge was formed between memories |
| `edge_strengthened` | `{ from_memory_id: String, to_memory_id: String, strength_before: f32, strength_after: f32, co_activations: u32, timestamp: DateTime<Utc> }` | Hebbian edge was strengthened (co-activation) |
| `edge_potentiated` | `{ from_memory_id: String, to_memory_id: String, final_strength: f32, total_co_activations: u32, timestamp: DateTime<Utc> }` | Edge became potentiated (permanent through LTP) |
| `edge_pruned` | `{ from_memory_id: String, to_memory_id: String, final_strength: f32, reason: PruningReason, timestamp: DateTime<Utc> }` | Edge was pruned (decayed below threshold) |
| `fact_extracted` | `{ fact_id: String, fact_content: String, confidence: f32, source_memory_count: usize, fact_type: String, timestamp: DateTime<Utc> }` | Semantic fact was extracted from episodic memories |
| `fact_reinforced` | `{ fact_id: String, fact_content: String, confidence_before: f32, confidence_after: f32, new_support_count: usize, timestamp: DateTime<Utc> }` | Existing fact was reinforced with new evidence |
| `fact_decayed` | `{ fact_id: String, fact_content: String, confidence_before: f32, confidence_after: f32, days_since_reinforcement: i64, timestamp: DateTime<Utc> }` | Fact confidence decayed due to lack of reinforcement |
| `fact_deleted` | `{ fact_id: String, fact_content: String, final_confidence: f32, support_count: usize, reason: String, timestamp: DateTime<Utc> }` | Fact was deleted (confidence fell below threshold) |
| `memory_promoted` | `{ memory_id: String, content_preview: String, from_tier: String, to_tier: String, timestamp: DateTime<Utc> }` | Memory was promoted to a higher tier |
| `maintenance_cycle_completed` | `{ memories_processed: usize, memories_decayed: usize, edges_pruned: usize, duration_ms: u64, timestamp: DateTime<Utc> }` | Maintenance cycle completed |
| `memory_replayed` | `{ memory_id: String, content_preview: String, activation_before: f32, activation_after: f32, replay_priority: f32, connected_memories_replayed: usize, timestamp: DateTime<Utc> }` | Memory was replayed during consolidation cycle |
| `replay_cycle_completed` | `{ memories_replayed: usize, edges_strengthened: usize, total_priority_score: f32, duration_ms: u64, timestamp: DateTime<Utc> }` | Replay cycle completed (batch of memories replayed) |
| `interference_detected` | { new_memory_id: String, old_memory_id: String, similarity: f32, interference_type: [`InterferenceType`](#interferencetype), timestamp: DateTime&lt;Utc&gt; } | Interference detected between memories |
| `memory_weakened` | { memory_id: String, content_preview: String, activation_before: f32, activation_after: f32, interfering_memory_id: String, interference_type: [`InterferenceType`](#interferencetype), timestamp: DateTime&lt;Utc&gt; } | Memory weakened due to interference |
| `retrieval_competition` | `{ query_preview: String, winner_memory_id: String, suppressed_memory_ids: Vec<String>, competition_factor: f32, timestamp: DateTime<Utc> }` | Retrieval competition occurred (similar memories competed) |
| `pattern_triggered_replay` | `{ trigger_type: String, memory_ids: Vec<String>, pattern_confidence: f32, trigger_description: String, timestamp: DateTime<Utc> }` | Pattern-triggered replay initiated (not timer-based) |
| `entity_pattern_detected` | `{ entity_group: Vec<String>, memory_ids: Vec<String>, overlap_score: f32, confidence: f32, timestamp: DateTime<Utc> }` | Entity co-occurrence pattern detected |
| `semantic_cluster_formed` | `{ memory_ids: Vec<String>, cluster_size: usize, avg_similarity: f32, centroid_id: String, timestamp: DateTime<Utc> }` | Semantic cluster formed (dense similarity group) |
| `temporal_cluster_formed` | `{ memory_ids: Vec<String>, session_duration_secs: i64, session_id: Option<String>, timestamp: DateTime<Utc> }` | Temporal cluster (session) detected |
| `salience_spike_detected` | `{ memory_id: String, content_preview: String, importance: f32, arousal: f32, surprise_factor: f32, timestamp: DateTime<Utc> }` | Salience spike detected (high importance/arousal memory) |
| `behavioral_change_detected` | `{ change_type: String, affected_memory_ids: Vec<String>, context: String, timestamp: DateTime<Utc> }` | Behavioral pattern change triggered replay |
| `pattern_detected` | `{ trigger_type: String, description: String, memory_ids: Vec<String>, timestamp: DateTime<Utc> }` | Generic pattern detected (covers any trigger type) |
| `edge_promotion_boost_applied` | `{ memory_id: String, entity_name: String, old_tier: String, new_tier: String, importance_boost: f64, new_importance: f64, timestamp: DateTime<Utc> }` | Edge tier promotion boosted a memory's importance |
| `graph_orphan_detected` | `{ memory_id: String, entity_count: usize, compensatory_boost: f64, timestamp: DateTime<Utc> }` | Memory became a graph orphan (lost all edges) |
| `graph_adjusted_promotion` | `{ memory_id: String, base_threshold: f64, adjusted_threshold: f64, l2_plus_edge_count: usize, promoted: bool, timestamp: DateTime<Utc> }` | Memory tier promotion threshold adjusted by graph health |
| `graph_decay_consolidated` | `{ pruned_count: usize, orphaned_entities: usize, timestamp: DateTime<Utc> }` | Graph decay consolidated into single call site (double-decay fix diagnostic) |
| `sleep_time_block_rewritten` | `{ user_id: String, block_key: String, old_version: u32, new_version: u32, mode: String, trigger: String, token_spend: u32, model: String, timestamp: DateTime<Utc> }` | A sleep-time pass produced a successful `ContextBlock` rewrite. Carries the block key, old/new version numbers, mode, trigger, and token spend. Significant — persisted via `record_consolidation_event_for_user`. |
| `sleep_time_observation_emitted` | `{ user_id: String, memory_id: String, mode: String, trigger: String, token_spend: u32, confidence: f32, timestamp: DateTime<Utc> }` | A sleep-time pass emitted (and persisted) an observation memory. `memory_id` is the resulting memory; `mode` is the producing sleep mode. Significant. |
| `sleep_time_rewrite_aborted` | `{ user_id: String, block_key: Option<String>, mode: String, reason: String, timestamp: DateTime<Utc> }` | A sleep-time pass was blocked by budget / lock / concurrency, or its rewriter response failed validation. `reason` carries a short diagnostic. Not significant — routine for ops, not audit. |
| `sleep_time_budget_exhausted` | `{ user_id: String, what: String, timestamp: DateTime<Utc> }` | A per-user or global budget cap was hit. `what` carries the specific cap name (e.g. "tokens_per_hour", "calls_per_day", "global_tokens_per_day"). Not significant individually — operators monitor aggregate frequency via metrics. |

### ConsolidationEventsRequest

Defined in `src/handlers/consolidation.rs`. Request for consolidation events

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `since` | `Option<String>` | no | default when absent |

### ConsolidationInterference

Defined in `src/memory/introspection.rs`. Interference event details (SHO-106)

| Field | Type | Required | Notes |
|---|---|---|---|
| `new_memory_id` | `String` | yes |  |
| `old_memory_id` | `String` | yes |  |
| `similarity` | `f32` | yes | score |
| `interference_type` | [`InterferenceType`](#interferencetype) | yes |  |
| `timestamp` | `DateTime<Utc>` | yes |  |

### ConsolidationReport

Defined in `src/memory/introspection.rs`. Aggregated consolidation report for a time period

| Field | Type | Required | Notes |
|---|---|---|---|
| `period` | [`ReportPeriod`](#reportperiod) | yes | Time period covered |
| `strengthened_memories` | Vec&lt;[`MemoryChange`](#memorychange)&gt; | yes | Memories that got stronger |
| `decayed_memories` | Vec&lt;[`MemoryChange`](#memorychange)&gt; | yes | Memories that decayed |
| `formed_associations` | Vec&lt;[`AssociationChange`](#associationchange)&gt; | yes | New associations formed |
| `strengthened_associations` | Vec&lt;[`AssociationChange`](#associationchange)&gt; | yes | Associations that got stronger |
| `potentiated_associations` | Vec&lt;[`AssociationChange`](#associationchange)&gt; | yes | Associations that became permanent (LTP) |
| `pruned_associations` | Vec&lt;[`AssociationChange`](#associationchange)&gt; | yes | Associations that were pruned |
| `extracted_facts` | Vec&lt;[`FactChange`](#factchange)&gt; | yes | Facts extracted from episodic memories |
| `reinforced_facts` | Vec&lt;[`FactChange`](#factchange)&gt; | yes | Facts that were reinforced |
| `decayed_facts` | Vec&lt;[`FactChange`](#factchange)&gt; | yes | Facts that decayed due to lack of reinforcement |
| `deleted_facts` | Vec&lt;[`FactChange`](#factchange)&gt; | yes | Facts that were deleted (confidence too low) |
| `replayed_memories` | Vec&lt;[`ReplayEvent`](#replayevent)&gt; | yes | Memories that were replayed for consolidation |
| `interference_events` | Vec&lt;[`ConsolidationInterference`](#consolidationinterference)&gt; | yes | Interference events detected |
| `weakened_memories` | Vec&lt;[`MemoryChange`](#memorychange)&gt; | yes | Memories weakened due to interference |
| `statistics` | [`ConsolidationStats`](#consolidationstats) | yes | Aggregate statistics |

### ConsolidationReportRequest

Defined in `src/handlers/consolidation.rs`. Request for consolidation report

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `since` | `Option<String>` | no | default when absent |
| `until` | `Option<String>` | no | default when absent |

### ConsolidationStats

Defined in `src/memory/introspection.rs`. Aggregate statistics for consolidation

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_memories` | `usize` | yes |  |
| `memories_strengthened` | `usize` | yes |  |
| `memories_decayed` | `usize` | yes |  |
| `memories_at_risk` | `usize` | yes |  |
| `edges_formed` | `usize` | yes |  |
| `edges_strengthened` | `usize` | yes |  |
| `edges_potentiated` | `usize` | yes |  |
| `edges_pruned` | `usize` | yes |  |
| `facts_extracted` | `usize` | yes |  |
| `facts_reinforced` | `usize` | yes |  |
| `facts_decayed` | `usize` | yes |  |
| `facts_deleted` | `usize` | yes |  |
| `maintenance_cycles` | `usize` | yes |  |
| `total_maintenance_duration_ms` | `u64` | yes |  |
| `memories_replayed` | `usize` | yes |  |
| `replay_cycles` | `usize` | yes |  |
| `total_replay_priority` | `f32` | yes |  |
| `interference_events` | `usize` | yes |  |
| `memories_weakened` | `usize` | yes |  |
| `retrieval_competitions` | `usize` | yes |  |

### ContentKind

Defined in `src/memory/facets.rs`. Coarse content kind. Will eventually replace the existing `ExperienceType` enum as the WHAT discriminant once migration completes.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `observation` | none | Default — a recorded observation. |
| `decision` | none | A decision made. |
| `learning` | none | A distilled insight (mirrors `RecordKind::Learning`). |
| `action` | none | An action taken. |
| `reflection` | none | A reflection or meta-thought. |
| `conversation` | none | Conversational exchange. |
| `code` | none | Code or code-related artifact. |
| `document` | none | Document content. |
| `other` | none | Anything else. |

### ContextId

Defined in `src/memory/types.rs`. Unique identifier for contexts

On the wire this is its inner value, `Uuid`.

### ContextStatus

Defined in `src/handlers/types.rs`. Context status from Claude Code (MCP server reports this via status line)

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | `Option<String>` | no |  |
| `tokens_used` | `u64` | yes |  |
| `tokens_budget` | `u64` | yes |  |
| `percent_used` | `u8` | yes |  |
| `current_task` | `Option<String>` | no |  |
| `model` | `Option<String>` | no |  |
| `updated_at` | `chrono::DateTime<chrono::Utc>` | yes |  |

### ContextStatusRequest

Defined in `src/handlers/health.rs`. Context status request from Claude Code status line

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | `String` | yes |  |
| `tokens_used` | `u64` | yes |  |
| `tokens_budget` | `u64` | yes |  |
| `current_dir` | `Option<String>` | no |  |
| `model` | `Option<String>` | no |  |

### ContextStatusUpdateResponse

Defined in `src/handlers/health.rs`. Reply of `POST /api/context/status`: `Updated` with 200, `Rejected` with 400 for a `session_id` over 128 characters or 503 when 10 000 sessions are already tracked.

Enum, untagged: the variant's payload alone.

| Variant | Payload | Notes |
|---|---|---|
| `Updated` | `{ success: bool, percent_used: u8 }` |  |
| `Rejected` | `{ error: String }` |  |

### ContextSummaryRequest

Defined in `src/handlers/recall.rs`. Context summary request

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `include_decisions` | `bool` | no | default when absent |
| `include_learnings` | `bool` | no | default when absent |
| `include_context` | `bool` | no | default when absent |
| `max_items` | `usize` | no | default when absent |

### ContextSummaryResponse

Defined in `src/handlers/recall.rs`. Context summary response - categorized memories for session bootstrap

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_memories` | `usize` | yes |  |
| `decisions` | Vec&lt;[`SummaryItem`](#summaryitem)&gt; | yes |  |
| `learnings` | Vec&lt;[`SummaryItem`](#summaryitem)&gt; | yes |  |
| `context` | Vec&lt;[`SummaryItem`](#summaryitem)&gt; | yes |  |
| `patterns` | Vec&lt;[`SummaryItem`](#summaryitem)&gt; | yes |  |
| `errors` | Vec&lt;[`SummaryItem`](#summaryitem)&gt; | yes |  |

### ConversationContext

Defined in `src/memory/types.rs`. Conversation-specific context

| Field | Type | Required | Notes |
|---|---|---|---|
| `conversation_id` | `Option<String>` | no | Current conversation ID |
| `topic` | `Option<String>` | no | Topic being discussed |
| `recent_messages` | `Vec<String>` | yes | Recent messages (last N turns) |
| `mentioned_entities` | `Vec<String>` | yes | Key entities mentioned |
| `active_intents` | `Vec<String>` | yes | Active questions/intents |
| `tone` | `Option<String>` | no | Conversation mood/tone |

### CountCard

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `key` | `String` | yes |  |
| `count` | `usize` | yes |  |

### CountMemoriesQuery

Defined in `src/handlers/crud.rs`. Query parameters for GET /api/memories/count

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `type` | `Option<String>` | no |  |
| `tags` | `Option<String>` | no | Comma-separated tags; every one must be present on the memory, matched exactly. See [`parse_tag_filter`]. |

### CountResponse

Defined in `src/handlers/crud.rs`. Response for GET /api/memories/count

| Field | Type | Required | Notes |
|---|---|---|---|
| `total` | `usize` | yes | Memories matching the filters. |
| `tags` | `Vec<String>` | yes | The tag filter as applied (parsed, trimmed, empties dropped), so a caller can see what the count was measured against. |

### CreateABTestRequest

Defined in `src/handlers/ab_testing.rs`. Request to create a new A/B test

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `description` | `Option<String>` | no | default when absent |
| `control_weights` | Option&lt;relevance::[`LearnedSignalWeights`](#learnedsignalweights)&gt; | no | default when absent |
| `treatment_weights` | Option&lt;relevance::[`LearnedSignalWeights`](#learnedsignalweights)&gt; | no | default when absent |
| `traffic_split` | `f32` | no | default when absent |
| `min_impressions` | `u64` | no | default when absent |
| `max_duration_hours` | `Option<u64>` | no | default when absent |
| `tags` | `Vec<String>` | no | default when absent |

### CreateBackupRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### CreateDatasetRequest

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `schema` | [`DatasetSchema`](#datasetschema) | yes |  |

### CreateDatasetResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `dataset` | [`DatasetRef`](#datasetref) | yes |  |

### CreateProjectRequest

Defined in `src/handlers/todos.rs`. Request to create a project

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `prefix` | `Option<String>` | no | default when absent |
| `description` | `Option<String>` | no | default when absent |
| `color` | `Option<String>` | no | default when absent |
| `parent` | `Option<String>` | no | default when absent |

### CreateReminderRequest

Defined in `src/handlers/todos.rs`. Request to create a new reminder

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `trigger` | [`ReminderTriggerRequest`](#remindertriggerrequest) | yes |  |
| `tags` | `Vec<String>` | no | default when absent |
| `priority` | `u8` | no | default when absent |

### CreateReminderResponse

Defined in `src/handlers/todos.rs`. Response for reminder creation

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `trigger_type` | `String` | yes |  |
| `due_at` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `created_at` | `chrono::DateTime<chrono::Utc>` | yes |  |

### CreateTodoRequest

Defined in `src/handlers/todos.rs`. Request to create a new todo

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `status` | `Option<String>` | no | default when absent |
| `priority` | `Option<String>` | no | default when absent |
| `project` | `Option<String>` | no | default when absent |
| `contexts` | `Option<Vec<String>>` | no | default when absent |
| `due_date` | `Option<String>` | no | default when absent |
| `blocked_on` | `Option<String>` | no | default when absent |
| `parent_id` | `Option<String>` | no | default when absent |
| `tags` | `Option<Vec<String>>` | no | default when absent |
| `notes` | `Option<String>` | no | default when absent |
| `recurrence` | `Option<String>` | no | default when absent |
| `external_id` | `Option<String>` | no | default when absent |
| `depends_on` | `Option<Vec<String>>` | no | default when absent; Structured dependencies: list of todo short IDs (e.g. ["SHO-1", "BOLT-3"]) or UUIDs. This todo cannot start until all listed todos are Done. `blocked_by` is upstream's name for the same list. An unknown reference is a 400. |
| `related_memory_ids` | `Option<Vec<String>>` | no | default when absent; Memory UUIDs that motivated this todo, verified to exist. |

### DatasetMeta

Defined in `src/datasets/store.rs`. Catalog-level metadata about a dataset.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `schema` | [`DatasetSchema`](#datasetschema) | yes |  |
| `row_count` | `u64` | yes |  |
| `created_at` | `DateTime<Utc>` | yes |  |

### DatasetMetadataResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `dataset` | [`DatasetRef`](#datasetref) | yes |  |
| `meta` | [`DatasetMeta`](#datasetmeta) | yes |  |

### DatasetRef

Defined in `src/datasets/store.rs`. Stable handle to a dataset across the API surface.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `table` | `String` | yes |  |

### DatasetRow

Defined in `src/datasets/store.rs`. A single row to be inserted.

| Field | Type | Required | Notes |
|---|---|---|---|
| `values` | `HashMap<String, serde_json::Value>` | yes |  |

### DatasetSchema

Defined in `src/datasets/schema.rs`. Structural schema of a single dataset.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `columns` | Vec&lt;[`ColumnDef`](#columndef)&gt; | yes |  |
| `primary_key` | `Vec<String>` | yes |  |

### DatasetUserQuery

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |

### DeclaredEntityInput

Defined in `src/declared_entities.rs`. One declared entity as it arrives on the wire.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `label` | `String` | yes |  |
| `weight` | `Option<f64>` | no | default when absent |

### DeclaredWriter

Defined in `src/memory/facets.rs`. Writer identity AS CLAIMED BY THE CALLER. Every field is an assertion by whoever made the request: recorded because it is useful, trusted for nothing.

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | `Option<String>` | no | default when absent; The caller's own session identifier. |
| `previous_session_id` | `Option<String>` | no | default when absent; The id this session carried BEFORE a restart, where the caller knows it. Without it, the records either side of a restart are two unrelated writers; with it they are one thread a reader can follow backwards. |
| `session_name` | `Option<String>` | no | default when absent; Human-facing name the fleet addresses this session by, e.g. `"commitwork-bb"`. Contended by construction — names get reassigned, and two live sessions have been observed answering to one name — so it is a label for reading, never a key. |
| `transcript_id` | `Option<String>` | no | default when absent; Transcript the write happened in. A CORRELATION KEY, not an address: one transcript can be held by two live processes at once, so it groups records without identifying a writer. |
| `model_id` | `Option<String>` | no | default when absent; Model that ran the command producing this record, e.g. `"claude-opus-5[1m]"`. The field most likely to be wanted later and impossible to recover afterwards: when a memory turns out to be wrong, this decides whether it is one bad record or a class of them, and nothing else in the record can answer that. |
| `workspace` | `Option<PathBuf>` | no | default when absent; Workspace or repository root the CALLER was working in — as distinct from `worktree_path` above, which is the server's. |

### DecompressMemoryRequest

Defined in `src/handlers/compression.rs`. Request for decompressing a memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `id` | `String` | yes |  |

### DecompressMemoryResponse

Defined in `src/handlers/compression.rs`. Reply of `POST /api/memory/decompress`. `memory` is present only when a compressed record was decompressed.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `String` | yes |  |
| `was_compressed` | `bool` | yes |  |
| `memory` | Option&lt;[`DecompressedMemory`](#decompressedmemory)&gt; | no | omitted when empty |

### DecompressedMemory

Defined in `src/handlers/compression.rs`. The decompressed record of a `POST /api/memory/decompress` reply.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `importance` | `f32` | yes | score |

### DeleteEntityRequest

Defined in `src/handlers/graph.rs`. Request body for `DELETE /api/graph/entity`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes | Owning tenant. Checked against the authenticated binding by `resolve_request_user_id`, which rejects and audits a mismatch. |
| `entity_id` | `String` | yes | The entity's UUID. Deletion is by id, never by name: a name can collide across tenants and is user-controlled, an id cannot be guessed into. |

### DeleteEntityResponse

Defined in `src/handlers/graph.rs`. Reply of `DELETE /api/graph/entity/delete`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted` | `bool` | yes | False when no entity had that id. |
| `entity_id` | `String` | yes |  |
| `user_id` | `String` | yes |  |

### DeleteMemoryResponse

Defined in `src/handlers/crud.rs`. Response for memory delete operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `id` | `String` | yes |  |
| `message` | `String` | yes |  |

### DeleteProjectRequest

Defined in `src/handlers/todos.rs`. Request to delete a project

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `delete_todos` | `bool` | no | default when absent |

### DeleteReminderQuery

Defined in `src/handlers/todos.rs`. Query for delete reminder

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### DeleteUserResponse

Defined in `src/handlers/users.rs`. Response for user deletion

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `user_id` | `String` | yes |  |
| `message` | `String` | yes |  |

### DependencyChainQuery

Defined in `src/handlers/todos.rs`. Params for dependency chain walk

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `direction` | `String` | no | default when absent |
| `max_depth` | `usize` | no | default when absent |

### DerivedAnswer

Defined in `src/handlers/temporal_derivation.rs`. A derived answer to a "when did/is X?" temporal question (B4).

| Field | Type | Required | Notes |
|---|---|---|---|
| `derived_date` | `String` | yes | ISO `YYYY-MM-DD` (anchor paths) or the fact's resolved-time repr (temporal-fact path, may be coarser e.g. `2023-06`). |
| `method` | `String` | yes | `"anchor_relative"` \| `"event_anchor"` \| `"temporal_fact"`. |
| `supporting_facts` | Vec&lt;[`TemporalFactWire`](#temporalfactwire)&gt; | yes | Stored facts that justified the derived date (empty for pure query-computed anchor-relative answers). |
| `source_memories` | `Vec<String>` | yes | Source memory ids backing the supporting facts. |

### DetectedEntity

Defined in `src/relevance.rs`. Detected entity with confidence

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes | Entity name |
| `entity_type` | `String` | yes | Entity type (Person, Organization, etc.) |
| `confidence` | `f32` | yes | score; Detection confidence (0.0-1.0) |

### DetectedEntityInfo

Defined in `src/handlers/recall.rs`. Entity detected in the query context

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `entity_type` | `String` | yes |  |

### DialSlider

Defined in `src/handlers/dials.rs`. The ranges of one recall slider.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `&'static str` | yes | The `CognitiveDials` field the slider sets. |
| `group` | `&'static str` | yes |  |
| `label` | `&'static str` | yes |  |
| `default` | `f64` | yes |  |
| `recommended` | `[f64; 2]` | yes | `[low, high]` of the recommended band. |
| `absolute` | `[f64; 2]` | yes | `[low, high]` the slider track spans. |
| `step` | `f64` | yes |  |

### DialsQuery

Defined in `src/handlers/dials.rs`. Query for `GET /api/dials`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### DialsSchemaResponse

Defined in `src/handlers/dials.rs`. Reply of `GET /api/dials/schema`: one entry per recall slider.

| Field | Type | Required | Notes |
|---|---|---|---|
| `sliders` | Vec&lt;[`DialSlider`](#dialslider)&gt; | yes |  |

### DismissReminderRequest

Defined in `src/handlers/todos.rs`. Request to dismiss a reminder

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### DocumentContext

Defined in `src/memory/types.rs`. Document-specific context

| Field | Type | Required | Notes |
|---|---|---|---|
| `document_id` | `Option<String>` | no | Current document ID |
| `document_type` | `Option<String>` | no | Document type |
| `current_section` | `Option<String>` | no | Section/chapter being read |
| `related_documents` | `Vec<String>` | yes | Related documents |
| `citations` | `Vec<String>` | yes | Citations/references |
| `categories` | `Vec<String>` | yes | Document tags/categories |

### DrainBacklogResponse

Defined in `src/handlers/consolidation.rs`. Reply of `POST /api/embeddings/backlog/drain`. `reconciled` is `null` unless the request set `reconcile`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `reconciled` | Option&lt;crate::memory::embedding_backlog::[`ReconcileReport`](#reconcilereport)&gt; | no |  |
| `drained` | crate::memory::embedding_backlog::[`DrainReport`](#drainreport) | yes |  |

### DrainEmbeddingBacklogRequest

Defined in `src/handlers/consolidation.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `Option<usize>` | no |  |
| `reconcile` | `bool` | no | default when absent |

### DrainReport

Defined in `src/memory/embedding_backlog.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `attempted` | `usize` | yes |  |
| `drained` | `usize` | yes |  |
| `dropped` | `usize` | yes |  |
| `failed` | `usize` | yes |  |
| `remaining` | `usize` | yes |  |
| `stopped` | `Option<String>` | no | omitted when empty |

### DropDatasetResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `dropped` | `bool` | yes |  |

### DueRemindersResponse

Defined in `src/handlers/todos.rs`. Response for due reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `reminders` | Vec&lt;[`ReminderItem`](#reminderitem)&gt; | yes |  |
| `count` | `usize` | yes |  |

### DueTodosRequest

Defined in `src/handlers/todos.rs`. Request to get due todos

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `include_overdue` | `bool` | no | default when absent |

### EdgeSource

Defined in `src/graph_memory.rs`. Creation source for relationship edges (FIX-06: dream replay tagging)

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `CoOccurrence` | none | Created from entity co-occurrence in user-stored memories |
| `DreamReplay` | none | Created by waking dream replay (maintenance consolidation) |
| `SleepReplay` | none | Created by sleep-phase replay (deep consolidation) |
| `Coactivation` | none | Created by Hebbian coactivation during retrieval |
| `Explicit` | none | Created from MIF import or explicit user input |
| `CollectivePrior` | none | Edge seeded from population-level collective knowledge. Imported from collective store during new-user initialization. Starts at low strength, rapidly potentiated if confirmed by user behavior. |
| `Unknown` | none | Unknown edge source (forward compatibility for future variants). Deserialized from a variant not recognized by this version. |

### EdgeTier

Defined in `src/graph_memory.rs`. Memory tier for edge consolidation

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `L1Working` | none | Working memory tier: new edges, dense, aggressive decay |
| `L2Episodic` | none | Episodic memory tier: proven edges, moderate decay |
| `L3Semantic` | none | Semantic memory tier: consolidated edges, near-permanent |

### EmbeddingBacklogQuery

Defined in `src/handlers/consolidation.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### EmotionalContext

Defined in `src/memory/types.rs`. Emotional context - captures affective state during memory formation

| Field | Type | Required | Notes |
|---|---|---|---|
| `valence` | `f32` | no | default when absent; Emotional valence: -1.0 (very negative) to 1.0 (very positive) 0.0 = neutral |
| `arousal` | `f32` | no | default when absent; Arousal level: 0.0 (calm) to 1.0 (highly aroused/excited) |
| `dominant_emotion` | `Option<String>` | no | default when absent; Dominant emotion label (optional, for categorical access) E.g., "joy", "frustration", "surprise", "satisfaction", "anxiety" |
| `content_sentiment` | `Option<f32>` | no | default when absent; Sentiment of the content itself (not the user's reaction) Useful for distinguishing "user felt frustrated" vs "content describes frustration" |
| `confidence` | `f32` | no | default when absent; score; Confidence in the emotional assessment (0.0 to 1.0) Lower if inferred from text, higher if explicitly stated |

### EncodedExport

Defined in `src/handlers/mif.rs`. A non-JSON export of `POST /api/export/mif`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `format` | `String` | yes |  |
| `data` | `String` | yes | The adapter's output, as text. |
| `memory_count` | `usize` | yes |  |
| `todo_count` | `usize` | yes |  |

### EndSessionRequest

Defined in `src/handlers/sessions.rs`. Request for ending a session

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `reason` | `String` | no | default when absent |

### EndSessionResponse

Defined in `src/handlers/sessions.rs`. Response for ending a session

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `session` | Option&lt;[`Session`](#session)&gt; | no |  |

### EngramBinding

Defined in `src/memory/facets.rs`. Cross-facet binding strength + presence — the hippocampal-index analogue.

| Field | Type | Required | Notes |
|---|---|---|---|
| `conjunctive_key` | `Option<String>` | no | Stable hash over the engram's W-anchors (who_id, where_id, when_bucket, why_id). Collision = candidate duplicate / reconsolidation target. |
| `binding_strength` | `f32` | yes | Recollection strength of the binding itself, 0.0 = gist only, 1.0 = full episodic detail. Decays faster than the W-facets it binds — models hippocampal-dependent detail loss while neocortical gist persists. |
| `present` | [`WFacetMask`](#wfacetmask) | yes | Which W-facets carry usable data for this engram. |

### EnqueueRequest

Defined in `src/handlers/sleep_time.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `mode` | `String` | yes |  |
| `trigger` | `String` | yes |  |

### EnrolBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `unlock` | [`UnlockBody`](#unlockbody) | yes |  |
| `provider` | `String` | yes | `secure-enclave`: the Monitor ran `veld-enclave enrol` on this Mac and posts the secret it released. |
| `id` | `String` | yes |  |
| `label` | `Option<String>` | no |  |
| `secret` | `String` | yes |  |

### EntityCard

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `labels` | `Vec<String>` | yes |  |
| `mention_count` | `usize` | yes |  |
| `first_learned` | `DateTime<Utc>` | yes |  |
| `last_seen` | `DateTime<Utc>` | yes |  |

### EntityFrequencyRequest

Defined in `src/handlers/graph.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `since` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `until` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `bucket` | [`Bucket`](#bucket) | no | default when absent |
| `limit` | `Option<usize>` | no |  |
| `shelf` | Option&lt;[`Shelf`](#shelf)&gt; | no | default when absent |

### EntityLabel

Defined in `src/graph_memory.rs`. Entity labels for ontological classification of graph nodes.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Person` | none |  |
| `Organization` | none |  |
| `Location` | none |  |
| `Technology` | none |  |
| `Concept` | none |  |
| `Event` | none |  |
| `Date` | none |  |
| `Product` | none |  |
| `Skill` | none |  |
| `Keyword` | none | YAKE-extracted discriminative keyword (not a named entity) Used for graph-based retrieval of rare/important terms like "sunrise" |
| `Project` | none | Git repos, Jira projects, feature work, mission plans |
| `Task` | none | Issues, tickets, work items, action items |
| `Document` | none | READMEs, specs, design docs, RFCs, runbooks |
| `Repository` | none | Git repositories, container registries |
| `Service` | none | Microservices, APIs, daemons, ROS2 nodes |
| `Database` | none | PostgreSQL, Redis, RocksDB, DynamoDB instances |
| `Metric` | none | Latency p99, error rate, SLOs, telemetry signals |
| `Configuration` | none | Env vars, feature flags, config files, parameters |
| `Environment` | none | staging, production, dev, CI, robot deployment zones |
| `Pipeline` | none | CI/CD pipelines, data pipelines, ETL, mission workflows |
| `Team` | none | Engineering squads, SRE teams, robot fleet groups |
| `Role` | none | SRE, tech lead, PM, architect, operator |
| `Module` | none | Code modules, packages, crates, libraries, ROS2 packages |
| `Other` | `String` |  |
| `FilePath` | none | A file in a named tree, canonically `repo:relative/path` |
| `Commit` | none | A commit, canonically `repo@sha` |
| `Symbol` | none | A code symbol: function, type, constant |
| `Session` | none | An agent session, canonically `cc:<sessionId>` |
| `Agent` | none | An agent within a session, canonically `cc:<sessionId>/<agentId>` |
| `Tool` | none | A tool or skill an agent invoked, canonically `tool:<name>` / `skill:/<name>` |
| `Template` | none | A prompt or tool template, canonically `template:<tool>@<sha8>` |

### EntityNode

Defined in `src/graph_memory.rs`. Entity node in the knowledge graph

| Field | Type | Required | Notes |
|---|---|---|---|
| `uuid` | `Uuid` | yes | Unique identifier |
| `name` | `String` | yes | Entity name (e.g., "John", "Paris", "Rust programming") |
| `labels` | Vec&lt;[`EntityLabel`](#entitylabel)&gt; | yes | Entity labels/types (e.g., ["Person"], ["Location", "City"]) |
| `created_at` | `DateTime<Utc>` | yes | When this entity was first created in the graph |
| `last_seen_at` | `DateTime<Utc>` | yes | When this entity was last observed |
| `mention_count` | `usize` | yes | How many times this entity has been mentioned |
| `summary` | `String` | yes | Summary of this entity's context (built from surrounding edges) |
| `attributes` | `HashMap<String, String>` | yes | Additional attributes based on entity type |
| `name_embedding` | `Option<Vec<f32>>` | no | Semantic embedding of the entity name (for similarity search) |
| `salience` | `f32` | no | default when absent; Salience score (0.0 - 1.0): How important is this entity? Higher salience = larger gravitational well in the memory universe Factors: proper noun status, mention frequency, recency, user-defined importance |
| `is_proper_noun` | `bool` | no | default when absent; Whether this is a proper noun (names, places, products) Proper nouns have higher base salience than common nouns |
| `pii_classification` | [`PiiClassification`](#piiclassification) | no | default when absent; PII classification for cross-user safety. Derived from entity labels: Person → PersonalIdentity, Organization → OrganizationIdentity, otherwise Clean. |

### EntityProvenance

Defined in `src/handlers/graph.rs`. Per-tag provenance folded over every memory that stored the tag. Absent when no row names the tag: records written before the store existed are undetermined, not caller-authored.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | `usize` | yes |  |
| `redacted` | `usize` | yes |  |
| `sources` | `std::collections::BTreeMap<&'static str, usize>` | yes |  |

### EntityRef

Defined in `src/memory/types.rs`. Entity reference - lightweight link from Memory to GraphMemory entities

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_id` | `Uuid` | yes | UUID of the entity in GraphMemory |
| `name` | `String` | yes | Entity name for quick access without graph lookup |
| `relation` | `String` | yes | Relationship type (e.g., "mentioned", "subject", "location") |

### EntityRelationship

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `from` | `String` | yes |  |
| `to` | `String` | yes |  |
| `relation_type` | `String` | yes |  |
| `strength` | `f32` | yes |  |
| `context` | `String` | yes |  |

### EntityResolution

Defined in `src/memory/types.rs`. Entity resolution result for retrieval feedback (FIX-09)

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes | Entity name from the query |
| `status` | `String` | yes | Whether the entity was found in the graph |
| `resolved_to` | `Option<String>` | no | omitted when empty; UUID of the resolved entity (if found) |

### EntityResolveRequest

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `names` | `Vec<String>` | yes |  |

### EntityResolveResponse

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `resolved` | Vec&lt;[`ResolvedEntity`](#resolvedentity)&gt; | yes |  |
| `unresolved` | `Vec<String>` | yes |  |

### EnvironmentContext

Defined in `src/memory/types.rs`. Environment context

| Field | Type | Required | Notes |
|---|---|---|---|
| `os` | `Option<String>` | no | Operating system |
| `device` | `Option<String>` | no | Device type |
| `screen_size` | `Option<String>` | no | Screen resolution/size |
| `location` | `Option<String>` | no | Location (if available) |
| `network` | `Option<String>` | no | Network status |
| `resources` | `HashMap<String, String>` | yes | System resource usage |

### EpisodeContext

Defined in `src/memory/types.rs`. Episode context - groups memories into coherent episodes

| Field | Type | Required | Notes |
|---|---|---|---|
| `episode_id` | `Option<String>` | no | default when absent; Unique episode identifier All memories in the same conversation/session share this ID |
| `sequence_number` | `Option<u32>` | no | default when absent; Sequence number within the episode (1, 2, 3, ...) Enables temporal ordering within an episode |
| `preceding_memory_id` | `Option<String>` | no | default when absent; ID of the immediately preceding memory (temporal chain) |
| `episode_type` | `Option<String>` | no | default when absent; Episode type (conversation, task, session, etc.) |
| `episode_start` | `Option<DateTime<Utc>>` | no | default when absent; Episode start time |
| `is_episode_start` | `bool` | no | default when absent; Is this the first memory in the episode? |
| `is_episode_end` | `bool` | no | default when absent; Is this the last memory in the episode? |
| `parent_episode_id` | `Option<String>` | no | default when absent; Parent episode (for hierarchical episodes) E.g., a conversation within a larger task session |

### EpisodeSource

Defined in `src/graph_memory.rs`. Episode source types

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Message` | none |  |
| `Document` | none |  |
| `Event` | none |  |
| `Observation` | none |  |

### EpisodicNode

Defined in `src/graph_memory.rs`. Episodic node representing a discrete experience/memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `uuid` | `Uuid` | yes | Unique identifier |
| `name` | `String` | yes | Human-readable name/title |
| `content` | `String` | yes | Episode content (the actual experience data) |
| `valid_at` | `DateTime<Utc>` | yes | When the original event occurred (event time) |
| `created_at` | `DateTime<Utc>` | yes | When this was ingested into the system (ingestion time) |
| `entity_refs` | `Vec<Uuid>` | yes | Entities extracted from this episode |
| `source` | [`EpisodeSource`](#episodesource) | yes | Source type |
| `metadata` | `HashMap<String, String>` | yes | Additional metadata |

### ExecuteResponse

Defined in `src/handlers/query.rs`. Response for `POST /api/query/execute`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `results` | Vec&lt;[`ScoredMemoryIdWire`](#scoredmemoryidwire)&gt; | yes |  |
| `total` | `usize` | yes |  |
| `derived_answer` | Option&lt;[`DerivedAnswer`](#derivedanswer)&gt; | no | omitted when empty; B4 temporal answer-derivation. `None` (and omitted from JSON) when no temporal answer is derivable — response is then byte-identical to the pre-B4 shape. Never affects `results`/`total`. |

### Experience

Defined in `src/memory/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `experience_type` | [`ExperienceType`](#experiencetype) | no | default when absent; Type of experience (defaults to Observation) |
| `content` | `String` | yes | Content of the experience (REQUIRED) |
| `context` | Option&lt;[`RichContext`](#richcontext)&gt; | no | default when absent; RICH CONTEXT instead of simple string (optional, null by default) |
| `entities` | `Vec<String>` | no | default when absent; Extracted entities (empty by default, auto-extracted if empty) |
| `metadata` | `HashMap<String, String>` | no | default when absent; Additional metadata (empty by default) |
| `embeddings` | `Option<Vec<f32>>` | no | default when absent; Content embeddings (optional, auto-generated if null) |
| `embeddings_secondary` | `Option<Vec<f32>>` | no | default when absent; Secondary content embeddings (optional, from Nomic-embed-text-v1.5 768d). Stored alongside primary embeddings for dual-index competitive retrieval. Both models compete at query time — the one finding a better match wins. |
| `embedding_degraded` | `bool` | no | default when absent; Whether the embedding was generated by a fallback hash (not semantic). Memories with degraded embeddings are re-embedded during maintenance when the embedding model recovers. |
| `image_embeddings` | `Option<Vec<f32>>` | no | default when absent; Image embeddings from CLIP/SigLIP (512-768 dims) Generated from attached images via vision encoder |
| `audio_embeddings` | `Option<Vec<f32>>` | no | default when absent; Audio embeddings from Whisper/wav2vec (768-1024 dims) Generated from attached audio via speech encoder |
| `video_embeddings` | `Option<Vec<f32>>` | no | default when absent; Video embeddings (768+ dims) Can be single embedding (averaged frames) or keyframe embedding |
| `media_refs` | Vec&lt;[`MediaRef`](#mediaref)&gt; | no | default when absent; References to attached media files Stores metadata and URIs, not raw bytes |
| `related_memories` | Vec&lt;[`MemoryId`](#memoryid)&gt; | no | default when absent; Related memories (empty by default) |
| `causal_chain` | Vec&lt;[`MemoryId`](#memoryid)&gt; | no | default when absent; Causality chain - what led to this (empty by default) |
| `outcomes` | `Vec<String>` | no | default when absent; Outcome/result - what happened after (empty by default) |
| `robot_id` | `Option<String>` | no | default when absent; Robot/drone identifier for multi-agent systems |
| `mission_id` | `Option<String>` | no | default when absent; Mission identifier this experience belongs to |
| `geo_location` | `Option<[f64; 3]>` | no | default when absent; GPS coordinates (latitude, longitude, altitude) |
| `local_position` | `Option<[f32; 3]>` | no | default when absent; Local coordinates (x, y, z in meters) |
| `heading` | `Option<f32>` | no | default when absent; Heading in degrees (0-360) |
| `action_type` | `Option<String>` | no | default when absent; Action that was performed (for action-outcome learning) |
| `reward` | `Option<f32>` | no | default when absent; Reward signal for reinforcement learning (-1.0 to 1.0) |
| `sensor_data` | `HashMap<String, f64>` | no | default when absent; Sensor readings at time of experience |
| `decision_context` | `Option<HashMap<String, String>>` | no | default when absent; Decision context: What state/conditions led to this decision? E.g., "battery_low=true, obstacle_ahead=true, weather=windy" |
| `action_params` | `Option<HashMap<String, String>>` | no | default when absent; Action parameters: Specific parameters of the action taken E.g., {"speed": "0.5", "turn_angle": "45", "altitude_change": "-10"} |
| `outcome_type` | `Option<String>` | no | default when absent; Outcome type: success, failure, partial, aborted, timeout |
| `outcome_details` | `Option<String>` | no | default when absent; Outcome details: What specifically happened? |
| `confidence` | `Option<f32>` | no | default when absent; score; Confidence score for this decision (0.0-1.0) How confident was the system when making this decision? |
| `alternatives_considered` | `Vec<String>` | no | default when absent; Alternative actions considered but not taken For learning "what else could have been done" |
| `weather` | `Option<HashMap<String, String>>` | no | default when absent; Weather conditions: {"wind_speed": "15", "visibility": "good", "precipitation": "none"} |
| `terrain_type` | `Option<String>` | no | default when absent; Terrain type: indoor, outdoor, urban, rural, water, aerial |
| `lighting` | `Option<String>` | no | default when absent; Lighting conditions: bright, dim, dark, variable |
| `nearby_agents` | `Vec<HashMap<String, String>>` | no | default when absent; Other agents detected: [{"id": "drone_002", "distance": "50m", "type": "friendly"}] |
| `is_failure` | `bool` | no | default when absent; Is this a failure/error event? |
| `is_anomaly` | `bool` | no | default when absent; Is this an anomaly (unexpected sensor reading, behavior, etc.)? |
| `severity` | `Option<String>` | no | default when absent; Severity level: info, warning, error, critical |
| `recovery_action` | `Option<String>` | no | default when absent; Recovery action taken (if this was a failure) |
| `root_cause` | `Option<String>` | no | default when absent; Root cause (if known) |
| `pattern_id` | `Option<String>` | no | default when absent; Pattern ID this experience matches (if recognized) |
| `predicted_outcome` | `Option<String>` | no | default when absent; Predicted outcome before action was taken |
| `prediction_accurate` | `Option<bool>` | no | default when absent; Was the prediction correct? |
| `tags` | `Vec<String>` | no | default when absent; Tags for quick filtering: ["obstacle", "battery", "navigation", "emergency"] |
| `temporal_refs` | `Vec<String>` | no | default when absent; Extracted temporal references from content (dates mentioned in text) Used for temporal filtering in retrieval (key for multi-hop accuracy) E.g., ["2023-05-07", "2023-06-15"] for memories mentioning specific dates |
| `ner_entities` | Vec&lt;[`NerEntityRecord`](#nerentityrecord)&gt; | no | default when absent; Structured NER entities with type and confidence (preserves NER classification) Populated by handler during remember/upsert. Consumed by graph insertion to create entities with proper labels (Person, Organization, Location, etc.) |
| `cooccurrence_pairs` | `Vec<(String, String)>` | no | default when absent; Co-occurrence pairs extracted from content (sentence-level proximity) Pre-computed by handler to avoid redundant content parsing in downstream passes |

### ExperienceType

Defined in `src/memory/types.rs`. Experience types that can be recorded

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Conversation` | none |  |
| `Decision` | none |  |
| `Error` | none |  |
| `Learning` | none |  |
| `Discovery` | none |  |
| `Pattern` | none |  |
| `Context` | none |  |
| `Task` | none |  |
| `CodeEdit` | none |  |
| `FileAccess` | none |  |
| `Search` | none |  |
| `Command` | none |  |
| `Observation` | none |  |
| `Intention` | none | Prospective memory - future intention/reminder (SHO-116) Filtered from normal recall, surfaces via dedicated reminder queries or when context triggers via spreading activation |

### ExportMifResponse

Defined in `src/handlers/mif.rs`. Reply of `POST /api/export/mif`: the MIF document itself for `format` `veld` or `json`, else the adapter's text output wrapped with its counts.

Enum, untagged: the variant's payload alone.

| Variant | Payload | Notes |
|---|---|---|
| `Document` | Box&lt;[`MifDocument`](#mifdocument)&gt; |  |
| `Encoded` | [`EncodedExport`](#encodedexport) |  |

### ExternalIdRecord

Defined in `src/handlers/crud.rs`. What veld holds under an external id: enough to tell whether a write whose response never arrived (a 408) landed, without reading the record.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `external_id` | `String` | yes |  |
| `version` | `u32` | yes |  |
| `content_sha256` | `Option<String>` | no | SHA-256 (hex) of the content as written. `None` when a lossy compressor replaced the body, so no written content can be compared with it. |
| `forgotten` | `bool` | yes |  |

### FactCard

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `fact` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `support_count` | `usize` | yes |  |
| `source_memories` | `Vec<String>` | yes |  |
| `related_entities` | `Vec<String>` | yes |  |
| `fact_type` | `String` | yes |  |
| `created_at` | `DateTime<Utc>` | yes |  |
| `last_reinforced` | `DateTime<Utc>` | yes |  |
| `supersedes` | `Vec<String>` | yes |  |
| `single_source` | `bool` | yes | A single supporting memory: shown, but labelled as one memory's claim. |

### FactChange

Defined in `src/memory/introspection.rs`. Change in a semantic fact

| Field | Type | Required | Notes |
|---|---|---|---|
| `fact_id` | `String` | yes |  |
| `fact_content` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `support_count` | `usize` | yes |  |
| `fact_type` | `String` | yes |  |
| `timestamp` | `DateTime<Utc>` | yes |  |

### FactCluster

Defined in `src/memory/compression.rs`. A cluster of related facts grouped by shared entities, with a template- generated narrative and heuristic causal chain detection. Ported from `shodh-memory`'s fact narrative subsystem; clustering algorithm is document-frequency-based topic discovery (lowest-DF non-hub entity = topic).

| Field | Type | Required | Notes |
|---|---|---|---|
| `topic` | `String` | yes | Most common entity in the cluster (used as topic heading) |
| `entities` | `Vec<String>` | yes | All entities appearing in the cluster's facts (sorted, deduped) |
| `facts` | Vec&lt;[`SemanticFact`](#semanticfact)&gt; | yes | Facts in this cluster, sorted by created_at ascending |
| `narrative` | `String` | yes | Template-generated narrative summary (no LLM required) |
| `avg_confidence` | `f32` | yes | score; Average confidence across facts in the cluster |
| `total_support` | `usize` | yes | Sum of support_count across all facts (drives cluster ranking) |
| `causal_chains` | Vec&lt;[`CausalFactLink`](#causalfactlink)&gt; | yes | Detected causal relationships between facts in this cluster |

### FactNarrativesRequest

Defined in `src/handlers/facts.rs`. Request for fact narratives. Limit is clamped to 50 server-side; values outside `[1, 50]` are coerced into range (no 400 on overshoot — agents that ask for "all" via a large limit get the cap automatically).

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |
| `entity_filter` | `Option<String>` | no | default when absent |

### FactNarrativesResponse

Defined in `src/handlers/facts.rs`. Response carrying clustered fact narratives. `total_clusters` is the count returned; `total_facts` is the sum of `facts.len()` across all clusters (a hint at the candidate-set size, not the user's full fact corpus).

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `clusters` | Vec&lt;[`FactCluster`](#factcluster)&gt; | yes |  |
| `total_facts` | `usize` | yes |  |
| `total_clusters` | `usize` | yes |  |

### FactStats

Defined in `src/memory/facts.rs`. Statistics about semantic facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_facts` | `usize` | yes |  |
| `by_type` | `std::collections::HashMap<String, usize>` | yes |  |
| `avg_confidence` | `f32` | yes | score |
| `avg_support` | `f32` | yes |  |

### FactType

Defined in `src/memory/compression.rs`. Types of semantic facts

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Preference` | none | User preference: "prefers concise code" |
| `Capability` | none | System capability: "can handle 10k requests/sec" |
| `Relationship` | none | Relationship: "auth module depends on JWT library" |
| `Procedure` | none | Procedure: "to deploy, run cargo build --release" |
| `Definition` | none | Definition: "MemoryId is a UUID wrapper" |
| `Pattern` | none | Pattern: "errors often occur after deployment" |

### FactsByEntityRequest

Defined in `src/handlers/facts.rs`. Request for facts by entity

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### FactsListRequest

Defined in `src/handlers/facts.rs`. Request for listing facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### FactsPreviewPurgeBucket

Defined in `src/handlers/facts.rs`. Bucketed match count returned by `facts_preview_purge`. Exact counts are withheld to prevent the preview from becoming an oracle for fact existence (breakers ORACLE-AS-DOS / RING_1_7.01-02): repeated probes against tight patterns could otherwise enumerate the user's fact corpus distribution.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `none` | none |  |
| `few` | none |  |
| `some` | none |  |
| `many` | none |  |

### FactsPreviewPurgeRequest

Defined in `src/handlers/facts.rs`. Request for `POST /api/facts/preview-purge`. Note the deliberate absence of a `dry_run` field — this endpoint is preview-only by name and by schema. Agents cannot escalate to destructive purge via this surface; the actual delete lives behind a separate route landed in Phase C.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `pattern` | `String` | yes |  |

### FactsPreviewPurgeResponse

Defined in `src/handlers/facts.rs`. Response for `POST /api/facts/preview-purge`. Match count is bucketed (not exact), and `fact_ids` are NOT returned — Phase B is intentionally information-stingy. Agents that need to see fact content should call `recall` or `fact_narratives`; the preview only answers "is your pattern safe to actually purge".

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `match_bucket` | [`FactsPreviewPurgeBucket`](#factspreviewpurgebucket) | yes |  |
| `total_scanned` | `usize` | yes | Exact count of facts scanned (NOT the match count). Bounded by the preview corpus cap (default 10_000) — values at the cap indicate the user's corpus may have been truncated and the bucket is a lower bound. |
| `dry_run` | `bool` | yes | Always `true` — documents the preview-only contract in the response body for clients that key off the field. |
| `audit_recorded` | `bool` | yes | `true` when the audit-log entry was successfully enqueued (async write; failures here do not block the response). |

### FactsPurgeRequest

Defined in `src/handlers/facts.rs`. Request for `POST /api/facts/purge`. The destructive route — distinct from `/api/facts/preview-purge` so that a misconfigured client cannot escalate to deletion via field-flipping on a single endpoint.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `pattern` | `String` | yes |  |
| `dry_run` | `bool` | no | default when absent |

### FactsPurgeResponse

Defined in `src/handlers/facts.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `dry_run` | `bool` | yes |  |
| `purged` | `usize` | yes | Exact count of matching facts (when dry_run) or the count of facts actually soft-purged (when !dry_run). Exposes precise numbers here because the route requires the same auth as `preview-purge` and is already implicitly destructive — the oracle concern (bucketed counts on the preview surface) does not apply to the explicit operator path. |
| `total_scanned` | `usize` | yes |  |
| `context_blocks_flagged` | `usize` | yes | Number of context blocks flagged DIRTY because their content contained a UUID-shape substring matching a purged fact id. Advisory only — sleep-time V2 reconciles. Always `0` for dry runs. |
| `audit_recorded` | `bool` | yes |  |

### FactsResponse

Defined in `src/handlers/facts.rs`. Response containing facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `facts` | Vec&lt;[`SemanticFact`](#semanticfact)&gt; | yes |  |
| `total` | `usize` | yes |  |

### FactsSearchRequest

Defined in `src/handlers/facts.rs`. Request for searching facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### FeedbackProcessed

Defined in `src/handlers/recall.rs`. Feedback processing results

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_evaluated` | `usize` | yes |  |
| `reinforced` | `Vec<String>` | yes |  |
| `weakened` | `Vec<String>` | yes |  |

### Field

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | [`FieldState`](#fieldstate) | yes |  |
| `value` | `Option<T>` | no | omitted when empty |
| `reason` | `Option<String>` | no | omitted when empty |
| `witness` | `&'static str` | yes |  |

### FieldState

Defined in `src/handlers/ui.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `measured` | none |  |
| `unmeasured` | none |  |
| `undetermined` | none |  |

### FileAnswerRequest

Defined in `src/handlers/remember.rs`. Request to file a synthesised answer as a derived memory (A5).

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `cited_sources` | `Vec<String>` | no | default when absent; Memory ids that fed the synthesis. Recorded as `src:<id>` tags. |
| `memory_type` | `Option<String>` | no | default when absent; Optional experience type (default `Learning`). |

### FileAnswerResponse

Defined in `src/handlers/remember.rs`. Reply of `POST /api/file_answer`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `id` | `String` | yes | The memory written. |
| `is_derived` | `bool` | yes | Always true: an answer is a derived memory. |
| `cited_sources` | `Vec<String>` | yes |  |

### FileListResponse

Defined in `src/handlers/files.rs`. Response for file list operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `files` | Vec&lt;[`FileMemorySummary`](#filememorysummary)&gt; | yes |  |
| `total` | `usize` | yes |  |

### FileMemoryStats

Defined in `src/memory/files.rs`. Statistics about file memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_files` | `usize` | yes |  |
| `total_size_bytes` | `u64` | yes |  |
| `total_lines` | `usize` | yes |  |
| `total_accesses` | `u32` | yes |  |
| `by_type` | `HashMap<String, usize>` | yes |  |
| `by_source` | `HashMap<String, usize>` | yes |  |

### FileMemorySummary

Defined in `src/handlers/files.rs`. Summary of a file memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `path` | `String` | yes |  |
| `absolute_path` | `String` | yes |  |
| `file_type` | `String` | yes |  |
| `summary` | `String` | yes |  |
| `key_items` | `Vec<String>` | yes |  |
| `access_count` | `u32` | yes |  |
| `last_accessed` | `String` | yes |  |
| `heat_score` | `u8` | yes |  |
| `size_bytes` | `u64` | yes |  |
| `line_count` | `usize` | yes |  |

### FileStatsResponse

Defined in `src/handlers/files.rs`. Response for file stats

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `stats` | [`FileMemoryStats`](#filememorystats) | yes |  |

### FindEntityRequest

Defined in `src/handlers/graph.rs`. Request to find an entity

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity_name` | `String` | yes |  |

### FlatTodoRequest

Defined in `src/handlers/todos.rs`. Body of the flat `/api/todos/complete` and `/api/todos/delete` aliases, which have no path capture and name their target here.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `todo_id` | `String` | yes |  |

### ForgetByAgeRequest

Defined in `src/handlers/crud.rs`. Forget memories by age

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `days_old` | `u32` | yes |  |

### ForgetByCriteriaResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/forget/age` and `POST /api/forget/importance`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `forgotten_count` | `usize` | yes |  |
| `criteria` | `String` | yes | The criterion applied, in words. |

### ForgetByDateRequest

Defined in `src/handlers/crud.rs`. Forget memories by date range

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `start` | `chrono::DateTime<chrono::Utc>` | yes | Start of date range (inclusive) - ISO 8601 format |
| `end` | `chrono::DateTime<chrono::Utc>` | yes | End of date range (inclusive) - ISO 8601 format |

### ForgetByDateResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/forget/date`. `start` and `end` are the window applied, RFC 3339.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted_count` | `usize` | yes |  |
| `start` | `String` | yes |  |
| `end` | `String` | yes |  |

### ForgetByIdRequest

Defined in `src/handlers/crud.rs`. Forget a single memory by ID (POST body variant)

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memory_id` | `String` | yes |  |

### ForgetByImportanceRequest

Defined in `src/handlers/crud.rs`. Forget memories by importance threshold

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `threshold` | `f32` | yes |  |

### ForgetByPatternRequest

Defined in `src/handlers/crud.rs`. Forget memories matching a pattern

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `pattern` | `String` | yes |  |

### ForgetByPatternResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/forget/pattern`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `forgotten_count` | `usize` | yes |  |
| `pattern` | `String` | yes |  |

### ForgetByTagsRequest

Defined in `src/handlers/crud.rs`. Forget memories by tags

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `tags` | `Vec<String>` | yes | Tags to match for deletion (deletes memories matching ANY of these tags) |

### ForgetByTagsResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/forget/tags`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `deleted_count` | `usize` | yes |  |
| `tags` | `Vec<String>` | yes |  |

### Frequency

Defined in `src/entity_frequency.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `since` | `DateTime<Utc>` | yes |  |
| `until` | `DateTime<Utc>` | yes |  |
| `bucket` | [`Bucket`](#bucket) | yes |  |
| `witness` | `&'static str` | yes |  |
| `episodes_total` | `usize` | yes |  |
| `episodes_in_window` | `usize` | yes |  |
| `episodes_undecodable` | `usize` | yes |  |
| `refs_unresolved` | `usize` | yes |  |
| `entities_in_window` | `usize` | yes |  |
| `buckets` | Vec&lt;[`BucketCounts`](#bucketcounts)&gt; | yes |  |
| `totals` | `BTreeMap<String, usize>` | yes |  |
| `undetermined` | Option&lt;[`Undetermined`](#undetermined)&gt; | no | omitted when empty |

### GapAnalysisRequest

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `min_edge_strength` | `f32` | no | default when absent |
| `max_gaps_per_type` | `usize` | no | default when absent |
| `offset` | `usize` | no | default when absent |
| `limit` | `usize` | no | default when absent |
| `min_embedding_similarity` | `Option<f32>` | no | default when absent; score; Minimum embedding similarity to flag a U-shape (default 0.3) |
| `star_min_spokes` | `Option<usize>` | no | default when absent; Minimum spokes for star gap detection (default 3) |
| `orbit_min_cluster_size` | `Option<usize>` | no | default when absent; Minimum cluster size for orbit detection (default 3) |
| `namespace` | `Option<String>` | no | default when absent; Only entities named `{namespace}:…`. |
| `project` | `Option<String>` | no | default when absent; Only entities linked to the tag entity `project:{project}`. |
| `entity_types` | `Vec<String>` | no | default when absent; Only entities carrying one of these labels. |
| `max_nodes` | `Option<usize>` | no | default when absent; Analyze at most this many entities, the most salient in scope. |

### GapAnalysisResponse

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `gaps` | Vec&lt;[`GapSummary`](#gapsummary)&gt; | yes |  |
| `total_count` | `usize` | yes |  |
| `type_counts` | `std::collections::HashMap<String, usize>` | yes |  |
| `duration_ms` | `u64` | yes | Mirror sync and detection of the run that produced the result. |
| `detect_ms` | `u64` | yes | Detection alone. |
| `status` | `&'static str` | yes | `fresh`: computed within the last five minutes. `stale`: an older result, served while a new run goes on (or after it failed). `running`: no result yet, the run goes on (status 202). |
| `computed_at` | `Option<String>` | no | When the result was computed, RFC 3339; absent while `running`. |
| `age_secs` | `Option<i64>` | no | Seconds since `computed_at`. |
| `refreshing` | `bool` | yes | A run for these parameters is under way. |
| `scope` | Option&lt;[`ScopeReport`](#scopereport)&gt; | no | The part of the graph the result covers; absent while `running`. |
| `mirror` | Option&lt;[`MirrorReport`](#mirrorreport)&gt; | no | How the run brought the graph mirror up to the graph; absent while `running`. |

### GapStatsRequest

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### GapStatsResponse

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `edge_count` | `usize` | yes |  |

### GapSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `gap_type` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `embedding_similarity` | `Option<f32>` | no | score |
| `impact_score` | `f32` | yes | score |
| `entity_names` | `Vec<String>` | yes |  |
| `shape` | [`ShapeSummary`](#shapesummary) | yes |  |

### GetAllEntitiesRequest

Defined in `src/handlers/graph.rs`. Request to get all entities

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `Option<usize>` | no |  |
| `shelf` | Option&lt;[`Shelf`](#shelf)&gt; | no | default when absent; Keep only entities whose lexical shelf matches, applied before `limit`. |
| `with_sources` | `bool` | no | default when absent; Join each entity's tag-source provenance (one scan of the tag_source store). |
| `include_embeddings` | `bool` | no | default when absent; `name_embedding` is 768 floats per row; a client that draws names does not need it. |

### GetDueRemindersRequest

Defined in `src/handlers/todos.rs`. Request to get due reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `mark_triggered` | `bool` | no | default when absent |

### GetEpisodeRequest

Defined in `src/handlers/graph.rs`. Request to get an episode

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `episode_uuid` | `String` | yes |  |

### GetSessionRequest

Defined in `src/handlers/sessions.rs`. Request for getting a specific session

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### GetSessionResponse

Defined in `src/handlers/sessions.rs`. Response for getting a session

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `session` | Option&lt;[`Session`](#session)&gt; | no |  |

### GetUncompressedRequest

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `days_old` | `u32` | yes |  |

### GitHubSyncRequest

Defined in `src/integrations/github.rs`. Request for bulk syncing GitHub issues/PRs/commits

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes | User ID to associate memories with |
| `token` | `String` | yes | GitHub personal access token |
| `owner` | `String` | yes | Repository owner (user or org) |
| `repo` | `String` | yes | Repository name |
| `sync_issues` | `bool` | no | default when absent; Sync issues (default: true) |
| `sync_prs` | `bool` | no | default when absent; Sync pull requests (default: true) |
| `sync_commits` | `bool` | no | default when absent; Sync commits (default: false) |
| `state` | `String` | no | default when absent; Only sync items with state (open, closed, all) - default: all |
| `limit` | `Option<usize>` | no | default when absent; Limit number of items to sync per type |
| `branch` | `Option<String>` | no | default when absent; Branch to sync commits from (default: default branch) |

### GitHubSyncResponse

Defined in `src/integrations/github.rs`. Response from bulk sync

| Field | Type | Required | Notes |
|---|---|---|---|
| `synced_count` | `usize` | yes | Total items synced |
| `issues_synced` | `usize` | yes | Issues synced |
| `prs_synced` | `usize` | yes | PRs synced |
| `commits_synced` | `usize` | yes | Commits synced |
| `created_count` | `usize` | yes | Items created (new) |
| `updated_count` | `usize` | yes | Items updated (existing) |
| `error_count` | `usize` | yes | Items that failed |
| `errors` | `Vec<String>` | yes | omitted when empty; Error messages if any |

### GoalRef

Defined in `src/memory/facets.rs`. A reference to a goal a record served.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Stable identifier — a Plan record id, or a free-form name until goals get a first-class record kind. |
| `label` | `Option<String>` | no | Optional human-readable label. |

### GraphCounts

Defined in `src/handlers/health.rs`. Node and edge counts of a tenant's graph.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `relationship_count` | `usize` | yes |  |
| `episode_count` | `usize` | yes |  |

### GraphDataResponse

Defined in `src/handlers/visualization.rs`. Graph data response for d3.js

| Field | Type | Required | Notes |
|---|---|---|---|
| `nodes` | Vec&lt;[`GraphNode`](#graphnode)&gt; | yes |  |
| `edges` | Vec&lt;[`GraphEdge`](#graphedge)&gt; | yes |  |
| `stats` | [`GraphDataStats`](#graphdatastats) | yes |  |

### GraphDataStats

Defined in `src/handlers/visualization.rs`. Graph statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_nodes` | `usize` | yes |  |
| `total_edges` | `usize` | yes |  |
| `l1_edges` | `usize` | yes |  |
| `l2_edges` | `usize` | yes |  |
| `l3_edges` | `usize` | yes |  |

### GraphEdge

Defined in `src/handlers/visualization.rs`. Graph edge for d3.js visualization

| Field | Type | Required | Notes |
|---|---|---|---|
| `source` | `String` | yes |  |
| `target` | `String` | yes |  |
| `edge_type` | `String` | yes |  |
| `tier` | `String` | yes |  |
| `strength` | `f32` | yes |  |
| `distinct_sources` | `usize` | yes | Distinct sources whose strengthening this edge rests on. |
| `last_corroborated` | `Option<chrono::DateTime<chrono::Utc>>` | no | When a source new to this edge last strengthened it. |

### GraphEntitiesResponse

Defined in `src/handlers/graph.rs`. Reply of `POST /api/graph/entities/all`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entities` | Vec&lt;[`GraphEntityRow`](#graphentityrow)&gt; | yes | At most `limit` entities on the requested shelf. |
| `count` | `usize` | yes | Entities returned. |
| `total` | `usize` | yes | Entities in the graph, every shelf. |
| `shelves` | `std::collections::BTreeMap<&'static str, usize>` | yes | Entities per shelf (`cast`, `vocabulary`, `noise`), every shelf. |
| `witness` | `&'static str` | yes |  |

### GraphEntityRow

Defined in `src/handlers/graph.rs`. One entity of a `POST /api/graph/entities/all` reply: the node's own fields, its hygiene classification, and its tag provenance when `with_sources` was set.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | [`EntityNode`](#entitynode) | yes | flattened into the parent |
| `hygiene` | [`Classification`](#classification) | yes |  |
| `provenance` | Option&lt;Option&lt;[`EntityProvenance`](#entityprovenance)&gt;&gt; | no | omitted when empty; Absent unless `with_sources`; then `null` for an entity no tag row names. |

### GraphNode

Defined in `src/handlers/visualization.rs`. Graph node for d3.js visualization

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `label` | `String` | yes |  |
| `node_type` | `String` | yes |  |
| `tier` | `String` | yes |  |
| `strength` | `f32` | yes |  |
| `size` | `f32` | yes |  |

### GraphPredicate

Defined in `src/query_planner/predicate.rs`. Graph reachability predicates against the knowledge graph.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `LinkedToEntity` | `{ entity_id: Uuid }` |  |
| `SharesEntity` | `{ other_memory_id: String }` |  |
| `EpisodeContains` | `{ episode_id: Uuid }` |  |

### GraphStats

Defined in `src/graph_memory.rs`. Graph statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `relationship_count` | `usize` | yes |  |
| `episode_count` | `usize` | yes |  |

### GraphTraversal

Defined in `src/graph_memory.rs`. Result of graph traversal

| Field | Type | Required | Notes |
|---|---|---|---|
| `entities` | Vec&lt;[`TraversedEntity`](#traversedentity)&gt; | yes | Entities found during traversal with hop distance info |
| `relationships` | Vec&lt;[`RelationshipEdge`](#relationshipedge)&gt; | yes |  |

### GraphViewParams

Defined in `src/handlers/visualization.rs`. Query parameters for graph view

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |

### GravitationalConnection

Defined in `src/graph_memory.rs`. A gravitational connection between stars (represents a relationship)

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `from_id` | `String` | yes |  |
| `to_id` | `String` | yes |  |
| `relation_type` | `String` | yes |  |
| `strength` | `f32` | yes |  |
| `from_position` | [`Position3D`](#position3d) | yes |  |
| `to_position` | [`Position3D`](#position3d) | yes |  |

### HealthResponse

Defined in `src/handlers/health.rs`. Health response for main health endpoint

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `String` | yes |  |
| `version` | `String` | yes |  |
| `build` | `String` | yes |  |
| `built_at` | `String` | yes |  |
| `requested_storage_backend` | `String` | yes |  |
| `effective_storage_backend` | `String` | yes |  |
| `users_count` | `usize` | yes |  |
| `users_in_cache` | `usize` | yes |  |
| `user_evictions` | `usize` | yes |  |
| `max_cache_size` | `usize` | yes |  |
| `embedder` | Option&lt;crate::embeddings::[`PrimaryEmbedderInfo`](#primaryembedderinfo)&gt; | no | omitted when empty; The primary embedder this process chose (id, model, dimension, kind, reason). `null` until the first user's memory system has been constructed. Measured 2026-09-10: the running server was on hash-bigram vectors and nothing on this endpoint said so. |
| `tenants_with_data` | crate::storage::boot_guard::[`BackendCensus`](#backendcensus) | yes | Tenants holding data on each backend at boot. `users_count` over an empty store is the 2026-09-10 outage; this is what tells the two apart. |
| `write_latch` | Vec&lt;crate::storage::write_latch::[`LatchSnapshot`](#latchsnapshot)&gt; | yes | omitted when empty; One entry per store currently refusing writes; absent when none is. Measured 2026-10-11: this endpoint served `healthy` through 11,088 consecutive failed writes. |

### HiddenCount

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `tags` | `usize` | yes |  |
| `occurrences` | `usize` | yes |  |

### ImportResult

Defined in `src/mif/import.rs`. Result of an import operation.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_imported` | `usize` | yes |  |
| `todos_imported` | `usize` | yes |  |
| `projects_imported` | `usize` | yes |  |
| `reminders_imported` | `usize` | yes |  |
| `edges_imported` | `usize` | yes |  |
| `entities_imported` | `usize` | yes |  |
| `duplicates_skipped` | `usize` | yes |  |
| `errors` | `Vec<String>` | yes |  |

### IndexCodebaseRequest

Defined in `src/handlers/files.rs`. Request to scan/index a codebase

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `codebase_path` | `String` | yes |  |
| `force` | `bool` | no | default when absent |

### IndexCoverage

Defined in `src/handlers/health.rs`. Populations, not error counts: a memory stored and never indexed is unretrievable and raises nothing for a counter to see.

| Field | Type | Required | Notes |
|---|---|---|---|
| `stored` | `usize` | yes |  |
| `vector_indexed` | `usize` | yes |  |
| `lexically_indexed` | `usize` | yes |  |
| `shortfall` | `bool` | yes | True when either index holds fewer memories than the store. |

### IndexHealthResponse

Defined in `src/handlers/health.rs`. Reply of `GET /health/index`: vector-index health summed over the cached tenants.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `&'static str` | yes | Always `ok`. |
| `users_checked` | `usize` | yes |  |
| `total_vectors` | `usize` | yes |  |
| `total_incremental_inserts` | `usize` | yes |  |
| `users_needing_rebuild_count` | `usize` | yes |  |
| `users_with_index_shortfall_count` | `usize` | yes | Tenants with memories stored but not in the vector or lexical index. |
| `users_coverage_unreadable_count` | `usize` | yes |  |
| `rebuild_threshold` | `usize` | yes |  |
| `timestamp` | `String` | yes | RFC 3339. |

### IndexIntegrityReport

Defined in `src/memory/types.rs`. Report from index integrity verification

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_storage` | `usize` | yes | Total memories in RocksDB storage |
| `total_indexed` | `usize` | yes | Total memories in vector index |
| `orphaned_count` | `usize` | yes | Number of orphaned memories (storage - indexed) |
| `orphaned_ids` | Vec&lt;[`MemoryId`](#memoryid)&gt; | yes | First 100 orphaned memory IDs for debugging |
| `is_healthy` | `bool` | yes | Whether the index is healthy (no orphans) |

### IndexResponse

Defined in `src/handlers/files.rs`. Response for index operation

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `result` | [`IndexingResult`](#indexingresult) | yes |  |
| `message` | `String` | yes |  |

### IndexingResult

Defined in `src/memory/files.rs`. Result of indexing a codebase

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_files` | `usize` | yes | Total files attempted |
| `indexed_files` | `usize` | yes | Files successfully indexed |
| `skipped_files` | `usize` | yes | Files skipped due to errors |
| `errors` | `Vec<String>` | yes | Error messages for failed files |

### IngestMetadata

Defined in `src/handlers/ingest.rs`. Extraction metadata returned alongside the stored memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | `Option<String>` | no | omitted when empty |
| `headings` | `Vec<String>` | yes |  |
| `entities_hint` | `Vec<String>` | yes |  |
| `line_count` | `usize` | yes |  |
| `word_count` | `usize` | yes |  |

### IngestRequest

Defined in `src/handlers/ingest.rs`. Ingest request - auto-detect format, extract text, store as memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `filename` | `Option<String>` | no | default when absent; Optional filename for format detection (e.g. "notes.md", "data.csv") |
| `tags` | `Vec<String>` | no | default when absent |
| `importance` | `Option<f32>` | no | default when absent; score |
| `memory_type` | `Option<String>` | no | default when absent |
| `tabular` | `bool` | no | default when absent; W7 tabular mode: when `true` and the content is CSV or JSON, store the input as a relational dataset (one row per record) instead of a single free-text memory. Requires a configured relational backend. Default `false` preserves the memory-storing behaviour. |
| `dataset_name` | `Option<String>` | no | default when absent; Optional dataset name for tabular mode. Defaults to the filename stem, or `ingest_dataset` when no filename is given. |

### IngestResponse

Defined in `src/handlers/ingest.rs`. Ingest response - remember response plus extraction metadata

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `success` | `bool` | yes |  |
| `format_detected` | `String` | yes |  |
| `metadata` | [`IngestMetadata`](#ingestmetadata) | yes |  |
| `dataset` | `Option<String>` | no | omitted when empty; Tabular mode only: the dataset that was created. |
| `rows_inserted` | `Option<u64>` | no | omitted when empty; Tabular mode only: how many rows were inserted. |

### InitBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `passphrase` | `String` | yes |  |
| `label` | `Option<String>` | no |  |

### InsertRowsRequest

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `rows` | Vec&lt;[`DatasetRow`](#datasetrow)&gt; | yes |  |

### InsertRowsResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `inserted` | `u64` | yes |  |

### IntentLogFileMeta

Defined in `src/backup.rs`. One file copied into the backup under `intent_log/`. Carries enough information to (a) verify the file survived round-trip via SHA-256, and (b) decide on PITR whether the LSN the operator asked for is in range. The `lsn_range` is `None` for non-journal files (e.g. the checkpoint store) since those don't carry LSNs in their own framing.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes | File name as it appears in the source directory (and as it is re-created in the restore-target directory). |
| `sha256` | `String` | yes | SHA-256 of the file contents, lower-case hex. |
| `size_bytes` | `u64` | yes | Raw byte length of the file at backup time. |
| `lsn_range` | `Option<(u64, u64)>` | no | default when absent; `Some((lo, hi))` for intent-log files where `lo` is the first LSN and `hi` is the last LSN present, both inclusive. `None` for auxiliary files (checkpoint store, future tombstone manifests). |

### InterferenceEvent

Defined in `src/memory/types.rs`. Interference event reported in retrieval stats (FIX-03)

| Field | Type | Required | Notes |
|---|---|---|---|
| `affected_memory_id` | `String` | yes | Memory ID that was affected |
| `interfering_memory_id` | `String` | yes | Memory ID that caused the interference |
| `similarity` | `f32` | yes | score; Cosine similarity between the two memories |
| `decay_applied` | `f32` | yes | Amount of importance decay applied |
| `interference_type` | `String` | yes | Type: "retroactive" or "proactive" |

### InterferenceType

Defined in `src/memory/introspection.rs`. Types of memory interference (SHO-106)

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `retroactive` | none | New learning disrupts old memories |
| `proactive` | none | Old memories interfere with new learning |
| `retrieval_competition` | none | Similar memories compete during retrieval |

### InvalidateRelationshipRequest

Defined in `src/handlers/graph.rs`. Request to invalidate a relationship

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `relationship_uuid` | `String` | yes |  |

### InvalidateRelationshipResponse

Defined in `src/handlers/graph.rs`. Reply of `POST /api/graph/relationship/invalidate`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `String` | yes |  |

### IsolatedEntity

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `isolation` | `f32` | yes |  |
| `anisotropy` | `f32` | yes |  |

### IssueBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `unlock` | [`UnlockBody`](#unlockbody) | yes |  |
| `label` | `Option<String>` | no |  |

### Knows

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `settled_facts` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`FactCard`](#factcard)&gt;&gt; | yes |  |
| `changed_its_mind` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`FactCard`](#factcard)&gt;&gt; | yes |  |
| `contradictions` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`FactCard`](#factcard)&gt;&gt; | yes |  |
| `rulings` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `scars` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `who_and_what` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`EntityCard`](#entitycard)&gt;&gt; | yes |  |
| `strongest_associations` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`AssociationCard`](#associationcard)&gt;&gt; | yes |  |
| `about_to_forget` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `pinned` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `known_but_never_asked` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `most_asked` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `asked_but_not_known` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MissedQuery`](#missedquery)&gt;&gt; | yes |  |
| `suspected_unknowns` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;String&gt;&gt; | yes |  |
| `this_week` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`CountCard`](#countcard)&gt;&gt; | yes |  |
| `where_learned` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`CountCard`](#countcard)&gt;&gt; | yes |  |
| `sense_of_self` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`MemoryCard`](#memorycard)&gt;&gt; | yes |  |
| `undetermined` | [`Field`](#field)&lt;[`UiShelf`](#uishelf)&lt;[`CountCard`](#countcard)&gt;&gt; | yes |  |

### KnowsQuery

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |

### LatchSnapshot

Defined in `src/storage/write_latch.rs`. What `/health` publishes about one latched store.

| Field | Type | Required | Notes |
|---|---|---|---|
| `store` | `String` | yes | Truncated SHA-256 of the store path. See [`store_id`]. |
| `condition` | `&'static str` | yes | Always [`LATCH_CONDITION`]; named so the body says what is wrong. |
| `since` | `DateTime<Utc>` | yes |  |
| `failed_writes` | `u64` | yes |  |
| `recovery_attempts` | `u32` | yes |  |
| `hard` | `bool` | yes | A flush+retry has already failed; only a restart clears it. |

### LearnedSignalWeights

Defined in `src/relevance.rs`. Learned weights for score fusion, adapted via feedback

| Field | Type | Required | Notes |
|---|---|---|---|
| `semantic` | `f32` | yes | Weight for semantic similarity score |
| `entity` | `f32` | yes | Weight for entity matching score |
| `tag` | `f32` | yes | Weight for tag matching score |
| `importance` | `f32` | yes | score; Weight for importance score |
| `momentum` | `f32` | no | default when absent; Weight for feedback momentum EMA (FBK-5) Higher momentum = memory has been consistently helpful Lower/negative momentum = memory has been consistently misleading |
| `access_count` | `f32` | no | default when absent; Weight for access count (CTX-3) Higher access count = memory has proven retrieval value |
| `graph_strength` | `f32` | no | default when absent; Weight for graph Hebbian strength (CTX-3) Higher strength = stronger entity relationships in knowledge graph |
| `update_count` | `u32` | yes | Number of feedback updates applied |
| `last_updated` | `Option<DateTime<Utc>>` | no | Last time weights were updated |

### LineageAddEdgeRequest

Defined in `src/handlers/lineage.rs`. Request to add an explicit lineage edge

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `from_memory_id` | `String` | yes |  |
| `to_memory_id` | `String` | yes |  |
| `relation` | `String` | yes |  |

### LineageBranch

Defined in `src/memory/lineage.rs`. A branch in the lineage graph (for project pivots)

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Unique branch identifier |
| `name` | `String` | yes | Human-readable branch name |
| `description` | `Option<String>` | no | Description of what this branch represents |
| `parent_branch` | `Option<String>` | no | Parent branch (None for main branch) |
| `branch_point` | Option&lt;[`MemoryId`](#memoryid)&gt; | no | Memory where this branch diverged from parent |
| `created_at` | `DateTime<Utc>` | yes | When the branch was created |
| `active` | `bool` | yes | Whether this branch is currently active |
| `tags` | `Vec<String>` | yes | Tags for categorization |

### LineageBranchesResponse

Defined in `src/handlers/lineage.rs`. Response for branches list

| Field | Type | Required | Notes |
|---|---|---|---|
| `branches` | Vec&lt;[`LineageBranch`](#lineagebranch)&gt; | yes |  |
| `total` | `usize` | yes |  |

### LineageConfirmResponse

Defined in `src/handlers/lineage.rs`. Reply of `POST /api/lineage/confirm`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `confirmed` | `bool` | yes | False when no inferred edge with that id existed. |

### LineageCreateBranchRequest

Defined in `src/handlers/lineage.rs`. Request to create a branch

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `parent_branch` | `String` | yes |  |
| `branch_point_memory_id` | `String` | yes |  |
| `description` | `Option<String>` | no |  |

### LineageEdge

Defined in `src/memory/lineage.rs`. A causal edge between two memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Unique edge identifier |
| `from` | [`MemoryId`](#memoryid) | yes | Source memory (cause) |
| `to` | [`MemoryId`](#memoryid) | yes | Target memory (effect) |
| `relation` | [`CausalRelation`](#causalrelation) | yes | Type of causal relationship |
| `confidence` | `f32` | yes | score; Confidence in this causal link (0.0-1.0) |
| `source` | [`LineageSource`](#lineagesource) | yes | How this edge was created |
| `branch_id` | `Option<String>` | no | Branch this edge belongs to (None = main branch) |
| `created_at` | `DateTime<Utc>` | yes | When the edge was created |
| `last_reinforced` | `DateTime<Utc>` | yes | Last time this edge was reinforced/confirmed |
| `reinforcement_count` | `u32` | yes | Number of times this edge was reinforced |

### LineageEdgeRequest

Defined in `src/handlers/lineage.rs`. Request to confirm/reject an edge

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `edge_id` | `String` | yes |  |

### LineageEdgesResponse

Defined in `src/handlers/lineage.rs`. Response for lineage edges list

| Field | Type | Required | Notes |
|---|---|---|---|
| `edges` | Vec&lt;[`LineageEdge`](#lineageedge)&gt; | yes |  |
| `total` | `usize` | yes |  |

### LineageListRequest

Defined in `src/handlers/lineage.rs`. Request to list lineage edges

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### LineageRejectResponse

Defined in `src/handlers/lineage.rs`. Reply of `POST /api/lineage/reject`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `rejected` | `bool` | yes | False when no inferred edge with that id existed. |

### LineageSource

Defined in `src/memory/lineage.rs`. Source of a lineage edge

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Inferred` | none | Automatically inferred by the system |
| `Confirmed` | none | Explicitly confirmed by user/agent |
| `Explicit` | none | Manually added by user/agent |

### LineageStats

Defined in `src/memory/lineage.rs`. Statistics about the lineage graph

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_edges` | `usize` | yes |  |
| `inferred_edges` | `usize` | yes |  |
| `confirmed_edges` | `usize` | yes |  |
| `explicit_edges` | `usize` | yes |  |
| `total_branches` | `usize` | yes |  |
| `active_branches` | `usize` | yes |  |
| `edges_by_relation` | `HashMap<String, usize>` | yes |  |
| `avg_confidence` | `f32` | yes | score |

### LineageTraceRequest

Defined in `src/handlers/lineage.rs`. Request to trace lineage from a memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memory_id` | `String` | yes |  |
| `direction` | `String` | no | default when absent |
| `max_depth` | `usize` | no | default when absent |

### LineageTraceResponse

Defined in `src/handlers/lineage.rs`. Response for lineage trace

| Field | Type | Required | Notes |
|---|---|---|---|
| `root` | `String` | yes |  |
| `direction` | `String` | yes |  |
| `edges` | Vec&lt;[`LineageEdge`](#lineageedge)&gt; | yes |  |
| `path` | `Vec<String>` | yes |  |
| `depth` | `usize` | yes |  |

### LinearSyncRequest

Defined in `src/integrations/linear.rs`. Request for bulk syncing Linear issues

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes | User ID to associate memories with |
| `api_key` | `String` | yes | Linear API key |
| `team_id` | `Option<String>` | no | default when absent; Optional team ID to filter issues |
| `updated_after` | `Option<String>` | no | default when absent; Optional: only sync issues updated after this date (ISO 8601) |
| `limit` | `Option<usize>` | no | default when absent; Optional: limit number of issues to sync |

### LinearSyncResponse

Defined in `src/integrations/linear.rs`. Response from bulk sync

| Field | Type | Required | Notes |
|---|---|---|---|
| `synced_count` | `usize` | yes | Number of issues synced |
| `created_count` | `usize` | yes | Number of issues created (new) |
| `updated_count` | `usize` | yes | Number of issues updated (existing) |
| `error_count` | `usize` | yes | Number of issues that failed |
| `errors` | `Vec<String>` | yes | omitted when empty; Error messages if any |

### LinkKindWire

Defined in `src/handlers/datasets.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `entity` | none |  |
| `memory` | none |  |

### LinkRequest

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `row_pk` | [`RowPk`](#rowpk) | yes |  |
| `kind` | [`LinkKindWire`](#linkkindwire) | yes |  |
| `target_id` | `String` | yes |  |

### LinkResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `linked` | `bool` | yes |  |

### ListBackupsRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### ListBackupsResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `backups` | Vec&lt;crate::backup::[`BackupMetadata`](#backupmetadata)&gt; | yes |  |
| `count` | `usize` | yes |  |

### ListDatasetsResponse

Defined in `src/handlers/datasets.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `datasets` | Vec&lt;[`DatasetMeta`](#datasetmeta)&gt; | yes |  |
| `total` | `usize` | yes |  |

### ListFilesRequest

Defined in `src/handlers/files.rs`. Request for listing files

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `Option<usize>` | no | default when absent |

### ListMemoriesQuery

Defined in `src/handlers/crud.rs`. Query parameters for GET /api/memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `Option<usize>` | no |  |
| `offset` | `Option<usize>` | no |  |
| `type` | `Option<String>` | no |  |
| `query` | `Option<String>` | no |  |
| `tags` | `Option<String>` | no | Comma-separated tags; every one must be present on the memory, matched exactly. See [`parse_tag_filter`]. |

### ListMemoriesRequest

Defined in `src/handlers/crud.rs`. Request for POST /api/memories - list memories with user_id in body

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `Option<usize>` | no | Max items to return. Defaults to 100, clamped to [`MAX_LIST_LIMIT`]. |
| `offset` | `Option<usize>` | no | Items to skip before collecting `limit` results. Defaults to 0. |
| `type` | `Option<String>` | no |  |
| `query` | `Option<String>` | no |  |
| `tags` | `Option<String>` | no | Comma-separated tags; every one must be present on the memory, matched exactly. See [`parse_tag_filter`]. |

### ListMemoryItem

Defined in `src/handlers/crud.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes | Content preview, cut at [`LIST_CONTENT_PREVIEW_CHARS`]. Check `content_truncated` before treating it as the record; GET /api/memory/{id} returns the whole text. |
| `content_truncated` | `bool` | yes | True when `content` is shorter than the stored content. |
| `content_length` | `usize` | yes | Character length of the full stored content. |
| `memory_type` | `String` | yes |  |
| `importance` | `f32` | yes | score |
| `tags` | `Vec<String>` | yes |  |
| `created_at` | `String` | yes |  |
| `tier` | `String` | yes |  |

### ListProjectsRequest

Defined in `src/handlers/todos.rs`. Request to list projects

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### ListQuery

Defined in `src/handlers/crud.rs`. Query parameters for listing memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `limit` | `Option<usize>` | no | Max items to return. Defaults to 100, clamped to [`MAX_LIST_LIMIT`]. |
| `offset` | `Option<usize>` | no | Items to skip before collecting `limit` results. Defaults to 0. Combine with `total` in the response to page past `MAX_LIST_LIMIT`. |
| `type` | `Option<String>` | no |  |
| `query` | `Option<String>` | no | Text search query - filters by content or tags (case-insensitive) |
| `tags` | `Option<String>` | no | Comma-separated tags; every one must be present on the memory, matched exactly. See [`parse_tag_filter`]. |

### ListRemindersRequest

Defined in `src/handlers/todos.rs`. Request to list reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `status` | `Option<String>` | no |  |

### ListRemindersResponse

Defined in `src/handlers/todos.rs`. Response for listing reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `reminders` | Vec&lt;[`ReminderItem`](#reminderitem)&gt; | yes |  |
| `count` | `usize` | yes |  |

### ListResponse

Defined in `src/handlers/crud.rs`. List response - simplified memory list

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | Vec&lt;[`ListMemoryItem`](#listmemoryitem)&gt; | yes |  |
| `total` | `usize` | yes | Total matching the filters, independent of `limit`/`offset`: compare `memories.len() + offset` against it to know whether pages remain. |

### ListSessionsRequest

Defined in `src/handlers/sessions.rs`. Request for listing sessions

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### ListSessionsResponse

Defined in `src/handlers/sessions.rs`. Response for listing sessions

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `sessions` | Vec&lt;[`SessionSummary`](#sessionsummary)&gt; | yes |  |
| `count` | `usize` | yes |  |

### ListTodosRequest

Defined in `src/handlers/todos.rs`. Request to list todos with filters

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `status` | `Option<Vec<String>>` | no | default when absent |
| `project` | `Option<String>` | no | default when absent |
| `context` | `Option<String>` | no | default when absent |
| `include_completed` | `Option<bool>` | no | default when absent |
| `due` | `Option<String>` | no | default when absent |
| `limit` | `Option<usize>` | no | default when absent |
| `offset` | `Option<usize>` | no | default when absent |
| `parent_id` | `Option<String>` | no | default when absent |
| `query` | `Option<String>` | no | default when absent |
| `priority` | `Option<String>` | no | default when absent |

### LiveResponse

Defined in `src/handlers/health.rs`. Reply of `GET /health/live`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `&'static str` | yes | Always `alive`. |
| `timestamp` | `String` | yes | RFC 3339. |

### LockBlockRequest

Defined in `src/handlers/sleep_time.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `block_key` | `String` | yes |  |
| `locked` | `bool` | yes |  |

### LoginRequest

Defined in `src/handlers/user_auth.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |
| `password` | `String` | yes |  |
| `totp` | `Option<String>` | no | default when absent |

### LtpStatus

Defined in `src/graph_memory.rs`. Long-Term Potentiation status for edges (PIPE-4)

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `None` | none | Not potentiated - normal decay applies |
| `Burst` | `{ detected_at: DateTime<Utc> }` | Burst potentiated: 5+ activations in 24 hours Temporary protection (2x slower decay) that expires after 48h Represents early-phase LTP (protein synthesis independent) |
| `Weekly` | none | Weekly potentiated: 3+/week for 2+ weeks Moderate protection (3x slower decay) Represents late-phase LTP (habit formation) |
| `Full` | none | Fully potentiated: 10+ activations OR 5+ over 30 days Maximum protection (10x slower decay) Represents systems consolidation (semantic memory) |

### MapperEdgeSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `from` | `usize` | yes |  |
| `to` | `usize` | yes |  |
| `weight` | `usize` | yes |  |

### MapperNodeSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `usize` | yes |  |
| `member_names` | `Vec<String>` | yes |  |
| `size` | `usize` | yes |  |
| `avg_filter_value` | `f32` | yes |  |

### MapperRequest

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `filter` | `String` | no | default when absent |
| `num_intervals` | `usize` | no | default when absent |
| `overlap` | `f32` | no | default when absent |
| `offset` | `usize` | no | default when absent |
| `limit` | `usize` | no | default when absent |

### MapperResponse

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_count` | `usize` | yes |  |
| `nodes` | Vec&lt;[`MapperNodeSummary`](#mappernodesummary)&gt; | yes |  |
| `edges` | Vec&lt;[`MapperEdgeSummary`](#mapperedgesummary)&gt; | yes |  |
| `num_components` | `usize` | yes |  |
| `num_loops` | `usize` | yes |  |
| `flare_count` | `usize` | yes |  |
| `branch_count` | `usize` | yes |  |
| `filter` | `String` | yes |  |
| `stats` | [`MapperStatsSummary`](#mapperstatssummary) | yes |  |

### MapperStatsSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `interval_count` | `usize` | yes |  |
| `cluster_count` | `usize` | yes |  |
| `edge_count` | `usize` | yes |  |
| `duration_ms` | `u64` | yes |  |

### MediaRef

Defined in `src/memory/types.rs`. Reference to attached media with metadata

| Field | Type | Required | Notes |
|---|---|---|---|
| `media_type` | [`MediaType`](#mediatype) | yes | Media type |
| `uri` | `String` | yes | External reference (file path, blob hash, or URL) Not the actual bytes - keeps Memory struct lean |
| `filename` | `Option<String>` | no | Original filename (for display) |
| `mime_type` | `Option<String>` | no | MIME type (e.g., "image/jpeg", "audio/wav") |
| `size_bytes` | `Option<u64>` | no | File size in bytes |
| `media_metadata` | `HashMap<String, String>` | no | default when absent; Media-specific metadata Images: {"width": "1920", "height": "1080", "format": "jpeg"} Audio: {"duration_ms": "30000", "sample_rate": "16000", "channels": "1"} Video: {"duration_ms": "60000", "width": "1920", "height": "1080", "fps": "30"} |
| `timestamp_ms` | `Option<u64>` | no | Timestamp within media (for video/audio segments) Allows referencing a specific moment, not just the whole file |
| `duration_ms` | `Option<u64>` | no | Duration of the segment (for audio/video clips) |

### MediaType

Defined in `src/memory/types.rs`. Media type for multimodal memories

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Image` | none | Image file (JPEG, PNG, WebP, etc.) |
| `Audio` | none | Audio file (WAV, MP3, FLAC, etc.) |
| `Video` | none | Video file (MP4, WebM, etc.) |
| `Document` | none | Document file (PDF with images/diagrams) |

### Memory

Defined in `src/memory/types.rs`. Stored memory with metadata

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`MemoryId`](#memoryid) | yes |  |
| `experience` | [`Experience`](#experience) | yes |  |
| `kind` | [`RecordKind`](#recordkind) | yes | What kind of record this is — memory / plan / prompt / learning (W3). One core record, several lifecycles; retrieval stays unified. Legacy and default: `RecordKind::Memory`. |
| `what` | [`WhatFacet`](#whatfacet) | yes | WHAT facet — content with explicit gist/verbatim separation (W3.2c). During the transition `experience.content` is still the primary content; `verbatim` / `gist` are filled in by encoding/consolidation. See `docs/neuroscience-5w-memory-design.md`. |
| `when` | [`WhenFacet`](#whenfacet) | yes | WHEN facet — encoding/event time + TCM drift vector (W3.2c). `encoded_at` mirrors `created_at` during the transition; `event_time` is the separately-encoded time the described event actually happened. |
| `metadata` | Arc&lt;parking_lot::Mutex&lt;[`MemoryMetadata`](#memorymetadata)&gt;&gt; | yes |  |
| `created_at` | `DateTime<Utc>` | yes |  |
| `compressed` | `bool` | yes |  |
| `persistent` | `bool` | yes | Forbids LOSSY compression of this record. Default TRUE: content survives unless a caller deliberately opts out. The inverse default is what destroyed a 19KB handoff — it fell through select_strategy to Hybrid, whose semantic pass replaces content with 50 words. Absent on an older serialized record means true (`default_true`), so the fail-safe direction applies to everything already in the store. |
| `tier` | [`MemoryTier`](#memorytier) | yes | Current tier in the memory hierarchy Memories flow: Working → Session → LongTerm → Archive |
| `entity_refs` | Vec&lt;[`EntityRef`](#entityref)&gt; | yes | Entity references - bidirectional links to GraphMemory Populated during record() via entity extraction Enables spreading activation without graph lookup |
| `last_retrieval_id` | `Option<Uuid>` | no | Retrieval tracking ID - set when memory is retrieved Used for Hebbian feedback loop (reinforce_recall) |
| `agent_id` | `Option<String>` | no |  |
| `run_id` | `Option<String>` | no |  |
| `actor_id` | `Option<String>` | no |  |
| `score` | `Option<f32>` | no |  |
| `external_id` | `Option<String>` | no | External identifier for linking to external systems Format: "{source}:{id}" e.g. "linear:SHO-39", "github:pr-123" When set, enables upsert semantics: same external_id = update existing memory |
| `version` | `u32` | yes | Version counter - incremented on each update (starts at 1) |
| `history` | Vec&lt;[`MemoryRevision`](#memoryrevision)&gt; | yes | Audit history - tracks all changes to this memory Only populated for memories with external_id (mutable memories) Empty for regular immutable memories |
| `related_todo_ids` | Vec&lt;[`TodoId`](#todoid)&gt; | yes | Related todo IDs for bidirectional linking with todo system Populated when a todo references this memory or vice versa |
| `parent_id` | Option&lt;[`MemoryId`](#memoryid)&gt; | no | Parent memory ID for hierarchical organization Enables tree structures: parent -> children Example: "71-research" -> "algebraic relationships" -> "21 × 27 ≡ -1" |
| `valid_until` | `Option<DateTime<Utc>>` | no | When set, this memory is considered expired/superseded after this timestamp. Set by conflict detection when a newer memory contradicts this one. Expired memories are filtered out early in the retrieval pipeline. |

### MemoryCard

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `head` | `String` | yes |  |
| `experience_type` | `String` | yes |  |
| `created_at` | `DateTime<Utc>` | yes |  |
| `importance` | `f32` | yes | score |
| `access_count` | `u32` | yes |  |
| `anchored` | `bool` | yes |  |
| `authored_tags` | `Option<Vec<String>>` | no | Tags whose source is caller or intent; `None` when the record has no tag-source row. |
| `estimated_days_to_decay` | `Option<f32>` | no | omitted when empty |
| `recalled` | `Option<usize>` | no | omitted when empty; Times this memory came back in a recall, counted from the recall event log. |

### MemoryChange

Defined in `src/memory/introspection.rs`. Change in a memory's state

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory_id` | `String` | yes |  |
| `content_preview` | `String` | yes |  |
| `activation_before` | `f32` | yes |  |
| `activation_after` | `f32` | yes |  |
| `change_reason` | `String` | yes |  |
| `at_risk` | `bool` | yes |  |
| `timestamp` | `DateTime<Utc>` | yes |  |

### MemoryHealthResponse

Defined in `src/handlers/crud.rs`. Memory health response — exposes internal state Claude can't normally see

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory_id` | `String` | yes |  |
| `importance` | `f32` | yes | score |
| `tier` | `String` | yes |  |
| `access_count` | `u32` | yes |  |
| `created_at` | `String` | yes |  |
| `last_accessed` | `Option<String>` | no |  |
| `feedback_momentum` | `Option<f32>` | no | Feedback momentum EMA (-1.0 to +1.0) |
| `feedback_signal_count` | `Option<u32>` | no | Number of feedback signals received |
| `edge_count` | `usize` | yes | Edge count from this memory's entities |
| `health_status` | `String` | yes | Health classification: "healthy", "at_risk", "degraded" |
| `estimated_days_to_decay` | `Option<f32>` | no | Estimated days until memory falls below retrieval threshold |

### MemoryId

Defined in `src/memory/types.rs`. Unique identifier for memories

On the wire this is its inner value, `Uuid`.

### MemoryMetadata

Defined in `src/memory/types.rs`. Mutable metadata for memory (interior mutability) Separated from immutable core data to enable zero-copy updates via Arc<Memory>

| Field | Type | Required | Notes |
|---|---|---|---|
| `importance` | `f32` | yes | score |
| `access_count` | `u32` | yes |  |
| `last_accessed` | `DateTime<Utc>` | yes |  |
| `temporal_relevance` | `f32` | yes |  |
| `activation` | `f32` | yes | Activation level (spreading activation algorithm) Decays over time, boosted by access and co-activation Range: 0.0 (dormant) to 1.0 (highly active) |
| `anchored` | `bool` | no | default when absent; Anchored memories resist decay — importance cannot fall below ANCHOR_IMPORTANCE_FLOOR. Set via the `/api/anchor` endpoint or `anchor` MCP tool. Anchored memories also skip compression. |
| `access_history` | `Vec<DateTime<Utc>>` | no | default when absent; Per-memory access timestamp history for temporal autocorrelation analysis. Capped at MAX_ACCESS_HISTORY entries (oldest evicted first). Enables rhythm detection: bursty vs steady access patterns per memory. |
| `elaboration_score` | `f32` | no | default when absent; score; Elaboration quality score (0.0 = bare S-rep content, 1.0 = fully contextualized C-rep). Computed at encoding time based on RichContext field count, entity diversity, temporal specificity, emotional signal presence, and content density. Reference: Ehlers & Clark (2000) — S-rep vs C-rep distinction. |
| `fragment_demotion` | `f32` | no | default when absent; Fragment demotion factor (1.0 = no demotion, FRAGMENT_DEMOTION_FLOOR = heavily demoted). Applied when a consolidated fact exists for this source fragment. Reference: Berntsen (2021) — functional constraints on involuntary memory. |
| `calibrated_confidence` | `f32` | no | default when absent; score; Bayesian-calibrated confidence: tracks retrieval correctness via alpha (helpful count) / beta (misleading count) Beta distribution. calibrated_confidence = alpha / (alpha + beta), with Bayesian weight ramp: w = min(1.0, (alpha + beta - 2) / 8) Range: 0.0 (untested or consistently misleading) to 1.0 (consistently helpful) |
| `confidence_alpha` | `f32` | no | default when absent; Beta distribution alpha parameter (successful retrievals + prior) |
| `confidence_beta` | `f32` | no | default when absent; Beta distribution beta parameter (failed retrievals + prior) |

### MemoryNeuron

Defined in `src/handlers/visualization.rs`. Individual memory neuron for visualization

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content_preview` | `String` | yes |  |
| `activation` | `f32` | yes |  |
| `importance` | `f32` | yes | score |
| `tier` | `String` | yes |  |
| `access_count` | `u32` | yes |  |
| `created_at` | `String` | yes |  |

### MemoryRevision

Defined in `src/memory/types.rs`. A revision in memory history - tracks what changed and when

| Field | Type | Required | Notes |
|---|---|---|---|
| `previous_content` | `String` | yes | Previous content before this change |
| `change_type` | [`ChangeType`](#changetype) | yes | What type of change occurred |
| `changed_at` | `DateTime<Utc>` | yes | When the change happened |
| `changed_by` | `Option<String>` | no | Optional: who/what triggered the change |
| `change_reason` | `Option<String>` | no | Optional: brief description of change |

### MemoryStats

Defined in `src/memory/types.rs`. Memory statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_memories` | `usize` | yes |  |
| `working_memory_count` | `usize` | yes |  |
| `session_memory_count` | `usize` | yes |  |
| `long_term_memory_count` | `usize` | yes |  |
| `vector_index_count` | `usize` | yes |  |
| `compressed_count` | `usize` | yes |  |
| `promotions_to_session` | `usize` | yes |  |
| `promotions_to_longterm` | `usize` | yes |  |
| `total_retrievals` | `usize` | yes |  |
| `average_importance` | `f32` | yes | score |
| `graph_nodes` | `usize` | no | default when absent; Knowledge graph entity count |
| `graph_edges` | `usize` | no | default when absent; Knowledge graph relationship count |

### MemoryTier

Defined in `src/memory/types.rs`. Memory tier in the cognitive hierarchy

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Working` | none | Active, immediate context (Cowan's focus of attention) |
| `Session` | none | Current task/session context |
| `LongTerm` | none | Consolidated durable memories |
| `Archive` | none | Compressed archival storage |

### MemoryUniverse

Defined in `src/graph_memory.rs`. The complete memory universe visualization

| Field | Type | Required | Notes |
|---|---|---|---|
| `stars` | Vec&lt;[`UniverseStar`](#universestar)&gt; | yes |  |
| `connections` | Vec&lt;[`GravitationalConnection`](#gravitationalconnection)&gt; | yes |  |
| `total_entities` | `usize` | yes |  |
| `total_connections` | `usize` | yes |  |
| `bounds` | [`UniverseBounds`](#universebounds) | yes |  |

### MemoryWithHierarchy

Defined in `src/handlers/crud.rs`. Response for GET /api/memories/{memory_id} - includes hierarchy context

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory` | [`Memory`](#memory) | yes | flattened into the parent; The memory itself (flattened) |
| `children_ids` | `Vec<String>` | yes | Children memory IDs (if any) |
| `children_count` | `usize` | yes | Number of children |
| `full_content` | `bool` | yes | True when `experience.content` is the complete stored text. False when the record is compressed and `?decompress=1` was not asked for, or when the record is a legacy one whose body was replaced at storage time and cannot be restored; `content_note` says which. Measured 2026-09-02: `?full=true` and `?decompress=true` were silently ignored and a 410-byte preview came back as if it were a 19,396-byte document. |
| `lossless` | `bool` | yes | Whether the full content is recoverable at all (false only for legacy lossy records). |
| `content_note` | `Option<String>` | no | omitted when empty; Why `full_content` is false, when it is. |

### MergeEntitiesRequest

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `source_name` | `String` | yes |  |
| `target_name` | `String` | yes |  |

### MergeEntitiesResponse

Defined in `src/handlers/prompt_gen.rs`. Reply of `POST /api/entity/merge`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `source` | `String` | yes | The entity deleted. |
| `target` | `String` | yes | The entity its edges now point at. |
| `edges_moved` | `usize` | yes |  |
| `status` | `String` | yes | Always `merged`. |

### MergeMemoriesResponse

Defined in `src/handlers/memory_ops.rs`. Reply of `POST /api/memory/merge`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `merged_id` | `String` | yes | The memory written from the merged content. |
| `source_ids` | `Vec<String>` | yes | The memories forgotten into it, as sent. |

### MergeRequest

Defined in `src/handlers/memory_ops.rs`. Request to merge several memories into one.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `ids` | `Vec<String>` | yes |  |
| `separator` | `Option<String>` | no | default when absent; Separator placed between merged bodies (default `\n\n---\n\n`). |
| `memory_type` | `Option<String>` | no | default when absent; Experience type for the merged memory (default `Context`). |

### MifAdapter

Defined in `src/handlers/mif.rs`. One import/export adapter.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `format` | `String` | yes | The value to send as `format`. |

### MifAdaptersResponse

Defined in `src/handlers/mif.rs`. Reply of `GET /api/mif/adapters`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `adapters` | Vec&lt;[`MifAdapter`](#mifadapter)&gt; | yes |  |
| `default_export` | `String` | yes | The `format` `POST /api/export/mif` uses when none is sent. |
| `default_import` | `String` | yes |  |

### MifDocument

Defined in `src/mif/schema.rs`. MIF v2 document — the top-level interchange container.

| Field | Type | Required | Notes |
|---|---|---|---|
| `mif_version` | `String` | yes |  |
| `generator` | [`MifGenerator`](#mifgenerator) | yes |  |
| `export_meta` | [`MifExportMeta`](#mifexportmeta) | yes |  |
| `memories` | Vec&lt;[`MifMemory`](#mifmemory)&gt; | no | default when absent |
| `knowledge_graph` | Option&lt;[`MifKnowledgeGraph`](#mifknowledgegraph)&gt; | no | default when absent; omitted when empty |
| `todos` | Vec&lt;[`MifTodo`](#miftodo)&gt; | no | default when absent |
| `projects` | Vec&lt;[`MifProject`](#mifproject)&gt; | no | default when absent |
| `reminders` | Vec&lt;[`MifReminder`](#mifreminder)&gt; | no | default when absent |
| `vendor_extensions` | `HashMap<String, serde_json::Value>` | no | default when absent; Vendor-specific metadata. Key = vendor name (e.g., "veld"). Allows lossless round-trip of system-specific data without polluting the core schema. |

### MifEmbedding

Defined in `src/mif/schema.rs`. Model-tagged embedding vector for cross-system portability.

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | `String` | yes |  |
| `dimensions` | `usize` | yes |  |
| `vector` | `Vec<f32>` | yes |  |
| `normalized` | `bool` | no | default when absent |

### MifEntityRef

Defined in `src/mif/schema.rs`. An entity mentioned in a memory, with its type preserved.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `entity_type` | `String` | no | default when absent; Lowercase entity type: "person", "organization", "location", "technology", "concept", "event", "date", "product", "skill", "keyword", "unknown". |
| `confidence` | `f32` | no | default when absent; score |

### MifExportMeta

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `created_at` | `DateTime<Utc>` | yes |  |
| `user_id` | `String` | yes |  |
| `checksum` | `String` | yes |  |
| `privacy` | Option&lt;[`MifPrivacy`](#mifprivacy)&gt; | no | default when absent; omitted when empty |

### MifExportRequest

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `include_embeddings` | `bool` | no | default when absent |
| `include_graph` | `bool` | no | default when absent |
| `since` | `Option<String>` | no | default when absent |
| `redact_pii` | `bool` | no | default when absent |
| `format` | `String` | no | default when absent |

### MifGenerator

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `version` | `String` | yes |  |

### MifGraphEntity

Defined in `src/mif/schema.rs`. A node in the knowledge graph.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `name` | `String` | yes |  |
| `types` | `Vec<String>` | no | default when absent; Lowercase type strings: ["person"], ["technology", "concept"], etc. |
| `attributes` | `HashMap<String, String>` | no | default when absent |
| `summary` | `String` | no | default when absent |
| `created_at` | `DateTime<Utc>` | yes |  |
| `last_seen_at` | `DateTime<Utc>` | yes |  |

### MifGraphEpisode

Defined in `src/mif/schema.rs`. An episodic node — a temporal grouping of entities.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `name` | `String` | yes |  |
| `content` | `String` | no | default when absent |
| `entity_ids` | `Vec<Uuid>` | no | default when absent |
| `source` | `String` | no | default when absent |
| `metadata` | `HashMap<String, String>` | no | default when absent |
| `created_at` | `DateTime<Utc>` | yes |  |

### MifGraphRelationship

Defined in `src/mif/schema.rs`. An edge in the knowledge graph.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `source_entity_id` | `Uuid` | yes |  |
| `target_entity_id` | `Uuid` | yes |  |
| `relation_type` | `String` | yes | Lowercase snake_case: "works_with", "part_of", "located_in", "causes", etc. |
| `context` | `String` | no | default when absent |
| `confidence` | `Option<f32>` | no | default when absent; omitted when empty; score; Normalized confidence/strength (0.0–1.0). Vendor-neutral. |
| `created_at` | `DateTime<Utc>` | yes |  |
| `valid_at` | `DateTime<Utc>` | yes |  |
| `invalidated_at` | `Option<DateTime<Utc>>` | no | default when absent; omitted when empty |

### MifImportRequest

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `skip_duplicates` | `bool` | no | default when absent |
| `format` | `Option<String>` | no | default when absent |
| `data` | `serde_json::Value` | yes |  |

### MifImportResponse

Defined in `src/handlers/mif.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `result` | import::[`ImportResult`](#importresult) | yes |  |

### MifKnowledgeGraph

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entities` | Vec&lt;[`MifGraphEntity`](#mifgraphentity)&gt; | no | default when absent |
| `relationships` | Vec&lt;[`MifGraphRelationship`](#mifgraphrelationship)&gt; | no | default when absent |
| `episodes` | Vec&lt;[`MifGraphEpisode`](#mifgraphepisode)&gt; | no | default when absent |

### MifMemory

Defined in `src/mif/schema.rs`. A single memory in vendor-neutral form.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `content` | `String` | yes |  |
| `memory_type` | `String` | yes | Lowercase type string: "observation", "decision", "learning", "error", "discovery", "pattern", "context", "task", "code_edit", "file_access", "search", "command", "conversation", "intention". |
| `created_at` | `DateTime<Utc>` | yes |  |
| `tags` | `Vec<String>` | no | default when absent |
| `entities` | Vec&lt;[`MifEntityRef`](#mifentityref)&gt; | no | default when absent |
| `metadata` | `HashMap<String, String>` | no | default when absent |
| `embeddings` | Option&lt;[`MifEmbedding`](#mifembedding)&gt; | no | default when absent; omitted when empty |
| `source` | Option&lt;[`MifSource`](#mifsource)&gt; | no | default when absent; omitted when empty |
| `parent_id` | `Option<Uuid>` | no | default when absent; omitted when empty |
| `related_memory_ids` | `Vec<Uuid>` | no | default when absent |
| `related_todo_ids` | `Vec<Uuid>` | no | default when absent |
| `agent_id` | `Option<String>` | no | default when absent; omitted when empty |
| `external_id` | `Option<String>` | no | default when absent; omitted when empty |
| `version` | `u32` | no | default when absent |

### MifPrivacy

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `pii_detected` | `bool` | yes |  |
| `secrets_detected` | `bool` | yes |  |
| `redacted_fields` | `Vec<String>` | no | default when absent |

### MifProject

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `name` | `String` | yes |  |
| `prefix` | `String` | no | default when absent |
| `description` | `Option<String>` | no | default when absent; omitted when empty |
| `status` | `String` | no | default when absent; "active", "archived", "completed" |
| `created_at` | `DateTime<Utc>` | yes |  |
| `color` | `Option<String>` | no | default when absent; omitted when empty |
| `icon` | `Option<String>` | no | default when absent; omitted when empty |

### MifReminder

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `content` | `String` | yes |  |
| `trigger` | [`MifTrigger`](#miftrigger) | yes |  |
| `status` | `String` | yes | "pending", "triggered", "dismissed", "expired" |
| `priority` | `u8` | no | default when absent |
| `tags` | `Vec<String>` | no | default when absent |
| `created_at` | `DateTime<Utc>` | yes |  |
| `triggered_at` | `Option<DateTime<Utc>>` | no | default when absent; omitted when empty |
| `dismissed_at` | `Option<DateTime<Utc>>` | no | default when absent; omitted when empty |

### MifSource

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `source_type` | `String` | no | default when absent |
| `session_id` | `Option<String>` | no | default when absent; omitted when empty |
| `agent` | `Option<String>` | no | default when absent; omitted when empty |

### MifTodo

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `content` | `String` | yes |  |
| `status` | `String` | yes | "backlog", "todo", "in_progress", "blocked", "done", "cancelled" |
| `priority` | `String` | yes | "urgent", "high", "medium", "low", "none" |
| `created_at` | `DateTime<Utc>` | yes |  |
| `updated_at` | `DateTime<Utc>` | yes |  |
| `due_date` | `Option<DateTime<Utc>>` | no | default when absent; omitted when empty |
| `completed_at` | `Option<DateTime<Utc>>` | no | default when absent; omitted when empty |
| `project_id` | `Option<Uuid>` | no | default when absent; omitted when empty |
| `parent_id` | `Option<Uuid>` | no | default when absent; omitted when empty |
| `tags` | `Vec<String>` | no | default when absent |
| `contexts` | `Vec<String>` | no | default when absent |
| `notes` | `Option<String>` | no | default when absent; omitted when empty |
| `blocked_on` | `Option<String>` | no | default when absent; omitted when empty |
| `recurrence` | `Option<String>` | no | default when absent; omitted when empty |
| `comments` | Vec&lt;[`MifTodoComment`](#miftodocomment)&gt; | no | default when absent |
| `related_memory_ids` | `Vec<Uuid>` | no | default when absent |
| `external_id` | `Option<String>` | no | default when absent; omitted when empty |

### MifTodoComment

Defined in `src/mif/schema.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `Uuid` | yes |  |
| `content` | `String` | yes |  |
| `comment_type` | `String` | no | default when absent; "comment", "progress", "resolution", "activity" |
| `created_at` | `DateTime<Utc>` | yes |  |
| `author` | `Option<String>` | no | default when absent; omitted when empty |

### MifTrigger

Defined in `src/mif/schema.rs`.

Enum, internally tagged: the payload's fields plus `"type": <variant>`.

| Variant | Payload | Notes |
|---|---|---|
| `time` | `{ at: DateTime<Utc> }` |  |
| `duration` | `{ seconds: u64, from: DateTime<Utc> }` |  |
| `context` | `{ keywords: Vec<String>, threshold: f32 }` |  |

### MigrateLegacyRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### MigrateLegacyResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `migrated_count` | `usize` | yes |  |
| `already_current_count` | `usize` | yes |  |
| `failed_count` | `usize` | yes |  |

### MirrorReport

Defined in `src/memory/gap_topology/run.rs`. How a run's mirror was brought up to the graph.

| Field | Type | Required | Notes |
|---|---|---|---|
| `mode` | [`SyncMode`](#syncmode) | yes |  |
| `best_effort` | `bool` | yes | A full copy that could not account for every graph write made during it. |
| `changed_during_copy` | `Option<usize>` | no | Graph writes made during a full copy and applied after it. |
| `rows_written` | `usize` | yes | Mirror rows written or removed. |
| `duration_ms` | `u64` | yes |  |

### MissedQuery

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | `String` | yes |  |
| `times` | `usize` | yes |  |
| `last_asked` | `DateTime<Utc>` | yes |  |

### MultiModalSearchRequest

Defined in `src/handlers/search.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query_text` | `String` | yes |  |
| `mode` | `String` | yes |  |
| `limit` | `Option<usize>` | no |  |

### NerEntityRecord

Defined in `src/memory/types.rs`. Raw experience data to be stored (ENHANCED with smart defaults)

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | `String` | yes |  |
| `entity_type` | `String` | yes | NER type: "PER", "ORG", "LOC", "MISC" |
| `confidence` | `f32` | yes | score |
| `start_char` | `Option<usize>` | no | default when absent; Character offset of entity start in source content |
| `end_char` | `Option<usize>` | no | default when absent; Character offset of entity end in source content |

### OriginCounts

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `authored` | `usize` | yes |  |
| `derived` | `usize` | yes |  |
| `undetermined` | `usize` | yes |  |

### OwnSessionBody

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | `String` | yes |  |

### PasskeyResultBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `credential_id` | `Option<String>` | no |  |
| `prf` | `Option<String>` | no |  |
| `error` | `Option<String>` | no |  |

### PasskeyStartBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `unlock` | [`UnlockBody`](#unlockbody) | yes |  |
| `label` | `Option<String>` | no |  |

### PersistencePairSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `dimension` | `usize` | yes |  |
| `dimension_name` | `String` | yes |  |
| `birth` | `f32` | yes |  |
| `death` | `Option<f32>` | no |  |
| `persistence` | `f32` | yes |  |
| `birth_entities` | `Vec<String>` | yes |  |
| `sandwich_lower` | `f32` | yes |  |
| `sandwich_upper` | `f32` | yes |  |

### PersistenceRequest

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `step_size` | `f32` | no | default when absent |
| `min_persistence` | `f32` | no | default when absent |
| `offset` | `usize` | no | default when absent |
| `limit` | `usize` | no | default when absent |

### PersistenceResponse

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_count` | `usize` | yes |  |
| `pairs` | Vec&lt;[`PersistencePairSummary`](#persistencepairsummary)&gt; | yes |  |
| `betti_curves` | Vec&lt;[`BettiCurveSummary`](#betticurvesummary)&gt; | yes |  |
| `stats` | [`PersistenceStatsSummary`](#persistencestatssummary) | yes |  |

### PersistenceStatsSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `filtration_levels` | `usize` | yes |  |
| `total_pairs` | `usize` | yes |  |
| `persistent_features` | `usize` | yes |  |
| `noise_features` | `usize` | yes |  |
| `duration_ms` | `u64` | yes |  |

### PiiClassification

Defined in `src/graph_memory.rs`. PII classification for entities, used to control cross-user data flow.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `PersonalIdentity` | none | Contains personally identifiable information (names, contacts) |
| `OrganizationIdentity` | none | Contains organization-specific information |
| `Clean` | none | No PII detected — safe for cross-user aggregation |

### Place

Defined in `src/memory/facets.rs`. One typed place layer — where in physical, organizational, or conceptual space.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Repo` | `{ slug: String, branch: Option<String>, commit: Option<String>, remote: Option<String>, pull_request: Option<String>, dirty: Option<bool> }` | A version-control repository. Absorbs the first-pass `RepositoryContext`. |
| `Module` | `{ path: String }` | A module / package / namespace within a project. |
| `File` | `{ path: String, symbol: Option<String> }` | A single source file, optionally scoped to a symbol. |
| `Geo` | `{ lat: f64, lon: f64, alt: Option<f64> }` | A geographic location (robotics, geolocation-tagged memory). |
| `Host` | `{ name: String, environment: Option<String> }` | A network host or deployment environment. |
| `Url` | `{ href: String }` | A web URL. |
| `Named` | `{ label: String }` | A free-form named place when nothing else fits. |
| `LocalFrame` | `{ frame: String, x: f32, y: f32, z: f32, orientation: Option<Orientation> }` | A position in a named local cartesian frame (robot frame, site frame, floor plan, etc.). Absorbs the legacy `Experience.local_position` flat field — the migrator emits `frame: "robot"` for that source. |
| `GeoFix` | `{ lat: f64, lon: f64, alt: Option<f64>, accuracy_m: Option<f32>, speed_m_s: Option<f32>, course_deg: Option<f32>, captured_at: Option<DateTime<Utc>> }` | A geographic fix with full GPS/INS metadata — for callers that have it. `Place::Geo` stays the simple-case variant; `GeoFix` is the rich one. `resolved_geo()` reads both. |

### PlanResponse

Defined in `src/handlers/query.rs`. Response for `POST /api/query/plan`: the human-readable plan plus the op count. The full `PhysicalPlan` is intentionally not serialised — the `explain` string is its canonical, stable rendering.

| Field | Type | Required | Notes |
|---|---|---|---|
| `explain` | `String` | yes |  |
| `op_count` | `usize` | yes |  |

### PlumbingRule

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `namespaces` | `Vec<String>` | yes |  |
| `bare` | `Vec<String>` | yes |  |
| `from_env` | `Vec<String>` | yes |  |

### Population

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | `usize` | yes |  |
| `with_tag_sources` | `usize` | yes |  |
| `undetermined` | `usize` | yes |  |
| `distinct_tags` | `usize` | yes |  |

### Position3D

Defined in `src/graph_memory.rs`. 3D position in the memory universe

| Field | Type | Required | Notes |
|---|---|---|---|
| `x` | `f32` | yes |  |
| `y` | `f32` | yes |  |
| `z` | `f32` | yes |  |

### Prediction

Defined in `src/memory/facets.rs`. A prediction made at encoding and its observed outcome.

| Field | Type | Required | Notes |
|---|---|---|---|
| `expected` | `Option<String>` | no | What was predicted. |
| `observed` | `Option<String>` | no | What was observed. |
| `accurate` | `Option<bool>` | no | Whether the prediction held — `None` if not yet evaluated. |

### PrimaryEmbedderInfo

Defined in `src/embeddings/mod.rs`. The primary embedder this process chose, for `/health` and anyone else who needs to know what the vectors in the store were made by. Recorded once at the first `MemorySystem` construction; every user's system reads the same environment, so first-wins is exact.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Canonical id, the value the corpus marker `.veld_embedder` holds. |
| `model` | `String` | yes |  |
| `dimension` | `usize` | yes |  |
| `kind` | `&'static str` | yes | `nomic-onnx` \| `minilm-onnx` \| `arctic-onnx` \| `http` — where the vectors come from. |
| `reason` | `String` | yes | Why this one: the env that asked for it, or the fallback step that reached it. |

### ProactiveContextRequest

Defined in `src/handlers/recall.rs`. Request for proactive context - returns relevant memories + triggered reminders

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `context` | `String` | yes |  |
| `max_results` | `usize` | no | default when absent |
| `semantic_threshold` | `f32` | no | default when absent; Minimum semantic similarity threshold (0.0-1.0) |
| `entity_match_weight` | `f32` | no | default when absent; score; Weight for entity matching in relevance scoring |
| `recency_weight` | `f32` | no | default when absent; score; Weight for recency boost |
| `memory_types` | `Vec<String>` | no | default when absent; Filter to specific memory types |
| `auto_ingest` | `bool` | no | default when absent; Whether to auto-ingest the context as a Conversation memory. Off unless asked: hooks already record each turn, and a default ingest stored the same context twice. |
| `previous_response` | `Option<String>` | no | default when absent; Agent's previous response (for implicit feedback extraction) |
| `user_followup` | `Option<String>` | no | default when absent; User's followup message after agent response (for delayed signals) |
| `tool_actions` | Vec&lt;crate::memory::feedback::[`ToolAction`](#toolaction)&gt; | no | default when absent; Tool/actuator actions performed since last proactive_context call. Used for tool-aware feedback attribution: matches actions against previously surfaced memories to detect concrete usage. Claude Code: collected by hooks (Read, Edit, Bash calls). Robotics: constructed from action-outcome Experience fields. |
| `rrf_k` | `Option<f32>` | no | default when absent; RRF k parameter for reciprocal rank fusion (default: 20.0) Higher k = more equal weighting; lower k = sharper top-rank discrimination |
| `competition_mode` | Option&lt;crate::memory::[`CompetitionMode`](#competitionmode)&gt; | no | default when absent; Optional retrieval-time competition policy override. |
| `surfaced_memory_ids` | `Vec<String>` | no | default when absent; IDs of memories surfaced in the previous proactive_context call. Used for explicit feedback: which surfaced memories were actually relevant. |
| `episode_id` | `Option<String>` | no | default when absent; Episode ID for auto-ingested memories (FIX-05: quality parity) When set, auto-ingested memories inherit this episode context |
| `sequence_number` | `Option<u32>` | no | default when absent; Sequence number for auto-ingested memories (FIX-05) |
| `emotional_valence` | `Option<f32>` | no | default when absent; Emotional valence for auto-ingested memories (FIX-05) |
| `emotional_arousal` | `Option<f32>` | no | default when absent; Emotional arousal for auto-ingested memories (FIX-05) |
| `repo` | `Option<String>` | no | default when absent; Surface only memories scoped to this repository slug, and scope auto-ingested memories to it. Todos, reminders and facts are not scoped. |

### ProactiveContextResponse

Defined in `src/handlers/recall.rs`. Response for proactive context

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | Vec&lt;[`ProactiveSurfacedMemory`](#proactivesurfacedmemory)&gt; | yes | Relevant memories based on context |
| `due_reminders` | Vec&lt;[`ReminderItem`](#reminderitem)&gt; | yes | Due time-based reminders |
| `context_reminders` | Vec&lt;[`ReminderItem`](#reminderitem)&gt; | yes | Context-triggered reminders (keyword match) |
| `memory_count` | `usize` | yes | Total counts |
| `reminder_count` | `usize` | yes |  |
| `ingested_memory_id` | `Option<String>` | no | omitted when empty; ID of auto-ingested memory (if auto_ingest=true) |
| `feedback_processed` | Option&lt;[`FeedbackProcessed`](#feedbackprocessed)&gt; | no | omitted when empty; Feedback processing results (if previous_response was provided) |
| `relevant_todos` | Vec&lt;[`ProactiveTodoItem`](#proactivetodoitem)&gt; | no | default when absent; omitted when empty; Relevant todos based on context |
| `todo_count` | `usize` | no | default when absent; Todo count |
| `relevant_facts` | Vec&lt;[`ProactiveFact`](#proactivefact)&gt; | no | default when absent; omitted when empty; Consolidated facts from knowledge graph |
| `latency_ms` | `f64` | yes | Processing latency in milliseconds |
| `detected_entities` | Vec&lt;[`DetectedEntityInfo`](#detectedentityinfo)&gt; | no | default when absent; omitted when empty; Entities detected in the query context |

### ProactiveFact

Defined in `src/handlers/recall.rs`. Consolidated fact surfaced in proactive context

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `fact` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `support_count` | `usize` | yes |  |
| `related_entities` | `Vec<String>` | yes |  |

### ProactiveSurfacedMemory

Defined in `src/handlers/recall.rs`. Surfaced memory in proactive context response

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `memory_type` | `String` | yes |  |
| `score` | `f32` | yes |  |
| `importance` | `f32` | yes | score |
| `created_at` | `String` | yes |  |
| `tags` | `Vec<String>` | yes |  |
| `tier` | `String` | yes |  |
| `relevance_reason` | `String` | yes | Why this memory was surfaced ("semantic", "entity", "combined", "co_activation") |
| `matched_entities` | `Vec<String>` | no | default when absent; omitted when empty; Entities from this memory that matched the query context |
| `hours_since_created` | `f32` | yes | Hours since this memory was first created |
| `hours_since_last_accessed` | `f32` | yes | Hours since this memory was last accessed/retrieved |
| `access_count` | `u32` | yes | Number of times this memory has been retrieved |
| `activation_level` | `f32` | yes | Current activation level (0.0 = dormant, 1.0 = labile/active) |
| `retrieval_trigger` | `String` | yes | How this memory was triggered: "semantic" \| "entity" \| "co_activation" \| "combined" |
| `intrusion_score` | `f32` | yes | score; Intrusion score: high activation + recent access = likely co-activation intrusion. Low activation + old access = genuine semantic match. Formula: activation * (1.0 / hours_since_last_accessed.max(0.1)) |
| `competition_conflict` | `Option<bool>` | no | omitted when empty; Whether this memory is in active competition with another result. |
| `competition_opponent` | `Option<String>` | no | omitted when empty; Opponent memory ID when competition_conflict is true. |

### ProactiveTodoItem

Defined in `src/handlers/recall.rs`. Todo item in proactive context response

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `short_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `status` | `String` | yes |  |
| `priority` | `String` | yes |  |
| `project` | `Option<String>` | no |  |
| `due_date` | `Option<String>` | no |  |
| `relevance_reason` | `String` | yes |  |
| `similarity_score` | `Option<f32>` | no | omitted when empty; score; Semantic similarity score (0.0 - 1.0) |

### Producer

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `share` | `f32` | yes |  |

### Project

Defined in `src/memory/types.rs`. A project that groups related todos

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`ProjectId`](#projectid) | yes | Unique identifier |
| `user_id` | `String` | yes | User who owns this project |
| `name` | `String` | yes | Project name |
| `prefix` | `Option<String>` | no | default when absent; Short prefix for todo IDs (e.g., "BOLT", "MEM") If not set, derived from first letters of project name |
| `description` | `Option<String>` | no | Optional description |
| `status` | [`ProjectStatus`](#projectstatus) | no | default when absent; Project status |
| `color` | `Option<String>` | no | Optional color (hex) |
| `parent_id` | Option&lt;[`ProjectId`](#projectid)&gt; | no | default when absent; Parent project ID (for sub-projects) |
| `created_at` | `DateTime<Utc>` | yes | When created |
| `completed_at` | `Option<DateTime<Utc>>` | no | When completed |
| `codebase_path` | `Option<String>` | no | default when absent; Absolute path to the codebase root directory |
| `codebase_indexed` | `bool` | no | default when absent; Whether the codebase has been indexed |
| `codebase_indexed_at` | `Option<DateTime<Utc>>` | no | default when absent; When the codebase was last indexed |
| `codebase_file_count` | `usize` | no | default when absent; Number of files indexed in the codebase |
| `embedding` | `Option<Vec<f32>>` | no | default when absent; omitted when empty; Vector embedding for semantic search (MiniLM-L6-v2, 384 dimensions) Computed from name + description for similarity matching |
| `related_memory_ids` | Vec&lt;[`MemoryId`](#memoryid)&gt; | no | default when absent; Related memory IDs for project-level context |
| `todo_counts` | [`ProjectTodoCounts`](#projecttodocounts) | no | default when absent; Aggregated todo count by status (cached for quick display) |

### ProjectContext

Defined in `src/memory/types.rs`. Project-specific context

| Field | Type | Required | Notes |
|---|---|---|---|
| `project_id` | `Option<String>` | no | Project ID |
| `name` | `Option<String>` | no | Project name |
| `project_type` | `Option<String>` | no | Project type (web, mobile, ML, etc) |
| `technologies` | `Vec<String>` | yes | Tech stack |
| `current_phase` | `Option<String>` | no | Current sprint/milestone |
| `active_files` | `Vec<String>` | yes | Related files being worked on |
| `current_task` | `Option<String>` | no | Current task/feature |
| `dependencies` | `Vec<String>` | yes | Project dependencies |

### ProjectId

Defined in `src/memory/types.rs`. Unique identifier for projects

On the wire this is its inner value, `Uuid`.

### ProjectListResponse

Defined in `src/handlers/todos.rs`. Response for project list

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `count` | `usize` | yes |  |
| `projects` | Vec&lt;([`Project`](#project), [`ProjectStats`](#projectstats))&gt; | yes |  |
| `formatted` | `String` | yes |  |

### ProjectResponse

Defined in `src/handlers/todos.rs`. Response for project operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `project` | Option&lt;[`Project`](#project)&gt; | no |  |
| `stats` | Option&lt;[`ProjectStats`](#projectstats)&gt; | no |  |
| `formatted` | `String` | yes |  |

### ProjectStats

Defined in `src/memory/todos.rs`. Stats for a single project

| Field | Type | Required | Notes |
|---|---|---|---|
| `total` | `usize` | yes |  |
| `backlog` | `usize` | yes |  |
| `todo` | `usize` | yes |  |
| `in_progress` | `usize` | yes |  |
| `blocked` | `usize` | yes |  |
| `done` | `usize` | yes |  |
| `cancelled` | `usize` | yes |  |

### ProjectStatus

Defined in `src/memory/types.rs`. Project status

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `active` | none |  |
| `on_hold` | none |  |
| `completed` | none |  |
| `archived` | none |  |

### ProjectTodoCounts

Defined in `src/memory/types.rs`. Cached todo counts for quick project display

| Field | Type | Required | Notes |
|---|---|---|---|
| `total` | `usize` | yes |  |
| `backlog` | `usize` | yes |  |
| `todo` | `usize` | yes |  |
| `in_progress` | `usize` | yes |  |
| `blocked` | `usize` | yes |  |
| `done` | `usize` | yes |  |

### PromptGenRequest

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `goal` | `String` | yes | The goal, question, or task the prompt is being assembled for. |
| `context` | `Option<String>` | no | default when absent; Optional current context (e.g., the user's latest message or file content). Used to bias retrieval toward relevant memories. |
| `max_memories` | `usize` | no | default when absent; Maximum memories to include (default: 10) |
| `max_facts` | `usize` | no | default when absent; Maximum facts to include (default: 10) |
| `max_entities` | `usize` | no | default when absent; Maximum entities to include (default: 15) |
| `max_todos` | `usize` | no | default when absent; Maximum todos to include (default: 5) |
| `include_graph` | `bool` | no | default when absent; Include entity graph neighborhood (default: true) |
| `include_reminders` | `bool` | no | default when absent; Include due reminders (default: true) |
| `include_todos` | `bool` | no | default when absent; Include active todos (default: true) |
| `include_facts` | `bool` | no | default when absent; Include extracted facts (default: true) |
| `resolve_entities` | `Vec<String>` | no | default when absent; Specific entity names to resolve and include (exact match with alias resolution) |
| `memory_types` | `Vec<String>` | no | default when absent; Memory types to filter for (empty = all types) |
| `mode` | `String` | no | default when absent; Retrieval mode: semantic, associative, temporal, hybrid (default: hybrid) |

### PromptGenResponse

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `sections` | Vec&lt;[`PromptSection`](#promptsection)&gt; | yes | Assembled prompt context as structured sections |
| `prompt` | `String` | yes | The full assembled prompt text (all sections concatenated) |
| `entities` | Vec&lt;[`ResolvedEntity`](#resolvedentity)&gt; | yes | Resolved entities with their structured details |
| `relationships` | Vec&lt;[`EntityRelationship`](#entityrelationship)&gt; | yes | Active relationships between resolved entities |
| `meta` | [`PromptMeta`](#promptmeta) | yes | Metadata about what was assembled |

### PromptMeta

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_retrieved` | `usize` | yes |  |
| `facts_retrieved` | `usize` | yes |  |
| `entities_resolved` | `usize` | yes |  |
| `todos_included` | `usize` | yes |  |
| `reminders_included` | `usize` | yes |  |
| `latency_ms` | `f64` | yes |  |

### PromptSection

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `String` | yes |  |
| `content` | `String` | yes |  |
| `item_count` | `usize` | yes |  |

### Provenance

Defined in `src/memory/facets.rs`. Source-monitoring metadata: where the information came from, how credibly, and through how many hops.

| Field | Type | Required | Notes |
|---|---|---|---|
| `source_kind` | `Option<String>` | no | The source channel kind (e.g. `"user"`, `"tool"`, `"github_webhook"`). |
| `source_id` | `Option<String>` | no | Identifier of the source within its kind. |
| `credibility` | `f32` | yes | Operator-assigned credibility weight, 0.0–1.0. Feeds retrieval ranking. |
| `verified` | `bool` | yes | Whether this engram has been independently verified. |
| `chain` | Vec&lt;[`ProvenanceHop`](#provenancehop)&gt; | yes | Structured relay chain — each hop the information passed through. |

### ProvenanceHop

Defined in `src/memory/facets.rs`. One hop in a provenance chain.

| Field | Type | Required | Notes |
|---|---|---|---|
| `source_kind` | `Option<String>` | no |  |
| `source_id` | `Option<String>` | no |  |
| `credibility` | `Option<f32>` | no |  |

### PurgeBackupsRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `keep_count` | `usize` | yes |  |

### PurgeBackupsResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `purged_count` | `usize` | yes |  |

### PurgeReason

Defined in `src/memory/compression.rs`. Reason a fact was administratively purged. Closed enum (no `Other` escape hatch) to keep `purge_reason` non-leaky: free-form operator notes belong in the audit log, not on the fact record.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `UserRequest` | none | Explicit user request to remove specific content. |
| `AdminCleanup` | none | Operator-initiated cleanup not tied to a particular pattern. |
| `PatternMatch` | `{ pattern_hash: String }` | Matched a substring pattern. Pattern is stored as a SHA-256 hash to avoid leaking content into the record; the raw pattern (if forensics needs it) lives in the audit log keyed by purge event id. |
| `ConfidenceFloor` | `{ threshold: f32 }` | Calibrated confidence fell below the given threshold. |

### Query

Defined in `src/query_planner/predicate.rs`. A single planner-level query. Combines relational filters, an optional vector scan, and graph reachability predicates against a single tenant (`user_id`).

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes | Tenant scope. The executor always passes this to every querier so backends can enforce per-user isolation. |
| `relational` | Vec&lt;[`RelationalPredicate`](#relationalpredicate)&gt; | yes | Structured relational predicates (column equality, IN-set, range). |
| `vector` | Option&lt;[`VectorPredicate`](#vectorpredicate)&gt; | no | Optional vector ANN predicate. At most one — multi-vector planning is W7. |
| `graph` | Vec&lt;[`GraphPredicate`](#graphpredicate)&gt; | yes | Graph reachability predicates (linked-to-entity, shares-entity, episode). |
| `limit` | `usize` | yes | Final top-N to return after intersection. |
| `stage_caps` | [`StageCaps`](#stagecaps) | yes | Per-stage candidate caps. Clamps scan estimates so the planner never asks a backend to materialize more than its budget allows. |
| `query_text` | `Option<String>` | no | default when absent; Optional raw natural-language query text. The planner itself never reads this — it exists so the HTTP `/api/query/execute` surface can run B4 temporal answer-derivation ("when did/is X?") alongside the structured plan. Absent in existing/structured-only payloads (`#[serde(default)]`), so the wire format and planner behaviour are unchanged when it is `None`. |

### QueryMetacognition

Defined in `src/memory/types.rs`. Query-level metacognition (M3): a returned feeling-of-knowing readout.

| Field | Type | Required | Notes |
|---|---|---|---|
| `fok` | `f32` | yes | Calibrated feeling-of-knowing for the whole query, 0.0–1.0. |
| `label` | `String` | yes | Coarse label: "high" \| "medium" \| "low". |
| `peak_confidence` | `f32` | yes | score; How sharply the top result dominates: top/(top+second). ~0.5 = ambiguous, approaching 1.0 = a lone strong match. |
| `answerability` | `f32` | yes | Squashed top score top/(top+1): a bounded proxy for "is there an answer". |
| `cross_embedder_agreement` | `Option<f32>` | no | omitted when empty; Cross-embedder (S3) agreement over the top results, when available. |
| `coverage` | `Option<f32>` | no | omitted when empty; Query-level entity coverage (M4): fraction of the query's focal entities covered by any top result. None when the query has no resolved entities. |
| `in_known_gap` | `bool` | yes | Whether the query sits in/near a known unresolved knowledge gap (M5 link). |
| `signal_strength` | `String` | yes | Which signals were available: "score_only" \| "score+gap" \| … |

### QueryRequest

Defined in `src/handlers/datasets.rs`. Parametrised SELECT.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `where` | `Option<String>` | no |  |
| `params` | `Vec<serde_json::Value>` | no | default when absent |
| `limit` | `Option<u32>` | no |  |

### ReadyResponse

Defined in `src/handlers/health.rs`. Reply of `GET /health/ready`. `reason` and `hard_latched_stores` appear only while a store's write latch is hard-set, with the 503 that goes with it.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `&'static str` | yes | `ready`, `not_ready` or `degraded`. |
| `version` | `&'static str` | yes |  |
| `effective_storage_backend` | `&'static str` | yes |  |
| `users_in_cache` | `usize` | yes |  |
| `tenant_open_waiters` | `usize` | yes | Requests waiting for a tenant store to open. |
| `timestamp` | `String` | yes | RFC 3339. |
| `reason` | `Option<&'static str>` | no | omitted when empty |
| `hard_latched_stores` | `Option<usize>` | no | omitted when empty |

### Reason

Defined in `src/entity_hygiene.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `uuid` | none |  |
| `runid` | none |  |
| `timestamp` | none |  |
| `json` | none |  |
| `session` | none |  |
| `lineref` | none |  |
| `counter` | none |  |
| `id` | none |  |
| `hex` | none |  |
| `url` | none |  |
| `ref` | none |  |
| `fragment` | none |  |
| `number` | none |  |
| `path` | none |  |
| `code-ident` | none |  |
| `filename` | none |  |
| `vocab` | none |  |

### RebuildGraphResponse

Defined in `src/handlers/graph.rs`. Reply of `POST /api/graph/{user_id}/rebuild`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `processed_memories` | `usize` | yes | Memories whose entities were extracted again. |
| `total_memories` | `usize` | yes |  |
| `entities_created` | `usize` | yes | Entities in the graph after the rebuild. |
| `relationships_created` | `usize` | yes | Relationships in the graph after the rebuild. |

### RebuildIndexRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### RebuildIndexResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `storage_count` | `usize` | yes |  |
| `indexed_count` | `usize` | yes |  |
| `is_healthy` | `bool` | yes |  |

### RecallByDateRequest

Defined in `src/handlers/recall.rs`. Recall memories by date range

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `start` | `chrono::DateTime<chrono::Utc>` | yes | Start of date range (inclusive) - ISO 8601 format |
| `end` | `chrono::DateTime<chrono::Utc>` | yes | End of date range (inclusive) - ISO 8601 format |
| `limit` | `Option<usize>` | no | Maximum number of results (default: 50) |

### RecallByTagsRequest

Defined in `src/handlers/recall.rs`. Recall memories by tags

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `tags` | `Vec<String>` | yes | Tags to search for (returns memories matching ANY of these tags) |
| `limit` | `Option<usize>` | no | Maximum number of results (default: 50) |
| `include_embeddings` | `bool` | no | default when absent; Both embedding vectors are about 17 KB per memory; a caller that reads text does not need them. |
| `include_derived` | `bool` | no | default when absent; Also match tags veld derived from the content (NER, keywords, ingest hints). Off by default: a keyword such as `handoff` or `decision` pulled out of the text would otherwise answer for the writer's own tag of the same name. |

### RecallEvent

Defined in `src/memory/recall_events.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `v` | `u32` | yes |  |
| `recall_id` | `String` | yes |  |
| `at` | `DateTime<Utc>` | yes |  |
| `endpoint` | `String` | yes |  |
| `query` | `String` | yes |  |
| `query_chars` | `usize` | yes | Length of the query as asked, before truncation to `RECALL_QUERY_MAX_CHARS`. |
| `query_redactions` | `usize` | yes | Tokens in the query replaced by a credential class label. |
| `episode_id` | `Option<String>` | no |  |
| `result_count` | `usize` | yes |  |
| `best_score` | `Option<f32>` | no | score |
| `hits` | Vec&lt;[`RecallHit`](#recallhit)&gt; | yes |  |

### RecallExperience

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `content` | `String` | yes |  |
| `memory_type` | `Option<String>` | no |  |
| `tags` | `Vec<String>` | yes |  |

### RecallFact

Defined in `src/handlers/types.rs`. Semantic fact returned in recall results

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `fact` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `support_count` | `usize` | yes |  |
| `related_entities` | `Vec<String>` | yes |  |

### RecallHit

Defined in `src/memory/recall_events.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory_id` | `String` | yes |  |
| `rank` | `usize` | yes |  |
| `score` | `f32` | yes |  |

### RecallLineageEdge

Defined in `src/handlers/types.rs`. Causal lineage edge returned in recall results

| Field | Type | Required | Notes |
|---|---|---|---|
| `from` | `String` | yes |  |
| `to` | `String` | yes |  |
| `relation` | `String` | yes |  |
| `confidence` | `f32` | yes | score |

### RecallLogQuery

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `since` | `Option<DateTime<Utc>>` | no |  |
| `limit` | `Option<usize>` | no |  |

### RecallLogReport

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `horizon` | `Option<DateTime<Utc>>` | no | Oldest retained event. Nothing before it can be answered. |
| `since` | `DateTime<Utc>` | yes |  |
| `total` | `usize` | yes |  |
| `misses` | `usize` | yes |  |
| `attributed` | `usize` | yes |  |
| `unattributed` | `usize` | yes |  |
| `by_endpoint` | `BTreeMap<String, usize>` | yes |  |
| `top_misses` | Vec&lt;[`MissedQuery`](#missedquery)&gt; | yes |  |
| `events` | Vec&lt;[`RecallEvent`](#recallevent)&gt; | yes | Newest first. |

### RecallMemory

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `experience` | [`RecallExperience`](#recallexperience) | yes |  |
| `importance` | `f32` | yes | score |
| `created_at` | `String` | yes |  |
| `score` | `f32` | yes |  |
| `tier` | `String` | yes |  |
| `competition_conflict` | `Option<bool>` | no | omitted when empty |
| `competition_opponent` | `Option<String>` | no | omitted when empty |
| `binding_strength` | `Option<f32>` | no | omitted when empty; R3 conjunctive-binding reliability, 0.0 (gist only) .. 1.0 (all W's bound). None when the engram carries no rich context. Low values mean the W-conjunction is weak — treat as gist, not a confident bound episode. |
| `source_type` | `Option<String>` | no | omitted when empty; C3 injection trust boundary: the engram's source tier (User/File/System = trusted; Web/ExternalApi/AiGenerated/Inferred = untrusted). Surfaced so a consumer treats untrusted-source content as data, never as instructions. |
| `source_trusted` | `Option<bool>` | no | omitted when empty; C3: whether the source tier is trusted (derived from source_type). |
| `writer` | Option&lt;crate::memory::facets::[`AgentSession`](#agentsession)&gt; | no | omitted when empty; WHO WROTE THIS, so a reader can weigh a memory instead of inheriting it. |
| `injection_suspected` | `Option<bool>` | no | omitted when empty; C3.1 (W-006): `Some(true)` when this memory came from a non-trusted source AND its content matches prompt-injection-style directives. Best-effort defense-in-depth — a consumer should fence such content as data, never instructions. Absent when there is no suspicion (the common case), so it adds nothing to the wire for clean recalls. |

### RecallReminder

Defined in `src/handlers/types.rs`. Reminder/prospective task returned in recall results

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `keywords` | `Vec<String>` | yes | Keywords that triggered this reminder (from OnContext trigger) |
| `match_type` | `String` | yes | How the reminder was matched (keyword_match, semantic_match) |
| `priority` | `u8` | yes |  |
| `created_at` | `String` | yes |  |

### RecallRequest

Defined in `src/handlers/types.rs`. Simplified recall request - just query text

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |
| `mode` | `String` | no | default when absent; Retrieval mode: "semantic", "associative", or "hybrid" (default) |
| `rrf_k` | `Option<f32>` | no | default when absent; RRF k parameter for reciprocal rank fusion (default: 20.0) Higher k = more equal weighting; lower k = sharper top-rank discrimination |
| `rerank_count` | `Option<usize>` | no | default when absent; Number of top results to rerank with cross-encoder (FIX-11, default: 20) |
| `competition_mode` | Option&lt;crate::memory::[`CompetitionMode`](#competitionmode)&gt; | no | default when absent; Optional retrieval-time competition policy override. |
| `as_of` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent; R6 bi-temporal as-of: return memories valid AS OF this instant (point-in-time query) instead of the default "currently valid" filter. |
| `tags` | `Option<Vec<String>>` | no | default when absent; Filter by tags (any match). Ported from shodh-memory `5aa1d5d`: the field was absent here, so serde dropped it and a tagged recall returned untagged records while looking filtered. MemoryQuery already enforced the filter; only the wiring was missing. |
| `repo` | `Option<String>` | no | default when absent; Return only memories scoped to this repository slug (ASCII case-insensitive). Todos, reminders and facts are not scoped. |
| `episode_id` | `Option<String>` | no | default when absent; The asking session's transcript id, recorded on the recall event log. |
| `layers` | `Option<String>` | no | default when absent; Pipeline rung for per-layer attribution: "vamana_only", "+spreading", "+bm25", "+rerank", "+facts" or "full". Omitted means full, the production pipeline. |

### RecallResponse

Defined in `src/handlers/types.rs`. Simplified recall response - returns just text snippets

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | Vec&lt;[`RecallMemory`](#recallmemory)&gt; | yes |  |
| `count` | `usize` | yes |  |
| `retrieval_stats` | Option&lt;crate::memory::types::[`RetrievalStats`](#retrievalstats)&gt; | no | omitted when empty; Retrieval statistics (for observability) |
| `todos` | Vec&lt;[`RecallTodo`](#recalltodo)&gt; | no | default when absent; omitted when empty; Related todos found via semantic search |
| `todo_count` | `Option<usize>` | no | omitted when empty; Number of todos found |
| `facts` | Vec&lt;[`RecallFact`](#recallfact)&gt; | no | default when absent; omitted when empty; Related semantic facts extracted from memories |
| `fact_count` | `Option<usize>` | no | omitted when empty; Number of facts found |
| `triggered_reminders` | Vec&lt;[`RecallReminder`](#recallreminder)&gt; | no | default when absent; omitted when empty; Triggered reminders (future intentions that influenced retrieval) These are prospective tasks whose context matched the query |
| `reminder_count` | `Option<usize>` | no | omitted when empty; Number of triggered reminders |
| `lineage` | Vec&lt;[`RecallLineageEdge`](#recalllineageedge)&gt; | no | default when absent; omitted when empty; Causal lineage edges connecting recalled memories |
| `lineage_count` | `Option<usize>` | no | omitted when empty; Number of lineage edges found |
| `tokens_estimated` | `Option<usize>` | no | omitted when empty; Estimated token cost of the response body (chars / 4 heuristic, matches OpenAI's rough English ratio). Published for parity with Mem0's token-efficiency benchmarks. |
| `metacognition` | Option&lt;crate::memory::types::[`QueryMetacognition`](#querymetacognition)&gt; | no | omitted when empty; Query-level metacognition (M3): a returned feeling-of-knowing scalar + label. Advisory; a terminal readout never fed into learning. |
| `derived_answer` | Option&lt;crate::handlers::temporal_derivation::[`DerivedAnswer`](#derivedanswer)&gt; | no | omitted when empty; B4 temporal answer-derivation. `None` (and omitted from JSON) when no temporal answer is derivable — response is then byte-identical to the pre-B4 shape. Never affects `memories`/`count`. |

### RecallTodo

Defined in `src/handlers/types.rs`. Todo returned in recall results

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `short_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `status` | `String` | yes |  |
| `priority` | `String` | yes |  |
| `project` | `Option<String>` | no |  |
| `due_date` | `Option<String>` | no |  |
| `score` | `f32` | yes |  |

### ReconcileReport

Defined in `src/memory/embedding_backlog.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `stored` | `usize` | yes |  |
| `indexed` | `usize` | yes |  |
| `enqueued` | `usize` | yes |  |
| `dropped_indexed` | `usize` | yes |  |
| `dropped_gone` | `usize` | yes |  |
| `pending` | `usize` | yes |  |

### RecordClickRequest

Defined in `src/handlers/ab_testing.rs`. Request to record a click

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memory_id` | `uuid::Uuid` | yes |  |

### RecordFeedbackRequest

Defined in `src/handlers/ab_testing.rs`. Request to record feedback

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `positive` | `bool` | yes |  |

### RecordImpressionRequest

Defined in `src/handlers/ab_testing.rs`. Request to record an impression

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `relevance_score` | `Option<f64>` | no | default when absent |
| `latency_us` | `Option<u64>` | no | default when absent |

### RecordKind

Defined in `src/memory/facets.rs`. The kind of record a core row carries.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `memory` | none | A remembered experience — the default; everything Veld stores today. |
| `plan` | none | A plan: a goal plus ordered steps with a status lifecycle. |
| `prompt` | none | A reusable prompt with provenance. |
| `learning` | none | A distilled insight derived from one or more memories. |

### RecordsQuery

Defined in `src/handlers/replication.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `after_lsn` | `Option<u64>` | no | default when absent |
| `limit` | `Option<usize>` | no |  |

### RecoverRequest

Defined in `src/handlers/user_auth.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |
| `recovery_code` | `String` | yes |  |
| `new_password` | `String` | yes |  |

### Recurrence

Defined in `src/memory/types.rs`. Recurrence pattern for repeating todos

Enum, internally tagged: the payload's fields plus `"type": <variant>`.

| Variant | Payload | Notes |
|---|---|---|
| `daily` | none | Every day |
| `weekly` | `{ days: Vec<u8> }` | Specific days of week (0=Sun, 6=Sat) |
| `monthly` | `{ day: u8 }` | Specific day of month (1-31) |
| `every_n_days` | `{ n: u32 }` | Every N days |

### RecurrencePattern

Defined in `src/memory/facets.rs`. A detected recurrence pattern. Free-form for now; structured RRULE-style fields can be added later without breaking existing data.

| Field | Type | Required | Notes |
|---|---|---|---|
| `period` | `Option<String>` | no | e.g. `"weekly"`, `"daily"`, `"monthly"`. |
| `description` | `Option<String>` | no | Free-form description of the recurrence rule. |

### ReembedResponse

Defined in `src/handlers/consolidation.rs`. Reply of `POST /api/index/reembed`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `reembedded` | `usize` | yes | Memories embedded again. |
| `failed` | `usize` | yes | Memories whose embedding failed; they keep their previous vector. |

### RefinerMode

Defined in `src/memory/hybrid_search.rs`. Final-stage refiner selection for hybrid search.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `none` | none | Skip both cross-encoder and refiner. Returns raw RRF scores. |
| `cross_encoder` | none | Apply only the cross-encoder rerank (current production default). |
| `llm` | none | Apply only the LLM refiner; skip the cross-encoder. |
| `stacked` | none | Cross-encoder first, then LLM refiner over its output. |

### RegisterRequest

Defined in `src/handlers/user_auth.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |
| `password` | `String` | yes |  |

### ReinforceFeedbackRequest

Defined in `src/handlers/types.rs`. Request to provide feedback on retrieval outcome

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `ids` | `Vec<String>` | yes | Memory IDs to reinforce |
| `outcome` | `String` | yes | "helpful", "misleading", or "neutral" |

### ReinforceFeedbackResponse

Defined in `src/handlers/recall.rs`. Response from reinforcement

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_processed` | `usize` | yes |  |
| `associations_strengthened` | `usize` | yes |  |
| `importance_boosts` | `usize` | yes |  |
| `importance_decays` | `usize` | yes |  |

### RelationType

Defined in `src/graph_memory.rs`. Relationship types for ontological edge classification.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `WorksWith` | none | Work relationships |
| `WorksAt` | none |  |
| `EmployedBy` | none |  |
| `PartOf` | none | Structural relationships |
| `Contains` | none |  |
| `OwnedBy` | none |  |
| `LocatedIn` | none | Location relationships |
| `LocatedAt` | none |  |
| `Uses` | none | Usage relationships |
| `CreatedBy` | none |  |
| `DevelopedBy` | none |  |
| `Causes` | none | Causal relationships |
| `ResultsIn` | none |  |
| `Learned` | none | Learning relationships |
| `Knows` | none |  |
| `Teaches` | none |  |
| `RelatedTo` | none | Generic relationships (never penalized by ontological filtering) |
| `AssociatedWith` | none |  |
| `CoRetrieved` | none | Memory co-retrieval (Hebbian association between memories) |
| `CoOccurs` | none | Sentence co-occurrence (entities appearing in same sentence) Key for multi-hop: "Melanie" <-> "sunrise" when "Melanie painted a sunrise" |
| `Manages` | none | Management / Governance Person/Team manages Project/Service/Team |
| `DependsOn` | none | Service/Module depends on Service/Module |
| `Requires` | none | Task requires Skill/Technology |
| `Configures` | none | Configuration configures Service/Environment |
| `Prefers` | none | Preference / Recommendation Person prefers Technology/Configuration/Approach |
| `Recommends` | none | Person/Role recommends Technology/Approach |
| `Documents` | none | Document / Knowledge Document documents Service/API/Project |
| `Implements` | none | Module/Service implements Concept/Spec |
| `DeploysTo` | none | Operational Pipeline deploys Service to Environment |
| `Monitors` | none | Metric/Service monitors Service/Pipeline |
| `Triggers` | none | Event/Pipeline triggers Pipeline/Task |
| `SupersededBy` | none | Comparison / Evolution Technology/Config superseded by newer version |
| `AlternativeTo` | none | Technology is alternative to Technology |
| `AssignedTo` | none | Assignment Task assigned to Person/Team |
| `Approves` | none | Person/Role approves Task/Document |
| `Custom` | `String` | Custom relationship |

### RelationalPredicate

Defined in `src/query_planner/predicate.rs`. Relational filters expressed as column-level predicates.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `Equals` | `{ column: String, value: serde_json::Value }` |  |
| `In` | `{ column: String, values: Vec<serde_json::Value> }` |  |
| `Range` | `{ column: String, lo: Option<serde_json::Value>, hi: Option<serde_json::Value> }` |  |
| `UserIdEquals` | `String` |  |

### RelationshipEdge

Defined in `src/graph_memory.rs`. Relationship edge between entities

| Field | Type | Required | Notes |
|---|---|---|---|
| `uuid` | `Uuid` | yes | Unique identifier for this edge |
| `from_entity` | `Uuid` | yes | Source entity UUID |
| `to_entity` | `Uuid` | yes | Target entity UUID |
| `relation_type` | [`RelationType`](#relationtype) | yes | Type of relationship |
| `strength` | `f32` | yes | Confidence/strength of this relationship (0.0 to 1.0) Dynamic: increases with co-activation, decays without use |
| `created_at` | `DateTime<Utc>` | yes | When this relationship was created |
| `valid_at` | `DateTime<Utc>` | yes | When this relationship was last observed (temporal tracking) |
| `invalidated_at` | `Option<DateTime<Utc>>` | no | Whether this relationship has been invalidated (temporal edge invalidation) |
| `source_episode_id` | `Option<Uuid>` | no | Source episode that created this relationship |
| `context` | `String` | yes | Additional context about the relationship |
| `last_activated` | `DateTime<Utc>` | no | default when absent; When this synapse was last activated (used in retrieval/traversal) Used to calculate time-based decay |
| `activation_count` | `u32` | no | default when absent; Number of times both entities were co-accessed (Hebbian co-activation) Higher count = stronger learned association |
| `ltp_status` | [`LtpStatus`](#ltpstatus) | no | default when absent; Long-Term Potentiation status (PIPE-4: multi-scale LTP) Replaces simple bool with tiered protection levels: - None: Normal decay - Burst: Temporary 2x protection (5+ activations in 24h) - Weekly: Moderate 3x protection (3+/week for 2 weeks) - Full: Maximum 10x protection (10+ activations or 5+ over 30 days) |
| `tier` | [`EdgeTier`](#edgetier) | no | default when absent; Memory tier for consolidation (L1→L2→L3) Edges start in L1 (working memory) and promote based on Hebbian strength |
| `activation_timestamps` | `Option<VecDeque<DateTime<Utc>>>` | no | default when absent; Activation timestamp history for temporal pattern detection (PIPE-4) Only populated for L2+ edges (L1 edges die too quickly to need history) Capacity: L2 = 20 timestamps, L3 = 50 timestamps Enables: burst detection, weekly patterns, temporal query relevance |
| `entity_confidence` | `Option<f32>` | no | default when absent; score; Entity extraction confidence (PIPE-5: Unified LTP Readiness) Average confidence of the entities connected by this edge. Affects LTP threshold: high confidence → faster LTP (7 activations) Low confidence → slower LTP (13 activations). Based on synaptic tagging: behaviorally relevant synapses consolidate faster. |
| `created_by` | [`EdgeSource`](#edgesource) | no | default when absent; Creation source for this edge (FIX-06: dream replay tagging) Tracks whether this edge was created from direct co-occurrence, dream replay, Hebbian coactivation, or explicit user input. Enables retrieval to distinguish speculative from empirical associations. |
| `forward_strength` | `f32` | no | default when absent; Forward activation strength (from_entity triggered, to_entity co-retrieved) Increases when from_entity is the query match and to_entity is co-retrieved, once per strengthening source, as `strength` does. |
| `backward_strength` | `f32` | no | default when absent; Backward activation strength (to_entity triggered, from_entity co-retrieved) Increases when to_entity is the query match and from_entity is co-retrieved, once per strengthening source, as `strength` does. |
| `strengthening_sources` | `BTreeMap<String, u32>` | no | default when absent; Distinct sources (session, tool, ingest) that have strengthened this edge, each with its event count. A repeat from a recorded source refreshes `last_activated` and nothing else; a new source strengthens. |
| `last_corroborated` | `Option<DateTime<Utc>>` | no | default when absent; When a source not yet in `strengthening_sources` last strengthened this edge. |

### RelationshipType

Defined in `src/memory/types.rs`. Types of relationships

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `IsA` | none |  |
| `HasA` | none |  |
| `Uses` | none |  |
| `RelatedTo` | none |  |
| `Causes` | none |  |
| `PartOf` | none |  |
| `Similar` | none |  |
| `Opposite` | none |  |

### RelevanceConfig

Defined in `src/relevance.rs`. Configuration for relevance triggers

| Field | Type | Required | Notes |
|---|---|---|---|
| `semantic_threshold` | `f32` | no | default when absent; Minimum semantic similarity score (0.0-1.0) to consider a memory relevant |
| `entity_threshold` | `f32` | no | default when absent; Minimum entity match score (0.0-1.0) to trigger entity-based surfacing |
| `max_results` | `usize` | no | default when absent; Maximum number of memories to surface |
| `memory_types` | `Vec<String>` | no | default when absent; Memory types to include (empty = all types) |
| `enable_entity_matching` | `bool` | no | default when absent; Whether to include entity-based matching |
| `enable_semantic_matching` | `bool` | no | default when absent; Whether to include semantic similarity matching |
| `min_importance` | `f32` | no | default when absent; score; Minimum importance score for memories to be surfaced |
| `recency_boost_hours` | `u64` | no | default when absent; Time window for recency boost (memories within this window get boosted) |
| `recency_boost_multiplier` | `f32` | no | default when absent; Recency boost multiplier (1.0 = no boost) |
| `graph_boost_multiplier` | `f32` | no | default when absent; Graph connectivity boost multiplier (FBK-7) Applied to memories that have graph relationships with detected entities |

### RelevanceDebug

Defined in `src/relevance.rs`. Debug information for relevance calculation

| Field | Type | Required | Notes |
|---|---|---|---|
| `ner_ms` | `f64` | yes | Time spent on NER |
| `entity_match_ms` | `f64` | yes | Time spent on entity matching |
| `semantic_search_ms` | `f64` | yes | Time spent on semantic search |
| `ranking_ms` | `f64` | yes | Time spent on scoring and ranking |
| `memories_scanned` | `usize` | yes | Number of memories scanned |
| `entity_matches` | `usize` | yes | Number of entity matches found |
| `semantic_matches` | `usize` | yes | Number of semantic matches found |

### RelevanceReason

Defined in `src/relevance.rs`. Reason why a memory was surfaced

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `EntityMatch` | none | Matched one or more entities in the context |
| `SemanticSimilarity` | none | High semantic similarity to current context |
| `Combined` | none | Both entity match and semantic similarity |
| `RecentImportant` | none | Recent and important (fallback when no strong matches) |

### RelevanceRequest

Defined in `src/relevance.rs`. Request for proactive memory surfacing

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes | User ID for memory lookup |
| `context` | `String` | yes | Current context text (conversation turn, user message, etc.) |
| `episode_id` | `Option<String>` | no | default when absent; The asking session's transcript id, recorded on the recall event log. |
| `entities` | `Vec<String>` | no | default when absent; Optional pre-extracted entities (skip NER if provided) |
| `config` | [`RelevanceConfig`](#relevanceconfig) | no | default when absent; Configuration overrides |

### RelevanceResponse

Defined in `src/relevance.rs`. Response from proactive memory surfacing

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | Vec&lt;[`SurfacedMemory`](#surfacedmemory)&gt; | yes | Surfaced memories ordered by relevance |
| `detected_entities` | Vec&lt;[`DetectedEntity`](#detectedentity)&gt; | yes | Entities detected in the context |
| `latency_ms` | `f64` | yes | Processing time in milliseconds |
| `latency_target_met` | `bool` | yes | Whether latency target (<30ms) was met |
| `debug` | Option&lt;[`RelevanceDebug`](#relevancedebug)&gt; | no | omitted when empty; Debug info (only in debug builds) |

### RememberRequest

Defined in `src/handlers/remember.rs`. Remember request - store a new memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `tags` | `Vec<String>` | no | default when absent |
| `memory_type` | `Option<String>` | no | default when absent |
| `external_id` | `Option<String>` | no | default when absent |
| `created_at` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent |
| `emotional_valence` | `Option<f32>` | no | default when absent |
| `emotional_arousal` | `Option<f32>` | no | default when absent |
| `emotion` | `Option<String>` | no | default when absent |
| `source_type` | `Option<String>` | no | default when absent |
| `credibility` | `Option<f32>` | no | default when absent |
| `episode_id` | `Option<String>` | no | default when absent |
| `sequence_number` | `Option<u32>` | no | default when absent |
| `preceding_memory_id` | `Option<String>` | no | default when absent |
| `agent_id` | `Option<String>` | no | default when absent |
| `parent_agent_id` | `Option<String>` | no | default when absent |
| `run_id` | `Option<String>` | no | default when absent |
| `parent_id` | `Option<String>` | no | default when absent; Parent memory ID for hierarchical organization Use this to create memory trees (e.g., "71-research" -> "algebraic" -> "21×27≡-1") |
| `filename` | `Option<String>` | no | default when absent; Optional filename for format detection (e.g. "notes.md", "data.json"). When provided, the content is run through the ingest text-extraction pipeline before storage. |
| `intent_tags` | `Vec<String>` | no | default when absent; Semantic intent tags from Claude (FIX-10) Merged into entity tags with "intent:" prefix for importance boost |
| `writer` | Option&lt;crate::memory::facets::[`DeclaredWriter`](#declaredwriter)&gt; | no | default when absent; WHO IS WRITING THIS, as the caller describes itself — session, model, transcript, workspace. Optional, and recorded verbatim under `session.declared`; nothing here is verified and nothing here grants anything. See [`crate::memory::facets::DeclaredWriter`] for why it has to come from the caller: the server's own session detection describes the server. |
| `entities` | Vec&lt;[`DeclaredEntityInput`](#declaredentityinput)&gt; | no | default when absent; Typed entities the writer declares this memory is about. See [`crate::declared_entities`]: validated (400 on an unknown label, a weight outside [0, 1], or more than 50), exempt from the derived-entity cut, and stored as authored tags. |
| `repo` | `Option<String>` | no | default when absent; Repository slug this memory belongs to, e.g. `"Portll/veld"`. Stored as the leading [`crate::memory::facets::Place::Repo`]; a recall with `repo` set returns only memories scoped to it. |

### RememberResponse

Defined in `src/handlers/remember.rs`. Remember response

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `success` | `bool` | yes |  |

### ReminderActionResponse

Defined in `src/handlers/todos.rs`. Response for dismiss/delete operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `String` | yes |  |

### ReminderItem

Defined in `src/handlers/todos.rs`. Individual reminder in list response

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `trigger_type` | `String` | yes |  |
| `status` | `String` | yes |  |
| `due_at` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `created_at` | `chrono::DateTime<chrono::Utc>` | yes |  |
| `triggered_at` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `dismissed_at` | `Option<chrono::DateTime<chrono::Utc>>` | no |  |
| `priority` | `u8` | yes |  |
| `tags` | `Vec<String>` | yes |  |
| `overdue_seconds` | `Option<i64>` | no |  |

### ReminderTriggerRequest

Defined in `src/handlers/todos.rs`. Trigger configuration for reminder creation

Enum, internally tagged: the payload's fields plus `"type": <variant>`.

| Variant | Payload | Notes |
|---|---|---|
| `time` | `{ at: chrono::DateTime<chrono::Utc> }` |  |
| `duration` | `{ after_seconds: u64 }` |  |
| `context` | `{ keywords: Vec<String>, threshold: f32 }` |  |

### ReorderTodoRequest

Defined in `src/handlers/todos.rs`. Request to reorder a todo

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `direction` | `String` | yes |  |
| `todo_id` | `Option<String>` | no | default when absent; The target todo, named in the body by the flat `/api/todos/reorder` alias. |

### RepairIndexRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### RepairIndexResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `total_storage` | `usize` | yes |  |
| `total_indexed` | `usize` | yes |  |
| `repaired` | `usize` | yes |  |
| `failed` | `usize` | yes |  |
| `is_healthy` | `bool` | yes |  |

### ReplayEvent

Defined in `src/memory/introspection.rs`. Replay event details (SHO-105)

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory_id` | `String` | yes |  |
| `content_preview` | `String` | yes |  |
| `activation_before` | `f32` | yes |  |
| `activation_after` | `f32` | yes |  |
| `replay_priority` | `f32` | yes |  |
| `connected_memories` | `usize` | yes |  |
| `timestamp` | `DateTime<Utc>` | yes |  |

### ReportPeriod

Defined in `src/memory/introspection.rs`. Time period for a report

| Field | Type | Required | Notes |
|---|---|---|---|
| `start` | `DateTime<Utc>` | yes |  |
| `end` | `DateTime<Utc>` | yes |  |

### ResolvedEntity

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `canonical_name` | `String` | yes |  |
| `aliases` | `Vec<String>` | yes |  |
| `entity_type` | `String` | yes |  |
| `attributes` | `HashMap<String, String>` | yes |  |
| `mention_count` | `usize` | yes |  |
| `salience` | `f32` | yes |  |
| `is_proper_noun` | `bool` | yes |  |
| `related_facts` | `Vec<String>` | yes |  |

### RestoreBackupRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `backup_id` | `Option<u32>` | no | default when absent |
| `max_lsn` | `Option<u64>` | no | default when absent; Point-in-time-restore (PITR). When set, the restored intent log is truncated so the final surviving frame has `lsn == max_lsn`. Frames in the archive with higher LSNs are dropped on disk. No effect on the main RocksDB restore. |

### RestoreBackupResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `message` | `String` | yes |  |
| `restored_stores` | `Vec<String>` | yes |  |

### RetrievalStats

Defined in `src/memory/types.rs`. Retrieval statistics for SHO-26 associative retrieval (debugging/observability)

| Field | Type | Required | Notes |
|---|---|---|---|
| `mode` | `String` | yes | Mode used for this retrieval (semantic, associative, hybrid) |
| `semantic_candidates` | `usize` | yes | Number of memories retrieved via semantic similarity |
| `graph_candidates` | `usize` | yes | Number of memories retrieved via graph traversal |
| `graph_density` | `f32` | yes | Graph density (edges per memory) - determines graph weight |
| `graph_weight` | `f32` | yes | score; Actual graph weight used (density-dependent: 0.1 to 0.5) |
| `semantic_weight` | `f32` | yes | score; Actual semantic weight used (1.0 - graph_weight - linguistic_weight) |
| `linguistic_weight` | `f32` | yes | score; Linguistic weight used (fixed at 0.15) |
| `graph_hops` | `usize` | yes | Number of graph hops performed in spreading activation |
| `entities_activated` | `usize` | yes | Number of unique entities activated during graph traversal |
| `avg_salience_boost` | `f32` | yes | Average salience boost applied to initial activations (0.0 if no entities) Tracks ACT-R inspired salience weighting effectiveness |
| `retrieval_time_us` | `u64` | yes | Total time spent on retrieval (microseconds) |
| `embedding_time_us` | `u64` | yes | Time spent on embedding generation (microseconds) |
| `graph_time_us` | `u64` | yes | Time spent on graph traversal (microseconds) |
| `weight_profile` | `String` | no | default when absent; omitted when empty; Weight profile that was used: density, query_type:*, or default. |
| `semantic_variance_pct` | `f32` | no | default when absent; Average percentage of final score contributed by semantic signal (0.0-1.0). |
| `graph_variance_pct` | `f32` | no | default when absent; Average percentage of final score contributed by graph signal (0.0-1.0). |
| `linguistic_variance_pct` | `f32` | no | default when absent; Average percentage of final score contributed by linguistic signal (0.0-1.0). |
| `suppressor_count` | `usize` | no | default when absent; Number of suppressor events detected during reranking/competition. |
| `interference_detected` | `bool` | no | default when absent; omitted when empty; Whether interference was detected during this retrieval |
| `interference_events` | Vec&lt;[`InterferenceEvent`](#interferenceevent)&gt; | no | default when absent; omitted when empty; Memory IDs that triggered interference (retroactive or proactive) |
| `recommended_mode` | `Option<String>` | no | default when absent; omitted when empty; Recommended retrieval mode based on query analysis and graph state |
| `mode_rationale` | `Option<String>` | no | default when absent; omitted when empty; Rationale for the mode recommendation |
| `entity_resolution` | Vec&lt;[`EntityResolution`](#entityresolution)&gt; | no | default when absent; omitted when empty; Entity lookup results: which query entities were found/not found in graph |

### RetrieveResponse

Defined in `src/handlers/types.rs`. Response for list/search operations returning multiple memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories` | `Vec<serde_json::Value>` | yes |  |
| `count` | `usize` | yes |  |

### RevokeBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `unlock` | [`UnlockBody`](#unlockbody) | yes |  |
| `id` | `String` | yes |  |

### RevokeSessionBody

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |
| `session_id` | `String` | yes |  |

### RichContext

Defined in `src/memory/types.rs`. Rich multi-dimensional context

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`ContextId`](#contextid) | yes |  |
| `conversation` | [`ConversationContext`](#conversationcontext) | yes | Conversation context - what's being discussed |
| `user` | [`UserContext`](#usercontext) | yes | User context - who the user is, their patterns |
| `project` | [`ProjectContext`](#projectcontext) | yes | Project context - what they're working on |
| `temporal` | [`TemporalContext`](#temporalcontext) | yes | Temporal context - when and patterns over time |
| `semantic` | [`SemanticContext`](#semanticcontext) | yes | Semantic context - relationships and meaning |
| `code` | [`CodeContext`](#codecontext) | yes | Code context - related code elements |
| `document` | [`DocumentContext`](#documentcontext) | yes | Document context - related documents |
| `environment` | [`EnvironmentContext`](#environmentcontext) | yes | Environment context - system state, location, etc |
| `emotional` | [`EmotionalContext`](#emotionalcontext) | no | default when absent; Emotional context - affective state during memory formation (SHO-104) Captures valence, arousal, and dominant emotion for emotional memory enhancement |
| `source` | [`SourceContext`](#sourcecontext) | no | default when absent; Source context - tracks where information came from (SHO-104) Enables source monitoring for memory accuracy and credibility weighting |
| `episode` | [`EpisodeContext`](#episodecontext) | no | default when absent; Episode context - groups memories into coherent episodes (SHO-104) Enables temporal ordering and event segmentation within conversations |
| `place` | [`WhereFacet`](#wherefacet) | no | default when absent; Spatial / locational facet — the WHERE of an engram (W3, layered `Place` model). Absorbs the first-pass `RepositoryContext` as `Place::Repo`. Older serialized contexts deserialize with an empty default. See `docs/neuroscience-5w-memory-design.md`. |
| `who` | [`WhoFacet`](#whofacet) | no | default when absent; Provenance + agent identity — the WHO of an engram (W3.3 facet). |
| `why` | [`WhyFacet`](#whyfacet) | no | default when absent; Goals + typed causal links + event-model — the WHY of an engram (W3.3). |
| `binding` | [`EngramBinding`](#engrambinding) | no | default when absent; Conjunctive binding — cross-W presence + binding strength (W3.3). |
| `session` | [`AgentSession`](#agentsession) | no | default when absent; Agent-session identity — which agent recorded this engram, from which worktree/branch, when. Source-of-truth for the worktree-per-agent topology and the planned branch-aware tooling (worktree viewer, branch-scoped recall). Defaults to empty so pre-W3.5 records deserialize cleanly. |
| `parent` | Option&lt;Box&lt;[`RichContext`](#richcontext)&gt;&gt; | no | Parent context (for hierarchical context) |
| `embeddings` | `Option<Vec<f32>>` | no | Context embeddings for similarity search |
| `decay_rate` | `f32` | yes | Context decay factor (how relevant this context is over time) |
| `created_at` | `DateTime<Utc>` | yes | Created timestamp |
| `updated_at` | `DateTime<Utc>` | yes | Last updated |

### RoboticsSearchRequest

Defined in `src/handlers/search.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `mode` | `String` | yes |  |
| `query_text` | `Option<String>` | no |  |
| `robot_id` | `Option<String>` | no |  |
| `mission_id` | `Option<String>` | no |  |
| `lat` | `Option<f64>` | no |  |
| `lon` | `Option<f64>` | no |  |
| `radius_meters` | `Option<f64>` | no |  |
| `action_type` | `Option<String>` | no |  |
| `min_reward` | `Option<f32>` | no |  |
| `max_reward` | `Option<f32>` | no |  |
| `limit` | `Option<usize>` | no |  |

### RotatePassphraseBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `old` | `String` | yes |  |
| `new` | `String` | yes |  |

### RowPk

Defined in `src/datasets/link.rs`. Concrete primary-key value tuple identifying a single row.

| Field | Type | Required | Notes |
|---|---|---|---|
| `values` | `Vec<serde_json::Value>` | yes |  |

### ScanResponse

Defined in `src/handlers/files.rs`. Response for scan operation

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `total_files` | `usize` | yes |  |
| `eligible_files` | `usize` | yes |  |
| `skipped_files` | `usize` | yes |  |
| `limit_reached` | `bool` | yes |  |
| `message` | `String` | yes |  |

### ScopeReport

Defined in `src/memory/gap_topology/scope.rs`. The scope a result covers.

| Field | Type | Required | Notes |
|---|---|---|---|
| `namespace` | `Option<String>` | no |  |
| `project` | `Option<String>` | no |  |
| `entity_types` | `Vec<String>` | yes |  |
| `max_nodes` | `Option<usize>` | no |  |
| `nodes_in_scope` | `usize` | yes | Entities the filters matched; every mirrored entity without filters. |
| `nodes_analyzed` | `usize` | yes | Entities the analysis ran over. |
| `sampled` | `bool` | yes | Fewer entities were analyzed than matched. |
| `sample_method` | `Option<&'static str>` | no | How the sample was drawn, when `sampled`. |

### ScoredMemoryIdWire

Defined in `src/handlers/query.rs`. One scored result row. Mirrors the executor's `ScoredMemoryId` as a serialisable wire type so the planner crate stays free of HTTP concerns.

| Field | Type | Required | Notes |
|---|---|---|---|
| `memory_id` | `String` | yes |  |
| `score` | `f32` | yes |  |

### SearchFilesRequest

Defined in `src/handlers/files.rs`. Request to search files

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |

### SecondaryIndexSummary

Defined in `src/handlers/health.rs`. The secondary index, as `GET /api/health/index` summarises it.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_vectors` | `usize` | yes |  |
| `needs_rebuild` | `bool` | yes |  |
| `deletion_ratio` | `f32` | yes |  |

### SeedRequest

Defined in `src/handlers/seed.rs`. Seed project request

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `project_path` | `String` | yes |  |
| `max_files` | `Option<usize>` | no | default when absent |

### SeedResponse

Defined in `src/handlers/seed.rs`. Seed project response

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_created` | `usize` | yes |  |
| `files_scanned` | `usize` | yes |  |
| `project_name` | `String` | yes |  |

### SemanticContext

Defined in `src/memory/types.rs`. Semantic context

| Field | Type | Required | Notes |
|---|---|---|---|
| `concepts` | `Vec<String>` | yes | Main concepts/topics |
| `related_concepts` | `Vec<String>` | yes | Related concepts |
| `relationships` | Vec&lt;[`ConceptRelationship`](#conceptrelationship)&gt; | yes | Concept relationships |
| `domain` | `Option<String>` | no | Domain/field |
| `abstraction_level` | `Option<String>` | no | Abstraction level (high-level vs detailed) |
| `tags` | `Vec<String>` | yes | Semantic tags |

### SemanticFact

Defined in `src/memory/compression.rs`. A semantic fact extracted from episodic memories

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Unique identifier |
| `fact` | `String` | yes | The factual statement |
| `confidence` | `f32` | yes | score; Confidence in this fact (0.0 - 1.0) |
| `support_count` | `usize` | yes | How many episodic memories support this fact |
| `source_memories` | Vec&lt;[`MemoryId`](#memoryid)&gt; | yes | Source memory IDs that contributed to this fact |
| `related_entities` | `Vec<String>` | yes | Keywords/entities this fact relates to |
| `created_at` | `chrono::DateTime<chrono::Utc>` | yes | When this fact was first extracted |
| `last_reinforced` | `chrono::DateTime<chrono::Utc>` | yes | When this fact was last reinforced |
| `fact_type` | [`FactType`](#facttype) | yes | Category of fact (preference, capability, relationship, procedure) |
| `valid_from` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent; When this fact became true in the world. `None` is treated as `created_at` by the store at first read. Set explicitly when the observation has a knowable start time. |
| `valid_until` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent; When this fact stopped being true. `None` = currently valid. Set by `detect_and_resolve_contradictions` when a newer contradicting fact supersedes this one. |
| `superseded_by` | `Option<String>` | no | default when absent; ID of the fact that superseded this one (back-pointer for audit). Only meaningful when `valid_until` is `Some`. |
| `supersedes` | `Vec<String>` | no | default when absent; IDs of facts that this fact replaced (forward-pointer for audit). Populated on a *new* fact when it invalidates one or more older facts. |
| `purged_at` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent; When this fact was administratively purged. `None` = never purged. |
| `purge_reason` | Option&lt;[`PurgeReason`](#purgereason)&gt; | no | default when absent; Why this fact was purged. `None` when `purged_at` is `None`. |

### Session

Defined in `src/memory/sessions.rs`. A user session

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`SessionId`](#sessionid) | yes | Session ID |
| `user_id` | `String` | yes | User ID |
| `status` | [`SessionStatus`](#sessionstatus) | yes | Session status |
| `started_at` | `DateTime<Utc>` | yes | When session started |
| `ended_at` | `Option<DateTime<Utc>>` | no | When session ended (if completed/abandoned) |
| `duration_secs` | `Option<i64>` | no | Session duration in seconds |
| `temporal` | [`SessionTimeContext`](#sessiontimecontext) | yes | Temporal context (time of day, relative date, etc.) |
| `stats` | [`SessionStats`](#sessionstats) | yes | Session statistics |
| `timeline` | Vec&lt;[`SessionEvent`](#sessionevent)&gt; | yes | Timeline of events |
| `label` | `Option<String>` | no | Optional session name/label (user-provided) |
| `metadata` | `HashMap<String, serde_json::Value>` | yes | Metadata |

### SessionEvent

Defined in `src/memory/sessions.rs`. Event types that can occur during a session

Enum, internally tagged: the payload's fields plus `"type": <variant>`.

| Variant | Payload | Notes |
|---|---|---|
| `session_start` | `{ timestamp: DateTime<Utc> }` | Session started |
| `memory_created` | `{ timestamp: DateTime<Utc>, memory_id: String, memory_type: String, content_preview: String, entities: Vec<String> }` | Memory was created |
| `memories_surfaced` | `{ timestamp: DateTime<Utc>, query_preview: String, memory_count: usize, memory_ids: Vec<String>, avg_score: f32 }` | Memories were surfaced for a query |
| `memory_used` | `{ timestamp: DateTime<Utc>, memory_id: String, derived_ratio: f32 }` | Memory was used in response (entity flow detected usage) |
| `todo_created` | `{ timestamp: DateTime<Utc>, todo_id: String, content: String, project: Option<String> }` | Todo was created |
| `todo_completed` | `{ timestamp: DateTime<Utc>, todo_id: String }` | Todo was completed |
| `topic_change` | `{ timestamp: DateTime<Utc>, similarity: f32 }` | Topic changed (detected via low similarity) |
| `query_processed` | `{ timestamp: DateTime<Utc>, query_preview: String, tokens_estimated: usize }` | Query was processed |
| `session_end` | `{ timestamp: DateTime<Utc>, reason: String }` | Session ended |

### SessionId

Defined in `src/memory/sessions.rs`. Unique session identifier

On the wire this is its inner value, `Uuid`.

### SessionStats

Defined in `src/memory/sessions.rs`. Session statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `memories_created` | `usize` | yes | Number of memories created |
| `memories_surfaced` | `usize` | yes | Number of memories surfaced |
| `memories_used` | `usize` | yes | Number of memories actually used (derived_ratio > 0) |
| `memory_hit_rate` | `f32` | yes | Memory hit rate (used / surfaced) |
| `todos_created` | `usize` | yes | Number of todos created |
| `todos_completed` | `usize` | yes | Number of todos completed |
| `todo_completion_rate` | `f32` | yes | Todo completion rate |
| `queries_count` | `usize` | yes | Number of queries processed |
| `tokens_estimated` | `usize` | yes | Estimated tokens used |
| `topic_changes` | `usize` | yes | Number of topic changes |

### SessionStatus

Defined in `src/memory/sessions.rs`. Session status

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `active` | none | Currently active session |
| `completed` | none | Session completed normally |
| `abandoned` | none | Session timed out or abandoned |

### SessionStoreStats

Defined in `src/memory/sessions.rs`. Session store statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `active_sessions` | `usize` | yes |  |
| `completed_sessions` | `usize` | yes |  |
| `users_with_sessions` | `usize` | yes |  |

### SessionStoreStatsResponse

Defined in `src/handlers/sessions.rs`. Response for session store stats

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `stats` | [`SessionStoreStats`](#sessionstorestats) | yes |  |

### SessionSummary

Defined in `src/memory/sessions.rs`. Lightweight session summary (without full timeline)

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`SessionId`](#sessionid) | yes |  |
| `user_id` | `String` | yes |  |
| `status` | [`SessionStatus`](#sessionstatus) | yes |  |
| `started_at` | `DateTime<Utc>` | yes |  |
| `ended_at` | `Option<DateTime<Utc>>` | no |  |
| `duration_secs` | `i64` | yes |  |
| `temporal` | [`SessionTimeContext`](#sessiontimecontext) | yes | Temporal context with human-readable labels |
| `label` | `Option<String>` | no | User-provided label |
| `stats` | [`SessionStats`](#sessionstats) | yes |  |

### SessionTimeContext

Defined in `src/memory/sessions.rs`. Temporal context for a session

| Field | Type | Required | Notes |
|---|---|---|---|
| `time_of_day` | [`TimeOfDay`](#timeofday) | yes | Time of day when session started |
| `day_of_week` | `u32` | yes | Day of week (Monday=0, Sunday=6) |
| `day_name` | `String` | yes | Day of week name |
| `month_name` | `String` | yes | Month name |
| `day` | `u32` | yes | Day of month |
| `year` | `i32` | yes | Year |
| `label` | `String` | yes | Human-readable label like "Morning session of Dec 20th" |
| `relative` | `String` | yes | Relative time label like "Today", "Yesterday", "Last week" |

### SessionsQuery

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |

### SetAttributeRequest

Defined in `src/handlers/prompt_gen.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity_name` | `String` | yes |  |
| `key` | `String` | yes |  |
| `value` | `String` | yes |  |

### SetAttributeResponse

Defined in `src/handlers/prompt_gen.rs`. Reply of `POST /api/entity/attribute`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | `String` | yes | The canonical entity name. |
| `attribute` | `String` | yes |  |
| `value` | `String` | yes |  |
| `status` | `String` | yes | Always `set`. |

### SetBlockRequest

Defined in `src/handlers/context_blocks.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `max_tokens` | `Option<usize>` | no |  |

### SetDialsRequest

Defined in `src/handlers/dials.rs`. Body for `PUT /api/dials` (full replace).

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `dials` | [`CognitiveDials`](#cognitivedials) | yes |  |

### ShapeSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `node_count` | `usize` | yes |  |
| `existing_edges` | `usize` | yes |  |
| `missing_edges` | `usize` | yes |  |
| `sparsity` | `f32` | yes |  |

### Shelf

Defined in `src/entity_hygiene.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `cast` | none |  |
| `vocabulary` | none |  |
| `noise` | none |  |

### SleepPhaseRequest

Defined in `src/handlers/types.rs`. Request to trigger sleep-phase consolidation

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `replay_multiplier` | `Option<usize>` | no | default when absent; Multiplier for replay batch size (default: 3x normal) |

### SleepPhaseResponse

Defined in `src/handlers/types.rs`. Response from sleep-phase consolidation

| Field | Type | Required | Notes |
|---|---|---|---|
| `accepted` | `bool` | yes |  |
| `message` | `String` | yes |  |

### SleightDimensionsRequest

Defined in `src/handlers/external_dimensions.rs`. Request body for POST /api/sleight/dimensions

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `density` | `f32` | yes | Entity density in the relevant region (0.0 = sparse, 1.0 = saturated) |
| `coherence` | `f32` | yes | Semantic coherence of neighbors (0.0 = unrelated, 1.0 = tight cluster) |
| `closure` | `f32` | yes | Fraction of potential triangles closed (0.0 = all open, 1.0 = fully closed) |
| `confidence` | `f32` | yes | score; Average edge confidence in the region (0.0 = weak, 1.0 = strong) |
| `isotropy` | `f32` | yes | Directional balance of knowledge (0.0 = anisotropic, 1.0 = isotropic) |

### SleightDimensionsResponse

Defined in `src/handlers/external_dimensions.rs`. Response body for POST /api/sleight/dimensions

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |

### SourceContext

Defined in `src/memory/types.rs`. Source context - tracks where information came from

| Field | Type | Required | Notes |
|---|---|---|---|
| `source_type` | [`SourceType`](#sourcetype) | no | default when absent; Type of source |
| `source_id` | `Option<String>` | no | default when absent; Specific source identifier E.g., "user:alice", "api:openai", "file:readme.md", "url:https://..." |
| `source_name` | `Option<String>` | no | default when absent; Human-readable source name |
| `credibility` | `f32` | no | default when absent; Credibility score (0.0 to 1.0) How reliable is this source? Affects retrieval ranking. - 1.0: Direct user input, verified facts - 0.8: Trusted documentation, official sources - 0.5: General web content, unverified - 0.3: Inferred/generated content |
| `verified` | `bool` | no | default when absent; Was this information verified/confirmed? |
| `source_chain` | `Vec<String>` | no | default when absent; Chain of sources (for information that was relayed) E.g., ["api:openai", "doc:react-docs", "user:alice"] |

### SourceType

Defined in `src/memory/types.rs`. Types of information sources

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `User` | none | Direct user input (highest credibility default) |
| `System` | none | System-generated (logs, errors, metrics) |
| `ExternalApi` | none | External API response |
| `File` | none | File content |
| `Web` | none | Web/URL content |
| `AiGenerated` | none | AI/LLM generated content |
| `Inferred` | none | Inferred by the system |
| `Unknown` | none | Unknown source |

### SplitMemoryResponse

Defined in `src/handlers/memory_ops.rs`. Reply of `POST /api/memory/split`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `parent_id` | `String` | yes | The memory that was split, as sent. |
| `child_ids` | `Vec<String>` | yes | One new memory per range, in range order. |

### SplitRequest

Defined in `src/handlers/memory_ops.rs`. Request to split one memory into several.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `id` | `String` | yes |  |
| `ranges` | `Vec<(usize, usize)>` | yes | `[start, end)` ranges as **character** indices into the content (W-021 — never byte offsets). |
| `memory_type` | `Option<String>` | no | default when absent; Experience type for each child (default `Context`). |

### SseQuery

Defined in `src/handlers/webhooks.rs`. Query parameters for SSE endpoints

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |

### StageCaps

Defined in `src/query_planner/predicate.rs`. Per-stage candidate caps. The planner clamps every estimate against the matching cap so a 10_000-element vector scan can be budget-trimmed to e.g. 1_000.

| Field | Type | Required | Notes |
|---|---|---|---|
| `vector` | `usize` | yes |  |
| `graph` | `usize` | yes |  |
| `relational` | `usize` | yes |  |

### StampedCore

Defined in `src/tag_cores.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `records` | `usize` | yes |  |
| `tags` | `Vec<String>` | yes |  |
| `sources` | `BTreeMap<&'static str, usize>` | yes | How many of the tag placements in this core came from each source. |
| `redacted_placements` | `usize` | yes |  |
| `sample_memory_ids` | Vec&lt;[`MemoryId`](#memoryid)&gt; | yes |  |

### StampedCores

Defined in `src/tag_cores.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `witness` | `&'static str` | yes |  |
| `rows_seen` | `usize` | yes |  |
| `distinct_tags` | `usize` | yes |  |
| `min_records` | `usize` | yes |  |
| `min_tags` | `usize` | yes |  |
| `cores_found` | `usize` | yes |  |
| `cores` | Vec&lt;[`StampedCore`](#stampedcore)&gt; | yes |  |
| `undetermined` | Option&lt;[`Undetermined`](#undetermined)&gt; | no | omitted when empty |

### StampedCoresRequest

Defined in `src/handlers/graph.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `min_records` | `Option<usize>` | no |  |
| `min_tags` | `Option<usize>` | no |  |
| `limit` | `Option<usize>` | no |  |

### StatsQuery

Defined in `src/handlers/users.rs`. Query parameters for stats endpoint

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |

### StorageStats

Defined in `src/memory/storage.rs`. Storage statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_count` | `usize` | yes |  |
| `compressed_count` | `usize` | yes |  |
| `total_size_bytes` | `usize` | yes |  |
| `average_importance` | `f32` | yes | score |
| `importance_sum` | `f32` | yes |  |
| `total_retrievals` | `usize` | no | default when absent; Total number of recall/retrieval operations (persisted) |

### StorageStatsRequest

Defined in `src/handlers/compression.rs`. Request for storage statistics

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### SummaryItem

Defined in `src/handlers/recall.rs`. Summary item - simplified memory for context

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `importance` | `f32` | yes | score |
| `created_at` | `String` | yes |  |

### SurfacedMemory

Defined in `src/relevance.rs`. A surfaced memory with relevance scoring

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes | Memory ID |
| `content` | `String` | yes | Memory content |
| `memory_type` | `String` | yes | Memory type (Decision, Learning, etc.) |
| `importance` | `f32` | yes | score; Base importance score |
| `relevance_score` | `f32` | yes | score; Relevance score for this context (0.0-1.0) |
| `relevance_reason` | [`RelevanceReason`](#relevancereason) | yes | Why this memory was surfaced |
| `matched_entities` | `Vec<String>` | yes | Matched entities (if entity-based) |
| `semantic_similarity` | `Option<f32>` | no | score; Semantic similarity score (if semantic-based) |
| `created_at` | `DateTime<Utc>` | yes | When the memory was created |
| `tags` | `Vec<String>` | yes | Tags associated with the memory |

### SyncMode

Defined in `src/memory/slow_store/mod.rs`. How a sync brought the mirror up to the graph.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `full` | none | Rebuilt from a full pass over the graph. |
| `incremental` | none | Only the rows changed since the last sync. |
| `unchanged` | none | Nothing changed since the last sync. |
| `skipped` | none | Another sync was in flight, or the last rebuild is within the TTL. |

### TagClass

Defined in `src/handlers/ui.rs`.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `visible` | none |  |
| `plumbing` | none |  |
| `malformed` | none |  |
| `timestamp` | none |  |

### TagRow

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `tag` | `String` | yes |  |
| `class` | [`TagClass`](#tagclass) | yes |  |
| `memories` | `usize` | yes | Memories carrying the tag (document frequency). |
| `weight` | `f32` | yes | score; `memories × ln(N / memories)`: a tag on every memory weighs nothing. |
| `origin` | [`OriginCounts`](#origincounts) | yes |  |
| `sources` | `BTreeMap<&'static str, usize>` | yes |  |
| `first_seen` | `DateTime<Utc>` | yes |  |
| `last_seen` | `DateTime<Utc>` | yes |  |
| `weekly` | `Vec<usize>` | yes | Memories created per week, oldest first, the last entry being the current week. |
| `producer` | [`Producer`](#producer) | yes |  |
| `cluster` | `Option<usize>` | no | omitted when empty |
| `co_occurring` | Vec&lt;[`CoTag`](#cotag)&gt; | yes |  |

### TagsQuery

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no |  |
| `include_hidden` | `bool` | no | default when absent |
| `limit` | `Option<usize>` | no |  |

### TagsReport

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `population` | [`Population`](#population) | yes |  |
| `hidden` | BTreeMap&lt;[`TagClass`](#tagclass), [`HiddenCount`](#hiddencount)&gt; | yes |  |
| `tags` | Vec&lt;[`TagRow`](#tagrow)&gt; | yes |  |
| `clusters` | Vec&lt;[`Cluster`](#cluster)&gt; | yes |  |
| `rule` | [`PlumbingRule`](#plumbingrule) | yes |  |
| `include_hidden` | `bool` | yes |  |
| `limit` | `usize` | yes |  |
| `week_starts` | `Vec<DateTime<Utc>>` | yes |  |

### TemporalContext

Defined in `src/memory/types.rs`. Temporal context

| Field | Type | Required | Notes |
|---|---|---|---|
| `time_of_day` | `Option<String>` | no | Time of day |
| `day_of_week` | `Option<String>` | no | Day of week |
| `session_duration_minutes` | `Option<u32>` | no | Session duration |
| `time_since_last_interaction` | `Option<i64>` | no | Time since last interaction |
| `patterns` | Vec&lt;[`TimePattern`](#timepattern)&gt; | yes | Recurring patterns detected |
| `trends` | `Vec<String>` | yes | Historical trends |

### TemporalFactEntry

Defined in `src/handlers/facts.rs`. A single temporal fact in the API response

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | `String` | yes |  |
| `event` | `String` | yes |  |
| `event_type` | `String` | yes |  |
| `timestamp` | `String` | yes |  |
| `source_memory_id` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `source_text` | `String` | yes |  |
| `valid_from` | `Option<String>` | no | When this fact became true (ISO 8601) |
| `valid_until` | `Option<String>` | no | When this fact was invalidated (ISO 8601). None means still valid. |
| `is_valid` | `bool` | yes | Whether this fact is currently valid |

### TemporalFactWire

Defined in `src/handlers/temporal_derivation.rs`. A temporal fact projected to the wire (B4 supporting evidence).

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | `String` | yes |  |
| `event` | `String` | yes |  |
| `resolved_time` | `String` | yes |  |
| `source_memory_id` | `String` | yes |  |
| `confidence` | `f32` | yes | score |
| `source_text` | `String` | yes |  |

### TemporalFactsRequest

Defined in `src/handlers/facts.rs`. Request for listing/filtering temporal facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity` | `Option<String>` | no |  |
| `event` | `Option<String>` | no |  |
| `limit` | `usize` | no | default when absent |
| `include_expired` | `bool` | no | default when absent; When true, include facts that have been invalidated by contradiction detection. Default: false (only return currently-valid facts). |

### TemporalFactsResponse

Defined in `src/handlers/facts.rs`. Response containing temporal facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `facts` | Vec&lt;[`TemporalFactEntry`](#temporalfactentry)&gt; | yes |  |
| `total` | `usize` | yes |  |

### TemporalFactsSearchRequest

Defined in `src/handlers/facts.rs`. Request for searching temporal facts

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |
| `include_expired` | `bool` | no | default when absent; When true, include facts that have been invalidated by contradiction detection. |

### TierMoveRequest

Defined in `src/handlers/crud.rs`. Request body for the tier-move endpoint.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `memory_id` | `String` | yes |  |
| `target_tier` | `String` | yes | One of: "working", "session", "longterm" / "long_term", "archive" |

### TierMoveResponse

Defined in `src/handlers/crud.rs`. Response for tier-move operations.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `memory_id` | `String` | yes |  |
| `previous_tier` | `String` | yes |  |
| `current_tier` | `String` | yes |  |

### TimeOfDay

Defined in `src/memory/sessions.rs`. Time of day classification

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `early_morning` | none |  |
| `morning` | none |  |
| `afternoon` | none |  |
| `evening` | none |  |
| `night` | none |  |

### TimePattern

Defined in `src/memory/types.rs`. Time pattern for recurring behavior

| Field | Type | Required | Notes |
|---|---|---|---|
| `pattern_type` | `String` | yes |  |
| `frequency` | `String` | yes |  |
| `time_range` | `Option<(String, String)>` | no |  |
| `confidence` | `f32` | yes | score |

### TimePrecision

Defined in `src/memory/facets.rs`. Granularity of a time reference.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `year` | none |  |
| `month` | none |  |
| `day` | none |  |
| `hour` | none |  |
| `minute` | none |  |
| `instant` | none |  |

### TimeSpan

Defined in `src/memory/facets.rs`. A bounded time interval with precision metadata.

| Field | Type | Required | Notes |
|---|---|---|---|
| `start` | `DateTime<Utc>` | yes | Interval start. |
| `end` | `Option<DateTime<Utc>>` | no | Interval end. `None` = point-in-time. |
| `precision` | [`TimePrecision`](#timeprecision) | yes | Granularity of the reference: was it `"2026"`, `"May 2026"`, an instant? |

### Todo

Defined in `src/memory/types.rs`. A GTD-style todo item

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`TodoId`](#todoid) | yes | Unique identifier |
| `seq_num` | `u32` | no | default when absent; Sequential number within project (BOLT-1, MEM-2, etc.) |
| `project_prefix` | `Option<String>` | no | default when absent; Cached project prefix for display (e.g., "BOLT", "MEM") Set when todo is created with a project |
| `user_id` | `String` | yes | User who owns this todo |
| `content` | `String` | yes | What needs to be done |
| `status` | [`TodoStatus`](#todostatus) | no | default when absent; Current status (Linear-style workflow) |
| `priority` | [`TodoPriority`](#todopriority) | no | default when absent; Priority level |
| `project_id` | Option&lt;[`ProjectId`](#projectid)&gt; | no | Associated project (optional) |
| `parent_id` | Option&lt;[`TodoId`](#todoid)&gt; | no | Parent todo for subtasks |
| `contexts` | `Vec<String>` | no | default when absent; GTD contexts (@computer, @phone, @errands, etc.) |
| `tags` | `Vec<String>` | no | default when absent; Tags for categorization |
| `external_id` | `Option<String>` | no | default when absent; omitted when empty; External ID for linking to external systems (e.g., "todoist:123", "linear:SHO-39") Used for two-way sync with external todo/task management systems |
| `due_date` | `Option<DateTime<Utc>>` | no | Due date/time (optional) |
| `recurrence` | Option&lt;[`Recurrence`](#recurrence)&gt; | no | Recurrence pattern (optional) |
| `blocked_on` | `Option<String>` | no | Who/what this is blocked on (when status=Blocked) |
| `notes` | `Option<String>` | no | Additional notes |
| `created_at` | `DateTime<Utc>` | yes | When created |
| `updated_at` | `DateTime<Utc>` | yes | When last modified |
| `completed_at` | `Option<DateTime<Utc>>` | no | When completed (if Done) |
| `sort_order` | `i32` | no | default when absent; Manual sort order within status group (lower = higher in list) |
| `comments` | Vec&lt;[`TodoComment`](#todocomment)&gt; | no | default when absent; Comments and activity history for this todo |
| `embedding` | `Option<Vec<f32>>` | no | default when absent; omitted when empty; Vector embedding for semantic search (MiniLM-L6-v2, 384 dimensions) Computed from content + notes + tags for similarity matching |
| `related_memory_ids` | Vec&lt;[`MemoryId`](#memoryid)&gt; | no | default when absent; Related memory IDs for bidirectional linking Memories that are semantically or explicitly linked to this todo |
| `depends_on` | Vec&lt;[`TodoId`](#todoid)&gt; | no | default when absent; Structured dependencies: this todo cannot start until all listed todos are Done. Replaces freeform `blocked_on` for machine-readable dependency tracking. |

### TodoComment

Defined in `src/memory/types.rs`. A comment/activity on a todo item Used to track progress, notes, and actions taken

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | [`TodoCommentId`](#todocommentid) | yes | Unique identifier |
| `todo_id` | [`TodoId`](#todoid) | yes | The todo this comment belongs to |
| `author` | `String` | yes | Author of the comment (user_id or system) |
| `content` | `String` | yes | Comment content (supports markdown) |
| `comment_type` | [`TodoCommentType`](#todocommenttype) | no | default when absent; Type of activity/comment |
| `created_at` | `DateTime<Utc>` | yes | When the comment was created |
| `updated_at` | `Option<DateTime<Utc>>` | no | When the comment was last edited (if ever) |

### TodoCommentId

Defined in `src/memory/types.rs`. Unique identifier for todo comments

On the wire this is its inner value, `Uuid`.

### TodoCommentType

Defined in `src/memory/types.rs`. Type of todo comment

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `comment` | none | User comment |
| `activity` | none | System-generated activity (status change, assignment, etc.) |
| `progress` | none | Progress update |
| `resolution` | none | Resolution/fix description |

### TodoCompleteResponse

Defined in `src/handlers/todos.rs`. Response for todo complete with potential next recurrence

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `todo` | Option&lt;[`Todo`](#todo)&gt; | no |  |
| `next_recurrence` | Option&lt;[`Todo`](#todo)&gt; | no |  |
| `unblocked` | Vec&lt;[`Todo`](#todo)&gt; | yes | Todos whose dependencies are all closed now that this one is done. |
| `formatted` | `String` | yes |  |

### TodoId

Defined in `src/memory/types.rs`. Unique identifier for todos

On the wire this is its inner value, `Uuid`.

### TodoListResponse

Defined in `src/handlers/todos.rs`. Response for todo list operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `count` | `usize` | yes |  |
| `todos` | Vec&lt;[`Todo`](#todo)&gt; | yes |  |
| `projects` | Vec&lt;[`Project`](#project)&gt; | yes |  |
| `formatted` | `String` | yes |  |

### TodoPriority

Defined in `src/memory/types.rs`. Todo priority (Linear-style)

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `urgent` | none | !!! Urgent (P1) |
| `high` | none | !! High (P2) |
| `medium` | none | ! Medium (P3) - default |
| `low` | none | (none) Low (P4) |
| `none` | none | No priority set |

### TodoQuery

Defined in `src/handlers/todos.rs`. Query params for single todo operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### TodoResponse

Defined in `src/handlers/todos.rs`. Response for todo operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `todo` | Option&lt;[`Todo`](#todo)&gt; | no |  |
| `project` | Option&lt;[`Project`](#project)&gt; | no |  |
| `formatted` | `String` | yes |  |

### TodoStatsRequest

Defined in `src/handlers/todos.rs`. Request for todo stats

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### TodoStatsResponse

Defined in `src/handlers/todos.rs`. Response for todo stats

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `stats` | [`UserTodoStats`](#usertodostats) | yes |  |
| `formatted` | `String` | yes |  |

### TodoStatus

Defined in `src/memory/types.rs`. Todo status (Linear-style workflow)

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `backlog` | none | ◌ Not started, someday/maybe |
| `todo` | none | ○ Ready to do |
| `in_progress` | none | ◐ Actively working on |
| `blocked` | none | ⊘ Waiting for someone/something |
| `done` | none | ● Completed |
| `cancelled` | none | ⊗ Won't do |

### ToolAction

Defined in `src/memory/feedback.rs`. A tool or actuator action performed between feedback cycles.

| Field | Type | Required | Notes |
|---|---|---|---|
| `tool_name` | `String` | yes | Tool or actuator name: "Edit", "Bash", "navigate", "grasp" |
| `inputs` | `HashMap<String, String>` | no | default when absent; Key-value inputs. Claude Code: {"file_path": "/src/main.rs", "command": "cargo build"}. Robotics: {"target": "waypoint_7", "speed": "0.5"}. |
| `success` | `bool` | yes | Whether the action succeeded. Claude Code: true unless tool_output contains error markers. Robotics: derived from outcome_type (success/partial = true). |
| `output_snippet` | `Option<String>` | no | default when absent; First 200 chars of output. Used for content matching. |
| `reward` | `Option<f32>` | no | default when absent; Reward signal (robotics only, -1.0 to 1.0). None for Claude Code tools. |

### TrackedRetrieveRequest

Defined in `src/handlers/types.rs`. Request for tracked retrieval (returns tracking ID for later feedback)

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `query` | `String` | yes |  |
| `limit` | `usize` | no | default when absent |
| `mode` | `String` | no | default when absent |
| `competition_mode` | Option&lt;crate::memory::[`CompetitionMode`](#competitionmode)&gt; | no | default when absent |

### TrackedRetrieveResponse

Defined in `src/handlers/types.rs`. Response with tracking ID for feedback

| Field | Type | Required | Notes |
|---|---|---|---|
| `tracking_id` | `String` | yes |  |
| `ids` | `Vec<String>` | yes |  |
| `memories` | Vec&lt;[`RecallMemory`](#recallmemory)&gt; | yes |  |

### TraverseGraphRequest

Defined in `src/handlers/graph.rs`. Request to traverse graph

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `entity_name` | `String` | yes |  |
| `max_depth` | `Option<usize>` | no |  |

### TraversedEntity

Defined in `src/graph_memory.rs`. Entity with hop distance from traversal origin

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity` | [`EntityNode`](#entitynode) | yes |  |
| `hop_distance` | `usize` | yes | Number of hops from the starting entity (0 = start entity) |
| `decay_factor` | `f32` | yes | Decay factor based on hop distance: 1.0 at hop 0, decays with each hop |

### UiEnvelope

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema` | `u32` | yes |  |
| `user_id` | `String` | yes |  |
| `binding` | [`Binding`](#binding) | yes |  |
| `measured_at` | `DateTime<Utc>` | yes |  |
| `witness` | `&'static str` | yes |  |
| `data` | `T` | yes |  |

### UiShelf

Defined in `src/handlers/ui.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `count` | `usize` | yes |  |
| `items` | `Vec<T>` | yes |  |

### Undetermined

Defined in `src/entity_frequency.rs`. Why the result cannot be read as a measurement of the window. Absent when it can.

| Field | Type | Required | Notes |
|---|---|---|---|
| `reason` | `String` | yes |  |

### UniverseBounds

Defined in `src/graph_memory.rs`. Bounds of the memory universe

| Field | Type | Required | Notes |
|---|---|---|---|
| `min` | [`Position3D`](#position3d) | yes |  |
| `max` | [`Position3D`](#position3d) | yes |  |

### UniverseStar

Defined in `src/graph_memory.rs`. A star in the memory universe (represents an entity)

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `name` | `String` | yes |  |
| `entity_type` | `Option<String>` | no |  |
| `salience` | `f32` | yes |  |
| `mention_count` | `usize` | yes |  |
| `is_proper_noun` | `bool` | yes |  |
| `position` | [`Position3D`](#position3d) | yes |  |
| `color` | `String` | yes |  |
| `size` | `f32` | yes |  |

### UnlockBody

Defined in `src/handlers/keys.rs`. A factor as it arrives in a request body.

Enum, externally tagged: a bare string for a unit variant, `{"<variant>": <payload>}` otherwise.

| Variant | Payload | Notes |
|---|---|---|
| `passphrase` | `String` |  |
| `recovery_code` | `String` |  |
| `secure_enclave` | `{ id: String, secret: String }` |  |
| `passkey` | `{ id: String, secret: String }` |  |

### UnlockOnlyBody

Defined in `src/handlers/keys.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `unlock` | [`UnlockBody`](#unlockbody) | yes |  |

### UpdateCommentRequest

Defined in `src/handlers/todos.rs`. Request to update a comment

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |

### UpdateMemoryRequest

Defined in `src/handlers/crud.rs`. Request for updating memory content

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `embeddings` | `Option<Vec<f32>>` | no |  |

### UpdateMemoryResponse

Defined in `src/handlers/crud.rs`. Response for memory update operations

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `id` | `String` | yes |  |
| `message` | `String` | yes |  |

### UpdateProjectRequest

Defined in `src/handlers/todos.rs`. Request to update a project

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `name` | `Option<String>` | no | default when absent |
| `prefix` | `Option<String>` | no | default when absent |
| `description` | `Option<Option<String>>` | no | default when absent |
| `status` | Option&lt;[`ProjectStatus`](#projectstatus)&gt; | no | default when absent |
| `color` | `Option<Option<String>>` | no | default when absent |

### UpdateTodoRequest

Defined in `src/handlers/todos.rs`. Request to update a todo

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `content` | `Option<String>` | no | default when absent |
| `status` | `Option<String>` | no | default when absent |
| `priority` | `Option<String>` | no | default when absent |
| `project` | `Option<String>` | no | default when absent |
| `contexts` | `Option<Vec<String>>` | no | default when absent |
| `due_date` | `Option<String>` | no | default when absent |
| `blocked_on` | `Option<String>` | no | default when absent |
| `notes` | `Option<String>` | no | default when absent |
| `tags` | `Option<Vec<String>>` | no | default when absent |
| `sort_order` | `Option<i32>` | no | default when absent |
| `parent_id` | `Option<String>` | no | default when absent |
| `external_id` | `Option<String>` | no | default when absent |
| `depends_on` | `Option<Vec<String>>` | no | default when absent; Structured dependencies: list of todo short IDs (e.g. ["SHO-1", "BOLT-3"]) or UUIDs. Replaces the set; an empty list clears it. Unknown references and cycles are a 400. |
| `related_memory_ids` | `Option<Vec<String>>` | no | default when absent; Replaces the set of linked memory UUIDs; an empty list clears it. |
| `todo_id` | `Option<String>` | no | default when absent; The target todo, named in the body by the flat `/api/todos/update` alias; the path-style route takes it from the URL. |

### UpsertRequest

Defined in `src/handlers/remember.rs`. Upsert request - create or update memory

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `external_id` | `String` | yes |  |
| `content` | `String` | yes |  |
| `tags` | `Vec<String>` | no | default when absent |
| `memory_type` | `Option<String>` | no | default when absent |
| `change_type` | `String` | no | default when absent |
| `changed_by` | `Option<String>` | no | default when absent |
| `change_reason` | `Option<String>` | no | default when absent |
| `created_at` | `Option<chrono::DateTime<chrono::Utc>>` | no | default when absent; Provenance, as [`RememberRequest`] accepts it. Stored as `remember` stores it on create; on update a provided field overwrites and an omitted one keeps the stored value. |
| `episode_id` | `Option<String>` | no | default when absent |
| `agent_id` | `Option<String>` | no | default when absent |
| `writer` | Option&lt;crate::memory::facets::[`DeclaredWriter`](#declaredwriter)&gt; | no | default when absent; See [`RememberRequest::writer`]. |
| `entities` | Vec&lt;[`DeclaredEntityInput`](#declaredentityinput)&gt; | no | default when absent; See [`RememberRequest::entities`]. On an update the graph episode is rebuilt from these. |
| `repo` | `Option<String>` | no | default when absent; See [`RememberRequest::repo`]. On update a provided repo replaces the stored one. |

### UpsertResponse

Defined in `src/handlers/remember.rs`. Upsert response

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `String` | yes |  |
| `success` | `bool` | yes |  |
| `was_update` | `bool` | yes |  |
| `version` | `u32` | yes |  |

### UserContext

Defined in `src/memory/types.rs`. User-specific context

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `Option<String>` | no | User ID |
| `name` | `Option<String>` | no | User name |
| `preferences` | `HashMap<String, String>` | yes | User preferences |
| `work_patterns` | Vec&lt;[`TimePattern`](#timepattern)&gt; | yes | User's typical working hours |
| `expertise` | `Vec<String>` | yes | User's expertise areas |
| `goals` | `Vec<String>` | yes | User's goals/objectives |
| `learning_style` | `Option<String>` | no | User's learning style |

### UserIndexHealth

Defined in `src/handlers/health.rs`. Vector-index health of one tenant.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_vectors` | `usize` | yes |  |
| `incremental_inserts` | `usize` | yes |  |
| `deleted_count` | `usize` | yes |  |
| `deletion_ratio` | `f32` | yes |  |
| `needs_rebuild` | `bool` | yes |  |
| `needs_compaction` | `bool` | yes |  |
| `secondary` | Option&lt;[`UserSecondaryIndexHealth`](#usersecondaryindexhealth)&gt; | no | `null` when there is no secondary index. |

### UserIndexHealthResponse

Defined in `src/handlers/health.rs`. Reply of `GET /api/health/index`: one tenant's index coverage and health.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `String` | yes | Always `ok`. |
| `user_id` | `String` | yes |  |
| `index_coverage` | [`IndexCoverage`](#indexcoverage) | yes |  |
| `total_vectors` | `usize` | yes |  |
| `incremental_inserts` | `usize` | yes |  |
| `needs_rebuild` | `bool` | yes |  |
| `rebuild_threshold` | `usize` | yes |  |
| `degradation_percent` | `f64` | yes | `incremental_inserts` as a percentage of `rebuild_threshold`, capped at 100. |
| `secondary` | Option&lt;[`SecondaryIndexSummary`](#secondaryindexsummary)&gt; | no | `null` when there is no secondary index. |
| `timestamp` | `String` | yes | RFC 3339. |

### UserMemoryCounts

Defined in `src/handlers/health.rs`. Memory counts of one tenant.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_memories` | `usize` | yes |  |
| `working_memory_count` | `usize` | yes |  |
| `session_memory_count` | `usize` | yes |  |
| `long_term_memory_count` | `usize` | yes |  |
| `vector_index_count` | `usize` | yes |  |

### UserReadyResponse

Defined in `src/handlers/health.rs`. Reply of `GET /api/health/ready`. The counters are present only when the tenant is cached and its store is not locked; `reason` says why they are absent.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `&'static str` | yes | `ready` or `not_ready`: the server, not the tenant. |
| `user_id` | `String` | yes |  |
| `user_ready` | `bool` | yes |  |
| `cached` | `bool` | yes | Whether the tenant's store is open in memory. |
| `exists_on_disk` | `Option<bool>` | no | omitted when empty; Only when not cached. |
| `reason` | `Option<&'static str>` | no | omitted when empty |
| `memory_stats` | Option&lt;[`UserMemoryCounts`](#usermemorycounts)&gt; | no | omitted when empty |
| `index_health` | Option&lt;[`UserIndexHealth`](#userindexhealth)&gt; | no | omitted when empty |
| `graph_stats` | Option&lt;Option&lt;[`GraphCounts`](#graphcounts)&gt;&gt; | no | omitted when empty; Present with the counters; `null` when the tenant has no graph. |
| `timestamp` | `String` | yes | RFC 3339. |

### UserSecondaryIndexHealth

Defined in `src/handlers/health.rs`. Health of a tenant's secondary vector index.

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_vectors` | `usize` | yes |  |
| `incremental_inserts` | `usize` | yes |  |
| `deleted_count` | `usize` | yes |  |
| `deletion_ratio` | `f32` | yes |  |
| `needs_rebuild` | `bool` | yes |  |
| `needs_compaction` | `bool` | yes |  |

### UserTodoStats

Defined in `src/memory/todos.rs`. Overall todo stats for a user

| Field | Type | Required | Notes |
|---|---|---|---|
| `total` | `usize` | yes |  |
| `backlog` | `usize` | yes |  |
| `todo` | `usize` | yes |  |
| `in_progress` | `usize` | yes |  |
| `blocked` | `usize` | yes |  |
| `done` | `usize` | yes |  |
| `cancelled` | `usize` | yes |  |
| `overdue` | `usize` | yes |  |
| `due_today` | `usize` | yes |  |
| `projects` | `usize` | yes |  |

### UsernameBody

Defined in `src/handlers/user_admin.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |

### UsernameRequest

Defined in `src/handlers/user_auth.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | `String` | yes |  |

### VectorPredicate

Defined in `src/query_planner/predicate.rs`. ANN vector search predicate. `kind` selects which Vamana/HNSW index to hit (e.g. `"vamana-text-primary"`); concrete resolution happens in `VectorQuerier`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `kind` | `String` | yes |  |
| `query_vector` | `Vec<f32>` | yes |  |
| `top_k` | `usize` | yes |  |

### VerifiedWriter

Defined in `src/memory/facets.rs`. Writer facts THE SERVER ESTABLISHED ITSELF. Kept structurally separate from [`DeclaredWriter`] rather than distinguished by naming convention, so that reading a claim as though it were evidence takes a deliberate act.

| Field | Type | Required | Notes |
|---|---|---|---|
| `principal` | `Option<String>` | no | default when absent; Principal resolved from the presented credential, where the deployment binds keys to users. `None` when binding is not enabled: a valid key was still presented, but it names nobody. |
| `received_at` | `Option<DateTime<Utc>>` | no | default when absent; Server clock at acceptance. Deliberately distinct from the record's `created_at`, which a caller is allowed to override. |

### VerifyBackupRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `backup_id` | `u32` | yes |  |

### VerifyBackupResponse

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `is_valid` | `bool` | yes |  |
| `message` | `String` | yes |  |

### VerifyIndexRequest

Defined in `src/handlers/types.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |

### VoidSummary

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `boundary_entities` | `Vec<String>` | yes |  |
| `radius` | `f32` | yes |  |
| `confidence` | `f32` | yes | score |

### VoronoiAnalysisRequest

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | `String` | yes |  |
| `k` | `usize` | no | default when absent |
| `offset` | `usize` | no | default when absent |
| `limit` | `usize` | no | default when absent |

### VoronoiAnalysisResponse

Defined in `src/handlers/gap_analysis.rs`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_count` | `usize` | yes |  |
| `voids_found` | `usize` | yes |  |
| `avg_anisotropy` | `f32` | yes |  |
| `total_count` | `usize` | yes |  |
| `most_isolated` | Vec&lt;[`IsolatedEntity`](#isolatedentity)&gt; | yes |  |
| `voids` | Vec&lt;[`VoidSummary`](#voidsummary)&gt; | yes |  |

### WFacetMask

Defined in `src/memory/facets.rs`. Which W-facets are populated and reliable for an engram. Serialized as an object of bools so adding a sixth W later doesn't break existing data.

| Field | Type | Required | Notes |
|---|---|---|---|
| `what` | `bool` | yes |  |
| `where` | `bool` | yes | `where` is a Rust keyword — field name is escaped, JSON key is `where`. |
| `when` | `bool` | yes |  |
| `who` | `bool` | yes |  |
| `why` | `bool` | yes |  |

### WebhookResponse

Defined in `src/handlers/integrations.rs`. Reply of `POST /webhook/linear` and `POST /webhook/github`.

Enum, untagged: the variant's payload alone.

| Variant | Payload | Notes |
|---|---|---|
| `Stored` | [`WebhookStored`](#webhookstored) | `status` is `success`: the event was written as a memory. |
| `Acknowledged` | `{ status: String, action: String }` | `status` is `acknowledged`: the event needs no memory (a removal). |
| `Ignored` | `{ status: String, reason: String }` | `status` is `ignored`: an event type the webhook does not process. |

### WebhookStored

Defined in `src/handlers/integrations.rs`. The memory a webhook event was written to.

| Field | Type | Required | Notes |
|---|---|---|---|
| `status` | `String` | yes |  |
| `id` | `String` | yes |  |
| `external_id` | `String` | yes | `linear:<identifier>` or `github:<owner>/<repo>#<number>`. |
| `was_update` | `bool` | yes | True when a memory with that `external_id` already existed and was updated. |
| `action` | `String` | yes |  |
| `event_type` | `Option<String>` | no | omitted when empty; GitHub only: `issues` or `pull_request`. |

### WhatFacet

Defined in `src/memory/facets.rs`. The WHAT of an engram — content with explicit gist/verbatim separation so consolidation can semanticize without destroying the engram.

| Field | Type | Required | Notes |
|---|---|---|---|
| `verbatim` | `Option<String>` | no | Original surface form. May be shed as the memory ages (hippocampal detail loss). |
| `gist` | `Option<String>` | no | Abstracted summary; survives consolidation. Becomes the primary content once the engram semanticizes. |
| `content_kind` | [`ContentKind`](#contentkind) | yes | Type of content. Will eventually replace `ExperienceType` as the WHAT discriminant. |
| `content_salience` | `f32` | yes | Content salience, 0.0–1.0. Distinct from emotional arousal. |
| `abstraction_level` | `f32` | yes | Abstraction level: 0.0 = raw episode, 1.0 = fully semanticized schema. Increments during consolidation as `verbatim` is shed. |

### WhenFacet

Defined in `src/memory/facets.rs`. The WHEN of an engram — encoding time, event time, ordinal position, and a Temporal Context Model (TCM) drift vector for graded contiguity.

| Field | Type | Required | Notes |
|---|---|---|---|
| `encoded_at` | `Option<DateTime<Utc>>` | no | When Veld stored this engram. |
| `event_time` | Option&lt;[`TimeSpan`](#timespan)&gt; | no | When the described event actually happened — distinct from encoding. |
| `episode_ordinal` | `Option<u32>` | no | Ordinal position within the episode. |
| `context_drift` | `Vec<f32>` | yes | Temporal Context Model drift vector — a low-dim slowly-drifting state captured at encoding. Cosine distance between two engrams' drift vectors = subjective-time proximity. Graded contiguity signal that replaces brittle `episode_id` equality at retrieval. |
| `recurrence` | Option&lt;[`RecurrencePattern`](#recurrencepattern)&gt; | no | Detected recurrence pattern (e.g. "every Monday standup"). |

### WhereFacet

Defined in `src/memory/facets.rs`. Spatial / locational facet — the WHERE of an engram.

| Field | Type | Required | Notes |
|---|---|---|---|
| `places` | Vec&lt;[`Place`](#place)&gt; | yes | Layered places, ordered coarse → fine. |
| `conceptual_anchors` | `Vec<String>` | yes | Conceptual neighborhood in the knowledge graph — named by dominant entities/concepts. The "cognitive-map" where: where in idea-space. Members are entity names; kept as strings for cross-module portability. |
| `heading` | `Option<f32>` | no | Heading / direction of activity. Robotics: physical heading in radians. Reserved for "direction of work" classification on the code side. |

### WhoFacet

Defined in `src/memory/facets.rs`. The WHO of an engram — source-monitoring provenance plus the agents involved.

| Field | Type | Required | Notes |
|---|---|---|---|
| `provenance` | [`Provenance`](#provenance) | yes | Source-monitoring channel. Subsumes the first-pass `SourceContext`. |
| `agents` | Vec&lt;[`AgentRef`](#agentref)&gt; | yes | Persons / agents present in this engram with their roles. |

### WhyFacet

Defined in `src/memory/facets.rs`. The WHY of an engram — goals served, typed causal links, the event model active at encoding, and prediction / boundary signals.

| Field | Type | Required | Notes |
|---|---|---|---|
| `goal_stack` | Vec&lt;[`GoalRef`](#goalref)&gt; | yes | Goals served by this engram, immediate goal first, nesting up. |
| `causes` | Vec&lt;[`CausalLink`](#causallink)&gt; | yes | Typed causal links to other memories (replaces the bare `causal_chain: Vec<MemoryId>`). |
| `event_model` | `Option<String>` | no | The active event model at encoding — Event Segmentation Theory's "what is happening and why" frame. |
| `boundary` | Option&lt;[`BoundaryCause`](#boundarycause)&gt; | no | Did this engram open a new event? Why did the boundary fire? |
| `prediction` | Option&lt;[`Prediction`](#prediction)&gt; | no | Prediction at encoding and whether it held — schema-violation signal. |
| `schema_id` | `Option<String>` | no | Identifier of the schema this engram instantiates (vmPFC schema slot). |

### WikiRenderResponse

Defined in `src/handlers/crud.rs`. Reply of `POST /api/wiki/render`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `success` | `bool` | yes |  |
| `enabled` | `bool` | yes | Always true in a reply: a disabled render answers 503 instead. |
| `target_dir` | `String` | yes | Directory the wiki was rendered into. |
| `pages` | `usize` | yes | Pages written. |
| `edits_applied` | `usize` | yes | Obsidian edits folded back into the store before rendering. |

---

*Generated from the handler source; a change to a request or response type reaches this page on the next docs build, and `tests/mcp_tool_roundtrip.rs` is where a tool is held to it.*
