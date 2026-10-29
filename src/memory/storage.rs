//! Storage backend for the memory system

use anyhow::{anyhow, Context, Result};
use bincode;
use chrono::{DateTime, Utc};
#[cfg(feature = "rocksdb")]
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DBCompressionType,
    IteratorMode, Options, ReadOptions, WriteBatch, WriteBufferManager, WriteOptions, DB,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "rocksdb")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "rocksdb")]
use std::sync::Arc;

use super::types::*;

/// Helper trait to safely iterate over RocksDB results with error logging.
/// Unlike `.flatten()` which silently ignores errors, this logs them.
#[cfg(feature = "rocksdb")]
trait LogErrors<T> {
    fn log_errors(self) -> impl Iterator<Item = T>;
}

#[cfg(feature = "rocksdb")]
impl<I, T, E> LogErrors<T> for I
where
    I: Iterator<Item = Result<T, E>>,
    E: std::fmt::Display,
{
    fn log_errors(self) -> impl Iterator<Item = T> {
        self.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("RocksDB iterator error (continuing): {}", e);
                None
            }
        })
    }
}

/// Prefixes of the rocksdb default-family rows that are not memories even when their
/// key happens to be 16 bytes long: the subsystems that share the family.
#[cfg(feature = "rocksdb")]
pub(crate) const NON_MEMORY_PREFIXES: &[&[u8]] = &[
    b"stats:",
    b"vmapping:",
    b"interference:",
    b"interference_meta:",
    b"_watermark:",
    b"facts:",
    b"facts_by_entity:",
    b"facts_by_type:",
    b"facts_embedding:",
    b"temporal_facts:",
    b"temporal_by_time:",
    b"temporal_by_entity:",
    b"lineage:",
    b"learning:",
    b"geo:",
];

/// Whether a default-family key is a memory record's: its 16 id bytes, and under no
/// prefix another subsystem writes with.
#[cfg(feature = "rocksdb")]
pub(crate) fn is_memory_key(key: &[u8]) -> bool {
    key.len() == 16 && !NON_MEMORY_PREFIXES.iter().any(|p| key.starts_with(p))
}

/// Precision of the geohash every `geo:` index row is keyed by (~1.2 m x 0.6 m cells).
#[cfg(feature = "rocksdb")]
const GEO_INDEX_PRECISION: usize = 10;

/// The geohash cells, all of one precision, that together hold every point within
/// `radius_meters` of the centre: the centre's cell and its eight neighbours, at the
/// finest precision whose cells are at least as tall and as wide as the circle reaches
/// from anywhere inside the centre cell. `None` when no precision is coarse enough —
/// the circle reaches a pole, or the centre lies off the globe — and the whole index
/// is read instead.
///
/// The reach is the haversine's own bound: two points `d` apart differ in latitude by
/// at most `d / R`, and in longitude by at most `2·asin(sin(d / 2R) / cos φ)`, φ the
/// greatest latitude either can hold. A cell grid chosen from a nominal cell size
/// instead missed memories inside the radius whenever the circle crossed more than
/// one neighbour, which redb's latitude band never does.
#[cfg(feature = "rocksdb")]
fn covering_geohash_cells(lat: f64, lon: f64, radius_meters: f64) -> Option<Vec<String>> {
    use super::types::{geohash_encode, geohash_neighbors};

    if !(-90.0..=90.0).contains(&lat) {
        return None;
    }
    let lon = (lon + 180.0).rem_euclid(360.0) - 180.0;
    let lat_reach = (radius_meters / EARTH_RADIUS_METRES).to_degrees();
    let polar_lat = lat.abs() + lat_reach;
    if polar_lat.is_nan() || polar_lat >= 90.0 {
        return None;
    }
    let lon_reach_sine =
        (radius_meters / (2.0 * EARTH_RADIUS_METRES)).sin() / polar_lat.to_radians().cos();
    if lon_reach_sine.is_nan() || lon_reach_sine >= 1.0 {
        return None;
    }
    let lon_reach = (2.0 * lon_reach_sine.asin()).to_degrees();

    (1..=GEO_INDEX_PRECISION).rev().find_map(|precision| {
        let lat_bits = (5 * precision / 2) as i32;
        let lon_bits = 5 * precision as i32 - lat_bits;
        let lat_span = 180.0 / 2f64.powi(lat_bits);
        let lon_span = 360.0 / 2f64.powi(lon_bits);
        (lat_span >= lat_reach && lon_span >= lon_reach).then(|| {
            let centre = geohash_encode(lat, lon, precision);
            let mut cells = geohash_neighbors(&centre);
            cells.push(centre);
            cells.sort();
            cells.dedup();
            cells
        })
    })
}

// ============================================================================
// BACKWARD-COMPATIBLE DESERIALIZATION
// Handles versioned format (SHO magic + checksum), current, and legacy formats
// ============================================================================

const STORAGE_MAGIC: &[u8; 3] = b"SHO";

/// Maximum allocation during bincode deserialization (64 MB).
/// Prevents OOM from corrupt varint length fields — a corrupt value like
/// 7957695011165131522 (7.9 exabytes) would be rejected before allocation.
const BINCODE_ALLOC_LIMIT: usize = 64 * 1024 * 1024;

use std::collections::HashMap;

/// Default experience type for legacy deserialization
fn default_legacy_experience_type() -> ExperienceType {
    ExperienceType::Observation
}

/// Minimal memory format - EXACT match for hex pattern: UUID (16 bytes) + varint string length + string
/// This is the simplest possible format with no extra fields. bincode doesn't support #[serde(default)]
/// so we can't have optional fields - the struct must match the binary data exactly.
#[derive(Deserialize)]
struct MinimalMemory {
    id: MemoryId,
    content: String,
}

/// Memory with experience type prefix - format: UUID + u8 (unknown) + u8 (experience_type) + String
/// Matches the failing entries that have 2 extra bytes before content:
/// - Byte 16: u8 value (seen as 28/0x1c)
/// - Byte 17: u8 experience type index (seen as 7 = Task)
/// - Byte 18+: varint length + string content
#[derive(Deserialize)]
struct MemoryWithTypePrefix {
    id: MemoryId,
    _unknown_field: u8,  // byte 16 - purpose unclear, maybe version?
    experience_type: u8, // byte 17 - experience type enum index
    content: String,     // byte 18+
}

/// Memory with 3-byte header - format: UUID + 3 bytes + raw content
/// Hex analysis shows: bytes 16-18 are header, byte 19+ is UTF-8 content starting with `[`
/// - Byte 16: 0x1c (28) - unknown
/// - Byte 17: 0x07 (7) - possibly experience type
/// - Byte 18: 0xa4 (164) - unknown (NOT valid UTF-8 start)
/// - Byte 19+: UTF-8 content
#[derive(Deserialize)]
struct MemoryWith3ByteHeader {
    id: MemoryId,
    _header1: u8,    // byte 16
    _header2: u8,    // byte 17
    _header3: u8,    // byte 18
    content: String, // byte 19+
}

impl MemoryWith3ByteHeader {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

impl MemoryWithTypePrefix {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let exp_type = match self.experience_type {
            0 => ExperienceType::Observation,
            1 => ExperienceType::Decision,
            2 => ExperienceType::Learning,
            3 => ExperienceType::Error,
            4 => ExperienceType::Discovery,
            5 => ExperienceType::Pattern,
            6 => ExperienceType::Context,
            7 => ExperienceType::Task,
            8 => ExperienceType::CodeEdit,
            9 => ExperienceType::FileAccess,
            10 => ExperienceType::Search,
            11 => ExperienceType::Command,
            12 => ExperienceType::Conversation,
            _ => ExperienceType::Observation,
        };
        let experience = Experience {
            experience_type: exp_type,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

impl MinimalMemory {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5, // default importance
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Very simple legacy Memory - some early versions stored content directly without Experience wrapper
/// This matches the hex pattern: UUID (16 bytes) + varint string length + string content
#[derive(Deserialize)]
struct SimpleLegacyMemory {
    id: MemoryId,
    content: String, // Direct content field, no Experience wrapper
    #[serde(default)]
    importance: f32,
    #[serde(default)]
    access_count: u32,
    #[serde(default)]
    created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    last_accessed: Option<DateTime<Utc>>,
    #[serde(default)]
    compressed: bool,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    actor_id: Option<String>,
    #[serde(default)]
    temporal_relevance: f32,
    #[serde(default)]
    score: Option<f32>,
}

impl SimpleLegacyMemory {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            if self.importance > 0.0 {
                self.importance
            } else {
                0.5
            },
            self.access_count,
            self.created_at.unwrap_or(now),
            self.last_accessed.unwrap_or(now),
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy Experience type from v0.1.0 - EXACT match for bincode 1.x deserialization
/// Includes all fields that were in the original Experience struct in EXACT ORDER.
/// bincode 1.x serializes fields positionally, so order matters!
#[derive(Deserialize)]
struct LegacyExperienceV1 {
    // Core fields (always present)
    #[serde(default = "default_legacy_experience_type")]
    experience_type: ExperienceType,
    content: String,
    #[serde(default)]
    context: Option<RichContext>,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    metadata: HashMap<String, String>,
    #[serde(default)]
    embeddings: Option<Vec<f32>>,
    #[serde(default)]
    related_memories: Vec<MemoryId>,
    #[serde(default)]
    causal_chain: Vec<MemoryId>,
    #[serde(default)]
    outcomes: Vec<String>,
    // Robotics fields
    #[serde(default)]
    robot_id: Option<String>,
    #[serde(default)]
    mission_id: Option<String>,
    #[serde(default)]
    geo_location: Option<[f64; 3]>,
    #[serde(default)]
    local_position: Option<[f32; 3]>,
    #[serde(default)]
    heading: Option<f32>,
    #[serde(default)]
    action_type: Option<String>,
    #[serde(default)]
    reward: Option<f32>,
    #[serde(default)]
    sensor_data: HashMap<String, f64>,
    // Decision & learning fields
    #[serde(default)]
    decision_context: Option<HashMap<String, String>>,
    #[serde(default)]
    action_params: Option<HashMap<String, String>>,
    #[serde(default)]
    outcome_type: Option<String>,
    #[serde(default)]
    outcome_details: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    alternatives_considered: Vec<String>,
    // Environmental context
    #[serde(default)]
    weather: Option<HashMap<String, String>>,
    #[serde(default)]
    terrain_type: Option<String>,
    #[serde(default)]
    lighting: Option<String>,
    #[serde(default)]
    nearby_agents: Vec<HashMap<String, String>>,
    // Failure & anomaly tracking
    #[serde(default)]
    is_failure: bool,
    #[serde(default)]
    is_anomaly: bool,
    #[serde(default)]
    severity: Option<String>,
    #[serde(default)]
    recovery_action: Option<String>,
    #[serde(default)]
    root_cause: Option<String>,
    // Learned patterns & predictions
    #[serde(default)]
    pattern_id: Option<String>,
    #[serde(default)]
    predicted_outcome: Option<String>,
    #[serde(default)]
    prediction_accurate: Option<bool>,
    #[serde(default)]
    tags: Vec<String>,
}

impl LegacyExperienceV1 {
    fn into_experience(self) -> Experience {
        Experience {
            experience_type: self.experience_type,
            content: self.content,
            context: self.context,
            entities: self.entities,
            metadata: self.metadata,
            embeddings: self.embeddings,
            image_embeddings: None,
            audio_embeddings: None,
            video_embeddings: None,
            media_refs: Vec::new(),
            related_memories: self.related_memories,
            causal_chain: self.causal_chain,
            outcomes: self.outcomes,
            robot_id: self.robot_id,
            mission_id: self.mission_id,
            geo_location: self.geo_location,
            local_position: self.local_position,
            heading: self.heading,
            action_type: self.action_type,
            reward: self.reward,
            sensor_data: self.sensor_data,
            decision_context: self.decision_context,
            action_params: self.action_params,
            outcome_type: self.outcome_type,
            outcome_details: self.outcome_details,
            confidence: self.confidence,
            alternatives_considered: self.alternatives_considered,
            weather: self.weather,
            terrain_type: self.terrain_type,
            lighting: self.lighting,
            nearby_agents: self.nearby_agents,
            is_failure: self.is_failure,
            is_anomaly: self.is_anomaly,
            severity: self.severity,
            recovery_action: self.recovery_action,
            root_cause: self.root_cause,
            pattern_id: self.pattern_id,
            predicted_outcome: self.predicted_outcome,
            prediction_accurate: self.prediction_accurate,
            tags: self.tags,
            temporal_refs: Vec::new(),
            ner_entities: Vec::new(),
            cooccurrence_pairs: Vec::new(),
            embedding_degraded: false,
            embeddings_secondary: None,
            embedding_space: None,
            embedding_space_secondary: None,
            other_vectors: Vec::new(),
        }
    }
}

/// Legacy v0.1.0 Memory with full Experience - for bincode 1.x deserialization
#[derive(Deserialize)]
struct LegacyMemoryV1Full {
    #[serde(rename = "memory_id")]
    id: MemoryId,
    experience: LegacyExperienceV1,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV1Full {
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy v0.1.0 format - matches the initial release serialization
/// Uses LegacyExperienceV1 (no multimodal fields) because bincode is positional
#[derive(Deserialize)]
struct LegacyMemoryV1 {
    #[serde(rename = "memory_id")]
    id: MemoryId,
    experience: LegacyExperienceV1, // Must use legacy Experience - bincode ignores #[serde(default)]
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV1 {
    /// Convert legacy v1 format to current Memory format
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy v2 format - after cognitive extensions but before external linking
/// Has tier, entity_refs, activation but no external_id/version/history
#[derive(Deserialize)]
struct LegacyMemoryV2 {
    id: MemoryId,
    experience: LegacyExperienceV1, // bincode is positional - must match original fields
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<uuid::Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV2 {
    /// Convert legacy v2 format to current Memory format
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            self.tier,
            self.entity_refs,
            self.activation,
            self.last_retrieval_id,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,       // external_id not in v2
            1,          // version
            Vec::new(), // history not in v2
            Vec::new(), // related_todo_ids not in v2
        )
    }
}

/// Deserialize memory with multi-version fallback for backwards compatibility
///
/// Tries formats in order from newest to oldest:
/// 1. Current format (with external linking, todos)
/// 2. Legacy v2 (cognitive extensions, no external linking)
/// 3. Legacy v1 (original v0.1.0 format)
///
/// Returns (Memory, needs_migration) where needs_migration=true means the data
/// was in a legacy format and should be re-written for future performance.
/// Content in the retired field-level format is opened here and the record flagged,
/// so the lazy migration rewrites it sealed under the keystore.
#[cfg(feature = "rocksdb")]
fn open_legacy_field((mut memory, needs_migration): (Memory, bool)) -> Result<(Memory, bool)> {
    match crate::legacy_field_encryption::open_legacy_content(&memory.experience.content)? {
        Some(plain) => {
            memory.experience.content = plain;
            Ok((memory, true))
        }
        None => Ok((memory, needs_migration)),
    }
}

fn deserialize_memory(data: &[u8]) -> Result<(Memory, bool)> {
    // Reject obviously corrupt entries before any deserialization attempt.
    // A single memory should never exceed 64MB — corrupt varint length fields
    // in bincode1 can request 7.9 exabytes and OOM-kill the process.
    if data.len() > BINCODE_ALLOC_LIMIT {
        return Err(anyhow!(
            "Memory data too large: {} bytes (limit: {} bytes). Likely corrupt.",
            data.len(),
            BINCODE_ALLOC_LIMIT
        ));
    }

    // Check for versioned format: SHO + version byte + payload + 4-byte CRC32
    if data.len() >= 8 && &data[0..3] == STORAGE_MAGIC {
        let version = data[3];
        let payload_end = data.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            data[payload_end],
            data[payload_end + 1],
            data[payload_end + 2],
            data[payload_end + 3],
        ]);
        let computed_checksum = crc32_simple(&data[..payload_end]);
        if stored_checksum != computed_checksum {
            tracing::warn!(
                "Checksum mismatch: stored={:08x} computed={:08x}",
                stored_checksum,
                computed_checksum
            );
        }
        let payload = &data[4..payload_end];
        // Try current format first, then fallback to legacy formats
        deserialize_with_fallback(payload).map_err(|e| anyhow!("v{version} decode failed: {e}"))
    } else {
        // Legacy format: raw bincode (no SHO header)
        deserialize_with_fallback(data).map_err(|e| anyhow!("legacy decode failed: {e}"))
    }
}

/// The bytes a portable migration carries for one stored memory record.
///
/// A record already in the current plain-bincode format is carried as stored, so a
/// re-derivation from the target reproduces it byte for byte; anything else is decoded
/// through the legacy chain and re-encoded once. `Memory` holds `HashMap`s, so
/// re-encoding is not byte-stable and must not be applied to already-current records.
/// Sealed records are refused here; the callers that carry them carry them verbatim.
pub(crate) fn portable_memory_value(data: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>> {
    if crate::keystore::is_encrypted_record(data) {
        return Err(anyhow!(
            "record is encrypted; a portable dump cannot carry it"
        ));
    }
    if is_current_memory_encoding(data) {
        return Ok(std::borrow::Cow::Borrowed(data));
    }
    let (memory, _) = deserialize_memory(data)?;
    let encoded = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .context("re-encoding a decoded memory")?;
    Ok(std::borrow::Cow::Owned(encoded))
}

/// Whether `data` is exactly one `Memory` in the current plain-bincode layout, which
/// the readers of both engines decode.
pub(crate) fn is_current_memory_encoding(data: &[u8]) -> bool {
    let limited = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
    matches!(
        bincode::serde::decode_from_slice::<Memory, _>(data, limited),
        Ok((_, used)) if used == data.len()
    )
}

#[cfg(feature = "rocksdb")]
#[cfg(test)]
pub(crate) fn deserialize_memory_for_test(data: &[u8]) -> Result<(Memory, bool)> {
    deserialize_memory(data)
}

/// Whether stored vector-mapping bytes decode, with the same allocation bound as reads.
pub(crate) fn check_vector_mapping_value(data: &[u8]) -> Result<()> {
    let limited = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
    bincode::serde::decode_from_slice::<VectorMappingEntry, _>(data, limited)
        .map(|_| ())
        .map_err(|e| anyhow!("vector mapping does not decode: {e}"))
}

/// Legacy MemoryFlat for bincode 2.x data written BEFORE multimodal Experience fields
/// This matches the format at MIF commit (dee4b03) - has tier/entity_refs but Experience without multimodal
#[derive(Deserialize)]
struct LegacyMemoryFlatV2 {
    id: MemoryId,
    experience: LegacyExperienceV1, // No multimodal fields
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<uuid::Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    #[serde(default)]
    related_todo_ids: Vec<TodoId>,
}

impl LegacyMemoryFlatV2 {
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            self.tier,
            self.entity_refs,
            self.activation,
            self.last_retrieval_id,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            self.external_id,
            self.version,
            self.history,
            self.related_todo_ids,
        )
    }
}

/// Legacy MemoryFlat from before W3.4 (kind / what / when were appended).
/// Matches the MemoryFlat schema as it existed prior to the W3.4 extension —
/// every field through `fragment_demotion` is identical. Tried BEFORE
/// `LegacyMemoryFlatV2` so data that was written with `parent_id` / `valid_until`
/// / metadata calibration still preserves them (V2 would silently drop them).
#[derive(Deserialize)]
#[allow(dead_code)] // legacy deserialize struct: fields kept for round-trip, not all read
struct LegacyMemoryFlatV3 {
    id: MemoryId,
    experience: Experience,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<uuid::Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    #[serde(default)]
    related_todo_ids: Vec<TodoId>,
    #[serde(default)]
    parent_id: Option<MemoryId>,
    #[serde(default)]
    anchored: bool,
    #[serde(default)]
    calibrated_confidence: Option<f32>,
    #[serde(default)]
    confidence_alpha: Option<f32>,
    #[serde(default)]
    confidence_beta: Option<f32>,
    #[serde(default)]
    valid_until: Option<DateTime<Utc>>,
    #[serde(default)]
    access_history: Option<Vec<DateTime<Utc>>>,
    #[serde(default)]
    elaboration_score: Option<f32>,
    #[serde(default)]
    fragment_demotion: Option<f32>,
}

impl LegacyMemoryFlatV3 {
    /// V3 → current: rebuilds the memory via `from_legacy` (which sets the W3
    /// facets to defaults — legacy rows are plain memories with mirror-of-
    /// `created_at` `encoded_at`), then restores the V3-era pub fields
    /// `parent_id` / `valid_until` that `from_legacy` doesn't carry.
    /// Metadata-only fields (anchored, confidence calibration, access_history,
    /// elaboration_score, fragment_demotion) get the same defaults the V2
    /// fallback uses today — acceptable migration loss on this narrow window.
    fn into_memory(self) -> Memory {
        let mut mem = Memory::from_legacy(
            self.id,
            self.experience,
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            self.tier,
            self.entity_refs,
            self.activation,
            self.last_retrieval_id,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            self.external_id,
            self.version,
            self.history,
            self.related_todo_ids,
        );
        mem.parent_id = self.parent_id;
        mem.valid_until = self.valid_until;
        mem
    }
}

// The four `...PreWriter...` shapes below are decoded only for the byte offset where
// their end falls, so bincode must walk every field and none is read: hence the allows.

/// `AgentSession` as persisted before 4a54053 (2026-09-10) appended `declared` and `verified`.
#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyAgentSessionPreWriter {
    worktree_path: Option<PathBuf>,
    branch: Option<String>,
    agent_id: Option<String>,
    vscode_window_id: Option<String>,
    started_at: Option<DateTime<Utc>>,
    parent_repo: Option<PathBuf>,
}

/// The record up to the end of a pre-4a54053 session: decoding this and nothing more yields the
/// offset where the missing bytes belong. Trailing bytes are the rest of the record.
#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyRichContextThroughPreWriterSession {
    id: ContextId,
    conversation: ConversationContext,
    user: UserContext,
    project: ProjectContext,
    temporal: TemporalContext,
    semantic: SemanticContext,
    code: CodeContext,
    document: DocumentContext,
    environment: EnvironmentContext,
    emotional: EmotionalContext,
    source: SourceContext,
    episode: EpisodeContext,
    place: WhereFacet,
    who: WhoFacet,
    why: WhyFacet,
    binding: EngramBinding,
    session: LegacyAgentSessionPreWriter,
}
#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyExperienceThroughPreWriterSession {
    experience_type: ExperienceType,
    content: String,
    context: Option<LegacyRichContextThroughPreWriterSession>,
}
#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyMemoryThroughPreWriterSession {
    id: MemoryId,
    experience: LegacyExperienceThroughPreWriterSession,
}

/// Default `DeclaredWriter` (six `None`s) followed by default `VerifiedWriter` (two `None`s).
const PRE_WRITER_SESSION_GAP: [u8; 8] = [0; 8];

/// How many fields may have been appended to `MemoryFlat` since a record was written, at most.
/// Each appended field encodes its default as one zero byte (see the rule on `MemoryFlat`).
const MAX_TRAILING_DEFAULTS: usize = 16;

/// A record in the current layout, or in one that ends before fields appended since: `data`,
/// then as many zero bytes as the missing fields' defaults take, must decode as `Memory` and
/// consume every byte. Only one count can do that, since a decode reads a fixed sequence of
/// fields, so the result is exact. 2026-10-19: the embedding-space fields were the first
/// appended under this rule, and every record on disk ends before them.
fn decode_with_trailing_defaults(data: &[u8]) -> Option<(Memory, usize)> {
    let cfg = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
    if let Ok((memory, n)) = bincode::serde::decode_from_slice::<Memory, _>(data, cfg) {
        if n == data.len() {
            return Some((memory, 0));
        }
    }
    let mut padded = Vec::with_capacity(data.len() + MAX_TRAILING_DEFAULTS);
    padded.extend_from_slice(data);
    for appended in 1..=MAX_TRAILING_DEFAULTS {
        padded.push(0);
        if let Ok((memory, n)) = bincode::serde::decode_from_slice::<Memory, _>(&padded, cfg) {
            if n == padded.len() {
                return Some((memory, appended));
            }
        }
    }
    None
}

/// Recovers a record persisted with a populated context before 4a54053 appended two structs to
/// `AgentSession`. bincode is positional, so `#[serde(default)]` on the appended fields does
/// nothing for bytes already on disk: the decoder reads the next field's bytes as the missing
/// ones and every later field misreads. The gap is exactly the encoding of the two defaults,
/// so splicing it in where the old session ends gives the wire the ordinary arms already read.
/// Measured 2026-09-15: 24 of 9950 live records, every one with `context: Some`, all written
/// between 2026-08-03 and 2026-09-06 and never re-serialised since.
///
/// Both candidate layouts after the splice must consume every byte: the same splice applied to
/// a record that already carries the two structs shifts its tail by eight and cannot.
fn recover_pre_writer_session(data: &[u8]) -> Option<(Memory, &'static str)> {
    let cfg = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
    let (prefix, gap_at) =
        bincode::serde::decode_from_slice::<LegacyMemoryThroughPreWriterSession, _>(data, cfg)
            .ok()?;
    // Without a context the session never reaches the wire and the layouts are identical.
    prefix.experience.context.as_ref()?;
    let mut spliced = Vec::with_capacity(data.len() + PRE_WRITER_SESSION_GAP.len());
    spliced.extend_from_slice(&data[..gap_at]);
    spliced.extend_from_slice(&PRE_WRITER_SESSION_GAP);
    spliced.extend_from_slice(&data[gap_at..]);

    if let Some((memory, _)) = decode_with_trailing_defaults(&spliced) {
        return Some((memory, "bincode2_pre_writer_session"));
    }
    // edfb43f (2026-09-02) inserted `persistent` mid-struct, so the same window of records is
    // also V3-shaped, with the W3.4 facets trailing.
    let (v3, n) = bincode::serde::decode_from_slice::<LegacyMemoryFlatV3, _>(&spliced, cfg).ok()?;
    let ((kind, what, when), m) = bincode::serde::decode_from_slice::<
        (RecordKind, WhatFacet, WhenFacet),
        _,
    >(&spliced[n..], cfg)
    .ok()?;
    if n + m != spliced.len() {
        return None;
    }
    let mut memory = v3.into_memory();
    memory.kind = kind;
    memory.what = what;
    memory.when = when;
    Some((memory, "bincode2_pre_writer_session_v3"))
}

/// Decodes a `Memory` snapshot carried by an intent-log payload or a projection checkpoint.
/// Those bytes are the same encoding the store holds, so they age the same way: a payload
/// journaled before a positional layout change needs the legacy arms, not a bare
/// `decode_from_slice`. The bm25 projection sat at lsn 12753 from 2026-09-03 to 2026-09-16
/// because it decoded with the current layout only.
///
/// The redb store reads its records here too, for the same reason: a plain current-layout
/// decode fails on every record written before a field was appended to `MemoryFlat`.
pub fn decode_memory_snapshot(bytes: &[u8]) -> Result<Memory> {
    deserialize_memory(bytes).map(|(memory, _)| memory)
}

/// Try deserializing with multiple format fallbacks
/// Supports bincode 2.x (current), MessagePack, and bincode 1.x (legacy) wire formats
///
/// Returns (Memory, is_legacy) where is_legacy=true means the data was in an old format
/// and should be re-written to current format for future performance.
fn deserialize_with_fallback(data: &[u8]) -> Result<(Memory, bool)> {
    fn record_branch(branch: &str) {
        crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL
            .with_label_values(&[branch])
            .inc();
    }

    // Try current format first (bincode 2.x with current Memory/Experience)
    // This is the hot path — avoid any allocations before this check.
    match bincode::serde::decode_from_slice::<Memory, _>(
        data,
        bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
    ) {
        Ok((memory, _)) => Ok((memory, false)), // Current format, no migration needed
        Err(e) => {
            // Current format failed — enter fallback chain.
            // From here on we collect errors for diagnostics.
            deserialize_legacy_fallback(data, e, record_branch)
        }
    }
}

/// Fallback deserialization chain for legacy memory formats.
///
/// Separated from `deserialize_with_fallback` so the hot path (current format)
/// stays allocation-free and the compiler can inline/optimize it independently.
fn deserialize_legacy_fallback(
    data: &[u8],
    first_error: bincode::error::DecodeError,
    record_branch: fn(&str),
) -> Result<(Memory, bool)> {
    // Guard: reject data larger than 64MB before attempting any deserialization.
    // This catches corrupt entries before bincode1 (which lacks with_limit) tries
    // to allocate based on a corrupt varint length field.
    if data.len() > BINCODE_ALLOC_LIMIT {
        return Err(anyhow!(
            "Data too large for memory deserialization: {} bytes (limit: {} bytes)",
            data.len(),
            BINCODE_ALLOC_LIMIT
        ));
    }
    // Log detailed errors for first entry only to help debug format issues
    static DEBUG_ENTRY_LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    let is_first_failure = !DEBUG_ENTRY_LOGGED.load(std::sync::atomic::Ordering::Relaxed);

    // Collect all errors for debugging
    // The two TINY bincode 2.x candidates (MinimalMemory, MemoryWithTypePrefix) must consume
    // the whole record. decode_from_slice does not object to trailing bytes, so a varint
    // decoder of a 16-byte id plus a string could "succeed" on a bincode 1.x fixint record:
    // the 8-byte length prefix 10 00 00 00 00 00 00 00 reads as varint 16 followed by seven
    // zero bytes, the uuid lands displaced by seven and zero-padded, and whether the rest
    // parsed as a String depended on uuid byte 9 — the random failure of
    // test_deserialize_with_fallback_bincode1_minimal_fixture. The FULL-memory candidates
    // (LegacyMemoryFlatV3/V2) keep tolerating trailing bytes: for them a partial read is a
    // later schema with appended fields, not a misread — see the note on each arm.
    let mut errors: Vec<(&str, String)> = Vec::new();
    errors.push(("bincode2 Memory", first_error.to_string()));

    // A current-layout record written before the latest appended fields. Tried before every
    // other arm: the V3 arm below tolerates trailing bytes, so it would read such a record as
    // V3 and drop the W3 facets that follow V3's last field.
    if let Some((memory, _)) = decode_with_trailing_defaults(data) {
        record_branch("bincode2_trailing_defaults");
        return Ok((memory, true));
    }
    errors.push((
        "bincode2 Memory with appended defaults",
        "no count of appended defaults consumed the record".to_string(),
    ));

    // Try the prior-current schema (pre-W3.4 — before kind/what/when were
    // appended to MemoryFlat). Tried first in the fallback chain so data
    // written between "parent_id added" and "W3.4" preserves parent_id /
    // valid_until rather than falling lossy to V2.
    match bincode::serde::decode_from_slice::<LegacyMemoryFlatV3, _>(
        data,
        bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
    ) {
        // Trailing bytes are ALLOWED here: this is a full-memory struct, and a record that
        // decodes as it with bytes left over is a later schema that appended fields — measured
        // 2026-09-13 on the live store, 24 of 9932 records were exactly that and became
        // "corrupted" the moment a full-consumption guard was applied to this arm too.
        Ok((legacy, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x pre-W3.4 format");
            record_branch("bincode2_legacy_flat_v3");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 LegacyMemoryFlatV3", e.to_string())),
    }

    if let Some((memory, branch)) = recover_pre_writer_session(data) {
        tracing::debug!("Recovered memory persisted with a pre-4a54053 AgentSession");
        record_branch(branch);
        return Ok((memory, true));
    }
    errors.push((
        "bincode2 pre-writer AgentSession splice",
        "no layout consumed the spliced record".to_string(),
    ));

    // Try bincode 2.x MINIMAL format (just UUID + content string)
    // This matches the hex pattern: 16-byte UUID + varint length + string bytes
    match bincode::serde::decode_from_slice::<MinimalMemory, _>(data, bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>()) {
        Ok((minimal, n)) if n == data.len() => {
            tracing::debug!("Migrated memory from bincode 2.x minimal format");
            record_branch("bincode2_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Ok((_, n)) => errors.push((
            "bincode2 MinimalMemory",
            format!("decoded {n} of {} bytes — a legacy record has no trailing bytes, so this was a misread", data.len()),
        )),
        Err(e) => errors.push(("bincode2 MinimalMemory", e.to_string())),
    }

    // Try bincode 2.x with type prefix: UUID + u8 + u8 (experience_type) + String
    // Matches entries with 2 extra bytes before content (byte 16=unknown, byte 17=exp_type)
    match bincode::serde::decode_from_slice::<MemoryWithTypePrefix, _>(
        data,
        bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
    ) {
        Ok((typed, n)) if n == data.len() => {
            tracing::debug!("Migrated memory from bincode 2.x with type prefix");
            record_branch("bincode2_type_prefix");
            return Ok((typed.into_memory(), true));
        }
        Ok((_, n)) => errors.push((
            "bincode2 MemoryWithTypePrefix",
            format!("decoded {n} of {} bytes — a legacy record has no trailing bytes, so this was a misread", data.len()),
        )),
        Err(e) => errors.push(("bincode2 MemoryWithTypePrefix", e.to_string())),
    }

    // Try bincode 2.x with OLD Experience (before multimodal fields were added)
    match bincode::serde::decode_from_slice::<LegacyMemoryFlatV2, _>(
        data,
        bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
    ) {
        // Trailing bytes are ALLOWED here: this is a full-memory struct, and a record that
        // decodes as it with bytes left over is a later schema that appended fields — measured
        // 2026-09-13 on the live store, 24 of 9932 records were exactly that and became
        // "corrupted" the moment a full-consumption guard was applied to this arm too.
        Ok((legacy, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x pre-multimodal format");
            record_branch("bincode2_legacy_flat_v2");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 LegacyMemoryFlatV2", e.to_string())),
    }

    // Try bincode 1.x with LegacyMemoryV1 (v0.1.0 format)
    match bincode1::deserialize::<LegacyMemoryV1>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v0.1.0 format");
            record_branch("bincode1_legacy_v1");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV1", e.to_string())),
    }

    // Try bincode 1.x MINIMAL format (just UUID + content)
    match bincode1::deserialize::<MinimalMemory>(data) {
        Ok(minimal) => {
            tracing::debug!("Migrated memory from bincode 1.x minimal format");
            record_branch("bincode1_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 MinimalMemory", e.to_string())),
    }

    // Try bincode 1.x with SIMPLE legacy format (content as direct field, no Experience wrapper)
    match bincode1::deserialize::<SimpleLegacyMemory>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x simple format");
            record_branch("bincode1_simple");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 SimpleLegacyMemory", e.to_string())),
    }

    // Try bincode 1.x with fixint encoding (u64 lengths instead of varint)
    use bincode1::Options;
    let fixint_config = bincode1::options()
        .with_fixint_encoding()
        .allow_trailing_bytes();

    // Try bincode 1.x fixint MinimalMemory
    match fixint_config.deserialize::<MinimalMemory>(data) {
        Ok(minimal) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint minimal format");
            record_branch("bincode1_fixint_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint MinimalMemory", e.to_string())),
    }

    match fixint_config.deserialize::<SimpleLegacyMemory>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint simple format");
            record_branch("bincode1_fixint_simple");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint SimpleLegacyMemory", e.to_string())),
    }

    // Try MessagePack minimal format
    match rmp_serde::from_slice::<MinimalMemory>(data) {
        Ok(minimal) => {
            tracing::debug!("Migrated memory from MessagePack minimal format");
            record_branch("msgpack_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("msgpack MinimalMemory", e.to_string())),
    }

    // Try MessagePack format (rmp-serde) - self-describing format
    match rmp_serde::from_slice::<SimpleLegacyMemory>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from MessagePack simple format");
            record_branch("msgpack_simple");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("msgpack SimpleLegacyMemory", e.to_string())),
    }

    // Try bincode 1.x format with original Experience (no multimodal fields)
    match bincode1::deserialize::<LegacyMemoryV1Full>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v1 full format");
            record_branch("bincode1_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV1Full", e.to_string())),
    }

    // Try bincode 1.x with fixint encoding for full legacy format
    match fixint_config.deserialize::<LegacyMemoryV1Full>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint v1 full format");
            record_branch("bincode1_fixint_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint LegacyMemoryV1Full", e.to_string())),
    }

    // Try MessagePack with full legacy format
    match rmp_serde::from_slice::<LegacyMemoryV1Full>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from MessagePack v1 full format");
            record_branch("msgpack_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("msgpack LegacyMemoryV1Full", e.to_string())),
    }

    // Try bincode 1.x format (used in versions prior to bincode 2.0 migration)
    if let Ok(legacy) = bincode1::deserialize::<LegacyMemoryV1>(data) {
        tracing::debug!("Migrated memory from bincode 1.x format");
        record_branch("bincode1_legacy_v1_repeat");
        return Ok((legacy.into_memory(), true));
    }

    // Try legacy v2 format with bincode 1.x (cognitive extensions era)
    match bincode1::deserialize::<LegacyMemoryV2>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v2 format");
            record_branch("bincode1_legacy_v2");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV2", e.to_string())),
    }

    // Try bincode 2.x with 3-byte header: UUID + 3 bytes + String
    // Hex analysis shows content starts at byte 19 (byte 18 is 0xa4, not valid UTF-8 start)
    match bincode::serde::decode_from_slice::<MemoryWith3ByteHeader, _>(
        data,
        bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
    ) {
        Ok((mem, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x with 3-byte header");
            record_branch("bincode2_3byte_header");
            return Ok((mem.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 MemoryWith3ByteHeader", e.to_string())),
    }

    // Log debug info on first failure only (at debug level to reduce noise)
    if is_first_failure {
        DEBUG_ENTRY_LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);

        let hex_preview: String = data
            .iter()
            .take(32)
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::debug!(
            "Unknown memory format ({} bytes): {}...",
            data.len(),
            hex_preview
        );
    }

    // All formats failed
    record_branch("decode_failed");
    Err(anyhow!(
        "Failed to deserialize memory: incompatible format ({} bytes)",
        data.len()
    ))
}

/// Simple CRC32 implementation (IEEE polynomial)
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Column family name for secondary indices (tags, types, timestamps, etc.)
#[cfg(feature = "rocksdb")]
const CF_INDEX: &str = "memory_index";
/// Sentinel (in `CF_INDEX`) holding the last-seen keystore generation; rollback guard.
#[cfg(feature = "rocksdb")]
const KEYSTORE_GENERATION_KEY: &[u8] = crate::keystore::GENERATION_SENTINEL_KEY;
/// Marker (in `CF_INDEX`) that the content-hash rows are one per memory. A `meta:` key,
/// so `rebuild_index` keeps it.
#[cfg(feature = "rocksdb")]
const CONTENT_HASH_LAYOUT_MARKER: &[u8] = b"meta:content_hash_rows_per_memory";
/// General-purpose key-value column family for small per-user blobs (e.g. the
/// `CognitiveDials` tuning surface) that don't fit the memory/index schemas.
/// Created on open via `create_missing_column_families`, so it materializes on
/// existing databases too.
#[cfg(feature = "rocksdb")]
const CF_KV: &str = "kv";
/// `CF_INDEX` time index of the memories compression has not reached yet:
/// `uncompressed:{sortable created_at millis}:{uuid}`. Lets the compression pass seek to
/// its candidates instead of decoding the whole store.
#[cfg(feature = "rocksdb")]
const UNCOMPRESSED_INDEX_PREFIX: &str = "uncompressed:";
/// Set once the uncompressed time index also covers records written before it existed.
/// A `meta:` key, so `rebuild_index` keeps it while re-deriving the index entries.
#[cfg(feature = "rocksdb")]
const UNCOMPRESSED_INDEX_SENTINEL: &[u8] = b"meta:uncompressed_index_v1";

/// Index key of a memory in the uncompressed time index. Millis are mapped onto `u64` with
/// the sign bit flipped, so zero-padded decimal order is chronological order, pre-1970
/// timestamps included.
#[cfg(feature = "rocksdb")]
fn uncompressed_index_key(created_at: DateTime<Utc>, id: &MemoryId) -> String {
    format!(
        "{UNCOMPRESSED_INDEX_PREFIX}{:020}:{}",
        sortable_millis(created_at),
        id.0
    )
}

#[cfg(feature = "rocksdb")]
fn sortable_millis(at: DateTime<Utc>) -> u64 {
    (at.timestamp_millis() as u64) ^ (1u64 << 63)
}

/// Whether a whole-store scan returns forgotten (soft-deleted) records.
#[cfg(feature = "rocksdb")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncludeForgotten {
    Yes,
    No,
}

#[cfg(feature = "rocksdb")]
/// Storage engine for long-term memory persistence
///
/// Uses a single RocksDB instance with 2 column families:
/// - default: main memory data (also shared by LearningHistoryStore, TemporalFactStore, etc. via key prefixes)
/// - `memory_index`: secondary indices for tag/type/timestamp queries
pub struct MemoryStorage {
    db: Arc<DB>,
    /// Base storage path for all memory data
    storage_path: PathBuf,
    /// Whether every write syncs the write-ahead log: `StorageDurability::Immediate`.
    sync_writes: bool,
    /// Optional record-level encryptor (AES-256-GCM envelope, encryption-v2).
    /// When present, the whole serialized record is encrypted before storage and
    /// decrypted on read. Backed by the keystore (`storage_path/keystore.json`),
    /// unsealed via `VELD_MASTER_PASSPHRASE`; `None` means plaintext (backward
    /// compatible). Holds one cryptor per keystore epoch so records written before
    /// a key rotation stay readable (writes always use the active epoch).
    record_cryptors: Option<crate::keystore::RecordCryptors>,
    /// Optional index-term blinder (encryption-v2 P3). When present, exact-match
    /// index terms (tag/entity/…) are stored as HMAC tokens, not plaintext.
    /// Derived from the master key alongside `record_cryptors`.
    index_blinder: Option<crate::keystore::IndexBlinder>,
    /// See [`Self::undecodable_records_skipped`].
    undecodable_skipped: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "rocksdb")]
/// Total memtable memory all per-user and shared RocksDB instances may hold.
const PROCESS_WRITE_BUFFER_LIMIT_BYTES: usize = 256 * 1024 * 1024;

#[cfg(feature = "rocksdb")]
/// Table options every column family uses: the shared block cache, a 10-bit full bloom
/// filter (about 1% false positives), and index/filter blocks held in that cache with
/// the L0 ones pinned so point lookups never page them back in.
///
/// No prefix extractor is installed on the index family on purpose. Its key families
/// (`tag:`, `inbound_ref:`, `date:`, `geo:`, `content_hash:` ...) have no common
/// length, and the scans over it seek with arbitrary-length bounds; with a prefix bloom
/// RocksDB may skip SST files for a seek whose bound is not the extractor's prefix,
/// silently dropping rows. The whole-key bloom serves the point lookups that dominate.
pub fn block_based_options(cache: &Cache) -> BlockBasedOptions {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_block_cache(cache);
    block_opts.set_cache_index_and_filter_blocks(true);
    block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
    block_opts
}

#[cfg(feature = "rocksdb")]
/// Options for a secondary column family (index, KV): Lz4, two memtables of
/// `write_buffer_bytes`, and [`block_based_options`] over `cache`.
pub fn column_family_options(cache: &Cache, write_buffer_bytes: usize) -> Options {
    let mut cf_opts = Options::default();
    cf_opts.create_if_missing(true);
    cf_opts.set_compression_type(DBCompressionType::Lz4);
    cf_opts.set_max_write_buffer_number(2);
    cf_opts.set_write_buffer_size(write_buffer_bytes);
    cf_opts.set_block_based_table_factory(&block_based_options(cache));
    cf_opts
}

#[cfg(feature = "rocksdb")]
/// The one write buffer manager every shared-cache RocksDB instance in the process
/// attaches to (`Options::set_write_buffer_manager`), capping total memtable memory at
/// 256 MiB and charging it to the cache. Writers are not stalled at the limit; RocksDB
/// flushes instead. The first caller's cache is the one charged, so every caller must
/// pass the same process-wide cache.
pub fn shared_write_buffer_manager(cache: &Cache) -> &'static WriteBufferManager {
    static MANAGER: std::sync::OnceLock<WriteBufferManager> = std::sync::OnceLock::new();
    MANAGER.get_or_init(|| {
        WriteBufferManager::new_write_buffer_manager_with_cache(
            PROCESS_WRITE_BUFFER_LIMIT_BYTES,
            false,
            cache.clone(),
        )
    })
}

#[cfg(feature = "rocksdb")]
impl MemoryStorage {
    /// Whether this store seals its records (a keystore is unsealed).
    pub fn encrypts_records(&self) -> bool {
        self.record_cryptors.is_some()
    }

    /// CF accessor for the memory_index column family
    fn index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_INDEX)
            .expect("memory_index CF must exist")
    }

    /// Create a new memory storage.
    ///
    /// If `shared_cache` is provided, all block-cache reads are charged against
    /// the shared LRU cache (recommended for multi-tenant server mode). When
    /// `None`, a small per-instance cache is created (standalone / test use).
    pub fn new(path: &Path, shared_cache: Option<&Cache>) -> Result<Self> {
        Self::new_with_tuning(path, shared_cache, crate::config::StorageTuning::process())
    }

    /// [`Self::new`] with explicit durability, write-buffer and block-cache settings in
    /// place of the process's.
    pub fn new_with_tuning(
        path: &Path,
        shared_cache: Option<&Cache>,
        tuning: crate::config::StorageTuning,
    ) -> Result<Self> {
        Self::open(path, shared_cache, true, tuning)
    }

    /// Open the store with the keystore already beside it, if any, and never create one:
    /// a store written in plaintext stays plaintext whatever VELD_MASTER_PASSPHRASE holds.
    pub fn open_without_creating_keystore(
        path: &Path,
        shared_cache: Option<&crate::storage::SharedBlockCache>,
    ) -> Result<Self> {
        Self::open(
            path,
            shared_cache,
            false,
            crate::config::StorageTuning::process(),
        )
    }

    fn open(
        path: &Path,
        shared_cache: Option<&crate::storage::SharedBlockCache>,
        create_keystore: bool,
        tuning: crate::config::StorageTuning,
    ) -> Result<Self> {
        let write_buffer_bytes = tuning.rocksdb_write_buffer_bytes;
        let sync_writes = tuning.rocksdb_sync_writes();

        // Create directories if they don't exist
        let storage_path = path.join("storage");
        std::fs::create_dir_all(&storage_path)?;

        // Configure RocksDB options for PRODUCTION durability + performance
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(DBCompressionType::Lz4);

        // ========================================================================
        // DURABILITY SETTINGS - Critical for data persistence across restarts
        // ========================================================================
        //
        // RocksDB data flow: Write → WAL → Memtable → SST files
        //
        // Every write reaches the WAL before it returns (manual WAL flush is off),
        // so it survives a process crash. Whether it also survives a power loss
        // is `StorageDurability`: `sync_writes` syncs the WAL on every write.
        // ========================================================================

        // WAL stays in default location (same as data dir) - avoids corruption issues
        opts.set_manual_wal_flush(false); // Auto-flush WAL entries

        // Write performance — sized for edge deployment (VELD_ROCKSDB_WRITE_BUFFER_BYTES for heavy workloads)
        opts.set_max_write_buffer_number(2);
        opts.set_write_buffer_size(write_buffer_bytes);
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB SST files
        opts.set_max_bytes_for_level_base(256 * 1024 * 1024); // 256MB L1
        opts.set_max_background_jobs(4);
        opts.set_level_compaction_dynamic_level_bytes(true);

        // Read performance — shared block cache for multi-tenant, small local for standalone
        use Cache;
        let local_cache;
        let cache = match shared_cache {
            Some(c) => c,
            None => {
                local_cache = Cache::new_lru_cache(tuning.rocksdb_standalone_block_cache_bytes());
                &local_cache
            }
        };
        opts.set_block_based_table_factory(&block_based_options(cache));
        // Memtable memory is capped process-wide, and only where a shared cache exists:
        // a standalone store has a private 16MB cache that the manager must not be charged to.
        if shared_cache.is_some() {
            opts.set_write_buffer_manager(shared_write_buffer_manager(cache));
        }

        // Open single database with column families. Every family gets the same table
        // options (shared cache, bloom filter, pinned L0 metadata); the index and KV
        // families only differ from the record family in carrying no tuned file sizes.
        let main_opts = opts.clone();
        let index_cache = cache.clone();
        let db = Arc::new(Self::open_or_repair_cf(&opts, &storage_path, move || {
            vec![
                ColumnFamilyDescriptor::new("default", main_opts.clone()),
                ColumnFamilyDescriptor::new(
                    CF_INDEX,
                    column_family_options(&index_cache, write_buffer_bytes),
                ),
                ColumnFamilyDescriptor::new(
                    CF_KV,
                    column_family_options(&index_cache, write_buffer_bytes),
                ),
            ]
        })?);

        // Migrate from old separate-DB layout if needed
        Self::migrate_from_separate_dbs(path, &db, sync_writes)?;
        let converted = Self::convert_single_holder_content_hash_rows(&db, sync_writes)?;
        if converted > 0 {
            tracing::info!(
                rows = converted,
                "content-hash index converted to one row per memory"
            );
        }

        tracing::info!(
            durability = %tuning.durability,
            sync_writes,
            "Storage initialized"
        );

        // Record-level encryption (encryption-v2): the keystore lives next to the
        // data and is unsealed with VELD_MASTER_PASSPHRASE. First run with a
        // passphrase and no keystore creates one. A keystore present without the
        // passphrase is a hard error (it would otherwise serve ciphertext as
        // plaintext). No keystore + no passphrase = plaintext (backward compatible).
        let (record_cryptors, index_blinder) =
            match Self::open_record_cryptor(&db, &storage_path, create_keystore, sync_writes)? {
                Some((cryptors, blinder)) => (Some(cryptors), Some(blinder)),
                None => (None, None),
            };
        crate::legacy_field_encryption::check_config(record_cryptors.is_some())?;

        Ok(Self {
            db,
            storage_path: path.to_path_buf(),
            sync_writes,
            record_cryptors,
            index_blinder,
            undecodable_skipped: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Memory records an enumeration has skipped because they would not decode, since
    /// this store opened.
    pub fn undecodable_records_skipped(&self) -> u64 {
        self.undecodable_skipped
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A record an enumeration read at `key`, decoded; `None`, logged and counted, when
    /// it will not decode. Skipped rather than failing the enumeration, on both engines:
    /// one bad record would otherwise make every listing of the store fail, and
    /// `cleanup_corrupted` is the path that removes it.
    fn decode_or_skip(&self, key: &[u8], value: &[u8], read: &'static str) -> Option<Memory> {
        match self.decode_stored(key, value) {
            Ok((memory, _)) => Some(memory),
            Err(e) => {
                self.note_undecodable(key, &e, read);
                None
            }
        }
    }

    fn note_undecodable(&self, key: &[u8], error: &anyhow::Error, read: &'static str) {
        let skipped = self
            .undecodable_skipped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let memory_id = uuid::Uuid::from_slice(key)
            .map(|id| id.to_string())
            .unwrap_or_else(|_| hex::encode(key));
        tracing::warn!(
            memory_id = %memory_id,
            read,
            skipped_since_open = skipped,
            error = %error,
            "memory record does not decode; skipped"
        );
    }

    /// Options for a write at this store's durability.
    fn write_options(&self) -> WriteOptions {
        crate::storage::rocksdb_write_options(self.sync_writes)
    }

    /// Load — or, on first run, create — the record encryptor + index blinder from the keystore.
    #[allow(clippy::type_complexity)]
    fn open_record_cryptor(
        db: &DB,
        storage_path: &Path,
        create_keystore: bool,
        sync_writes: bool,
    ) -> Result<
        Option<(
            crate::keystore::RecordCryptors,
            crate::keystore::IndexBlinder,
        )>,
    > {
        let keystore = storage_path.join("keystore.json");
        if !create_keystore && !keystore.exists() {
            return Ok(None);
        }
        let Some(keys) = crate::keystore::open_record_keys(&keystore)? else {
            return Ok(None);
        };
        Self::check_keystore_generation(db, keys.generation, sync_writes)?;
        Ok(Some((keys.cryptors, keys.blinder)))
    }

    /// Rollback guard: refuse a keystore whose generation is older than the last
    /// one this database observed; advance the stored sentinel otherwise.
    fn check_keystore_generation(db: &DB, generation: u64, sync_writes: bool) -> Result<()> {
        let cf = db
            .cf_handle(CF_INDEX)
            .ok_or_else(|| anyhow!("memory_index CF missing for keystore generation sentinel"))?;
        let stored = db
            .get_cf(cf, KEYSTORE_GENERATION_KEY)
            .context("read keystore generation sentinel")?
            .and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok())
            .map(u64::from_le_bytes)
            .unwrap_or(0);
        if let Some(next) = crate::keystore::keystore_generation_sentinel(stored, generation)? {
            db.put_cf_opt(
                cf,
                KEYSTORE_GENERATION_KEY,
                next.to_le_bytes(),
                &crate::storage::rocksdb_write_options(sync_writes),
            )
            .context("write keystore generation sentinel")?;
        }
        Ok(())
    }

    // ========================================================================
    // RECORD-LEVEL ENCRYPTION HELPERS (encryption-v2)
    // serialize_memory encrypts the whole serialized record after bincode;
    // decode_stored decrypts before deserialize. All storage paths route raw
    // bytes through these instead of calling bincode/deserialize_memory directly.
    // ========================================================================

    /// Serialize a memory to bytes. With record-level encryption enabled the whole
    /// serialized record is envelope-encrypted (`ENC\0|ver|epoch|nonce|ct`);
    /// otherwise it is plaintext bincode (backward compatible).
    fn serialize_memory(&self, memory: &Memory) -> Result<Vec<u8>> {
        let plaintext = bincode::serde::encode_to_vec(memory, bincode::config::standard())
            .context("Failed to serialize memory")?;
        match &self.record_cryptors {
            Some(cryptors) => cryptors
                .active()
                // Bind the record to its identity: AAD = the memory-id (== the
                // primary-CF key, id.0.as_bytes()). A ciphertext lifted from one
                // key and replanted at another fails to decrypt (anti-swap /
                // anti-relocation), and tamper of the stored id is caught.
                .encrypt_record(&plaintext, memory.id.0.as_bytes())
                .context("Failed to encrypt memory record"),
            None => Ok(plaintext),
        }
    }

    /// Decode a stored record: decrypt first if it carries the record-envelope
    /// marker, then deserialize. Plaintext (legacy) records pass straight through.
    ///
    /// Errors if a record is encrypted but no key is configured, or if decryption
    /// fails — callers must NOT treat that as a plaintext record.
    ///
    /// `id` is the record's identity (its primary-CF key, id.0.as_bytes()), bound
    /// into the AEAD AAD by serialize_memory. Pass the rocksdb key the value was
    /// read at; a mismatch fails decryption (anti-swap). Unused on the plaintext
    /// (legacy) path.
    fn decode_stored(&self, id: &[u8], data: &[u8]) -> Result<(Memory, bool)> {
        if crate::keystore::is_encrypted_record(data) {
            let cryptors = self.record_cryptors.as_ref().ok_or_else(|| {
                anyhow!("encountered an encrypted record but no keystore/key is configured")
            })?;
            let epoch = crate::keystore::record_epoch(data).unwrap_or(0);
            // Select the DEK for THIS record's epoch — records written before a key
            // rotation decrypt under their original (retired) epoch key.
            let cryptor = cryptors.for_epoch(epoch).ok_or_else(|| {
                anyhow!(
                    "record epoch {epoch} has no DEK in the keystore (active epoch {}); \
                     the keystore may have been replaced or the epoch's key pruned",
                    cryptors.active_epoch()
                )
            })?;
            let plaintext = cryptor
                .decrypt_record(data, id)
                .context("Failed to decrypt memory record")?;
            open_legacy_field(deserialize_memory(&plaintext)?)
        } else {
            open_legacy_field(deserialize_memory(data)?)
        }
    }

    /// Blind an exact-match index term (encryption-v2 P3). Returns the HMAC token
    /// when index blinding is enabled, else the term unchanged (plaintext index,
    /// backward compatible). Equal terms map to equal tokens so lookups still work.
    fn blind_term(&self, term: &str) -> String {
        match &self.index_blinder {
            Some(blinder) => blinder.blind(term),
            None => term.to_string(),
        }
    }

    /// Open a RocksDB database with column families, automatically repairing if corruption is detected.
    ///
    /// On hard kills (ONNX deadlock, OOM, kill -9), RocksDB SST files can be left
    /// in a partially written state. This attempts repair before giving up.
    ///
    /// Takes a builder closure because `ColumnFamilyDescriptor` is not `Clone` —
    /// we need to rebuild descriptors for the retry path after repair.
    fn open_or_repair_cf<F>(opts: &Options, path: &Path, build_cfs: F) -> Result<DB>
    where
        F: Fn() -> Vec<ColumnFamilyDescriptor>,
    {
        match DB::open_cf_descriptors(opts, path, build_cfs()) {
            Ok(db) => Ok(db),
            Err(open_err) => {
                let err_str = open_err.to_string();
                // Only attempt repair for corruption-related errors
                if err_str.contains("Corruption")
                    || err_str.contains("bad block")
                    || err_str.contains("checksum mismatch")
                    || err_str.contains("MANIFEST")
                    || err_str.contains("CURRENT")
                {
                    tracing::warn!(
                        error = %open_err,
                        "RocksDB corruption detected in memory storage, attempting repair"
                    );
                    if let Err(repair_err) = DB::repair(opts, path) {
                        tracing::error!(
                            error = %repair_err,
                            "RocksDB repair failed for memory storage"
                        );
                        return Err(anyhow::anyhow!(
                            "Failed to open or repair memory storage: open={open_err}, repair={repair_err}"
                        ));
                    }
                    tracing::info!("RocksDB repair succeeded for memory storage, reopening");
                    DB::open_cf_descriptors(opts, path, build_cfs()).map_err(|e| {
                        anyhow::anyhow!("Failed to open memory storage after repair: {e}")
                    })
                } else {
                    Err(anyhow::anyhow!("Failed to open memory storage: {open_err}"))
                }
            }
        }
    }

    /// Migrate from old separate-DB layout (memories/ + memory_index/) to single CF-based DB.
    ///
    /// Detects whether old directories exist and the new CFs are empty, then bulk-copies
    /// all data into the unified DB. Old directories are renamed to *.pre_cf_migration
    /// for rollback safety.
    fn migrate_from_separate_dbs(base_path: &Path, db: &DB, sync_writes: bool) -> Result<()> {
        let write_opts = crate::storage::rocksdb_write_options(sync_writes);
        let old_memories_dir = base_path.join("memories");
        let old_index_dir = base_path.join("memory_index");

        let has_old_memories = old_memories_dir.is_dir();
        let has_old_index = old_index_dir.is_dir();

        if !has_old_memories && !has_old_index {
            return Ok(());
        }

        tracing::info!("Detected old separate-DB layout, migrating to column families...");
        let mut total_migrated = 0usize;

        // Migrate main memories → default CF
        if has_old_memories {
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_memories_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for item in old_db.iterator(IteratorMode::Start) {
                        let (key, value) = item.map_err(|e| {
                            anyhow::anyhow!("RocksDB iterator error during memory migration: {e}")
                        })?;
                        batch.put(&key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write_opt(std::mem::take(&mut batch), &write_opts)?;
                            tracing::info!("  memories: migrated {count} entries...");
                        }
                    }
                    if !batch.is_empty() {
                        db.write_opt(batch, &write_opts)?;
                    }
                    drop(old_db);
                    total_migrated += count;
                    tracing::info!("  memories: migrated {count} entries to default CF");

                    let backup_name = base_path.join("memories.pre_cf_migration");
                    if backup_name.exists() {
                        let _ = std::fs::remove_dir_all(&backup_name);
                    }
                    if let Err(e) = std::fs::rename(&old_memories_dir, &backup_name) {
                        tracing::warn!("Could not rename old memories dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old memories DB for migration: {e}");
                }
            }
        }

        // Migrate index → memory_index CF
        if has_old_index {
            let index_cf = db
                .cf_handle(CF_INDEX)
                .expect("memory_index CF must exist during migration");

            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_index_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for item in old_db.iterator(IteratorMode::Start) {
                        let (key, value) = item.map_err(|e| {
                            anyhow::anyhow!("RocksDB iterator error during index migration: {e}")
                        })?;
                        batch.put_cf(&index_cf, &key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write_opt(std::mem::take(&mut batch), &write_opts)?;
                            tracing::info!("  index: migrated {count} entries...");
                        }
                    }
                    if !batch.is_empty() {
                        db.write_opt(batch, &write_opts)?;
                    }
                    drop(old_db);
                    total_migrated += count;
                    tracing::info!("  index: migrated {count} entries to {CF_INDEX} CF");

                    let backup_name = base_path.join("memory_index.pre_cf_migration");
                    if backup_name.exists() {
                        let _ = std::fs::remove_dir_all(&backup_name);
                    }
                    if let Err(e) = std::fs::rename(&old_index_dir, &backup_name) {
                        tracing::warn!("Could not rename old memory_index dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old memory_index DB for migration: {e}");
                }
            }
        }

        if total_migrated > 0 {
            tracing::info!(
                "Memory storage migration complete: {total_migrated} total entries migrated"
            );
        }

        Ok(())
    }

    /// Get the base storage path
    pub fn path(&self) -> &Path {
        &self.storage_path
    }

    /// Run a write, clearing the write latch on success and making one bounded
    /// flush+retry attempt when RocksDB refused it for want of disk space.
    ///
    /// `op` is called at most twice, so it must be replayable.
    fn guarded_write<T>(&self, op: impl Fn() -> Result<T>) -> Result<T> {
        crate::storage::write_latch::guard(&self.db, op)
    }

    /// [`Self::guarded_write`] for a batch already built.
    fn guarded_batch(&self, batch: &WriteBatch, opts: &WriteOptions) -> Result<()> {
        crate::storage::write_latch::guard_batch(&self.db, batch, opts)
    }

    /// Store a memory at the store's durability (`VELD_STORAGE_DURABILITY`).
    /// - eventual (default): <1ms per write, data survives process crashes
    /// - immediate: 2-10ms per write, data survives power loss
    ///
    /// For robotics/edge: Use eventual + periodic flush() calls for best latency.
    /// For compliance/critical: Set VELD_STORAGE_DURABILITY=immediate.
    ///
    /// The record, the removal of every index row the version it replaces owned, and
    /// the new version's index rows are one atomic write, so a store over a live id
    /// never leaves lookups answering for values the memory no longer has.
    pub fn store(&self, memory: &Memory) -> Result<()> {
        let value = self
            .serialize_memory(memory)
            .context(format!("Failed to serialize memory {}", memory.id.0))?;

        // Rebuilt per attempt: `write_opt` consumes the batch and the write latch may replay it.
        let build = || {
            let mut batch = WriteBatch::default();
            self.queue_replaced_index_removals(&memory.id, &mut batch);
            batch.put(memory.id.0.as_bytes(), &value);
            self.queue_index_writes(memory, &mut batch);
            batch
        };
        let opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .write_opt(build(), &opts)
                .with_context(|| format!("Failed to write memory {} to RocksDB", memory.id.0))
        })
    }

    /// [`Self::store`] for many memories in one atomic write. Each record's replaced
    /// index rows are read from the store as it stands before the write, so a memory
    /// should appear once.
    pub fn store_many(&self, memories: &[Memory]) -> Result<()> {
        if memories.is_empty() {
            return Ok(());
        }
        let values = memories
            .iter()
            .map(|memory| {
                self.serialize_memory(memory)
                    .context(format!("Failed to serialize memory {}", memory.id.0))
            })
            .collect::<Result<Vec<_>>>()?;

        // Rebuilt per attempt: `write_opt` consumes the batch and the write latch may replay it.
        let build = || {
            let mut batch = WriteBatch::default();
            for (memory, value) in memories.iter().zip(&values) {
                self.queue_replaced_index_removals(&memory.id, &mut batch);
                batch.put(memory.id.0.as_bytes(), value);
                self.queue_index_writes(memory, &mut batch);
            }
            batch
        };
        let opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .write_opt(build(), &opts)
                .with_context(|| format!("Failed to write {} memories to RocksDB", memories.len()))
        })
    }

    /// Queue the removal of the index rows of whatever version is stored under `id`.
    /// Its rows are named by its fields, so it is read before the write that replaces it.
    fn queue_replaced_index_removals(&self, id: &MemoryId, batch: &mut WriteBatch) {
        match self.get_if_present(id) {
            Ok(Some(previous)) => self.queue_index_removals(id, &previous, batch),
            Ok(None) => {}
            Err(e) => self.remove_undecodable_inbound_refs(id, &e),
        }
    }

    /// Update secondary indices for efficient retrieval
    /// Rebuild the entire secondary index (`memory_index` CF) from the primary
    /// records. Required after the index-blinding key changes (e.g. KEK rotation):
    /// clears the index then re-derives every memory's entries with the current
    /// blinder. Idempotent and record-safe — records are never touched; `meta:`
    /// keys are preserved. Run offline / single-writer. Returns the record count.
    pub fn rebuild_index(&self) -> Result<usize> {
        let idx = self.index_cf();
        let stale: Vec<Box<[u8]>> = self
            .db
            .iterator_cf(idx, IteratorMode::Start)
            .filter_map(|r| r.ok().map(|(k, _)| k))
            .filter(|k| !k.starts_with(b"meta:"))
            .collect();
        let mut batch = rocksdb::WriteBatch::default();
        for k in &stale {
            batch.delete_cf(idx, k.as_ref());
        }
        self.guarded_batch(&batch, &self.write_options())
            .context("Failed to clear stale index entries during rebuild")?;

        // Forgotten memories too: a soft delete rewrites the record and leaves its index
        // rows, so a rebuild that skipped them would answer differently from the store
        // it replaces, and from redb's rebuild.
        let memories = self.decode_all_memories(IncludeForgotten::Yes);
        for memory in &memories {
            self.update_indices(memory)?;
        }
        self.flush()?;
        Ok(memories.len())
    }

    fn update_indices(&self, memory: &Memory) -> Result<()> {
        let mut batch = WriteBatch::default();
        self.queue_index_writes(memory, &mut batch);
        self.guarded_batch(&batch, &self.write_options())
    }

    /// Add to `batch` every index row `memory` owns.
    fn queue_index_writes(&self, memory: &Memory, batch: &mut WriteBatch) {
        let idx = self.index_cf();

        // === Standard Indices ===

        // Index by date (for temporal queries)
        // BUG-001 FIX: Include memory_id in key to allow multiple memories per day
        // Old format: date:YYYYMMDD (overwrites on same day)
        // New format: date:YYYYMMDD:uuid (unique per memory)
        let date_key = format!(
            "date:{}:{}",
            memory.created_at.format("%Y%m%d"),
            memory.id.0
        );
        batch.put_cf(idx, date_key.as_bytes(), b"1");

        // Index by type
        let type_key = format!(
            "type:{:?}:{}",
            memory.experience.experience_type, memory.id.0
        );
        batch.put_cf(idx, type_key.as_bytes(), b"1");

        // Index by importance (quantized into buckets)
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, memory.id.0);
        batch.put_cf(idx, importance_key.as_bytes(), b"1");

        // Index by entities (case-insensitive for tag search compatibility)
        for entity in &memory.experience.entities {
            let normalized_entity = entity.to_lowercase();
            let entity_key = format!(
                "entity:{}:{}",
                self.blind_term(&normalized_entity),
                memory.id.0
            );
            batch.put_cf(idx, entity_key.as_bytes(), b"1");
        }

        // Index by tags (separate from entities for explicit tag queries)
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", self.blind_term(&normalized_tag), memory.id.0);
            batch.put_cf(idx, tag_key.as_bytes(), b"1");
        }

        // Index by episode_id (for temporal/episodic retrieval)
        // Episode is the primary temporal grouping - memories in same episode are highly related
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!(
                    "episode:{}:{}",
                    self.blind_term(&episode_id.to_string()),
                    memory.id.0
                );
                batch.put_cf(idx, episode_key.as_bytes(), b"1");

                // Also index by sequence within episode for temporal ordering
                if let Some(seq) = ctx.episode.sequence_number {
                    let seq_key = format!(
                        "episode_seq:{}:{}:{}",
                        self.blind_term(&episode_id.to_string()),
                        seq,
                        memory.id.0
                    );
                    batch.put_cf(idx, seq_key.as_bytes(), b"1");
                }
            }
        }

        // === Robotics Indices ===

        // Index by robot_id (for multi-robot systems). Read via
        // resolved_robot_id so the W3 WhoFacet SelfAgent indexes too.
        if let Some(robot_id) = memory.experience.resolved_robot_id() {
            let robot_key = format!("robot:{}:{}", self.blind_term(robot_id), memory.id.0);
            batch.put_cf(idx, robot_key.as_bytes(), b"1");
        }

        // Index by mission_id (for mission context retrieval). Read via
        // resolved_mission_id so the W3 WhyFacet goal_stack first entry counts.
        if let Some(mission_id) = memory.experience.resolved_mission_id() {
            let mission_key = format!("mission:{}:{}", self.blind_term(mission_id), memory.id.0);
            batch.put_cf(idx, mission_key.as_bytes(), b"1");
        }

        // Index by geo_location (for spatial queries) using geohash.
        // Key format: geo:GEOHASH:memory_id (geohash precision 10 = ~1.2m x 60cm).
        // Read via resolved_geo so both Place::Geo (W3) and the legacy flat
        // field are indexed.
        //
        // SECURITY (residual): the geohash is stored UNBLINDED, unlike the
        // exact-match index terms. Spatial search scans by geohash *prefix*
        // (proximity), so HMAC-blinding would destroy the prefix structure and
        // break nearby-queries. Precise location (~1.2m) is therefore visible in
        // the index CF to anyone who can read it — same class as the date/type/
        // importance range keys. See docs/ENCRYPTION-SECURITY.md.
        if let Some((lat, lon, _alt)) = memory.experience.resolved_geo() {
            let geohash = super::types::geohash_encode(lat, lon, GEO_INDEX_PRECISION);
            let geo_key = format!("geo:{}:{}", geohash, memory.id.0);
            batch.put_cf(idx, geo_key.as_bytes(), b"1");
        }

        // Index by action_type (for action-based retrieval)
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!(
                "action:{}:{}",
                self.blind_term(&action_type.to_string()),
                memory.id.0
            );
            batch.put_cf(idx, action_key.as_bytes(), b"1");
        }

        // Index by reward (bucketed, for RL-style queries)
        // Bucket: -1.0 to 1.0 mapped to 0-20
        if let Some(reward) = memory.experience.reward {
            let clamped_reward = reward.clamp(-1.0, 1.0);
            let reward_bucket = ((clamped_reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, memory.id.0);
            batch.put_cf(idx, reward_key.as_bytes(), b"1");
        }

        // === Content Hash Index (idempotency, issue #109) ===
        // One row per memory, `content_hash:{hash}:{memory_id}`, so memories that share
        // content each keep a row and deleting one never hides the others.
        batch.put_cf(idx, self.content_hash_row(memory).as_bytes(), b"1");

        // === External Linking Index ===
        // Index by external_id for upsert operations (Linear, GitHub, etc.)
        // Key format: external:{source}:{id}:{memory_id} -> memory_id
        // Enables O(1) lookup when syncing from external systems
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!(
                "external:{}:{}",
                self.blind_term(&external_id.to_string()),
                memory.id.0
            );
            // Store memory_id as value for direct lookup
            batch.put_cf(idx, external_key.as_bytes(), memory.id.0.as_bytes());
        }

        // === Hierarchy Index ===
        // Index by parent_id for tree queries (list children, build tree)
        // Key format: parent:{parent_id}:{child_id} -> 1
        // Enables O(1) lookup of all children for a parent
        if let Some(ref parent_id) = memory.parent_id {
            let parent_key = format!(
                "parent:{}:{}",
                self.blind_term(&parent_id.0.to_string()),
                memory.id.0
            );
            batch.put_cf(idx, parent_key.as_bytes(), b"1");
        }

        // === Inbound-Reference Reverse Index (R7 / F2) ===
        // For every forward MemoryId reference this memory holds (it points AT a
        // target), record the inverse edge so we can answer "does anything still
        // point at memory X?" in O(1) via a prefix scan, without scanning the
        // whole corpus. Consumed by a later merge/GC step (R6) to gate
        // garbage-collection on inbound_refs == 0. Key format:
        //   inbound_ref:{blinded(target_id)}:{holder_id} -> {holder_id bytes}
        // De-duplicated so a memory that references the same target twice (e.g.
        // both related_memories and a CausalLink) yields a single entry.
        for target in memory.forward_ref_targets() {
            let inbound_key = self.inbound_ref_key(&target, &memory.id);
            batch.put_cf(idx, inbound_key.as_bytes(), memory.id.0.as_bytes());
        }

        if !memory.compressed && !memory.is_forgotten() {
            let key = uncompressed_index_key(memory.created_at, &memory.id);
            batch.put_cf(idx, key.as_bytes(), b"1");
        }
    }

    /// `inbound_ref:{blinded(target)}:{holder}`, the reverse-index entry recording that
    /// `holder` points at `target`.
    fn inbound_ref_key(&self, target: &MemoryId, holder: &MemoryId) -> String {
        format!(
            "inbound_ref:{}:{}",
            self.blind_term(&target.0.to_string()),
            holder.0
        )
    }

    /// Public query API (R7) — enumerate the holders that still point AT `target`.
    ///
    /// Prefix-scans the inbound-reference reverse index for
    /// `inbound_ref:{blinded(target)}:` and returns the distinct holder
    /// `MemoryId`s. A later merge/GC step (R6) consumes this to know which
    /// records must be repointed before a target can be reclaimed. Per-user
    /// isolated: this store is the per-user `MemoryStorage`, so it can never
    /// return another tenant's holders.
    pub fn inbound_refs(&self, target: &MemoryId) -> Result<Vec<MemoryId>> {
        let prefix = format!("inbound_ref:{}:", self.blind_term(&target.0.to_string()));
        let idx = self.index_cf();
        let iter = self.db.iterator_cf(
            idx,
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        let mut holders = Vec::new();
        for (key, value) in iter.log_errors() {
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            // The holder id is stored as the value (16 raw UUID bytes). Fall back
            // to parsing it out of the key suffix if the value is unexpected.
            let holder = if value.len() == 16 {
                uuid::Uuid::from_slice(&value).ok().map(MemoryId)
            } else {
                let key_str = String::from_utf8_lossy(&key);
                key_str
                    .strip_prefix(&prefix)
                    .and_then(|s| uuid::Uuid::parse_str(s).ok())
                    .map(MemoryId)
            };
            if let Some(id) = holder {
                holders.push(id);
            }
        }
        Ok(holders)
    }

    /// Public query API (R7) — short-circuit existence check for the GC gate.
    ///
    /// Returns `true` as soon as a single holder is found, without materializing
    /// the full holder list. R6's GC reads this to refuse reclaiming a target
    /// that anything still references (the "zero inbound refs" gate).
    pub fn has_inbound_refs(&self, target: &MemoryId) -> Result<bool> {
        let prefix = format!("inbound_ref:{}:", self.blind_term(&target.0.to_string()));
        let idx = self.index_cf();
        let mut iter = self.db.iterator_cf(
            idx,
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        match iter.next() {
            Some(Ok((key, _))) => Ok(key.starts_with(prefix.as_bytes())),
            _ => Ok(false),
        }
    }

    // =========================================================================
    // R7b — Fold-3 fact-source coverage for the inbound-ref GC gate
    // =========================================================================
    //
    // `has_inbound_refs` (R7) only covers the Memory-store reverse index.
    // Two additional citation sources exist that can make a memory non-reclaimable
    // even when its Memory-store ref-count drops to zero:
    //
    //   (F3a) `SemanticFact.source_memories: Vec<MemoryId>`  — facts.rs
    //   (F3b) `TemporalFact.source_memory_id: MemoryId`      — temporal_facts.rs
    //
    // Both stores share this RocksDB instance via `self.db` (default CF, key
    // prefixes `facts:{user_id}:…` and `temporal_facts:{user_id}:…`).  No new
    // state is required — we do a short-circuit prefix scan and bail as soon as
    // one citing record is found.  This is a READ-ONLY path; nothing is written.
    //
    // Call `has_any_inbound_refs(target, user_id)` from the GC gate in place of
    // (or in addition to) `has_inbound_refs`; the former is the complete guard.

    /// Returns `true` as soon as a `SemanticFact` is found whose `source_memories`
    /// contains `target`.  Scans `facts:{user_id}:*` records in the default CF.
    /// Short-circuits on the first hit; inactive/expired facts are included because
    /// a fact that references a memory is still a provenance citation regardless of
    /// its validity window.
    fn semantic_fact_cites_memory(&self, target: &MemoryId, user_id: &str) -> Result<bool> {
        use super::facts::decode_for_gc as decode_semantic;
        let prefix = format!("facts:{}:", user_id);
        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Skip secondary index keys (entity/type indices have >2 colons after
            // the `facts:` prefix — exact primary keys are `facts:{uid}:{fact_id}`
            // which have exactly 2 colons).
            if key_str.matches(':').count() > 2 {
                continue;
            }
            if let Ok(fact) = decode_semantic(&value) {
                if fact.source_memories.contains(target) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Returns `true` as soon as a `TemporalFact` is found whose `source_memory_id`
    /// equals `target`.  Scans `temporal_facts:{user_id}:*` records in the default CF.
    /// Short-circuits on the first hit; expired facts are included (same provenance
    /// reasoning as for `semantic_fact_cites_memory`).
    fn temporal_fact_cites_memory(&self, target: &MemoryId, user_id: &str) -> Result<bool> {
        use super::temporal_facts::decode_for_gc as decode_temporal;
        let prefix = format!("temporal_facts:{}:", user_id);
        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Skip secondary index entries (`temporal_by_entity:…`, `temporal_by_event:…`)
            // — primary keys have the form `temporal_facts:{uid}:{fact_id}` (2 colons
            // in the `temporal_facts:` portion).
            if key_str.matches(':').count() > 2 {
                continue;
            }
            if let Ok(fact) = decode_temporal(&value) {
                if fact.source_memory_id == *target {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Full GC guard (R7b) — returns `true` if `target` has ANY inbound reference
    /// from the three citation sources:
    ///
    ///   1. Memory-store reverse index (R7 / F2) — `has_inbound_refs`
    ///   2. `SemanticFact.source_memories`        — `semantic_fact_cites_memory`
    ///   3. `TemporalFact.source_memory_id`        — `temporal_fact_cites_memory`
    ///
    /// The future merge/GC pass (R6) MUST call this instead of bare
    /// `has_inbound_refs` to avoid orphaning facts whose only provenance anchor
    /// is a `source_memory_id` / `source_memories` entry.
    ///
    /// `user_id` is the tenant prefix used by both fact stores to namespace their
    /// keys.  Pass the same `user_id` that was used when storing the facts.
    pub fn has_any_inbound_refs(&self, target: &MemoryId, user_id: &str) -> Result<bool> {
        if self.has_inbound_refs(target)? {
            return Ok(true);
        }
        if self.semantic_fact_cites_memory(target, user_id)? {
            return Ok(true);
        }
        self.temporal_fact_cites_memory(target, user_id)
    }

    /// Compute SHA256 hex digest for content dedup indexing (issue #109)
    fn sha256_content_hash(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// `content_hash:{blinded sha256}:`, the prefix every content-hash row for `content`
    /// shares.
    fn content_hash_prefix(&self, content: &str) -> String {
        format!(
            "content_hash:{}:",
            self.blind_term(&Self::sha256_content_hash(content))
        )
    }

    fn content_hash_row(&self, memory: &Memory) -> String {
        self.content_hash_prefix(&memory.experience.content) + &memory.id.0.to_string()
    }

    /// Look up an existing memory by content hash (idempotency dedup, issue #109).
    ///
    /// When several memories hold this content the lowest id answers, which is the
    /// first row in key order (a hyphenated uuid sorts as its bytes do) and redb's
    /// answer. The record is read to confirm the content, so a digest collision cannot
    /// answer, and rows whose memory is gone or rewritten are removed. A candidate that
    /// will not decode answers `None`, as on redb.
    pub fn get_by_content_hash(&self, content: &str) -> Option<MemoryId> {
        let prefix = self.content_hash_prefix(content);
        let idx = self.index_cf();

        let mut found = None;
        let mut dangling: Vec<Box<[u8]>> = Vec::new();
        for (key, _value) in self
            .db
            .prefix_iterator_cf(idx, prefix.as_bytes())
            .log_errors()
        {
            let Some(id_bytes) = key.strip_prefix(prefix.as_bytes()) else {
                break;
            };
            let Some(id) = std::str::from_utf8(id_bytes)
                .ok()
                .and_then(|id_str| uuid::Uuid::parse_str(id_str).ok())
                .map(MemoryId)
            else {
                continue;
            };
            match self.get_if_present(&id).ok()? {
                Some(memory) if memory.experience.content == content => {
                    found = Some(id);
                    break;
                }
                _ => dangling.push(key),
            }
        }

        if !dangling.is_empty() {
            let mut batch = WriteBatch::default();
            for key in &dangling {
                batch.delete_cf(idx, key);
            }
            if let Err(e) = self.guarded_batch(&batch, &self.write_options()) {
                tracing::warn!(
                    rows = dangling.len(),
                    error = %e,
                    "could not remove content-hash index rows that name no matching memory"
                );
            }
        }

        found
    }

    /// Rewrite the content-hash rows of a store written before they were one per
    /// memory, `content_hash:{hash}` holding the id of the memory stored last, into
    /// `content_hash:{hash}:{id}`. One batch, so a crash leaves the old layout whole.
    /// The marker is written only into a store that holds rows, so an opened empty
    /// store stays empty for the boot guard and for `veld migrate`.
    fn convert_single_holder_content_hash_rows(db: &DB, sync_writes: bool) -> Result<usize> {
        let Some(idx) = db.cf_handle(CF_INDEX) else {
            return Ok(0);
        };
        if db.get_cf(idx, CONTENT_HASH_LAYOUT_MARKER)?.is_some() {
            return Ok(0);
        }

        let prefix = b"content_hash:";
        let mut saw_rows = false;
        let mut batch = WriteBatch::default();
        let mut converted = 0usize;
        for (key, value) in db.prefix_iterator_cf(idx, prefix).log_errors() {
            let Some(rest) = key.strip_prefix(prefix.as_slice()) else {
                break;
            };
            saw_rows = true;
            if rest.contains(&b':') {
                continue;
            }
            batch.delete_cf(idx, &key);
            if let Ok(holder) = uuid::Uuid::from_slice(&value) {
                let mut row = key.to_vec();
                row.push(b':');
                row.extend_from_slice(holder.to_string().as_bytes());
                batch.put_cf(idx, row, b"1");
                converted += 1;
            }
        }
        if saw_rows {
            batch.put_cf(idx, CONTENT_HASH_LAYOUT_MARKER, b"1");
            db.write_opt(batch, &crate::storage::rocksdb_write_options(sync_writes))
                .context("Failed to convert content-hash index rows to one row per memory")?;
        }
        Ok(converted)
    }

    /// Retrieve a memory by ID
    ///
    /// Performs lazy migration: if memory is in legacy format, re-writes it
    /// in current format for faster future reads.
    pub fn get(&self, id: &MemoryId) -> Result<Memory> {
        self.get_if_present(id)?
            .ok_or_else(|| anyhow!("Memory not found: {id:?}"))
    }

    /// [`Self::get`], answering an absent record with `None` instead of an error, so a
    /// caller can tell a missing memory from one that will not decode.
    fn get_if_present(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let key = id.0.as_bytes();
        let Some(value) = self.db.get(key)? else {
            return Ok(None);
        };
        let (memory, needs_migration) = self
            .decode_stored(key, &value)
            .with_context(|| format!("Failed to decode memory {} ({} bytes)", id.0, value.len()))?;

        // Lazy migration: re-write legacy formats in current format
        if needs_migration {
            if let Err(e) = self.migrate_memory_format(key, &memory) {
                // Migration failure is non-fatal - log and continue
                tracing::debug!("Lazy migration skipped for memory {}: {}", memory.id.0, e);
            }
        }

        Ok(Some(memory))
    }

    /// Re-write a memory in current format (lazy migration helper). Writes only under the key
    /// the bytes were read from: a decoder that took the id from the value itself would
    /// otherwise create a row under an id nothing else references.
    fn migrate_memory_format(&self, key: &[u8], memory: &Memory) -> Result<()> {
        if memory.id.0.as_bytes() != key {
            return Err(anyhow!(
                "decoded id {} does not match the stored key; not rewriting",
                memory.id.0
            ));
        }
        let value = self
            .serialize_memory(memory)
            .context("Failed to serialize for migration")?;

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(false); // Async is fine for migration

        self.guarded_write(|| Ok(self.db.put_opt(key, &value, &write_opts)?))?;
        tracing::debug!("Migrated memory {} to current format", memory.id.0);
        Ok(())
    }

    /// Find a memory by its external_id (e.g., "linear:SHO-39", "github:pr-123")
    ///
    /// Returns the memory if found, None if no memory with this external_id exists.
    /// Used for upsert operations when syncing from external sources.
    ///
    /// An index row naming a memory that is gone, or that no longer carries this
    /// external id, is skipped and removed: the record is the truth, as it is for redb.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        // Index key format: external:{external_id}:{memory_id}
        let prefix = format!("external:{}:", self.blind_term(external_id));
        let idx = self.index_cf();

        let iter = self.db.iterator_cf(
            idx,
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        let mut found = None;
        let mut dangling: Vec<Box<[u8]>> = Vec::new();
        for (key, _value) in iter.log_errors() {
            let Some(id_bytes) = key.strip_prefix(prefix.as_bytes()) else {
                break;
            };
            let Some(id) = std::str::from_utf8(id_bytes)
                .ok()
                .and_then(|id_str| uuid::Uuid::parse_str(id_str).ok())
                .map(MemoryId)
            else {
                continue;
            };
            match self.get_if_present(&id)? {
                Some(memory) if memory.external_id.as_deref() == Some(external_id) => {
                    found = Some(memory);
                    break;
                }
                _ => dangling.push(key),
            }
        }

        if !dangling.is_empty() {
            let mut batch = WriteBatch::default();
            for key in &dangling {
                batch.delete_cf(idx, key);
            }
            if let Err(e) = self.guarded_batch(&batch, &self.write_options()) {
                tracing::warn!(
                    rows = dangling.len(),
                    error = %e,
                    "could not remove external-id index rows that name no matching memory"
                );
            }
        }

        Ok(found)
    }

    /// Update an existing memory: [`Self::store`], which replaces the previous version's
    /// index rows in the write that replaces the record (ALGO-004: Hebbian feedback
    /// moves importance between buckets, and a stale bucket row would keep answering).
    pub fn update(&self, memory: &Memory) -> Result<()> {
        self.store(memory)
    }

    /// Delete a memory, its vector mapping and its index rows: [`Self::delete_with_vectors`].
    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        self.delete_with_vectors(id)
    }

    /// Add to `batch` the deletion of every index row `memory`, stored under `id`, owns.
    /// BUG-005 FIX: Direct key deletion instead of O(n) scan with contains()
    /// We reconstruct index keys from memory metadata for O(k) deletion
    fn queue_index_removals(&self, id: &MemoryId, memory: &Memory, batch: &mut WriteBatch) {
        let idx = self.index_cf();

        // Reconstruct and delete all index keys directly (O(k) instead of O(n))

        // Date index
        let date_key = format!("date:{}:{}", memory.created_at.format("%Y%m%d"), id.0);
        batch.delete_cf(idx, date_key.as_bytes());

        // Type index
        let type_key = format!("type:{:?}:{}", memory.experience.experience_type, id.0);
        batch.delete_cf(idx, type_key.as_bytes());

        // Importance index
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, id.0);
        batch.delete_cf(idx, importance_key.as_bytes());

        // Entity indices (must match the to_lowercase() normalization in update_indices)
        for entity in &memory.experience.entities {
            let normalized_entity = entity.to_lowercase();
            let entity_key = format!("entity:{}:{}", self.blind_term(&normalized_entity), id.0);
            batch.delete_cf(idx, entity_key.as_bytes());
        }

        // Tag indices
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", self.blind_term(&normalized_tag), id.0);
            batch.delete_cf(idx, tag_key.as_bytes());
        }

        // Episode indices
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!(
                    "episode:{}:{}",
                    self.blind_term(&episode_id.to_string()),
                    id.0
                );
                batch.delete_cf(idx, episode_key.as_bytes());

                if let Some(seq) = ctx.episode.sequence_number {
                    let seq_key = format!(
                        "episode_seq:{}:{}:{}",
                        self.blind_term(&episode_id.to_string()),
                        seq,
                        id.0
                    );
                    batch.delete_cf(idx, seq_key.as_bytes());
                }
            }
        }

        // Robot index — must mirror update_indices's resolved_robot_id read.
        if let Some(robot_id) = memory.experience.resolved_robot_id() {
            let robot_key = format!("robot:{}:{}", self.blind_term(robot_id), id.0);
            batch.delete_cf(idx, robot_key.as_bytes());
        }

        // Mission index — must mirror update_indices's resolved_mission_id read.
        if let Some(mission_id) = memory.experience.resolved_mission_id() {
            let mission_key = format!("mission:{}:{}", self.blind_term(mission_id), id.0);
            batch.delete_cf(idx, mission_key.as_bytes());
        }

        // Geo index — must mirror update_indices's resolved_geo read.
        if let Some((lat, lon, _alt)) = memory.experience.resolved_geo() {
            let geohash = super::types::geohash_encode(lat, lon, GEO_INDEX_PRECISION);
            let geo_key = format!("geo:{}:{}", geohash, id.0);
            batch.delete_cf(idx, geo_key.as_bytes());
        }

        // Action index
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!(
                "action:{}:{}",
                self.blind_term(&action_type.to_string()),
                id.0
            );
            batch.delete_cf(idx, action_key.as_bytes());
        }

        // Reward index (must match the clamp in update_indices)
        if let Some(reward) = memory.experience.reward {
            let clamped_reward = reward.clamp(-1.0, 1.0);
            let reward_bucket = ((clamped_reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, id.0);
            batch.delete_cf(idx, reward_key.as_bytes());
        }

        batch.delete_cf(
            idx,
            self.content_hash_prefix(&memory.experience.content) + &id.0.to_string(),
        );

        // External linking index
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!(
                "external:{}:{}",
                self.blind_term(&external_id.to_string()),
                id.0
            );
            batch.delete_cf(idx, external_key.as_bytes());
        }

        // Parent index (hierarchy)
        if let Some(ref parent_id) = memory.parent_id {
            let parent_key = format!(
                "parent:{}:{}",
                self.blind_term(&parent_id.0.to_string()),
                id.0
            );
            batch.delete_cf(idx, parent_key.as_bytes());
        }

        // === Inbound-Reference Reverse Index teardown (R7 / F2) ===
        // `memory` is the record as stored, read before the write that replaces it, so its
        // targets are exactly the entries `queue_index_writes` wrote for it: targets an update DROPS are among them and are removed here, and the
        // new version's targets are written fresh by the store that follows. Entries
        // for which this memory is the TARGET belong to other holders and stay.
        for target in memory.forward_ref_targets() {
            batch.delete_cf(idx, self.inbound_ref_key(&target, id).as_bytes());
        }

        batch.delete_cf(
            idx,
            uncompressed_index_key(memory.created_at, id).as_bytes(),
        );
    }

    /// A stored record that does not decode cannot name its index rows. Its inbound_ref
    /// entries can still be found by their holder suffix, so those are removed by scan;
    /// the rest stay until `rebuild_index`.
    fn remove_undecodable_inbound_refs(&self, id: &MemoryId, error: &anyhow::Error) {
        tracing::warn!(
            memory_id = %id.0,
            error = %error,
            "stored record does not decode; removing its inbound_ref entries by scan"
        );
        if let Err(e) = self.remove_inbound_refs_by_holder_scan(id) {
            tracing::warn!(memory_id = %id.0, error = %e, "inbound_ref scan removal failed");
        }
    }

    /// Remove every `inbound_ref:*:{holder}` entry by scanning the whole reverse index.
    /// The holder is the key's suffix, not its prefix, so this is O(index); it is only for
    /// a holder whose stored record cannot be decoded to name its targets.
    fn remove_inbound_refs_by_holder_scan(&self, holder: &MemoryId) -> Result<()> {
        let idx = self.index_cf();
        let holder_suffix = format!(":{}", holder.0);
        let scan_prefix = b"inbound_ref:";
        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_cf_opt(
            idx,
            read_opts,
            IteratorMode::From(scan_prefix, rocksdb::Direction::Forward),
        );
        let mut batch = WriteBatch::default();
        for (key, _value) in iter.log_errors() {
            if !key.starts_with(scan_prefix) {
                break;
            }
            if key.ends_with(holder_suffix.as_bytes()) {
                batch.delete_cf(idx, key.as_ref());
            }
        }
        self.guarded_batch(&batch, &self.write_options())
    }

    /// Search memories by various criteria.
    ///
    /// The index scans below generate candidates — whole UTC days, importance tenths,
    /// reward twentieths, geohash cells — and every candidate is then held to
    /// [`SearchCriteria::matches`], the exact predicate redb answers with, so a bound
    /// inside a bucket returns what it names and nothing else in the bucket.
    pub fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        let exact = criteria.clone();
        let mut memory_ids = Vec::new();

        match criteria {
            // === Standard Criteria ===
            SearchCriteria::ByDate { start, end } => {
                memory_ids = self.search_by_date_range(start, end)?;
            }
            SearchCriteria::ByType(exp_type) => {
                memory_ids = self.search_by_type(exp_type)?;
            }
            SearchCriteria::ByImportance { min, max } => {
                memory_ids = self.search_by_importance(min, max)?;
            }
            SearchCriteria::ByEntity(entity) => {
                memory_ids = self.search_by_entity(&entity)?;
            }
            SearchCriteria::ByTags(tags) => {
                memory_ids = self.search_by_tags(&tags)?;
            }

            // === Temporal/Episode Criteria ===
            SearchCriteria::ByEpisode(episode_id) => {
                memory_ids = self.search_by_episode(&episode_id)?;
            }
            SearchCriteria::ByEpisodeSequence {
                episode_id,
                min_sequence,
                max_sequence,
            } => {
                memory_ids =
                    self.search_by_episode_sequence(&episode_id, min_sequence, max_sequence)?;
            }

            // === Robotics Criteria ===
            SearchCriteria::ByRobot(robot_id) => {
                memory_ids = self.search_by_robot(&robot_id)?;
            }
            SearchCriteria::ByMission(mission_id) => {
                memory_ids = self.search_by_mission(&mission_id)?;
            }
            SearchCriteria::ByLocation {
                lat,
                lon,
                radius_meters,
            } => {
                memory_ids = self.search_by_location(lat, lon, radius_meters)?;
            }
            SearchCriteria::ByActionType(action_type) => {
                memory_ids = self.search_by_action_type(&action_type)?;
            }
            SearchCriteria::ByReward { min, max } => {
                memory_ids = self.search_by_reward(min, max)?;
            }

            // === Compound Criteria ===
            SearchCriteria::Combined(criterias) => {
                // Intersection of all criteria results
                // Use HashSet for O(1) lookups instead of O(n) Vec::contains
                use std::collections::HashSet;
                let mut result_sets: Vec<HashSet<MemoryId>> = Vec::new();
                for c in criterias {
                    result_sets.push(
                        self.search(c)?
                            .into_iter()
                            .map(|m| m.id)
                            .collect::<HashSet<_>>(),
                    );
                }

                if !result_sets.is_empty() {
                    let first_set = result_sets.remove(0);
                    memory_ids = first_set
                        .into_iter()
                        .filter(|id| result_sets.iter().all(|set| set.contains(id)))
                        .collect();
                }
            }

            // === Hierarchy Criteria ===
            SearchCriteria::ByParent(parent_id) => {
                memory_ids = self.search_by_parent(&parent_id)?;
            }
            SearchCriteria::RootsOnly => {
                memory_ids = self.search_roots()?;
            }
        }

        // Fetch full memories, filtering out forgotten ones. A row whose memory is gone
        // is stale and passed over; one whose memory will not decode is counted.
        let mut seen = std::collections::HashSet::new();
        let mut memories = Vec::new();
        for id in memory_ids {
            if !seen.insert(id.clone()) {
                continue;
            }
            match self.get_if_present(&id) {
                Ok(Some(memory)) if !memory.is_forgotten() && exact.matches(&memory) => {
                    memories.push(memory)
                }
                Ok(_) => {}
                Err(e) => self.note_undecodable(id.0.as_bytes(), &e, "search"),
            }
        }

        Ok(memories)
    }

    fn search_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let start_key = format!("date:{}", start.format("%Y%m%d"));
        // BUG-001 FIX: End key needs ~ suffix to include all UUIDs for that date
        // Keys are: date:YYYYMMDD:uuid, so date:20251207~ comes after all Dec 7 entries
        let end_key = format!("date:{}~", end.format("%Y%m%d"));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _value) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if &*key_str > end_key.as_str() {
                break;
            }
            // BUG-001 FIX: Extract memory_id from key (format: date:YYYYMMDD:uuid)
            if key_str.starts_with("date:") {
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 3 {
                    // parts[0] = "date", parts[1] = "YYYYMMDD", parts[2] = uuid
                    if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_type(&self, exp_type: ExperienceType) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("type:{exp_type:?}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    fn search_by_importance(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let min_bucket = (min * 10.0) as u32;
        let max_bucket = (max * 10.0) as u32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("importance:{bucket}:");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                // Extract ID from key
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_entity(&self, entity: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        // Normalize to lowercase for case-insensitive matching
        let normalized_entity = entity.to_lowercase();
        let prefix = format!("entity:{}:", self.blind_term(&normalized_entity));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by tags (returns memories matching ANY of the provided tags)
    fn search_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        use std::collections::HashSet;

        // Union of all tag matches
        let mut all_ids = HashSet::new();

        for tag in tags {
            // Normalize to lowercase for case-insensitive matching
            let normalized_tag = tag.to_lowercase();
            let prefix = format!("tag:{}:", self.blind_term(&normalized_tag));
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );
            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        all_ids.insert(MemoryId(uuid));
                    }
                }
            }
        }

        // A HashSet has no order; sorted by id the result is repeatable. recall_by_tags orders by
        // recency itself before it truncates.
        let mut ids: Vec<MemoryId> = all_ids.into_iter().collect();
        ids.sort();
        Ok(ids)
    }

    /// Search memories by episode ID
    /// Returns all memories in the specified episode
    fn search_by_episode(&self, episode_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("episode:{}:", self.blind_term(episode_id));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by episode with sequence filtering
    /// Returns memories in temporal order within the episode
    fn search_by_episode_sequence(
        &self,
        episode_id: &str,
        min_sequence: Option<u32>,
        max_sequence: Option<u32>,
    ) -> Result<Vec<MemoryId>> {
        let mut results: Vec<(u32, MemoryId)> = Vec::new();

        // Scan the episode_seq index which has format:
        //   episode_seq:{blinded episode_id}:{seq}:{memory_id}
        // The episode_id is blinded (HMAC) to match update_indices; seq stays
        // plaintext as the in-episode ordering key (sorted in memory below).
        let prefix = format!("episode_seq:{}:", self.blind_term(episode_id));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }

            // Parse: episode_seq:{episode_id}:{seq}:{memory_id}
            if let Some(rest) = key_str.strip_prefix(&prefix) {
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() == 2 {
                    if let (Ok(seq), Ok(uuid)) =
                        (parts[0].parse::<u32>(), uuid::Uuid::parse_str(parts[1]))
                    {
                        // Apply sequence filters
                        let passes_min = min_sequence.is_none_or(|min| seq >= min);
                        let passes_max = max_sequence.is_none_or(|max| seq <= max);

                        if passes_min && passes_max {
                            results.push((seq, MemoryId(uuid)));
                        }
                    }
                }
            }
        }

        // Sort by sequence number for temporal ordering
        results.sort_by_key(|(seq, _)| *seq);

        Ok(results.into_iter().map(|(_, id)| id).collect())
    }

    // ========================================================================
    // ROBOTICS SEARCH METHODS
    // ========================================================================

    /// Search memories by robot/drone identifier
    fn search_by_robot(&self, robot_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("robot:{}:", self.blind_term(robot_id));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by mission identifier
    fn search_by_mission(&self, mission_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("mission:{}:", self.blind_term(mission_id));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by geographic location using geohash prefix scanning
    ///
    /// Performance: O(k) where k = memories in ~9 geohash cells covering the radius
    /// Previous approach was O(n) where n = all geo-indexed memories
    /// Candidates for a radius search: every memory indexed in a geohash cell the
    /// circle can reach, each held to the exact haversine by `search`.
    fn search_by_location(
        &self,
        center_lat: f64,
        center_lon: f64,
        radius_meters: f64,
    ) -> Result<Vec<MemoryId>> {
        let finite_query = center_lat.is_finite()
            && center_lon.is_finite()
            && !radius_meters.is_nan()
            && radius_meters >= 0.0;
        if !finite_query {
            return Ok(Vec::new());
        }
        let cells = covering_geohash_cells(center_lat, center_lon, radius_meters)
            .unwrap_or_else(|| vec![String::new()]);

        let mut ids = Vec::new();
        for cell in cells {
            // No trailing colon: a coarse cell prefixes every stored precision-10 hash in it.
            let prefix = format!("geo:{cell}");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );
            for (key, _value) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                // Key format: geo:GEOHASH:memory_id
                if let Some(uuid) = key_str
                    .rsplit(':')
                    .next()
                    .and_then(|id| uuid::Uuid::parse_str(id).ok())
                {
                    ids.push(MemoryId(uuid));
                }
            }
        }
        Ok(ids)
    }

    /// Search memories by action type
    fn search_by_action_type(&self, action_type: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("action:{}:", self.blind_term(action_type));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by reward range (for RL-style queries)
    fn search_by_reward(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();

        // Reward is bucketed similar to importance (-10 to 10 buckets)
        // Clamp to prevent bucket overflow from out-of-range values
        let clamped_min = min.clamp(-1.0, 1.0);
        let clamped_max = max.clamp(-1.0, 1.0);
        let min_bucket = ((clamped_min + 1.0) * 10.0) as i32; // -1.0 -> 0, 1.0 -> 20
        let max_bucket = ((clamped_max + 1.0) * 10.0) as i32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("reward:{bucket}:");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    // =========================================================================
    // HIERARCHY SEARCH METHODS
    // =========================================================================

    /// Get all children of a parent memory
    fn search_by_parent(&self, parent_id: &MemoryId) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("parent:{}:", self.blind_term(&parent_id.0.to_string()));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Key format: parent:{parent_uuid}:{child_uuid}
            if let Some(child_id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(child_id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Get all root memories (memories with no parent)
    fn search_roots(&self) -> Result<Vec<MemoryId>> {
        let mut roots = Vec::new();

        // Iterate all memories and check for parent_id = None
        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = self.decode_stored(&key, &value) {
                if memory.parent_id.is_none() {
                    roots.push(memory.id);
                }
            }
        }

        Ok(roots)
    }

    /// Get children of a memory (public API)
    pub fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        let child_ids = self.search_by_parent(parent_id)?;
        let mut children = Vec::new();
        for id in child_ids {
            if let Ok(memory) = self.get(&id) {
                children.push(memory);
            }
        }
        Ok(children)
    }

    /// Get the parent chain (ancestors) of a memory
    pub fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        let mut ancestors = Vec::new();
        let mut current_id = memory_id.clone();

        // Walk up the parent chain (max 100 to prevent infinite loops)
        for _ in 0..100 {
            let memory = self.get(&current_id)?;
            if let Some(parent_id) = &memory.parent_id {
                let parent = self.get(parent_id)?;
                ancestors.push(parent.clone());
                current_id = parent_id.clone();
            } else {
                break; // Reached root
            }
        }

        Ok(ancestors)
    }

    /// Get the full hierarchy context for a memory
    /// Returns (ancestors, memory, children)
    pub fn get_hierarchy_context(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)> {
        let memory = self.get(memory_id)?;
        let ancestors = self.get_ancestors(memory_id)?;
        let children = self.get_children(memory_id)?;
        Ok((ancestors, memory, children))
    }

    /// Get all memories in a subtree rooted at the given memory
    pub fn get_subtree(&self, root_id: &MemoryId, max_depth: usize) -> Result<Vec<Memory>> {
        let mut result = Vec::new();
        let mut queue = vec![(root_id.clone(), 0usize)];

        while let Some((id, depth)) = queue.pop() {
            if depth > max_depth {
                continue;
            }
            if let Ok(memory) = self.get(&id) {
                result.push(memory);
                // Add children to queue
                if depth < max_depth {
                    let child_ids = self.search_by_parent(&id)?;
                    for child_id in child_ids {
                        queue.push((child_id, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Every memory key under `prefix`, values not loaded.
    ///
    /// Memory keys only — the 16-byte ids `get_all_ids` reads — although the default
    /// family also holds vector mappings, counters, facts and KV rows: redb keeps
    /// memories in a table of their own, and `PrimaryMemoryStore::scan_prefix_keys`
    /// answers the same keys on both engines. The default keyspace is read through
    /// `default_key_value_store`.
    ///
    /// The prefix bound is enforced HERE. rocksdb's `prefix_iterator` may run past
    /// the prefix, so callers were carrying `if !key.starts_with(p) { break }` by
    /// hand — a guard whose absence deletes keys nobody asked about.
    pub fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        let mut keys = Vec::new();
        for item in self.db.prefix_iterator(prefix) {
            let (key, _) = item?;
            if !key.starts_with(prefix) {
                break;
            }
            if is_memory_key(&key) {
                keys.push(key.to_vec());
            }
        }
        Ok(keys)
    }

    /// Apply a backend-agnostic batch atomically against this store's database.
    pub fn write_batch(&self, batch: &crate::storage::WriteBatch) -> Result<()> {
        // Scoped import: this module already has a `WriteBatch` from rocksdb in
        // flight, and a file-level `use` of the trait would put two same-named
        // things one glance apart in a 4000-line file.
        use crate::storage::BatchWriter as _;
        crate::storage::RocksDbBatchWriter::with_sync_writes(self.db(), self.sync_writes)
            .write_batch(batch)
    }

    /// Get all memory IDs without loading full Memory objects.
    ///
    /// Returns only 16-byte UUID keys (lightweight). Use with `get()` to load
    /// individual memories in batches to avoid OOM on large datasets.
    pub fn get_all_ids(&self) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for (key, _) in iter.flatten() {
            if key.len() == 16 {
                let uuid_bytes: [u8; 16] = key[..16].try_into().unwrap();
                ids.push(MemoryId(uuid::Uuid::from_bytes(uuid_bytes)));
            }
        }
        Ok(ids)
    }

    /// Every live memory in long-term storage: forgotten (soft-deleted) records are
    /// left out, as `PrimaryMemoryStore::get_all` documents for every engine.
    ///
    /// Only returns entries with valid 16-byte UUID keys (consistent with get_stats)
    pub fn get_all(&self) -> Result<Vec<Memory>> {
        Ok(self.decode_all_memories(IncludeForgotten::No))
    }

    /// Every decodable memory record, in key order.
    fn decode_all_memories(&self, forgotten: IncludeForgotten) -> Vec<Memory> {
        let mut memories = Vec::new();

        // fill_cache(false) prevents this maintenance scan from polluting
        // the block cache with cold data, reducing C++ peak memory
        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for (key, value) in iter.flatten() {
            // Only process valid 16-byte UUID keys (consistent with get_stats)
            if key.len() != 16 {
                continue;
            }
            if let Some(memory) = self.decode_or_skip(&key, &value, "memory scan") {
                if forgotten == IncludeForgotten::Yes || !memory.is_forgotten() {
                    memories.push(memory);
                }
            }
        }

        memories
    }

    /// Memories not yet compressed or forgotten, created before `cutoff`.
    ///
    /// A range seek over the uncompressed time index that decodes only the candidates.
    /// Entries the index still holds for a record that has since been compressed,
    /// forgotten or deleted are dropped as they are met.
    pub fn get_uncompressed_older_than(&self, cutoff: DateTime<Utc>) -> Result<Vec<Memory>> {
        self.ensure_uncompressed_index()?;

        let idx = self.index_cf();
        let prefix = UNCOMPRESSED_INDEX_PREFIX.as_bytes();
        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        read_opts.set_iterate_upper_bound(format!(
            "{UNCOMPRESSED_INDEX_PREFIX}{:020}",
            sortable_millis(cutoff)
        ));
        let iter = self.db.iterator_cf_opt(
            idx,
            read_opts,
            IteratorMode::From(prefix, rocksdb::Direction::Forward),
        );

        let mut memories = Vec::new();
        let mut stale = WriteBatch::default();
        for (key, _) in iter.log_errors() {
            if !key.starts_with(prefix) {
                break;
            }
            let Some(id) = std::str::from_utf8(&key)
                .ok()
                .and_then(|k| k.rsplit(':').next())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .map(MemoryId)
            else {
                stale.delete_cf(idx, key.as_ref());
                continue;
            };
            match self.get_uncached(&id) {
                Ok(Some(memory)) => {
                    let current = !memory.compressed
                        && !memory.is_forgotten()
                        && uncompressed_index_key(memory.created_at, &id).as_bytes() == &key[..];
                    if !current {
                        stale.delete_cf(idx, key.as_ref());
                    } else if memory.created_at < cutoff {
                        memories.push(memory);
                    }
                }
                Ok(None) => stale.delete_cf(idx, key.as_ref()),
                Err(e) => {
                    tracing::debug!("compression candidate {} skipped: {e}", id.0);
                }
            }
        }
        if !stale.is_empty() {
            self.guarded_batch(&stale, &self.write_options())?;
        }

        Ok(memories)
    }

    /// Read and decode one record without filling the block cache or rewriting legacy
    /// encodings: for maintenance passes that visit records nobody asked for.
    fn get_uncached(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let key = id.0.as_bytes();
        match self.db.get_opt(key, &read_opts)? {
            Some(value) => Ok(Some(self.decode_stored(key, &value)?.0)),
            None => Ok(None),
        }
    }

    /// Index every record written before the uncompressed time index existed, once.
    /// A write racing the backfill can leave an entry for a record that is compressed by
    /// the time it lands; `get_uncompressed_older_than` drops such entries when it meets
    /// them.
    fn ensure_uncompressed_index(&self) -> Result<()> {
        let idx = self.index_cf();
        if self.db.get_cf(idx, UNCOMPRESSED_INDEX_SENTINEL)?.is_some() {
            return Ok(());
        }

        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let mut batch = WriteBatch::default();
        let mut indexed = 0usize;
        for (key, value) in self
            .db
            .iterator_opt(IteratorMode::Start, read_opts)
            .log_errors()
        {
            if key.len() != 16 {
                continue;
            }
            if let Some(memory) = self.decode_or_skip(&key, &value, "ensure_uncompressed_index") {
                if !memory.compressed && !memory.is_forgotten() {
                    let index_key = uncompressed_index_key(memory.created_at, &memory.id);
                    batch.put_cf(idx, index_key.as_bytes(), b"1");
                    indexed += 1;
                    if batch.len() >= 10_000 {
                        self.guarded_batch(&std::mem::take(&mut batch), &self.write_options())
                            .context("Failed to write the uncompressed time index backfill")?;
                    }
                }
            }
        }
        batch.put_cf(idx, UNCOMPRESSED_INDEX_SENTINEL, b"1");
        self.guarded_batch(&batch, &self.write_options())
            .context("Failed to write the uncompressed time index backfill")?;
        tracing::info!(indexed, "Backfilled the uncompressed-memory time index");
        Ok(())
    }

    /// Mark memories as forgotten (soft delete) with atomic batch write.
    /// Returns the IDs of memories that were flagged, so callers can clean up
    /// secondary indices (vector, BM25, graph).
    pub fn mark_forgotten_by_age(&self, cutoff: DateTime<Utc>) -> Result<Vec<MemoryId>> {
        let mut batch = rocksdb::WriteBatch::default();
        let mut flagged_ids = Vec::new();
        let now = Utc::now().to_rfc3339();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            match self.decode_stored(&key, &value) {
                Ok((mut memory, _)) => {
                    if memory.is_forgotten() {
                        continue;
                    }
                    if memory.created_at < cutoff {
                        flagged_ids.push(memory.id.clone());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten".to_string(), "true".to_string());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten_at".to_string(), now.clone());

                        let updated_value = self.serialize_memory(&memory)?;
                        batch.put(&key, updated_value);
                        batch.delete_cf(
                            self.index_cf(),
                            uncompressed_index_key(memory.created_at, &memory.id).as_bytes(),
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "skipping a record that failed to decrypt/deserialize during forget-by-age (corruption, wrong key, or unsupported epoch)"
                    );
                }
            }
        }

        if !flagged_ids.is_empty() {
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(true);
            self.guarded_batch(&batch, &write_opts)?;
        }

        Ok(flagged_ids)
    }

    /// Mark memories with low importance as forgotten with atomic batch write.
    /// Returns the IDs of memories that were flagged, so callers can clean up
    /// secondary indices (vector, BM25, graph).
    pub fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<Vec<MemoryId>> {
        let mut batch = rocksdb::WriteBatch::default();
        let mut flagged_ids = Vec::new();
        let now = Utc::now().to_rfc3339();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            match self.decode_stored(&key, &value) {
                Ok((mut memory, _)) => {
                    if memory.is_forgotten() {
                        continue;
                    }
                    if memory.importance() < threshold {
                        flagged_ids.push(memory.id.clone());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten".to_string(), "true".to_string());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten_at".to_string(), now.clone());

                        let updated_value = self.serialize_memory(&memory)?;
                        batch.put(&key, updated_value);
                        batch.delete_cf(
                            self.index_cf(),
                            uncompressed_index_key(memory.created_at, &memory.id).as_bytes(),
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "skipping a record that failed to decrypt/deserialize during forget-by-importance (corruption, wrong key, or unsupported epoch)"
                    );
                }
            }
        }

        if !flagged_ids.is_empty() {
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(true);
            self.guarded_batch(&batch, &write_opts)?;
        }

        Ok(flagged_ids)
    }

    /// Remove memories matching a pattern with durable writes
    pub fn remove_matching(&self, regex: &regex::Regex) -> Result<usize> {
        let mut count = 0;
        let mut to_delete: Vec<MemoryId> = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            // Only process valid 16-byte UUID keys
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = self.decode_stored(&key, &value) {
                if regex.is_match(&memory.experience.content) {
                    to_delete.push(memory.id);
                    count += 1;
                }
            }
        }

        // Delete each memory through the proper delete() path which cleans up indices first
        for memory_id in to_delete {
            if let Err(e) = self.delete(&memory_id) {
                tracing::warn!("Failed to delete matching memory {}: {}", memory_id.0, e);
            }
        }

        Ok(count)
    }

    /// Update access count for a memory
    pub fn update_access(&self, id: &MemoryId) -> Result<()> {
        if let Ok(memory) = self.get(id) {
            // ZERO-COPY: Update metadata through interior mutability
            memory.update_access();

            // Persist updated metadata
            self.update(&memory)?;
        }
        Ok(())
    }

    /// Get statistics about stored memories
    pub fn get_stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::default();
        let mut raw_count = 0;
        let mut skipped_non_memory = 0;
        let mut deserialize_errors = 0;
        let stats_prefix = b"stats:";

        let mut read_opts = ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for item in iter {
            match item {
                Ok((key, value)) => {
                    raw_count += 1;

                    // Skip stats entries - they use a different format
                    if key.starts_with(stats_prefix) {
                        skipped_non_memory += 1;
                        continue;
                    }

                    // Valid memory keys should be exactly 16 bytes (UUID bytes)
                    if key.len() != 16 {
                        skipped_non_memory += 1;
                        continue;
                    }

                    match self.decode_stored(&key, &value) {
                        Ok((memory, _)) => {
                            if memory.is_forgotten() {
                                continue;
                            }
                            stats.total_count += 1;
                            stats.total_size_bytes += value.len();
                            if memory.compressed {
                                stats.compressed_count += 1;
                            }
                            stats.importance_sum += memory.importance();
                        }
                        Err(e) => {
                            deserialize_errors += 1;
                            // The id is what makes an undecodable record actionable; a length
                            // alone identified nothing when 24 surfaced on the 2026-09-13 deploy.
                            // No byte prefix: content can begin inside the first 24 bytes of a
                            // record and 25 tenants share this log. A digest identifies the bytes
                            // without printing them.
                            let id = uuid::Uuid::from_slice(&key)
                                .map(|u| u.to_string())
                                .unwrap_or_else(|_| hex::encode(&key));
                            let digest = {
                                use sha2::{Digest, Sha256};
                                hex::encode(&Sha256::digest(&value)[..8])
                            };
                            tracing::warn!(
                                "Corrupted memory entry id={} (key len: {}, value len: {}, sha256[..8]={}): {}",
                                id,
                                key.len(),
                                value.len(),
                                digest,
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Iterator error: {}", e);
                }
            }
        }

        tracing::debug!(
            "get_stats: raw_count={}, memories={}, skipped={}, corrupted={}",
            raw_count,
            stats.total_count,
            skipped_non_memory,
            deserialize_errors
        );

        if stats.total_count > 0 {
            stats.average_importance = stats.importance_sum / stats.total_count as f32;
        }

        // Load persisted retrieval counter
        stats.total_retrievals = self.get_retrieval_count().unwrap_or(0);

        Ok(stats)
    }

    /// Get the persisted retrieval counter
    pub fn get_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        match self.db.get(RETRIEVAL_KEY)? {
            Some(data) => {
                if data.len() >= 8 {
                    Ok(usize::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])))
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    /// Increment and persist the retrieval counter, returns new value
    pub fn increment_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        let current = self.get_retrieval_count().unwrap_or(0);
        let new_count = current + 1;
        let opts = self.write_options();
        self.guarded_write(|| {
            Ok(self
                .db
                .put_opt(RETRIEVAL_KEY, new_count.to_le_bytes(), &opts)?)
        })?;
        Ok(new_count)
    }

    /// Remove memory records that fail to deserialize even with legacy fallbacks, with
    /// their index rows, in one atomic write. Returns the number of records deleted.
    ///
    /// Only memory keys ([`is_memory_key`]) are inspected. The default family is shared with
    /// vector mappings, counters, facts, lineage, ACLs, merge audit and tombstone rows
    /// and KV rows written through `default_key_value_store`; none of them is a memory,
    /// so none of them can be a corrupted one, and a skip list of their prefixes
    /// deleted every subsystem it had not heard of. redb's cleanup reads its memories
    /// table alone, and the two answer the same.
    ///
    /// A sealed record that will not open is never deleted: wrong key, missing key
    /// and tamper are indistinguishable under AES-GCM, and deleting on a key
    /// misconfiguration would destroy data a restored keystore opens.
    pub fn cleanup_corrupted(&self) -> Result<usize> {
        let mut doomed: Vec<uuid::Uuid> = Vec::new();
        for (key, value) in self.db.iterator(IteratorMode::Start).flatten() {
            if !is_memory_key(&key) {
                continue;
            }
            let Ok(id) = uuid::Uuid::from_slice(&key) else {
                continue;
            };
            if crate::keystore::is_encrypted_record(&value) {
                if self.decode_stored(&key, &value).is_err() {
                    tracing::warn!(
                        memory_id = %id,
                        bytes = value.len(),
                        "sealed record does not open (wrong/missing key or tamper); not deleting it"
                    );
                }
            } else if deserialize_memory(&value).is_err() {
                tracing::debug!(memory_id = %id, bytes = value.len(), "record does not decode; deleting it");
                doomed.push(id);
            }
        }

        if doomed.is_empty() {
            return Ok(0);
        }
        tracing::info!(
            records = doomed.len(),
            "cleaning up corrupted memory records"
        );
        let mut batch = WriteBatch::default();
        for id in &doomed {
            batch.delete(id.as_bytes());
        }
        self.queue_index_rows_naming(&doomed, &mut batch);
        self.guarded_batch(&batch, &self.write_options())?;
        self.flush()?;
        Ok(doomed.len())
    }

    /// Queue the removal of every index row that names one of `ids`, read from the index
    /// itself: a record that will not decode cannot say which rows were its own. Every
    /// row key ends `:{id}`.
    fn queue_index_rows_naming(&self, ids: &[uuid::Uuid], batch: &mut WriteBatch) {
        let named: std::collections::HashSet<uuid::Uuid> = ids.iter().copied().collect();
        let idx = self.index_cf();
        for (key, _value) in self.db.iterator_cf(idx, IteratorMode::Start).log_errors() {
            let names_one = key
                .rsplit(|byte| *byte == b':')
                .next()
                .and_then(|suffix| std::str::from_utf8(suffix).ok())
                .and_then(|suffix| uuid::Uuid::parse_str(suffix).ok())
                .is_some_and(|id| named.contains(&id));
            if names_one {
                batch.delete_cf(idx, key.as_ref());
            }
        }
    }

    /// Migrate legacy memories to current format for improved performance
    /// Returns (migrated_count, already_current_count, failed_count)
    ///
    /// This function:
    /// 1. Iterates all memories in storage
    /// 2. Attempts to deserialize with format fallback
    /// 3. Re-saves successfully deserialized legacy memories in current format
    /// 4. Reports migration statistics
    pub fn migrate_legacy(&self) -> Result<(usize, usize, usize)> {
        let mut migrated = 0;
        let mut already_current = 0;
        let mut failed = 0;
        let stats_prefix = b"stats:";

        let iter = self.db.iterator(IteratorMode::Start);
        let mut to_migrate = Vec::new();

        for (key, value) in iter.flatten() {
            // Skip stats entries
            if key.starts_with(stats_prefix) {
                continue;
            }

            // Skip non-UUID keys
            if key.len() != 16 {
                continue;
            }

            // Try current format first (quick check). An encrypted record is already
            // in the v2 envelope format, so treat it as current rather than
            // decrypt + re-encrypt it on every migration pass.
            let is_current = crate::keystore::is_encrypted_record(&value)
                || bincode::serde::decode_from_slice::<Memory, _>(
                    &value,
                    bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
                )
                .is_ok();

            if is_current {
                already_current += 1;
                continue;
            }

            // Not current format - try with fallback
            match self.decode_stored(&key, &value) {
                Ok((memory, _)) => {
                    // Successfully deserialized legacy format - queue for migration
                    to_migrate.push((key.to_vec(), memory));
                }
                Err(_) => {
                    failed += 1;
                }
            }
        }

        // Re-save migrated memories in current format
        if !to_migrate.is_empty() {
            tracing::info!(
                "Migrating {} legacy memories to current format",
                to_migrate.len()
            );

            let write_opts = self.write_options();

            for (key, memory) in to_migrate {
                match self.serialize_memory(&memory) {
                    Ok(serialized) => {
                        if let Err(e) = self.guarded_write(|| {
                            Ok(self.db.put_opt(&key, &serialized, &write_opts)?)
                        }) {
                            tracing::warn!("Failed to migrate memory: {e}");
                            failed += 1;
                        } else {
                            migrated += 1;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to serialize migrated memory: {e}");
                        failed += 1;
                    }
                }
            }

            // Flush to persist migrations
            self.flush()?;
        }

        tracing::info!(
            "Migration complete: {} migrated, {} already current, {} failed",
            migrated,
            already_current,
            failed
        );

        Ok((migrated, already_current, failed))
    }

    /// Flush all column families to ensure data is persisted (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        use rocksdb::FlushOptions;

        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true); // Block until flush is complete

        // Single DB flush covers both default and index CFs
        self.guarded_write(|| {
            self.db
                .flush_opt(&flush_opts)
                .map_err(|e| anyhow::anyhow!("Failed to flush memory storage: {e}"))
        })?;

        // Explicitly flush the index CF (RocksDB flush_opt only flushes default CF)
        self.guarded_write(|| {
            self.db
                .flush_cf_opt(self.index_cf(), &flush_opts)
                .map_err(|e| anyhow::anyhow!("Failed to flush index CF: {e}"))
        })?;

        Ok(())
    }

    /// Get a reference to the underlying RocksDB instance
    ///
    /// Used by SemanticFactStore to share the same database for fact storage.
    /// Facts use a different key prefix ("facts:") to avoid collisions.
    pub fn db(&self) -> Arc<DB> {
        self.db.clone()
    }

    /// Backend-agnostic key-value handle over the general-purpose `kv` column family.
    ///
    /// For small per-user blobs (e.g. the `CognitiveDials` tuning surface) that don't
    /// belong in the memory or index schemas. Returns a
    /// [`crate::storage::KeyValueStore`] so callers stay portable across the redb
    /// cutover — the concrete rocksdb CF lives behind the trait here, and the redb
    /// path (`RedbKeyValueStore`) implements the same trait. The store is per-user
    /// (this `MemoryStorage` opens the user's own database), so callers use bare keys.
    /// Backend-agnostic handle over the DEFAULT keyspace.
    ///
    /// Not the same store as [`Self::key_value_store`], which binds the `kv`
    /// column family. Data written before the column families exists here, and
    /// moving it is a migration, not a refactor — see the type's own note.
    pub fn default_key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(
            crate::storage::legacy_rocksdb::RocksDbDefaultKeyValueStore::with_sync_writes(
                self.db.clone(),
                self.sync_writes,
            ),
        )
    }

    pub fn key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(
            crate::storage::legacy_rocksdb::RocksDbColumnFamilyStore::with_sync_writes(
                self.db.clone(),
                CF_KV,
                self.sync_writes,
            ),
        )
    }
}

/// Search criteria for memory retrieval
#[derive(Debug, Clone)]
pub enum SearchCriteria {
    // === Standard Criteria ===
    ByDate {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    ByType(ExperienceType),
    ByImportance {
        min: f32,
        max: f32,
    },
    ByEntity(String),
    /// Filter by tags (matches memories containing ANY of these tags)
    ByTags(Vec<String>),

    // === Temporal/Episode Criteria ===
    /// Filter by episode ID - memories in the same episode are highly related
    ByEpisode(String),
    /// Filter by episode with sequence ordering - returns memories in temporal order
    ByEpisodeSequence {
        episode_id: String,
        /// If provided, only return memories with sequence >= this value
        min_sequence: Option<u32>,
        /// If provided, only return memories with sequence <= this value
        max_sequence: Option<u32>,
    },

    // === Robotics Criteria ===
    /// Filter by robot/drone identifier
    ByRobot(String),
    /// Filter by mission identifier
    ByMission(String),
    /// Spatial filter: memories within radius of (lat, lon)
    ByLocation {
        lat: f64,
        lon: f64,
        radius_meters: f64,
    },
    /// Filter by action type
    ByActionType(String),
    /// Filter by reward range (for RL-style queries)
    ByReward {
        min: f32,
        max: f32,
    },

    // === Compound Criteria ===
    /// Memories every listed criterion matches. An empty list matches nothing on every
    /// engine: `POST /api/search/advanced` refuses a search with no criterion rather
    /// than answering it with the whole corpus, and an empty `Combined` is that same
    /// search reached another way.
    Combined(Vec<SearchCriteria>),

    // === Hierarchy Criteria ===
    /// Filter by parent memory ID - returns all children of a memory
    ByParent(MemoryId),
    /// Filter for root memories (no parent)
    RootsOnly,
}

/// Mean Earth radius the spatial criteria measure with, in metres.
pub(crate) const EARTH_RADIUS_METRES: f64 = 6_371_000.0;

/// Great-circle distance in metres (haversine).
pub(crate) fn haversine_metres(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (phi1, phi2) = (lat1.to_radians(), lat2.to_radians());
    let delta_phi = (lat2 - lat1).to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();
    let a = (delta_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (delta_lambda / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_METRES * a.sqrt().asin()
}

impl SearchCriteria {
    /// Whether `memory` satisfies these criteria, exactly.
    ///
    /// The one definition every engine's `search` answers with: an index only narrows
    /// the candidates — RocksDB's day, tenth, twentieth and geohash-cell buckets, redb's
    /// value-ordered rows — and every memory `search` returns has passed this, so a
    /// coarse bucket or a stale row can cost a read but never change a result.
    pub fn matches(&self, memory: &Memory) -> bool {
        let episode = memory
            .experience
            .context
            .as_ref()
            .map(|context| &context.episode);
        match self {
            SearchCriteria::ByDate { start, end } => {
                memory.created_at >= *start && memory.created_at <= *end
            }
            SearchCriteria::ByType(experience_type) => {
                &memory.experience.experience_type == experience_type
            }
            SearchCriteria::ByImportance { min, max } => {
                let importance = memory.importance();
                importance >= *min && importance <= *max
            }
            SearchCriteria::ByEntity(entity) => {
                let needle = entity.to_lowercase();
                memory
                    .experience
                    .entities
                    .iter()
                    .any(|candidate| candidate.to_lowercase() == needle)
            }
            SearchCriteria::ByTags(tags) => {
                let owned: std::collections::HashSet<String> = memory
                    .experience
                    .tags
                    .iter()
                    .map(|tag| tag.to_lowercase())
                    .collect();
                tags.iter().any(|tag| owned.contains(&tag.to_lowercase()))
            }
            SearchCriteria::ByEpisode(episode_id) => {
                episode.and_then(|e| e.episode_id.as_deref()) == Some(episode_id.as_str())
            }
            SearchCriteria::ByEpisodeSequence {
                episode_id,
                min_sequence,
                max_sequence,
            } => {
                if episode.and_then(|e| e.episode_id.as_deref()) != Some(episode_id.as_str()) {
                    return false;
                }
                match episode.and_then(|e| e.sequence_number) {
                    Some(sequence) => {
                        min_sequence.is_none_or(|lo| sequence >= lo)
                            && max_sequence.is_none_or(|hi| sequence <= hi)
                    }
                    None => false,
                }
            }
            SearchCriteria::ByRobot(robot_id) => {
                memory.experience.resolved_robot_id() == Some(robot_id.as_str())
            }
            SearchCriteria::ByMission(mission_id) => {
                memory.experience.resolved_mission_id() == Some(mission_id.as_str())
            }
            SearchCriteria::ByLocation {
                lat,
                lon,
                radius_meters,
            } => match memory.experience.resolved_geo() {
                Some((memory_lat, memory_lon, _alt)) => {
                    haversine_metres(*lat, *lon, memory_lat, memory_lon) <= *radius_meters
                }
                None => false,
            },
            SearchCriteria::ByActionType(action_type) => {
                memory.experience.action_type.as_deref() == Some(action_type.as_str())
            }
            SearchCriteria::ByReward { min, max } => match memory.experience.reward {
                Some(reward) => reward >= *min && reward <= *max,
                None => false,
            },
            SearchCriteria::Combined(criterias) => {
                !criterias.is_empty() && criterias.iter().all(|criteria| criteria.matches(memory))
            }
            SearchCriteria::ByParent(parent_id) => memory.parent_id.as_ref() == Some(parent_id),
            SearchCriteria::RootsOnly => memory.parent_id.is_none(),
        }
    }
}

/// Storage statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_count: usize,
    pub compressed_count: usize,
    pub total_size_bytes: usize,
    pub average_importance: f32,
    pub importance_sum: f32,
    /// Total number of recall/retrieval operations (persisted)
    #[serde(default)]
    pub total_retrievals: usize,
}

// =============================================================================
// ATOMIC VECTOR INDEX MAPPING STORAGE
// =============================================================================
//
// This module provides atomic storage for vector index mappings alongside memory data.
// By storing IdMapping in RocksDB (not separate files), we ensure:
//
// 1. ATOMIC WRITES: Memory + vector mapping written in single WriteBatch
// 2. NO ORPHANS: If memory exists, its vector mapping exists (or can be rebuilt)
// 3. CRASH SAFETY: RocksDB WAL protects both memory data and mappings
// 4. SINGLE SOURCE OF TRUTH: RocksDB is THE authority, Vamana is just a cache
//
// MULTIMODALITY READY:
// - Each modality (text, image, audio, video) has separate vector space
// - Text: 384-dim MiniLM (current)
// - Image: 1024-dim ImageBind (future)
// - Audio: 1024-dim ImageBind (future)
// - Video: 1024-dim ImageBind (future)
// - Cross-modal search possible via ImageBind's unified embedding space
//
// Key format: "vmapping:{memory_id}" -> bincode(VectorMappingEntry)
// =============================================================================

/// Supported embedding modalities
///
/// When adding a new modality:
/// 1. Add variant here
/// 2. Create corresponding Vamana index with correct dimension
/// 3. Implement embedder for the modality
/// 4. Update search to include the modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (MiniLM-L6-v2, 384-dim)
    Text,
    /// Image embeddings (future: ImageBind, 1024-dim)
    Image,
    /// Audio embeddings (future: ImageBind, 1024-dim)
    Audio,
    /// Video embeddings (future: ImageBind, 1024-dim)
    Video,
    /// Multi-modal unified embeddings (future: ImageBind, 1024-dim)
    /// Used when content has multiple modalities fused together
    Unified,
}

impl Modality {
    /// Get embedding dimension for this modality
    pub fn dimension(&self) -> usize {
        match self {
            Modality::Text => 384, // MiniLM-L6-v2
            // ImageBind projects all modalities to 1024-dim shared space
            Modality::Image => 1024,
            Modality::Audio => 1024,
            Modality::Video => 1024,
            Modality::Unified => 1024,
        }
    }

    /// Get the string key for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Unified => "unified",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Vector IDs for a specific modality
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModalityVectors {
    /// Vector IDs in this modality's Vamana index
    pub vector_ids: Vec<u32>,
    /// Embedding dimension (for validation)
    pub dimension: usize,
    /// Chunk boundaries (for long content)
    /// Each entry is (start_char, end_char) in original content
    pub chunk_ranges: Option<Vec<(usize, usize)>>,
}

/// Vector mapping entry for a single memory - MULTIMODALITY READY
///
/// Stores vector IDs for each modality separately, allowing:
/// - Text-only memories (current)
/// - Image-only memories (future)
/// - Multi-modal memories (text + image + audio)
/// - Cross-modal search via unified embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMappingEntry {
    /// Vector IDs per modality
    /// Key: Modality enum (serializes as string)
    /// Value: Vector IDs + metadata for that modality
    pub modalities: HashMap<Modality, ModalityVectors>,
    /// Timestamp when mapping was created (for debugging)
    pub created_at: i64,
    /// Schema version for forward compatibility
    pub version: u8,
}

impl Default for VectorMappingEntry {
    fn default() -> Self {
        Self {
            modalities: HashMap::new(),
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }
}

impl VectorMappingEntry {
    /// Create a new mapping with text vectors (most common case)
    pub fn with_text(vector_ids: Vec<u32>) -> Self {
        let mut modalities = HashMap::new();
        modalities.insert(
            Modality::Text,
            ModalityVectors {
                vector_ids,
                dimension: 384,
                chunk_ranges: None,
            },
        );
        Self {
            modalities,
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }

    /// Get text vector IDs (convenience method for current text-only usage)
    pub fn text_vectors(&self) -> Option<&Vec<u32>> {
        self.modalities.get(&Modality::Text).map(|m| &m.vector_ids)
    }

    /// Get all vector IDs across all modalities (for deletion)
    pub fn all_vector_ids(&self) -> Vec<(Modality, u32)> {
        self.modalities
            .iter()
            .flat_map(|(modality, mv)| mv.vector_ids.iter().map(|id| (*modality, *id)))
            .collect()
    }

    /// Check if this entry has any vectors
    pub fn is_empty(&self) -> bool {
        self.modalities.values().all(|mv| mv.vector_ids.is_empty())
    }

    /// Add vectors for a modality
    pub fn add_modality(&mut self, modality: Modality, vector_ids: Vec<u32>) {
        self.modalities.insert(
            modality,
            ModalityVectors {
                dimension: modality.dimension(),
                vector_ids,
                chunk_ranges: None,
            },
        );
    }

    /// Future: Add image vectors
    #[allow(dead_code)]
    pub fn with_image(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Image, vector_ids);
        self
    }

    /// Future: Add audio vectors
    #[allow(dead_code)]
    pub fn with_audio(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Audio, vector_ids);
        self
    }

    /// Future: Add video vectors
    #[allow(dead_code)]
    pub fn with_video(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Video, vector_ids);
        self
    }
}

#[cfg(feature = "rocksdb")]
impl MemoryStorage {
    // =========================================================================
    // ATOMIC VECTOR MAPPING OPERATIONS
    // =========================================================================

    /// Store memory and its text vector mapping atomically
    ///
    /// Uses WriteBatch to ensure both operations succeed or both fail.
    /// This is the ONLY way orphaned memories can be prevented.
    ///
    /// For text-only memories (current implementation). Use store_with_multimodal_vectors
    /// for memories with image/audio/video content.
    pub fn store_with_vectors(&self, memory: &Memory, vector_ids: Vec<u32>) -> Result<()> {
        self.store_with_multimodal_vectors(memory, Modality::Text, vector_ids)
    }

    /// Store memory with vectors for a specific modality
    ///
    /// MULTIMODALITY READY: Supports text, image, audio, video modalities.
    /// Each modality is stored separately, allowing cross-modal search.
    ///
    /// The record, its mapping, and the replacement of its index rows are one atomic
    /// write, as [`Self::store`] makes them.
    pub fn store_with_multimodal_vectors(
        &self,
        memory: &Memory,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mut batch = WriteBatch::default();

        // 1. Serialize memory (encrypts content field if encryptor is configured)
        let memory_key = memory.id.0.as_bytes();
        let memory_value = self
            .serialize_memory(memory)
            .context(format!("Failed to serialize memory {}", memory.id.0))?;
        self.queue_replaced_index_removals(&memory.id, &mut batch);
        batch.put(memory_key, &memory_value);
        self.queue_index_writes(memory, &mut batch);

        // 2. Serialize vector mapping with modality support
        let mapping_key = format!("vmapping:{}", memory.id.0);

        // Load existing mapping (for adding new modality to existing memory)
        let mut mapping_entry = self.get_vector_mapping(&memory.id)?.unwrap_or_default();

        // Add/update the modality vectors
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value =
            bincode::serde::encode_to_vec(&mapping_entry, bincode::config::standard())
                .context("Failed to serialize vector mapping")?;
        batch.put(mapping_key.as_bytes(), &mapping_value);

        self.guarded_batch(&batch, &self.write_options())
            .context("Atomic write of memory, vector mapping and index rows failed")?;
        Ok(())
    }

    /// Get vector mapping for a memory
    pub fn get_vector_mapping(&self, memory_id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        match self.db.get(mapping_key.as_bytes())? {
            Some(data) => {
                let (entry, _): (VectorMappingEntry, _) = bincode::serde::decode_from_slice(
                    &data,
                    bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
                )
                .context("Failed to deserialize vector mapping")?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Get all vector mappings (for rebuilding Vamana index on startup)
    ///
    /// Returns iterator-style results to avoid loading everything into memory at once.
    /// Sorted by memory_id for deterministic Vamana rebuilding.
    pub fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>> {
        let mut mappings = Vec::new();
        let prefix = b"vmapping:";

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter {
            match item {
                Ok((key, value)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if !key_str.starts_with("vmapping:") {
                        break;
                    }

                    // Extract memory_id from key
                    if let Some(id_str) = key_str.strip_prefix("vmapping:") {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            if let Ok((entry, _)) =
                                bincode::serde::decode_from_slice::<VectorMappingEntry, _>(
                                    &value,
                                    bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
                                )
                            {
                                mappings.push((MemoryId(uuid), entry));
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Error reading vector mapping: {}", e);
                }
            }
        }

        Ok(mappings)
    }

    /// Replace a memory's whole vector mapping, every modality at once.
    pub fn put_vector_mapping(
        &self,
        memory_id: &MemoryId,
        mapping: &VectorMappingEntry,
    ) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        let mapping_value = bincode::serde::encode_to_vec(mapping, bincode::config::standard())
            .context("Failed to serialize vector mapping")?;
        let opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .put_opt(mapping_key.as_bytes(), &mapping_value, &opts)
                .context("Failed to write vector mapping")
        })
    }

    /// Delete vector mapping for a memory (called when deleting memory)
    pub fn delete_vector_mapping(&self, memory_id: &MemoryId) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        let write_opts = self.write_options();
        self.guarded_write(|| Ok(self.db.delete_opt(mapping_key.as_bytes(), &write_opts)?))
    }

    /// Update text vector mapping for a memory (for reindex operations)
    ///
    /// Convenience method for text-only reindexing.
    pub fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()> {
        self.update_modality_vectors(memory_id, Modality::Text, vector_ids)
    }

    /// Update vector mapping for a specific modality
    ///
    /// MULTIMODALITY READY: Preserves vectors for other modalities while updating one.
    pub fn update_modality_vectors(
        &self,
        memory_id: &MemoryId,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);

        // Load existing mapping to preserve other modalities
        let mut mapping_entry = self.get_vector_mapping(memory_id)?.unwrap_or_default();

        // Update the specific modality
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value =
            bincode::serde::encode_to_vec(&mapping_entry, bincode::config::standard())?;

        let write_opts = self.write_options();
        self.guarded_write(|| {
            Ok(self
                .db
                .put_opt(mapping_key.as_bytes(), &mapping_value, &write_opts)?)
        })
    }

    /// Delete a memory, its vector mapping and its index rows in one atomic write.
    pub fn delete_with_vectors(&self, id: &MemoryId) -> Result<()> {
        let mut batch = WriteBatch::default();

        // The record is read before anything is deleted: its index rows are named by its
        // fields, so once it is gone nothing can say which rows were its own.
        match self.get_if_present(id) {
            Ok(Some(memory)) => self.queue_index_removals(id, &memory, &mut batch),
            Ok(None) => {}
            Err(e) => self.remove_undecodable_inbound_refs(id, &e),
        }

        batch.delete(id.0.as_bytes());
        let mapping_key = format!("vmapping:{}", id.0);
        batch.delete(mapping_key.as_bytes());

        self.guarded_batch(&batch, &self.write_options())
    }

    /// Count memories with vector mappings (for health checks)
    pub fn count_vector_mappings(&self) -> usize {
        let prefix = b"vmapping:";
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for (key, _) in iter.flatten() {
            if key.starts_with(prefix) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Check integrity: find memories without vector mappings
    ///
    /// Returns memories that have embeddings but no corresponding vector mapping.
    /// These need to be reindexed.
    pub fn find_memories_without_mappings(&self) -> Result<Vec<MemoryId>> {
        let mut orphans = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            // Skip non-memory keys
            if key.len() != 16 {
                continue;
            }

            // Try to deserialize as memory
            if let Ok((memory, _)) = self.decode_stored(&key, &value) {
                // Check if vector mapping exists and has text vectors
                let has_mapping = match self.get_vector_mapping(&memory.id) {
                    Ok(Some(entry)) => entry.text_vectors().is_some_and(|v| !v.is_empty()),
                    _ => false,
                };

                // Memory has embeddings but no mapping - needs reindex
                if !has_mapping && memory.experience.embeddings.is_some() {
                    orphans.push(memory.id);
                }
            }
        }

        Ok(orphans)
    }

    /// Get all text vector IDs from mappings (for Vamana statistics)
    pub fn get_all_text_vector_ids(&self) -> Result<Vec<u32>> {
        let mut all_ids = Vec::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            if let Some(text_vecs) = entry.text_vectors() {
                all_ids.extend(text_vecs.iter().copied());
            }
        }

        Ok(all_ids)
    }

    /// Get vector count per modality (for health monitoring)
    pub fn get_modality_stats(&self) -> Result<HashMap<Modality, usize>> {
        let mut stats: HashMap<Modality, usize> = HashMap::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            for (modality, mv) in entry.modalities {
                *stats.entry(modality).or_insert(0) += mv.vector_ids.len();
            }
        }

        Ok(stats)
    }

    // =========================================================================
    // INTERFERENCE PERSISTENCE (SHO-106 RIF)
    // =========================================================================
    //
    // Persists InterferenceDetector state to the main RocksDB database using
    // key prefix "interference:{memory_id}" for per-memory records and
    // "interference_meta:total" for the aggregate event counter.
    //
    // This ensures retrieval-induced forgetting history survives server restarts.
    // =========================================================================

    /// Persist interference records for a single memory
    ///
    /// Key format: `interference:{memory_id}` → JSON `Vec<InterferenceRecord>`
    pub fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[super::replay::InterferenceRecord],
    ) -> Result<()> {
        let key = format!("interference:{memory_id}");
        let value =
            serde_json::to_vec(records).context("Failed to serialize interference records")?;

        let write_opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .put_opt(key.as_bytes(), &value, &write_opts)
                .context("Failed to persist interference records")
        })
    }

    /// Load all interference records from storage on startup
    ///
    /// Scans all `interference:` prefixed keys and deserializes records.
    /// Returns `(history_map, total_event_count)` for bulk-loading into InterferenceDetector.
    pub fn load_all_interference_records(
        &self,
    ) -> Result<(
        HashMap<String, Vec<super::replay::InterferenceRecord>>,
        usize,
    )> {
        let prefix = b"interference:";
        let mut history: HashMap<String, Vec<super::replay::InterferenceRecord>> = HashMap::new();
        let mut total_events: usize = 0;

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter.log_errors() {
            let (key, value) = item;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with("interference:") {
                break;
            }

            if let Some(memory_id) = key_str.strip_prefix("interference:") {
                match serde_json::from_slice::<Vec<super::replay::InterferenceRecord>>(&value) {
                    Ok(records) => {
                        total_events += records.len();
                        history.insert(memory_id.to_string(), records);
                    }
                    Err(e) => {
                        tracing::warn!(
                            key = %key_str,
                            error = %e,
                            "Failed to deserialize interference records, skipping"
                        );
                    }
                }
            }
        }

        // Load persisted total count (may be higher than sum of records due to eviction)
        let persisted_total = self
            .db
            .get(b"interference_meta:total")
            .ok()
            .flatten()
            .and_then(|v| {
                if v.len() == 8 {
                    Some(u64::from_le_bytes(v[..8].try_into().unwrap()) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(total_events);

        Ok((history, persisted_total.max(total_events)))
    }

    /// Delete interference records for a single memory (called on forget/delete)
    pub fn delete_interference_records(&self, memory_id: &str) -> Result<()> {
        let key = format!("interference:{memory_id}");
        let write_opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .delete_opt(key.as_bytes(), &write_opts)
                .context("Failed to delete interference records")
        })
    }

    /// Persist the total interference event count
    ///
    /// Key: `interference_meta:total` → 8-byte little-endian u64
    pub fn save_interference_event_count(&self, count: usize) -> Result<()> {
        let write_opts = self.write_options();
        self.guarded_write(|| {
            self.db
                .put_opt(
                    b"interference_meta:total",
                    (count as u64).to_le_bytes(),
                    &write_opts,
                )
                .context("Failed to persist interference event count")
        })
    }

    /// Load the fact extraction watermark for a user.
    ///
    /// Key: `_watermark:fact_extraction:{user_id}` → 8-byte little-endian i64 (unix millis)
    /// Returns None if no watermark has been persisted yet.
    pub fn get_fact_watermark(&self, user_id: &str) -> Option<i64> {
        let key = format!("_watermark:fact_extraction:{user_id}");
        match self.db.get(key.as_bytes()) {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                Some(i64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => None,
        }
    }

    /// Persist the fact extraction watermark for a user.
    ///
    /// Key: `_watermark:fact_extraction:{user_id}` → 8-byte little-endian i64 (unix millis)
    pub fn set_fact_watermark(&self, user_id: &str, timestamp_millis: i64) {
        let key = format!("_watermark:fact_extraction:{user_id}");
        let write_opts = self.write_options();
        if let Err(e) = self.guarded_write(|| {
            Ok(self
                .db
                .put_opt(key.as_bytes(), timestamp_millis.to_le_bytes(), &write_opts)?)
        }) {
            tracing::warn!("Failed to persist fact extraction watermark: {e}");
        }
    }

    /// Delete ALL interference records (GDPR forget_all)
    ///
    /// Batch-deletes all `interference:` and `interference_meta:` keys.
    pub fn clear_all_interference_records(&self) -> Result<usize> {
        let prefix = b"interference";
        let mut batch = WriteBatch::default();
        let mut count = 0;

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter.log_errors() {
            let (key, _) = item;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with("interference") {
                break;
            }
            batch.delete(&key);
            count += 1;
        }

        if count > 0 {
            let write_opts = self.write_options();
            self.guarded_batch(&batch, &write_opts)?;
        }

        Ok(count)
    }
}

/// Wire fixtures shared by the decoder tests and the projection tests: a current record with a
/// populated session, and the same bytes as a pre-4a54053 binary wrote them.
#[cfg(test)]
pub(crate) mod wire_fixtures {
    use super::*;

    /// A current record with a populated context and session. `declared`/`verified` stay at
    /// their defaults so that, once stripped, the bytes are what a pre-4a54053 binary wrote.
    pub(crate) fn memory_with_session(content: &str) -> Memory {
        let mut context = crate::memory::context::ContextBuilder::new().build();
        context.session = AgentSession {
            worktree_path: Some(PathBuf::from("/wt/feature")),
            branch: Some("feature/x".into()),
            agent_id: None,
            vscode_window_id: None,
            started_at: Some(Utc::now()),
            parent_repo: Some(PathBuf::from("/repo")),
            declared: Default::default(),
            verified: Default::default(),
        };
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            context: Some(context),
            embeddings: Some(vec![0.25; 4]),
            tags: vec!["synthetic".into()],
            ..Default::default()
        };
        let mut memory = Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            experience,
            0.7,
            None,
            None,
            None,
            None,
        );
        memory.when.episode_ordinal = Some(3);
        memory
    }

    pub(crate) fn encode(memory: &Memory) -> Vec<u8> {
        bincode::serde::encode_to_vec(memory, bincode::config::standard()).unwrap()
    }

    /// Strips the eight default-writer bytes 4a54053 appended, producing the pre-4a54053 wire.
    pub(crate) fn strip_writer_structs(current: &[u8]) -> Vec<u8> {
        let cfg = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
        let (_, gap_at) =
            bincode::serde::decode_from_slice::<LegacyMemoryThroughPreWriterSession, _>(
                current, cfg,
            )
            .unwrap();
        assert_eq!(
            &current[gap_at..gap_at + 8],
            &PRE_WRITER_SESSION_GAP,
            "fixture must carry default writers"
        );
        [&current[..gap_at], &current[gap_at + 8..]].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct LegacyMinimalFixture {
        id: MemoryId,
        content: String,
    }

    // LEGACY_FALLBACK_BRANCH_TOTAL is a process-wide metric. Two tests that decode legacy
    // records in parallel see each other's increments, so a `before + 1` assertion reads 2 —
    // the other half of why test_deserialize_with_fallback_bincode1_minimal_fixture "failed in
    // company". Every test that touches the decoder's counters takes this lock.
    static FALLBACK_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn counter_guard() -> std::sync::MutexGuard<'static, ()> {
        FALLBACK_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sample_memory(id: MemoryId, content: &str) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            ..Default::default()
        };
        Memory::from_legacy(
            id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }

    #[test]
    fn test_deserialize_with_fallback_records_current_bincode2_branch() {
        let _serial = counter_guard();
        let id = MemoryId(uuid::Uuid::new_v4());
        let memory = sample_memory(id.clone(), "current format memory");
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["bincode2_memory"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(!is_legacy);
        // Current format is not a fallback — metric should NOT increment
        assert_eq!(after, before);
    }

    #[test]
    fn test_deserialize_with_fallback_bincode1_minimal_fixture() {
        let _serial = counter_guard();
        let id = MemoryId(uuid::Uuid::new_v4());
        let fixture = LegacyMinimalFixture {
            id: id.clone(),
            content: "legacy bincode1 minimal".to_string(),
        };
        let bytes = bincode1::serialize(&fixture).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["bincode1_minimal"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(is_legacy);
        assert_eq!(after, before + 1);
    }

    /// The misread, made deterministic: uuid byte 9 = 0x00 is the value that let the bincode2
    /// MinimalMemory candidate read the bincode1 fixint prefix as a 16-byte id and an empty
    /// content, and decode_from_slice's tolerance of trailing bytes let it return Ok. With a
    /// random id this failed on a small fraction of runs and read as order-dependent.
    #[test]
    fn test_bincode1_minimal_fixture_is_not_misread_by_a_varint_candidate() {
        let _serial = counter_guard();
        let bytes16 = [
            0x2fu8, 0x4b, 0xff, 0x82, 0x3e, 0x81, 0x4a, 0x0a, 0xb3, 0x00, 0x33, 0x37, 0xeb, 0x6f,
            0xf4, 0x3c,
        ];
        let id = MemoryId(uuid::Uuid::from_bytes(bytes16));
        let fixture = LegacyMinimalFixture {
            id: id.clone(),
            content: "legacy bincode1 minimal".to_string(),
        };
        let bytes = bincode1::serialize(&fixture).unwrap();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        assert_eq!(
            decoded.id, id,
            "the id came back displaced: a varint candidate misread the fixint prefix"
        );
        assert_eq!(decoded.experience.content, "legacy bincode1 minimal");
        assert!(is_legacy);
    }

    /// The full-memory legacy arm must keep accepting a record with bytes left over: that shape
    /// is a later schema that appended fields, and 24 of 9932 live records were exactly that.
    /// Applying the tiny-candidate full-consumption guard to this arm turned them into
    /// "corrupted memory entry" on the 2026-09-13 deploy.
    #[test]
    fn test_v3_record_with_trailing_bytes_still_decodes() {
        let _serial = counter_guard();
        #[derive(Serialize)]
        struct V3PlusAppendedFields {
            id: MemoryId,
            experience: Experience,
            importance: f32,
            access_count: u32,
            created_at: DateTime<Utc>,
            last_accessed: DateTime<Utc>,
            compressed: bool,
            tier: MemoryTier,
            entity_refs: Vec<EntityRef>,
            activation: f32,
            last_retrieval_id: Option<uuid::Uuid>,
            agent_id: Option<String>,
            run_id: Option<String>,
            actor_id: Option<String>,
            temporal_relevance: f32,
            score: Option<f32>,
            external_id: Option<String>,
            version: u32,
            history: Vec<MemoryRevision>,
            related_todo_ids: Vec<TodoId>,
            parent_id: Option<MemoryId>,
            anchored: bool,
            calibrated_confidence: Option<f32>,
            confidence_alpha: Option<f32>,
            confidence_beta: Option<f32>,
            valid_until: Option<DateTime<Utc>>,
            access_history: Option<Vec<DateTime<Utc>>>,
            elaboration_score: Option<f32>,
            fragment_demotion: Option<f32>,
            appended_later: u64,
            appended_later_too: String,
        }
        let id = MemoryId(uuid::Uuid::new_v4());
        let now = Utc::now();
        let fixture = V3PlusAppendedFields {
            id: id.clone(),
            experience: Experience {
                content: "a record from a schema between V3 and today".to_string(),
                ..Default::default()
            },
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            compressed: false,
            tier: MemoryTier::LongTerm,
            entity_refs: Vec::new(),
            activation: 0.0,
            last_retrieval_id: None,
            agent_id: None,
            run_id: None,
            actor_id: None,
            temporal_relevance: 1.0,
            score: None,
            external_id: None,
            version: 1,
            history: Vec::new(),
            related_todo_ids: Vec::new(),
            parent_id: None,
            anchored: false,
            calibrated_confidence: None,
            confidence_alpha: None,
            confidence_beta: None,
            valid_until: None,
            access_history: None,
            elaboration_score: None,
            fragment_demotion: None,
            appended_later: 7,
            appended_later_too: "trailing".to_string(),
        };
        let bytes = bincode::serde::encode_to_vec(&fixture, bincode::config::standard()).unwrap();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes)
            .expect("a V3 record with appended fields must still decode through the V3 arm");
        assert_eq!(decoded.id, id);
        assert_eq!(
            decoded.experience.content,
            "a record from a schema between V3 and today"
        );
        assert!(is_legacy);
    }

    use super::wire_fixtures::{encode, memory_with_session, strip_writer_structs};

    #[cfg(feature = "rocksdb")]
    /// A value that is not a memory but whose tail is UTF-8 used to "decode" through a raw-parse
    /// arm that took the id from the value's first 16 bytes, and `get` then rewrote that
    /// fabrication under the fabricated id. The arm is gone; this pins both halves.
    #[test]
    fn a_non_memory_value_with_a_utf8_tail_is_undecodable_and_never_rewritten() {
        let (storage, _dir) = test_storage();
        let key = MemoryId(uuid::Uuid::new_v4());
        let fabricated = uuid::Uuid::new_v4();
        let mut value = fabricated.as_bytes().to_vec();
        value.extend_from_slice(&[0x00, 0x01]);
        value.extend_from_slice(b"this was never a memory");
        storage.db.put(key.0.as_bytes(), &value).unwrap();

        assert!(
            storage.get(&key).is_err(),
            "a fabricated memory must not be served"
        );
        assert!(
            storage.db.get(fabricated.as_bytes()).unwrap().is_none(),
            "no phantom row"
        );
        assert_eq!(
            storage.db.get(key.0.as_bytes()).unwrap().unwrap(),
            value,
            "original bytes intact"
        );
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn lazy_migration_refuses_a_key_the_decoded_id_does_not_match() {
        let (storage, _dir) = test_storage();
        let memory = memory_with_session("legacy-shaped");
        let other = uuid::Uuid::new_v4();
        assert!(storage
            .migrate_memory_format(other.as_bytes(), &memory)
            .is_err());
        assert!(storage.db.get(memory.id.0.as_bytes()).unwrap().is_none());
        assert!(storage.db.get(other.as_bytes()).unwrap().is_none());
        storage
            .migrate_memory_format(memory.id.0.as_bytes(), &memory)
            .unwrap();
        assert!(storage.db.get(memory.id.0.as_bytes()).unwrap().is_some());
    }

    /// Strips the encoding of the fields appended for embedding spaces — two `None` spaces and
    /// an empty `other_vectors` — the wire every record had before they were appended.
    pub(crate) fn strip_space_fields(current: &[u8]) -> Vec<u8> {
        assert_eq!(
            &current[current.len() - 3..],
            &[0, 0, 0],
            "fixture must carry no spaces"
        );
        current[..current.len() - 3].to_vec()
    }

    #[test]
    fn a_record_from_before_the_space_fields_reads_back_whole() {
        let original = memory_with_session("written before spaces were named");
        let old = strip_space_fields(&encode(&original));
        assert!(
            bincode::serde::decode_from_slice::<Memory, _>(&old, bincode::config::standard())
                .is_err(),
            "the current layout must refuse the old wire, or this test guards nothing"
        );
        let (back, is_legacy) = deserialize_with_fallback(&old).expect("trailing-defaults arm");
        assert!(is_legacy, "must be re-serialised on the way back");
        assert_eq!(back.id, original.id);
        assert_eq!(
            back.when.episode_ordinal,
            Some(3),
            "the W3 facets after V3's last field survive, so the V3 arm did not read it"
        );
        assert_eq!(back.experience.embeddings, original.experience.embeddings);
        assert_eq!(back.experience.embedding_space, None);
        assert_eq!(
            back.experience
                .context
                .as_ref()
                .unwrap()
                .session
                .branch
                .as_deref(),
            Some("feature/x")
        );
        assert_eq!(decode_memory_snapshot(&old).unwrap().id, original.id);
    }

    /// The portable dump, and so the redb backup and `veld migrate`, carries a record's
    /// spaces with it.
    #[test]
    fn a_portable_dump_carries_the_spaces() {
        use crate::storage::portable::{read_all, PortableRecord, PortableWriter};
        let mut original = memory_with_session("dumped and restored");
        original.experience.embedding_space = SpaceId::parse("minilm-l6-v2@4");
        original
            .experience
            .keep_vector(SpaceId::new("nomic-embed-text-v1.5", 8), vec![0.5; 8]);
        let value = portable_memory_value(&encode(&original))
            .unwrap()
            .into_owned();
        let mut buf = Vec::new();
        let mut writer = PortableWriter::new(&mut buf).unwrap();
        writer
            .write_record(&PortableRecord {
                cf: "default".to_string(),
                key: original.id.0.as_bytes().to_vec(),
                value,
            })
            .unwrap();
        writer.finish().unwrap();
        let records = read_all(&buf[..]).unwrap();
        let back = decode_memory_snapshot(&records[0].value).unwrap();
        assert_eq!(
            back.experience.embedding_space,
            original.experience.embedding_space
        );
        assert_eq!(
            back.experience.other_vectors,
            original.experience.other_vectors
        );
    }

    #[test]
    fn a_record_with_its_spaces_is_current_and_keeps_them() {
        let mut original = memory_with_session("written with spaces");
        original.experience.embedding_space = SpaceId::parse("minilm-l6-v2@4");
        original
            .experience
            .set_secondary_vector(vec![0.5; 8], SpaceId::new("nomic-embed-text-v1.5", 8));
        let bytes = encode(&original);
        assert!(is_current_memory_encoding(&bytes));
        let (back, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        assert!(!is_legacy);
        assert_eq!(
            back.experience.embedding_space,
            original.experience.embedding_space
        );
        assert_eq!(
            back.experience
                .embedding_space_secondary
                .as_ref()
                .map(SpaceId::as_str),
            Some("nomic-embed-text-v1.5@8")
        );
    }

    #[test]
    fn a_pre_writer_record_from_before_the_space_fields_recovers_losslessly() {
        let original = memory_with_session("pre-writer and pre-space");
        let old = strip_space_fields(&strip_writer_structs(&encode(&original)));
        let (back, is_legacy) = deserialize_with_fallback(&old).expect("splice arm");
        assert!(is_legacy);
        assert_eq!(back.id, original.id);
        assert_eq!(back.when.episode_ordinal, Some(3));
        let session = &back.experience.context.as_ref().unwrap().session;
        assert_eq!(session.branch.as_deref(), Some("feature/x"));
    }

    #[test]
    fn a_record_with_a_pre_writer_session_recovers_through_the_splice() {
        let original = memory_with_session("written with a six-field session");
        let old = strip_writer_structs(&encode(&original));

        assert!(
            bincode::serde::decode_from_slice::<Memory, _>(&old, bincode::config::standard())
                .is_err(),
            "the current layout must refuse the old wire, or this test guards nothing"
        );
        let (back, is_legacy) = deserialize_with_fallback(&old).expect("recovery arm");
        assert!(is_legacy, "must be re-serialised on the way back");
        assert_eq!(back.id, original.id);
        assert_eq!(back.experience.content, original.experience.content);
        let session = &back.experience.context.as_ref().unwrap().session;
        assert_eq!(session.branch.as_deref(), Some("feature/x"));
        assert_eq!(session.declared, Default::default());
        assert_eq!(back.when.episode_ordinal, Some(3));
    }

    #[test]
    fn a_v3_record_with_a_pre_writer_session_recovers_and_keeps_its_facets() {
        let original = memory_with_session("written before persistent was inserted");
        // A V3 record predates the space fields too.
        let current = strip_space_fields(&encode(&original));
        let cfg = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
        let (_, persistent_at) = bincode::serde::decode_from_slice::<
            (
                MemoryId,
                Experience,
                f32,
                u32,
                DateTime<Utc>,
                DateTime<Utc>,
                bool,
            ),
            _,
        >(&current, cfg)
        .unwrap();
        assert_eq!(
            current[persistent_at], 1,
            "persistent=true sits after compressed"
        );
        let without_persistent =
            [&current[..persistent_at], &current[persistent_at + 1..]].concat();
        let old = strip_writer_structs(&without_persistent);

        let (back, is_legacy) = deserialize_with_fallback(&old).expect("V3 recovery arm");
        assert!(is_legacy);
        assert_eq!(back.experience.content, original.experience.content);
        assert_eq!(
            back.experience
                .context
                .as_ref()
                .unwrap()
                .session
                .branch
                .as_deref(),
            Some("feature/x")
        );
        assert_eq!(
            back.when.episode_ordinal,
            Some(3),
            "trailing W3.4 facets must survive, not default"
        );
        assert_eq!(back.kind, original.kind);
    }

    #[test]
    fn the_splice_refuses_records_that_already_carry_the_writer_structs() {
        let mut populated = memory_with_session("current layout, writer populated");
        populated
            .experience
            .context
            .as_mut()
            .unwrap()
            .session
            .declared
            .session_name = Some("commitwork-11".into());
        assert!(recover_pre_writer_session(&encode(&populated)).is_none());

        let defaults = memory_with_session("current layout, writer defaulted");
        assert!(recover_pre_writer_session(&encode(&defaults)).is_none());

        let mut no_context = memory_with_session("no context on the wire");
        no_context.experience.context = None;
        assert!(recover_pre_writer_session(&encode(&no_context)).is_none());
    }

    #[cfg(feature = "rocksdb")]
    /// Probe-only: AgentSession as written before 4a54053 appended `declared`/`verified`.
    /// bincode is positional, so `#[serde(default)]` on the appended fields does nothing for
    /// records already on disk with a populated context.
    // Every `Probe...` shape is decoded only to see whether a record parses as that layout,
    // and how far: no field is read.
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeAgentSessionPreWriter {
        worktree_path: Option<PathBuf>,
        branch: Option<String>,
        agent_id: Option<String>,
        vscode_window_id: Option<String>,
        started_at: Option<DateTime<Utc>>,
        parent_repo: Option<PathBuf>,
    }
    #[cfg(feature = "rocksdb")]
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeRichContextPreWriter {
        id: ContextId,
        conversation: ConversationContext,
        user: UserContext,
        project: ProjectContext,
        temporal: TemporalContext,
        semantic: SemanticContext,
        code: CodeContext,
        document: DocumentContext,
        environment: EnvironmentContext,
        emotional: EmotionalContext,
        source: SourceContext,
        episode: EpisodeContext,
        place: WhereFacet,
        who: WhoFacet,
        why: WhyFacet,
        binding: EngramBinding,
        session: ProbeAgentSessionPreWriter,
        parent: Option<Box<RichContext>>,
        embeddings: Option<Vec<f32>>,
        decay_rate: f32,
        created_at: DateTime<Utc>,
        updated_at: DateTime<Utc>,
    }
    #[cfg(feature = "rocksdb")]
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeExperiencePreWriter {
        experience_type: ExperienceType,
        content: String,
        context: Option<ProbeRichContextPreWriter>,
        entities: Vec<String>,
        metadata: HashMap<String, String>,
        embeddings: Option<Vec<f32>>,
        embeddings_secondary: Option<Vec<f32>>,
        embedding_degraded: bool,
        image_embeddings: Option<Vec<f32>>,
        audio_embeddings: Option<Vec<f32>>,
        video_embeddings: Option<Vec<f32>>,
        media_refs: Vec<MediaRef>,
        related_memories: Vec<MemoryId>,
        causal_chain: Vec<MemoryId>,
        outcomes: Vec<String>,
        robot_id: Option<String>,
        mission_id: Option<String>,
        geo_location: Option<[f64; 3]>,
        local_position: Option<[f32; 3]>,
        heading: Option<f32>,
        action_type: Option<String>,
        reward: Option<f32>,
        sensor_data: HashMap<String, f64>,
        decision_context: Option<HashMap<String, String>>,
        action_params: Option<HashMap<String, String>>,
        outcome_type: Option<String>,
        outcome_details: Option<String>,
        confidence: Option<f32>,
        alternatives_considered: Vec<String>,
        weather: Option<HashMap<String, String>>,
        terrain_type: Option<String>,
        lighting: Option<String>,
        nearby_agents: Vec<HashMap<String, String>>,
        is_failure: bool,
        is_anomaly: bool,
        severity: Option<String>,
        recovery_action: Option<String>,
        root_cause: Option<String>,
        pattern_id: Option<String>,
        predicted_outcome: Option<String>,
        prediction_accurate: Option<bool>,
        tags: Vec<String>,
        temporal_refs: Vec<String>,
        ner_entities: Vec<NerEntityRecord>,
        cooccurrence_pairs: Vec<(String, String)>,
    }
    #[cfg(feature = "rocksdb")]
    /// LegacyMemoryFlatV3 (before edfb43f inserted `persistent` mid-struct on 2026-09-02) plus the
    /// W3.4 trailing facets, so full consumption can be required.
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeMemoryFlatPreWriter {
        id: MemoryId,
        experience: ProbeExperiencePreWriter,
        importance: f32,
        access_count: u32,
        created_at: DateTime<Utc>,
        last_accessed: DateTime<Utc>,
        compressed: bool,
        tier: MemoryTier,
        entity_refs: Vec<EntityRef>,
        activation: f32,
        last_retrieval_id: Option<uuid::Uuid>,
        agent_id: Option<String>,
        run_id: Option<String>,
        actor_id: Option<String>,
        temporal_relevance: f32,
        score: Option<f32>,
        external_id: Option<String>,
        version: u32,
        history: Vec<MemoryRevision>,
        related_todo_ids: Vec<TodoId>,
        parent_id: Option<MemoryId>,
        anchored: bool,
        calibrated_confidence: Option<f32>,
        confidence_alpha: Option<f32>,
        confidence_beta: Option<f32>,
        valid_until: Option<DateTime<Utc>>,
        access_history: Option<Vec<DateTime<Utc>>>,
        elaboration_score: Option<f32>,
        fragment_demotion: Option<f32>,
        kind: RecordKind,
        what: WhatFacet,
        when: WhenFacet,
    }

    #[cfg(feature = "rocksdb")]
    /// Probe-only layout: veld's Experience WITHOUT the two fields upstream shodh-memory never
    /// had (embeddings_secondary, embedding_degraded), inside the V3 memory shape. A record that
    /// decodes here and nowhere else was written by an upstream-layout binary.
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeExperienceNoSecondary {
        experience_type: ExperienceType,
        content: String,
        context: Option<RichContext>,
        entities: Vec<String>,
        metadata: HashMap<String, String>,
        embeddings: Option<Vec<f32>>,
        image_embeddings: Option<Vec<f32>>,
        audio_embeddings: Option<Vec<f32>>,
        video_embeddings: Option<Vec<f32>>,
        media_refs: Vec<MediaRef>,
        related_memories: Vec<MemoryId>,
        causal_chain: Vec<MemoryId>,
        outcomes: Vec<String>,
        robot_id: Option<String>,
        mission_id: Option<String>,
        geo_location: Option<[f64; 3]>,
        local_position: Option<[f32; 3]>,
        heading: Option<f32>,
        #[serde(default)]
        _rest: (),
    }
    #[cfg(feature = "rocksdb")]
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ProbeMemoryNoSecondary {
        id: MemoryId,
        experience: ProbeExperienceNoSecondary,
    }

    #[cfg(feature = "rocksdb")]
    /// Offline diagnostic over a COPY of a store: for each id in VELD_PROBE_IDS (comma-separated),
    /// report which decoder candidate accepts the record, or the distinct error every candidate
    /// gives, and confirm the good records are untouched. Prints no content.
    ///
    ///   VELD_PROBE_STORE=/tmp/copy-of/portll/storage VELD_PROBE_IDS=uuid,uuid \
    ///     cargo test --lib -- --ignored probe_undecodable_records --nocapture
    #[test]
    #[ignore = "offline diagnostic over a copied store — run explicitly"]
    fn probe_undecodable_records() {
        let path = std::env::var("VELD_PROBE_STORE").expect("VELD_PROBE_STORE unset");
        let ids: Vec<MemoryId> = std::env::var("VELD_PROBE_IDS")
            .expect("VELD_PROBE_IDS unset")
            .split(',')
            .map(|s| MemoryId(uuid::Uuid::parse_str(s.trim()).expect("uuid")))
            .collect();
        let mut records: std::collections::BTreeMap<Vec<u8>, Vec<u8>> = Default::default();
        crate::storage::visit_records_read_only(std::path::Path::new(&path), |k, v| {
            records.insert(k.to_vec(), v.to_vec());
        })
        .expect("open the copied store read-only");
        let cfg = bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>();
        let mut verdicts: std::collections::BTreeMap<String, usize> = Default::default();
        for id in &ids {
            let v = records
                .get(id.0.as_bytes().as_slice())
                .expect("record present in default CF");
            let verdict = match deserialize_memory(v) {
                Ok((m, legacy)) => format!(
                    "DECODES legacy={legacy} content_len={}",
                    m.experience.content.len()
                ),
                Err(_) => {
                    let mut parts = Vec::new();
                    parts.push(format!(
                        "Memory: {}",
                        bincode::serde::decode_from_slice::<Memory, _>(v, cfg)
                            .err()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "ok".into())
                    ));
                    parts.push(format!(
                        "V3: {}",
                        bincode::serde::decode_from_slice::<LegacyMemoryFlatV3, _>(v, cfg)
                            .map(|(_, n)| format!("ok consumed {n}/{}", v.len()))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!(
                        "V2: {}",
                        bincode::serde::decode_from_slice::<LegacyMemoryFlatV2, _>(v, cfg)
                            .map(|(_, n)| format!("ok consumed {n}/{}", v.len()))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!(
                        "bincode1 V1: {}",
                        bincode1::deserialize::<LegacyMemoryV1>(v)
                            .map(|_| "ok".to_string())
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    let fix = bincode::config::standard()
                        .with_fixed_int_encoding()
                        .with_limit::<BINCODE_ALLOC_LIMIT>();
                    let leg = bincode::config::legacy().with_limit::<BINCODE_ALLOC_LIMIT>();
                    parts.push(format!(
                        "Memory FIXINT: {}",
                        bincode::serde::decode_from_slice::<Memory, _>(v, fix)
                            .map(|(m, n)| format!(
                                "ok consumed {n}/{} content_len={}",
                                v.len(),
                                m.experience.content.len()
                            ))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!(
                        "V3 FIXINT: {}",
                        bincode::serde::decode_from_slice::<LegacyMemoryFlatV3, _>(v, fix)
                            .map(|(_, n)| format!("ok consumed {n}/{}", v.len()))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!(
                        "Memory LEGACY(bincode1-compatible): {}",
                        bincode::serde::decode_from_slice::<Memory, _>(v, leg)
                            .map(|(m, n)| format!(
                                "ok consumed {n}/{} content_len={}",
                                v.len(),
                                m.experience.content.len()
                            ))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!(
                        "V3 LEGACY: {}",
                        bincode::serde::decode_from_slice::<LegacyMemoryFlatV3, _>(v, leg)
                            .map(|(_, n)| format!("ok consumed {n}/{}", v.len()))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    parts.push(format!("V3 + PRE-WRITER AgentSession (before edfb43f and 4a54053): {}", bincode::serde::decode_from_slice::<ProbeMemoryFlatPreWriter, _>(v, cfg).map(|(m, n)| format!("ok consumed {n}/{} content_len={} emb={} emb2={} created={} session_started={}", v.len(), m.experience.content.len(), m.experience.embeddings.as_ref().map(|e| e.len()).unwrap_or(0), m.experience.embeddings_secondary.as_ref().map(|e| e.len()).unwrap_or(0), m.created_at.format("%Y-%m-%d"), m.experience.context.as_ref().and_then(|c| c.session.started_at).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_else(|| "none".into()))).unwrap_or_else(|e| e.to_string())));
                    parts.push(format!(
                        "recover_pre_writer_session: {}",
                        recover_pre_writer_session(v)
                            .map(|(m, branch)| format!(
                                "{branch} content_len={} kind={:?} when.encoded_at={}",
                                m.experience.content.len(),
                                m.kind,
                                m.when
                                    .encoded_at
                                    .map(|d| d.format("%Y-%m-%d").to_string())
                                    .unwrap_or_else(|| "none".into())
                            ))
                            .unwrap_or_else(|| "none".into())
                    ));
                    parts.push(format!(
                        "V3 with an Experience lacking embeddings_secondary/embedding_degraded: {}",
                        bincode::serde::decode_from_slice::<ProbeMemoryNoSecondary, _>(v, cfg)
                            .map(|(m, n)| format!(
                                "ok consumed {n}/{} content_len={} entities={}",
                                v.len(),
                                m.experience.content.len(),
                                m.experience.entities.len()
                            ))
                            .unwrap_or_else(|e| e.to_string())
                    ));
                    // Structural walk of the prefix, numbers only: uuid, experience_type varint,
                    // content length, then the raw tag bytes that follow the content.
                    fn varint(b: &[u8], i: &mut usize) -> u64 {
                        let x = b[*i];
                        *i += 1;
                        match x {
                            0..=250 => x as u64,
                            251 => {
                                let v = u16::from_le_bytes([b[*i], b[*i + 1]]) as u64;
                                *i += 2;
                                v
                            }
                            252 => {
                                let v = u32::from_le_bytes([b[*i], b[*i + 1], b[*i + 2], b[*i + 3]])
                                    as u64;
                                *i += 4;
                                v
                            }
                            _ => {
                                let v = u64::from_le_bytes(b[*i..*i + 8].try_into().unwrap());
                                *i += 8;
                                v
                            }
                        }
                    }
                    let mut i = 0usize;
                    let uuid_len = varint(v, &mut i);
                    i += uuid_len as usize;
                    let etype = varint(v, &mut i);
                    let clen = varint(v, &mut i);
                    i += clen as usize;
                    // Deeper walk past Some(RichContext { id }): 48 bytes, printable runs masked.
                    let deep = &v[(i + 2 + 16).min(v.len())..(i + 2 + 16 + 48).min(v.len())];
                    let mut masked = String::new();
                    let mut run = 0usize;
                    for b in deep {
                        if (0x20..0x7f).contains(b) {
                            run += 1;
                            if run == 4 {
                                masked.push_str("…txt… ");
                            } else if run < 4 {
                                masked.push_str(&format!("{b:02x} "));
                            }
                        } else {
                            run = 0;
                            masked.push_str(&format!("{b:02x} "));
                        }
                    }
                    parts.push(format!("after RichContext.id: [{}]", masked.trim_end()));
                    let start = (i + 2 + 16).min(v.len());
                    let zeros = v[start..].iter().take_while(|b| **b == 0).count();
                    let next: Vec<String> = v
                        [(start + zeros).min(v.len())..(start + zeros + 8).min(v.len())]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    parts.push(format!(
                        "zero-run after RichContext.id = {zeros} bytes, then [{}]",
                        next.join(" ")
                    ));
                    let after: Vec<String> = v[i..(i + 12).min(v.len())]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    parts.push(format!("walk: uuid_len={uuid_len} experience_type={etype} content_len={clen} bytes_after_content=[{}] total={}", after.join(" "), v.len()));
                    // Field-by-field decode of RichContext then the Experience tail, each step
                    // reporting bytes consumed, so the first refusing field is named not guessed.
                    use crate::memory::facets::{
                        AgentSession, EngramBinding, WhereFacet, WhoFacet, WhyFacet,
                    };
                    fn step<T: serde::de::DeserializeOwned>(
                        v: &[u8],
                        off: &mut usize,
                        name: &str,
                        log: &mut Vec<String>,
                    ) -> bool {
                        match bincode::serde::decode_from_slice::<T, _>(
                            &v[*off..],
                            bincode::config::standard().with_limit::<BINCODE_ALLOC_LIMIT>(),
                        ) {
                            Ok((_, n)) => {
                                log.push(format!("{name}+{n}"));
                                *off += n;
                                true
                            }
                            Err(e) => {
                                let peek: Vec<String> = v[*off..(*off + 8).min(v.len())]
                                    .iter()
                                    .map(|b| format!("{b:02x}"))
                                    .collect();
                                log.push(format!(
                                    "{name} FAILS at offset {} of {}: {e} peek=[{}]",
                                    *off,
                                    v.len(),
                                    peek.join(" ")
                                ));
                                false
                            }
                        }
                    }
                    let mut log = Vec::new();
                    let mut off = i + 1;
                    let ok = step::<ContextId>(v, &mut off, "ctx.id", &mut log)
                        && step::<ConversationContext>(v, &mut off, "conversation", &mut log)
                        && step::<UserContext>(v, &mut off, "user", &mut log)
                        && step::<ProjectContext>(v, &mut off, "project", &mut log)
                        && step::<TemporalContext>(v, &mut off, "temporal", &mut log)
                        && step::<SemanticContext>(v, &mut off, "semantic", &mut log)
                        && step::<CodeContext>(v, &mut off, "code", &mut log)
                        && step::<DocumentContext>(v, &mut off, "document", &mut log)
                        && step::<EnvironmentContext>(v, &mut off, "environment", &mut log)
                        && step::<EmotionalContext>(v, &mut off, "emotional", &mut log)
                        && step::<SourceContext>(v, &mut off, "source", &mut log)
                        && step::<EpisodeContext>(v, &mut off, "episode", &mut log)
                        && step::<WhereFacet>(v, &mut off, "place", &mut log)
                        && step::<WhoFacet>(v, &mut off, "who", &mut log)
                        && step::<WhyFacet>(v, &mut off, "why", &mut log)
                        && step::<EngramBinding>(v, &mut off, "binding", &mut log)
                        && (step::<AgentSession>(v, &mut off, "session", &mut log)
                            || step::<ProbeAgentSessionPreWriter>(
                                v,
                                &mut off,
                                "session(pre-writer)",
                                &mut log,
                            ))
                        && step::<Option<Box<RichContext>>>(v, &mut off, "parent", &mut log)
                        && step::<Option<Vec<f32>>>(v, &mut off, "ctx.embeddings", &mut log)
                        && step::<f32>(v, &mut off, "decay_rate", &mut log)
                        && step::<DateTime<Utc>>(v, &mut off, "ctx.created_at", &mut log)
                        && step::<DateTime<Utc>>(v, &mut off, "ctx.updated_at", &mut log)
                        && step::<Vec<String>>(v, &mut off, "entities", &mut log)
                        && step::<std::collections::HashMap<String, String>>(
                            v, &mut off, "metadata", &mut log,
                        )
                        && step::<Option<Vec<f32>>>(v, &mut off, "embeddings", &mut log)
                        && step::<Option<Vec<f32>>>(v, &mut off, "embeddings_secondary", &mut log)
                        && step::<bool>(v, &mut off, "embedding_degraded", &mut log)
                        && step::<Option<Vec<f32>>>(v, &mut off, "image_embeddings", &mut log)
                        && step::<Option<Vec<f32>>>(v, &mut off, "audio_embeddings", &mut log)
                        && step::<Option<Vec<f32>>>(v, &mut off, "video_embeddings", &mut log)
                        && step::<Vec<MediaRef>>(v, &mut off, "media_refs", &mut log)
                        && step::<Vec<MemoryId>>(v, &mut off, "related_memories", &mut log)
                        && step::<Vec<MemoryId>>(v, &mut off, "causal_chain", &mut log)
                        && step::<Vec<String>>(v, &mut off, "outcomes", &mut log)
                        && step::<Option<String>>(v, &mut off, "robot_id", &mut log)
                        && step::<Option<String>>(v, &mut off, "mission_id", &mut log)
                        && step::<Option<[f64; 3]>>(v, &mut off, "geo_location", &mut log)
                        && step::<Option<[f32; 3]>>(v, &mut off, "local_position", &mut log)
                        && step::<Option<f32>>(v, &mut off, "heading", &mut log)
                        && step::<Option<String>>(v, &mut off, "action_type", &mut log)
                        && step::<Option<f32>>(v, &mut off, "reward", &mut log)
                        && step::<HashMap<String, f64>>(v, &mut off, "sensor_data", &mut log)
                        && step::<Option<HashMap<String, String>>>(
                            v,
                            &mut off,
                            "decision_context",
                            &mut log,
                        )
                        && step::<Option<HashMap<String, String>>>(
                            v,
                            &mut off,
                            "action_params",
                            &mut log,
                        )
                        && step::<Option<String>>(v, &mut off, "outcome_type", &mut log)
                        && step::<Option<String>>(v, &mut off, "outcome_details", &mut log)
                        && step::<Option<f32>>(v, &mut off, "confidence", &mut log)
                        && step::<Vec<String>>(v, &mut off, "alternatives_considered", &mut log)
                        && step::<Option<HashMap<String, String>>>(
                            v, &mut off, "weather", &mut log,
                        )
                        && step::<Option<String>>(v, &mut off, "terrain_type", &mut log)
                        && step::<Option<String>>(v, &mut off, "lighting", &mut log)
                        && step::<Vec<HashMap<String, String>>>(
                            v,
                            &mut off,
                            "nearby_agents",
                            &mut log,
                        )
                        && step::<bool>(v, &mut off, "is_failure", &mut log)
                        && step::<bool>(v, &mut off, "is_anomaly", &mut log)
                        && step::<Option<String>>(v, &mut off, "severity", &mut log)
                        && step::<Option<String>>(v, &mut off, "recovery_action", &mut log)
                        && step::<Option<String>>(v, &mut off, "root_cause", &mut log)
                        && step::<Option<String>>(v, &mut off, "pattern_id", &mut log)
                        && step::<Option<String>>(v, &mut off, "predicted_outcome", &mut log)
                        && step::<Option<bool>>(v, &mut off, "prediction_accurate", &mut log)
                        && step::<Vec<String>>(v, &mut off, "tags", &mut log)
                        && step::<Vec<String>>(v, &mut off, "temporal_refs", &mut log)
                        && step::<Vec<NerEntityRecord>>(v, &mut off, "ner_entities", &mut log)
                        && step::<Vec<(String, String)>>(
                            v,
                            &mut off,
                            "cooccurrence_pairs",
                            &mut log,
                        )
                        && step::<f32>(v, &mut off, "importance", &mut log)
                        && step::<u32>(v, &mut off, "access_count", &mut log)
                        && step::<DateTime<Utc>>(v, &mut off, "created_at", &mut log)
                        && step::<DateTime<Utc>>(v, &mut off, "last_accessed", &mut log)
                        && step::<bool>(v, &mut off, "compressed", &mut log)
                        && step::<MemoryTier>(v, &mut off, "tier", &mut log)
                        && step::<Vec<EntityRef>>(v, &mut off, "entity_refs", &mut log)
                        && step::<f32>(v, &mut off, "activation", &mut log)
                        && step::<Option<uuid::Uuid>>(v, &mut off, "last_retrieval_id", &mut log)
                        && step::<Option<String>>(v, &mut off, "agent_id", &mut log)
                        && step::<Option<String>>(v, &mut off, "run_id", &mut log)
                        && step::<Option<String>>(v, &mut off, "actor_id", &mut log)
                        && step::<f32>(v, &mut off, "temporal_relevance", &mut log)
                        && step::<Option<f32>>(v, &mut off, "score", &mut log)
                        && step::<Option<String>>(v, &mut off, "external_id", &mut log)
                        && step::<u32>(v, &mut off, "version", &mut log)
                        && step::<Vec<MemoryRevision>>(v, &mut off, "history", &mut log)
                        && step::<Vec<TodoId>>(v, &mut off, "related_todo_ids", &mut log)
                        && step::<Option<MemoryId>>(v, &mut off, "parent_id", &mut log)
                        && step::<bool>(v, &mut off, "anchored", &mut log)
                        && step::<Option<f32>>(v, &mut off, "calibrated_confidence", &mut log)
                        && step::<Option<f32>>(v, &mut off, "confidence_alpha", &mut log)
                        && step::<Option<f32>>(v, &mut off, "confidence_beta", &mut log)
                        && step::<Option<DateTime<Utc>>>(v, &mut off, "valid_until", &mut log)
                        && step::<Option<Vec<DateTime<Utc>>>>(
                            v,
                            &mut off,
                            "access_history",
                            &mut log,
                        )
                        && step::<Option<f32>>(v, &mut off, "elaboration_score", &mut log)
                        && step::<Option<f32>>(v, &mut off, "fragment_demotion", &mut log)
                        && step::<RecordKind>(v, &mut off, "kind", &mut log)
                        && step::<WhatFacet>(v, &mut off, "what", &mut log)
                        && step::<WhenFacet>(v, &mut off, "when", &mut log);
                    log.push(format!("END off={off} of {}", v.len()));
                    parts.push(format!("fieldwalk ok={ok} [{}]", log.join(" ")));
                    format!("UNDECODABLE — {}", parts.join(" | "))
                }
            };
            *verdicts.entry(verdict).or_default() += 1;
        }
        for (v, n) in &verdicts {
            eprintln!("{n} × {v}");
        }
        let mut good = 0usize;
        let mut good_bad = 0usize;
        let mut good_fixint = 0usize;
        let mut good_prewriter = 0usize;
        let mut good_recovered = 0usize;
        let mut good_with_context = 0usize;
        for (k, v) in &records {
            if k.len() != 16 || ids.iter().any(|i| i.0.as_bytes() == &k[..]) {
                continue;
            }
            good += 1;
            let has_context = match deserialize_memory(v) {
                Err(_) => {
                    good_bad += 1;
                    false
                }
                Ok((m, _)) => {
                    if m.experience.context.is_some() {
                        good_with_context += 1
                    };
                    m.experience.context.is_some()
                }
            };
            let fix = bincode::config::standard()
                .with_fixed_int_encoding()
                .with_limit::<BINCODE_ALLOC_LIMIT>();
            if bincode::serde::decode_from_slice::<Memory, _>(v, fix).is_ok() {
                good_fixint += 1;
            }
            // The two layouts only differ inside a populated context, so ambiguity is measured there.
            if has_context
                && matches!(bincode::serde::decode_from_slice::<ProbeMemoryFlatPreWriter, _>(v, cfg), Ok((_, n)) if n == v.len())
            {
                good_prewriter += 1;
            }
            if recover_pre_writer_session(v).is_some() {
                good_recovered += 1;
            }
        }
        eprintln!("control: {good} other records ({good_with_context} with context=Some), {good_bad} undecodable, {good_fixint} ALSO accepted by the fixint candidate (must be 0), {good_prewriter} of the context=Some records ALSO fully consumed by the pre-writer candidate (must be 0), {good_recovered} accepted by recover_pre_writer_session (must be 0)");
    }

    #[test]
    fn test_deserialize_with_fallback_msgpack_minimal_fixture() {
        let _serial = counter_guard();
        let id = MemoryId(uuid::Uuid::new_v4());
        let fixture = LegacyMinimalFixture {
            id: id.clone(),
            content: "legacy msgpack minimal".to_string(),
        };
        let bytes = rmp_serde::to_vec(&fixture).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["msgpack_minimal"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(is_legacy);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn test_crc32_simple() {
        let data = b"test data for CRC32";
        let crc1 = crc32_simple(data);
        let crc2 = crc32_simple(data);

        assert_eq!(crc1, crc2);
        assert_ne!(crc1, 0);

        let crc3 = crc32_simple(b"different data");
        assert_ne!(crc1, crc3);
    }

    #[test]
    fn test_crc32_empty() {
        let crc = crc32_simple(b"");
        assert_eq!(
            crc, 0,
            "IEEE CRC32 of empty input is 0 (init 0xFFFFFFFF XOR final 0xFFFFFFFF)"
        );
    }

    #[test]
    fn test_modality_dimension() {
        assert_eq!(Modality::Text.dimension(), 384);
        // ImageBind projects all non-text modalities to 1024-dim shared space
        assert_eq!(Modality::Image.dimension(), 1024);
        assert_eq!(Modality::Audio.dimension(), 1024);
        assert_eq!(Modality::Video.dimension(), 1024);
        assert_eq!(Modality::Unified.dimension(), 1024);
    }

    #[test]
    fn test_modality_as_str() {
        assert_eq!(Modality::Text.as_str(), "text");
        assert_eq!(Modality::Image.as_str(), "image");
        assert_eq!(Modality::Audio.as_str(), "audio");
        assert_eq!(Modality::Video.as_str(), "video");
    }

    #[test]
    fn test_vector_mapping_entry_with_text() {
        let entry = VectorMappingEntry::with_text(vec![1, 2, 3]);

        assert_eq!(entry.text_vectors(), Some(&vec![1, 2, 3]));
        assert!(!entry.is_empty());
    }

    #[test]
    fn test_vector_mapping_entry_multimodal() {
        let entry = VectorMappingEntry::with_text(vec![1])
            .with_image(vec![2])
            .with_audio(vec![3])
            .with_video(vec![4]);

        let all = entry.all_vector_ids();
        assert_eq!(all.len(), 4);

        assert!(all.contains(&(Modality::Text, 1)));
        assert!(all.contains(&(Modality::Image, 2)));
        assert!(all.contains(&(Modality::Audio, 3)));
        assert!(all.contains(&(Modality::Video, 4)));
    }

    #[test]
    fn test_vector_mapping_entry_empty() {
        let entry = VectorMappingEntry::default();

        assert!(entry.is_empty());
        assert!(entry.text_vectors().is_none());
        assert!(entry.all_vector_ids().is_empty());
    }

    #[test]
    fn test_vector_mapping_entry_add_modality() {
        let mut entry = VectorMappingEntry::default();
        entry.add_modality(Modality::Text, vec![1, 2]);

        assert_eq!(entry.text_vectors(), Some(&vec![1, 2]));
    }

    #[test]
    fn test_storage_stats_default() {
        let stats = StorageStats::default();

        assert_eq!(stats.total_count, 0);
        assert_eq!(stats.compressed_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.total_retrievals, 0);
    }

    #[test]
    fn test_search_criteria_variants() {
        let criteria1 = SearchCriteria::ByEntity("test".to_string());
        let criteria2 = SearchCriteria::ByImportance { min: 0.5, max: 1.0 };
        let criteria3 = SearchCriteria::ByType(ExperienceType::Observation);

        assert!(matches!(criteria1, SearchCriteria::ByEntity(_)));
        assert!(matches!(criteria2, SearchCriteria::ByImportance { .. }));
        assert!(matches!(criteria3, SearchCriteria::ByType(_)));
    }

    #[test]
    fn test_search_criteria_by_date() {
        let now = Utc::now();
        let start = now - chrono::Duration::days(7);
        let criteria = SearchCriteria::ByDate { start, end: now };

        if let SearchCriteria::ByDate { start: s, end: e } = criteria {
            assert!(s < e);
        } else {
            panic!("Expected ByDate");
        }
    }

    #[test]
    fn test_search_criteria_combined() {
        let criteria = SearchCriteria::Combined(vec![
            SearchCriteria::ByEntity("test".to_string()),
            SearchCriteria::ByImportance { min: 0.5, max: 1.0 },
        ]);

        if let SearchCriteria::Combined(inner) = criteria {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("Expected Combined");
        }
    }

    #[test]
    fn test_modality_vectors_struct() {
        let mv = ModalityVectors {
            vector_ids: vec![1, 2, 3],
            dimension: 384,
            chunk_ranges: None,
        };

        assert_eq!(mv.vector_ids.len(), 3);
        assert_eq!(mv.dimension, 384);
        assert!(mv.chunk_ranges.is_none());
    }

    // ========================================================================
    // R7 — inbound-reference reverse index (F2)
    // ========================================================================

    #[cfg(feature = "rocksdb")]
    /// Spin up a real per-user MemoryStorage backed by a temp dir.
    fn test_storage() -> (MemoryStorage, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().expect("create tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("open storage");
        (storage, dir)
    }

    #[cfg(feature = "rocksdb")]
    /// Build a memory whose experience.related_memories points at `targets`.
    fn memory_referencing(id: MemoryId, targets: &[MemoryId]) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: format!("holder {}", id.0),
            related_memories: targets.to_vec(),
            ..Default::default()
        };
        Memory::from_legacy(
            id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn r7_forward_ref_targets_harvests_all_memory_id_fields() {
        let a = MemoryId(uuid::Uuid::new_v4());
        let b = MemoryId(uuid::Uuid::new_v4());
        let c = MemoryId(uuid::Uuid::new_v4());
        let d = MemoryId(uuid::Uuid::new_v4());
        let holder = MemoryId(uuid::Uuid::new_v4());

        let now = Utc::now();
        let mut ctx = crate::memory::context::ContextBuilder::new().build();
        ctx.why.causes = vec![CausalLink {
            other: d.clone(),
            relation: crate::memory::facets::CausalKind::Caused,
            confidence: 0.7,
            inferred: false,
        }];
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "multi-ref holder".to_string(),
            related_memories: vec![a.clone()],
            causal_chain: vec![b.clone()],
            context: Some(ctx),
            ..Default::default()
        };
        let mut memory = Memory::from_legacy(
            holder.clone(),
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        );
        memory.parent_id = Some(c.clone());

        let targets = memory.forward_ref_targets();
        assert!(targets.contains(&a), "related_memories harvested");
        assert!(targets.contains(&b), "causal_chain harvested");
        assert!(targets.contains(&c), "parent_id harvested");
        assert!(targets.contains(&d), "WhyFacet CausalLink.other harvested");
        assert_eq!(targets.len(), 4, "exactly the four distinct targets");
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn r7_forward_ref_targets_dedups_and_excludes_self() {
        let a = MemoryId(uuid::Uuid::new_v4());
        let holder = MemoryId(uuid::Uuid::new_v4());
        // Reference A twice (related + causal) and reference self once.
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "dup holder".to_string(),
            related_memories: vec![a.clone(), holder.clone()],
            causal_chain: vec![a.clone()],
            ..Default::default()
        };
        let memory = Memory::from_legacy(
            holder.clone(),
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        );
        let targets = memory.forward_ref_targets();
        assert_eq!(targets, vec![a], "A once, self excluded");
    }

    #[cfg(feature = "rocksdb")]
    /// (1) store M referencing A,B → inbound_refs(A) contains M; has_inbound_refs(A) true.
    #[test]
    fn r7_store_records_inbound_refs() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let b = MemoryId(uuid::Uuid::new_v4());
        let m = MemoryId(uuid::Uuid::new_v4());

        storage
            .store(&memory_referencing(m.clone(), &[a.clone(), b.clone()]))
            .unwrap();

        let holders_a = storage.inbound_refs(&a).unwrap();
        assert!(holders_a.contains(&m), "A's inbound refs include M");
        assert!(storage.has_inbound_refs(&a).unwrap(), "A has inbound refs");
        assert!(storage.inbound_refs(&b).unwrap().contains(&m), "B too");
        assert!(storage.has_inbound_refs(&b).unwrap());

        // A target nobody references has no inbound refs.
        let unref = MemoryId(uuid::Uuid::new_v4());
        assert!(!storage.has_inbound_refs(&unref).unwrap());
        assert!(storage.inbound_refs(&unref).unwrap().is_empty());
    }

    #[cfg(feature = "rocksdb")]
    /// (2) delete M → has_inbound_refs(A) false (teardown).
    #[test]
    fn r7_delete_tears_down_inbound_refs() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let m = MemoryId(uuid::Uuid::new_v4());

        storage
            .store(&memory_referencing(m.clone(), std::slice::from_ref(&a)))
            .unwrap();
        assert!(storage.has_inbound_refs(&a).unwrap());

        storage.delete(&m).unwrap();
        assert!(
            !storage.has_inbound_refs(&a).unwrap(),
            "deleting holder M tears down A's inbound ref"
        );
        assert!(storage.inbound_refs(&a).unwrap().is_empty());
    }

    #[cfg(feature = "rocksdb")]
    /// (3) update M to DROP ref to A → inbound_refs(A) no longer contains M.
    /// This is the update-diff hazard: update() queues the old version's removals
    /// and the new version's rows in one batch; Option-b teardown (remove all
    /// inbound_ref:*:{M}) must purge the stale A-entry even though the NEW memory
    /// no longer names A.
    #[test]
    fn r7_update_drop_ref_clears_stale_inbound() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let b = MemoryId(uuid::Uuid::new_v4());
        let m = MemoryId(uuid::Uuid::new_v4());

        // M references A and B.
        storage
            .store(&memory_referencing(m.clone(), &[a.clone(), b.clone()]))
            .unwrap();
        assert!(storage.inbound_refs(&a).unwrap().contains(&m));
        assert!(storage.inbound_refs(&b).unwrap().contains(&m));

        // Update M to reference only B (drops A).
        storage
            .update(&memory_referencing(m.clone(), std::slice::from_ref(&b)))
            .unwrap();

        assert!(
            !storage.inbound_refs(&a).unwrap().contains(&m),
            "stale inbound ref to dropped target A removed on update"
        );
        assert!(!storage.has_inbound_refs(&a).unwrap());
        assert!(
            storage.inbound_refs(&b).unwrap().contains(&m),
            "retained target B still has M as inbound holder"
        );
    }

    #[cfg(feature = "rocksdb")]
    /// (4) per-user isolation: user-A's index never returns user-B's holders.
    /// Each user has its own MemoryStorage (separate DB), so a target id that
    /// exists in both stores resolves only to the local holder.
    #[test]
    fn r7_per_user_isolation() {
        let (store_a, _dir_a) = test_storage();
        let (store_b, _dir_b) = test_storage();

        // Same target id referenced by a different holder in each user store.
        let target = MemoryId(uuid::Uuid::new_v4());
        let holder_a = MemoryId(uuid::Uuid::new_v4());
        let holder_b = MemoryId(uuid::Uuid::new_v4());

        store_a
            .store(&memory_referencing(
                holder_a.clone(),
                std::slice::from_ref(&target),
            ))
            .unwrap();
        store_b
            .store(&memory_referencing(
                holder_b.clone(),
                std::slice::from_ref(&target),
            ))
            .unwrap();

        let a_holders = store_a.inbound_refs(&target).unwrap();
        let b_holders = store_b.inbound_refs(&target).unwrap();

        assert_eq!(
            a_holders,
            vec![holder_a.clone()],
            "store A sees only its holder"
        );
        assert_eq!(
            b_holders,
            vec![holder_b.clone()],
            "store B sees only its holder"
        );
        assert!(
            !a_holders.contains(&holder_b),
            "no cross-tenant leader leak"
        );
        assert!(
            !b_holders.contains(&holder_a),
            "no cross-tenant holder leak"
        );
    }

    // ========================================================================
    // R7b — Fold-3 fact-source coverage tests
    // ========================================================================
    //
    // Three properties must hold for the GC gate to be safe:
    //   (a) A memory cited ONLY by SemanticFact.source_memories → has_any_inbound_refs == true
    //   (b) A memory cited ONLY by TemporalFact.source_memory_id → has_any_inbound_refs == true
    //   (c) A memory with no citations from any source → has_any_inbound_refs == false

    #[cfg(feature = "rocksdb")]
    /// (R7b-a) Memory cited only by a SemanticFact.source_memories entry reports
    /// has_any_inbound_refs == true even though the Memory-store reverse index is
    /// empty for it (no Memory-to-Memory edges point at it).
    #[test]
    fn r7b_semantic_fact_source_memories_detected() {
        use super::super::compression::{FactType, SemanticFact};
        use super::super::facts::SemanticFactStore;

        let (storage, _dir) = test_storage();
        let target = MemoryId(uuid::Uuid::new_v4());

        // Sanity: Memory-store reverse index is empty for `target`.
        assert!(
            !storage.has_inbound_refs(&target).unwrap(),
            "no memory-store refs yet"
        );

        // Store a SemanticFact whose source_memories contains target, using the
        // same underlying DB as the MemoryStorage.
        let fact_store = SemanticFactStore::new(storage.db());
        let fact = SemanticFact {
            id: "sf-r7b-a".to_string(),
            fact: "test fact for r7b".to_string(),
            confidence: 0.9,
            support_count: 1,
            source_memories: vec![target.clone()],
            related_entities: vec![],
            created_at: Utc::now(),
            last_reinforced: Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: vec![],
            purged_at: None,
            purge_reason: None,
        };
        fact_store.store("test-user", &fact).unwrap();

        // Full GC guard must now report true.
        assert!(
            storage.has_any_inbound_refs(&target, "test-user").unwrap(),
            "memory cited by SemanticFact.source_memories must have inbound refs"
        );
        // Bare Memory-store reverse index is still empty — this distinguishes the
        // two checks and confirms R7b adds coverage on top of R7.
        assert!(
            !storage.has_inbound_refs(&target).unwrap(),
            "Memory-store reverse index must NOT be affected"
        );
    }

    #[cfg(feature = "rocksdb")]
    /// (R7b-b) Memory cited only by a TemporalFact.source_memory_id entry reports
    /// has_any_inbound_refs == true.
    #[test]
    fn r7b_temporal_fact_source_memory_id_detected() {
        use super::super::temporal_facts::{
            EventType, ResolvedTime, TemporalFact, TemporalFactStore,
        };

        let (storage, _dir) = test_storage();
        let target = MemoryId(uuid::Uuid::new_v4());

        assert!(
            !storage.has_inbound_refs(&target).unwrap(),
            "no memory-store refs yet"
        );

        let temporal_store = TemporalFactStore::new(storage.db());
        let fact = TemporalFact {
            id: "tf-r7b-b".to_string(),
            entity: "Alice".to_string(),
            event: "camping".to_string(),
            event_stems: vec!["camp".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("next month".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: target.clone(),
            conversation_date: Utc::now(),
            confidence: 0.85,
            source_text: "Alice is planning to go camping next month.".to_string(),
            valid_from: None,
            valid_until: None,
        };
        temporal_store.store("test-user", &fact).unwrap();

        assert!(
            storage.has_any_inbound_refs(&target, "test-user").unwrap(),
            "memory cited by TemporalFact.source_memory_id must have inbound refs"
        );
        assert!(
            !storage.has_inbound_refs(&target).unwrap(),
            "Memory-store reverse index must NOT be affected"
        );
    }

    #[cfg(feature = "rocksdb")]
    /// (R7b-c) A memory with no refs from any source (no Memory-to-Memory edges,
    /// no SemanticFact citations, no TemporalFact citations) reports false.
    #[test]
    fn r7b_unreferenced_memory_reports_false() {
        let (storage, _dir) = test_storage();
        let orphan = MemoryId(uuid::Uuid::new_v4());

        assert!(
            !storage.has_any_inbound_refs(&orphan, "test-user").unwrap(),
            "completely unreferenced memory must report false"
        );
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn removing_one_holder_leaves_the_other_holders_of_a_target() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let b = MemoryId(uuid::Uuid::new_v4());
        let m1 = MemoryId(uuid::Uuid::new_v4());
        let m2 = MemoryId(uuid::Uuid::new_v4());
        let m3 = MemoryId(uuid::Uuid::new_v4());

        storage
            .store(&memory_referencing(m1.clone(), std::slice::from_ref(&a)))
            .unwrap();
        storage
            .store(&memory_referencing(m2.clone(), &[a.clone(), b.clone()]))
            .unwrap();
        storage
            .store(&memory_referencing(m3.clone(), std::slice::from_ref(&a)))
            .unwrap();

        storage.delete(&m1).unwrap();
        let mut holders = storage.inbound_refs(&a).unwrap();
        holders.sort_by_key(|h| h.0);
        let mut expected = vec![m2.clone(), m3.clone()];
        expected.sort_by_key(|h| h.0);
        assert_eq!(holders, expected, "only m1's entry for A goes");

        storage
            .update(&memory_referencing(m2.clone(), std::slice::from_ref(&b)))
            .unwrap();
        assert_eq!(storage.inbound_refs(&a).unwrap(), vec![m3.clone()]);
        assert_eq!(storage.inbound_refs(&b).unwrap(), vec![m2]);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn an_undecodable_holder_still_has_its_inbound_refs_removed() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let m = MemoryId(uuid::Uuid::new_v4());
        let other = MemoryId(uuid::Uuid::new_v4());

        storage
            .store(&memory_referencing(m.clone(), std::slice::from_ref(&a)))
            .unwrap();
        storage
            .store(&memory_referencing(other.clone(), std::slice::from_ref(&a)))
            .unwrap();
        storage.db.put(m.0.as_bytes(), [0xffu8; 3]).unwrap();
        assert!(storage.get(&m).is_err(), "the fixture must be undecodable");

        storage.delete(&m).unwrap();
        assert_eq!(storage.inbound_refs(&a).unwrap(), vec![other]);
    }

    #[cfg(feature = "rocksdb")]
    fn aged_memory(age_days: i64, compressed: bool) -> Memory {
        let mut memory = memory_referencing(MemoryId(uuid::Uuid::new_v4()), &[]);
        memory.created_at = Utc::now() - chrono::Duration::days(age_days);
        memory.compressed = compressed;
        memory
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn uncompressed_candidates_come_from_the_time_index() {
        let (storage, _dir) = test_storage();
        let old = aged_memory(30, false);
        let young = aged_memory(1, false);
        let old_compressed = aged_memory(30, true);
        for memory in [&old, &young, &old_compressed] {
            storage.store(memory).unwrap();
        }
        let cutoff = Utc::now() - chrono::Duration::days(7);

        let found: Vec<MemoryId> = storage
            .get_uncompressed_older_than(cutoff)
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(found, vec![old.id.clone()]);

        let mut compressed = old.clone();
        compressed.compressed = true;
        storage.update(&compressed).unwrap();
        assert!(storage
            .get_uncompressed_older_than(cutoff)
            .unwrap()
            .is_empty());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn records_written_before_the_time_index_are_backfilled_once() {
        let (storage, _dir) = test_storage();
        let old = aged_memory(30, false);
        storage.store(&old).unwrap();
        let idx = storage.index_cf();
        storage
            .db
            .delete_cf(
                idx,
                uncompressed_index_key(old.created_at, &old.id).as_bytes(),
            )
            .unwrap();
        storage
            .db
            .delete_cf(idx, UNCOMPRESSED_INDEX_SENTINEL)
            .unwrap();

        let cutoff = Utc::now() - chrono::Duration::days(7);
        let found = storage.get_uncompressed_older_than(cutoff).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, old.id);
        assert!(storage
            .db
            .get_cf(idx, UNCOMPRESSED_INDEX_SENTINEL)
            .unwrap()
            .is_some());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn a_stale_time_index_entry_is_dropped_when_met() {
        let (storage, _dir) = test_storage();
        let old = aged_memory(30, false);
        storage.store(&old).unwrap();
        let mut compressed = old.clone();
        compressed.compressed = true;
        storage.store(&compressed).unwrap();
        let key = uncompressed_index_key(old.created_at, &old.id);
        let idx = storage.index_cf();
        // What a write racing the backfill leaves behind (see `ensure_uncompressed_index`).
        storage.db.put_cf(idx, key.as_bytes(), b"1").unwrap();

        let cutoff = Utc::now() - chrono::Duration::days(7);
        assert!(storage
            .get_uncompressed_older_than(cutoff)
            .unwrap()
            .is_empty());
        assert!(storage.db.get_cf(idx, key.as_bytes()).unwrap().is_none());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn the_time_index_orders_pre_epoch_timestamps_first() {
        let before = DateTime::<Utc>::from_timestamp_millis(-1).unwrap();
        let epoch = DateTime::<Utc>::from_timestamp_millis(0).unwrap();
        let after = DateTime::<Utc>::from_timestamp_millis(1).unwrap();
        assert!(sortable_millis(before) < sortable_millis(epoch));
        assert!(sortable_millis(epoch) < sortable_millis(after));
        let id = MemoryId(uuid::Uuid::nil());
        assert!(uncompressed_index_key(before, &id) < uncompressed_index_key(after, &id));
    }

    #[cfg(feature = "rocksdb")]
    fn index_has(storage: &MemoryStorage, key: &str) -> bool {
        storage
            .db
            .get_cf(storage.index_cf(), key.as_bytes())
            .unwrap()
            .is_some()
    }

    #[cfg(feature = "rocksdb")]
    fn index_keys_for(memory: &Memory) -> Vec<String> {
        vec![
            format!(
                "date:{}:{}",
                memory.created_at.format("%Y%m%d"),
                memory.id.0
            ),
            format!(
                "type:{:?}:{}",
                memory.experience.experience_type, memory.id.0
            ),
            format!(
                "importance:{}:{}",
                (memory.importance() * 10.0) as u32,
                memory.id.0
            ),
            uncompressed_index_key(memory.created_at, &memory.id),
        ]
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn store_commits_record_and_indices_together() {
        let (storage, _dir) = test_storage();
        let target = MemoryId(uuid::Uuid::new_v4());
        let m = memory_referencing(MemoryId(uuid::Uuid::new_v4()), &[target.clone()]);

        storage.store(&m).unwrap();

        assert!(storage.db.get(m.id.0.as_bytes()).unwrap().is_some());
        for key in index_keys_for(&m) {
            assert!(
                index_has(&storage, &key),
                "index key {key} missing after store"
            );
        }
        assert!(storage.has_inbound_refs(&target).unwrap());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn update_rewrites_record_and_indices_in_one_commit() {
        let (storage, _dir) = test_storage();
        let a = MemoryId(uuid::Uuid::new_v4());
        let b = MemoryId(uuid::Uuid::new_v4());
        let id = MemoryId(uuid::Uuid::new_v4());
        storage
            .store(&memory_referencing(id.clone(), &[a.clone()]))
            .unwrap();

        let updated = memory_referencing(id.clone(), &[b.clone()]);
        storage.update(&updated).unwrap();

        assert!(
            !storage.has_inbound_refs(&a).unwrap(),
            "dropped ref must go"
        );
        assert!(storage.has_inbound_refs(&b).unwrap(), "new ref must land");
        for key in index_keys_for(&updated) {
            assert!(
                index_has(&storage, &key),
                "index key {key} missing after update"
            );
        }
        assert!(storage.get(&id).is_ok());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn multimodal_store_commits_mapping_and_indices_together() {
        let (storage, _dir) = test_storage();
        let m = memory_referencing(MemoryId(uuid::Uuid::new_v4()), &[]);

        storage
            .store_with_multimodal_vectors(&m, Modality::Text, vec![7, 8])
            .unwrap();

        assert!(storage.get(&m.id).is_ok());
        assert!(storage.get_vector_mapping(&m.id).unwrap().is_some());
        for key in index_keys_for(&m) {
            assert!(index_has(&storage, &key), "index key {key} missing");
        }
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn shared_cache_and_write_buffer_manager_open_round_trips_data() {
        let cache = Cache::new_lru_cache(8 * 1024 * 1024);
        let dir = tempfile::TempDir::new().unwrap();
        let m = memory_referencing(MemoryId(uuid::Uuid::new_v4()), &[]);
        {
            let storage = MemoryStorage::new(dir.path(), Some(&cache)).unwrap();
            storage.store(&m).unwrap();
            storage.flush().unwrap();
        }
        let storage = MemoryStorage::new(dir.path(), Some(&cache)).unwrap();
        assert!(storage.get(&m.id).is_ok());
        for key in index_keys_for(&m) {
            assert!(
                index_has(&storage, &key),
                "index key {key} lost across reopen"
            );
        }
        assert!(super::shared_write_buffer_manager(&cache).enabled());
    }

    // ========================================================================
    // Deletion, external-id lookup and tuning
    // ========================================================================

    #[cfg(feature = "rocksdb")]
    /// A memory carrying a row in every keyed index: external id, episode and sequence,
    /// tags, entities, parent (which is also an inbound reference).
    fn fully_indexed_memory(parent: &MemoryId, external_id: &str) -> Memory {
        let episode = EpisodeContext {
            episode_id: Some("episode-under-delete".to_string()),
            sequence_number: Some(3),
            ..Default::default()
        };
        let mut memory = sample_memory(MemoryId(uuid::Uuid::new_v4()), "indexed everywhere");
        memory.experience.tags = vec!["Doomed".to_string(), "kept-tag".to_string()];
        memory.experience.entities = vec!["Wellington".to_string()];
        memory.experience.context = Some(
            crate::memory::context::ContextBuilder::new()
                .with_episode(episode)
                .build(),
        );
        memory.parent_id = Some(parent.clone());
        memory.external_id = Some(external_id.to_string());
        memory
    }

    #[cfg(feature = "rocksdb")]
    /// Every `memory_index` key that ends with `id` — the rows the memory owns.
    fn index_rows_naming(storage: &MemoryStorage, id: &MemoryId) -> Vec<String> {
        let suffix = format!(":{}", id.0);
        storage
            .db
            .iterator_cf(storage.index_cf(), IteratorMode::Start)
            .log_errors()
            .map(|(key, _)| String::from_utf8_lossy(&key).into_owned())
            .filter(|key| key.ends_with(&suffix))
            .collect()
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn delete_with_vectors_removes_every_index_row_the_memory_owned() {
        let (storage, _dir) = test_storage();
        let parent = sample_memory(MemoryId(uuid::Uuid::new_v4()), "the parent");
        storage.store(&parent).unwrap();
        let memory = fully_indexed_memory(&parent.id, "linear:SHO-1");
        storage.store_with_vectors(&memory, vec![7]).unwrap();
        assert!(
            !index_rows_naming(&storage, &memory.id).is_empty(),
            "the fixture wrote no index rows to remove"
        );

        storage.delete_with_vectors(&memory.id).unwrap();

        assert_eq!(
            index_rows_naming(&storage, &memory.id),
            Vec::<String>::new()
        );
        assert!(storage
            .find_by_external_id("linear:SHO-1")
            .unwrap()
            .is_none());
        assert!(storage.get_children(&parent.id).unwrap().is_empty());
        assert!(!storage.has_inbound_refs(&parent.id).unwrap());
        assert!(storage
            .search(SearchCriteria::ByTags(vec!["doomed".to_string()]))
            .unwrap()
            .is_empty());
        assert!(storage
            .search(SearchCriteria::ByEpisode(
                "episode-under-delete".to_string()
            ))
            .unwrap()
            .is_empty());
        assert!(storage.get_vector_mapping(&memory.id).unwrap().is_none());
        assert!(storage.get_by_content_hash("indexed everywhere").is_none());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn a_single_holder_content_hash_row_is_converted_on_open() {
        let dir = tempfile::TempDir::new().expect("create tempdir");
        let stored = sample_memory(MemoryId(uuid::Uuid::new_v4()), "written by an older veld");
        {
            let storage = MemoryStorage::new(dir.path(), None).expect("open storage");
            storage.store(&stored).unwrap();
            let idx = storage.index_cf();
            let row = storage.content_hash_row(&stored);
            let single_holder = row
                .strip_suffix(&format!(":{}", stored.id.0))
                .expect("row ends with its id")
                .to_string();
            let mut batch = WriteBatch::default();
            batch.delete_cf(idx, row.as_bytes());
            batch.delete_cf(idx, CONTENT_HASH_LAYOUT_MARKER);
            batch.put_cf(idx, single_holder.as_bytes(), stored.id.0.as_bytes());
            storage.db.write(batch).unwrap();
            assert_eq!(
                storage.get_by_content_hash("written by an older veld"),
                None,
                "the fixture holds only the old row"
            );
        }

        let storage = MemoryStorage::new(dir.path(), None).expect("reopen storage");
        assert_eq!(
            storage.get_by_content_hash("written by an older veld"),
            Some(stored.id.clone())
        );
        let idx = storage.index_cf();
        let content_rows: Vec<String> = storage
            .db
            .prefix_iterator_cf(idx, b"content_hash:")
            .log_errors()
            .map(|(key, _)| String::from_utf8_lossy(&key).into_owned())
            .take_while(|key| key.starts_with("content_hash:"))
            .collect();
        assert_eq!(content_rows, vec![storage.content_hash_row(&stored)]);
        assert!(storage
            .db
            .get_cf(idx, CONTENT_HASH_LAYOUT_MARKER)
            .unwrap()
            .is_some());

        drop(storage);
        let reopened = MemoryStorage::new(dir.path(), None).expect("open a third time");
        assert_eq!(
            reopened.get_by_content_hash("written by an older veld"),
            Some(stored.id.clone()),
            "the conversion is idempotent"
        );
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn a_store_over_a_live_id_replaces_the_previous_versions_index_rows() {
        type Write = fn(&MemoryStorage, &Memory) -> Result<()>;
        let writes: [(&str, Write); 3] = [
            ("store", |storage, memory| storage.store(memory)),
            ("update", |storage, memory| storage.update(memory)),
            ("store_with_vectors", |storage, memory| {
                storage.store_with_vectors(memory, vec![3])
            }),
        ];
        for (path, write) in writes {
            let (storage, _dir) = test_storage();
            let parent = sample_memory(MemoryId(uuid::Uuid::new_v4()), "the parent");
            storage.store(&parent).unwrap();
            let original = fully_indexed_memory(&parent.id, "linear:SHO-7");
            storage.store(&original).unwrap();

            let mut rewritten = sample_memory(original.id.clone(), "rewritten content");
            rewritten.experience.tags = vec!["kept-tag".to_string()];
            write(&storage, &rewritten).unwrap();

            let (fresh, _fresh_dir) = test_storage();
            fresh.store(&rewritten).unwrap();
            let mut expected = index_rows_naming(&fresh, &rewritten.id);
            expected.sort();
            let mut held = index_rows_naming(&storage, &original.id);
            held.sort();
            assert_eq!(
                held, expected,
                "{path}: the index holds rows only the previous version had"
            );
            assert!(
                storage
                    .find_by_external_id("linear:SHO-7")
                    .unwrap()
                    .is_none(),
                "{path}"
            );
            assert!(
                storage.get_children(&parent.id).unwrap().is_empty(),
                "{path}"
            );
            assert!(!storage.has_inbound_refs(&parent.id).unwrap(), "{path}");
            assert!(
                storage
                    .search(SearchCriteria::ByTags(vec!["doomed".to_string()]))
                    .unwrap()
                    .is_empty(),
                "{path}"
            );
            assert!(
                storage
                    .search(SearchCriteria::ByEpisode(
                        "episode-under-delete".to_string()
                    ))
                    .unwrap()
                    .is_empty(),
                "{path}"
            );
            assert_eq!(
                storage.get_by_content_hash("indexed everywhere"),
                None,
                "{path}"
            );
            assert_eq!(
                storage.get_by_content_hash("rewritten content"),
                Some(original.id.clone()),
                "{path}"
            );
        }
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn find_by_external_id_answers_none_for_a_row_whose_memory_is_gone_and_removes_it() {
        let (storage, _dir) = test_storage();
        let parent = sample_memory(MemoryId(uuid::Uuid::new_v4()), "the parent");
        let memory = fully_indexed_memory(&parent.id, "github:pr-9");
        storage.store(&memory).unwrap();
        // A record removed without its index rows: what delete_with_vectors left behind
        // before it read the record first, and what any write bypassing the store leaves.
        storage.db.delete(memory.id.0.as_bytes()).unwrap();
        let dangling = |storage: &MemoryStorage| {
            index_rows_naming(storage, &memory.id)
                .into_iter()
                .filter(|key| key.starts_with("external:"))
                .count()
        };
        assert_eq!(dangling(&storage), 1);

        assert!(storage
            .find_by_external_id("github:pr-9")
            .unwrap()
            .is_none());
        assert_eq!(dangling(&storage), 0, "the dangling row was left in place");
        assert!(storage
            .find_by_external_id("github:pr-9")
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn find_by_external_id_reaches_the_live_memory_past_a_dangling_row() {
        let (storage, _dir) = test_storage();
        let parent = MemoryId(uuid::Uuid::new_v4());
        let gone = fully_indexed_memory(&parent, "github:issue-4");
        let live = fully_indexed_memory(&parent, "github:issue-4");
        storage.store(&gone).unwrap();
        storage.store(&live).unwrap();
        storage.db.delete(gone.id.0.as_bytes()).unwrap();

        let found = storage
            .find_by_external_id("github:issue-4")
            .unwrap()
            .expect("the live memory with this external id");
        assert_eq!(found.id, live.id);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn a_store_opened_with_explicit_tuning_reads_back_after_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let tuned = crate::config::StorageTuning {
            durability: crate::config::StorageDurability::Immediate,
            redb_cache_bytes: 1024 * 1024,
            rocksdb_write_buffer_bytes: 1024 * 1024,
            rocksdb_block_cache_bytes: Some(1024 * 1024),
        };
        let memory = sample_memory(MemoryId(uuid::Uuid::new_v4()), "survives the reopen");
        {
            let storage = MemoryStorage::new_with_tuning(dir.path(), None, tuned).unwrap();
            assert!(storage.sync_writes);
            storage.store_with_vectors(&memory, vec![1, 2]).unwrap();
        }
        let eventual = crate::config::StorageTuning {
            durability: crate::config::StorageDurability::Eventual,
            ..tuned
        };
        let reopened = MemoryStorage::new_with_tuning(dir.path(), None, eventual).unwrap();
        assert!(!reopened.sync_writes);
        assert_eq!(
            reopened.get(&memory.id).unwrap().experience.content,
            "survives the reopen"
        );
        assert!(reopened.get_vector_mapping(&memory.id).unwrap().is_some());
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn rebuild_index_keeps_the_rows_of_forgotten_memories() {
        let (storage, _dir) = test_storage();
        let parent = MemoryId(uuid::Uuid::new_v4());
        let memory = fully_indexed_memory(&parent, "linear:SHO-77");
        storage.store(&memory).unwrap();
        memory.set_importance(0.01);
        storage.update(&memory).unwrap();
        assert_eq!(
            storage.mark_forgotten_by_importance(0.5).unwrap(),
            vec![memory.id.clone()]
        );
        let before = {
            let mut rows = index_rows_naming(&storage, &memory.id);
            rows.sort();
            rows
        };

        storage.rebuild_index().unwrap();

        let mut after = index_rows_naming(&storage, &memory.id);
        after.sort();
        assert_eq!(after, before);
        assert!(storage.get_all().unwrap().is_empty());
        assert_eq!(
            storage
                .find_by_external_id("linear:SHO-77")
                .unwrap()
                .map(|m| m.id),
            Some(memory.id)
        );
    }
}
