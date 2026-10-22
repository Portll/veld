//! Embedding loading from the SQLite slow store.
//!
//! Provides bulk and targeted embedding retrieval for gap topology scoring,
//! Voronoi analysis, and Planet X detection.

use anyhow::Result;
use std::collections::{BinaryHeap, HashMap};

use super::SlowStore;

/// One row held by [`SlowStore::sample_embeddings`]: its sample key, uuid,
/// name and still-sealed embedding. Ordered by key, then uuid.
type SampledRow = (u64, String, String, Vec<u8>);

impl SlowStore {
    /// Load embeddings for a set of entity UUIDs.
    ///
    /// Returns a map from UUID string to embedding vector.
    /// Used by gap topology scoring to compute cosine similarity between gap endpoints.
    pub fn load_embeddings(&self, uuids: &[&str]) -> Result<HashMap<String, Vec<f32>>> {
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.read()?;
        let placeholders = vec!["?"; uuids.len()].join(",");
        let sql = format!(
            "SELECT uuid, embedding FROM entities \
             WHERE embedding IS NOT NULL AND uuid IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql)?;

        let mut result = HashMap::new();
        let rows = stmt.query_map(rusqlite::params_from_iter(uuids.iter().copied()), |row| {
            let uuid_str: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((uuid_str, blob))
        })?;
        for row in rows {
            let (uuid_str, blob) = row?;
            // Decrypt the embedding BLOB if the embeddings toggle sealed it
            // (AAD-bound to the entity uuid).
            let blob = super::open_field(self.encryptor.as_ref(), blob, uuid_str.as_bytes())?;
            let floats = decode_f32_le(&blob);
            if !floats.is_empty() {
                result.insert(uuid_str, floats);
            }
        }

        Ok(result)
    }

    /// Load ALL entity embeddings from the slow store.
    ///
    /// Returns (uuid, name, embedding) triples for all entities that have embeddings.
    /// Used by Voronoi analysis to reason about the shape of knowledge space.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, String, Vec<f32>)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare_cached(
            "SELECT uuid, name, embedding FROM entities WHERE embedding IS NOT NULL",
        )?;

        // `name` stays plaintext (gap queries read it); the embedding BLOB may be
        // sealed. Collect first so the conn/stmt borrow ends before we decrypt.
        let raw: Vec<(String, String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })?
            .filter_map(|r| match r {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("slow_store load_all_embeddings: skipping unreadable row: {e}");
                    None
                }
            })
            .collect();

        let mut results = Vec::with_capacity(raw.len());
        for (uuid, name, blob) in raw {
            // Decrypt the embedding column if the embeddings toggle sealed it
            // (AAD-bound to the entity uuid).
            let blob = super::open_field(self.encryptor.as_ref(), blob, uuid.as_bytes())?;
            let floats = decode_f32_le(&blob);
            if !floats.is_empty() {
                results.push((uuid, name, floats));
            }
        }

        Ok(results)
    }

    /// Up to `max` entity embeddings as `(uuid, name, embedding)`, in uuid order:
    /// every entity with an embedding when there are no more than `max`, else a
    /// uniform sample of `max` of them.
    ///
    /// The sample is a one-pass reservoir that keeps the `max` rows with the
    /// lowest [`sample_key`] of their uuid under `seed`. The key comes from the
    /// uuid rather than from a random stream drawn as rows arrive, because rows
    /// arrive in the order the last mirror rebuild inserted them: a stream would
    /// draw a different sample of an unchanged graph after every rebuild, where
    /// this draws the same one, and an entity leaves it only when one with a
    /// lower key arrives. Rows outside the sample are never copied, decrypted or
    /// decoded.
    pub fn sample_embeddings(
        &self,
        max: usize,
        seed: u64,
    ) -> Result<Vec<(String, String, Vec<f32>)>> {
        if max == 0 {
            return Ok(Vec::new());
        }
        let mut kept: BinaryHeap<SampledRow> = BinaryHeap::new();
        {
            let conn = self.read()?;
            let mut stmt = conn.prepare_cached(
                "SELECT uuid, name, embedding FROM entities WHERE embedding IS NOT NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let uuid = match row.get_ref(0).and_then(|v| Ok(v.as_str()?)) {
                    Ok(uuid) => uuid,
                    Err(e) => {
                        tracing::warn!(
                            "slow_store sample_embeddings: skipping unreadable row: {e}"
                        );
                        continue;
                    }
                };
                let key = sample_key(uuid, seed);
                if kept.len() >= max
                    && kept
                        .peek()
                        .is_some_and(|top| (key, uuid) >= (top.0, top.1.as_str()))
                {
                    continue;
                }
                let (name, blob) = match (row.get::<_, String>(1), row.get::<_, Vec<u8>>(2)) {
                    (Ok(name), Ok(blob)) => (name, blob),
                    (Err(e), _) | (_, Err(e)) => {
                        tracing::warn!(
                            "slow_store sample_embeddings: skipping unreadable row: {e}"
                        );
                        continue;
                    }
                };
                kept.push((key, uuid.to_string(), name, blob));
                if kept.len() > max {
                    kept.pop();
                }
            }
        }

        let mut results = Vec::with_capacity(kept.len());
        for (_, uuid, name, blob) in kept {
            // Decrypt the embedding column if the embeddings toggle sealed it
            // (AAD-bound to the entity uuid).
            let blob = super::open_field(self.encryptor.as_ref(), blob, uuid.as_bytes())?;
            let floats = decode_f32_le(&blob);
            if !floats.is_empty() {
                results.push((uuid, name, floats));
            }
        }
        results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }

    /// Load all directed edges (from_uuid, to_uuid) from the slow store.
    ///
    /// Used by Planet X detection to find where edges converge.
    pub fn load_all_edge_pairs(&self) -> Result<Vec<(String, String)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare_cached("SELECT from_entity, to_entity FROM edges")?;

        let results = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| match r {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("slow_store load_all_edge_pairs: skipping unreadable row: {e}");
                    None
                }
            })
            .collect();

        Ok(results)
    }
}

/// Little-endian `f32`s from an embedding BLOB; a trailing partial value is dropped.
fn decode_f32_le(blob: &[u8]) -> Vec<f32> {
    let (values, _) = blob.as_chunks::<4>();
    values.iter().copied().map(f32::from_le_bytes).collect()
}

/// A uniformly spread 64-bit key for `uuid` under `seed`: FNV-1a over its bytes,
/// then the SplitMix64 finaliser, which spreads FNV's weak low bits across the
/// word. Spelled out rather than taken from `std`, whose hashers promise no
/// stable output across releases, so a sample stays the same across builds.
pub(super) fn sample_key(uuid: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &byte in uuid.as_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut z = hash.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
