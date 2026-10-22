//! Vamana index persistence for instant startup
//!
//! Binary file format for persisting Vamana graph to disk.
//! Uses mmap for zero-copy loading.
//!
//! # File Format (v1)
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (64 bytes)                       │
//! │ ├── magic: [u8; 4] = "VAMA"             │
//! │ ├── version: u32 = 1                    │
//! │ ├── num_vectors: u64                    │
//! │ ├── dimension: u32                      │
//! │ ├── max_degree: u32                     │
//! │ ├── medoid: u32                         │
//! │ ├── distance_metric: u8                 │
//! │ ├── deleted_count: u32                  │
//! │ ├── incremental_inserts: u64            │
//! │ ├── checksum: u64                       │
//! │ └── reserved: [u8; 15]                  │
//! ├─────────────────────────────────────────┤
//! │ Deleted IDs Section                     │
//! │ └── [u32; deleted_count]                │
//! ├─────────────────────────────────────────┤
//! │ Graph Section                           │
//! │ ├── For each node:                      │
//! │ │   ├── neighbor_count: u16             │
//! │ │   └── neighbors: [u32; neighbor_count]│
//! ├─────────────────────────────────────────┤
//! │ Vectors Section (aligned to 64 bytes)   │
//! │ └── [[f32; dimension]; num_vectors]     │
//! └─────────────────────────────────────────┘
//! ```

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::Path;
use tracing::info;

use super::vamana::{DistanceMetric, VamanaConfig, VamanaIndex, VamanaNode};
use super::vamana_store::{DeletedSet, VectorStore};

const MAGIC: [u8; 4] = *b"VAMA";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const ALIGNMENT: usize = 64;
const MAX_REASONABLE_DIMENSION: usize = 16_384;

/// Header for persisted Vamana index
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct VamanaHeader {
    magic: [u8; 4],
    version: u32,
    num_vectors: u64,
    dimension: u32,
    max_degree: u32,
    medoid: u32,
    distance_metric: u8,
    deleted_count: u32,
    incremental_inserts: u64,
    checksum: u64,
    reserved: [u8; 15],
}

impl VamanaHeader {
    fn new(
        num_vectors: usize,
        dimension: usize,
        max_degree: usize,
        medoid: u32,
        distance_metric: DistanceMetric,
        deleted_count: usize,
        incremental_inserts: usize,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            num_vectors: num_vectors as u64,
            dimension: dimension as u32,
            max_degree: max_degree as u32,
            medoid,
            distance_metric: match distance_metric {
                DistanceMetric::NormalizedDotProduct => 0,
                DistanceMetric::Euclidean => 1,
                DistanceMetric::Cosine => 2,
            },
            deleted_count: deleted_count as u32,
            incremental_inserts: incremental_inserts as u64,
            checksum: 0, // Computed after serialization
            reserved: [0u8; 15],
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.num_vectors.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.dimension.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.max_degree.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.medoid.to_le_bytes());
        bytes[28] = self.distance_metric;
        bytes[29..33].copy_from_slice(&self.deleted_count.to_le_bytes());
        bytes[33..41].copy_from_slice(&self.incremental_inserts.to_le_bytes());
        bytes[41..49].copy_from_slice(&self.checksum.to_le_bytes());
        // reserved bytes already 0
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(anyhow!("Header too small"));
        }

        let magic: [u8; 4] = bytes[0..4].try_into()?;
        if magic != MAGIC {
            return Err(anyhow!("Invalid magic bytes: {:?}", magic));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into()?);
        if version != VERSION {
            return Err(anyhow!("Unsupported version: {}", version));
        }

        Ok(Self {
            magic,
            version,
            num_vectors: u64::from_le_bytes(bytes[8..16].try_into()?),
            dimension: u32::from_le_bytes(bytes[16..20].try_into()?),
            max_degree: u32::from_le_bytes(bytes[20..24].try_into()?),
            medoid: u32::from_le_bytes(bytes[24..28].try_into()?),
            distance_metric: bytes[28],
            deleted_count: u32::from_le_bytes(bytes[29..33].try_into()?),
            incremental_inserts: u64::from_le_bytes(bytes[33..41].try_into()?),
            checksum: u64::from_le_bytes(bytes[41..49].try_into()?),
            reserved: [0u8; 15],
        })
    }

    fn distance_metric_enum(&self) -> DistanceMetric {
        match self.distance_metric {
            0 => DistanceMetric::NormalizedDotProduct,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::Cosine,
            _ => DistanceMetric::NormalizedDotProduct,
        }
    }
}

/// Compute checksum for data integrity
fn compute_checksum(data: &[u8]) -> u64 {
    // Simple xxhash-style checksum
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Sibling temp file a save writes before renaming it over `path`.
fn saving_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".saving");
    std::path::PathBuf::from(name)
}

/// Make a rename in `path`'s directory durable. Directories cannot be opened as files
/// on Windows, where the rename is durable on return.
fn sync_parent_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Replace the file at `path` with `bytes` as an index save does: written to a sibling temp
/// file that is synced, renamed over `path`, and the directory synced, so a crash leaves the
/// old file or the new one and never a part of either.
pub(crate) fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let tmp_path = saving_path(path);
    let written = (|| -> Result<()> {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })()
    .and_then(|()| {
        std::fs::rename(&tmp_path, path)?;
        sync_parent_dir(path)
    });
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    written
}

/// Align offset to boundary
fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

fn checked_add(a: usize, b: usize, context: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("Overflow while computing {context}"))
}

fn checked_mul(a: usize, b: usize, context: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("Overflow while computing {context}"))
}

fn ensure_available(file_len: usize, offset: usize, bytes: usize, context: &str) -> Result<()> {
    let end = checked_add(offset, bytes, context)?;
    if end > file_len {
        return Err(anyhow!(
            "Persisted Vamana index truncated while reading {context}: need {} bytes at offset {}, file len {}",
            bytes,
            offset,
            file_len
        ));
    }
    Ok(())
}

fn validate_layout(header: &VamanaHeader, file_len: usize) -> Result<(usize, usize, usize)> {
    if file_len < HEADER_SIZE {
        return Err(anyhow!(
            "Persisted Vamana index too small: {} bytes",
            file_len
        ));
    }

    let num_vectors: usize = header
        .num_vectors
        .try_into()
        .map_err(|_| anyhow!("num_vectors does not fit in usize"))?;
    let dimension: usize = header
        .dimension
        .try_into()
        .map_err(|_| anyhow!("dimension does not fit in usize"))?;
    let deleted_count: usize = header
        .deleted_count
        .try_into()
        .map_err(|_| anyhow!("deleted_count does not fit in usize"))?;

    if dimension == 0 || dimension > MAX_REASONABLE_DIMENSION {
        return Err(anyhow!("Invalid persisted Vamana dimension: {}", dimension));
    }

    if deleted_count > num_vectors {
        return Err(anyhow!(
            "Invalid persisted deleted_count {} for {} vectors",
            deleted_count,
            num_vectors
        ));
    }

    let medoid = header.medoid;
    if num_vectors > 0 && medoid as usize >= num_vectors {
        return Err(anyhow!(
            "Invalid persisted medoid {} for {} vectors",
            medoid,
            num_vectors
        ));
    }

    let deleted_section_size = checked_mul(deleted_count, 4, "deleted section size")?;
    let graph_min_section_size = checked_mul(num_vectors, 2, "minimum graph section size")?;
    let graph_offset = checked_add(HEADER_SIZE, deleted_section_size, "graph section offset")?;
    let vectors_offset = align_to(
        checked_add(
            graph_offset,
            graph_min_section_size,
            "minimum vectors offset",
        )?,
        ALIGNMENT,
    );
    let vector_section_size = checked_mul(
        checked_mul(num_vectors, dimension, "vector count × dimension")?,
        4,
        "vector bytes",
    )?;
    let minimum_file_size = checked_add(vectors_offset, vector_section_size, "minimum file size")?;

    if minimum_file_size > file_len {
        return Err(anyhow!(
            "Persisted Vamana layout exceeds file size: need at least {} bytes, file has {}",
            minimum_file_size,
            file_len
        ));
    }

    Ok((num_vectors, dimension, deleted_count))
}

impl VamanaIndex {
    /// Save index to file
    ///
    /// Serializes the entire index (graph + vectors) to a binary file.
    /// Can be loaded back with `load_from_file()`.
    ///
    /// The index is copied out under its locks and written without them: writing a
    /// large index takes tens of seconds (77.7 s measured), and every insert waits on
    /// these locks. The bytes go to a sibling temp file that is synced, then renamed
    /// over `path` and the directory synced, so a crash or SIGKILL mid-save leaves the
    /// previous file whole instead of a truncated one that forces a rebuild on open.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let start = std::time::Instant::now();

        let (vector_data, neighbor_counts, neighbors, deleted, medoid, num_vectors, inserts) = {
            let graph = self.graph.read();
            let vectors = self.vectors.read();
            let deleted_ids = self.deleted_ids.read();
            let vector_data: Vec<f32> = vectors.to_flat();
            let neighbor_counts: Vec<u16> = graph
                .iter()
                .map(|node| node.neighbors.len() as u16)
                .collect();
            let neighbors: Vec<u32> = graph
                .iter()
                .flat_map(|node| node.neighbors.iter().copied())
                .collect();
            let deleted: Vec<u32> = deleted_ids.iter().collect();
            (
                vector_data,
                neighbor_counts,
                neighbors,
                deleted,
                *self.medoid.read(),
                self.num_vectors.load(std::sync::atomic::Ordering::Acquire),
                self.incremental_inserts
                    .load(std::sync::atomic::Ordering::Acquire),
            )
        };

        let mut header = VamanaHeader::new(
            num_vectors,
            self.config.dimension,
            self.config.max_degree,
            medoid,
            self.config.distance_metric,
            deleted.len(),
            inserts,
        );

        let deleted_section_size = deleted.len() * 4;
        let graph_section_size = neighbor_counts.len() * 2 + neighbors.len() * 4;
        let vectors_offset = align_to(
            HEADER_SIZE + deleted_section_size + graph_section_size,
            ALIGNMENT,
        );
        let total_size = vectors_offset + vector_data.len() * 4;

        let tmp_path = saving_path(path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        let written = (|| -> Result<()> {
            file.set_len(total_size as u64)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };

            let mut offset = HEADER_SIZE;
            for &id in &deleted {
                mmap[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
                offset += 4;
            }
            let mut next = 0usize;
            for &count in &neighbor_counts {
                mmap[offset..offset + 2].copy_from_slice(&count.to_le_bytes());
                offset += 2;
                for &neighbor in &neighbors[next..next + count as usize] {
                    mmap[offset..offset + 4].copy_from_slice(&neighbor.to_le_bytes());
                    offset += 4;
                }
                next += count as usize;
            }
            offset = vectors_offset;
            for val in &vector_data {
                mmap[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                offset += 4;
            }

            header.checksum = compute_checksum(&mmap[HEADER_SIZE..]);
            mmap[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
            mmap.flush()?;
            drop(mmap);
            file.sync_all()?;
            Ok(())
        })();
        drop(file);
        if let Err(e) = written.and_then(|()| {
            std::fs::rename(&tmp_path, path)?;
            sync_parent_dir(path)
        }) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        info!(
            "Saved Vamana index: {} vectors, {} bytes in {:?}",
            num_vectors,
            total_size,
            start.elapsed()
        );

        Ok(())
    }

    /// Load index from file
    ///
    /// Uses mmap for zero-copy access to vectors.
    /// Returns immediately - vectors are demand-paged by OS.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let start = std::time::Instant::now();

        if !path.exists() {
            return Err(anyhow!("Index file not found: {:?}", path));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        ensure_available(mmap.len(), 0, HEADER_SIZE, "header")?;

        // Read header
        let header = VamanaHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        let (num_vectors, dimension, deleted_count) = validate_layout(&header, mmap.len())?;

        // Verify checksum
        let stored_checksum = header.checksum;
        let computed_checksum = compute_checksum(&mmap[HEADER_SIZE..]);
        if stored_checksum != computed_checksum {
            return Err(anyhow!(
                "Checksum mismatch: stored={}, computed={}",
                stored_checksum,
                computed_checksum
            ));
        }

        // Read deleted IDs
        let mut offset = HEADER_SIZE;
        let mut deleted_ids = DeletedSet::default();
        for _ in 0..deleted_count {
            ensure_available(mmap.len(), offset, 4, "deleted id")?;
            let id = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
            if id as usize >= num_vectors {
                return Err(anyhow!(
                    "Invalid persisted deleted id {} for {} vectors",
                    id,
                    num_vectors
                ));
            }
            deleted_ids.insert(id);
            offset += 4;
        }

        // Read graph
        let mut graph = Vec::with_capacity(num_vectors);
        for node_id in 0..num_vectors {
            ensure_available(mmap.len(), offset, 2, "graph neighbor count")?;
            let count = u16::from_le_bytes(mmap[offset..offset + 2].try_into()?) as usize;
            offset += 2;
            if count > num_vectors {
                return Err(anyhow!(
                    "Invalid persisted neighbor count {} for node {} (num_vectors={})",
                    count,
                    node_id,
                    num_vectors
                ));
            }
            ensure_available(
                mmap.len(),
                offset,
                checked_mul(count, 4, "graph neighbor bytes")?,
                "graph neighbors",
            )?;
            let mut neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                let neighbor = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                neighbors.push(neighbor);
                offset += 4;
            }
            graph.push(VamanaNode {
                id: node_id as u32,
                neighbors,
            });
        }

        // Calculate vectors offset (aligned)
        let vectors_offset = align_to(offset, ALIGNMENT);
        let vector_section_size = checked_mul(
            checked_mul(num_vectors, dimension, "vector count × dimension")?,
            4,
            "vector bytes",
        )?;
        ensure_available(
            mmap.len(),
            vectors_offset,
            vector_section_size,
            "vector section",
        )?;

        // Where the file's little-endian floats are the machine's own, the vector section is
        // read in place: loading copies nothing, and pages a search never reads stay on disk.
        // Saves replace the file by rename, which leaves a mapped inode intact on unix; a
        // Windows rename cannot replace a mapped file, so there the section is copied.
        let vectors = if cfg!(all(unix, target_endian = "little")) {
            VectorStore::mapped(dimension, mmap, vectors_offset, num_vectors)?
        } else {
            VectorStore::from_le_bytes(
                dimension,
                &mmap[vectors_offset..vectors_offset + vector_section_size],
                num_vectors,
            )?
        };

        let config = VamanaConfig {
            max_degree: header.max_degree as usize,
            search_list_size: 75, // Default, not persisted
            alpha: 1.2,           // Default, not persisted
            dimension,
            use_mmap: false, // A rebuild of a loaded index keeps its vectors on the heap
            distance_metric: header.distance_metric_enum(),
            quantized_traversal: false, // Callers opt in with set_quantized_traversal
        };

        let index = VamanaIndex {
            config,
            graph: std::sync::Arc::new(parking_lot::RwLock::new(graph)),
            vectors: std::sync::Arc::new(parking_lot::RwLock::new(vectors)),
            medoid: std::sync::Arc::new(parking_lot::RwLock::new(header.medoid)),
            num_vectors: std::sync::atomic::AtomicUsize::new(num_vectors),
            storage_path: Some(path.to_path_buf()),
            incremental_inserts: std::sync::atomic::AtomicUsize::new(
                header.incremental_inserts as usize,
            ),
            rebuilding: std::sync::atomic::AtomicBool::new(false),
            deleted_ids: std::sync::Arc::new(parking_lot::RwLock::new(deleted_ids)),
            visited: Default::default(),
            exact_search_max: super::vamana::EXACT_SEARCH_MAX_VECTORS,
        };

        info!(
            "Loaded Vamana index: {} vectors in {:?} (read in place: {})",
            num_vectors,
            start.elapsed(),
            index.vectors.read().is_mapped()
        );

        Ok(index)
    }

    /// Check if a persisted index exists and is valid
    pub fn index_file_exists(path: &Path) -> bool {
        if !path.exists() {
            return false;
        }

        // Try to read and validate header
        if let Ok(file) = File::open(path) {
            let mut header_bytes = [0u8; HEADER_SIZE];
            let mut reader = BufReader::new(file);
            if reader.read_exact(&mut header_bytes).is_ok() {
                return VamanaHeader::from_bytes(&header_bytes).is_ok();
            }
        }
        false
    }

    /// The content checksum in the header of the index file at `path`, read without mapping
    /// or verifying the rest. It identifies one saved file: a file that differs from it in
    /// any byte after the header carries another, so a file written beside an index can
    /// name the index it belongs to.
    pub fn saved_checksum(path: &Path) -> Result<u64> {
        let mut header = [0u8; HEADER_SIZE];
        File::open(path)?.read_exact(&mut header)?;
        Ok(VamanaHeader::from_bytes(&header)?.checksum)
    }

    /// Verify index file integrity without fully loading
    pub fn verify_index_file(path: &Path) -> Result<bool> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Ok(false);
        }

        let header = VamanaHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        if validate_layout(&header, mmap.len()).is_err() {
            return Ok(false);
        }
        let stored_checksum = header.checksum;
        let computed_checksum = compute_checksum(&mmap[HEADER_SIZE..]);

        Ok(stored_checksum == computed_checksum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("test.vamana");

        // Create index with some vectors (disable mmap for test)
        let config = VamanaConfig {
            dimension: 4,
            max_degree: 8,
            use_mmap: false,
            ..Default::default()
        };
        let mut index = VamanaIndex::new(config).unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        index.build(vectors.clone()).unwrap();

        // Save
        index.save_to_file(&index_path).unwrap();
        assert!(index_path.exists());

        // Verify
        assert!(VamanaIndex::verify_index_file(&index_path).unwrap());

        // Load
        let loaded = VamanaIndex::load_from_file(&index_path).unwrap();
        assert_eq!(loaded.len(), 5);

        // Search should work
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.search(&query, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Should find vector 0 first
    }

    /// A loaded index reads its vectors in place where the platform allows, searches exactly as
    /// the saved one did, and takes inserts without copying the mapped rows.
    #[test]
    fn a_loaded_index_searches_like_the_saved_one() {
        use rand::{Rng, SeedableRng};
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("loaded.vamana");
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let vectors: Vec<Vec<f32>> = (0..400)
            .map(|_| {
                let v: Vec<f32> = (0..12).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect();
        let mut saved = VamanaIndex::new(VamanaConfig {
            dimension: 12,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();
        saved.build(vectors.clone()).unwrap();
        saved.mark_deleted(7);
        saved.save_to_file(&path).unwrap();

        let mut loaded = VamanaIndex::load_from_file(&path).unwrap();
        assert_eq!(
            loaded.vectors.read().is_mapped(),
            cfg!(all(unix, target_endian = "little"))
        );
        for exact_max in [usize::MAX, 0] {
            saved.set_exact_search_max(exact_max);
            loaded.set_exact_search_max(exact_max);
            for q in vectors.iter().step_by(23) {
                assert_eq!(loaded.search(q, 8).unwrap(), saved.search(q, 8).unwrap());
            }
        }

        let id = loaded.add_vector(vectors[0].clone()).unwrap();
        assert_eq!(id, 400);
        assert_eq!(
            loaded.vectors.read().is_mapped(),
            cfg!(all(unix, target_endian = "little"))
        );
        assert_eq!(loaded.vector(id).unwrap(), vectors[0]);
        assert_eq!(loaded.vector(5).unwrap(), vectors[5]);
    }

    #[test]
    fn test_checksum_detects_corruption() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("corrupt.vamana");

        // Create and save index (disable mmap for test)
        let config = VamanaConfig {
            dimension: 4,
            use_mmap: false,
            ..Default::default()
        };
        let mut index = VamanaIndex::new(config).unwrap();
        index
            .build(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]])
            .unwrap();
        index.save_to_file(&index_path).unwrap();

        // Corrupt the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&index_path)
            .unwrap();
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        mmap[HEADER_SIZE + 10] ^= 0xFF; // Flip some bits
        mmap.flush().unwrap();

        // Verify should fail
        assert!(!VamanaIndex::verify_index_file(&index_path).unwrap());
    }

    #[test]
    fn test_header_serialization() {
        let header = VamanaHeader::new(
            1000,
            384,
            32,
            42,
            DistanceMetric::NormalizedDotProduct,
            5,
            100,
        );

        let bytes = header.to_bytes();
        let restored = VamanaHeader::from_bytes(&bytes).unwrap();

        // Copy values to avoid packed struct reference issues
        let num_vectors = restored.num_vectors;
        let dimension = restored.dimension;
        let max_degree = restored.max_degree;
        let medoid = restored.medoid;
        let deleted_count = restored.deleted_count;
        let incremental_inserts = restored.incremental_inserts;

        assert_eq!(num_vectors, 1000);
        assert_eq!(dimension, 384);
        assert_eq!(max_degree, 32);
        assert_eq!(medoid, 42);
        assert_eq!(deleted_count, 5);
        assert_eq!(incremental_inserts, 100);
    }

    #[test]
    fn test_rejects_invalid_layout_before_allocation() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("invalid-layout.vamana");

        let mut header = VamanaHeader::new(
            1_000_000,
            384,
            32,
            0,
            DistanceMetric::NormalizedDotProduct,
            0,
            0,
        );
        header.checksum = compute_checksum(&[]);
        std::fs::write(&index_path, header.to_bytes()).unwrap();

        assert!(!VamanaIndex::verify_index_file(&index_path).unwrap());
        assert!(VamanaIndex::load_from_file(&index_path).is_err());
    }
    #[test]
    fn a_save_replaces_the_previous_file_whole_and_leaves_no_temp() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("index.vamana");
        let config = VamanaConfig {
            dimension: 4,
            max_degree: 8,
            use_mmap: false,
            ..Default::default()
        };
        let axes = |n: usize| -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    let mut v = vec![0.1; 4];
                    v[i % 4] = 1.0 + i as f32;
                    v
                })
                .collect()
        };
        let mut first = VamanaIndex::new(config.clone()).unwrap();
        first.build(axes(3)).unwrap();
        first.save_to_file(&path).unwrap();

        // What a save killed before its rename leaves beside the live file.
        std::fs::write(saving_path(&path), b"half-written").unwrap();
        assert_eq!(VamanaIndex::load_from_file(&path).unwrap().len(), 3);

        let mut second = VamanaIndex::new(config).unwrap();
        second.build(axes(5)).unwrap();
        second.save_to_file(&path).unwrap();
        assert_eq!(VamanaIndex::load_from_file(&path).unwrap().len(), 5);
        assert!(!saving_path(&path).exists());
    }
}
