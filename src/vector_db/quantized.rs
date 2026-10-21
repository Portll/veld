//! A flat index of product-quantized codes, for an embedding space that is not the primary.
//!
//! Each vector is held as one byte per subspace: a 384-dimensional vector in 48 bytes instead of
//! 1536. A search scores every live code against the query by asymmetric distance, so there is
//! no graph to rebuild as vectors come and go; a deleted slot is skipped until compaction drops
//! it. Until enough vectors exist to train the quantizer on, they are held exact and searched
//! exactly. Scores from codes are estimates: the caller reranks the best candidates against the
//! exact vectors the memory records keep.
//!
//! Vectors are normalized on the way in, so a code's squared distance `d` to a normalized query
//! estimates the cosine as `1 - d / 2`.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::distance_inline::{dot_product_inline, normalize_inplace};
use super::pq::{PQConfig, ProductQuantizer, NUM_CENTROIDS};

/// Vectors held exact before the quantizer is trained on them.
pub const TRAIN_AT: usize = 4096;
/// The most vectors a quantizer is trained on; the rest are only encoded.
pub const TRAIN_SAMPLE: usize = 4096;
/// A quantizer trained when the index held `n` vectors is due a retrain at `n` times this.
pub const RETRAIN_GROWTH: usize = 8;
/// Share of slots deleted at which compaction is due.
pub const DELETION_RATIO_THRESHOLD: f32 = 0.30;

const FILE_MAGIC: &[u8; 8] = b"VELDPQ01";
const KMEANS_ITERATIONS: usize = 20;

/// The quantizer layout for `dimension`: subvectors of 8 floats, or 4 or 2 when 8 does not
/// divide it. `None` when none does; such a space stays exact.
pub fn config_for(dimension: usize) -> Option<PQConfig> {
    let subvec_dim = [8, 4, 2]
        .into_iter()
        .find(|s| dimension >= *s && dimension.is_multiple_of(*s))?;
    Some(PQConfig {
        dimension,
        num_subvectors: dimension / subvec_dim,
        subvec_dim,
        num_centroids: NUM_CENTROIDS,
        kmeans_iterations: KMEANS_ITERATIONS,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedIndex {
    dimension: usize,
    quantizer: Option<ProductQuantizer>,
    /// Normalized vectors by slot, until the quantizer is trained; empty after.
    exact: Vec<Vec<f32>>,
    /// One code per subspace per slot, slot-major, once the quantizer is trained.
    codes: Vec<u8>,
    slots: usize,
    deleted: HashSet<u32>,
    /// Live vectors when the quantizer was trained.
    trained_on: usize,
    /// Vectors added since the index was last built or compacted.
    inserted: usize,
}

fn normalized(mut vector: Vec<f32>) -> Vec<f32> {
    normalize_inplace(&mut vector);
    vector
}

/// Every `len / take`-th index, `take` of them at most.
fn stride_sample(len: usize, take: usize) -> impl Iterator<Item = usize> {
    let step = len.div_ceil(take.max(1)).max(1);
    (0..len).step_by(step)
}

impl QuantizedIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            quantizer: None,
            exact: Vec::new(),
            codes: Vec::new(),
            slots: 0,
            deleted: HashSet::new(),
            trained_on: 0,
            inserted: 0,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Slots, deleted ones included: the next vector's id.
    pub fn len(&self) -> usize {
        self.slots
    }

    pub fn is_empty(&self) -> bool {
        self.slots == 0
    }

    pub fn live(&self) -> usize {
        self.slots - self.deleted.len()
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }

    pub fn is_deleted(&self, id: u32) -> bool {
        self.deleted.contains(&id)
    }

    pub fn deletion_ratio(&self) -> f32 {
        if self.slots == 0 {
            0.0
        } else {
            self.deleted.len() as f32 / self.slots as f32
        }
    }

    pub fn needs_compaction(&self) -> bool {
        self.deletion_ratio() >= DELETION_RATIO_THRESHOLD
    }

    pub fn is_quantized(&self) -> bool {
        self.quantizer.is_some()
    }

    /// Whether the index has grown past what its quantizer was trained on, so a build from the
    /// stored vectors would fit it better.
    pub fn needs_retrain(&self) -> bool {
        self.is_quantized() && self.live() >= self.trained_on.saturating_mul(RETRAIN_GROWTH)
    }

    pub fn inserted_since_build(&self) -> usize {
        self.inserted
    }

    /// Bytes held: codes, exact vectors and codebooks.
    pub fn memory_bytes(&self) -> usize {
        let codebooks = self.quantizer.as_ref().map_or(0, |q| {
            q.centroids.len() * q.config.num_centroids * q.config.subvec_dim * 4
        });
        self.codes.len() + self.exact.len() * self.dimension * 4 + codebooks
    }

    fn check(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            bail!(
                "a {}-dimensional vector for a {}-dimensional index",
                vector.len(),
                self.dimension
            );
        }
        Ok(())
    }

    /// Replace the contents with `vectors`, slot by slot, trained and encoded when there are
    /// enough of them.
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        for v in &vectors {
            self.check(v)?;
        }
        let exact: Vec<Vec<f32>> = vectors.into_par_iter().map(normalized).collect();
        let mut fresh = Self::new(self.dimension);
        fresh.slots = exact.len();
        fresh.exact = exact;
        if fresh.slots >= TRAIN_AT {
            if let Some(sample) = fresh.training_sample() {
                let quantizer = Self::train(self.dimension, &sample)?;
                fresh.install_quantizer(quantizer)?;
            }
        }
        *self = fresh;
        Ok(())
    }

    /// Add `vector` in the next slot and return its id.
    pub fn add(&mut self, vector: Vec<f32>) -> Result<u32> {
        self.check(&vector)?;
        let id = u32::try_from(self.slots).map_err(|_| anyhow!("index is full"))?;
        let vector = normalized(vector);
        match &self.quantizer {
            Some(q) => self.codes.extend(q.encode(&vector)?),
            None => self.exact.push(vector),
        }
        self.slots += 1;
        self.inserted += 1;
        Ok(id)
    }

    pub fn mark_deleted(&mut self, id: u32) {
        if (id as usize) < self.slots {
            self.deleted.insert(id);
        }
    }

    /// Whether the index holds enough exact vectors to be trained and is not yet.
    pub fn wants_quantizer(&self) -> bool {
        !self.is_quantized() && self.live() >= TRAIN_AT && config_for(self.dimension).is_some()
    }

    /// Up to `TRAIN_SAMPLE` live exact vectors, spread over the slots; `None` once quantized or
    /// when the dimension cannot be split.
    pub fn training_sample(&self) -> Option<Vec<Vec<f32>>> {
        if self.is_quantized() || config_for(self.dimension).is_none() {
            return None;
        }
        let live: Vec<usize> = (0..self.slots)
            .filter(|i| !self.deleted.contains(&(*i as u32)))
            .collect();
        if live.is_empty() {
            return None;
        }
        Some(
            stride_sample(live.len(), TRAIN_SAMPLE)
                .map(|i| self.exact[live[i]].clone())
                .collect(),
        )
    }

    /// Fit a quantizer for `dimension` on `sample`, its subspaces in parallel.
    pub fn train(dimension: usize, sample: &[Vec<f32>]) -> Result<ProductQuantizer> {
        let config = config_for(dimension)
            .with_context(|| format!("{dimension} dimensions do not split into subvectors"))?;
        ProductQuantizer::train(config, sample)
    }

    /// Encode every held exact vector with `quantizer` and drop them: from here the index holds
    /// codes only.
    pub fn install_quantizer(&mut self, quantizer: ProductQuantizer) -> Result<()> {
        if quantizer.config.dimension != self.dimension {
            bail!(
                "a quantizer for {} dimensions on a {}-dimensional index",
                quantizer.config.dimension,
                self.dimension
            );
        }
        let encoded: Vec<Vec<u8>> = self
            .exact
            .par_iter()
            .map(|v| quantizer.encode(v))
            .collect::<Result<_>>()?;
        self.codes = encoded.into_iter().flatten().collect();
        self.exact = Vec::new();
        self.trained_on = self.live();
        self.quantizer = Some(quantizer);
        Ok(())
    }

    /// The `k` live slots nearest `query`, best first, ties broken by id, each with its cosine:
    /// exact while the index is unquantized, estimated from the codes after.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        self.check(query)?;
        let query = normalized(query.to_vec());
        let live = |i: &usize| !self.deleted.contains(&(*i as u32));
        let mut scored: Vec<(u32, f32)> = match &self.quantizer {
            None => (0..self.slots)
                .filter(live)
                .map(|i| (i as u32, dot_product_inline(&query, &self.exact[i])))
                .collect(),
            Some(q) => {
                let table = q.build_distance_table(&query)?;
                let m = q.config.num_subvectors;
                (0..self.slots)
                    .filter(live)
                    .map(|i| {
                        let d = q.distance_with_table(&table, &self.codes[i * m..(i + 1) * m]);
                        (i as u32, 1.0 - d / 2.0)
                    })
                    .collect()
            }
        };
        let order =
            |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if scored.len() > k && k > 0 {
            scored.select_nth_unstable_by(k - 1, order);
            scored.truncate(k);
        }
        scored.sort_by(order);
        scored.truncate(k);
        Ok(scored)
    }

    /// Drop deleted slots, keeping the rest in order. Returns each old slot's new id, `None`
    /// for a dropped one.
    pub fn compact(&mut self) -> Vec<Option<u32>> {
        let mut remap = vec![None; self.slots];
        let mut next = 0u32;
        let m = self.quantizer.as_ref().map(|q| q.config.num_subvectors);
        let mut codes = Vec::with_capacity(m.map_or(0, |m| self.live() * m));
        let mut exact = Vec::with_capacity(if m.is_some() { 0 } else { self.live() });
        for (i, slot) in remap.iter_mut().enumerate() {
            if self.deleted.contains(&(i as u32)) {
                continue;
            }
            match m {
                Some(m) => codes.extend_from_slice(&self.codes[i * m..(i + 1) * m]),
                None => exact.push(std::mem::take(&mut self.exact[i])),
            }
            *slot = Some(next);
            next += 1;
        }
        self.codes = codes;
        self.exact = exact;
        self.slots = next as usize;
        self.deleted.clear();
        self.inserted = 0;
        remap
    }

    /// Write the index with a checksum, through a temporary file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        let body = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .context("encoding the quantized index")?;
        let mut out = Vec::with_capacity(body.len() + 40);
        out.extend_from_slice(FILE_MAGIC);
        out.extend_from_slice(&Sha256::digest(&body));
        out.extend_from_slice(&body);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
        Ok(())
    }

    /// Read an index `save_to_file` wrote; an error when the file is not one, or its checksum
    /// fails.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() < 40 || &bytes[..8] != FILE_MAGIC {
            bail!("{} is not a quantized index", path.display());
        }
        let (digest, body) = bytes[8..].split_at(32);
        if Sha256::digest(body).as_slice() != digest {
            bail!("{} fails its checksum", path.display());
        }
        let (index, _): (Self, usize) =
            bincode::serde::decode_from_slice(body, bincode::config::standard())
                .context("decoding the quantized index")?;
        let m = index.quantizer.as_ref().map(|q| q.config.num_subvectors);
        let consistent = match m {
            Some(m) => index.codes.len() == index.slots * m && index.exact.is_empty(),
            None => index.exact.len() == index.slots && index.codes.is_empty(),
        };
        if !consistent || index.deleted.iter().any(|&id| id as usize >= index.slots) {
            bail!("{} describes an inconsistent index", path.display());
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % centres.len()];
                normalized(c.iter().map(|x| x + rng.gen_range(-0.3..0.3)).collect())
            })
            .collect()
    }

    fn exact_top(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot_product_inline(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn a_dimension_splits_into_the_widest_subvector_that_divides_it() {
        assert_eq!(config_for(384).unwrap().subvec_dim, 8);
        assert_eq!(config_for(12).unwrap().subvec_dim, 4);
        assert_eq!(config_for(6).unwrap().subvec_dim, 2);
        assert!(config_for(7).is_none());
    }

    #[test]
    fn a_small_index_is_exact() {
        let corpus = vectors(200, 16, 1);
        let mut index = QuantizedIndex::new(16);
        for v in &corpus {
            index.add(v.clone()).unwrap();
        }
        assert!(!index.is_quantized());
        let hits = index.search(&corpus[17], 5).unwrap();
        assert_eq!(hits[0].0, 17);
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(
            hits.iter().map(|h| h.0).collect::<Vec<_>>(),
            exact_top(&corpus, &corpus[17], 5)
        );
    }

    #[test]
    fn an_index_trains_once_it_holds_enough_and_its_codes_find_the_neighbours() {
        let corpus = vectors(TRAIN_AT + 500, 32, 2);
        let mut index = QuantizedIndex::new(32);
        for v in &corpus[..TRAIN_AT] {
            index.add(v.clone()).unwrap();
        }
        assert!(index.wants_quantizer());
        let sample = index.training_sample().unwrap();
        assert_eq!(sample.len(), TRAIN_SAMPLE.min(TRAIN_AT));
        let q = QuantizedIndex::train(32, &sample).unwrap();
        index.install_quantizer(q).unwrap();
        for v in &corpus[TRAIN_AT..] {
            index.add(v.clone()).unwrap();
        }
        assert!(index.is_quantized());
        assert!(!index.needs_retrain());
        assert!(index.memory_bytes() < corpus.len() * 32 * 4 / 2);

        let mut found = 0;
        for probe in (0..corpus.len()).step_by(97) {
            let truth: HashSet<u32> = exact_top(&corpus, &corpus[probe], 10).into_iter().collect();
            let got = index.search(&corpus[probe], 40).unwrap();
            found += got.iter().filter(|(i, _)| truth.contains(i)).count();
        }
        let probes = corpus.len().div_ceil(97);
        let recall = found as f32 / (probes * 10) as f32;
        assert!(recall > 0.8, "recall@10 within 40 candidates: {recall}");
    }

    #[test]
    fn deleted_slots_are_skipped_and_compaction_renumbers_the_rest() {
        let corpus = vectors(50, 8, 3);
        let mut index = QuantizedIndex::new(8);
        index.build(corpus.clone()).unwrap();
        index.mark_deleted(4);
        assert!(index
            .search(&corpus[4], 50)
            .unwrap()
            .iter()
            .all(|h| h.0 != 4));
        let remap = index.compact();
        assert_eq!(remap[4], None);
        assert_eq!(remap[5], Some(4));
        assert_eq!(index.len(), 49);
        assert_eq!(index.search(&corpus[5], 1).unwrap()[0].0, 4);
    }

    #[test]
    fn a_saved_index_loads_and_a_damaged_one_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq.idx");
        let corpus = vectors(TRAIN_AT, 16, 4);
        let mut index = QuantizedIndex::new(16);
        index.build(corpus.clone()).unwrap();
        assert!(index.is_quantized());
        index.mark_deleted(9);
        index.save_to_file(&path).unwrap();
        let loaded = QuantizedIndex::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), TRAIN_AT);
        assert!(loaded.is_deleted(9));
        assert_eq!(
            loaded.search(&corpus[3], 5).unwrap(),
            index.search(&corpus[3], 5).unwrap()
        );

        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        assert!(QuantizedIndex::load_from_file(&path).is_err());
    }

    #[test]
    fn a_vector_of_another_width_is_refused() {
        let mut index = QuantizedIndex::new(8);
        assert!(index.add(vec![1.0; 9]).is_err());
        assert!(index.search(&[1.0; 4], 1).is_err());
    }
}
