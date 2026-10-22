//! Vector rows of a [`super::vamana::VamanaIndex`], and the id sets its searches use.

use anyhow::{anyhow, Result};
use memmap2::Mmap;

/// Rows read in place from a mapped file: the vector file a build wrote, or the vector section
/// of a saved index.
struct MappedRows {
    map: Mmap,
    /// Byte offset of row 0 within `map`.
    offset: usize,
    rows: usize,
}

/// Every vector of an index, row-major and contiguous. Rows `0..mapped.rows` are read from a
/// mapped file and never copied; rows appended after that, and every row of an index with
/// nothing mapped, live in one growable heap buffer.
pub(crate) struct VectorStore {
    dimension: usize,
    mapped: Option<MappedRows>,
    tail: Vec<f32>,
    tail_rows: usize,
    /// int8 code of every row, row-major, for graph walks; `None` unless enabled.
    codes: Option<Vec<i8>>,
}

impl VectorStore {
    pub(crate) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            mapped: None,
            tail: Vec::new(),
            tail_rows: 0,
            codes: None,
        }
    }

    /// `rows` copied into one buffer. Every row must be `dimension` long: rows are addressed by
    /// stride, so one short row would shift every row after it.
    pub(crate) fn from_rows(dimension: usize, rows: Vec<Vec<f32>>) -> Result<Self> {
        let mut store = Self::new(dimension);
        store.tail.reserve_exact(rows.len() * dimension);
        for (id, row) in rows.into_iter().enumerate() {
            if row.len() != dimension {
                return Err(anyhow!(
                    "vector {id} has dimension {}, the index has {dimension}",
                    row.len()
                ));
            }
            store.push(&row);
        }
        Ok(store)
    }

    /// `rows` vectors of native-endian f32 read in place from `map`, starting `offset` bytes in.
    pub(crate) fn mapped(dimension: usize, map: Mmap, offset: usize, rows: usize) -> Result<Self> {
        let bytes = rows
            .checked_mul(dimension)
            .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow!("{rows} vectors of dimension {dimension} overflow"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("vector section end overflows"))?;
        if end > map.len() {
            return Err(anyhow!(
                "vector section [{offset}..{end}] exceeds the mapped {} bytes",
                map.len()
            ));
        }
        let start = map.as_ptr() as usize + offset;
        if start % std::mem::align_of::<f32>() != 0 {
            return Err(anyhow!(
                "vector section at offset {offset} is not aligned for f32"
            ));
        }
        Ok(Self {
            dimension,
            mapped: Some(MappedRows { map, offset, rows }),
            tail: Vec::new(),
            tail_rows: 0,
            codes: None,
        })
    }

    /// `rows` vectors of little-endian f32 copied out of `bytes`.
    pub(crate) fn from_le_bytes(dimension: usize, bytes: &[u8], rows: usize) -> Result<Self> {
        let floats = rows
            .checked_mul(dimension)
            .ok_or_else(|| anyhow!("{rows} vectors of dimension {dimension} overflow"))?;
        if bytes.len() != floats * std::mem::size_of::<f32>() {
            return Err(anyhow!(
                "{} bytes do not hold {rows} vectors of dimension {dimension}",
                bytes.len()
            ));
        }
        let mut store = Self::new(dimension);
        store.tail = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        store.tail_rows = rows;
        Ok(store)
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn len(&self) -> usize {
        self.mapped_rows() + self.tail_rows
    }

    /// Whether any rows are read from a mapped file.
    pub(crate) fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    fn mapped_rows(&self) -> usize {
        self.mapped.as_ref().map_or(0, |m| m.rows)
    }

    fn mapped_floats(&self) -> &[f32] {
        match &self.mapped {
            // SAFETY: `mapped()` checked that `rows × dimension` floats starting `offset` bytes
            // into the map lie inside it and that their start is f32-aligned. The map is
            // read-only and lives as long as `self`, which the returned slice borrows.
            Some(m) => unsafe {
                std::slice::from_raw_parts(
                    m.map.as_ptr().add(m.offset) as *const f32,
                    m.rows * self.dimension,
                )
            },
            None => &[],
        }
    }

    /// The vector stored at `id`.
    #[inline]
    pub(crate) fn row(&self, id: u32) -> Option<&[f32]> {
        let id = id as usize;
        let dim = self.dimension;
        let mapped = self.mapped_rows();
        if id < mapped {
            let start = id * dim;
            self.mapped_floats().get(start..start + dim)
        } else if id - mapped < self.tail_rows {
            let start = (id - mapped) * dim;
            self.tail.get(start..start + dim)
        } else {
            None
        }
    }

    /// Append `row`, which the caller has checked is `dimension` long. Mapped rows stay mapped.
    pub(crate) fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dimension);
        self.tail.extend_from_slice(row);
        self.tail_rows += 1;
        if let Some(codes) = self.codes.as_mut() {
            codes.extend(row.iter().map(|&x| quantize_component(x)));
        }
    }

    /// Keep, or drop, an int8 code of every row for graph walks. Kept codes follow every push.
    pub(crate) fn set_codes(&mut self, on: bool) {
        if !on {
            self.codes = None;
            return;
        }
        if self.codes.is_some() {
            return;
        }
        let mut codes = Vec::with_capacity(self.len() * self.dimension);
        for id in 0..self.len() as u32 {
            if let Some(row) = self.row(id) {
                codes.extend(row.iter().map(|&x| quantize_component(x)));
            }
        }
        self.codes = Some(codes);
    }

    pub(crate) fn has_codes(&self) -> bool {
        self.codes.is_some()
    }

    /// The int8 code of row `id`, when codes are kept.
    #[inline]
    pub(crate) fn code_row(&self, id: u32) -> Option<&[i8]> {
        let start = id as usize * self.dimension;
        self.codes.as_ref()?.get(start..start + self.dimension)
    }

    /// The first `n` rows as contiguous runs, each with the id of its first row: the mapped
    /// rows, then the heap rows.
    pub(crate) fn runs(&self, n: usize) -> impl Iterator<Item = (u32, &[f32])> {
        let dim = self.dimension;
        let mapped = self.mapped_rows().min(n);
        let tail = self.tail_rows.min(n - mapped);
        [
            (0u32, &self.mapped_floats()[..mapped * dim]),
            (mapped as u32, &self.tail[..tail * dim]),
        ]
        .into_iter()
        .filter(|(_, run)| !run.is_empty())
    }

    /// Every row, copied.
    pub(crate) fn to_rows(&self) -> Vec<Vec<f32>> {
        (0..self.len() as u32)
            .filter_map(|id| self.row(id).map(<[f32]>::to_vec))
            .collect()
    }

    /// Every row, copied into one buffer.
    pub(crate) fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.len() * self.dimension);
        flat.extend_from_slice(self.mapped_floats());
        flat.extend_from_slice(&self.tail[..self.tail_rows * self.dimension]);
        flat
    }
}

/// The int8 code of a unit vector's component: times 127, rounded, so [-1, 1] maps onto
/// [-127, 127] and anything outside saturates.
#[inline]
pub(crate) fn quantize_component(x: f32) -> i8 {
    (x * 127.0).round().clamp(-127.0, 127.0) as i8
}

/// A query's int8 code, scaled so its largest component maps to ±127. Ranking rows by dot
/// product does not depend on the query's scale, so the whole int8 range goes to resolution.
pub(crate) fn quantize_query(query: &[f32]) -> Vec<i8> {
    let largest = query.iter().fold(0f32, |m, x| m.max(x.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return vec![0; query.len()];
    }
    let scale = 127.0 / largest;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Soft-deleted vector ids, one bit per id below the highest marked.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DeletedSet {
    words: Vec<u64>,
    count: usize,
}

impl DeletedSet {
    #[inline]
    pub(crate) fn contains(&self, id: u32) -> bool {
        let id = id as usize;
        self.words
            .get(id / 64)
            .is_some_and(|word| word & (1 << (id % 64)) != 0)
    }

    /// Mark `id`; false when it already was.
    pub(crate) fn insert(&mut self, id: u32) -> bool {
        let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        let fresh = self.words[word] & bit == 0;
        self.words[word] |= bit;
        self.count += usize::from(fresh);
        fresh
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn clear(&mut self) {
        self.words.clear();
        self.count = 0;
    }

    /// Marked ids, ascending.
    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(index, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                Some(index as u32 * 64 + bit)
            })
        })
    }
}

impl FromIterator<u32> for DeletedSet {
    fn from_iter<I: IntoIterator<Item = u32>>(ids: I) -> Self {
        let mut set = Self::default();
        for id in ids {
            set.insert(id);
        }
        set
    }
}

/// The nodes one graph walk has reached. A node is reached when its stamp equals the walk's
/// epoch, so a new walk starts with one increment instead of clearing or allocating a set.
pub(crate) struct VisitedSet {
    stamps: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    fn new() -> Self {
        Self {
            stamps: Vec::new(),
            epoch: 0,
        }
    }

    /// Forget every node, with room for ids below `n`.
    fn reset(&mut self, n: usize) {
        if self.stamps.len() < n {
            self.stamps.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }

    /// Mark `id` reached; false when it already was.
    #[inline]
    pub(crate) fn insert(&mut self, id: u32) -> bool {
        let id = id as usize;
        if id >= self.stamps.len() {
            self.stamps.resize(id + 1, 0);
        }
        let stamp = &mut self.stamps[id];
        if *stamp == self.epoch {
            false
        } else {
            *stamp = self.epoch;
            true
        }
    }
}

/// Visited sets kept between walks. A set holds four bytes per node, so the pool keeps no more
/// than this many; a walk beyond them allocates its own and drops it.
const POOLED_VISITED_SETS: usize = 32;

#[derive(Default)]
pub(crate) struct VisitedPool {
    free: parking_lot::Mutex<Vec<VisitedSet>>,
}

impl VisitedPool {
    /// An empty visited set with room for ids below `n`, returned to the pool when dropped.
    pub(crate) fn take(&self, n: usize) -> PooledVisited<'_> {
        let mut set = self.free.lock().pop().unwrap_or_else(VisitedSet::new);
        set.reset(n);
        PooledVisited {
            pool: self,
            set: Some(set),
        }
    }
}

pub(crate) struct PooledVisited<'a> {
    pool: &'a VisitedPool,
    set: Option<VisitedSet>,
}

impl std::ops::Deref for PooledVisited<'_> {
    type Target = VisitedSet;
    fn deref(&self) -> &VisitedSet {
        self.set.as_ref().expect("visited set is held until drop")
    }
}

impl std::ops::DerefMut for PooledVisited<'_> {
    fn deref_mut(&mut self) -> &mut VisitedSet {
        self.set.as_mut().expect("visited set is held until drop")
    }
}

impl Drop for PooledVisited<'_> {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            let mut free = self.pool.free.lock();
            if free.len() < POOLED_VISITED_SETS {
                free.push(set);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn rows(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32).collect())
            .collect()
    }

    #[test]
    fn rows_read_back_and_runs_cover_them_in_order() {
        let source = rows(5, 3);
        let mut store = VectorStore::from_rows(3, source.clone()).unwrap();
        store.push(&[100.0, 101.0, 102.0]);
        assert_eq!(store.len(), 6);
        for (id, row) in source.iter().enumerate() {
            assert_eq!(store.row(id as u32).unwrap(), row.as_slice());
        }
        assert_eq!(store.row(5).unwrap(), &[100.0, 101.0, 102.0]);
        assert!(store.row(6).is_none());
        let runs: Vec<(u32, usize)> = store.runs(4).map(|(id, r)| (id, r.len())).collect();
        assert_eq!(runs, vec![(0, 12)]);
    }

    #[test]
    fn a_short_row_is_refused() {
        assert!(VectorStore::from_rows(3, vec![vec![1.0, 2.0, 3.0], vec![1.0]]).is_err());
    }

    /// Appending to a mapped store leaves the mapped rows in the file and adds a heap tail.
    #[test]
    fn appends_after_a_mapped_prefix_do_not_copy_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.bin");
        let source = rows(4, 2);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0u8; 64]).unwrap();
        for v in source.iter().flatten() {
            file.write_all(&v.to_ne_bytes()).unwrap();
        }
        drop(file);
        let map = unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
        let mut store = VectorStore::mapped(2, map, 64, 4).unwrap();
        store.push(&[9.0, 9.5]);
        assert!(store.is_mapped());
        assert_eq!(store.tail.len(), 2, "only the appended row is on the heap");
        assert_eq!(store.len(), 5);
        assert_eq!(store.row(3).unwrap(), source[3].as_slice());
        assert_eq!(store.row(4).unwrap(), &[9.0, 9.5]);
        let runs: Vec<(u32, usize)> = store.runs(5).map(|(id, r)| (id, r.len())).collect();
        assert_eq!(runs, vec![(0, 8), (4, 2)]);
        assert_eq!(store.to_flat().len(), 10);
    }

    #[test]
    fn codes_follow_rows_and_pushes() {
        let mut store =
            VectorStore::from_rows(3, vec![vec![1.0, -1.0, 0.0], vec![0.5, -0.25, 2.0]]).unwrap();
        assert!(store.code_row(0).is_none());
        store.set_codes(true);
        assert_eq!(store.code_row(0).unwrap(), &[127, -127, 0]);
        assert_eq!(store.code_row(1).unwrap(), &[64, -32, 127], "2.0 saturates");
        store.push(&[0.0, 0.1, -0.1]);
        assert_eq!(store.code_row(2).unwrap(), &[0, 13, -13]);
        assert!(store.code_row(3).is_none());
        store.set_codes(false);
        assert!(!store.has_codes());
        assert_eq!(quantize_query(&[0.2, -0.05, 0.0]), vec![127, -32, 0]);
        assert_eq!(quantize_query(&[0.0, 0.0]), vec![0, 0]);
    }

    #[test]
    fn deleted_ids_are_a_set() {
        let mut deleted = DeletedSet::default();
        assert_eq!(deleted.len(), 0);
        assert!(deleted.insert(130));
        assert!(deleted.insert(3));
        assert!(!deleted.insert(130));
        assert!(deleted.insert(64));
        assert_eq!(deleted.len(), 3);
        assert!(deleted.contains(3) && deleted.contains(64) && deleted.contains(130));
        assert!(!deleted.contains(4) && !deleted.contains(63) && !deleted.contains(100_000));
        assert_eq!(deleted.iter().collect::<Vec<_>>(), vec![3, 64, 130]);
        assert_eq!(deleted.iter().collect::<DeletedSet>(), deleted);
        deleted.clear();
        assert_eq!(deleted.len(), 0);
        assert!(!deleted.contains(3));
    }

    #[test]
    fn a_visited_set_forgets_between_walks_and_across_epoch_wrap() {
        let pool = VisitedPool::default();
        {
            let mut walk = pool.take(10);
            assert!(walk.insert(4));
            assert!(!walk.insert(4));
            assert!(walk.insert(25), "an id past the sizing grows the set");
        }
        {
            let mut walk = pool.take(10);
            assert!(walk.insert(4), "a new walk starts empty");
            walk.epoch = u32::MAX;
            walk.stamps[7] = u32::MAX;
        }
        let mut walk = pool.take(10);
        assert_eq!(
            walk.epoch, 1,
            "the epoch wrapped and the stamps were cleared"
        );
        assert!(walk.insert(7));
        assert_eq!(
            pool.free.lock().len(),
            0,
            "the set is out of the pool while held"
        );
        drop(walk);
        assert_eq!(pool.free.lock().len(), 1);
    }

    #[test]
    fn a_mapping_too_short_for_its_rows_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.bin");
        std::fs::write(&path, [0u8; 64 + 12]).unwrap();
        let map = unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
        assert!(VectorStore::mapped(2, map, 64, 2).is_err());
    }
}
