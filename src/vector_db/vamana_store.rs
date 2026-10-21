//! Vector rows of a [`super::vamana::VamanaIndex`].

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
}

impl VectorStore {
    pub(crate) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            mapped: None,
            tail: Vec::new(),
            tail_rows: 0,
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
    fn a_mapping_too_short_for_its_rows_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.bin");
        std::fs::write(&path, [0u8; 64 + 12]).unwrap();
        let map = unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
        assert!(VectorStore::mapped(2, map, 64, 2).is_err());
    }
}
