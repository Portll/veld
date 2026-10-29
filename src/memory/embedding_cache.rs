//! Embedding caches keyed by space and text hash.
//!
//! The content and query caches used to be keyed by the text's SHA-256 alone, which is exact
//! only while one embedder writes every vector. Keyed by `(space, hash)`, a vector cached for
//! one model is never served for another. `get` and `insert` read and write the cache's own
//! space, the primary index's; `get_in` and `insert_in` name another.

use crate::embeddings::space::SpaceId;

pub struct EmbeddingCache {
    space: SpaceId,
    inner: moka::sync::Cache<(SpaceId, [u8; 32]), Vec<f32>>,
}

impl EmbeddingCache {
    pub fn new(space: SpaceId, max_capacity: u64) -> Self {
        Self {
            space,
            inner: moka::sync::Cache::builder()
                .max_capacity(max_capacity)
                .build(),
        }
    }

    /// The space `get` and `insert` use.
    pub fn space(&self) -> &SpaceId {
        &self.space
    }

    pub fn get(&self, hash: &[u8; 32]) -> Option<Vec<f32>> {
        self.inner.get(&(self.space.clone(), *hash))
    }

    pub fn insert(&self, hash: [u8; 32], vector: Vec<f32>) {
        self.inner.insert((self.space.clone(), hash), vector);
    }

    pub fn get_in(&self, space: &SpaceId, hash: &[u8; 32]) -> Option<Vec<f32>> {
        self.inner.get(&(space.clone(), *hash))
    }

    pub fn insert_in(&self, space: &SpaceId, hash: [u8; 32], vector: Vec<f32>) {
        self.inner.insert((space.clone(), hash), vector);
    }

    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }

    pub fn invalidate_all(&self) {
        self.inner.invalidate_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_text_in_two_spaces_is_two_entries() {
        let nomic = SpaceId::new("nomic-embed-text-v1.5", 768);
        let minilm = SpaceId::new("minilm-l6-v2", 384);
        let cache = EmbeddingCache::new(nomic.clone(), 16);
        let hash = [7u8; 32];
        cache.insert(hash, vec![1.0; 768]);
        cache.insert_in(&minilm, hash, vec![2.0; 384]);
        assert_eq!(cache.get(&hash).map(|v| v.len()), Some(768));
        assert_eq!(cache.get_in(&nomic, &hash).map(|v| v.len()), Some(768));
        assert_eq!(cache.get_in(&minilm, &hash).map(|v| v.len()), Some(384));
        let other = SpaceId::new("bge-m3", 1024);
        assert!(
            cache.get_in(&other, &hash).is_none(),
            "a space nothing was cached for gets nothing"
        );
    }
}
