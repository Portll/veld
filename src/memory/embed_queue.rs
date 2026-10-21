//! Memories written without a vector, waiting for the next batch.
//!
//! A request stores the memory, indexes it for keyword search and returns; the memory joins
//! this queue. A server loop drains each tenant's queue a batch at a time, so one embedder call
//! carries many texts and no request waits on the embedder. The queue is in memory: a memory
//! still queued at a crash is stored and unindexed, which the backlog reconcile finds and
//! queues again.

use std::collections::{HashMap, VecDeque};

use super::types::MemoryId;

/// Why a memory waits for its vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedPurpose {
    /// A write here: once its vector exists, the write gate and interference detection run.
    Write,
    /// A snapshot replicated from a leader, which already judged it: its vectors only. The
    /// leader's own frames carry whatever its write gate decided.
    Replica,
}

#[derive(Default)]
pub struct EmbedQueue {
    order: VecDeque<MemoryId>,
    queued: HashMap<MemoryId, EmbedPurpose>,
}

impl EmbedQueue {
    /// Add a memory written here, once: a memory already waiting keeps its place.
    pub fn push(&mut self, id: MemoryId) {
        self.push_as(id, EmbedPurpose::Write);
    }

    /// Add a memory for `purpose`, once. A memory already waiting keeps its place; a write
    /// here outranks a replica, so its write gate still runs.
    pub fn push_as(&mut self, id: MemoryId, purpose: EmbedPurpose) {
        match self.queued.get_mut(&id) {
            Some(waiting) => {
                if purpose == EmbedPurpose::Write {
                    *waiting = EmbedPurpose::Write;
                }
            }
            None => {
                self.queued.insert(id.clone(), purpose);
                self.order.push_back(id);
            }
        }
    }

    /// The oldest `max` memories with why each waits, removed from the queue.
    pub fn take(&mut self, max: usize) -> Vec<(MemoryId, EmbedPurpose)> {
        let n = max.min(self.order.len());
        self.order
            .drain(..n)
            .map(|id| {
                let purpose = self.queued.remove(&id).unwrap_or(EmbedPurpose::Write);
                (id, purpose)
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    fn ids(batch: Vec<(MemoryId, EmbedPurpose)>) -> Vec<MemoryId> {
        batch.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn a_memory_is_queued_once_and_taken_in_arrival_order() {
        let mut q = EmbedQueue::default();
        for n in [3u128, 1, 3, 2] {
            q.push(id(n));
        }
        assert_eq!(q.len(), 3);
        assert_eq!(ids(q.take(2)), vec![id(3), id(1)]);
        q.push(id(3));
        assert_eq!(ids(q.take(10)), vec![id(2), id(3)]);
        assert!(q.is_empty());
    }

    #[test]
    fn a_write_here_outranks_a_replica_and_keeps_its_place() {
        let mut q = EmbedQueue::default();
        q.push_as(id(1), EmbedPurpose::Replica);
        q.push_as(id(2), EmbedPurpose::Replica);
        q.push(id(1));
        q.push_as(id(2), EmbedPurpose::Replica);
        assert_eq!(
            q.take(10),
            vec![(id(1), EmbedPurpose::Write), (id(2), EmbedPurpose::Replica)]
        );
    }
}
