//! Memories written without a vector, waiting for the next batch.
//!
//! A request stores the memory, indexes it for keyword search and returns; the memory joins
//! this queue. A server loop drains each tenant's queue a batch at a time, so one embedder call
//! carries many texts and no request waits on the embedder. The queue is in memory: a memory
//! still queued at a crash is stored and unindexed, which the backlog reconcile finds and
//! queues again.

use std::collections::{HashSet, VecDeque};

use super::types::MemoryId;

#[derive(Default)]
pub struct EmbedQueue {
    order: VecDeque<MemoryId>,
    queued: HashSet<MemoryId>,
}

impl EmbedQueue {
    /// Add a memory, once: a memory already waiting keeps its place.
    pub fn push(&mut self, id: MemoryId) {
        if self.queued.insert(id.clone()) {
            self.order.push_back(id);
        }
    }

    /// The oldest `max` memories, removed from the queue.
    pub fn take(&mut self, max: usize) -> Vec<MemoryId> {
        let n = max.min(self.order.len());
        let batch: Vec<MemoryId> = self.order.drain(..n).collect();
        for id in &batch {
            self.queued.remove(id);
        }
        batch
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

    #[test]
    fn a_memory_is_queued_once_and_taken_in_arrival_order() {
        let mut q = EmbedQueue::default();
        for n in [3u128, 1, 3, 2] {
            q.push(id(n));
        }
        assert_eq!(q.len(), 3);
        assert_eq!(q.take(2), vec![id(3), id(1)]);
        q.push(id(3));
        assert_eq!(q.take(10), vec![id(2), id(3)]);
        assert!(q.is_empty());
    }
}
