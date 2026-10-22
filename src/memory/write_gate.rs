//! Per-user write serialization that a shutdown can close.
//!
//! Every create, upsert, embedding write-back and compression step takes the gate
//! before touching the store. Concurrent writers for one tenant queue on it, one at a
//! time, so a burst of upserts accepted just before a shutdown signal used to be
//! served one by one after "Server shutdown complete": on 2026-10-20 the live server
//! wrote for ten seconds past its flush until the runtime's shutdown bound cut the
//! queue off mid-request. Closing the gate lets the writer inside finish and turns
//! every queued or later writer away with [`WritesClosed`], which the HTTP layer
//! answers as a retryable 503.
//!
//! The gate also orders the tenant's intent log. A write path queues its frame
//! (`MultiUserMemoryManager::journal_enqueue`) while it still holds the gate, from the
//! snapshot the store took under the gate, so the log takes the frames for one memory
//! in the order the store took the writes; replication followers and projections apply
//! in log order, and a frame out of that order leaves them stale or resurrects a
//! deleted memory. The queue fixes a frame's LSN, so the gate is released before the
//! sync is waited for: held through the sync, it would keep every other writer to the
//! tenant out of the queue and no two writes would share a sync. Lock order: the gate,
//! then the journal's queue and writer mutexes, then the graph's lock. Nothing that
//! holds the journal mutex or a graph guard may enter the gate.
//!
//! The gate is re-entrant on its thread: a path that holds it across a store write and
//! the append calls the store's own gated methods, which enter it again.

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};

/// Proof that the current thread holds a tenant's write gate.
pub type WriteGuard<'a> = ReentrantMutexGuard<'a, ()>;

/// A writer arrived after the tenant was closed for shutdown. Nothing was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritesClosed;

impl std::fmt::Display for WritesClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tenant is closed to writes: the server is shutting down")
    }
}

impl std::error::Error for WritesClosed {}

#[derive(Debug, Default)]
pub struct WriteGate {
    lock: ReentrantMutex<()>,
    closed: AtomicBool,
}

impl WriteGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait for the gate, then refuse if it was closed meanwhile. The check comes after
    /// the lock so a writer that queued before the close cannot slip through after it.
    pub fn enter(&self) -> Result<WriteGuard<'_>, WritesClosed> {
        let guard = self.lock.lock();
        if self.closed.load(Ordering::Acquire) {
            return Err(WritesClosed);
        }
        Ok(guard)
    }

    /// Wait for the gate whether or not it is closed. For appending the intent-log
    /// frames of writes the store already took, and for the write paths that were never
    /// refused at shutdown (update, forget): a refusal there would change what those
    /// requests answer, while the frame of an accepted write still has to land.
    pub fn hold(&self) -> WriteGuard<'_> {
        self.lock.lock()
    }

    /// Close the gate and wait for the writer inside, if any. The flag is set before
    /// the lock is taken: taking it first would wait behind every queued writer, each
    /// of which would then write. On return no gated write is running or can start.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        drop(self.lock.lock());
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Whether `WritesClosed` is anywhere in the chain, however many contexts wrap it.
pub fn chain_is_writes_closed(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<WritesClosed>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn an_open_gate_admits_writers() {
        let gate = WriteGate::new();
        assert!(gate.enter().is_ok());
        assert!(gate.enter().is_ok());
        assert!(!gate.is_closed());
    }

    #[test]
    fn a_closed_gate_refuses_writers() {
        let gate = WriteGate::new();
        gate.close();
        assert_eq!(gate.enter().err(), Some(WritesClosed));
        assert!(gate.is_closed());
    }

    #[test]
    fn close_waits_for_the_writer_inside_and_refuses_the_queue() {
        let gate = Arc::new(WriteGate::new());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (finished_tx, finished_rx) = mpsc::channel();

        let inside = {
            let gate = Arc::clone(&gate);
            let finished_tx = finished_tx.clone();
            std::thread::spawn(move || {
                let _guard = gate.enter().expect("gate is open");
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                finished_tx.send("inside").unwrap();
            })
        };
        entered_rx.recv().unwrap();

        let queued: Vec<_> = (0..3)
            .map(|_| {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || gate.enter().map(|_| ()))
            })
            .collect();

        let closer = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                gate.close();
                finished_tx.send("closed").unwrap();
            })
        };
        while !gate.is_closed() {
            std::thread::yield_now();
        }
        assert!(
            finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "close returned while a writer was still inside"
        );

        release_tx.send(()).unwrap();
        assert_eq!(finished_rx.recv().unwrap(), "inside");
        assert_eq!(finished_rx.recv().unwrap(), "closed");
        inside.join().unwrap();
        closer.join().unwrap();
        for writer in queued {
            assert_eq!(writer.join().unwrap(), Err(WritesClosed));
        }
    }

    #[test]
    fn the_holder_enters_again_and_others_wait() {
        let gate = Arc::new(WriteGate::new());
        let outer = gate.enter().expect("gate is open");
        let inner = gate.enter().expect("the holder enters again");
        let (entered_tx, entered_rx) = mpsc::channel();
        let other = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let _guard = gate.hold();
                entered_tx.send(()).unwrap();
            })
        };
        assert!(
            entered_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "another thread entered while the gate was held"
        );
        drop(inner);
        assert!(
            entered_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "another thread entered while the outer hold remained"
        );
        drop(outer);
        entered_rx.recv().unwrap();
        other.join().unwrap();
    }

    #[test]
    fn a_closed_gate_is_still_held_for_journaling() {
        let gate = WriteGate::new();
        gate.close();
        assert_eq!(gate.enter().err(), Some(WritesClosed));
        drop(gate.hold());
    }

    #[test]
    fn the_refusal_is_found_under_context() {
        let err = anyhow::Error::new(WritesClosed).context("upsert claude-memory/x");
        assert!(chain_is_writes_closed(&err));
        assert!(!chain_is_writes_closed(&anyhow::anyhow!("disk full")));
    }
}
