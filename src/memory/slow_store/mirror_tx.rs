//! Short write transactions for the graph mirror.
//!
//! The mirror shares its database file, and so SQLite's one write lock, with
//! the intent-log projection, which writes on every remember. Each mirror
//! transaction holds the projection connection's mutex as its write turn and
//! commits once it has run for [`MIRROR_TX_BUDGET`], or at its next step once
//! a projection write is waiting, and hands the turn to that writer, which
//! would otherwise sleep in SQLite's busy handler. For the
//! length of a sync the projection connection does not checkpoint: the mirror
//! folds the WAL back between its transactions, on its own thread, so no
//! projection commit copies the mirror's pages and syncs the file.

use anyhow::Result;
use rusqlite::{Connection, Transaction, TransactionBehavior};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::SlowStore;

/// How long one mirror write transaction runs before it commits.
pub(super) const MIRROR_TX_BUDGET: Duration = Duration::from_micros(250);

/// Rows written between checks of the budget and of waiting writers.
pub(super) const MIRROR_STEP_ROWS: usize = 4;

/// Mirror transactions between checkpoints.
const CHECKPOINT_EVERY_TXS: u32 = 16;

/// Rows deleted per statement when a retired table is emptied.
const CLEAR_STEP_ROWS: usize = 32;

/// Whether a step of mirror writes left work for another.
pub(super) enum Step {
    More,
    Done,
}

/// Restores the projection connection's automatic checkpoint on drop.
pub(super) struct ProjectionCheckpointsPaused<'a> {
    store: &'a SlowStore,
    pages: i64,
}

impl Drop for ProjectionCheckpointsPaused<'_> {
    fn drop(&mut self) {
        let conn = self.store.conn.lock();
        if let Err(e) = conn.pragma_update(None, "wal_autocheckpoint", self.pages) {
            tracing::warn!(error = %e, "projection checkpoint threshold not restored");
        }
    }
}

impl SlowStore {
    /// Stop the projection connection's automatic checkpoint until the guard drops.
    pub(super) fn pause_projection_checkpoints(&self) -> Result<ProjectionCheckpointsPaused<'_>> {
        let conn = self.conn.lock();
        let pages: i64 = conn.query_row("PRAGMA wal_autocheckpoint", [], |r| r.get(0))?;
        conn.pragma_update(None, "wal_autocheckpoint", 0)?;
        Ok(ProjectionCheckpointsPaused { store: self, pages })
    }

    /// Run `step` in mirror write transactions of about [`MIRROR_TX_BUDGET`]
    /// each until it is done. `between_chunks` runs after each commit, with no
    /// transaction open and the write turn released.
    pub(super) fn budgeted_writes(
        &self,
        conn: &mut Connection,
        between_chunks: &mut dyn FnMut(),
        mut step: impl FnMut(&Transaction<'_>) -> Result<Step>,
    ) -> Result<()> {
        let mut commits = 0u32;
        loop {
            let turn = self.conn.lock();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let start = Instant::now();
            let done = loop {
                if matches!(step(&tx)?, Step::Done) {
                    break true;
                }
                if start.elapsed() >= MIRROR_TX_BUDGET
                    || self.writers_waiting.load(Ordering::Acquire) > 0
                {
                    break false;
                }
            };
            tx.commit()?;
            parking_lot::MutexGuard::unlock_fair(turn);
            commits += 1;
            if done || commits.is_multiple_of(CHECKPOINT_EVERY_TXS) {
                checkpoint(conn);
            }
            between_chunks();
            if done {
                return Ok(());
            }
        }
    }

    /// Empty `table` a few rows per transaction, then drop it; nothing when it
    /// does not exist. Dropping a large table in one statement holds the write
    /// lock for as long as it takes to free every page.
    pub(super) fn clear_and_drop(
        &self,
        conn: &mut Connection,
        table: &str,
        between_chunks: &mut dyn FnMut(),
    ) -> Result<()> {
        let exists = conn
            .prepare_cached("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1")?
            .exists([table])?;
        if !exists {
            return Ok(());
        }
        let delete =
            format!("DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} LIMIT {CLEAR_STEP_ROWS})");
        self.budgeted_writes(conn, between_chunks, |tx| {
            Ok(if tx.execute(&delete, [])? == 0 {
                Step::Done
            } else {
                Step::More
            })
        })?;
        let turn = self.conn.lock();
        conn.execute_batch(&format!("DROP TABLE {table};"))?;
        parking_lot::MutexGuard::unlock_fair(turn);
        Ok(())
    }
}

/// Fold committed pages into the database on the mirror's own thread, outside
/// the write turn, so a projection commit's automatic checkpoint finds little
/// to copy. A busy checkpoint is left to the next.
pub(super) fn checkpoint(conn: &Connection) {
    if let Err(e) = conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(())) {
        tracing::debug!(error = %e, "mirror checkpoint skipped");
    }
}
