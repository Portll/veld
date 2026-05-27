//! Multi-tenant access control for shared memories.
//!
//! # Model
//!
//! Veld already isolates memories per-user by key prefix (`user_id`). The
//! ACL layer on top of that lets the **owner** explicitly share specific
//! memories with **other** users — read-only or read-write — for
//! collaborative knowledge workspaces.
//!
//! ## Scope
//!
//! - Each memory has at most one [`MemoryAcl`] record. The owner is
//!   *implicit* (whoever stored the memory under their user_id prefix);
//!   the ACL only lists *additional* readers and writers.
//! - `readers` and `writers` are full user_id strings.
//! - `writers` does not imply `readers` — add the user to both if you
//!   want them to do both.
//! - Absence of an ACL record means "owner only" (the existing default).
//!
//! # Storage
//!
//! `acl:{owner_user_id}:{memory_id} → bincode(MemoryAcl)` over the shared
//! RocksDB instance. Same key prefix shape as the rest of veld's stores
//! so single-user scans don't fan out.

use anyhow::Result;
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use super::types::MemoryId;

/// Sharing record for a single memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryAcl {
    /// User IDs granted read access (in addition to the owner)
    #[serde(default)]
    pub readers: HashSet<String>,
    /// User IDs granted read+write access (in addition to the owner)
    #[serde(default)]
    pub writers: HashSet<String>,
}

impl MemoryAcl {
    /// True if `viewer` can read this memory (is owner, explicit reader,
    /// or writer — writers always include read).
    pub fn can_read(&self, owner: &str, viewer: &str) -> bool {
        owner == viewer
            || self.readers.contains(viewer)
            || self.writers.contains(viewer)
    }

    /// True if `actor` can modify this memory.
    pub fn can_write(&self, owner: &str, actor: &str) -> bool {
        owner == actor || self.writers.contains(actor)
    }
}

/// Persistence layer for memory ACLs.
pub struct MemoryAclStore {
    db: Arc<DB>,
}

impl MemoryAclStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn key(owner: &str, memory_id: &MemoryId) -> String {
        format!("acl:{}:{}", owner, memory_id.0)
    }

    /// Fetch the ACL for a memory. Returns the default (empty) ACL when no
    /// record exists, which means "owner-only".
    pub fn get(&self, owner: &str, memory_id: &MemoryId) -> Result<MemoryAcl> {
        let key = Self::key(owner, memory_id);
        match self.db.get(key.as_bytes())? {
            Some(bytes) => {
                let (acl, _): (MemoryAcl, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
                Ok(acl)
            }
            None => Ok(MemoryAcl::default()),
        }
    }

    fn put(&self, owner: &str, memory_id: &MemoryId, acl: &MemoryAcl) -> Result<()> {
        let key = Self::key(owner, memory_id);
        let bytes = bincode::serde::encode_to_vec(acl, bincode::config::standard())?;
        self.db.put(key.as_bytes(), &bytes)?;
        Ok(())
    }

    /// Grant read access to `target_user_id`.
    pub fn grant_read(
        &self,
        owner: &str,
        memory_id: &MemoryId,
        target_user_id: &str,
    ) -> Result<()> {
        let mut acl = self.get(owner, memory_id)?;
        acl.readers.insert(target_user_id.to_string());
        self.put(owner, memory_id, &acl)
    }

    /// Grant read+write access to `target_user_id`.
    pub fn grant_write(
        &self,
        owner: &str,
        memory_id: &MemoryId,
        target_user_id: &str,
    ) -> Result<()> {
        let mut acl = self.get(owner, memory_id)?;
        acl.writers.insert(target_user_id.to_string());
        self.put(owner, memory_id, &acl)
    }

    /// Revoke any read or write grant for `target_user_id`.
    pub fn revoke(
        &self,
        owner: &str,
        memory_id: &MemoryId,
        target_user_id: &str,
    ) -> Result<bool> {
        let mut acl = self.get(owner, memory_id)?;
        let r = acl.readers.remove(target_user_id);
        let w = acl.writers.remove(target_user_id);
        if r || w {
            self.put(owner, memory_id, &acl)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete the ACL record entirely (back to owner-only).
    pub fn clear(&self, owner: &str, memory_id: &MemoryId) -> Result<()> {
        let key = Self::key(owner, memory_id);
        self.db.delete(key.as_bytes())?;
        Ok(())
    }

    /// True if `viewer` can read the memory owned by `owner`.
    pub fn can_read(
        &self,
        owner: &str,
        memory_id: &MemoryId,
        viewer: &str,
    ) -> Result<bool> {
        Ok(self.get(owner, memory_id)?.can_read(owner, viewer))
    }

    /// True if `actor` can modify the memory owned by `owner`.
    pub fn can_write(
        &self,
        owner: &str,
        memory_id: &MemoryId,
        actor: &str,
    ) -> Result<bool> {
        Ok(self.get(owner, memory_id)?.can_write(owner, actor))
    }

    /// List all (memory_id, acl) pairs that `owner` has shared with anyone.
    /// Useful for an admin "what have I shared" view.
    pub fn list_shared(&self, owner: &str) -> Result<Vec<(MemoryId, MemoryAcl)>> {
        let prefix = format!("acl:{}:", owner);
        let mut out = Vec::new();
        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            let id_str = &key_str[prefix.len()..];
            let Ok(uuid) = uuid::Uuid::parse_str(id_str) else {
                continue;
            };
            let (acl, _): (MemoryAcl, _) =
                bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
            out.push((MemoryId(uuid), acl));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn store() -> (MemoryAclStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(dir.path()).unwrap());
        (MemoryAclStore::new(db), dir)
    }

    #[test]
    fn owner_can_always_read_and_write() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        assert!(s.can_read("alice", &mid, "alice").unwrap());
        assert!(s.can_write("alice", &mid, "alice").unwrap());
    }

    #[test]
    fn stranger_default_denied() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        assert!(!s.can_read("alice", &mid, "bob").unwrap());
        assert!(!s.can_write("alice", &mid, "bob").unwrap());
    }

    #[test]
    fn grant_read_then_check() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        s.grant_read("alice", &mid, "bob").unwrap();
        assert!(s.can_read("alice", &mid, "bob").unwrap());
        assert!(!s.can_write("alice", &mid, "bob").unwrap());
    }

    #[test]
    fn grant_write_implies_read() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        s.grant_write("alice", &mid, "bob").unwrap();
        assert!(s.can_write("alice", &mid, "bob").unwrap());
        assert!(s.can_read("alice", &mid, "bob").unwrap());
    }

    #[test]
    fn revoke_removes_access() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        s.grant_read("alice", &mid, "bob").unwrap();
        s.grant_write("alice", &mid, "carol").unwrap();

        assert!(s.revoke("alice", &mid, "bob").unwrap());
        assert!(!s.can_read("alice", &mid, "bob").unwrap());

        assert!(s.revoke("alice", &mid, "carol").unwrap());
        assert!(!s.can_write("alice", &mid, "carol").unwrap());

        assert!(!s.revoke("alice", &mid, "ghost").unwrap()); // no-op
    }

    #[test]
    fn list_shared_returns_all() {
        let (s, _d) = store();
        let m1 = MemoryId(Uuid::new_v4());
        let m2 = MemoryId(Uuid::new_v4());
        s.grant_read("alice", &m1, "bob").unwrap();
        s.grant_write("alice", &m2, "carol").unwrap();
        let shared = s.list_shared("alice").unwrap();
        assert_eq!(shared.len(), 2);
    }

    #[test]
    fn clear_removes_record() {
        let (s, _d) = store();
        let mid = MemoryId(Uuid::new_v4());
        s.grant_read("alice", &mid, "bob").unwrap();
        s.clear("alice", &mid).unwrap();
        assert!(!s.can_read("alice", &mid, "bob").unwrap());
    }
}
