//! `RegistryStore` — the org/workspace/project/folder/ACL graph, backed by five
//! column families in the shared RocksDB.
//!
//! Modeled on `TodoStore` (`src/memory/todos.rs`): it holds the shared
//! `Arc<DB>`, exposes [`RegistryStore::cf_descriptors`] for the shared-DB
//! bootstrap, and — crucially — writes every multi-key mutation in ONE
//! [`rocksdb::WriteBatch`] so an entity plus its secondary indexes and reverse
//! edges are crash-atomic. This deliberately does NOT copy the sequential
//! `put_cf` pattern in `todos.rs` (`store_project`), where a crash between puts
//! can orphan an index.

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, IteratorMode, Options, WriteBatch, DB};

/// Canonical entity records (JSON values): `acct:` / `org:` / `ws:` / `proj:` / `folder:`.
pub const CF_ENTITIES: &str = "registry_entities";
/// Edges: `ws_proj:` / `proj_ws:` / `folder_proj:` / `proj_folder:` / `proj_edge:`.
pub const CF_LINKS: &str = "registry_links";
/// Secondary lookup indexes: `scope2proj:` / `remote2scope:` / `org_proj:` / `wsname:`.
pub const CF_INDEX: &str = "registry_index";
/// Principal (tenant `user_id`) → authorized scope grants: `grant:` / `scopegrant:`.
pub const CF_ACL: &str = "registry_acl";
/// Schema version + counters.
pub const CF_META: &str = "registry_meta";

/// Serializes registry read-modify-write so uniqueness checks (`scope2proj`,
/// workspace name, `remote2scope`) are race-free — RocksDB has no unique index,
/// so the check-then-write must hold this lock. Mirrors `admin.rs`'s `BIND_LOCK`.
pub static REGISTRY_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// The registry backing store. Cheap to clone-share via `Arc`.
pub struct RegistryStore {
    db: Arc<DB>,
}

impl RegistryStore {
    /// Column-family descriptors the caller MUST include (alongside `"default"`
    /// and the other stores' CFs) when opening the shared DB.
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        [CF_ENTITIES, CF_LINKS, CF_INDEX, CF_ACL, CF_META]
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, opts.clone()))
            .collect()
    }

    /// Wrap a shared DB whose registry CFs have already been opened.
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn cf(&self, name: &str) -> &ColumnFamily {
        self.db
            .cf_handle(name)
            .unwrap_or_else(|| panic!("registry CF `{name}` must exist — add cf_descriptors() to the shared-DB bootstrap"))
    }

    /// Atomically apply a set of puts and deletes across CFs in ONE write batch.
    /// This is the sole mutation primitive; every create/link/delete routes
    /// through it so entity + indexes + reverse edges commit all-or-nothing.
    pub fn write_batch(
        &self,
        puts: &[(&str, String, Vec<u8>)],
        deletes: &[(&str, String)],
    ) -> Result<()> {
        let mut batch = WriteBatch::default();
        for (cf, key, value) in puts {
            batch.put_cf(self.cf(cf), key.as_bytes(), value);
        }
        for (cf, key) in deletes {
            batch.delete_cf(self.cf(cf), key.as_bytes());
        }
        self.db.write(batch)?;
        Ok(())
    }

    /// Fetch a single raw value by CF + key.
    pub fn get(&self, cf: &str, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get_cf(self.cf(cf), key.as_bytes())?)
    }

    /// `true` if a key exists (uniqueness pre-check under [`REGISTRY_LOCK`]).
    pub fn exists(&self, cf: &str, key: &str) -> Result<bool> {
        Ok(self.get(cf, key)?.is_some())
    }

    /// Prefix scan a CF, returning `(key, value)` for every key starting with
    /// `prefix`. Used for list_tree walks, ACL grant enumeration, and reverse
    /// edge fans.
    pub fn scan_prefix(&self, cf: &str, prefix: &str) -> Vec<(String, Vec<u8>)> {
        let handle = self.cf(cf);
        let iter = self.db.iterator_cf(
            handle,
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        let mut out = Vec::new();
        for item in iter.flatten() {
            let (k, v) = item;
            let key = String::from_utf8_lossy(&k).to_string();
            if !key.starts_with(prefix) {
                break;
            }
            out.push((key, v.to_vec()));
        }
        out
    }
}
