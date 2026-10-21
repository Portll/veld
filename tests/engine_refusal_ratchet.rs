//! ADR 0007 ratchet: every place one storage engine refuses, or has no counterpart,
//! is written down here by name, and the list may only shrink.
//!
//! The ADR's promise is that a capability added for RocksDB without a redb
//! implementation "shows up as a named refusal, so the distance to the gates is
//! visible". That holds only if a refusal that nobody named cannot appear. These
//! tests scan the storage layer, the backup module and the handlers for the
//! phrasings the code uses to refuse, and require the list below to match what they
//! find in both directions: a new refusal is a silent divergence until it is listed,
//! and a listed one that no longer exists is a stale claim that the gap is open.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One place an engine refuses, or is absent, with the reason it is allowed to.
/// `site` is `<path under src/>::<enclosing fn>`, which survives edits that only
/// move lines.
struct Refusal {
    site: &'static str,
    engine: &'static str,
    interface: &'static str,
    method: &'static str,
    reason: &'static str,
}

const KNOWN_REFUSALS: &[Refusal] = &[
    Refusal {
        site: "storage/legacy_rocksdb.rs::import_portable",
        engine: "rocksdb",
        interface: "MigrationStore",
        method: "import_portable",
        reason: "R-6b: RocksDB exports a portable dump but cannot import one; redb can",
    },
    Refusal {
        site: "storage/legacy_rocksdb.rs::migrate_in_place",
        engine: "rocksdb",
        interface: "MigrationStore",
        method: "migrate_in_place",
        reason: "R-6d, deliberate: export, import into a scratch target, verify, then switch config",
    },
    Refusal {
        site: "storage/redb.rs::migrate_in_place",
        engine: "redb",
        interface: "MigrationStore",
        method: "migrate_in_place",
        reason: "R-6d, deliberate and symmetric with RocksDB: `veld migrate` is the path",
    },
    Refusal {
        site: "storage/legacy_rocksdb.rs::open_shared_store",
        engine: "rocksdb",
        interface: "StorageFactory",
        method: "open_shared_store",
        reason: "a second opener over the live shared DB is unsafe; bootstrap hands the open handle over",
    },
    Refusal {
        site: "storage/legacy_rocksdb.rs::open_audit_store",
        engine: "rocksdb",
        interface: "StorageFactory",
        method: "open_audit_store",
        reason: "same: the audit family lives in the live shared DB",
    },
    Refusal {
        site: "storage/legacy_rocksdb.rs::open_backup_store",
        engine: "rocksdb",
        interface: "StorageFactory",
        method: "open_backup_store",
        reason: "needs a bound primary DB reference; backups go through the backup engine bootstrap",
    },
    Refusal {
        site: "storage/legacy_rocksdb.rs::open_migration_store",
        engine: "rocksdb",
        interface: "StorageFactory",
        method: "open_migration_store",
        reason: "no path constructor; RocksDbMigrationStore::new takes the live handle",
    },
    Refusal {
        site: "storage/mod.rs::require_rocksdb",
        engine: "redb",
        interface: "SharedEngineDb",
        method: "require_rocksdb",
        reason: "raw-column consumers (user-auth, GDPR purge scans, comprehensive backup) are still RocksDB-typed",
    },
    Refusal {
        site: "storage/migrate/mod.rs::extract_with",
        engine: "redb",
        interface: "migrate",
        method: "extract_with",
        reason: "a build without the storage-redb feature has no redb backend to read",
    },
    Refusal {
        site: "storage/migrate/mod.rs::apply",
        engine: "redb",
        interface: "migrate",
        method: "apply",
        reason: "a build without the storage-redb feature has no redb backend to write",
    },
    Refusal {
        site: "handlers/state.rs::no_rocksdb",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "no_rocksdb",
        reason: "the refusal a redb-only build returns when the configured backend is rocksdb",
    },
    Refusal {
        site: "handlers/state.rs::bootstrap_shared_stores",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "bootstrap_shared_stores",
        reason: "redb-only build: the rocksdb shared stores are not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::open_user_memory",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "open_user_memory",
        reason: "redb-only build: the rocksdb per-user memory store is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::open_user_graph_memory",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "open_user_graph_memory",
        reason: "redb-only build: the rocksdb per-user graph store is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::create_user_backup",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "create_user_backup",
        reason: "redb-only build: the rocksdb backup engine is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::list_backups_for_user",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "list_backups_for_user",
        reason: "redb-only build: the rocksdb backup engine is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::purge_backups_for_user",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "purge_backups_for_user",
        reason: "redb-only build: the rocksdb backup engine is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::restore_user_stores",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "restore_user_stores",
        reason: "redb-only build: the rocksdb backup engine is not compiled in",
    },
    Refusal {
        site: "handlers/state.rs::require_backup_engine",
        engine: "redb",
        interface: "MultiUserMemoryManager",
        method: "require_backup_engine",
        reason: "the rocksdb backup engine does not exist on a redb deployment, which uses RedbBackupEngine",
    },
    Refusal {
        site: "handlers/state.rs::require_redb_backup_engine",
        engine: "rocksdb",
        interface: "MultiUserMemoryManager",
        method: "require_redb_backup_engine",
        reason: "the redb backup engine does not exist on a rocksdb deployment, which uses VeldBackupEngine",
    },
];

/// Source that carries a refusal, relative to `src/`. A directory is walked.
const SCANNED: &[&str] = &[
    "storage/redb.rs",
    "storage/legacy_rocksdb.rs",
    "storage/mod.rs",
    "storage/migrate/mod.rs",
    "backup.rs",
    "handlers",
];

/// The words the code uses when it refuses on engine grounds. A pattern ending in
/// `(` is a call and does not match its own definition.
const REFUSAL_PHRASES: &[&str] = &[
    "not_implemented(",
    "is not implemented for",
    "store is not implemented",
    "deliberately not implemented",
    "requires the rocksdb",
    "requires the redb",
    "requires a bound",
    "needs the rocksdb engine",
    "no_rocksdb(",
    "has no redb backend",
    "must be opened from the live",
];

/// Counted exactly, like `tests/rocksdb_ratchet.rs`: a module that removes a
/// `#[cfg(feature = "rocksdb")]`-only item must lower its number, or the slack
/// would readmit growth unnoticed.
const ROCKSDB_ONLY_ITEMS: &[(&str, usize)] = &[("backup.rs", 20), ("handlers", 16)];

const ROCKSDB_ONLY_ATTRIBUTE: &str = "#[cfg(feature = \"rocksdb\")]";

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Found {
    site: String,
    line: usize,
    text: String,
}

fn function_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    if let Some(after) = rest.strip_prefix("pub ") {
        rest = after;
    } else if let Some(after) = rest.strip_prefix("pub(") {
        rest = &after[after.find(')')? + 1..];
    }
    loop {
        let trimmed = rest.trim_start();
        let stripped = ["const ", "async ", "unsafe ", "extern "]
            .iter()
            .find_map(|qualifier| trimmed.strip_prefix(qualifier));
        match stripped {
            Some(after) => rest = after,
            None => {
                rest = trimmed;
                break;
            }
        }
    }
    let name: String = rest
        .strip_prefix("fn ")?
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Index of the line after a top-level `#[cfg(test)] mod x { ... }` that starts
/// at `start`, or `None` when `start` is not the attribute of such a module.
fn end_of_test_module(lines: &[&str], start: usize) -> Option<usize> {
    let attribute = lines[start];
    let is_test_cfg = attribute == "#[cfg(test)]" || attribute.starts_with("#[cfg(all(test");
    if !is_test_cfg {
        return None;
    }
    let mut declaration = start + 1;
    while declaration < lines.len() && lines[declaration].starts_with("#[") {
        declaration += 1;
    }
    let header = lines.get(declaration)?;
    let opens_block =
        (header.starts_with("mod ") || header.starts_with("pub mod ")) && header.ends_with('{');
    if !opens_block {
        return None;
    }
    let close = (declaration + 1..lines.len()).find(|&i| lines[i] == "}")?;
    Some(close + 1)
}

fn find_refusals(rel: &str, source: &str) -> Vec<Found> {
    let lines: Vec<&str> = source.lines().collect();
    let mut found = Vec::new();
    let mut enclosing = String::from("<module>");
    let mut index = 0;
    while index < lines.len() {
        if let Some(after_module) = end_of_test_module(&lines, index) {
            index = after_module;
            continue;
        }
        let line = lines[index];
        if let Some(name) = function_name(line) {
            enclosing = name;
        }
        if !line.trim_start().starts_with("//") {
            let phrase = REFUSAL_PHRASES.iter().find(|phrase| {
                line.contains(**phrase)
                    && !(phrase.ends_with('(') && line.contains(&format!("fn {}", phrase)))
            });
            if let Some(phrase) = phrase {
                found.push(Found {
                    site: format!("{rel}::{enclosing}"),
                    line: index + 1,
                    text: format!("{} [{phrase}]", line.trim()),
                });
            }
        }
        index += 1;
    }
    found
}

fn rust_files(root: &Path, entry: &str) -> Vec<(String, PathBuf)> {
    fn walk(root: &Path, path: &Path, out: &mut Vec<(String, PathBuf)>) {
        if path.is_dir() {
            let mut children: Vec<PathBuf> = std::fs::read_dir(path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
                .map(|child| child.expect("dir entry").path())
                .collect();
            children.sort();
            for child in children {
                walk(root, &child, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path.to_path_buf()));
        }
    }
    let mut out = Vec::new();
    walk(root, &root.join(entry), &mut out);
    out
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn refusals_in_tree() -> BTreeMap<String, Found> {
    let root = src_root();
    let mut by_site = BTreeMap::new();
    for entry in SCANNED {
        for (rel, path) in rust_files(&root, entry) {
            for found in find_refusals(&rel, &read(&path)) {
                by_site.entry(found.site.clone()).or_insert(found);
            }
        }
    }
    by_site
}

#[test]
fn the_scanner_sees_refusals_and_skips_comments_definitions_and_tests() {
    let planted = r#"
fn not_implemented(operation: &str) -> Error {
    anyhow!("redb {} is gone", operation)
}

impl Thing {
    pub(crate) fn import_portable(&self) -> Result<()> {
        // requires the rocksdb engine, said in a comment
        Err(anyhow!("import_portable is not implemented for rocksdb"))
    }

    fn other(&self) -> Result<()> {
        Err(not_implemented("x"))
    }
}

#[cfg(test)]
mod tests {
    fn in_a_test() {
        let _ = "requires the redb backup engine";
    }
}

fn after_the_tests() -> Result<()> {
    bail!("{what} requires the rocksdb engine")
}
"#;
    let sites: Vec<String> = find_refusals("x.rs", planted)
        .into_iter()
        .map(|found| found.site)
        .collect();
    assert_eq!(
        sites,
        vec![
            "x.rs::import_portable",
            "x.rs::other",
            "x.rs::after_the_tests"
        ]
    );
    assert_eq!(
        function_name("    pub(crate) async fn run_it(&self)"),
        Some("run_it".to_string())
    );
    assert_eq!(function_name("let fn_like = 1;"), None);
}

#[test]
fn every_engine_refusal_is_named_and_none_has_gone_stale() {
    let found = refusals_in_tree();

    let mut listed: BTreeMap<&str, &Refusal> = BTreeMap::new();
    for refusal in KNOWN_REFUSALS {
        assert!(
            listed.insert(refusal.site, refusal).is_none(),
            "{} appears twice in KNOWN_REFUSALS",
            refusal.site
        );
    }

    let mut problems = Vec::new();
    for (site, hit) in &found {
        if !listed.contains_key(site.as_str()) {
            let (path, function) = site.split_once("::").unwrap();
            problems.push(format!(
                "UNLISTED refusal at src/{path}:{} — {}\n    add to KNOWN_REFUSALS:\n    Refusal {{ site: \"{site}\", engine: \"<rocksdb|redb>\", interface: \"<trait>\", method: \"{function}\", reason: \"<why this engine may refuse>\" }},\n    or implement it on both engines, which is the better outcome",
                hit.line, hit.text
            ));
        }
    }
    for (site, refusal) in &listed {
        if !found.contains_key(*site) {
            problems.push(format!(
                "STALE entry {} ({} {}::{}: {}) — the refusal is gone; remove this entry from KNOWN_REFUSALS",
                site, refusal.engine, refusal.interface, refusal.method, refusal.reason
            ));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}

/// Runtime items only: a rocksdb-only test module is not an engine divergence, so test
/// modules are skipped as the refusal scan skips them.
fn count_rocksdb_only_items(root: &Path, entry: &str) -> usize {
    rust_files(root, entry)
        .iter()
        .map(|(_, path)| {
            let source = read(path);
            let lines: Vec<&str> = source.lines().collect();
            let mut count = 0;
            let mut index = 0;
            while index < lines.len() {
                if let Some(after_module) = end_of_test_module(&lines, index) {
                    index = after_module;
                    continue;
                }
                if lines[index].trim() == ROCKSDB_ONLY_ATTRIBUTE {
                    count += 1;
                }
                index += 1;
            }
            count
        })
        .sum()
}

#[test]
fn rocksdb_only_items_in_backup_and_handlers_only_fall() {
    let root = src_root();
    let mut problems = Vec::new();
    for (entry, baseline) in ROCKSDB_ONLY_ITEMS {
        let now = count_rocksdb_only_items(&root, entry);
        if now > *baseline {
            problems.push(format!(
                "{entry}: {now} `{ROCKSDB_ONLY_ATTRIBUTE}` items, baseline {baseline} — a rocksdb-only branch needs a redb counterpart or a named refusal in KNOWN_REFUSALS"
            ));
        } else if now < *baseline {
            problems.push(format!(
                "{entry}: fell {baseline} -> {now} — lower its ROCKSDB_ONLY_ITEMS entry to {now}"
            ));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}

/// The capability table is reachable from an integration test, and so are the
/// two `MigrationStore` constructors, so the flag-versus-behaviour check below runs
/// against real stores. The `PrimaryMemoryStore` helpers in `storage::test_support`
/// are `#[cfg(test)]` and crate-private, which is why no other method is probed here.
#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
mod capability_flags {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use tempfile::TempDir;
    use veld::config::StorageBackend;
    use veld::storage::legacy_rocksdb::RocksDbMigrationStore;
    use veld::storage::{MigrationStore, RedbStorageFactory, StorageCapabilities, StorageFactory};

    use super::{read, src_root};

    const ENGINES: [StorageBackend; 2] = [StorageBackend::RocksDb, StorageBackend::Redb];

    type FlagReader = fn(&StorageCapabilities) -> bool;

    /// A flag whose method can be called on both engines, so the flag's claim is
    /// checked against what the method does.
    struct ProbedFlag {
        name: &'static str,
        read: FlagReader,
        method: &'static str,
        refuses: fn(StorageBackend) -> bool,
    }

    const PROBED: &[ProbedFlag] = &[ProbedFlag {
        name: "supports_migrate_in_place",
        read: |capabilities| capabilities.supports_migrate_in_place,
        method: "MigrationStore::migrate_in_place",
        refuses: migrate_in_place_refuses,
    }];

    /// Flags that say what the engine IS rather than what it can do; they differ by
    /// design and carry no behaviour to refuse.
    const IDENTITY_FLAGS: &[&str] = &["embedded", "default_target", "legacy_compatibility"];

    /// Behavioural flags with no refusing method to call. With nothing to witness a
    /// difference, they must be equal across engines.
    const UNPROBED_BEHAVIOURAL_FLAGS: &[(&str, FlagReader)] = &[
        ("supports_prefix_scan", |c| c.supports_prefix_scan),
        ("supports_transactional_batch", |c| {
            c.supports_transactional_batch
        }),
        ("supports_snapshots", |c| c.supports_snapshots),
        ("supports_shared_multi_tenant_store", |c| {
            c.supports_shared_multi_tenant_store
        }),
    ];

    /// Flags that are true while the method refuses. The flag over-claims; the list
    /// empties by implementing the method or by setting the flag false.
    const FLAGS_THAT_OVERCLAIM: &[(&str, StorageBackend)] = &[];

    fn migrate_in_place_refuses(engine: StorageBackend) -> bool {
        let scratch = TempDir::new().unwrap();
        let target = scratch.path().join("target");
        match engine {
            StorageBackend::RocksDb => {
                let mut options = rocksdb::Options::default();
                options.create_if_missing(true);
                let db = rocksdb::DB::open(&options, scratch.path().join("source")).unwrap();
                RocksDbMigrationStore::new(Arc::new(db))
                    .migrate_in_place("probe", StorageBackend::Redb, &target)
                    .is_err()
            }
            StorageBackend::Redb => RedbStorageFactory
                .open_migration_store(&scratch.path().join("source"))
                .unwrap()
                .migrate_in_place("probe", StorageBackend::RocksDb, &target)
                .is_err(),
        }
    }

    fn capability_fields() -> BTreeSet<String> {
        let source = read(&src_root().join("storage/mod.rs"));
        let mut in_struct = false;
        let mut fields = BTreeSet::new();
        for line in source.lines() {
            if line.starts_with("pub struct StorageCapabilities") {
                in_struct = true;
            } else if in_struct && line == "}" {
                break;
            } else if in_struct {
                if let Some(field) = line.trim().strip_prefix("pub ") {
                    fields.insert(field.split(':').next().unwrap().to_string());
                }
            }
        }
        fields
    }

    #[test]
    fn every_capability_flag_is_probed_unprobed_or_identity() {
        let accounted: BTreeSet<String> = PROBED
            .iter()
            .map(|flag| flag.name)
            .chain(UNPROBED_BEHAVIOURAL_FLAGS.iter().map(|(name, _)| *name))
            .chain(IDENTITY_FLAGS.iter().copied())
            .map(str::to_string)
            .collect();
        let declared = capability_fields();
        assert!(!declared.is_empty(), "found no StorageCapabilities fields");
        assert_eq!(
            declared, accounted,
            "StorageCapabilities gained or lost a flag: classify it as PROBED (with a method that refuses), UNPROBED_BEHAVIOURAL_FLAGS or IDENTITY_FLAGS"
        );
    }

    #[test]
    fn a_flag_false_on_one_engine_is_a_refusal_on_that_engine() {
        let mut problems = Vec::new();
        for flag in PROBED {
            for engine in ENGINES {
                let claims = (flag.read)(&StorageCapabilities::for_backend(engine));
                let refuses = (flag.refuses)(engine);
                let overclaims = FLAGS_THAT_OVERCLAIM.contains(&(flag.name, engine));
                match (claims, refuses, overclaims) {
                    (false, false, _) => problems.push(format!(
                        "{engine:?}: {} is false but {} did not refuse — the flag understates the engine; set it true",
                        flag.name, flag.method
                    )),
                    (true, true, false) => problems.push(format!(
                        "{engine:?}: {} is true but {} refuses — set the flag false or add (\"{}\", StorageBackend::{engine:?}) to FLAGS_THAT_OVERCLAIM",
                        flag.name, flag.method, flag.name
                    )),
                    (true, false, true) => problems.push(format!(
                        "{engine:?}: {} now works and the flag is true — remove (\"{}\", StorageBackend::{engine:?}) from FLAGS_THAT_OVERCLAIM",
                        flag.method, flag.name
                    )),
                    (false, true, true) => problems.push(format!(
                        "{engine:?}: {} is now false — remove (\"{}\", StorageBackend::{engine:?}) from FLAGS_THAT_OVERCLAIM",
                        flag.name, flag.name
                    )),
                    _ => {}
                }
            }
        }
        assert!(problems.is_empty(), "\n{}", problems.join("\n"));
    }

    #[test]
    fn a_behavioural_flag_with_no_witness_does_not_differ_between_engines() {
        let rocksdb = StorageCapabilities::for_backend(StorageBackend::RocksDb);
        let redb = StorageCapabilities::for_backend(StorageBackend::Redb);
        let differing: Vec<&str> = UNPROBED_BEHAVIOURAL_FLAGS
            .iter()
            .filter(|(_, read)| read(&rocksdb) != read(&redb))
            .map(|(name, _)| *name)
            .collect();
        assert!(
            differing.is_empty(),
            "{differing:?} differ between engines with no refusing method to witness it: move each to PROBED with a method that refuses on the engine whose flag is false"
        );
    }
}
