//! ADR 0006 WP3 ratchet: `rocksdb::` outside `src/storage/` may only fall.
//!
//! Upstream has no storage abstraction, so every merge from it reintroduces
//! engine-typed code. The baseline is exact rather than a ceiling: a module
//! that burns uses down must lower its number here, or the slack it leaves
//! would readmit growth unnoticed.

use std::collections::BTreeMap;
use std::path::Path;

const BASELINE: &[(&str, usize)] = &[
    ("backup", 4),
    ("graph_memory", 7),
    ("handlers", 17),
    ("memory", 57),
    ("memory/sleep_time", 0),
    ("user_auth", 4),
    ("user_memory", 0),
];

fn module_of(rel: &str) -> String {
    let parts: Vec<&str> = rel.split('/').collect();
    let key = if parts[0] == "memory" && parts.len() > 2 {
        format!("{}/{}", parts[0], parts[1])
    } else {
        parts[0].to_string()
    };
    key.trim_end_matches(".rs").to_string()
}

/// Runtime code only: a top-level `#[cfg(test)] mod` block is skipped, because a test
/// that drives a RocksDB store directly is not engine-typed code the port must remove.
fn count_code_uses(source: &str) -> usize {
    let lines: Vec<&str> = source.lines().collect();
    let mut count = 0;
    let mut index = 0;
    while index < lines.len() {
        if let Some(after_module) = end_of_test_module(&lines, index) {
            index = after_module;
            continue;
        }
        let line = lines[index];
        if !line.trim_start().starts_with("//") {
            count += line.matches("rocksdb::").count();
        }
        index += 1;
    }
    count
}

/// Index of the line after a top-level `#[cfg(test)] mod x { ... }` that starts at
/// `start`, or `None` when `start` is not the attribute of such a module.
fn end_of_test_module(lines: &[&str], start: usize) -> Option<usize> {
    let attribute = lines[start];
    if attribute != "#[cfg(test)]" && !attribute.starts_with("#[cfg(all(test") {
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

fn scan(src: &Path, dir: &Path, counts: &mut BTreeMap<String, usize>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            scan(src, &path, counts);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let rel = path
            .strip_prefix(src)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("storage/") {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let n = count_code_uses(&source);
        if n > 0 {
            *counts.entry(module_of(&rel)).or_default() += n;
        }
    }
}

#[test]
fn the_counter_sees_code_and_skips_comments() {
    let planted = "use rocksdb::{DB, Options};\n// rocksdb::DB in a comment\nlet x: rocksdb::Cache = y; rocksdb::DB::open();\n#[cfg(test)]\nmod tests {\n    use rocksdb::WriteBatch;\n}\nfn after() { rocksdb::DB::open(); }\n";
    assert_eq!(count_code_uses(planted), 4);
    assert_eq!(module_of("memory/sleep_time/queue.rs"), "memory/sleep_time");
    assert_eq!(module_of("memory/storage.rs"), "memory");
    assert_eq!(module_of("backup.rs"), "backup");
}

#[test]
fn rocksdb_outside_storage_only_falls() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut measured = BTreeMap::new();
    scan(&src, &src, &mut measured);

    let baseline: BTreeMap<String, usize> =
        BASELINE.iter().map(|(m, n)| (m.to_string(), *n)).collect();

    let mut problems = Vec::new();
    for module in baseline
        .keys()
        .chain(measured.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let was = baseline.get(module).copied().unwrap_or(0);
        let now = measured.get(module).copied().unwrap_or(0);
        if now > was {
            problems.push(format!(
                "{module}: {now} rocksdb:: uses, baseline {was} — route new storage access through the traits in src/storage/"
            ));
        } else if now < was {
            problems.push(format!(
                "{module}: fell {was} -> {now} — lower its BASELINE entry to {now}"
            ));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
