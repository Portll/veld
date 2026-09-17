//! A `.rs` file under `src/` that no `mod` declaration reaches is never compiled,
//! so every test and change inside it passes silently. `recall_harness/runner.rs`
//! collected tests and fixes for weeks that had never built.
//!
//! The baseline is exact: declaring a module fails this until its entry is removed.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

const KNOWN_UNDECLARED: &[&str] = &[
    "recall_harness/bridge_harness.rs",
    "recall_harness/decay_sim.rs",
    "recall_harness/forgetting_harness.rs",
    "recall_harness/lineage_harness.rs",
    "recall_harness/multihop.rs",
    "recall_harness/ontology_harness.rs",
    "recall_harness/runner.rs",
    "recall_harness/temporal_harness.rs",
];

fn declared_modules(source: &str) -> Vec<String> {
    source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .filter_map(|l| {
            let mut t = l.trim_start();
            if let Some(rest) = t.strip_prefix("pub") {
                let rest = rest.trim_start();
                t = match rest.strip_prefix('(') {
                    Some(r) => r.split_once(')')?.1.trim_start(),
                    None => rest,
                };
            }
            let name = t.strip_prefix("mod ")?.trim_start();
            let name = name.strip_suffix(';')?.trim_end();
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
                .then(|| name.to_string())
        })
        .collect()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn target_roots(manifest_dir: &Path, src: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("Cargo.toml");
    let mut roots: Vec<PathBuf> = manifest
        .lines()
        .filter_map(|l| l.trim().strip_prefix("path"))
        .filter_map(|r| r.trim_start().strip_prefix('='))
        .map(|r| r.trim().trim_matches('"'))
        .filter(|p| p.starts_with("src/"))
        .map(|p| manifest_dir.join(p))
        .collect();
    // Cargo's autobins: src/main.rs and src/bin/*.rs are targets without a manifest entry.
    roots.push(src.join("main.rs"));
    if let Ok(entries) = std::fs::read_dir(src.join("bin")) {
        roots.extend(entries.flatten().map(|e| e.path()));
    }
    roots.retain(|p| p.is_file());
    roots
}

fn child_dir(file: &Path) -> PathBuf {
    let stem = file.file_stem().unwrap().to_str().unwrap();
    let parent = file.parent().unwrap();
    let is_root_like = matches!(stem, "lib" | "main" | "mod")
        || parent.file_name().and_then(|n| n.to_str()) == Some("bin");
    if is_root_like {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    }
}

fn unreachable(manifest_dir: &Path, src: &Path) -> BTreeSet<String> {
    let mut files = Vec::new();
    rs_files(src, &mut files);
    let roots = target_roots(manifest_dir, src);

    let mut seen: HashSet<PathBuf> = roots.iter().cloned().collect();
    let mut stack = roots;
    while let Some(file) = stack.pop() {
        let source = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let dir = child_dir(&file);
        for name in declared_modules(&source) {
            for candidate in [
                dir.join(format!("{name}.rs")),
                dir.join(&name).join("mod.rs"),
            ] {
                if candidate.is_file() && seen.insert(candidate.clone()) {
                    stack.push(candidate);
                }
            }
        }
    }

    files
        .into_iter()
        .filter(|f| !seen.contains(f))
        .map(|f| {
            f.strip_prefix(src)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

#[test]
fn the_walker_finds_a_planted_orphan_and_follows_every_declaration_form() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("Cargo.toml"), "[lib]\npath = \"src/lib.rs\"\n").unwrap();
    let src = root.join("src");
    std::fs::create_dir_all(src.join("a/b")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "pub mod a;\n// mod commented;\npub(crate) mod c;\n",
    )
    .unwrap();
    std::fs::write(src.join("a/mod.rs"), "mod b;\n").unwrap();
    std::fs::write(src.join("a/b/mod.rs"), "").unwrap();
    std::fs::write(src.join("c.rs"), "").unwrap();
    std::fs::write(src.join("commented.rs"), "").unwrap();
    std::fs::write(src.join("a/orphan.rs"), "").unwrap();

    let found: Vec<String> = unreachable(root, &src).into_iter().collect();
    assert_eq!(
        found,
        vec!["a/orphan.rs".to_string(), "commented.rs".to_string()]
    );
}

#[test]
fn every_source_file_is_compiled_or_known() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let found = unreachable(manifest_dir, &manifest_dir.join("src"));
    let known: BTreeSet<String> = KNOWN_UNDECLARED.iter().map(|s| s.to_string()).collect();

    let new: Vec<_> = found.difference(&known).collect();
    let now_declared: Vec<_> = known.difference(&found).collect();
    assert!(
        new.is_empty() && now_declared.is_empty(),
        "\nnever compiled (declare with `mod`, or delete): {new:?}\nnow declared, remove from KNOWN_UNDECLARED: {now_declared:?}"
    );
}
