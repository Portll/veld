//! ADR 0006 WP3 ratchet: uses of the rocksdb crate outside `src/storage/` may only fall.
//!
//! Upstream has no storage abstraction, so every merge from it reintroduces
//! engine-typed code. The baseline is exact rather than a ceiling: a module
//! that burns uses down must lower its number here, or the slack it leaves
//! would readmit growth unnoticed. A use is a `rocksdb::` path or a use of a
//! name imported from the crate (`count_code_uses`).

use std::collections::BTreeMap;
use std::path::Path;

const BASELINE: &[(&str, usize)] = &[
    ("backup", 39),
    ("graph_memory", 21),
    ("handlers", 24),
    ("memory", 190),
    ("memory/sleep_time", 0),
    ("user_auth", 7),
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
///
/// Each `rocksdb::` path counts, and so does each use of a name an import binds to the
/// rocksdb crate: after `use rocksdb::{DB, Options}` every `DB` and `Options` counts, and
/// after `use rocksdb as r` every `r::`. Without that, one import admits unlimited uses and an
/// alias hides them all. A name counts within the block its import sits in, the whole file
/// for a top-level import; import statements are not uses. A glob import from the crate
/// hides which names it brings in, so it is refused rather than counted.
fn count_code_uses(source: &str) -> Result<usize, String> {
    let lines: Vec<&str> = source.lines().collect();
    let code = code_lines(&lines);
    let imports = imports(&lines, &code);
    if let Some(line) = imports.globs.first() {
        return Err(format!(
            "line {line}: a glob import from rocksdb hides which names it uses; import them by name"
        ));
    }
    let mut count = 0;
    for (index, line) in lines.iter().enumerate() {
        if !code[index] {
            continue;
        }
        count += line.matches("rocksdb::").count();
        if imports.statement_lines.contains(&index) {
            continue;
        }
        let mut names: Vec<&str> = imports
            .bindings
            .iter()
            .filter(|binding| binding.scope.contains(&index))
            .map(|binding| binding.name.as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        count += names
            .iter()
            .map(|name| count_name(line, name))
            .sum::<usize>();
    }
    Ok(count)
}

/// Whether each line is runtime code: outside a top-level test module, and not a comment.
fn code_lines(lines: &[&str]) -> Vec<bool> {
    let mut code = vec![false; lines.len()];
    let mut index = 0;
    while index < lines.len() {
        if let Some(after_module) = end_of_test_module(lines, index) {
            index = after_module;
            continue;
        }
        code[index] = !lines[index].trim_start().starts_with("//");
        index += 1;
    }
    code
}

/// A name an import binds to an item of the rocksdb crate, and the lines it is in scope for.
struct Binding {
    name: String,
    scope: std::ops::Range<usize>,
}

#[derive(Default)]
struct Imports {
    bindings: Vec<Binding>,
    /// Every line of every import statement, from any crate.
    statement_lines: Vec<usize>,
    /// 1-based lines of glob imports from the rocksdb crate.
    globs: Vec<usize>,
}

fn imports(lines: &[&str], code: &[bool]) -> Imports {
    let mut found = Imports::default();
    let mut index = 0;
    while index < lines.len() {
        let trimmed = lines[index].trim_start();
        let tree_start = if code[index] {
            import_tree_start(trimmed)
        } else {
            None
        };
        let Some(tree_start) = tree_start else {
            index += 1;
            continue;
        };
        let first = index;
        let mut statement = trimmed[tree_start..].to_string();
        while !statement.contains(';') && index + 1 < lines.len() {
            index += 1;
            statement.push(' ');
            statement.push_str(lines[index].trim());
        }
        found.statement_lines.extend(first..=index);
        let tree = statement.split(';').next().unwrap_or_default();
        let (names, glob) = rocksdb_names(tree);
        if glob {
            found.globs.push(first + 1);
        }
        let scope = scope_of(lines, first);
        found.bindings.extend(names.into_iter().map(|name| Binding {
            name,
            scope: scope.clone(),
        }));
        index += 1;
    }
    found
}

/// Offset of the use tree in a `use` or `extern crate` line, with or without visibility.
fn import_tree_start(trimmed: &str) -> Option<usize> {
    let mut rest = trimmed;
    if let Some(after) = rest.strip_prefix("pub(") {
        rest = after[after.find(')')? + 1..].trim_start();
    } else if let Some(after) = rest.strip_prefix("pub ") {
        rest = after;
    }
    ["use ", "extern crate "]
        .iter()
        .find_map(|keyword| rest.strip_prefix(keyword))
        .map(|tree| trimmed.len() - tree.len())
}

/// The block an import at `first` is visible in: the whole file at the top level, otherwise
/// up to the first later line indented less, which closes the block (rustfmt is enforced).
fn scope_of(lines: &[&str], first: usize) -> std::ops::Range<usize> {
    let indent = indentation(lines[first]);
    if indent == 0 {
        return 0..lines.len();
    }
    let end = (first + 1..lines.len())
        .find(|&i| !lines[i].trim().is_empty() && indentation(lines[i]) < indent)
        .unwrap_or(lines.len());
    first..end
}

fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The names a use tree binds to paths in the rocksdb crate, and whether it globs from it.
fn rocksdb_names(tree: &str) -> (Vec<String>, bool) {
    let tokens = tokenize(tree);
    let mut found = Vec::new();
    let mut position = 0;
    parse_tree(&tokens, &mut position, &[], &mut found);
    let mut names = Vec::new();
    let mut glob = false;
    for (path, name) in found {
        if path.first().map(String::as_str) != Some("rocksdb") {
            continue;
        }
        match name {
            None => glob = true,
            Some(name) if name != "_" && name != "rocksdb" => names.push(name),
            Some(_) => {}
        }
    }
    (names, glob)
}

fn tokenize(tree: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = tree.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            tokens.push("::".to_string());
        } else if c.is_alphanumeric() || c == '_' {
            let mut ident = c.to_string();
            while let Some(&next) = chars.peek() {
                if !(next.is_alphanumeric() || next == '_') {
                    break;
                }
                ident.push(next);
                chars.next();
            }
            tokens.push(ident);
        } else {
            tokens.push(c.to_string());
        }
    }
    tokens
}

/// A binding's full path and the name it binds, or `None` for a glob.
type Bound = (Vec<String>, Option<String>);

/// One use tree at `tokens[*position]` under `prefix`.
fn parse_tree(tokens: &[String], position: &mut usize, prefix: &[String], found: &mut Vec<Bound>) {
    let Some(token) = tokens.get(*position) else {
        return;
    };
    match token.as_str() {
        "::" if prefix.is_empty() => {
            *position += 1;
            parse_tree(tokens, position, prefix, found);
        }
        "{" => {
            *position += 1;
            while let Some(token) = tokens.get(*position) {
                if token == "}" {
                    *position += 1;
                    break;
                }
                if token == "," {
                    *position += 1;
                    continue;
                }
                let before = *position;
                parse_tree(tokens, position, prefix, found);
                if *position == before {
                    *position += 1;
                }
            }
        }
        "*" => {
            *position += 1;
            found.push((prefix.to_vec(), None));
        }
        segment => {
            *position += 1;
            let mut path = prefix.to_vec();
            if segment != "self" {
                path.push(segment.to_string());
            }
            if tokens.get(*position).map(String::as_str) == Some("::") {
                *position += 1;
                parse_tree(tokens, position, &path, found);
                return;
            }
            let mut name = path.last().cloned().unwrap_or_default();
            if tokens.get(*position).map(String::as_str) == Some("as") {
                if let Some(alias) = tokens.get(*position + 1) {
                    name = alias.clone();
                }
                *position += 2;
            }
            found.push((path, Some(name)));
        }
    }
}

/// Uses of `name` in `line` as a name of its own, not a segment of some other path: a type
/// or trait counts as a whole word, a module or crate alias where it starts a path.
fn count_name(line: &str, name: &str) -> usize {
    let starts_path_only = name.starts_with(|c: char| c.is_lowercase());
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    line.match_indices(name)
        .filter(|(at, _)| {
            let before = &line[..*at];
            let after = &line[at + name.len()..];
            !before.ends_with(is_ident)
                && !before.ends_with("::")
                && !after.starts_with(is_ident)
                && (!starts_path_only || after.starts_with("::"))
        })
        .count()
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

fn scan(src: &Path, dir: &Path, counts: &mut BTreeMap<String, usize>, problems: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            scan(src, &path, counts, problems);
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
        match count_code_uses(&source) {
            Ok(0) => {}
            Ok(n) => *counts.entry(module_of(&rel)).or_default() += n,
            Err(refusal) => problems.push(format!("src/{rel}: {refusal}")),
        }
    }
}

#[test]
fn the_counter_sees_code_and_skips_comments() {
    let planted = "use rocksdb::{DB, Options};\n// rocksdb::DB in a comment\nlet x: rocksdb::Cache = y; rocksdb::DB::open();\n#[cfg(test)]\nmod tests {\n    use rocksdb::WriteBatch;\n}\nfn after() { rocksdb::DB::open(); }\n";
    assert_eq!(count_code_uses(planted), Ok(4));
    assert_eq!(module_of("memory/sleep_time/queue.rs"), "memory/sleep_time");
    assert_eq!(module_of("memory/storage.rs"), "memory");
    assert_eq!(module_of("backup.rs"), "backup");
}

#[test]
fn the_counter_follows_the_names_an_import_binds() {
    let brace = "use rocksdb::{DB, Options as RocksOptions};\nfn open() -> DB {\n    let o = RocksOptions::default();\n    DB::open(&o, rocksdb::DB::path())\n}\n";
    assert_eq!(count_code_uses(brace), Ok(5));

    let multi_line =
        "use rocksdb::{\n    ColumnFamilyDescriptor,\n    Options,\n};\nfn f(o: Options) {}\n";
    assert_eq!(count_code_uses(multi_line), Ok(2));

    let alias = "use rocksdb as r;\nextern crate rocksdb as engine;\nfn open(path: &str) -> r::DB {\n    let r = 1;\n    engine::DB::open_default(path).unwrap()\n}\n";
    assert_eq!(count_code_uses(alias), Ok(2));

    let nested = "fn a() {\n    use rocksdb::WriteBatch;\n    let b = WriteBatch::default();\n}\nfn c() -> WriteBatch { make() }\n";
    assert_eq!(count_code_uses(nested), Ok(2));

    assert!(count_code_uses("use rocksdb::backup::*;\n").is_err());
}

#[test]
fn rocksdb_outside_storage_only_falls() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut measured = BTreeMap::new();
    let mut problems = Vec::new();
    scan(&src, &src, &mut measured, &mut problems);

    let baseline: BTreeMap<String, usize> =
        BASELINE.iter().map(|(m, n)| (m.to_string(), *n)).collect();

    for module in baseline
        .keys()
        .chain(measured.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let was = baseline.get(module).copied().unwrap_or(0);
        let now = measured.get(module).copied().unwrap_or(0);
        if now > was {
            problems.push(format!(
                "{module}: {now} rocksdb uses, baseline {was} — route new storage access through the traits in src/storage/"
            ));
        } else if now < was {
            problems.push(format!(
                "{module}: fell {was} -> {now} — lower its BASELINE entry to {now}"
            ));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
