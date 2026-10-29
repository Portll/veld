//! gen-schemas — every HTTP route's request and response types, with their fields, read from
//! the handler source: `src/handlers/router.rs` names the handlers, each handler's signature
//! names its body extractor (`Json`, `LenientJson`, anything ending in `Json`), its `Query`
//! and `Path` extractors and its `Json` reply, and the structs and enums behind those names
//! are walked with their serde attributes, so the page states the wire shape a client sends
//! and receives rather than a hand-kept copy of it. `Path` tuples are paired with the route's
//! `{name}` segments, and an `f32` field whose name ends in a score word is noted as a score.
//!
//! Emits `docs/src/reference/schemas.md` for people and `schemas.json` for tools and tests.
//!
//! A type name is resolved from the file that uses it: a definition in the same file, then
//! `src/handlers/types.rs` for handler files, then the file a `use` line or a fully qualified
//! path points at, then a name unique in the crate. A name that is still ambiguous is listed
//! with every candidate so the reader is told rather than guessed for.

use anyhow::{Context, Result};
use quote::ToTokens;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{
    Attribute, Expr, ExprCall, ExprMethodCall, ExprPath, Fields, FnArg, GenericArgument, Item,
    ItemEnum, ItemFn, ItemStruct, Lit, Meta, PathArguments, ReturnType, Token, Type, UseTree,
};
use veld_docs_generators::{docs_src_root, generated_header, read_source, repo_root, write_output};

// ────────────────────────────────────────────────────────────────────────────
// Routes
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct Route {
    method: String,
    path: String,
    handler: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_params: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    path_param_names: Vec<String>,
    response: String,
}

/// The `{name}` segments of a route path, in order.
fn path_segment_names(path: &str) -> Vec<String> {
    path.split('/')
        .filter_map(|seg| {
            seg.strip_prefix('{')
                .and_then(|s| s.strip_suffix('}'))
                .map(|s| s.trim_start_matches('*').to_string())
        })
        .collect()
}

/// `Path<(A, B)>` paired with `{a}` and `{b}` renders as `{a: A, b: B}`; a lone `Path<A>`
/// pairs with the single segment. Positions without a segment keep the bare type.
fn render_path_params(rendered: &str, names: &[String]) -> String {
    let inner = rendered
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(rendered);
    let types: Vec<&str> = if rendered.starts_with('(') {
        split_top_level(inner)
    } else {
        vec![rendered]
    };
    if types.len() != names.len() {
        return rendered.to_string();
    }
    let pairs: Vec<String> = names
        .iter()
        .zip(types)
        .map(|(n, t)| format!("{n}: {}", t.trim()))
        .collect();
    format!("{{{}}}", pairs.join(", "))
}

/// Split `A, B<C, D>, E` on the commas that are not inside angle brackets or parentheses.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

struct RouteCollector {
    routes: Vec<(String, String, String)>,
    skipped: usize,
}

impl RouteCollector {
    /// `get(a)` or `get(a).post(b)…`: every (method, handler) pair in a route's second argument.
    fn method_pairs(expr: &Expr, out: &mut Vec<(String, String)>) -> bool {
        match expr {
            Expr::Call(ExprCall { func, args, .. }) => {
                let method = match func.as_ref() {
                    Expr::Path(ExprPath { path, .. }) => {
                        path.segments.last().map(|s| s.ident.to_string())
                    }
                    _ => None,
                };
                let handler = args.first().and_then(handler_path);
                match (method, handler) {
                    (Some(m), Some(h)) => {
                        out.push((m, h));
                        true
                    }
                    _ => false,
                }
            }
            Expr::MethodCall(ExprMethodCall {
                receiver,
                method,
                args,
                ..
            }) => {
                let inner = Self::method_pairs(receiver, out);
                match args.first().and_then(handler_path) {
                    Some(h) => {
                        out.push((method.to_string(), h));
                        inner
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }
}

fn handler_path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => Some(
            path.segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        ),
        _ => None,
    }
}

impl<'ast> Visit<'ast> for RouteCollector {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "route" && call.args.len() == 2 {
            let path = match &call.args[0] {
                Expr::Lit(lit) => match &lit.lit {
                    Lit::Str(s) => Some(s.value()),
                    _ => None,
                },
                _ => None,
            };
            let mut pairs = Vec::new();
            let complete = Self::method_pairs(&call.args[1], &mut pairs);
            match path {
                Some(p) if complete && !pairs.is_empty() => {
                    for (m, h) in pairs {
                        self.routes.push((m.to_uppercase(), p.clone(), h));
                    }
                }
                _ => self.skipped += 1,
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Type index
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct FieldDoc {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    required: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    doc: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct VariantDoc {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    doc: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Shape {
    Struct {
        fields: Vec<FieldDoc>,
    },
    /// A tuple struct with one field is the wire form of that field.
    Newtype {
        inner: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        refs: Vec<String>,
    },
    Enum {
        representation: String,
        variants: Vec<VariantDoc>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct TypeDoc {
    name: String,
    file: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    doc: String,
    #[serde(flatten)]
    shape: Shape,
}

/// Everything the crate defines, with the file each definition lives in and that file's
/// `use` lines, so a name is resolved the way the compiler would see it from there.
struct Index {
    defs: BTreeMap<String, Vec<TypeDoc>>,
    files: BTreeMap<String, FileScope>,
}

struct FileScope {
    defined: BTreeSet<String>,
    /// `use a::b::Name;` → Name ↦ ["a","b"]; glob imports keep the module under "*".
    imports: BTreeMap<String, Vec<String>>,
    globs: Vec<Vec<String>>,
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn doc_of(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta {
                if let Expr::Lit(lit) = &nv.value {
                    if let Lit::Str(s) = &lit.lit {
                        lines.push(s.value().trim().to_string());
                    }
                }
            }
        }
    }
    lines
        .into_iter()
        .take_while(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default, Debug)]
struct SerdeAttrs {
    rename: Option<String>,
    rename_all: Option<String>,
    default: bool,
    skip: bool,
    skip_serializing_if: bool,
    flatten: bool,
    transparent: bool,
    tag: Option<String>,
    content: Option<String>,
    untagged: bool,
}

fn serde_attrs(attrs: &[Attribute]) -> SerdeAttrs {
    let mut out = SerdeAttrs::default();
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let Ok(metas) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        else {
            continue;
        };
        for meta in metas {
            match &meta {
                Meta::Path(p) => {
                    if p.is_ident("default") {
                        out.default = true;
                    } else if p.is_ident("skip") || p.is_ident("skip_deserializing") {
                        out.skip = true;
                    } else if p.is_ident("flatten") {
                        out.flatten = true;
                    } else if p.is_ident("transparent") {
                        out.transparent = true;
                    } else if p.is_ident("untagged") {
                        out.untagged = true;
                    }
                }
                Meta::NameValue(nv) => {
                    let value = match &nv.value {
                        Expr::Lit(lit) => match &lit.lit {
                            Lit::Str(s) => Some(s.value()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if nv.path.is_ident("rename") {
                        out.rename = value;
                    } else if nv.path.is_ident("rename_all") {
                        out.rename_all = value;
                    } else if nv.path.is_ident("default") {
                        out.default = true;
                    } else if nv.path.is_ident("skip_serializing_if") {
                        out.skip_serializing_if = true;
                    } else if nv.path.is_ident("tag") {
                        out.tag = value;
                    } else if nv.path.is_ident("content") {
                        out.content = value;
                    }
                }
                Meta::List(list) => {
                    if list.path.is_ident("default") {
                        out.default = true;
                    } else if list.path.is_ident("rename") {
                        out.rename = list
                            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                            .ok()
                            .and_then(|m| {
                                m.iter().find_map(|m| match m {
                                    Meta::NameValue(nv) if nv.path.is_ident("serialize") => {
                                        match &nv.value {
                                            Expr::Lit(lit) => match &lit.lit {
                                                Lit::Str(s) => Some(s.value()),
                                                _ => None,
                                            },
                                            _ => None,
                                        }
                                    }
                                    _ => None,
                                })
                            });
                    }
                }
            }
        }
    }
    out
}

fn apply_rename_all(name: &str, rule: Option<&str>) -> String {
    let words: Vec<String> = if name.contains('_') {
        name.split('_').map(str::to_lowercase).collect()
    } else {
        let mut words = Vec::new();
        let mut current = String::new();
        for c in name.chars() {
            if c.is_uppercase() && !current.is_empty() {
                words.push(current.to_lowercase());
                current = String::new();
            }
            current.push(c);
        }
        if !current.is_empty() {
            words.push(current.to_lowercase());
        }
        words
    };
    let pascal = |w: &str| {
        let mut c = w.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    };
    match rule {
        Some("snake_case") => words.join("_"),
        Some("SCREAMING_SNAKE_CASE") => words.join("_").to_uppercase(),
        Some("kebab-case") => words.join("-"),
        Some("SCREAMING-KEBAB-CASE") => words.join("-").to_uppercase(),
        Some("lowercase") => words.concat(),
        Some("UPPERCASE") => words.concat().to_uppercase(),
        Some("PascalCase") => words.iter().map(|w| pascal(w)).collect(),
        Some("camelCase") => {
            let mut s = String::new();
            for (i, w) in words.iter().enumerate() {
                if i == 0 {
                    s.push_str(w);
                } else {
                    s.push_str(&pascal(w));
                }
            }
            s
        }
        _ => name.to_string(),
    }
}

fn type_string(ty: &Type) -> String {
    let raw = ty.to_token_stream().to_string();
    raw.replace(" < ", "<")
        .replace(" >", ">")
        .replace("< ", "<")
        .replace(" ,", ",")
        .replace(" :: ", "::")
        .replace("& ", "&")
        .replace(" ;", ";")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("( ", "(")
        .replace(" )", ")")
}

/// Named idents inside a type, with the module path written before each (empty when bare).
fn named_idents(ty: &Type, out: &mut Vec<(Vec<String>, String)>) {
    match ty {
        Type::Path(p) => {
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if let Some((last, module)) = segs.split_last() {
                let first = last.chars().next().unwrap_or('a');
                if first.is_uppercase() {
                    out.push((module.to_vec(), last.clone()));
                }
            }
            for seg in &p.path.segments {
                if let PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        if let GenericArgument::Type(inner) = arg {
                            named_idents(inner, out);
                        }
                    }
                }
            }
        }
        Type::Reference(r) => named_idents(&r.elem, out),
        Type::Tuple(t) => t.elems.iter().for_each(|e| named_idents(e, out)),
        Type::Array(a) => named_idents(&a.elem, out),
        Type::Slice(s) => named_idents(&s.elem, out),
        Type::Paren(p) => named_idents(&p.elem, out),
        Type::Group(g) => named_idents(&g.elem, out),
        _ => {}
    }
}

fn is_option(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Option"))
}

const BUILTIN: &[&str] = &[
    "String",
    "Option",
    "Vec",
    "HashMap",
    "BTreeMap",
    "HashSet",
    "BTreeSet",
    "Box",
    "Arc",
    "Value",
    "DateTime",
    "Utc",
    "Uuid",
    "Duration",
    "PathBuf",
    "Cow",
    "NaiveDate",
    "NaiveDateTime",
    "Map",
    "Number",
    "IndexMap",
    "Instant",
    "SystemTime",
    "NonZeroU32",
    "NonZeroUsize",
];

fn collect_use(tree: &UseTree, prefix: &mut Vec<String>, scope: &mut FileScope) {
    match tree {
        UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            collect_use(&p.tree, prefix, scope);
            prefix.pop();
        }
        UseTree::Name(n) => {
            scope.imports.insert(n.ident.to_string(), prefix.clone());
        }
        UseTree::Rename(r) => {
            scope.imports.insert(r.rename.to_string(), prefix.clone());
        }
        UseTree::Glob(_) => scope.globs.push(prefix.clone()),
        UseTree::Group(g) => {
            for t in &g.items {
                collect_use(t, prefix, scope);
            }
        }
    }
}

const SCORE_SUFFIXES: &[&str] = &["_score", "similarity", "confidence", "weight", "importance"];

/// An `f32` whose wire name says it is a score rather than a measurement.
fn is_score_field(wire_name: &str, ty: &str) -> bool {
    matches!(ty, "f32" | "Option<f32>") && SCORE_SUFFIXES.iter().any(|s| wire_name.ends_with(s))
}

fn describe_struct(item: &ItemStruct, file: &str) -> TypeDoc {
    let outer = serde_attrs(&item.attrs);
    let doc = doc_of(&item.attrs);
    let shape = match &item.fields {
        Fields::Named(named) => {
            let mut fields = Vec::new();
            for f in &named.named {
                let attrs = serde_attrs(&f.attrs);
                if attrs.skip {
                    continue;
                }
                let ident = f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
                let wire = attrs
                    .rename
                    .clone()
                    .unwrap_or_else(|| apply_rename_all(&ident, outer.rename_all.as_deref()));
                let mut notes = Vec::new();
                let optional = is_option(&f.ty) || attrs.default;
                if attrs.default {
                    notes.push("default when absent".to_string());
                }
                if attrs.skip_serializing_if {
                    notes.push("omitted when empty".to_string());
                }
                if attrs.flatten {
                    notes.push("flattened into the parent".to_string());
                }
                let ty = type_string(&f.ty);
                if is_score_field(&wire, &ty) {
                    notes.push("score".to_string());
                }
                let mut refs = Vec::new();
                named_idents(&f.ty, &mut refs);
                fields.push(FieldDoc {
                    name: wire,
                    ty,
                    required: !optional,
                    notes,
                    doc: doc_of(&f.attrs),
                    refs: refs.into_iter().map(|(_, n)| n).collect(),
                });
            }
            Shape::Struct { fields }
        }
        Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
            let inner = &unnamed.unnamed[0].ty;
            let mut refs = Vec::new();
            named_idents(inner, &mut refs);
            Shape::Newtype {
                inner: type_string(inner),
                refs: refs.into_iter().map(|(_, n)| n).collect(),
            }
        }
        Fields::Unnamed(unnamed) => {
            let fields = unnamed
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let mut refs = Vec::new();
                    named_idents(&f.ty, &mut refs);
                    FieldDoc {
                        name: i.to_string(),
                        ty: type_string(&f.ty),
                        required: true,
                        notes: vec!["positional".to_string()],
                        doc: String::new(),
                        refs: refs.into_iter().map(|(_, n)| n).collect(),
                    }
                })
                .collect();
            Shape::Struct { fields }
        }
        Fields::Unit => Shape::Struct { fields: Vec::new() },
    };
    TypeDoc {
        name: item.ident.to_string(),
        file: file.to_string(),
        doc,
        shape,
    }
}

fn describe_enum(item: &ItemEnum, file: &str) -> TypeDoc {
    let outer = serde_attrs(&item.attrs);
    let representation = if outer.untagged {
        "untagged: the variant's payload alone".to_string()
    } else if let (Some(tag), Some(content)) = (&outer.tag, &outer.content) {
        format!("adjacently tagged: `{{\"{tag}\": <variant>, \"{content}\": <payload>}}`")
    } else if let Some(tag) = &outer.tag {
        format!("internally tagged: the payload's fields plus `\"{tag}\": <variant>`")
    } else {
        "externally tagged: a bare string for a unit variant, `{\"<variant>\": <payload>}` otherwise".to_string()
    };
    let variants = item
        .variants
        .iter()
        .filter(|v| !serde_attrs(&v.attrs).skip)
        .map(|v| {
            let attrs = serde_attrs(&v.attrs);
            let name = attrs.rename.clone().unwrap_or_else(|| {
                apply_rename_all(&v.ident.to_string(), outer.rename_all.as_deref())
            });
            let payload = match &v.fields {
                Fields::Unit => None,
                Fields::Unnamed(u) => Some(
                    u.unnamed
                        .iter()
                        .map(|f| type_string(&f.ty))
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                Fields::Named(n) => Some(format!(
                    "{{ {} }}",
                    n.named
                        .iter()
                        .map(|f| format!(
                            "{}: {}",
                            f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                            type_string(&f.ty)
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
            };
            VariantDoc {
                name,
                payload,
                doc: doc_of(&v.attrs),
            }
        })
        .collect();
    TypeDoc {
        name: item.ident.to_string(),
        file: file.to_string(),
        doc: doc_of(&item.attrs),
        shape: Shape::Enum {
            representation,
            variants,
        },
    }
}

fn index_items(items: &[Item], file: &str, index: &mut Index) {
    index
        .files
        .entry(file.to_string())
        .or_insert_with(|| FileScope {
            defined: BTreeSet::new(),
            imports: BTreeMap::new(),
            globs: Vec::new(),
        });
    let mut nested: Vec<(String, &[Item])> = Vec::new();
    for item in items {
        match item {
            Item::Struct(s) => {
                index
                    .files
                    .get_mut(file)
                    .expect("scope exists")
                    .defined
                    .insert(s.ident.to_string());
                let doc = describe_struct(s, file);
                index.defs.entry(doc.name.clone()).or_default().push(doc);
            }
            Item::Enum(e) => {
                index
                    .files
                    .get_mut(file)
                    .expect("scope exists")
                    .defined
                    .insert(e.ident.to_string());
                let doc = describe_enum(e, file);
                index.defs.entry(doc.name.clone()).or_default().push(doc);
            }
            Item::Use(u) => {
                let scope = index.files.get_mut(file).expect("scope exists");
                collect_use(&u.tree, &mut Vec::new(), scope);
            }
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    nested.push((format!("{file}::{}", m.ident), inner));
                }
            }
            _ => {}
        }
    }
    for (name, inner) in nested {
        index_items(inner, &name, index);
    }
}

fn build_index(src: &Path, root: &Path) -> Result<Index> {
    let mut index = Index {
        defs: BTreeMap::new(),
        files: BTreeMap::new(),
    };
    for entry in walkdir::WalkDir::new(src)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
    {
        let source = read_source(entry.path())?;
        let file = rel(entry.path(), root);
        let Ok(ast) = syn::parse_file(&source) else {
            eprintln!("gen-schemas: could not parse {file}; skipped");
            continue;
        };
        index_items(&ast.items, &file, &mut index);
    }
    Ok(index)
}

/// The file a module path such as `crate::memory::types` or `super::types` lives in, seen
/// from `from`, as a prefix every candidate definition's file must start with.
fn module_prefix(module: &[String], from: &str) -> Option<String> {
    let from_dir = Path::new(from)
        .parent()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let mut parts: Vec<String> = Vec::new();
    let mut iter = module.iter().peekable();
    let mut base = match iter.peek().map(|s| s.as_str()) {
        Some("crate") => {
            iter.next();
            "src".to_string()
        }
        Some("super") => {
            iter.next();
            Path::new(&from_dir)
                .parent()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|| "src".to_string())
        }
        Some("self") => {
            iter.next();
            from_dir.clone()
        }
        Some(_) => from_dir.clone(),
        None => return None,
    };
    if base.is_empty() {
        base = "src".to_string();
    }
    for seg in iter {
        parts.push(seg.clone());
    }
    Some(if parts.is_empty() {
        base
    } else {
        format!("{base}/{}", parts.join("/"))
    })
}

/// `name (file)`: the identity of a definition while the walk runs, before it is known
/// which names are unique among the definitions reached.
fn qualified_key(def: &TypeDoc) -> String {
    format!(
        "{} ({})",
        def.name,
        def.file.trim_start_matches("src/").trim_end_matches(".rs")
    )
}

/// Re-key the reached definitions for publication: a name is plain when it is unique among
/// the types on the page, and `name (file)` only when two reached definitions share it. A
/// twin elsewhere in the crate that no route reaches does not qualify the name.
fn publish(reached: BTreeMap<String, TypeDoc>) -> BTreeMap<String, TypeDoc> {
    let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
    for def in reached.values() {
        *by_name.entry(def.name.clone()).or_default() += 1;
    }
    reached
        .into_values()
        .map(|def| {
            let key = if by_name[&def.name] == 1 {
                def.name.clone()
            } else {
                qualified_key(&def)
            };
            (key, def)
        })
        .collect()
}

impl Index {

    fn resolve(&self, module: &[String], name: &str, from: &str) -> Resolution {
        if BUILTIN.contains(&name) {
            return Resolution::Builtin;
        }
        let Some(candidates) = self.defs.get(name) else {
            return Resolution::Unknown;
        };
        if candidates.len() == 1 {
            return Resolution::Found(candidates[0].clone());
        }
        let by_prefix = |prefix: &str| -> Option<TypeDoc> {
            let hits: Vec<&TypeDoc> = candidates
                .iter()
                .filter(|c| {
                    let f = c.file.trim_end_matches(".rs");
                    let f = f.trim_end_matches("/mod");
                    f == prefix
                        || c.file.starts_with(&format!("{prefix}/"))
                        || f.starts_with(prefix)
                })
                .collect();
            if hits.len() == 1 {
                Some(hits[0].clone())
            } else {
                None
            }
        };
        if !module.is_empty() {
            if let Some(prefix) = module_prefix(module, from) {
                if let Some(hit) = by_prefix(&prefix) {
                    return Resolution::Found(hit);
                }
            }
        }
        if let Some(same) = candidates.iter().find(|c| c.file == from) {
            return Resolution::Found(same.clone());
        }
        if let Some(scope) = self.files.get(from) {
            if let Some(path) = scope.imports.get(name) {
                if let Some(prefix) = module_prefix(path, from) {
                    if let Some(hit) = by_prefix(&prefix) {
                        return Resolution::Found(hit);
                    }
                }
            }
            for glob in &scope.globs {
                if let Some(prefix) = module_prefix(glob, from) {
                    if let Some(hit) = by_prefix(&prefix) {
                        return Resolution::Found(hit);
                    }
                }
            }
        }
        if from.starts_with("src/handlers/") {
            if let Some(hit) = candidates
                .iter()
                .find(|c| c.file == "src/handlers/types.rs")
            {
                return Resolution::Found(hit.clone());
            }
        }
        Resolution::Ambiguous(candidates.clone())
    }
}

enum Resolution {
    Found(TypeDoc),
    Ambiguous(Vec<TypeDoc>),
    Builtin,
    Unknown,
}

// ────────────────────────────────────────────────────────────────────────────
// Handlers
// ────────────────────────────────────────────────────────────────────────────

struct HandlerShape {
    body: Option<Type>,
    query: Option<Type>,
    path: Option<Type>,
    response: Option<Type>,
    opaque: bool,
    file: String,
}

fn last_segment(ty: &Type) -> Option<(String, Vec<Type>)> {
    if let Type::Path(p) = ty {
        let seg = p.path.segments.last()?;
        let args = match &seg.arguments {
            PathArguments::AngleBracketed(a) => a
                .args
                .iter()
                .filter_map(|a| match a {
                    GenericArgument::Type(t) => Some(t.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        return Some((seg.ident.to_string(), args));
    }
    None
}

fn find_fn<'a>(items: &'a [Item], name: &str) -> Option<&'a ItemFn> {
    for item in items {
        match item {
            Item::Fn(f) if f.sig.ident == name => return Some(f),
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    if let Some(f) = find_fn(inner, name) {
                        return Some(f);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn handler_shape(root: &Path, handler: &str) -> Result<Option<HandlerShape>> {
    let mut parts: Vec<&str> = handler.split("::").collect();
    let fn_name = parts.pop().context("handler has a function name")?;
    let module = parts.join("/");
    let candidates = [
        root.join("src/handlers").join(format!("{module}.rs")),
        root.join("src/handlers").join(&module).join("mod.rs"),
    ];
    let Some(file) = candidates.iter().find(|p| p.exists()) else {
        return Ok(None);
    };
    let source = read_source(file)?;
    let ast = syn::parse_file(&source).with_context(|| format!("parsing {}", file.display()))?;
    let Some(func) = find_fn(&ast.items, fn_name) else {
        return Ok(None);
    };
    let mut shape = HandlerShape {
        body: None,
        query: None,
        path: None,
        response: None,
        opaque: false,
        file: rel(file, root),
    };
    for input in &func.sig.inputs {
        if let FnArg::Typed(pt) = input {
            if let Some((name, args)) = last_segment(&pt.ty) {
                match (name.as_str(), args.first()) {
                    (n, Some(t)) if n.ends_with("Json") => shape.body = Some(t.clone()),
                    ("Query", Some(t)) => shape.query = Some(t.clone()),
                    ("Path", Some(t)) => shape.path = Some(t.clone()),
                    _ => {}
                }
            }
        }
    }
    match &func.sig.output {
        ReturnType::Default => {}
        ReturnType::Type(_, ty) => {
            let mut current: Type = (**ty).clone();
            if let Some((name, args)) = last_segment(&current) {
                if name == "Result" {
                    if let Some(ok) = args.first() {
                        current = ok.clone();
                    }
                }
            }
            if let Type::Tuple(t) = &current {
                if let Some(inner) = t
                    .elems
                    .iter()
                    .find(|e| last_segment(e).is_some_and(|(n, _)| n == "Json"))
                {
                    current = inner.clone();
                }
            }
            match last_segment(&current) {
                Some((name, args)) if name == "Json" => shape.response = args.first().cloned(),
                Some((name, _)) if name == "Response" => shape.opaque = true,
                _ => {
                    if matches!(current, Type::ImplTrait(_)) {
                        shape.opaque = true;
                    } else if !matches!(current, Type::Tuple(ref t) if t.elems.is_empty()) {
                        shape.response = Some(current.clone());
                    }
                }
            }
        }
    }
    Ok(Some(shape))
}

// ────────────────────────────────────────────────────────────────────────────
// Walk and render
// ────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Output {
    generated_from: Vec<&'static str>,
    routes: Vec<Route>,
    types: BTreeMap<String, TypeDoc>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    ambiguous: BTreeMap<String, Vec<String>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    handlers_not_found: Vec<String>,
}

struct Walker<'a> {
    index: &'a Index,
    types: BTreeMap<String, TypeDoc>,
    ambiguous: BTreeMap<String, Vec<String>>,
}

impl<'a> Walker<'a> {
    /// Resolve every named ident in `ty` from `from`, record the definitions reached, and
    /// return the type as written with each resolved name linked by its published key.
    fn visit_type(&mut self, ty: &Type, from: &str) -> String {
        let mut idents = Vec::new();
        named_idents(ty, &mut idents);
        for (module, name) in idents {
            self.visit_name(&module, &name, from);
        }
        type_string(ty)
    }

    fn visit_name(&mut self, module: &[String], name: &str, from: &str) {
        match self.index.resolve(module, name, from) {
            Resolution::Found(def) => {
                let key = qualified_key(&def);
                if self.types.contains_key(&key) {
                    return;
                }
                self.types.insert(key, def.clone());
                let file = def.file.clone();
                match &def.shape {
                    Shape::Struct { fields } => {
                        for f in fields {
                            for r in &f.refs {
                                self.visit_name(&[], r, &file);
                            }
                        }
                    }
                    Shape::Newtype { refs, .. } => {
                        for r in refs {
                            self.visit_name(&[], r, &file);
                        }
                    }
                    Shape::Enum { variants, .. } => {
                        for v in variants {
                            if let Some(payload) = &v.payload {
                                if let Ok(t) = syn::parse_str::<Type>(payload) {
                                    let mut inner = Vec::new();
                                    named_idents(&t, &mut inner);
                                    for (m, n) in inner {
                                        self.visit_name(&m, &n, &file);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Resolution::Ambiguous(cands) => {
                self.ambiguous
                    .entry(name.to_string())
                    .or_insert_with(|| cands.iter().map(|c| c.file.clone()).collect());
            }
            Resolution::Builtin | Resolution::Unknown => {}
        }
    }
}

fn anchor(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// `text` as a code span when none of its names is a documented type, otherwise the names
/// linked to their sections with the rest escaped, since a link cannot live inside a code span.
fn link_names(text: &str, keys: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    let mut word = String::new();
    let mut linked = false;
    let flush = |word: &mut String, out: &mut String, linked: &mut bool| {
        if word.is_empty() {
            return;
        }
        match keys.get(word.as_str()) {
            Some(key) => {
                out.push_str(&format!("[`{}`](#{})", word, anchor(key)));
                *linked = true;
            }
            None => out.push_str(word),
        }
        word.clear();
    };
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut out, &mut linked);
            match c {
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '|' => out.push_str("\\|"),
                _ => out.push(c),
            }
        }
    }
    flush(&mut word, &mut out, &mut linked);
    if linked {
        out
    } else {
        format!("`{}`", text.replace('|', "\\|"))
    }
}

fn group_of(path: &str) -> String {
    match path.strip_prefix("/api/") {
        Some(rest) => rest
            .split('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("misc")
            .to_string(),
        None => path
            .trim_start_matches('/')
            .split('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("misc")
            .to_string(),
    }
}

fn render_markdown(out: &Output, skipped: usize) -> String {
    let mut md = generated_header(
        "src/handlers/router.rs and src/handlers/*.rs",
        "gen-schemas",
    );
    md.push_str("# HTTP API schemas\n\n");
    md.push_str(&format!(
        "The request and response shape of every HTTP route, read from the handler signatures and the \
         types they name: **{}** routes over **{}** types. `schemas.json` beside this page holds the same \
         facts for tools and tests.\n\n",
        out.routes.len(),
        out.types.len()
    ));
    md.push_str(
        "**How to read a type.** *Required* means the field has no `Option` and no serde default, so a \
         request without it is rejected and a response always carries it. A field marked *omitted when \
         empty* is left out of a response when it has no value. Enum variants are listed in wire form. \
         `DateTime<Utc>` is an RFC 3339 string, `Uuid` and `MemoryId` are UUID strings, and `Value` is \
         any JSON. A response shown as *opaque* is built by hand in the handler; read the handler for it.\n\n",
    );
    if skipped > 0 {
        md.push_str(&format!(
            "> {skipped} route registration(s) use a non-literal path or an unrecognised method chain and are not listed; see src/handlers/router.rs.\n\n"
        ));
    }
    if !out.handlers_not_found.is_empty() {
        md.push_str(&format!(
            "> Handlers the generator could not locate: {}.\n\n",
            out.handlers_not_found.join(", ")
        ));
    }
    if !out.ambiguous.is_empty() {
        md.push_str(
            "> Names defined more than once that no `use` line disambiguated, left unlinked:\n",
        );
        for (name, files) in &out.ambiguous {
            md.push_str(&format!(">   `{name}`: {}\n", files.join(", ")));
        }
        md.push('\n');
    }

    // name → published key for linking
    let mut keys: BTreeMap<String, String> = BTreeMap::new();
    for (key, def) in &out.types {
        keys.entry(def.name.clone()).or_insert_with(|| key.clone());
    }

    md.push_str("## Routes\n\n");
    let mut groups: BTreeMap<String, Vec<&Route>> = BTreeMap::new();
    for r in &out.routes {
        groups.entry(group_of(&r.path)).or_default().push(r);
    }
    for (group, routes) in &groups {
        md.push_str(&format!("### /{group}\n\n"));
        md.push_str("| Method | Path | Request | Response |\n|---|---|---|---|\n");
        let mut sorted = routes.clone();
        sorted.sort_by(|a, b| a.path.cmp(&b.path).then(a.method.cmp(&b.method)));
        for r in sorted {
            let mut request = Vec::new();
            if let Some(b) = &r.body {
                request.push(format!("body {}", link_names(b, &keys)));
            }
            if let Some(q) = &r.query {
                request.push(format!("query {}", link_names(q, &keys)));
            }
            if let Some(p) = &r.path_params {
                request.push(format!("path {}", link_names(p, &keys)));
            }
            let request = if request.is_empty() {
                "none".to_string()
            } else {
                request.join("; ")
            };
            md.push_str(&format!(
                "| `{}` | `{}` | {} | {} |\n",
                r.method,
                r.path,
                request,
                link_names(&r.response, &keys)
            ));
        }
        md.push('\n');
    }

    md.push_str("## Types\n\n");
    for (key, def) in &out.types {
        md.push_str(&format!("### {key}\n\n"));
        md.push_str(&format!("Defined in `{}`.", def.file));
        if !def.doc.is_empty() {
            md.push_str(&format!(" {}", def.doc.replace('|', "\\|")));
        }
        md.push_str("\n\n");
        match &def.shape {
            Shape::Struct { fields } if fields.is_empty() => {
                md.push_str("An empty object.\n\n");
            }
            Shape::Struct { fields } => {
                md.push_str("| Field | Type | Required | Notes |\n|---|---|---|---|\n");
                for f in fields {
                    let mut notes = f.notes.clone();
                    if !f.doc.is_empty() {
                        notes.push(f.doc.clone());
                    }
                    md.push_str(&format!(
                        "| `{}` | {} | {} | {} |\n",
                        f.name,
                        link_names(&f.ty, &keys),
                        if f.required { "yes" } else { "no" },
                        notes.join("; ").replace('|', "\\|")
                    ));
                }
                md.push('\n');
            }
            Shape::Newtype { inner, .. } => {
                md.push_str(&format!(
                    "On the wire this is its inner value, {}.\n\n",
                    link_names(inner, &keys)
                ));
            }
            Shape::Enum {
                representation,
                variants,
            } => {
                md.push_str(&format!("Enum, {representation}.\n\n"));
                md.push_str("| Variant | Payload | Notes |\n|---|---|---|\n");
                for v in variants {
                    md.push_str(&format!(
                        "| `{}` | {} | {} |\n",
                        v.name,
                        v.payload
                            .as_ref()
                            .map(|p| link_names(p, &keys))
                            .unwrap_or_else(|| "none".to_string()),
                        v.doc.replace('|', "\\|")
                    ));
                }
                md.push('\n');
            }
        }
    }
    md.push_str(
        "---\n\n*Generated from the handler source; a change to a request or response type reaches this page on the next docs build, and `tests/mcp_tool_roundtrip.rs` is where a tool is held to it.*\n",
    );
    md
}

fn main() -> Result<()> {
    let root = repo_root();
    let router_path = root.join("src/handlers/router.rs");
    let source = read_source(&router_path)?;
    let file =
        syn::parse_file(&source).with_context(|| format!("parsing {}", router_path.display()))?;
    let mut collector = RouteCollector {
        routes: Vec::new(),
        skipped: 0,
    };
    collector.visit_file(&file);

    let index = build_index(&root.join("src"), &root)?;
    let mut walker = Walker {
        index: &index,
        types: BTreeMap::new(),
        ambiguous: BTreeMap::new(),
    };
    let mut routes = Vec::new();
    let mut not_found = Vec::new();
    let mut seen = BTreeSet::new();
    for (method, path, handler) in &collector.routes {
        if !seen.insert((method.clone(), path.clone())) {
            continue;
        }
        let shape = handler_shape(&root, handler)?;
        let Some(shape) = shape else {
            not_found.push(handler.clone());
            routes.push(Route {
                method: method.clone(),
                path: path.clone(),
                handler: handler.clone(),
                body: None,
                query: None,
                path_params: None,
                path_param_names: Vec::new(),
                response: "unknown".to_string(),
            });
            continue;
        };
        let from = shape.file.clone();
        let body = shape.body.as_ref().map(|t| walker.visit_type(t, &from));
        let query = shape.query.as_ref().map(|t| walker.visit_type(t, &from));
        let path_param_names = path_segment_names(path);
        let path_params = shape
            .path
            .as_ref()
            .map(|t| render_path_params(&walker.visit_type(t, &from), &path_param_names));
        let response = match (&shape.response, shape.opaque) {
            (Some(t), _) => walker.visit_type(t, &from),
            (None, true) => "opaque".to_string(),
            (None, false) => "no body".to_string(),
        };
        routes.push(Route {
            method: method.clone(),
            path: path.clone(),
            handler: handler.clone(),
            body,
            query,
            path_params,
            path_param_names,
            response,
        });
    }
    routes.sort_by(|a, b| a.path.cmp(&b.path).then(a.method.cmp(&b.method)));

    let output = Output {
        generated_from: vec!["src/handlers/router.rs", "src/handlers/*.rs", "src/**/*.rs"],
        routes,
        types: publish(walker.types),
        ambiguous: walker.ambiguous,
        handlers_not_found: not_found,
    };
    let reference = docs_src_root().join("reference");
    write_output(
        &reference.join("schemas.md"),
        &render_markdown(&output, collector.skipped),
    )?;
    let json = serde_json::to_string_pretty(&output)? + "\n";
    write_output(&reference.join("schemas.json"), &json)?;
    eprintln!(
        "gen-schemas: {} routes, {} types, {} ambiguous names, {} handlers not found",
        output.routes.len(),
        output.types.len(),
        output.ambiguous.len(),
        output.handlers_not_found.len()
    );
    Ok(())
}
