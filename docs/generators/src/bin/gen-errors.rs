//! gen-errors — parse `src/errors.rs` and emit a markdown reference table of
//! every `AppError` variant with its doc comment.
//!
//! Strategy: parse via syn, find the `enum AppError` (or any `pub enum`
//! ending in `Error`) and walk its variants. For each variant, capture the
//! `///` doc comment and any `#[error("...")]` attribute (from `thiserror`).

use anyhow::Result;
use syn::{Attribute, Expr, ExprLit, Fields, Item, ItemEnum, Lit, Meta};

use veld_docs_generators::{
    docs_src_root, generated_header, read_source, repo_root, write_output,
};

#[derive(Debug, Default)]
struct ErrorVariant {
    name: String,
    doc: String,
    error_msg: Option<String>,
    has_fields: bool,
}

fn extract_doc(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &attr.meta {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
            {
                lines.push(s.value().trim().to_string());
            }
        }
    }
    lines.join(" ")
}

fn extract_error_msg(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("error") {
            continue;
        }
        // #[error("...")] — pull the string literal out.
        if let Ok(list) = attr.meta.require_list() {
            let tokens = list.tokens.to_string();
            // crude but works: find the first quoted segment.
            if let Some(start) = tokens.find('"') {
                if let Some(end) = tokens[start + 1..].find('"') {
                    return Some(tokens[start + 1..start + 1 + end].to_string());
                }
            }
        }
    }
    None
}

fn collect_variants(en: &ItemEnum) -> Vec<ErrorVariant> {
    en.variants
        .iter()
        .map(|v| ErrorVariant {
            name: v.ident.to_string(),
            doc: extract_doc(&v.attrs),
            error_msg: extract_error_msg(&v.attrs),
            has_fields: !matches!(v.fields, Fields::Unit),
        })
        .collect()
}

fn main() -> Result<()> {
    let errors_path = repo_root().join("src").join("errors.rs");
    let source = read_source(&errors_path)?;
    let file = syn::parse_file(&source)?;

    let mut all_variants: Vec<(String, Vec<ErrorVariant>)> = Vec::new();
    for item in &file.items {
        if let Item::Enum(en) = item {
            if en.ident.to_string().ends_with("Error") {
                all_variants.push((en.ident.to_string(), collect_variants(en)));
            }
        }
    }

    if all_variants.is_empty() {
        anyhow::bail!(
            "no `*Error` enum found in {} — the structure may have changed",
            errors_path.display()
        );
    }

    let mut out = generated_header("src/errors.rs", "gen-errors");
    out.push_str("# Error Reference\n\n");
    out.push_str(&format!(
        "The error types defined in `src/errors.rs`. {} error enum(s) discovered.\n\n",
        all_variants.len()
    ));

    for (name, variants) in &all_variants {
        out.push_str(&format!("## `{name}`\n\n"));
        out.push_str(&format!(
            "{} variants.\n\n",
            variants.len()
        ));
        out.push_str("| Variant | Message / Doc |\n|---|---|\n");
        for v in variants {
            let msg = v
                .error_msg
                .as_deref()
                .map(|m| format!("`{}`", m.replace('|', "\\|")))
                .unwrap_or_else(|| v.doc.replace('|', "\\|"));
            let display_msg = if msg.trim().is_empty() {
                "—".to_string()
            } else {
                msg
            };
            let fields_suffix = if v.has_fields { " *(carries data)*" } else { "" };
            out.push_str(&format!("| `{}{}` | {} |\n", v.name, fields_suffix, display_msg));
        }
        out.push('\n');
    }
    out.push_str("---\n\n*HTTP status mappings live in the `IntoResponse` impl on `AppError`. Inspect [src/errors.rs](https://github.com/Portll/veld/blob/main/src/errors.rs) for the full mapping.*\n");

    let output_path = docs_src_root().join("reference").join("errors.md");
    write_output(&output_path, &out)?;
    let total: usize = all_variants.iter().map(|(_, v)| v.len()).sum();
    eprintln!("gen-errors: extracted {} variants across {} enums", total, all_variants.len());
    Ok(())
}
