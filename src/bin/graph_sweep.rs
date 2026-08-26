//! Sweep an EXISTING entity graph for credential-shaped names.
//!
//! `src/redaction.rs` filters entities on the way IN. It cannot help with anything already stored,
//! and the store predates it. This is the remediation half: classify what is already there, and
//! emit a deletion plan.
//!
//! Two properties are load-bearing.
//!
//! **It uses `veld::redaction::classify`, the same function the entry path uses.** A sweep with its
//! own reimplemented patterns would drift from the guard and certify the guard's blind spot as a
//! clean graph. There is one classifier and this calls it.
//!
//! **It never deletes and never holds a credential.** It reads a dump and writes a plan; feeding
//! that plan to `DELETE /api/graph/entity/delete` stays a human act. Declaration is split from
//! authority, and a tool that could empty a graph unattended is a worse defect than the one it
//! remediates.
//!
//! Produce the input with:
//!
//! ```text
//! curl -sX POST $VELD/api/graph/entities/all -H "X-API-Key: $KEY" \
//!   -H 'Content-Type: application/json' -d '{"user_id":"you","limit":50000}' -o dump.json
//! curl -s $VELD/api/graph/you/stats -H "X-API-Key: $KEY"   # -> entity_count, pass as --graph-total
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use veld::redaction::{classify, is_non_semantic_id};

#[derive(Parser)]
#[command(
    name = "graph-sweep",
    about = "Classify a stored entity graph against the redaction lexicon and emit a deletion plan"
)]
struct Args {
    /// The `/api/graph/entities/all` response.
    #[arg(long)]
    dump: PathBuf,

    /// `entity_count` from `/api/graph/{user}/stats`, taken at dump time.
    ///
    /// REQUIRED, and it is the whole point. `entities/all` silently defaults to `limit: 100`, so a
    /// dump taken without one covers 0.9% of an 11k graph while looking exactly like a full one.
    /// Without this the tool cannot tell a clean graph from a clean sample, and every headline
    /// number degrades gracefully under sampling except zero.
    #[arg(long)]
    graph_total: usize,

    /// Write the deletion plan here as JSONL of `{"uuid","class"}`. Names are never written.
    #[arg(long)]
    plan: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Dump {
    entities: Vec<Entity>,
}

#[derive(Deserialize)]
struct Entity {
    uuid: String,
    name: String,
    #[serde(default)]
    mention_count: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.dump)
        .with_context(|| format!("reading dump {}", args.dump.display()))?;
    // A dump that will not parse is an unknown graph, never an empty one.
    let dump: Dump = serde_json::from_str(&raw)
        .with_context(|| format!("parsing dump {} — refusing to report an empty sweep over an unreadable file", args.dump.display()))?;

    if dump.entities.is_empty() && args.graph_total > 0 {
        bail!(
            "dump holds 0 entities but --graph-total says {}. That is a failed read, not an empty graph.",
            args.graph_total
        );
    }

    let seen = dump.entities.len();
    let complete = seen >= args.graph_total;

    let mut by_class: BTreeMap<&'static str, Vec<(&str, &str, u64)>> = BTreeMap::new();
    let mut non_semantic = 0usize;

    for e in &dump.entities {
        if let Some(class) = classify(&e.name) {
            by_class
                .entry(class.label())
                .or_default()
                .push((&e.uuid, &e.name, e.mention_count));
        }
        // Ruled ALLOWED 2026-08-26: counted for reporting, never planned for deletion.
        if is_non_semantic_id(&e.name) {
            non_semantic += 1;
        }
    }

    let flagged: usize = by_class.values().map(Vec::len).sum();

    println!("graph-sweep");
    if complete {
        println!("  coverage   {seen} of {} entities (COMPLETE)", args.graph_total);
    } else {
        println!(
            "  coverage   {seen} of {} entities — PARTIAL, {} unread",
            args.graph_total,
            args.graph_total - seen
        );
    }
    println!("  flagged    {flagged}");
    println!("  uuid-named {non_semantic} (allowed by ruling; not swept)");

    if by_class.is_empty() {
        // The distinction this whole tool exists to preserve.
        if complete {
            println!("\n  no credential-shaped entity in the graph.");
        } else {
            println!(
                "\n  no credential-shaped entity in the {seen} READ. The other {} are unexamined,\n  \
                 and this is not a statement about them.",
                args.graph_total - seen
            );
        }
    } else {
        println!();
        for (class, hits) in &by_class {
            println!("  {class:<22} {}", hits.len());
        }
        // Names are printed, once, deliberately: an operator cannot approve a deletion they cannot
        // see, and this runs on their own terminal against their own store. The PLAN file — the
        // artifact that gets passed around — carries uuids only.
        println!("\n  review before deleting (uuid  mentions  name):");
        for (class, hits) in &by_class {
            for (uuid, name, mentions) in hits {
                println!("    {class}  {uuid}  {mentions:>5}  {name}");
            }
        }
    }

    if let Some(path) = &args.plan {
        let mut w = BufWriter::new(
            File::create(path).with_context(|| format!("creating plan {}", path.display()))?,
        );
        for (class, hits) in &by_class {
            for (uuid, _name, _) in hits {
                writeln!(w, r#"{{"uuid":"{uuid}","class":"{class}"}}"#)?;
            }
        }
        w.flush()?;
        println!("\n  plan written: {} ({flagged} entries, uuids only)", path.display());
        println!("  apply it yourself; this tool does not delete.");
    }

    // Exit non-zero on an incomplete sweep so a caller cannot read "no findings" as "graph clean".
    if !complete {
        std::process::exit(2);
    }
    Ok(())
}
