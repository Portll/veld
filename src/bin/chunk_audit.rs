//! Does `chunk_text` actually produce chunks the encoder can hold?
//!
//! `chunking.rs` exists because "MiniLM has a 256 token limit. Content beyond this is silently
//! dropped." It sizes chunks in CHARACTERS — `chunk_size: 800`, commented "~200 tokens * 4
//! chars/token ... leave headroom for tokenizer differences" — and `MiniLMEmbedder::encode` then
//! truncates whatever it is handed at `max_length` with no marker and no error.
//!
//! So the chunker's correctness rests entirely on one assumption: **that text never tokenizes
//! denser than 4 characters per token.** Nothing checks it, and if it is wrong the failure is
//! exactly the one chunking was built to prevent, arriving one layer lower down and just as
//! silently. `ChunkResult::coverage_ratio()` returns a hardcoded `1.0` with the comment "With
//! chunking, we cover everything", so it cannot report the case either.
//!
//! Four chars per token is a reasonable average for English prose. This store holds file paths,
//! SHAs, JSON, slugs, log lines and code — `/Users/portll/Repositories/Portll/commitwork` is 41
//! characters and nowhere near 10 wordpieces. Whether the assumption survives THIS corpus is a
//! measurement, which is what this is.
//!
//! It reads real text, runs the real `chunk_text`, and tokenizes each chunk with the real
//! tokenizer. No estimate is involved anywhere in the answer.
//!
//! ```text
//! chunk-audit --corpus /tmp/corpus.txt --tokenizer ~/…/minilm-l6/tokenizer.json --max-length 256
//! ```
//!
//! The corpus file is record-separator (0x1E) delimited. It is NOT committed and must not be:
//! this is memory content, and an audit fixture that quotes the store discloses the store.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use veld::embeddings::chunking::{chunk_text, ChunkConfig};

#[derive(Parser)]
#[command(name = "chunk-audit", about = "Verify chunk_text output fits the encoder window")]
struct Args {
    /// 0x1E-delimited text records.
    #[arg(long)]
    corpus: PathBuf,

    /// The encoder's own tokenizer.json — the only authority on how long a chunk is.
    #[arg(long)]
    tokenizer: PathBuf,

    /// `MiniLMConfig::max_length`, the width past which `encode` silently drops tokens.
    #[arg(long, default_value_t = 256)]
    max_length: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("reading corpus {}", args.corpus.display()))?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", args.tokenizer.display()))?;

    // The instrument shared the defect. all-MiniLM-L6-v2 ships tokenizer.json with
    // `truncation: {max_length: 128}`, so the first run of this audit measured every one of 1,278
    // chunks at exactly 128 tokens and reported "0 over the window — every chunk fits". A median,
    // p90, p99 and max that are all the same number is not a distribution; it is a ruler with one
    // mark on it. Turn the tokenizer's own truncation OFF so the count is the text's length rather
    // than the tokenizer's ceiling.
    let shipped_truncation = tokenizer.get_truncation().map(|t| t.max_length);
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("disabling tokenizer truncation: {e}"))?;
    if let Some(limit) = shipped_truncation {
        println!("  NOTE: tokenizer.json ships truncation at {limit} tokens; disabled for measuring.");
        if limit < args.max_length {
            println!(
                "        That is BELOW the configured window of {}, so {limit} — not {} — is the\n\
                 \x20       real ceiling in `encode`, and `max_length` never binds.",
                args.max_length, args.max_length
            );
        }
    }

    let records: Vec<&str> = raw.split('\u{1e}').map(str::trim).filter(|s| !s.is_empty()).collect();
    anyhow::ensure!(!records.is_empty(), "corpus parsed to 0 records — that is a failed read");

    let config = ChunkConfig::default();
    let (mut chunks_total, mut over, mut records_affected, mut tokens_lost) = (0usize, 0usize, 0usize, 0usize);
    let mut lengths: Vec<usize> = Vec::new();
    let mut worst: Vec<(usize, f64, String)> = Vec::new();

    for record in &records {
        let result = chunk_text(record, &config);
        let mut hurt = false;
        for chunk in &result.chunks {
            let encoding = tokenizer
                .encode(chunk.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
            let n = encoding.get_ids().len();
            chunks_total += 1;
            lengths.push(n);
            if n > args.max_length {
                over += 1;
                hurt = true;
                tokens_lost += n - args.max_length;
                let density = chunk.chars().count() as f64 / n as f64;
                if worst.len() < 5 {
                    worst.push((n, density, chunk.chars().take(90).collect()));
                }
            }
        }
        if hurt {
            records_affected += 1;
        }
    }

    lengths.sort_unstable();
    let pct = |p: f64| lengths[((lengths.len() - 1) as f64 * p) as usize];

    // Refuse a degenerate distribution rather than reporting it. This is the check that would have
    // caught the first run: a corpus of varied prose cannot produce one token count.
    anyhow::ensure!(
        lengths.first() != lengths.last() || lengths.len() < 3,
        "every one of {} chunks measured exactly {} tokens. That is a clamp somewhere in the \
         measurement, not a property of the text, and the audit refuses to report it as a result.",
        lengths.len(),
        lengths[0]
    );

    println!("chunk-audit  (chunk_size {} chars, window {} tokens)", config.chunk_size, args.max_length);
    println!("  records            {}", records.len());
    println!("  chunks             {chunks_total}");
    println!(
        "  chunk tokens       median {}  p90 {}  p99 {}  max {}",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        lengths[lengths.len() - 1]
    );
    println!(
        "  OVER the window    {over} chunks ({:.1}%) across {records_affected} records ({:.1}%)",
        100.0 * over as f64 / chunks_total as f64,
        100.0 * records_affected as f64 / records.len() as f64
    );
    println!("  tokens dropped     {tokens_lost} — silently, with coverage_ratio() still reporting 1.0");

    if !worst.is_empty() {
        println!("\n  densest offenders (tokens, chars/token, opening):");
        for (n, density, head) in &worst {
            println!("    {n:>4} tok  {density:.2} c/t  {head}");
        }
        println!("\n  The chunker assumes 4.00 chars/token. Anything below that overflows.");
    }

    // A non-zero overflow is the finding, and a caller must not be able to read past it.
    if over > 0 {
        std::process::exit(1);
    }
    println!("\n  every chunk fits.");
    Ok(())
}
