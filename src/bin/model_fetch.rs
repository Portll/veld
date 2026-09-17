//! Fetch embedding models without going through `veld init`.
//!
//! `veld init` is the only path that downloads models, and it also creates config and generates an
//! API key. On a box that already has a running server with a key in the keychain, that is not a
//! setup step, it is a rewrite of live configuration to obtain a file. This does the one thing.
//!
//! Downloads go through `embeddings::downloader`, so URLs stay pinned to a HuggingFace commit and
//! every file is SHA-256 verified. Fetching these by hand with `curl` would skip that, which is the
//! difference between a model and an arbitrary 90MB blob that ONNX will happily execute.
//!
//! ```text
//! model-fetch --status          # what is present, decide from that
//! model-fetch --minilm          # the embedder recall actually depends on
//! model-fetch --ner --nomic
//! ```

use anyhow::Result;
use clap::Parser;
use veld::embeddings::downloader;

#[derive(Parser)]
#[command(
    name = "model-fetch",
    about = "Download veld's ONNX models, checksum-verified"
)]
struct Args {
    /// all-MiniLM-L6-v2 — the sentence embedder. Without it the server silently runs a
    /// hash-bigram fallback that produces vectors of the right shape and no semantic meaning.
    #[arg(long)]
    minilm: bool,

    /// TinyBERT NER — entity extraction. Opt-in at runtime via VELD_NEURAL_NER too.
    #[arg(long)]
    ner: bool,

    /// nomic-embed-text-v1.5 — 768-dim, 8192-token context.
    #[arg(long)]
    nomic: bool,

    /// The ONNX runtime itself. Without it every model above is an unreadable file: the embedder
    /// falls back exactly as if no model were present, so "model downloaded" is not the same claim
    /// as "the model can run".
    #[arg(long)]
    onnxruntime: bool,

    /// Report what is on disk and change nothing.
    #[arg(long)]
    status: bool,
}

fn progress() -> downloader::ProgressCallback {
    // Only on a whole-percent change. The callback fires per chunk, and a `\r` that a pipe or a
    // CI log turns into a literal is 900 lines of noise around the one line that matters.
    let last = std::sync::atomic::AtomicU64::new(u64::MAX);
    std::sync::Arc::new(move |downloaded: u64, total: u64| {
        if total == 0 {
            return;
        }
        let pct = 100 * downloaded / total;
        if last.swap(pct, std::sync::atomic::Ordering::Relaxed) != pct {
            eprint!("\r    {pct:>3}%  ({:.1} MB)", downloaded as f64 / 1e6);
            if pct == 100 {
                eprintln!();
            }
        }
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.status || !(args.minilm || args.ner || args.nomic || args.onnxruntime) {
        println!(
            "minilm  {}",
            present(
                downloader::are_models_downloaded(),
                downloader::get_models_dir()
            )
        );
        println!(
            "ner     {}",
            present(
                downloader::are_ner_models_downloaded(),
                downloader::get_ner_models_dir()
            )
        );
        println!(
            "nomic   {}",
            present(
                downloader::are_nomic_models_downloaded(),
                downloader::get_nomic_models_dir()
            )
        );
        println!(
            "onnxrt  {}",
            present(
                downloader::is_onnx_runtime_downloaded(),
                downloader::get_onnx_runtime_dir()
            )
        );
        if !args.status {
            println!("\nnothing selected; pass --minilm / --ner / --nomic to fetch.");
        }
        return Ok(());
    }

    if args.onnxruntime {
        println!("onnx runtime:");
        let path = downloader::download_onnx_runtime(Some(progress()))?;
        println!("\r    done: {}", path.display());
    }
    if args.minilm {
        println!("minilm:");
        let path = downloader::download_models(Some(progress()))?;
        println!("\r    done: {}", path.display());
    }
    if args.ner {
        println!("ner:");
        let path = downloader::download_ner_models(Some(progress()))?;
        println!("\r    done: {}", path.display());
    }
    if args.nomic {
        println!("nomic:");
        let path = downloader::download_nomic_models(Some(progress()))?;
        println!("\r    done: {}", path.display());
    }

    // Say what is on disk AFTER the fetch, from the same predicate the server uses at startup.
    // A downloader that reports its own success is reporting that it finished, not that the server
    // will find anything.
    println!();
    println!(
        "minilm  {}",
        present(
            downloader::are_models_downloaded(),
            downloader::get_models_dir()
        )
    );
    println!(
        "ner     {}",
        present(
            downloader::are_ner_models_downloaded(),
            downloader::get_ner_models_dir()
        )
    );
    println!(
        "nomic   {}",
        present(
            downloader::are_nomic_models_downloaded(),
            downloader::get_nomic_models_dir()
        )
    );
    println!(
        "onnxrt  {}",
        present(
            downloader::is_onnx_runtime_downloaded(),
            downloader::get_onnx_runtime_dir()
        )
    );
    Ok(())
}

fn present(yes: bool, path: std::path::PathBuf) -> String {
    if yes {
        format!("PRESENT  {}", path.display())
    } else {
        format!("absent   ({})", path.display())
    }
}
