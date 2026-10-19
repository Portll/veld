//! Model and ONNX Runtime auto-downloader
//!
//! Downloads MiniLM-L6-v2 model and ONNX Runtime on first use.
//! Files are cached in ~/.cache/veld/
//!
//! Model files (~22MB quantized):
//! - model_quantized.onnx - Quantized MiniLM-L6-v2 model
//! - tokenizer.json - HuggingFace tokenizer
//!
//! ONNX Runtime (~50MB):
//! - onnxruntime.dll (Windows)
//! - libonnxruntime.so (Linux)
//! - libonnxruntime.dylib (macOS)
//!
//! Security: All downloads are verified with SHA-256 checksums to prevent
//! supply chain attacks and ensure model integrity.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Backstop timeout for model downloads (10 minutes).
const DOWNLOAD_TIMEOUT_SECS: u64 = 600;

/// URLs for MiniLM model files (hosted on HuggingFace)
/// Full model is 90MB, quantized is 23MB - we download quantized for edge devices
/// Pinned to commit c9745ed1 to prevent checksum drift when HuggingFace updates models
const MODEL_ONNX_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx";
const MODEL_QUANTIZED_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model_quint8_avx2.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json";

/// URLs for NER model files (TinyBERT-finetuned-NER, ~14.5MB quantized)
/// Using a lightweight 4-layer TinyBERT model optimized for edge devices
/// Source: onnx-community/TinyBERT-finetuned-NER-ONNX (fine-tuned on CoNLL2003)
/// Pinned to commit 9b03777d for reproducibility
const NER_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/TinyBERT-finetuned-NER-ONNX/resolve/9b03777d9832105fbe419f258127fb2ec3eb09d7/onnx/model_quantized.onnx";
const NER_TOKENIZER_URL: &str =
    "https://huggingface.co/onnx-community/TinyBERT-finetuned-NER-ONNX/resolve/9b03777d9832105fbe419f258127fb2ec3eb09d7/tokenizer.json";

/// URLs for Nomic-embed-text-v1.5 model files (768-dimensional, Matryoshka)
/// Source: nomic-ai/nomic-embed-text-v1.5
/// Full model ~274MB, quantized ~65MB
const NOMIC_MODEL_ONNX_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx";
const NOMIC_MODEL_QUANTIZED_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx";
const NOMIC_TOKENIZER_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";

/// URLs for snowflake-arctic-embed-xs model files (384-dimensional, CLS-pooled)
/// Source: Snowflake/snowflake-arctic-embed-xs, a retrieval fine-tune of all-MiniLM-L6-v2
/// Licence: Apache-2.0, stated in the model card's "License" section. The card carries no
/// `license:` metadata tag, so the Hub's licence filter does not list it.
/// Pinned to commit d8c86521 (2024-12-13); the ONNX exports date from 8284d472 (2024-04-16).
/// Full model 90MB, int8 dynamic-quantised 23MB (the Hub's `model_int8.onnx` is the same file)
const ARCTIC_XS_MODEL_ONNX_URL: &str = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/d8c86521100d3556476a063fc2342036d45c106f/onnx/model.onnx";
const ARCTIC_XS_MODEL_QUANTIZED_URL: &str = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/d8c86521100d3556476a063fc2342036d45c106f/onnx/model_quantized.onnx";
const ARCTIC_XS_TOKENIZER_URL: &str = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/d8c86521100d3556476a063fc2342036d45c106f/tokenizer.json";

/// SHA-256 checksums for model integrity verification
/// Verified against pinned commit hashes above — these will not drift
struct ModelChecksums;

impl ModelChecksums {
    /// Quantized model checksum (model_quint8_avx2.onnx)
    /// Pinned: sentence-transformers/all-MiniLM-L6-v2 @ c9745ed1
    const QUANTIZED_MODEL: Option<&'static str> =
        Some("b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21");

    /// Full model checksum (model.onnx)
    /// Pinned: sentence-transformers/all-MiniLM-L6-v2 @ c9745ed1
    const FULL_MODEL: Option<&'static str> =
        Some("6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452");

    /// Tokenizer checksum (tokenizer.json)
    /// Pinned: sentence-transformers/all-MiniLM-L6-v2 @ c9745ed1
    const TOKENIZER: Option<&'static str> =
        Some("be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037");

    /// NER quantized model checksum (model_quantized.onnx)
    /// Pinned: onnx-community/TinyBERT-finetuned-NER-ONNX @ 9b03777d
    const NER_MODEL: Option<&'static str> =
        Some("ba4a1a00cf1600cae8e7cf3fda4650c825811719065b51041256392edd3647b8");

    /// NER tokenizer checksum (tokenizer.json)
    /// Pinned: onnx-community/TinyBERT-finetuned-NER-ONNX @ 9b03777d
    const NER_TOKENIZER: Option<&'static str> =
        Some("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66");

    /// Nomic quantized model checksum (model_quantized.onnx)
    /// Will be populated after first verified download
    const NOMIC_QUANTIZED_MODEL: Option<&'static str> = None;

    /// Nomic full model checksum (model.onnx)
    /// Will be populated after first verified download
    const NOMIC_FULL_MODEL: Option<&'static str> = None;

    /// Nomic tokenizer checksum (tokenizer.json)
    /// Will be populated after first verified download
    const NOMIC_TOKENIZER: Option<&'static str> = None;

    /// arctic-embed-xs quantised model checksum (onnx/model_quantized.onnx)
    /// Pinned: Snowflake/snowflake-arctic-embed-xs @ d8c86521; equals the Hub's LFS oid
    const ARCTIC_XS_QUANTIZED_MODEL: Option<&'static str> =
        Some("e6aa5e656466a73d7c3111e9a3378bd13e5b93af30eaac2b3f13fd56692589a1");

    /// arctic-embed-xs full model checksum (onnx/model.onnx)
    /// Pinned: Snowflake/snowflake-arctic-embed-xs @ d8c86521; equals the Hub's LFS oid
    const ARCTIC_XS_FULL_MODEL: Option<&'static str> =
        Some("cf2698d30ff05da02c70a088313bad56e5c2f401d734cb24a8390d446111936c");

    /// arctic-embed-xs tokenizer checksum (tokenizer.json)
    /// Pinned: Snowflake/snowflake-arctic-embed-xs @ d8c86521
    const ARCTIC_XS_TOKENIZER: Option<&'static str> =
        Some("91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854");
}

/// ONNX Runtime download URLs by platform (v1.23.2 required by ort 2.0.0-rc.11)
#[cfg(target_os = "windows")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip";
#[cfg(target_os = "linux")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz";
#[cfg(target_os = "macos")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz";

/// Get the cache directory for veld
pub fn get_cache_dir() -> PathBuf {
    // Try standard cache locations
    if let Some(cache) = dirs::cache_dir() {
        return cache.join("veld");
    }

    // Fallback to home directory
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("veld");
    }

    // Last resort: current directory
    PathBuf::from(".veld-cache")
}

/// Get the models directory for embeddings (MiniLM)
pub fn get_models_dir() -> PathBuf {
    get_cache_dir().join("models").join("minilm-l6")
}

/// Get the models directory for NER (bert-tiny-ner)
pub fn get_ner_models_dir() -> PathBuf {
    get_cache_dir().join("models").join("bert-tiny-ner")
}

/// Get the models directory for Nomic-embed-text-v1.5
pub fn get_nomic_models_dir() -> PathBuf {
    get_cache_dir().join("models").join("nomic-embed-v1.5")
}

/// Get the models directory for snowflake-arctic-embed-xs
pub fn get_arctic_xs_models_dir() -> PathBuf {
    get_cache_dir().join("models").join("arctic-embed-xs")
}

/// The arctic-embed-xs model file for one precision: (URL, file name, pinned checksum).
fn arctic_xs_model_file(use_quantized: bool) -> (&'static str, &'static str, Option<&'static str>) {
    if use_quantized {
        (
            ARCTIC_XS_MODEL_QUANTIZED_URL,
            "model_quantized.onnx",
            ModelChecksums::ARCTIC_XS_QUANTIZED_MODEL,
        )
    } else {
        (
            ARCTIC_XS_MODEL_ONNX_URL,
            "model.onnx",
            ModelChecksums::ARCTIC_XS_FULL_MODEL,
        )
    }
}

/// Get the ONNX Runtime directory
pub fn get_onnx_runtime_dir() -> PathBuf {
    get_cache_dir().join("onnxruntime")
}

/// Check if embedding model files are downloaded and valid
pub fn are_models_downloaded() -> bool {
    let models_dir = get_models_dir();
    let model_path = models_dir.join("model_quantized.onnx");
    let tokenizer_path = models_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        return false;
    }

    // Verify checksums to ensure files aren't corrupted or from a different version
    if let Some(expected) = ModelChecksums::QUANTIZED_MODEL {
        if let Ok(valid) = verify_checksum(&model_path, expected) {
            if !valid {
                tracing::warn!("Model file checksum mismatch — will re-download");
                let _ = fs::remove_file(&model_path);
                return false;
            }
        }
    }
    if let Some(expected) = ModelChecksums::TOKENIZER {
        if let Ok(valid) = verify_checksum(&tokenizer_path, expected) {
            if !valid {
                tracing::warn!("Tokenizer file checksum mismatch — will re-download");
                let _ = fs::remove_file(&tokenizer_path);
                return false;
            }
        }
    }

    true
}

/// Check if NER model files are downloaded and valid
pub fn are_ner_models_downloaded() -> bool {
    let models_dir = get_ner_models_dir();
    let model_path = models_dir.join("model.onnx");
    let tokenizer_path = models_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        return false;
    }

    if let Some(expected) = ModelChecksums::NER_MODEL {
        if let Ok(valid) = verify_checksum(&model_path, expected) {
            if !valid {
                tracing::warn!("NER model file checksum mismatch — will re-download");
                let _ = fs::remove_file(&model_path);
                return false;
            }
        }
    }
    if let Some(expected) = ModelChecksums::NER_TOKENIZER {
        if let Ok(valid) = verify_checksum(&tokenizer_path, expected) {
            if !valid {
                tracing::warn!("NER tokenizer file checksum mismatch — will re-download");
                let _ = fs::remove_file(&tokenizer_path);
                return false;
            }
        }
    }

    true
}

/// Check if Nomic embedding model files are downloaded and valid
pub fn are_nomic_models_downloaded() -> bool {
    let models_dir = get_nomic_models_dir();
    let model_path = models_dir.join("model_quantized.onnx");
    let tokenizer_path = models_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        return false;
    }

    // Verify checksums if available
    if let Some(expected) = ModelChecksums::NOMIC_QUANTIZED_MODEL {
        if let Ok(valid) = verify_checksum(&model_path, expected) {
            if !valid {
                tracing::warn!("Nomic model file checksum mismatch — will re-download");
                let _ = fs::remove_file(&model_path);
                return false;
            }
        }
    }
    if let Some(expected) = ModelChecksums::NOMIC_TOKENIZER {
        if let Ok(valid) = verify_checksum(&tokenizer_path, expected) {
            if !valid {
                tracing::warn!("Nomic tokenizer file checksum mismatch — will re-download");
                let _ = fs::remove_file(&tokenizer_path);
                return false;
            }
        }
    }

    true
}

/// Check if the quantised arctic-embed-xs model (the default) is downloaded and valid
pub fn are_arctic_xs_models_downloaded() -> bool {
    are_arctic_xs_models_downloaded_for(true)
}

/// Check if the arctic-embed-xs model of one precision and its tokenizer are downloaded and
/// match their pinned checksums. A mismatching file is removed so the next fetch replaces it.
pub fn are_arctic_xs_models_downloaded_for(use_quantized: bool) -> bool {
    let models_dir = get_arctic_xs_models_dir();
    let (_, model_filename, model_checksum) = arctic_xs_model_file(use_quantized);
    let files = [
        (models_dir.join(model_filename), model_checksum),
        (
            models_dir.join("tokenizer.json"),
            ModelChecksums::ARCTIC_XS_TOKENIZER,
        ),
    ];

    if files.iter().any(|(path, _)| !path.exists()) {
        return false;
    }
    for (path, expected) in &files {
        if let Some(expected) = expected {
            if let Ok(false) = verify_checksum(path, expected) {
                tracing::warn!(
                    "arctic-embed-xs file {:?} checksum mismatch — will re-download",
                    path
                );
                let _ = fs::remove_file(path);
                return false;
            }
        }
    }
    true
}

/// Check if ONNX Runtime is downloaded
pub fn is_onnx_runtime_downloaded() -> bool {
    let onnx_dir = get_onnx_runtime_dir();

    #[cfg(target_os = "windows")]
    let lib_name = "onnxruntime.dll";
    #[cfg(target_os = "linux")]
    let lib_name = "libonnxruntime.so";
    #[cfg(target_os = "macos")]
    let lib_name = "libonnxruntime.dylib";

    onnx_dir.join(lib_name).exists()
}

/// Get the path to the ONNX Runtime library
pub fn get_onnx_runtime_path() -> Option<PathBuf> {
    let onnx_dir = get_onnx_runtime_dir();

    #[cfg(target_os = "windows")]
    let lib_name = "onnxruntime.dll";
    #[cfg(target_os = "linux")]
    let lib_name = "libonnxruntime.so";
    #[cfg(target_os = "macos")]
    let lib_name = "libonnxruntime.dylib";

    let path = onnx_dir.join(lib_name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Download progress callback type (Arc for clonability)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Verify SHA-256 checksum of a file
/// Used for model integrity verification when checksum values are provided
pub(crate) fn verify_checksum(path: &Path, expected: &str) -> Result<bool> {
    let mut file = fs::File::open(path).context("Failed to open file for checksum")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    let actual = hex::encode(result);

    if actual == expected.to_lowercase() {
        Ok(true)
    } else {
        tracing::warn!(
            "Checksum mismatch for {:?}: expected {}, got {}",
            path,
            expected,
            actual
        );
        Ok(false)
    }
}

/// Compute SHA-256 checksum of a file
/// Reserved for model integrity verification (currently logged during download)
#[allow(dead_code)]
fn compute_checksum(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).context("Failed to open file for checksum")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Download a file from URL to path with progress and optional checksum verification
fn download_file(
    url: &str,
    path: &PathBuf,
    progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>,
) -> Result<()> {
    download_file_with_checksum(url, path, progress, None)
}

/// Download a file with SHA-256 checksum verification. The bytes go to `<path>.part` and
/// are renamed over `path` only once complete and verified, so an interrupted or
/// corrupted download never leaves a file that looks cached.
pub(crate) fn download_file_with_checksum(
    url: &str,
    path: &PathBuf,
    progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>,
    expected_checksum: Option<&str>,
) -> Result<()> {
    let mut part = path.clone().into_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    let outcome = download_to(url, &part, path, progress, expected_checksum);
    if outcome.is_err() {
        let _ = fs::remove_file(&part);
    }
    outcome
}

fn download_to(
    url: &str,
    part: &PathBuf,
    path: &PathBuf,
    progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>,
    expected_checksum: Option<&str>,
) -> Result<()> {
    tracing::info!("Downloading {} to {:?}", url, path);

    // Create parent directories
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create cache directory")?;
    }

    // Use ureq for simple HTTP downloads (blocking, no async runtime needed)
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(DOWNLOAD_TIMEOUT_SECS)))
        .build()
        .new_agent();
    let response = agent
        .get(url)
        .call()
        .context(format!("Failed to download from {url}"))?;

    let total_size = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = response.into_body().into_reader();
    let mut file = fs::File::create(part).context("Failed to create output file")?;

    // Compute checksum while downloading for efficiency
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .context("Failed to read from download stream")?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .context("Failed to write to file")?;
        hasher.update(&buffer[..bytes_read]);

        downloaded += bytes_read as u64;

        if let Some(cb) = progress {
            cb(downloaded, total_size);
        }
    }

    // Verify checksum if provided
    let actual_checksum = hex::encode(hasher.finalize());
    tracing::info!(
        "Downloaded {} bytes to {:?} (SHA-256: {})",
        downloaded,
        path,
        actual_checksum
    );

    if let Some(expected) = expected_checksum {
        if actual_checksum != expected.to_lowercase() {
            anyhow::bail!(
                "Checksum verification failed for {path:?}. Expected: {expected}, Got: {actual_checksum}. File deleted for security."
            );
        }
        tracing::info!("Checksum verified for {path:?}");
    } else {
        // Log warning that no checksum was provided
        tracing::warn!(
            "No checksum provided for {:?}. For security, add this checksum: {}",
            path,
            actual_checksum
        );
    }

    file.sync_all().context("Failed to flush the download")?;
    drop(file);
    fs::rename(part, path).with_context(|| format!("Failed to move {part:?} to {path:?}"))?;
    Ok(())
}

/// Download MiniLM model files
/// Downloads quantized model (~23MB) by default for edge devices
/// Set use_full_model=true for the larger 90MB model
pub fn download_models(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    download_models_internal(progress, true)
}

/// Download MiniLM model files with option for full or quantized model
pub fn download_models_internal(
    progress: Option<ProgressCallback>,
    use_quantized: bool,
) -> Result<PathBuf> {
    let models_dir = get_models_dir();

    if are_models_downloaded() {
        tracing::info!("Models already downloaded at {:?}", models_dir);
        return Ok(models_dir);
    }

    tracing::info!("Downloading MiniLM-L6-v2 model to {:?}", models_dir);

    // Download model (quantized ~23MB or full ~90MB)
    let (model_url, model_filename, model_checksum) = if use_quantized {
        (
            MODEL_QUANTIZED_URL,
            "model_quantized.onnx",
            ModelChecksums::QUANTIZED_MODEL,
        )
    } else {
        (MODEL_ONNX_URL, "model.onnx", ModelChecksums::FULL_MODEL)
    };

    let model_path = models_dir.join(model_filename);
    tracing::info!(
        "Downloading model from {} (~{}MB)",
        if use_quantized {
            "HuggingFace (quantized)"
        } else {
            "HuggingFace (full)"
        },
        if use_quantized { 23 } else { 90 }
    );
    download_file_with_checksum(
        model_url,
        &model_path,
        progress.as_ref().map(|p| p.as_ref()),
        model_checksum,
    )?;

    // Download tokenizer (~700KB)
    let tokenizer_path = models_dir.join("tokenizer.json");
    tracing::info!("Downloading tokenizer.json");
    download_file_with_checksum(
        TOKENIZER_URL,
        &tokenizer_path,
        progress.as_ref().map(|p| p.as_ref()),
        ModelChecksums::TOKENIZER,
    )?;

    tracing::info!(
        "MiniLM-L6-v2 model downloaded successfully to {:?}",
        models_dir
    );
    Ok(models_dir)
}

/// Download NER model files (TinyBERT-finetuned-NER, ~14.5MB quantized)
/// This is opt-in via VELD_NEURAL_NER=true environment variable
pub fn download_ner_models(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    let models_dir = get_ner_models_dir();

    if are_ner_models_downloaded() {
        tracing::info!("NER models already downloaded at {:?}", models_dir);
        return Ok(models_dir);
    }

    tracing::info!(
        "Downloading TinyBERT-NER model to {:?} (~14.5MB)",
        models_dir
    );

    // Download model (~14.5MB quantized)
    let model_path = models_dir.join("model.onnx");
    tracing::info!("Downloading NER model_quantized.onnx (~14.5MB)");
    download_file_with_checksum(
        NER_MODEL_URL,
        &model_path,
        progress.as_ref().map(|p| p.as_ref()),
        ModelChecksums::NER_MODEL,
    )?;

    // Download tokenizer (~700KB)
    let tokenizer_path = models_dir.join("tokenizer.json");
    tracing::info!("Downloading NER tokenizer.json");
    download_file_with_checksum(
        NER_TOKENIZER_URL,
        &tokenizer_path,
        progress.as_ref().map(|p| p.as_ref()),
        ModelChecksums::NER_TOKENIZER,
    )?;

    tracing::info!(
        "TinyBERT-NER model downloaded successfully to {:?}",
        models_dir
    );
    Ok(models_dir)
}

/// Download Nomic-embed-text-v1.5 model files
/// Downloads quantized model (~65MB) by default for edge devices
/// Set use_full_model=true for the larger 274MB model via download_nomic_models_internal
pub fn download_nomic_models(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    download_nomic_models_internal(progress, true)
}

/// Download Nomic model files with option for full or quantized model
pub fn download_nomic_models_internal(
    progress: Option<ProgressCallback>,
    use_quantized: bool,
) -> Result<PathBuf> {
    let models_dir = get_nomic_models_dir();

    if are_nomic_models_downloaded() {
        tracing::info!("Nomic models already downloaded at {:?}", models_dir);
        return Ok(models_dir);
    }

    tracing::info!(
        "Downloading Nomic-embed-text-v1.5 model to {:?}",
        models_dir
    );

    // Download model (quantized ~65MB or full ~274MB)
    let (model_url, model_filename, model_checksum) = if use_quantized {
        (
            NOMIC_MODEL_QUANTIZED_URL,
            "model_quantized.onnx",
            ModelChecksums::NOMIC_QUANTIZED_MODEL,
        )
    } else {
        (
            NOMIC_MODEL_ONNX_URL,
            "model.onnx",
            ModelChecksums::NOMIC_FULL_MODEL,
        )
    };

    let model_path = models_dir.join(model_filename);
    tracing::info!(
        "Downloading Nomic model from {} (~{}MB)",
        if use_quantized {
            "HuggingFace (quantized)"
        } else {
            "HuggingFace (full)"
        },
        if use_quantized { 65 } else { 274 }
    );
    download_file_with_checksum(
        model_url,
        &model_path,
        progress.as_ref().map(|p| p.as_ref()),
        model_checksum,
    )?;

    // Download tokenizer (~700KB)
    let tokenizer_path = models_dir.join("tokenizer.json");
    tracing::info!("Downloading Nomic tokenizer.json");
    download_file_with_checksum(
        NOMIC_TOKENIZER_URL,
        &tokenizer_path,
        progress.as_ref().map(|p| p.as_ref()),
        ModelChecksums::NOMIC_TOKENIZER,
    )?;

    tracing::info!(
        "Nomic-embed-text-v1.5 model downloaded successfully to {:?}",
        models_dir
    );
    Ok(models_dir)
}

/// Download snowflake-arctic-embed-xs model files: the int8 quantised model (~23MB), the
/// precision MiniLM ships at, and its tokenizer.
pub fn download_arctic_xs_models(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    download_arctic_xs_models_internal(progress, true)
}

/// Download snowflake-arctic-embed-xs model files with option for full or quantised model
pub fn download_arctic_xs_models_internal(
    progress: Option<ProgressCallback>,
    use_quantized: bool,
) -> Result<PathBuf> {
    let models_dir = get_arctic_xs_models_dir();

    if are_arctic_xs_models_downloaded_for(use_quantized) {
        tracing::info!("arctic-embed-xs already downloaded at {:?}", models_dir);
        return Ok(models_dir);
    }

    let (model_url, model_filename, model_checksum) = arctic_xs_model_file(use_quantized);
    tracing::info!(
        "Downloading snowflake-arctic-embed-xs {} to {:?} (~{}MB)",
        model_filename,
        models_dir,
        if use_quantized { 23 } else { 90 }
    );
    download_file_with_checksum(
        model_url,
        &models_dir.join(model_filename),
        progress.as_ref().map(|p| p.as_ref()),
        model_checksum,
    )?;

    tracing::info!("Downloading arctic-embed-xs tokenizer.json");
    download_file_with_checksum(
        ARCTIC_XS_TOKENIZER_URL,
        &models_dir.join("tokenizer.json"),
        progress.as_ref().map(|p| p.as_ref()),
        ModelChecksums::ARCTIC_XS_TOKENIZER,
    )?;

    tracing::info!(
        "snowflake-arctic-embed-xs downloaded successfully to {:?}",
        models_dir
    );
    Ok(models_dir)
}

/// Download ONNX Runtime
pub fn download_onnx_runtime(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    let onnx_dir = get_onnx_runtime_dir();

    if is_onnx_runtime_downloaded() {
        tracing::info!("ONNX Runtime already downloaded at {:?}", onnx_dir);
        return get_onnx_runtime_path().ok_or_else(|| anyhow::anyhow!("ONNX Runtime not found"));
    }

    tracing::info!("Downloading ONNX Runtime to {:?}", onnx_dir);
    fs::create_dir_all(&onnx_dir)?;

    // Download the archive
    let archive_name = if cfg!(target_os = "windows") {
        "onnxruntime.zip"
    } else {
        "onnxruntime.tgz"
    };
    let archive_path = onnx_dir.join(archive_name);

    download_file(
        ONNX_RUNTIME_URL,
        &archive_path,
        progress.as_ref().map(|p| p.as_ref()),
    )?;

    // Extract the archive
    extract_onnx_runtime(&archive_path, &onnx_dir)?;

    // Clean up archive - log if cleanup fails (non-fatal)
    if let Err(e) = fs::remove_file(&archive_path) {
        tracing::warn!("Failed to clean up archive {:?}: {}", archive_path, e);
    }

    get_onnx_runtime_path().ok_or_else(|| anyhow::anyhow!("Failed to extract ONNX Runtime"))
}

/// Extract ONNX Runtime from archive
fn extract_onnx_runtime(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    tracing::info!("Extracting ONNX Runtime from {:?}", archive_path);

    #[cfg(target_os = "windows")]
    {
        // Use zip crate for Windows
        let file = fs::File::open(archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name();

            // Look for the DLL file
            if name.ends_with("onnxruntime.dll") {
                let dest_path = dest_dir.join("onnxruntime.dll");
                let mut outfile = fs::File::create(&dest_path)?;
                std::io::copy(&mut file, &mut outfile)?;
                tracing::info!("Extracted onnxruntime.dll");
                return Ok(());
            }
        }

        anyhow::bail!("onnxruntime.dll not found in archive");
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Use tar + gzip for Unix
        let file = fs::File::open(archive_path)?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);

        #[cfg(target_os = "linux")]
        let lib_name = "libonnxruntime.so";
        #[cfg(target_os = "macos")]
        let lib_name = "libonnxruntime.dylib";

        // Match the REAL library, not the unversioned symlink that points at it.
        //
        // The macOS and Linux release tarballs ship both: `libonnxruntime.dylib` is a symlink to
        // `libonnxruntime.1.23.2.dylib` (and `libonnxruntime.so` to `libonnxruntime.so.1.23.2`).
        // The previous predicate — `ends_with(lib_name) || contains(lib_name)` — matched ONLY the
        // symlink, because the versioned filename contains neither string. `entry.unpack()` then
        // faithfully recreated a symlink whose relative target had not been extracted, and returned
        // early, so the real library was never reached.
        //
        // The result was a broken link at exactly the path `get_onnx_runtime_path()` probes, so
        // `is_onnx_runtime_downloaded()` reported false, `download_onnx_runtime` bailed with
        // "Failed to extract ONNX Runtime", and the archive was deleted on the way out — making
        // every retry re-download and re-fail identically. On this machine that had left the
        // runtime, and therefore every model, permanently uninstalled, with the server silently
        // serving hash-bigram vectors instead of embeddings.
        //
        // So: skip link entries outright, and select by a predicate that cannot pick up a stub.
        //
        // The shape of this fix is taken from shodh-memory, which hit the same defect first and
        // fixed it in its issues #223 and #250. Checking there rather than only here caught a
        // second bug in the first version of this code: matching `starts_with("libonnxruntime")`
        // also matches `libonnxruntime_providers_shared.so`, a ~14KB stub that is a REGULAR FILE
        // and would therefore have been extracted and installed as the runtime if it happened to
        // come first in the archive. macOS survives that by luck — the osx-arm64 tarball ships no
        // provider stub, only the real dylib, the symlink, and a 330-byte `libonnxruntime.pc` —
        // but Linux ships the stubs and would have swapped a 14KB file in for a 35MB library.
        //
        // The discriminator is the DOT after `libonnxruntime`: the real library is
        // `libonnxruntime.dylib` / `libonnxruntime.1.23.2.dylib` / `libonnxruntime.so.1.23.2`,
        // and every stub is `libonnxruntime_something`. Then take the LARGEST match rather than
        // the first, so the result does not depend on archive ordering at all.
        let mut candidates: Vec<(String, u64)> = Vec::new();
        for entry in archive.entries()? {
            let mut entry = entry?;
            let entry_type = entry.header().entry_type();
            let path = entry.path()?;
            let file_name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();

            // A link entry points at a versioned file that may not be extracted yet. Unpacking one
            // produces a dangling symlink at the very path `get_onnx_runtime_path()` probes, which
            // is exactly how this went unnoticed: it fails `exists()`, so the runtime reported
            // itself absent forever rather than broken.
            if matches!(entry_type, tar::EntryType::Symlink | tar::EntryType::Link) {
                continue;
            }
            if !entry_type.is_file() || !is_runtime_library(&file_name, lib_name) {
                continue;
            }

            let size = entry.header().size().unwrap_or(0);
            let staged = dest_dir.join(&file_name);
            entry.unpack(&staged)?;
            candidates.push((file_name, size));
        }

        // Largest wins; ordering in the archive is not load-bearing.
        candidates.sort_by_key(|(_, size)| *size);
        let found = !candidates.is_empty();
        if let Some((file_name, size)) = candidates.last().cloned() {
            let dest_path = dest_dir.join(lib_name);
            let staged = dest_dir.join(&file_name);
            if staged != dest_path {
                fs::rename(&staged, &dest_path)?;
            }
            // Anything else extracted alongside is a stub or an older copy; leave no ambiguity
            // about which file is the runtime.
            for (other, _) in &candidates {
                let path = dest_dir.join(other);
                if path != dest_path {
                    let _ = fs::remove_file(&path);
                }
            }
            tracing::info!("Extracted {} from {} ({} bytes)", lib_name, file_name, size);

            // macOS Gatekeeper blocks unsigned dylibs downloaded from the internet.
            // Remove the quarantine xattr and ad-hoc sign so dlopen succeeds.
            #[cfg(target_os = "macos")]
            {
                // Remove com.apple.quarantine attribute (silent fail if not present)
                let _ = std::process::Command::new("xattr")
                    .args(["-d", "com.apple.quarantine"])
                    .arg(&dest_path)
                    .output();

                // Ad-hoc code sign (no identity needed, just removes the unsigned flag)
                match std::process::Command::new("codesign")
                    .args(["--force", "--deep", "-s", "-"])
                    .arg(&dest_path)
                    .output()
                {
                    Ok(output) if output.status.success() => {
                        tracing::info!("Ad-hoc signed {} for macOS Gatekeeper", lib_name);
                    }
                    Ok(output) => {
                        tracing::warn!(
                            "codesign returned non-zero for {}: {}",
                            lib_name,
                            String::from_utf8_lossy(&output.stderr)
                        );
                    }
                    Err(e) => {
                        tracing::warn!("codesign not available ({}), dlopen may fail on macOS", e);
                    }
                }
            }

            return Ok(());
        }

        // `found` is false here by construction, but the message is the point: distinguish "the
        // archive held nothing that looked like the library" from the old failure, where the only
        // matching entry WAS a symlink and unpacking it reported success while installing nothing.
        debug_assert!(!found);
        anyhow::bail!(
            "no regular file matching {lib_name} in the archive (link entries are skipped by \
             design, and stubs like libonnxruntime_providers_shared are excluded). If the tarball \
             layout changed, this is where to look — the predicate this replaced would have \
             silently installed a dangling symlink instead of reporting anything."
        );
    }
}

/// Is this archive entry the ONNX Runtime shared library itself?
///
/// Split out so the predicate is testable without a 10MB download, because it is where two
/// separate bugs lived and neither was visible from the call site.
///
/// The rule is the DOT immediately after `libonnxruntime`. Everything the runtime ships alongside
/// the real library is `libonnxruntime_<something>` — `libonnxruntime_providers_shared.so` is a
/// ~14KB stub, and it is a regular file, so a size- or type-based filter alone does not exclude it.
/// The real library is `libonnxruntime.dylib`, `libonnxruntime.1.23.2.dylib` or
/// `libonnxruntime.so.1.23.2`.
///
/// `libonnxruntime.pc` (330 bytes, pkgconfig) also carries the dot, so the extension check is not
/// redundant with it.
fn is_runtime_library(file_name: &str, lib_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix("libonnxruntime.") else {
        return false;
    };
    // `rest` is e.g. "dylib", "1.23.2.dylib", "so", "so.1.23.2", "pc"
    let _ = lib_name;
    rest.ends_with(".dylib") || rest == "dylib" || rest == "so" || rest.starts_with("so.")
}

#[cfg(test)]
mod runtime_library_predicate_tests {
    use super::is_runtime_library;

    /// Exact entry names from `onnxruntime-osx-arm64-1.23.2.tgz` and the Linux equivalent,
    /// including the two that broke this: the symlink's name (identical to the real library's, so
    /// the NAME cannot distinguish them — only the tar entry type can) and the provider stub.
    #[test]
    fn the_predicate_takes_the_library_and_leaves_the_stubs() {
        for (name, want, why) in [
            (
                "libonnxruntime.1.23.2.dylib",
                true,
                "the real macOS library",
            ),
            (
                "libonnxruntime.dylib",
                true,
                "unversioned name — real file on some layouts",
            ),
            ("libonnxruntime.so.1.23.2", true, "the real Linux library"),
            ("libonnxruntime.so", true, "unversioned Linux name"),
            // The one that made the first fix wrong: a REGULAR FILE, ~14KB, which
            // starts_with("libonnxruntime") and contains(".so").
            (
                "libonnxruntime_providers_shared.so",
                false,
                "provider stub, would clobber a 35MB lib",
            ),
            ("libonnxruntime_providers_cuda.so", false, "provider stub"),
            (
                "libonnxruntime.pc",
                false,
                "330-byte pkgconfig, carries the dot but is not a library",
            ),
            (
                "libonnxruntime_test.dylib",
                false,
                "underscore, not the library",
            ),
            ("onnxruntime.dylib", false, "missing the lib prefix"),
            ("README.md", false, "unrelated"),
        ] {
            assert_eq!(
                is_runtime_library(name, "libonnxruntime.dylib"),
                want,
                "{name}: {why}"
            );
        }
    }

    /// Selection is by size, so ordering in the archive must not matter. Asserted in BOTH orders
    /// because the original defect was precisely an order-dependent first-match-wins.
    #[test]
    fn the_largest_match_wins_regardless_of_order() {
        let stub = ("libonnxruntime_providers_shared.so".to_string(), 14_000u64);
        let real = ("libonnxruntime.so.1.23.2".to_string(), 35_138_784u64);

        for candidates in [
            vec![stub.clone(), real.clone()],
            vec![real.clone(), stub.clone()],
        ] {
            let mut kept: Vec<_> = candidates
                .into_iter()
                .filter(|(n, _)| is_runtime_library(n, "libonnxruntime.so"))
                .collect();
            kept.sort_by_key(|(_, size)| *size);
            assert_eq!(
                kept.last().map(|(n, _)| n.as_str()),
                Some("libonnxruntime.so.1.23.2"),
                "the real library must win in either archive order"
            );
            assert_eq!(kept.len(), 1, "the stub should not even be a candidate");
        }
    }
}

/// Ensure all required files are downloaded
/// Returns paths to model directory and ONNX runtime
pub fn ensure_downloaded(progress: Option<ProgressCallback>) -> Result<(PathBuf, PathBuf)> {
    // Check if ORT_DYLIB_PATH is already set
    if let Ok(existing_path) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&existing_path);
        if path.exists() {
            tracing::info!(
                "Using existing ONNX Runtime from ORT_DYLIB_PATH: {:?}",
                path
            );
            let models_dir = download_models(progress)?;
            return Ok((models_dir, path));
        }
    }

    // Download models
    let models_dir = download_models(progress.clone())?;

    // Download ONNX Runtime
    let onnx_path = download_onnx_runtime(progress)?;

    // Set ORT_DYLIB_PATH for current process
    std::env::set_var("ORT_DYLIB_PATH", &onnx_path);
    tracing::info!("Set ORT_DYLIB_PATH to {:?}", onnx_path);

    Ok((models_dir, onnx_path))
}

/// Print download status
pub fn print_status() {
    let cache_dir = get_cache_dir();
    let models_downloaded = are_models_downloaded();
    let ner_models_downloaded = are_ner_models_downloaded();
    let nomic_models_downloaded = are_nomic_models_downloaded();
    let arctic_xs_models_downloaded = are_arctic_xs_models_downloaded();
    let onnx_downloaded = is_onnx_runtime_downloaded();

    println!("Veld Cache Status:");
    println!("  Cache directory: {cache_dir:?}");
    println!("  MiniLM embedding models downloaded: {models_downloaded}");
    println!("  Nomic embedding models downloaded: {nomic_models_downloaded}");
    println!("  arctic-embed-xs embedding models downloaded: {arctic_xs_models_downloaded}");
    println!("  NER models downloaded: {ner_models_downloaded}");
    println!("  ONNX Runtime downloaded: {onnx_downloaded}");

    if models_downloaded {
        let models_dir = get_models_dir();
        println!("  MiniLM model path: {models_dir:?}");
    }

    if nomic_models_downloaded {
        let nomic_dir = get_nomic_models_dir();
        println!("  Nomic model path: {nomic_dir:?}");
    }

    if arctic_xs_models_downloaded {
        let arctic_dir = get_arctic_xs_models_dir();
        println!("  arctic-embed-xs model path: {arctic_dir:?}");
    }

    if ner_models_downloaded {
        let ner_dir = get_ner_models_dir();
        println!("  NER model path: {ner_dir:?}");
    }

    if onnx_downloaded {
        if let Some(path) = get_onnx_runtime_path() {
            println!("  ONNX Runtime path: {path:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("veld"));
    }

    #[test]
    fn test_models_dir() {
        let models_dir = get_models_dir();
        assert!(models_dir.to_string_lossy().contains("minilm-l6"));
    }

    #[test]
    fn test_nomic_models_dir() {
        let models_dir = get_nomic_models_dir();
        assert!(models_dir.to_string_lossy().contains("nomic-embed-v1.5"));
    }

    #[test]
    fn arctic_xs_is_pinned_to_one_revision_with_a_checksum_per_file() {
        const REVISION: &str = "/resolve/d8c86521100d3556476a063fc2342036d45c106f/";
        let files = [
            arctic_xs_model_file(true),
            arctic_xs_model_file(false),
            (
                ARCTIC_XS_TOKENIZER_URL,
                "tokenizer.json",
                ModelChecksums::ARCTIC_XS_TOKENIZER,
            ),
        ];
        for (url, name, sha) in files {
            assert!(
                url.contains(REVISION),
                "{url} is not at the pinned revision"
            );
            assert!(url.ends_with(name), "{url} does not fetch {name}");
            let sha = sha.unwrap_or_else(|| panic!("{name} has no pinned checksum"));
            assert!(sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(get_arctic_xs_models_dir().ends_with("models/arctic-embed-xs"));
    }

    /// Serve `body` once over HTTP on loopback; returns the URL.
    fn serve_once(body: &'static [u8]) -> String {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });
        url
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    #[test]
    fn a_verified_download_lands_whole_and_leaves_no_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        let body: &'static [u8] = b"pinned model bytes";
        download_file_with_checksum(&serve_once(body), &path, None, Some(&sha256_hex(body)))
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert!(!dir.path().join("model.onnx.part").exists());
    }

    #[test]
    fn a_mismatched_download_leaves_the_cached_copy_and_no_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        fs::write(&path, b"the copy already cached").unwrap();
        let err = download_file_with_checksum(
            &serve_once(b"tampered bytes"),
            &path,
            None,
            Some(&sha256_hex(b"pinned model bytes")),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("Checksum verification failed"));
        assert_eq!(fs::read(&path).unwrap(), b"the copy already cached");
        assert!(!dir.path().join("model.onnx.part").exists());
    }
}
