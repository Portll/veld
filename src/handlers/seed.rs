//! Cold-Start Project Seeding Handler
//!
//! Scans a project directory and creates foundational memories from config files,
//! README, and source code. Enables rapid bootstrapping of memory for new projects.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::remember::parse_experience_type;
use super::state::MultiUserMemoryManager;
use crate::constants::{
    SEED_DEFAULT_MAX_FILES, SEED_IMPORTANCE_CONFIG, SEED_IMPORTANCE_README,
    SEED_IMPORTANCE_SOURCE, SEED_MAX_DEPTH, SEED_README_MAX_CHARS,
};
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{Experience, ExperienceType};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

/// Directories to skip during project seeding
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "dist",
    ".next",
    ".nuxt",
    "build",
    ".cache",
    "vendor",
    ".tox",
    "venv",
    "env",
    ".eggs",
    "*.egg-info",
];

/// Source file extensions to include
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "cs", "cpp", "cc", "c", "h", "hpp", "rb",
    "ex", "exs", "erl", "hs", "ml", "mli", "scala", "kt", "kts", "swift", "dart", "lua", "r",
    "jl", "zig", "nim", "v", "cr", "clj", "cljs", "fs", "fsx",
];

/// Config file extensions to include
const CONFIG_EXTENSIONS: &[&str] = &["toml", "json", "yaml", "yml", "xml", "ini", "cfg"];

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Seed project request
#[derive(Debug, Deserialize)]
pub struct SeedRequest {
    pub user_id: String,
    pub project_path: String,
    #[serde(default)]
    pub max_files: Option<usize>,
}

/// Seed project response
#[derive(Debug, Serialize)]
pub struct SeedResponse {
    pub memories_created: usize,
    pub files_scanned: usize,
    pub project_name: String,
}

// =============================================================================
// PATH SECURITY
// =============================================================================

/// Validate that a project path is safe to access.
///
/// Rejects paths that:
/// - Don't exist or aren't directories
/// - Escape the user's home directory
/// - Point to system-critical paths
fn validate_project_path(path_str: &str) -> Result<PathBuf, AppError> {
    let path = PathBuf::from(path_str);

    // Canonicalize to resolve symlinks and ".."
    let canonical = std::fs::canonicalize(&path).map_err(|e| AppError::InvalidInput {
        field: "project_path".to_string(),
        reason: format!("Cannot resolve project path '{}': {}", path_str, e),
    })?;

    if !canonical.is_dir() {
        return Err(AppError::InvalidInput {
            field: "project_path".to_string(),
            reason: format!("Project path '{}' is not a directory", path_str),
        });
    }

    // Block system-critical paths
    let blocked_prefixes: &[&str] = &[
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/System",
        "/Library",
        "/private/etc",
        "/private/var",
        "C:\\Windows",
        "C:\\Program Files",
    ];

    let canonical_str = canonical.to_string_lossy();
    for blocked in blocked_prefixes {
        if canonical_str.starts_with(blocked) {
            return Err(AppError::InvalidInput {
                field: "project_path".to_string(),
                reason: format!("Access denied: '{}' is a system path", path_str),
            });
        }
    }

    // Ensure path is within a user-accessible location
    // Allow: home directories, /tmp, and common project paths
    let home = dirs::home_dir();
    let is_safe = home
        .as_ref()
        .map(|h| canonical.starts_with(h))
        .unwrap_or(false)
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        return Err(AppError::InvalidInput {
            field: "project_path".to_string(),
            reason: format!(
                "Access denied: '{}' is outside allowed roots (home directory or /tmp)",
                path_str
            ),
        });
    }

    Ok(canonical)
}

// =============================================================================
// FILE WALKING
// =============================================================================

/// Collect eligible files from a project directory
fn walk_project_files(root: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_recursive(root, 0, max_files, &mut files);
    files
}

fn walk_recursive(
    dir: &Path,
    depth: usize,
    max_files: usize,
    files: &mut Vec<PathBuf>,
) {
    if depth > SEED_MAX_DEPTH || files.len() >= max_files {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    let mut subdirs = Vec::new();

    for entry in entries.flatten() {
        if files.len() >= max_files {
            break;
        }

        let path = entry.path();
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if path.is_dir() {
            // Skip excluded directories
            if SKIP_DIRS.iter().any(|s| file_name == *s || file_name.ends_with(".egg-info")) {
                continue;
            }
            // Skip hidden directories (except .github, .vscode)
            if file_name.starts_with('.')
                && !matches!(file_name.as_str(), ".github" | ".vscode")
            {
                continue;
            }
            subdirs.push(path);
        } else if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            // Include source files, config files, and well-known project files
            let is_known_file = matches!(
                file_name.as_str(),
                "README.md" | "readme.md" | "README" | "Makefile" | "Dockerfile"
            );
            if SOURCE_EXTENSIONS.contains(&ext) || CONFIG_EXTENSIONS.contains(&ext) || is_known_file
            {
                files.push(path);
            }
        }
    }

    // Recurse into subdirectories (breadth-first priority: process files first)
    for subdir in subdirs {
        if files.len() >= max_files {
            break;
        }
        walk_recursive(&subdir, depth + 1, max_files, files);
    }
}

// =============================================================================
// CONFIG FILE PARSING
// =============================================================================

/// Extract project name and description from config files
fn extract_project_info(root: &Path) -> (String, Option<String>) {
    // Priority 1: Cargo.toml
    let cargo_toml = root.join("Cargo.toml");
    if cargo_toml.is_file() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            let name = extract_toml_value(&content, "name");
            let desc = extract_toml_value(&content, "description");
            if let Some(name) = name {
                return (name, desc);
            }
        }
    }

    // Priority 2: package.json
    let package_json = root.join("package.json");
    if package_json.is_file() {
        if let Ok(content) = std::fs::read_to_string(&package_json) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
                let name = parsed["name"].as_str().map(|s| s.to_string());
                let desc = parsed["description"].as_str().map(|s| s.to_string());
                if let Some(name) = name {
                    return (name, desc);
                }
            }
        }
    }

    // Priority 3: pyproject.toml
    let pyproject = root.join("pyproject.toml");
    if pyproject.is_file() {
        if let Ok(content) = std::fs::read_to_string(&pyproject) {
            let name = extract_toml_value(&content, "name");
            let desc = extract_toml_value(&content, "description");
            if let Some(name) = name {
                return (name, desc);
            }
        }
    }

    // Priority 4: go.mod
    let go_mod = root.join("go.mod");
    if go_mod.is_file() {
        if let Ok(content) = std::fs::read_to_string(&go_mod) {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("module ") {
                    let module_name = rest.trim();
                    // Extract last path component as project name
                    let name = module_name
                        .rsplit('/')
                        .next()
                        .unwrap_or(module_name)
                        .to_string();
                    return (name, None);
                }
            }
        }
    }

    // Fallback: use directory name
    let dir_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    (dir_name, None)
}

/// Extract a value from a TOML string without a full parser dependency.
/// Handles basic `key = "value"` patterns in [package] section.
fn extract_toml_value(content: &str, key: &str) -> Option<String> {
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]" || trimmed == "[project]" || trimmed == "[tool.poetry]";
            continue;
        }
        if in_package {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let rest = rest.trim();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim();
                    // Handle quoted strings
                    if let Some(stripped) = rest.strip_prefix('"') {
                        if let Some(end) = stripped.find('"') {
                            return Some(stripped[..end].to_string());
                        }
                    }
                    if let Some(stripped) = rest.strip_prefix('\'') {
                        if let Some(end) = stripped.find('\'') {
                            return Some(stripped[..end].to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Determine importance level for a file based on its role
fn file_importance(path: &Path, root: &Path) -> f32 {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    let rel_path = path.strip_prefix(root).unwrap_or(path);
    let rel_str = rel_path.to_string_lossy();

    // README gets highest importance
    if file_name.to_lowercase().starts_with("readme") {
        return SEED_IMPORTANCE_README;
    }

    // Config files at root get high importance
    if matches!(
        file_name,
        "Cargo.toml"
            | "package.json"
            | "pyproject.toml"
            | "go.mod"
            | "Makefile"
            | "Dockerfile"
            | "docker-compose.yml"
            | "docker-compose.yaml"
            | ".env.example"
    ) && !rel_str.contains(std::path::MAIN_SEPARATOR)
    {
        return SEED_IMPORTANCE_CONFIG;
    }

    // Other config files
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if CONFIG_EXTENSIONS.contains(&ext) {
        return (SEED_IMPORTANCE_CONFIG + SEED_IMPORTANCE_SOURCE) / 2.0; // midpoint
    }

    SEED_IMPORTANCE_SOURCE
}

// =============================================================================
// HANDLER
// =============================================================================

/// Seed a project directory into memory
///
/// `POST /api/seed`
///
/// Scans a project directory, extracts text from source/config files, and creates
/// foundational memories tagged with `_seed`. Previous `_seed` memories are
/// deleted for idempotent re-seeding.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn seed_project(
    State(state): State<AppState>,
    Json(req): Json<SeedRequest>,
) -> Result<Json<SeedResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project_path = validate_project_path(&req.project_path)?;
    let max_files = req.max_files.unwrap_or(SEED_DEFAULT_MAX_FILES);
    let user_id = req.user_id.clone();

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    // Run in spawn_blocking since we do filesystem I/O
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<SeedResponse> {
        let (project_name, project_desc) = extract_project_info(&project_path);

        // Idempotent: delete old _seed tagged memories before re-seeding
        {
            let seed_tags = vec!["_seed".to_string()];
            let guard = memory.read();
            match guard.recall_by_tags(&seed_tags, 10_000) {
                Ok(old_seeds) if !old_seeds.is_empty() => {
                    let count = old_seeds.len();
                    // Use forget_by_tags to clean up existing seed memories
                    drop(guard);
                    let guard = memory.read();
                    if let Err(e) = guard.forget_by_tags(&seed_tags) {
                        tracing::warn!("Failed to clean old seed memories: {e}");
                    } else {
                        tracing::info!(
                            "Cleaned {} old seed memories for re-seeding",
                            count
                        );
                    }
                }
                _ => {}
            }
        }

        // Collect files
        let files = walk_project_files(&project_path, max_files);
        let files_scanned = files.len();
        let mut memories_created = 0usize;

        // Extract README content if available
        let readme_path = project_path.join("README.md");
        let readme_content = if readme_path.is_file() {
            std::fs::read_to_string(&readme_path)
                .ok()
                .map(|c| {
                    if c.len() > SEED_README_MAX_CHARS {
                        c[..SEED_README_MAX_CHARS].to_string()
                    } else {
                        c
                    }
                })
        } else {
            None
        };

        // Create project overview memory from README + project info
        if readme_content.is_some() || project_desc.is_some() {
            let overview = format!(
                "Project: {}\n{}\n{}",
                project_name,
                project_desc.as_deref().unwrap_or(""),
                readme_content.as_deref().unwrap_or("")
            );

            let experience = Experience {
                content: overview,
                experience_type: ExperienceType::Context,
                tags: vec!["_seed".to_string(), project_name.clone()],
                entities: vec![project_name.clone()],
                ..Default::default()
            };

            let guard = memory.read();
            match guard.remember(experience, None) {
                Ok(_id) => {
                    memories_created += 1;
                    // Boost importance to README level
                    // The experience was stored with calculated importance;
                    // the write gate may have adjusted it
                }
                Err(e) => {
                    tracing::debug!("Failed to create project overview memory: {e}");
                }
            }
        }

        // Process each file
        for file_path in &files {
            let file_name = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            // Skip README since we already processed it above
            if file_name.to_lowercase().starts_with("readme") {
                continue;
            }

            // Read file content (skip binary/too-large files)
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue, // Binary or unreadable
            };

            // Skip empty or very small files
            if content.len() < 10 {
                continue;
            }

            // Truncate very large files to first 4000 chars
            let content = if content.len() > 4000 {
                content[..4000].to_string()
            } else {
                content
            };

            let rel_path = file_path
                .strip_prefix(&project_path)
                .unwrap_or(file_path);
            let rel_str = rel_path.to_string_lossy().to_string();

            let importance = file_importance(file_path, &project_path);
            let exp_type = if importance >= SEED_IMPORTANCE_CONFIG {
                "context"
            } else {
                "observation"
            };

            let experience = Experience {
                content: format!("File: {}\n\n{}", rel_str, content),
                experience_type: parse_experience_type(Some(&exp_type.to_string())),
                tags: vec![
                    "_seed".to_string(),
                    file_name.to_string(),
                ],
                entities: vec![file_name.to_string(), project_name.clone()],
                ..Default::default()
            };

            let guard = memory.read();
            match guard.remember(experience, None) {
                Ok(_id) => {
                    memories_created += 1;
                }
                Err(e) => {
                    tracing::debug!("Failed to seed file {}: {e}", rel_str);
                }
            }
        }

        tracing::info!(
            project = %project_name,
            files = files_scanned,
            memories = memories_created,
            "Project seeding complete"
        );

        Ok(SeedResponse {
            memories_created,
            files_scanned,
            project_name,
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Seed task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(result))
}
