//! External integrations for syncing data sources to Veld
//!
//! Supports:
//! - Linear: Issue tracking webhooks and bulk sync
//! - GitHub: PR/Issue webhooks and bulk sync

pub mod github;
pub mod linear;

pub use github::{GitHubSyncRequest, GitHubSyncResponse, GitHubWebhook, GitHubWebhookPayload};
pub use linear::{LinearSyncRequest, LinearSyncResponse, LinearWebhook, LinearWebhookPayload};
