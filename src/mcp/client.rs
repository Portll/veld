//! HTTP clients for the shodh-memory API.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// HTTP client for the shodh-memory API (async version for MCP tools)
#[derive(Clone)]
pub(crate) struct AsyncApiClient {
    client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) user_id: String,
}

impl std::fmt::Debug for AsyncApiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncApiClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .field("user_id", &self.user_id)
            .finish()
    }
}

impl AsyncApiClient {
    pub(crate) fn new(base_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            user_id,
        }
    }

    pub(crate) async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json().await?)
    }
}

/// HTTP client for the shodh-memory API (blocking version for hooks)
#[derive(Clone)]
pub(crate) struct BlockingApiClient {
    client: reqwest::blocking::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
}

impl std::fmt::Debug for BlockingApiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingApiClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .finish()
    }
}

impl BlockingApiClient {
    pub(crate) fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url,
            api_key,
        }
    }

    pub(crate) fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json()?)
    }
}
