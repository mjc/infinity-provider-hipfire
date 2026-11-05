//! Hipfire implementation of the Infinity [`ModelProvider`] trait.
//!
//! The process serves this provider over a Unix socket. Listing models and
//! streaming chat completions are not connected to hipfire serve yet.

use async_trait::async_trait;
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, ErrorClass, ModelEntry, ModelProvider, ModelStream,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// [`ModelProvider`] backed by a running `hipfire serve`.
pub struct HipfireProvider;

impl HipfireProvider {
    /// Create a provider from the process environment.
    ///
    /// The base URL and bearer file are not read yet. The default base URL,
    /// once wired, is `http://127.0.0.1:11435/v1`.
    pub fn from_env() -> Self {
        Self
    }
}

#[async_trait]
impl ModelProvider for HipfireProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, BoxError> {
        Ok(Vec::new())
    }

    async fn invoke_model(
        &self,
        _model_id: &str,
        _request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        Err(CompletionError::provider(
            ErrorClass::Fatal,
            "hipfire chat completions are not connected yet",
        ))
    }
}
