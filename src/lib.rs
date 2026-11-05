//! Hipfire implementation of the Infinity [`ModelProvider`] trait.
//!
//! The binary loads configuration from the environment, serves this provider
//! on a Unix socket, and prints that socket path as its first stdout line.
//! Listing and chat call a running `hipfire serve`.

mod classify;
mod config;
mod convert;
mod sse;

use async_trait::async_trait;
use futures_util::StreamExt;
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, ErrorClass, ModelEntry, ModelProvider, ModelStream,
};
use serde_json::Value;

pub use config::{Config, ConfigError};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// [`ModelProvider`] backed by a running `hipfire serve`.
pub struct HipfireProvider {
    http: reqwest::Client,
    config: Config,
}

impl HipfireProvider {
    /// Build a provider from the process environment.
    ///
    /// A missing or unreadable bearer file fails here, before any completion.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::new(Config::from_env()?)
    }

    /// Build a provider from validated configuration.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| ConfigError::new(format!("failed to build the HTTP client: {err}")))?;
        Ok(Self { http, config })
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(bearer) = &self.config.bearer {
            builder.header(reqwest::header::AUTHORIZATION, bearer.header().clone())
        } else {
            builder
        }
    }

    async fn send(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, CompletionError> {
        let request = self.authorize(builder);
        match tokio::time::timeout(self.config.timeout, request.send()).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(err)) if err.is_timeout() => Err(before_first_frame()),
            Ok(Err(err)) => Err(CompletionError::ResponseError(err.to_string())),
            Err(_elapsed) => Err(before_first_frame()),
        }
    }

    async fn get_text(&self, url: &url::Url) -> Result<(u16, bool, String), BoxError> {
        let response = self
            .send(self.http.get(url.clone()))
            .await
            .map_err(|err| err.to_string())?;
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .contains_key(reqwest::header::RETRY_AFTER);
        let bytes = response.bytes().await?;
        let body = String::from_utf8_lossy(&bytes).into_owned();
        Ok((status, retry_after, body))
    }
}

fn before_first_frame() -> CompletionError {
    CompletionError::ResponseError("timed out before the first frame".to_owned())
}

#[async_trait]
impl ModelProvider for HipfireProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, BoxError> {
        let models = self.get_text(&self.config.models_url).await?;
        let (model_status, _, model_body) = models;
        if !(200..300).contains(&model_status) {
            return Err(format!("hipfire model list returned HTTP {model_status}").into());
        }
        let models: Value = serde_json::from_str(&model_body)
            .map_err(|err| format!("hipfire model list was not JSON: {err}"))?;
        let data = models
            .get("data")
            .and_then(Value::as_array)
            .ok_or("hipfire model list has no data array")?;

        let health = self.get_text(&self.config.health_url).await;
        let (resident_model, n_ctx) = match health {
            Ok((status, _, body)) if (200..300).contains(&status) => {
                let health: Value = serde_json::from_str(&body)
                    .map_err(|err| format!("hipfire health was not JSON: {err}"))?;
                let n_ctx = health
                    .get("n_ctx")
                    .and_then(Value::as_u64)
                    .filter(|n_ctx| *n_ctx > 0)
                    .and_then(|n_ctx| usize::try_from(n_ctx).ok());
                (
                    health
                        .get("model")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    n_ctx,
                )
            }
            Ok((401 | 403 | 404, _, _)) => (None, None),
            Ok((status, _, _)) => {
                return Err(format!("hipfire health returned HTTP {status}").into());
            }
            Err(err) => return Err(format!("failed to read hipfire health: {err}").into()),
        };

        let mut entries = Vec::with_capacity(data.len());
        for (index, model) in data.iter().enumerate() {
            let Some(id) = model.get("id").and_then(Value::as_str) else {
                return Err(format!("hipfire model list entry {index} is missing an id").into());
            };
            let resident = resident_model.as_deref() == Some(id);
            let context_window = if resident {
                n_ctx.unwrap_or(self.config.default_context_window)
            } else {
                self.config.default_context_window
            };
            entries.push(ModelEntry {
                model_id: id.to_owned(),
                display_name: id.to_owned(),
                context_window,
                max_output_tokens: None,
                supports_image_input: has_image_modality(model),
            });
        }
        Ok(entries)
    }

    async fn invoke_model(
        &self,
        model_id: &str,
        request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        let body = convert::completion_body(model_id, &request, self.config.thinking)?;
        let response = self
            .send(self.http.post(self.config.chat_url.clone()).json(&body))
            .await?;
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .contains_key(reqwest::header::RETRY_AFTER);
        if !(200..300).contains(&status) {
            let body = match response.bytes().await {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(err) => {
                    return Err(CompletionError::provider(
                        ErrorClass::Transient,
                        format!("failed to read hipfire error response: {err}"),
                    ));
                }
            };
            let class = classify::classify(status, retry_after, &body);
            let message = classify::public_message(status, &body);
            return Err(CompletionError::provider(class, message));
        }
        Ok(stream_response(response, self.config.timeout))
    }
}

fn has_image_modality(model: &Value) -> bool {
    model
        .pointer("/architecture/input_modalities")
        .and_then(Value::as_array)
        .is_some_and(|modalities| {
            modalities
                .iter()
                .any(|modality| modality.as_str() == Some("image"))
        })
}

fn stream_response(response: reqwest::Response, idle_timeout: std::time::Duration) -> ModelStream {
    Box::pin(async_stream::stream! {
        let mut parser = sse::SseParser::new();
        let mut incoming = Box::pin(response.bytes_stream());
        let mut body_started = false;
        loop {
            let next = if body_started {
                match tokio::time::timeout(idle_timeout, incoming.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        yield Err(CompletionError::ResponseError("hipfire stream stalled after response bytes".to_owned()));
                        return;
                    }
                }
            } else {
                incoming.next().await
            };
            let Some(item) = next else { break };
            match item {
                Ok(bytes) => {
                    body_started = true;
                    let feed = parser.push(&bytes);
                    for chunk in feed.chunks {
                        yield Ok(chunk);
                    }
                    if let Some(error) = feed.error {
                        yield Err(error);
                        return;
                    }
                }
                Err(err) => {
                    if parser.saw_final() {
                        return;
                    }
                    yield Err(CompletionError::ResponseError(format!(
                        "hipfire stream ended before the completion finished: {err}"
                    )));
                    return;
                }
            }
        }
        let feed = parser.finish();
        for chunk in feed.chunks {
            yield Ok(chunk);
        }
        if let Some(error) = feed.error {
            yield Err(error);
        }
    })
}
