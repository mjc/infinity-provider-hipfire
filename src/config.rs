//! Process configuration for a `hipfire serve` endpoint.
//!
//! `HIPFIRE_BASE_URL` defaults to `http://127.0.0.1:11435/v1`. A LAN address
//! is operator configuration. `HIPFIRE_BEARER_FILE`, when set, is a file whose
//! contents are the bearer token; unset means the request is unauthenticated.
//! `HIPFIRE_TIMEOUT_SECS` bounds connect and response headers.
//! `HIPFIRE_DEFAULT_CONTEXT` is the context window for every model that is
//! not the resident one. `HIPFIRE_THINKING` is `true` or `false` (also `1`,
//! `0`, `on`, `off`, `yes`, `no`); unset leaves thinking off.

use std::fmt;
use std::fs;
use std::path::Path;
use std::time::Duration;

use reqwest::header::HeaderValue;
use url::Url;

/// Time allowed for connect and for response headers.
///
/// Hipfire can hold the status line for 15 seconds of silence on a cold load,
/// and an idle completion on the Tali serve returned in under 10 seconds.
/// Thirty seconds covers both. The streamed body is not under this budget.
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Context window used when `/health` does not report a resident `n_ctx`.
pub(crate) const DEFAULT_CONTEXT_WINDOW: usize = 8192;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435/v1";

/// Failure while reading provider configuration. The message names a path
/// and never the bearer file's contents.
#[derive(Debug)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

/// Bearer token read from a file. Debug output is redacted.
#[derive(Clone)]
pub(crate) struct Bearer {
    header: HeaderValue,
}

impl Bearer {
    fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(ConfigError::new(format!(
                    "bearer file {} is missing",
                    path.display()
                )));
            }
            Err(_) => {
                return Err(ConfigError::new(format!(
                    "bearer file {} is unreadable",
                    path.display()
                )));
            }
        };
        // Hipfire trims trailing whitespace and sends the rest as a header value.
        let token = text.trim_end();
        let header = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
            ConfigError::new(format!(
                "bearer file {} is not a usable token",
                path.display()
            ))
        })?;
        Ok(Self { header })
    }

    pub(crate) fn header(&self) -> &HeaderValue {
        &self.header
    }
}

impl fmt::Debug for Bearer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Bearer([redacted])")
    }
}

/// Validated endpoint and request policy.
#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) health_url: Url,
    pub(crate) models_url: Url,
    pub(crate) chat_url: Url,
    pub(crate) bearer: Option<Bearer>,
    pub(crate) timeout: Duration,
    pub(crate) default_context_window: usize,
    pub(crate) thinking: bool,
}

impl Config {
    /// Read configuration from the process environment.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_vars(|key| std::env::var(key).ok())
    }

    /// Read configuration from an explicit environment lookup.
    pub fn from_vars(mut vars: impl FnMut(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let base =
            non_blank(vars("HIPFIRE_BASE_URL")).unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let timeout = match non_blank(vars("HIPFIRE_TIMEOUT_SECS")) {
            Some(text) => parse_timeout(&text)?,
            None => DEFAULT_TIMEOUT,
        };
        let default_context_window = match non_blank(vars("HIPFIRE_DEFAULT_CONTEXT")) {
            Some(text) => parse_context(&text)?,
            None => DEFAULT_CONTEXT_WINDOW,
        };
        let thinking = match non_blank(vars("HIPFIRE_THINKING")) {
            Some(text) => parse_thinking(&text)?,
            None => false,
        };
        let bearer = match non_blank(vars("HIPFIRE_BEARER_FILE")) {
            Some(path) => Some(Bearer::from_file(Path::new(&path))?),
            None => None,
        };
        Self::from_parts(&base, bearer, timeout, default_context_window, thinking)
    }

    fn from_parts(
        base: &str,
        bearer: Option<Bearer>,
        timeout: Duration,
        default_context_window: usize,
        thinking: bool,
    ) -> Result<Self, ConfigError> {
        let mut base_url = Url::parse(base)
            .map_err(|_| ConfigError::new(format!("HIPFIRE_BASE_URL is not a URL: {base}")))?;
        if base_url.scheme() != "http" {
            return Err(ConfigError::new("HIPFIRE_BASE_URL must be an http URL"));
        }
        if base_url.host().is_none() {
            return Err(ConfigError::new("HIPFIRE_BASE_URL must include a host"));
        }
        if !base_url.username().is_empty() || base_url.password().is_some() {
            return Err(ConfigError::new(
                "HIPFIRE_BASE_URL must not include a username or password",
            ));
        }
        base_url.set_query(None);
        base_url.set_fragment(None);
        if !base_url.path().ends_with('/') {
            let path = format!("{}/", base_url.path());
            base_url.set_path(&path);
        }

        let models_url = base_url
            .join("models")
            .map_err(|_| ConfigError::new("HIPFIRE_BASE_URL cannot address /v1/models"))?;
        let chat_url = base_url.join("chat/completions").map_err(|_| {
            ConfigError::new("HIPFIRE_BASE_URL cannot address /v1/chat/completions")
        })?;
        // `/health` is on the origin, not under the `/v1` prefix.
        let mut health_url = base_url;
        health_url.set_path("/health");

        Ok(Self {
            health_url,
            models_url,
            chat_url,
            bearer,
            timeout,
            default_context_window,
            thinking,
        })
    }
}

fn non_blank(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    })
}

fn parse_timeout(text: &str) -> Result<Duration, ConfigError> {
    let seconds: u64 = text.parse().map_err(|_| {
        ConfigError::new("HIPFIRE_TIMEOUT_SECS must be a positive number of seconds")
    })?;
    if seconds == 0 {
        return Err(ConfigError::new(
            "HIPFIRE_TIMEOUT_SECS must be a positive number of seconds",
        ));
    }
    Ok(Duration::from_secs(seconds))
}

fn parse_context(text: &str) -> Result<usize, ConfigError> {
    let tokens: usize = text
        .parse()
        .map_err(|_| ConfigError::new("HIPFIRE_DEFAULT_CONTEXT must be a positive integer"))?;
    if tokens == 0 {
        return Err(ConfigError::new(
            "HIPFIRE_DEFAULT_CONTEXT must be a positive integer",
        ));
    }
    Ok(tokens)
}

fn parse_thinking(text: &str) -> Result<bool, ConfigError> {
    match text.to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Ok(true),
        "false" | "0" | "off" | "no" => Ok(false),
        _ => Err(ConfigError::new("HIPFIRE_THINKING must be true or false")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_loopback_and_leave_thinking_off() {
        let config = Config::from_vars(|_| None).unwrap();
        assert_eq!(
            config.chat_url.as_str(),
            "http://127.0.0.1:11435/v1/chat/completions"
        );
        assert_eq!(
            config.models_url.as_str(),
            "http://127.0.0.1:11435/v1/models"
        );
        assert_eq!(config.health_url.as_str(), "http://127.0.0.1:11435/health");
        assert_eq!(config.timeout, DEFAULT_TIMEOUT);
        assert_eq!(config.default_context_window, DEFAULT_CONTEXT_WINDOW);
        assert!(!config.thinking);
        assert!(config.bearer.is_none());
    }

    #[test]
    fn bearer_file_is_redacted_and_a_missing_file_names_only_the_path() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-provider-config-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let secret = "SENTINEL-config-bearer-do-not-print";
        let path = dir.join("key");
        fs::write(&path, format!("{secret}\n")).unwrap();

        let config = Config::from_vars(|key| match key {
            "HIPFIRE_BEARER_FILE" => Some(path.display().to_string()),
            _ => None,
        })
        .unwrap();
        let rendered = format!("{config:?}");
        assert!(rendered.contains("redacted"));
        assert!(!rendered.contains(secret));

        let missing = dir.join("absent");
        let err = Config::from_vars(|key| match key {
            "HIPFIRE_BEARER_FILE" => Some(missing.display().to_string()),
            _ => None,
        })
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("missing"));
        assert!(message.contains(&missing.display().to_string()));
        assert!(!message.contains(secret));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unreadable_bearer_file_does_not_echo_its_contents() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-provider-config-unreadable-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let secret = "SENTINEL-unreadable-bearer-do-not-print";
        let path = dir.join("key");
        fs::write(&path, secret).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o0);
        }
        fs::set_permissions(&path, perms).unwrap();
        assert!(
            fs::read_to_string(&path).is_err(),
            "mode 000 should hide the bearer file"
        );

        let err = Config::from_vars(|key| match key {
            "HIPFIRE_BEARER_FILE" => Some(path.display().to_string()),
            _ => None,
        })
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unreadable"));
        assert!(!message.contains(secret));
        let _ = fs::set_permissions(&path, readable());
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    fn readable() -> std::fs::Permissions {
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(0o644)
    }

    #[test]
    fn invalid_switches_fail_without_a_default() {
        let timeout = Config::from_vars(|key| match key {
            "HIPFIRE_TIMEOUT_SECS" => Some("0".into()),
            _ => None,
        })
        .unwrap_err();
        assert!(timeout.to_string().contains("HIPFIRE_TIMEOUT_SECS"));

        let thinking = Config::from_vars(|key| match key {
            "HIPFIRE_THINKING" => Some("maybe".into()),
            _ => None,
        })
        .unwrap_err();
        assert!(thinking.to_string().contains("HIPFIRE_THINKING"));
    }
}
