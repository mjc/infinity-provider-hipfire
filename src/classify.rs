//! Map a hipfire HTTP failure onto [`ErrorClass`].
//!
//! Hipfire's daemon class decides. The chat HTTP body stores OpenAI's
//! envelope in `error.type` (`invalid_request_error` or `server_error`) and,
//! for a typed daemon failure, renders the class as the
//! `[class retryable=…]` prefix of `error.message`. A body that already
//! carries `error.class`, or an `error.type` that is itself a daemon class,
//! uses that field. The message text is not searched for class names.

use infinity_provider_protocol::ErrorClass;
use serde_json::Value;

const DAEMON_CLASSES: &[&str] = &[
    "context_length",
    "transient",
    "overload",
    "validation",
    "unsupported",
    "malformed",
    "cancel",
    "transport",
    "internal",
    "adaptive_poison",
    "deterministic_mismatch",
];

pub(crate) fn classify(status: u16, retry_after: bool, body: &str) -> ErrorClass {
    if let Some(class) = daemon_class(body) {
        return class_to_error(&class);
    }
    if status == 401 || status == 404 || status == 413 {
        ErrorClass::Fatal
    } else if status == 503 && retry_after {
        ErrorClass::Transient
    } else {
        ErrorClass::Fatal
    }
}

pub(crate) fn public_message(status: u16, body: &str) -> String {
    if let Some(message) = error_message(body) {
        return message;
    }
    format!("hipfire returned HTTP {status}")
}

fn class_to_error(class: &str) -> ErrorClass {
    if class == "context_length" {
        ErrorClass::ContextOverflow
    } else if class == "transient" || class == "overload" {
        ErrorClass::Transient
    } else {
        ErrorClass::Fatal
    }
}

fn daemon_class(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    if let Some(class) = error.get("class").and_then(Value::as_str)
        && is_daemon_class(class)
    {
        return Some(class.to_owned());
    }
    if let Some(class) = error.get("type").and_then(Value::as_str)
        && is_daemon_class(class)
    {
        return Some(class.to_owned());
    }
    let message = error.get("message").and_then(Value::as_str)?;
    rendered_daemon_class(message).map(str::to_owned)
}

fn rendered_daemon_class(message: &str) -> Option<&str> {
    let trimmed = message.trim_start();
    let trimmed = match trimmed.strip_prefix("daemon error:") {
        Some(rest) => rest.trim_start(),
        None => trimmed,
    };
    let inside = trimmed.strip_prefix('[')?.split_once(']')?.0;
    let mut words = inside.split_whitespace();
    let class = words.next()?;
    let marker = words.next()?;
    if !marker.starts_with("retryable=") {
        return None;
    }
    if is_daemon_class(class) {
        Some(class)
    } else {
        None
    }
}

fn is_daemon_class(class: &str) -> bool {
    DAEMON_CLASSES.contains(&class)
}

fn error_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let message = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)?
        .trim();
    if message.is_empty() {
        None
    } else {
        Some(message.to_owned())
    }
}
