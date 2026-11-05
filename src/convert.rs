//! Infinity chat messages as an OpenAI chat-completions body.

use std::fmt;

use infinity_provider_protocol::{
    AssistantContent, CompletionError, CompletionRequest, Image, ImageSource, Message, ToolResult,
    ToolResultContent, UserContent,
};
use serde_json::{Map, Value};

pub(crate) fn completion_body(
    model_id: &str,
    request: &CompletionRequest,
    thinking: bool,
) -> Result<Value, CompletionError> {
    let mut body = Map::new();
    if let Some(extra) = request
        .additional_params
        .as_ref()
        .filter(|value| !value.is_null())
    {
        let Some(extra) = extra.as_object() else {
            return Err(request_error("additional_params must be a JSON object"));
        };
        body.extend(extra.clone());
    }

    let mut messages = Vec::new();
    if let Some(preamble) = &request.preamble {
        messages.push(serde_json::json!({
            "role": "system",
            "content": preamble,
        }));
    }
    for message in &request.chat_history {
        push_message(&mut messages, message)?;
    }

    body.insert("model".to_owned(), Value::String(model_id.to_owned()));
    body.insert("messages".to_owned(), Value::Array(messages));
    body.insert("stream".to_owned(), Value::Bool(true));
    match request.max_tokens {
        Some(max_tokens) => {
            body.insert("max_tokens".to_owned(), Value::from(max_tokens));
        }
        None => {
            body.remove("max_tokens");
        }
    }
    if request.tools.is_empty() {
        body.remove("tools");
    } else {
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect();
        body.insert("tools".to_owned(), Value::Array(tools));
    }

    let mut kwargs = match body.remove("chat_template_kwargs") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    // The think span counts against max_tokens, so the operator switch wins.
    kwargs.insert("enable_thinking".to_owned(), Value::Bool(thinking));
    body.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));

    let mut stream_options = match body.remove("stream_options") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    stream_options
        .entry("include_usage")
        .or_insert(Value::Bool(true));
    body.insert("stream_options".to_owned(), Value::Object(stream_options));

    Ok(Value::Object(body))
}

fn push_message(out: &mut Vec<Value>, message: &Message) -> Result<(), CompletionError> {
    match message {
        Message::User { content } => push_user(out, content),
        Message::Assistant { content } => {
            out.push(assistant_message(content)?);
            Ok(())
        }
    }
}

fn push_user(out: &mut Vec<Value>, blocks: &[UserContent]) -> Result<(), CompletionError> {
    let mut pending = Vec::new();
    for block in blocks {
        if let UserContent::ToolResult(result) = block {
            if !pending.is_empty() {
                out.push(user_message(&pending)?);
                pending.clear();
            }
            out.push(tool_message(result)?);
        } else {
            pending.push(block.clone());
        }
    }
    if !pending.is_empty() {
        out.push(user_message(&pending)?);
    }
    Ok(())
}

fn user_message(blocks: &[UserContent]) -> Result<Value, CompletionError> {
    let images = blocks.iter().any(matches_image);
    if !images {
        let mut text = String::new();
        for block in blocks {
            if let UserContent::Text(chunk) = block {
                push_line(&mut text, &chunk.text);
            }
        }
        return Ok(serde_json::json!({
            "role": "user",
            "content": text,
        }));
    }
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            UserContent::Text(chunk) => {
                parts.push(serde_json::json!({"type": "text", "text": chunk.text}));
            }
            UserContent::Image(image) => parts.push(image_part(image)),
            UserContent::ToolResult(_) => {}
        }
    }
    Ok(serde_json::json!({
        "role": "user",
        "content": parts,
    }))
}

fn tool_message(result: &ToolResult) -> Result<Value, CompletionError> {
    let saw_image = result.content.iter().any(|block| match block {
        ToolResultContent::Image(_) => true,
        ToolResultContent::Text(_) => false,
    });
    let content = if saw_image {
        let mut parts = Vec::new();
        for block in &result.content {
            match block {
                ToolResultContent::Text(chunk) => {
                    parts.push(serde_json::json!({"type": "text", "text": chunk.text}));
                }
                ToolResultContent::Image(image) => parts.push(image_part(image)),
            }
        }
        Value::Array(parts)
    } else {
        let mut text = String::new();
        for block in &result.content {
            if let ToolResultContent::Text(chunk) = block {
                push_line(&mut text, &chunk.text);
            }
        }
        Value::String(text)
    };
    Ok(serde_json::json!({
        "role": "tool",
        "tool_call_id": result.id,
        "content": content,
    }))
}

fn assistant_message(blocks: &[AssistantContent]) -> Result<Value, CompletionError> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            AssistantContent::Text(chunk) => push_line(&mut text, &chunk.text),
            AssistantContent::Reasoning(block) => {
                let rendered = block.display_text();
                if !rendered.is_empty() {
                    push_line(&mut reasoning, &rendered);
                }
            }
            AssistantContent::ToolCall(call) => {
                let arguments = serde_json::to_string(&call.function.arguments)?;
                tool_calls.push(serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": arguments,
                    }
                }));
            }
        }
    }
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String("assistant".to_owned()));
    if text.is_empty() && !tool_calls.is_empty() {
        message.insert("content".to_owned(), Value::Null);
    } else {
        message.insert("content".to_owned(), Value::String(text));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".to_owned(), Value::String(reasoning));
    }
    Ok(Value::Object(message))
}

fn image_part(image: &Image) -> Value {
    serde_json::json!({
        "type": "image_url",
        "image_url": { "url": image_url(image) }
    })
}

fn image_url(image: &Image) -> String {
    match &image.data {
        ImageSource::Url(url) => url.clone(),
        ImageSource::Base64(data) => {
            let mime = image
                .media_type
                .map(|media| media.to_mime_type())
                .unwrap_or("application/octet-stream");
            format!("data:{mime};base64,{data}")
        }
    }
}

fn matches_image(block: &UserContent) -> bool {
    match block {
        UserContent::Image(_) => true,
        UserContent::Text(_) | UserContent::ToolResult(_) => false,
    }
}

fn push_line(buf: &mut String, line: &str) {
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(line);
}

fn request_error(message: &'static str) -> CompletionError {
    CompletionError::RequestError(Box::new(WireError(message)))
}

#[derive(Debug)]
struct WireError(&'static str);

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for WireError {}
