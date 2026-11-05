//! Server-sent chat chunks as Infinity [`StreamChunk`]s.
//!
//! Text arrives as `delta.content`. Reasoning arrives as `delta.reasoning_content`
//! and is closed into one [`StreamChunk::Reasoning`] when the span ends.
//! Tool arguments arrive in one terminal chunk; the call is emitted once its
//! id, name, and JSON arguments are all present. A usage object becomes
//! [`StreamChunk::Final`]. `[DONE]` without a usage object still ends the
//! stream with an empty final chunk.

use std::collections::{BTreeMap, BTreeSet};

use infinity_provider_protocol::{
    CompletionError, FinalResponse, Reasoning, StreamChunk, ToolCall, Usage,
};
use serde_json::Value;

#[derive(Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    pending: Vec<String>,
    reasoning: String,
    reasoning_open: bool,
    saw_final: bool,
    tools: BTreeMap<u64, ToolAcc>,
    emitted_tools: BTreeSet<u64>,
}

#[derive(Default)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
}

pub(crate) struct Feed {
    pub chunks: Vec<StreamChunk>,
    pub error: Option<CompletionError>,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn saw_final(&self) -> bool {
        self.saw_final
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) -> Feed {
        self.buf.extend_from_slice(bytes);
        let mut chunks = Vec::new();
        while let Some(pos) = self.buf.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = match std::str::from_utf8(&line) {
                Ok(line) => line.trim_end_matches(['\r', '\n']),
                Err(_) => {
                    return Feed {
                        chunks,
                        error: Some(stream_ended("hipfire sent a chat frame that is not UTF-8")),
                    };
                }
            };
            match self.push_line(line) {
                Ok(more) => chunks.extend(more),
                Err(err) => {
                    return Feed {
                        chunks,
                        error: Some(err),
                    };
                }
            }
        }
        Feed {
            chunks,
            error: None,
        }
    }

    pub(crate) fn finish(&mut self) -> Feed {
        let mut chunks = Vec::new();
        if !self.buf.is_empty() {
            let line = match std::str::from_utf8(&self.buf) {
                Ok(line) => line.trim_end_matches(['\r', '\n']).to_owned(),
                Err(_) => {
                    return Feed {
                        chunks,
                        error: Some(stream_ended("hipfire sent a chat frame that is not UTF-8")),
                    };
                }
            };
            self.buf.clear();
            if !line.is_empty() {
                match self.push_line(&line) {
                    Ok(more) => chunks.extend(more),
                    Err(err) => {
                        return Feed {
                            chunks,
                            error: Some(err),
                        };
                    }
                }
            }
        }
        match self.dispatch() {
            Ok(more) => chunks.extend(more),
            Err(err) => {
                return Feed {
                    chunks,
                    error: Some(err),
                };
            }
        }
        self.close_reasoning(&mut chunks);
        if self.saw_final {
            return Feed {
                chunks,
                error: None,
            };
        }
        Feed {
            chunks,
            error: Some(stream_ended(
                "hipfire stream ended before the completion finished",
            )),
        }
    }

    fn push_line(&mut self, line: &str) -> Result<Vec<StreamChunk>, CompletionError> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return Ok(Vec::new());
        }
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.strip_prefix(' ').unwrap_or(data);
            self.pending.push(data.to_owned());
        }
        Ok(Vec::new())
    }

    fn dispatch(&mut self) -> Result<Vec<StreamChunk>, CompletionError> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let payload = self.pending.join("\n");
        self.pending.clear();
        let payload = payload.trim();
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        if payload == "[DONE]" {
            let mut chunks = Vec::new();
            self.close_reasoning(&mut chunks);
            self.emit_final(&mut chunks, None);
            return Ok(chunks);
        }
        let value: Value = serde_json::from_str(payload)
            .map_err(|_| stream_ended("hipfire sent a chat chunk that is not JSON"))?;
        self.event(&value)
    }

    fn event(&mut self, value: &Value) -> Result<Vec<StreamChunk>, CompletionError> {
        if value.get("error").and_then(Value::as_object).is_some() {
            let body = value.to_string();
            let class = crate::classify::classify(200, false, &body);
            let message = crate::classify::public_message(200, &body);
            return Err(CompletionError::provider(class, message));
        }
        let mut chunks = Vec::new();
        if let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str)
                && !text.is_empty()
            {
                self.reasoning.push_str(text);
                self.reasoning_open = true;
                chunks.push(StreamChunk::ReasoningDelta {
                    id: None,
                    text: text.to_owned(),
                });
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                self.close_reasoning(&mut chunks);
                chunks.push(StreamChunk::Text(text.to_owned()));
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                self.close_reasoning(&mut chunks);
                for call in calls {
                    self.absorb_tool(call, &mut chunks);
                }
            }
            if choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .is_some()
            {
                self.close_reasoning(&mut chunks);
            }
        }
        if let Some(usage) = value.get("usage").and_then(Value::as_object) {
            self.close_reasoning(&mut chunks);
            self.emit_final(&mut chunks, Some(usage_from(usage)));
        }
        Ok(chunks)
    }

    fn absorb_tool(&mut self, call: &Value, chunks: &mut Vec<StreamChunk>) {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let acc = self.tools.entry(index).or_default();
        if let Some(id) = call.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            acc.id = id.to_owned();
        }
        if let Some(name) = call.pointer("/function/name").and_then(Value::as_str)
            && !name.is_empty()
        {
            acc.name = name.to_owned();
        }
        if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str) {
            acc.arguments.push_str(arguments);
        }
        if self.emitted_tools.contains(&index) {
            return;
        }
        let Some(acc) = self.tools.get(&index) else {
            return;
        };
        if acc.id.is_empty() || acc.name.is_empty() || acc.arguments.is_empty() {
            return;
        }
        let Ok(arguments) = serde_json::from_str::<Value>(&acc.arguments) else {
            return;
        };
        let id = acc.id.clone();
        let name = acc.name.clone();
        chunks.push(StreamChunk::ToolCall(ToolCall::new(id, name, arguments)));
        self.emitted_tools.insert(index);
    }

    fn close_reasoning(&mut self, chunks: &mut Vec<StreamChunk>) {
        if self.reasoning_open {
            chunks.push(StreamChunk::Reasoning(Reasoning::new_with_signature(
                self.reasoning.clone(),
                None,
            )));
            self.reasoning_open = false;
        }
    }

    fn emit_final(&mut self, chunks: &mut Vec<StreamChunk>, usage: Option<Usage>) {
        if self.saw_final {
            return;
        }
        self.saw_final = true;
        chunks.push(StreamChunk::Final(FinalResponse { usage }));
    }
}

fn usage_from(value: &serde_json::Map<String, Value>) -> Usage {
    Usage {
        input_tokens: u64_field(value, "prompt_tokens"),
        output_tokens: u64_field(value, "completion_tokens"),
        total_tokens: u64_field(value, "total_tokens"),
        cached_input_tokens: value
            .get("prompt_tokens_details")
            .and_then(Value::as_object)
            .map(|details| u64_field(details, "cached_tokens"))
            .unwrap_or(0),
    }
}

fn u64_field(value: &serde_json::Map<String, Value>, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn stream_ended(message: &str) -> CompletionError {
    CompletionError::ResponseError(message.to_owned())
}
