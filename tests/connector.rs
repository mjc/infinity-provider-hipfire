//! The provider against a local HTTP double of the routes it calls.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use infinity_provider_hipfire::{Config, HipfireProvider};
use infinity_provider_protocol::{
    AssistantContent, CompletionError, CompletionRequest, ErrorClass, FinalResponse, Message,
    ModelProvider, Reasoning, StreamChunk, ToolCall, ToolDefinition, ToolResult, ToolResultContent,
    Usage, UserContent,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const SECRET: &str = "SENTINEL-connector-bearer-do-not-print";

#[tokio::test]
async fn lists_the_resident_model_from_health_context() {
    let double = Double::start(routes(&[
        (
            "GET /v1/models",
            json_reply(200, &[], models_body(&["qwen3.5:9b"], &["text"])),
        ),
        ("GET /health", json_reply(200, &[], health_body(8192))),
    ]))
    .await;
    let models = provider(&double, false, 5, 111)
        .list_models()
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "qwen3.5:9b");
    assert_eq!(models[0].display_name, "qwen3.5:9b");
    assert_eq!(models[0].context_window, 8192);
    assert_eq!(models[0].max_output_tokens, None);
    assert!(!models[0].supports_image_input);
}

#[tokio::test]
async fn resident_image_modality_does_not_apply_to_other_models() {
    let body = json!({
        "object": "list",
        "data": [
            {
                "id": "qwen3.5:9b",
                "object": "model",
                "architecture": { "input_modalities": ["text", "image"] }
            },
            {
                "id": "other:1",
                "object": "model",
                "architecture": { "input_modalities": ["text", "image"] }
            }
        ]
    });
    let double = Double::start(routes(&[
        ("GET /v1/models", json_reply(200, &[], body)),
        ("GET /health", json_reply(200, &[], health_body(8192))),
    ]))
    .await;
    let models = provider(&double, false, 5, 4096)
        .list_models()
        .await
        .unwrap();
    assert_eq!(models[0].model_id, "qwen3.5:9b");
    assert_eq!(models[0].context_window, 8192);
    assert!(models[0].supports_image_input);
    assert_eq!(models[1].model_id, "other:1");
    assert_eq!(models[1].context_window, 4096);
    assert!(!models[1].supports_image_input);
}

#[tokio::test]
async fn text_stream_requests_thinking_off_and_ends_in_usage() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
        "data: [DONE]\n\n",
    );
    let double = Double::start(routes(&[("POST /v1/chat/completions", sse_reply(sse))])).await;
    let provider = provider(&double, false, 5, 8192);
    let stream = provider
        .invoke_model(
            "qwen3.5:9b",
            sample_request(Some(
                json!({"temperature": 0.2, "chat_template_kwargs": {"enable_thinking": true}}),
            )),
        )
        .await
        .unwrap();
    let chunks = collect(stream).await;
    assert_ok(
        &chunks,
        &[
            StreamChunk::Text("Hello".into()),
            StreamChunk::Text(" world".into()),
            StreamChunk::Final(FinalResponse {
                usage: Some(Usage {
                    input_tokens: 3,
                    output_tokens: 2,
                    total_tokens: 5,
                    cached_input_tokens: 1,
                }),
            }),
        ],
    );
    let body = recorded_json(&double, "POST /v1/chat/completions");
    assert_eq!(body["model"], "qwen3.5:9b");
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "be brief");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "What is the weather?");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    assert!(double.authorization_header().is_none());
}

#[tokio::test]
async fn reasoning_only_stream_closes_one_reasoning_block() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" two\"},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let double = Double::start(routes(&[("POST /v1/chat/completions", sse_reply(sse))])).await;
    let stream = provider(&double, true, 5, 8192)
        .invoke_model("qwen3.5:9b", sample_request(None))
        .await
        .unwrap();
    let chunks = collect(stream).await;
    assert_ok(
        &chunks,
        &[
            StreamChunk::ReasoningDelta {
                id: None,
                text: "step".into(),
            },
            StreamChunk::ReasoningDelta {
                id: None,
                text: " two".into(),
            },
            StreamChunk::Reasoning(Reasoning::new_with_signature("step two", None)),
            StreamChunk::Final(FinalResponse { usage: None }),
        ],
    );
    let body = recorded_json(&double, "POST /v1/chat/completions");
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
}

#[tokio::test]
async fn one_terminal_tool_call_chunk_becomes_a_tool_call() {
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Berlin\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n",
        "data: [DONE]\n\n",
    );
    let double = Double::start(routes(&[("POST /v1/chat/completions", sse_reply(sse))])).await;
    let stream = provider(&double, false, 5, 8192)
        .invoke_model("qwen3.5:9b", sample_request(None))
        .await
        .unwrap();
    let chunks = collect(stream).await;
    assert_ok(
        &chunks,
        &[
            StreamChunk::ToolCall(ToolCall::new(
                "call-1",
                "get_weather",
                json!({"city": "Berlin"}),
            )),
            StreamChunk::Final(FinalResponse {
                usage: Some(Usage {
                    input_tokens: 2,
                    output_tokens: 3,
                    total_tokens: 5,
                    cached_input_tokens: 0,
                }),
            }),
        ],
    );
    let body = recorded_json(&double, "POST /v1/chat/completions");
    assert_eq!(body["messages"][2]["role"], "assistant");
    assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call-prev");
    assert_eq!(
        body["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
    assert_eq!(body["messages"][3]["role"], "tool");
    assert_eq!(body["messages"][3]["tool_call_id"], "call-prev");
    assert_eq!(body["messages"][3]["content"], "rain");
}

#[tokio::test]
async fn context_length_on_http_400_is_context_overflow() {
    let body = json!({
        "error": {
            "message": "the prompt does not fit",
            "type": "context_length"
        }
    });
    let err = http_error(400, &[], &body).await;
    assert_eq!(err.class(), ErrorClass::ContextOverflow);
    assert!(err.to_string().contains("the prompt does not fit"));
    assert!(!err.to_string().contains("context_length"));
}

#[tokio::test]
async fn validation_wins_over_a_message_that_names_context_length() {
    let body = json!({
        "error": {
            "message": "context_length exceeded",
            "type": "validation"
        }
    });
    let err = http_error(400, &[], &body).await;
    assert_eq!(err.class(), ErrorClass::Fatal);
}

#[tokio::test]
async fn rendered_daemon_class_is_read_from_the_hipfire_prefix() {
    let body = json!({
        "error": {
            "message": "daemon error: [context_length retryable=false rolled_back=false attempt=1] the prompt does not fit",
            "type": "invalid_request_error"
        }
    });
    let err = http_error(400, &[], &body).await;
    assert_eq!(err.class(), ErrorClass::ContextOverflow);
}

#[tokio::test]
async fn class_field_wins_over_the_openai_type() {
    let body = json!({
        "error": {
            "message": "overlong",
            "type": "invalid_request_error",
            "class": "context_length"
        }
    });
    let err = http_error(400, &[], &body).await;
    assert_eq!(err.class(), ErrorClass::ContextOverflow);
}

#[tokio::test]
async fn busy_queue_with_retry_after_is_transient() {
    let body = json!({
        "error": {
            "message": "fatal configuration rejected",
            "type": "server_error"
        }
    });
    let err = http_error(503, &[("retry-after", "1")], &body).await;
    assert_eq!(err.class(), ErrorClass::Transient);
}

#[tokio::test]
async fn typed_transient_and_named_fatal_statuses() {
    let transient = json!({"error": {"message": "try again", "type": "transient"}});
    assert_eq!(
        http_error(500, &[], &transient).await.class(),
        ErrorClass::Transient
    );
    let unsupported = json!({"error": {"message": "nope", "type": "unsupported"}});
    assert_eq!(
        http_error(400, &[], &unsupported).await.class(),
        ErrorClass::Fatal
    );
    assert_eq!(
        http_error(401, &[], &json!({})).await.class(),
        ErrorClass::Fatal
    );
    assert_eq!(
        http_error(404, &[], &json!({"error": {"message": "model not found locally: missing:model", "type": "invalid_request_error"}})).await.class(),
        ErrorClass::Fatal
    );
    assert_eq!(
        http_error(413, &[], &json!({})).await.class(),
        ErrorClass::Fatal
    );
}

#[tokio::test]
async fn timeout_before_the_first_frame_is_transient() {
    let double = Double::start(routes(&[("POST /v1/chat/completions", Reply::hang())])).await;
    let err = expect_err(
        provider(&double, false, 1, 8192)
            .invoke_model("qwen3.5:9b", sample_request(None))
            .await,
    );
    assert_eq!(err.class(), ErrorClass::Transient);
    assert!(err.to_string().contains("timed out before the first frame"));
}

#[tokio::test]
async fn a_stream_that_ends_after_bytes_is_transient() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";
    let double = Double::start(routes(&[("POST /v1/chat/completions", sse_reply(sse))])).await;
    let stream = provider(&double, false, 5, 8192)
        .invoke_model("qwen3.5:9b", sample_request(None))
        .await
        .unwrap();
    let chunks = collect(stream).await;
    match chunks.first() {
        Some(Ok(StreamChunk::Text(text))) => assert_eq!(text, "Hi"),
        other => panic!("expected text, got {other:?}"),
    }
    let err = chunks.last().unwrap().as_ref().unwrap_err();
    assert_eq!(err.class(), ErrorClass::Transient);
}

#[tokio::test]
async fn a_mid_stream_context_length_keeps_its_class() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"the prompt does not fit\",\"type\":\"context_length\"}}\n\n",
    );
    let double = Double::start(routes(&[("POST /v1/chat/completions", sse_reply(sse))])).await;
    let stream = provider(&double, false, 5, 8192)
        .invoke_model("qwen3.5:9b", sample_request(None))
        .await
        .unwrap();
    let chunks = collect(stream).await;
    match chunks.first() {
        Some(Ok(StreamChunk::Text(text))) => assert_eq!(text, "Hi"),
        other => panic!("expected text, got {other:?}"),
    }
    let err = chunks.last().unwrap().as_ref().unwrap_err();
    assert_eq!(err.class(), ErrorClass::ContextOverflow);
}

#[tokio::test]
async fn transport_errors_do_not_include_the_bearer() {
    let path = write_secret();
    let config = Config::from_vars(|key| match key {
        "HIPFIRE_BASE_URL" => Some("http://127.0.0.1:1/v1".into()),
        "HIPFIRE_BEARER_FILE" => Some(path.display().to_string()),
        "HIPFIRE_TIMEOUT_SECS" => Some("2".into()),
        _ => None,
    })
    .unwrap();
    let err = expect_err(
        HipfireProvider::new(config)
            .unwrap()
            .invoke_model("qwen3.5:9b", sample_request(None))
            .await,
    );
    assert_eq!(err.class(), ErrorClass::Transient);
    assert!(!err.to_string().contains(SECRET));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn bearer_is_sent_and_not_echoed_on_a_provider_error() {
    let path = write_secret();
    let body = json!({"error": {"message": "nope", "type": "validation"}});
    let double = Double::start(routes(&[(
        "POST /v1/chat/completions",
        json_reply(400, &[], body),
    )]))
    .await;
    let config = Config::from_vars(|key| match key {
        "HIPFIRE_BASE_URL" => Some(double.base.clone()),
        "HIPFIRE_BEARER_FILE" => Some(path.display().to_string()),
        "HIPFIRE_TIMEOUT_SECS" => Some("5".into()),
        _ => None,
    })
    .unwrap();
    let err = expect_err(
        HipfireProvider::new(config)
            .unwrap()
            .invoke_model("qwen3.5:9b", sample_request(None))
            .await,
    );
    assert_eq!(err.class(), ErrorClass::Fatal);
    assert!(!err.to_string().contains(SECRET));
    assert_eq!(
        double.authorization_header().as_deref(),
        Some(format!("Bearer {SECRET}").as_str())
    );
    let _ = std::fs::remove_file(path);
}

fn provider(double: &Double, thinking: bool, timeout_secs: u64, context: usize) -> HipfireProvider {
    let config = Config::from_vars(|key| match key {
        "HIPFIRE_BASE_URL" => Some(double.base.clone()),
        "HIPFIRE_TIMEOUT_SECS" => Some(timeout_secs.to_string()),
        "HIPFIRE_DEFAULT_CONTEXT" => Some(context.to_string()),
        "HIPFIRE_THINKING" => Some(if thinking { "true" } else { "false" }.into()),
        _ => None,
    })
    .unwrap();
    HipfireProvider::new(config).unwrap()
}

fn sample_request(additional_params: Option<Value>) -> CompletionRequest {
    CompletionRequest {
        preamble: Some("be brief".into()),
        chat_history: vec![
            Message::user("What is the weather?"),
            Message::Assistant {
                content: vec![AssistantContent::ToolCall(ToolCall::new(
                    "call-prev",
                    "get_weather",
                    json!({"city": "Paris"}),
                ))],
            },
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    id: "call-prev".into(),
                    call_id: None,
                    content: vec![ToolResultContent::text("rain")],
                })],
            },
        ],
        tools: vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Local weather".into(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }],
        max_tokens: Some(64),
        additional_params,
    }
}

async fn http_error(status: u16, headers: &[(&str, &str)], body: &Value) -> CompletionError {
    let double = Double::start(routes(&[(
        "POST /v1/chat/completions",
        json_reply(status, headers, body.clone()),
    )]))
    .await;
    expect_err(
        provider(&double, false, 5, 8192)
            .invoke_model("qwen3.5:9b", sample_request(None))
            .await,
    )
}

fn expect_err(
    result: Result<infinity_provider_protocol::ModelStream, CompletionError>,
) -> CompletionError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected a completion error"),
    }
}

async fn collect(
    stream: infinity_provider_protocol::ModelStream,
) -> Vec<Result<StreamChunk, CompletionError>> {
    stream.collect().await
}

fn assert_ok(chunks: &[Result<StreamChunk, CompletionError>], expected: &[StreamChunk]) {
    let actual: Vec<StreamChunk> = chunks
        .iter()
        .map(|chunk| match chunk {
            Ok(chunk) => chunk.clone(),
            Err(err) => panic!("stream error: {err}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

fn recorded_json(double: &Double, key: &str) -> Value {
    let recorded = double.recorded.lock().unwrap();
    let found = recorded.iter().find(|item| item.key == key).unwrap();
    serde_json::from_str(&found.body).unwrap()
}

fn write_secret() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "hipfire-provider-bearer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, format!("{SECRET}\n")).unwrap();
    path
}

fn models_body(ids: &[&str], modalities: &[&str]) -> Value {
    json!({
        "object": "list",
        "data": ids.iter().enumerate().map(|(index, id)| {
            let mut model = json!({"id": id, "object": "model", "owned_by": "hipfire"});
            if index == 0 {
                model["architecture"] = json!({"input_modalities": modalities});
            }
            model
        }).collect::<Vec<_>>()
    })
}

fn health_body(n_ctx: u64) -> Value {
    json!({
        "status": "ok",
        "model": "qwen3.5:9b",
        "loading_model": null,
        "n_ctx": n_ctx
    })
}

fn routes(items: &[(&str, Reply)]) -> HashMap<String, Reply> {
    items
        .iter()
        .map(|(key, reply)| ((*key).to_owned(), reply.clone()))
        .collect()
}

fn json_reply(status: u16, headers: &[(&str, &str)], body: Value) -> Reply {
    Reply {
        status,
        headers: headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect(),
        body: serde_json::to_vec(&body).unwrap(),
        hang: false,
    }
}

fn sse_reply(body: &str) -> Reply {
    Reply {
        status: 200,
        headers: vec![("content-type".into(), "text/event-stream".into())],
        body: body.as_bytes().to_vec(),
        hang: false,
    }
}

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    hang: bool,
}

impl Reply {
    fn hang() -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body: Vec::new(),
            hang: true,
        }
    }
}

struct Recorded {
    key: String,
    headers: Vec<(String, String)>,
    body: String,
}

struct Double {
    base: String,
    recorded: Arc<Mutex<Vec<Recorded>>>,
}

impl Double {
    async fn start(routes: HashMap<String, Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let saved = Arc::clone(&recorded);
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    break;
                };
                let routes = routes.clone();
                let recorded = Arc::clone(&saved);
                tokio::spawn(async move {
                    let _ = handle(socket, routes, recorded).await;
                });
            }
        });
        Self {
            base: format!("http://127.0.0.1:{port}/v1"),
            recorded,
        }
    }

    fn authorization_header(&self) -> Option<String> {
        let recorded = self.recorded.lock().unwrap();
        recorded.iter().find_map(|item| {
            item.headers.iter().find_map(|(name, value)| {
                if name.eq_ignore_ascii_case("authorization") {
                    Some(value.clone())
                } else {
                    None
                }
            })
        })
    }
}

async fn handle(
    mut socket: TcpStream,
    routes: HashMap<String, Reply>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    let header_end = loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > 1024 * 1024 {
            return Ok(());
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut already = buf[header_end + 4..].to_vec();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("").split('?').next().unwrap_or("");
    let mut headers = Vec::new();
    let mut content_length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_owned();
        let value = value.trim().to_owned();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.parse().unwrap_or(0);
        }
        headers.push((name, value));
    }
    while already.len() < content_length {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        already.extend_from_slice(&tmp[..n]);
    }
    already.truncate(content_length);
    let key = format!("{method} {path}");
    recorded.lock().unwrap().push(Recorded {
        key: key.clone(),
        headers,
        body: String::from_utf8_lossy(&already).into_owned(),
    });
    let Some(reply) = routes.get(&key) else {
        let missing = b"not found";
        let head = format!(
            "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            missing.len()
        );
        socket.write_all(head.as_bytes()).await?;
        socket.write_all(missing).await?;
        return Ok(());
    };
    if reply.hang {
        std::future::pending::<()>().await;
    }
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reply.status,
        reason(reply.status),
        reply.body.len()
    );
    for (name, value) in &reply.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    socket.write_all(head.as_bytes()).await?;
    socket.write_all(&reply.body).await?;
    Ok(())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}
