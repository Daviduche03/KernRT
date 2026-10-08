//! The OpenAI surface, driven in-process.
//!
//! A stub generator keeps these fast and file-free; the point under test is
//! the contract — request parsing, the response shapes, status codes, the
//! error envelope, and the SSE frame sequence. Tokenisation and the decode
//! loop are covered by the crate's own tests.
#![cfg(feature = "serve")]

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use kernrt::serve::{
    AppState, ChatMessage, ChatTemplate, Detect, Finish, GenOpts, GenReport, Generate,
    MessageFormats, Model, Registry, ServeError,
};
use tower::ServiceExt;

/// Emits a fixed script of text deltas, then reports a chosen finish reason.
struct StubGenerate {
    id: &'static str,
    pieces: &'static [&'static str],
    finish: Finish,
    template: Option<ChatTemplate>,
}

impl Generate for StubGenerate {
    fn id(&self) -> &str {
        self.id
    }

    fn render(&self, messages: &[ChatMessage]) -> Result<String, ServeError> {
        self.template
            .as_ref()
            .ok_or_else(|| {
                ServeError::bad_request(format!(
                    "model {:?} has no chat template; add \"chat_template\" to its manifest entry, \
                     or call /v1/completions with a raw prompt",
                    self.id
                ))
            })?
            .render(messages)
    }

    fn generate(
        &self,
        prompt: &str,
        opts: &GenOpts,
        emit: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<GenReport, ServeError> {
        if opts.top_p != 1.0 {
            return Err(ServeError::bad_request(
                "top_p is not supported yet; omit it or send 1.0",
            ));
        }
        assert!(!prompt.is_empty(), "the prompt must have been validated");
        for piece in self.pieces {
            if matches!(emit(piece), std::ops::ControlFlow::Break(_)) {
                break;
            }
        }
        Ok(GenReport {
            usage: kernrt::generate::Usage {
                prompt_tokens: 3,
                completion_tokens: self.pieces.len(),
            },
            finish: self.finish,
        })
    }
}

struct Detector;

impl Detect for Detector {
    fn id(&self) -> &str {
        "det"
    }
    fn input_size(&self) -> (usize, usize) {
        (2, 2)
    }
    fn detect(&self, _: &[u8]) -> Result<Vec<kernrt::arch::vision::yolo26::Detection>, ServeError> {
        Ok(Vec::new())
    }
}

fn app(models: Vec<Model>) -> Router {
    let mut registry = Registry::new();
    for model in models {
        registry.register(model);
    }
    kernrt::serve::router(AppState::new(registry))
}

fn plain_app() -> Router {
    app(vec![Model::generate(StubGenerate {
        id: "stub",
        pieces: &["Hel", "lo"],
        finish: Finish::Stop,
        template: None,
    })])
}

fn templated_app() -> Router {
    app(vec![Model::generate(StubGenerate {
        id: "stub",
        pieces: &["hi"],
        finish: Finish::Stop,
        template: Some(ChatTemplate {
            message: MessageFormats::PerRole {
                system: Some("[system] {content}\n".to_string()),
                user: "[user] {content}\n".to_string(),
                assistant: "[assistant] {content}\n".to_string(),
            },
            generation_prompt: "> ".to_string(),
        }),
    })])
}

fn post_json(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::builder().uri(path).body(Body::empty()).unwrap()
}

async fn send(app: Router, request: Request<Body>) -> (StatusCode, String, String) {
    let response = app.oneshot(request).await.expect("router panics");
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|value| value.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .expect("body reads");
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

fn json<T: serde::de::DeserializeOwned>(body: &str) -> T {
    serde_json::from_str(body).expect("response is valid JSON")
}

/// Pull the payloads out of an SSE body.
fn events(body: &str) -> Vec<serde_json::Value> {
    body.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let payload = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .collect::<Vec<_>>()
                .join("");
            serde_json::from_str(&payload).expect("event payload is JSON")
        })
        .collect()
}

#[tokio::test]
async fn a_completion_returns_the_documented_shape() {
    let (status, content_type, body) = send(
        plain_app(),
        post_json("/v1/completions", r#"{"model":"stub","prompt":"Once"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let response: serde_json::Value = json(&body);
    assert_eq!(response["object"], "text_completion");
    assert!(response["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(response["choices"][0]["text"], "Hello");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["choices"][0]["logprobs"], serde_json::Value::Null);
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(response["usage"]["completion_tokens"], 2);
    assert_eq!(response["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn a_length_finish_is_reported_as_length() {
    let router = app(vec![Model::generate(StubGenerate {
        id: "stub",
        pieces: &["x"],
        finish: Finish::Length,
        template: None,
    })]);
    let (_, _, body) = send(
        router,
        post_json("/v1/completions", r#"{"model":"stub","prompt":"p"}"#),
    )
    .await;
    let response: serde_json::Value = json(&body);
    assert_eq!(response["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn streaming_emits_text_frames_then_a_finish_frame() {
    let (status, content_type, body) = send(
        plain_app(),
        post_json(
            "/v1/completions",
            r#"{"model":"stub","prompt":"Once","stream":true}"#,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "text/event-stream");

    let frames = events(&body);
    assert_eq!(
        frames.len(),
        3,
        "two text frames and a finish frame: {body}"
    );
    assert_eq!(frames[0]["choices"][0]["text"], "Hel");
    assert_eq!(
        frames[0]["choices"][0]["finish_reason"],
        serde_json::Value::Null
    );
    assert_eq!(frames[1]["choices"][0]["text"], "lo");
    assert_eq!(frames[2]["choices"][0]["finish_reason"], "stop");
    assert!(
        frames[2]["usage"].is_null(),
        "usage is only sent when stream_options asks for it"
    );
    for frame in &frames {
        assert_eq!(frame["object"], "text_completion");
    }
}

#[tokio::test]
async fn streaming_sends_usage_only_when_asked() {
    let (_, _, body) = send(
        plain_app(),
        post_json(
            "/v1/completions",
            r#"{"model":"stub","prompt":"p","stream":true,"stream_options":{"include_usage":true}}"#,
        ),
    )
    .await;
    let frames = events(&body);
    assert_eq!(
        frames.len(),
        4,
        "two text, a finish, and a usage frame: {body}"
    );
    let usage = &frames[3];
    assert!(
        usage["choices"].as_array().unwrap().is_empty(),
        "the usage frame carries no choices: {usage}"
    );
    assert_eq!(usage["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn models_lists_generators_and_nothing_else() {
    let app = app(vec![
        Model::generate(StubGenerate {
            id: "stub",
            pieces: &[],
            finish: Finish::Stop,
            template: None,
        }),
        Model::detect(Detector),
    ]);
    let (status, _, body) = send(app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let list: serde_json::Value = json(&body);
    assert_eq!(list["object"], "list");
    let ids: Vec<&str> = list["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert_eq!(ids, vec!["stub"], "a detector is not a chat model");
}

#[tokio::test]
async fn chat_without_a_template_says_exactly_what_to_do() {
    let (_, _, body) = send(
        plain_app(),
        post_json(
            "/v1/chat/completions",
            r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    let envelope: serde_json::Value = json(&body);
    let message = envelope["error"]["message"].as_str().unwrap();
    assert_eq!(envelope["error"]["type"], "invalid_request_error");
    assert!(message.contains("has no chat template"), "{message}");
    assert!(message.contains("/v1/completions"), "{message}");
}

#[tokio::test]
async fn chat_renders_through_the_configured_template() {
    let (_, _, body) = send(
        templated_app(),
        post_json(
            "/v1/chat/completions",
            r#"{"model":"stub","messages":[
                {"role":"system","content":"be terse"},
                {"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    let response: serde_json::Value = json(&body);
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    assert_eq!(response["choices"][0]["message"]["content"], "hi");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
}

/// The template is what makes chat work, so the rendered prompt must be
/// observable. The stub echoes it back through the completion text.
#[tokio::test]
async fn the_template_is_applied_per_message_then_the_assistant_prefix() {
    let mut registry = Registry::new();
    registry.register(Model::generate(EchoRender));
    let (_, _, body) = send(
        kernrt::serve::router(AppState::new(registry)),
        post_json(
            "/v1/chat/completions",
            r#"{"model":"echo","messages":[
                {"role":"user","content":"one"},
                {"role":"assistant","content":"two"}]}"#,
        ),
    )
    .await;
    let response: serde_json::Value = json(&body);
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "[user] one\n[assistant] two\n> "
    );
}

/// Renders the request back as its "completion", so the test can assert on
/// the exact string `render` produced.
struct EchoRender;

impl Generate for EchoRender {
    fn id(&self) -> &str {
        "echo"
    }

    fn render(&self, messages: &[ChatMessage]) -> Result<String, ServeError> {
        ChatTemplate {
            message: MessageFormats::PerRole {
                system: Some("[system] {content}\n".to_string()),
                user: "[user] {content}\n".to_string(),
                assistant: "[assistant] {content}\n".to_string(),
            },
            generation_prompt: "> ".to_string(),
        }
        .render(messages)
    }

    fn generate(
        &self,
        prompt: &str,
        _: &GenOpts,
        emit: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<GenReport, ServeError> {
        let _ = emit(prompt);
        Ok(GenReport::default())
    }
}

#[tokio::test]
async fn chat_streams_a_role_frame_then_content_then_finish() {
    let (_, content_type, body) = send(
        templated_app(),
        post_json(
            "/v1/chat/completions",
            r#"{"model":"stub","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        ),
    )
    .await;
    assert_eq!(content_type, "text/event-stream");
    let frames = events(&body);
    assert_eq!(
        frames.len(),
        2,
        "one content frame and one finish frame: {body}"
    );
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(frames[0]["choices"][0]["delta"]["content"], "hi");
    assert_eq!(frames[1]["choices"][0]["finish_reason"], "stop");
    for frame in &frames {
        assert_eq!(frame["object"], "chat.completion.chunk");
    }
}

#[tokio::test]
async fn an_unknown_model_is_an_openai_shaped_404() {
    let (status, _, body) = send(
        plain_app(),
        post_json("/v1/completions", r#"{"model":"nope","prompt":"x"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let envelope: serde_json::Value = json(&body);
    assert_eq!(envelope["error"]["type"], "invalid_request_error");
    assert!(
        envelope["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no generate model named nope"),
        "{envelope}"
    );
}

#[tokio::test]
async fn unsupported_sampling_is_rejected_before_any_stream_starts() {
    let (status, content_type, body) = send(
        plain_app(),
        post_json(
            "/v1/completions",
            r#"{"model":"stub","prompt":"x","top_p":0.9,"stream":true}"#,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type, "application/json", "no event stream may open");
    let envelope: serde_json::Value = json(&body);
    assert!(
        envelope["error"]["message"]
            .as_str()
            .unwrap()
            .contains("top_p"),
        "{envelope}"
    );
}

#[tokio::test]
async fn every_error_uses_the_openai_envelope() {
    for (path, body) in [
        ("/v1/completions", r#"{"model":"stub","prompt":[]}"#),
        ("/v1/chat/completions", r#"{"model":"stub","messages":[]}"#),
        (
            "/v1/completions",
            r#"{"model":"stub","prompt":"x","max_tokens":0}"#,
        ),
    ] {
        let (status, _, body) = send(plain_app(), post_json(path, body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        let envelope: serde_json::Value = json(&body);
        assert!(
            envelope["error"]["message"].is_string(),
            "{path} gave {body}"
        );
        assert_eq!(envelope["error"]["type"], "invalid_request_error", "{path}");
    }
}

#[tokio::test]
async fn completions_do_not_leak_into_the_native_routes() {
    let (status, _, _) = send(
        plain_app(),
        post_json(
            "/decide",
            r#"{"state":"s","instructions":"i","criteria":[]}"#,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
