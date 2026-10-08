//! `POST /v1/chat/completions`: messages in, text out.
//!
//! The model's chat template does the rendering, so a model without one in
//! the manifest fails here with an HTTP status rather than producing a
//! plausible-looking but wrong prompt.

use std::ops::ControlFlow;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::convert;
use super::sse::{self, Piece};
use super::types::{
    ChatChoice, ChatChunk, ChatChunkChoice, ChatMessageOut, ChatRequest, ChatResponse, Delta, Usage,
};
use super::{created, finish_name, request_id};
use crate::serve::AppState;
use crate::serve::error::{ServeError, json_response};
use crate::serve::model::{GenOpts, Generate};

pub async fn create(
    State(state): State<Arc<AppState>>,
    axum::Json(request): axum::Json<ChatRequest>,
) -> Response {
    match run(&state, request).await {
        Ok(response) => response,
        Err(error) => error.openai(),
    }
}

async fn run(state: &AppState, request: ChatRequest) -> Result<Response, ServeError> {
    let model = state.registry.generate(Some(&request.model))?.clone();
    let opts = convert::chat_opts(&request)?;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);

    // Rendered before anything is spawned, so a missing template is still a
    // plain HTTP error and never an event stream that dies mid-flight.
    let messages = convert::chat_messages(&request)?;
    let prompt = model.render(&messages)?;

    let id = request_id("chatcmpl");
    let created = created();
    let name = request.model.clone();
    if request.stream {
        Ok(stream_response(
            model,
            prompt,
            opts,
            include_usage,
            id,
            created,
            name,
        ))
    } else {
        buffered_response(model, prompt, opts, id, created, name).await
    }
}

async fn buffered_response(
    model: Arc<dyn Generate>,
    prompt: String,
    opts: GenOpts,
    id: String,
    created: u64,
    name: String,
) -> Result<Response, ServeError> {
    let (report, text) = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let report = model.generate(&prompt, &opts, &mut |piece| {
            text.push_str(piece);
            ControlFlow::Continue(())
        });
        (report, text)
    })
    .await
    .map_err(|error| ServeError::internal(format!("generation task failed: {error}")))?;

    let report = report?;
    let response = ChatResponse {
        id,
        object: "chat.completion",
        created,
        model: name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content: text,
            },
            finish_reason: Some(finish_name(report.finish)),
        }],
        usage: Usage::new(report.usage),
    };
    let body = serde_json::to_string(&response)
        .map_err(|error| ServeError::internal(format!("cannot encode response: {error}")))?;
    Ok(json_response(StatusCode::OK, &body))
}

fn stream_response(
    model: Arc<dyn Generate>,
    prompt: String,
    opts: GenOpts,
    include_usage: bool,
    id: String,
    created: u64,
    name: String,
) -> Response {
    let (sender, receiver) = tokio::sync::mpsc::channel::<Piece>(32);
    tokio::task::spawn_blocking(move || {
        let report = {
            let mut emit = |piece: &str| match sender.blocking_send(Piece::Text(piece.to_string()))
            {
                Ok(()) => ControlFlow::Continue(()),
                Err(_) => ControlFlow::Break(()),
            };
            model.generate(&prompt, &opts, &mut emit)
        };
        if let Ok(report) = report {
            let _ = sender.blocking_send(Piece::Done(report));
        }
    });

    let mut announced = false;
    let frames = sse::body(receiver, move |piece| {
        let mut payloads = Vec::new();
        match piece {
            Piece::Text(text) => {
                // OpenAI announces the assistant role once, on the first
                // frame that carries content.
                let role = if announced {
                    None
                } else {
                    announced = true;
                    Some("assistant")
                };
                payloads.push(
                    serde_json::to_string(&ChatChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: name.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: Delta {
                                role,
                                content: Some(text.clone()),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    })
                    .unwrap_or_default(),
                );
            }
            Piece::Done(report) => {
                payloads.push(
                    serde_json::to_string(&ChatChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: name.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            finish_reason: Some(finish_name(report.finish)),
                        }],
                        usage: None,
                    })
                    .unwrap_or_default(),
                );
                if include_usage {
                    payloads.push(
                        serde_json::to_string(&ChatChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: name.clone(),
                            choices: Vec::new(),
                            usage: Some(Usage::new(report.usage)),
                        })
                        .unwrap_or_default(),
                    );
                }
            }
        }
        payloads
    });
    frames.into_response()
}
