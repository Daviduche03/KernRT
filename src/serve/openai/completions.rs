//! `POST /v1/completions`: a raw prompt in, text out.

use std::ops::ControlFlow;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::convert;
use super::sse::{self, Piece};
use super::types::{
    CompletionChoice, CompletionChunk, CompletionChunkChoice, CompletionRequest,
    CompletionResponse, Usage,
};
use super::{created, finish_name, request_id};
use crate::serve::AppState;
use crate::serve::error::{ServeError, json_response};
use crate::serve::model::{GenOpts, Generate};

pub async fn create(
    State(state): State<Arc<AppState>>,
    axum::Json(request): axum::Json<CompletionRequest>,
) -> Response {
    match run(&state, request).await {
        Ok(response) => response,
        Err(error) => error.openai(),
    }
}

async fn run(state: &AppState, request: CompletionRequest) -> Result<Response, ServeError> {
    let model = state.registry.generate(Some(&request.model))?.clone();
    let opts = convert::completion_opts(&request)?;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let prompt = convert::prompt_text(request.prompt)?;
    if prompt.is_empty() {
        return Err(ServeError::bad_request("prompt must not be empty"));
    }

    let id = request_id("cmpl");
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
    let response = CompletionResponse {
        id,
        object: "text_completion",
        created,
        model: name,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: Some(finish_name(report.finish)),
            logprobs: None,
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
                // The client went away; stop decoding rather than burn CPU.
                Err(_) => ControlFlow::Break(()),
            };
            model.generate(&prompt, &opts, &mut emit)
        };
        if let Ok(report) = report {
            let _ = sender.blocking_send(Piece::Done(report));
        }
    });

    let frames = sse::body(receiver, move |piece| {
        let mut payloads = Vec::new();
        match piece {
            Piece::Text(text) => {
                payloads.push(
                    serde_json::to_string(&CompletionChunk {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: name.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: text.clone(),
                            index: 0,
                            finish_reason: None,
                        }],
                        usage: None,
                    })
                    .unwrap_or_default(),
                );
            }
            Piece::Done(report) => {
                payloads.push(
                    serde_json::to_string(&CompletionChunk {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: name.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: String::new(),
                            index: 0,
                            finish_reason: Some(finish_name(report.finish)),
                        }],
                        usage: None,
                    })
                    .unwrap_or_default(),
                );
                if include_usage {
                    payloads.push(
                        serde_json::to_string(&CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
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
