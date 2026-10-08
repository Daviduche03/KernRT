//! One process, one port, two API surfaces.
//!
//! KernRT serves two shapes of model, and they do not fit one schema: a
//! detector takes raw pixels and returns boxes, a decision head returns
//! probabilities per candidate, and only a decoder speaks text. Forcing all
//! three through OpenAI's `text -> text` contract would flatten the structured
//! outputs into JSON strings and accept sampling parameters that mean nothing
//! for a bounding-box regressor.
//!
//! So the universality lives in the *server*, not the *contract*:
//!
//! * [`openai`] speaks OpenAI Chat Completions for what fits it.
//! * [`native`] speaks KernRT's own typed JSON for what does not.
//! * Both convert into [`model`]'s neutral types, which is the only thing
//!   [`crate::arch`] ever sees.
//!
//! ```text
//! cargo run --release --features serve --example serve -- models.json
//! ```
//!
//! ```text
//! GET  /health            liveness
//! POST /infer             image -> detections
//!                         (raw RGB, image/jpeg|png|..., or {"image_url": ...})
//! POST /decide            question -> probabilities
//! ```
//! plus the OpenAI routes documented in [`openai`].

pub mod adapters;
pub mod config;
pub mod error;
pub mod image;
pub mod model;
pub mod native;
pub mod openai;
pub mod registry;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::Uri;
use axum::response::Response;
use axum::routing::{get, post};

pub use config::{ChatTemplate, Entry, Manifest, MessageFormats};
pub use error::ServeError;
pub use model::{
    CandidateScore, ChatMessage, Decide, DecideRequest, Detect, Finish, GenOpts, GenReport,
    GenUsage, Generate, Kind, Model, Role,
};
pub use registry::Registry;

/// Upper bound on a request body.
///
/// axum defaults to 2 MB, which would reject a 1080p RGB frame (6.2 MB) long
/// before anything interesting happened. This covers a ~4600px square frame.
pub const BODY_LIMIT: usize = 64 * 1024 * 1024;

/// Shared handler state. Cheap to clone, since the registry hands out `Arc`s.
#[derive(Clone)]
pub struct AppState {
    pub registry: Registry,
}

impl AppState {
    pub fn new(registry: Registry) -> Self {
        Self { registry }
    }
}

/// Build the full router. Public so tests can drive it in-process with
/// `axum::Router::oneshot` instead of binding a port.
pub fn router(state: AppState) -> Router {
    Router::new()
        // KernRT-native: typed input, typed output, no schema forced on top.
        .route("/health", get(native::health))
        .route("/infer", post(native::detect::infer))
        .route("/decide", post(native::decide::decide))
        // OpenAI: what fits the `text -> text` contract.
        .route("/v1/models", get(openai::models_list::list))
        .route("/v1/completions", post(openai::completions::create))
        .route("/v1/chat/completions", post(openai::chat::create))
        // axum's built-in 404 has an empty body; callers parse an error
        // object, so answer in the dialect the path implies.
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(Arc::new(state))
}

/// Answer an unrouted path in the caller's own error dialect: OpenAI
/// clients read `{"error":{"message":..,"type":..}}`, KernRT clients read
/// `{"error":"..."}`. Falling back to a bare 404 would break both.
async fn not_found(uri: Uri) -> Response {
    let error = ServeError::not_found("not found");
    if uri.path().starts_with("/v1/") {
        error.openai()
    } else {
        error.native()
    }
}

/// Bind `address` and serve until the process is stopped.
///
/// Owns its own runtime so callers keep a plain `fn main` — no `#[tokio::main]`
/// anywhere in the crate.
pub fn serve(address: &str, registry: Registry) -> Result<(), ServeError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| ServeError::internal(format!("failed to start runtime: {error}")))?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .map_err(|error| ServeError::internal(format!("failed to bind {address}: {error}")))?;
        println!("kernrt serving on http://{address}");
        axum::serve(listener, router(AppState::new(registry)))
            .await
            .map_err(|error| ServeError::internal(format!("server stopped: {error}")))
    })
}
