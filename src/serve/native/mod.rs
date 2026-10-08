//! The KernRT-native surface: raw bytes in, typed JSON out.

pub mod decide;
pub mod detect;

use axum::response::Response;

use crate::serve::error::json_response;

pub async fn health() -> Response {
    json_response(axum::http::StatusCode::OK, "{\"status\":\"ok\"}")
}
