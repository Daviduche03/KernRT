//! One error type with two renderings.
//!
//! Native routes render `{"error":"..."}`; the OpenAI surface renders the
//! `{"error":{"message":..,"type":..}}` envelope. Handlers pick the rendering
//! explicitly, so no route can accidentally emit the wrong dialect.

use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

/// A request failed. Carries the HTTP status and a human-readable message.
#[derive(Debug, Clone)]
pub struct ServeError {
    status: StatusCode,
    message: String,
}

impl ServeError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        ServeError {
            status,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Native dialect body: `{"error":"<message>"}`.
    pub fn native_body(&self) -> String {
        format!("{{\"error\":{:?}}}", self.message)
    }

    /// OpenAI dialect body: `{"error":{"message":..,"type":..}}`.
    pub fn openai_body(&self) -> String {
        let kind = if self.status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        format!(
            "{{\"error\":{{\"message\":{:?},\"type\":{kind:?}}}}}",
            self.message
        )
    }

    pub fn native(&self) -> Response {
        json_response(self.status, &self.native_body())
    }

    pub fn openai(&self) -> Response {
        json_response(self.status, &self.openai_body())
    }
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ServeError {}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        self.native()
    }
}

/// JSON bytes with a status.
///
/// The status and content type are applied after `String`'s own
/// `IntoResponse`, which replaces its default `text/plain`.
pub(crate) fn json_response(status: StatusCode, body: &str) -> Response {
    let mut response = body.to_string().into_response();
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_body_escapes_the_message() {
        let error = ServeError::bad_request("expected 12 RGB bytes");
        assert_eq!(error.native_body(), r#"{"error":"expected 12 RGB bytes"}"#);
    }

    #[test]
    fn openai_body_carries_a_type() {
        let error = ServeError::bad_request("no such model");
        assert_eq!(
            error.openai_body(),
            r#"{"error":{"message":"no such model","type":"invalid_request_error"}}"#
        );
    }

    #[test]
    fn server_errors_get_a_server_type() {
        let error = ServeError::internal("boom");
        assert!(error.openai_body().contains(r#""type":"server_error""#));
    }
}
