//! The OpenAI-compatible surface.
//!
//! Only text generation fits this contract — it is `text -> text`. Detection
//! and scoring live in [`crate::serve::native`], because flattening boxes or
//! probabilities into a chat message's `content` string would trade typed
//! output for something the client has to `JSON.parse` against nothing.
//!
//! ```text
//! GET  /v1/models             generate models only
//! POST /v1/completions        raw prompt in, text out
//! POST /v1/chat/completions   messages in, text out (needs a chat_template)
//! ```
//!
//! `stream: true` is answered with `text/event-stream`.

pub mod chat;
pub mod completions;
pub mod convert;
pub mod models_list;
pub mod sse;
pub mod types;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::generate::Finish;

/// Unix seconds now. Unrepresentable clocks fall back to 0 rather than
/// failing a response.
pub(crate) fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// When this process started, so `/v1/models` reports a stable `created`.
pub(crate) fn started() -> u64 {
    static START: OnceLock<u64> = OnceLock::new();
    *START.get_or_init(created)
}

/// `cmpl-<unix>-<counter>`: unique per request, cheap, no uuid dependency.
pub(crate) fn request_id(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "{prefix}-{}-{}",
        created(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn finish_name(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}
