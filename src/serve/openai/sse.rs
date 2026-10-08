//! Server-sent events for `stream: true`.
//!
//! Generation runs on a blocking task and pushes decoded text into a channel;
//! this turns the channel into the SSE stream. Running it off the executor
//! keeps a long decode from stalling other requests.

use std::collections::VecDeque;
use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::Stream;
use tokio::sync::mpsc::Receiver;

use crate::generate::Report;

/// One step out of the generation loop.
pub(crate) enum Piece {
    /// A decoded text delta.
    Text(String),
    /// The run ended: token counts, and why it stopped.
    Done(Report),
}

/// Turns one piece into the JSON payloads to send as frames. `FnMut` because
/// the chat stream tags only its first frame with a role; an empty vector
/// drops the piece (used when usage was not requested).
type Renderer = Box<dyn FnMut(&Piece) -> Vec<String> + Send>;

struct State {
    receiver: Receiver<Piece>,
    /// Events produced by the last piece that have not been sent yet. A
    /// `Done` can need two frames — the closing chunk and a usage chunk —
    /// and a stream item is exactly one event, so they queue here.
    pending: VecDeque<Event>,
    render: Renderer,
}

/// The SSE body for one request.
///
/// `render` turns a piece into zero or more JSON payloads; returning an empty
/// vector drops the piece (used when usage was not requested). It is `FnMut`
/// because the chat stream tags only its first frame with a role.
pub(crate) fn body(
    receiver: Receiver<Piece>,
    render: impl FnMut(&Piece) -> Vec<String> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static> {
    let initial = State {
        receiver,
        pending: VecDeque::new(),
        render: Box::new(render),
    };
    let frames = futures_util::stream::unfold(initial, |mut state| async move {
        loop {
            let queued = state.pending.pop_front();
            if let Some(event) = queued {
                return Some((Ok::<_, Infallible>(event), state));
            }
            let piece = match state.receiver.recv().await {
                Some(piece) => piece,
                // The blocking task finished and dropped its sender.
                None => return None,
            };
            for payload in (state.render)(&piece) {
                state.pending.push_back(Event::default().data(payload));
            }
        }
    });
    Sse::new(frames).keep_alive(KeepAlive::default())
}
