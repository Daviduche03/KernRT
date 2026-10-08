//! Neutral request types and the three capability traits.
//!
//! Wire schemas live in [`crate::serve::openai`] and [`crate::serve::native`];
//! they convert into these types. Nothing in [`crate::arch`] learns about
//! HTTP, and nothing here learns about JSON field names — so adding a dialect
//! costs a new folder, not a new trait method.

use std::ops::ControlFlow;
use std::sync::Arc;

use crate::arch::decision::QuestionType;
use crate::arch::vision::yolo26::Detection;

use super::error::ServeError;

/// What a loaded model can do. Drives route dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Detect,
    Decide,
    Generate,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Detect => "detect",
            Kind::Decide => "decide",
            Kind::Generate => "generate",
        }
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Speaker of a chat message. Kept deliberately small: the OpenAI surface maps
/// its `string | array of parts` content down to a plain string here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    /// The spelling OpenAI's `messages[].role` uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Sampling controls, normalised to the forms the runtime actually honours.
///
/// [`GenOpts::top_p`] and [`GenOpts::stop`] exist because clients send them;
/// the generator rejects non-default values rather than silently ignoring
/// them. `temperature` defaults to 0.0 (greedy), matching every example in
/// this crate — deliberately not OpenAI's 1.0, so a default request is
/// reproducible.
#[derive(Debug, Clone, PartialEq)]
pub struct GenOpts {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            stop: Vec::new(),
            seed: None,
        }
    }
}

pub use crate::generate::{Finish, Report as GenReport, Usage as GenUsage};

/// Why a question is being asked, plus its candidates.
///
/// `criteria` is `(id, description)` in order: choice candidates format as
/// `"{id}: {description}"`, score levels pass their labels as descriptions.
#[derive(Debug, Clone, PartialEq)]
pub struct DecideRequest {
    pub state: String,
    pub kind: QuestionType,
    pub instructions: String,
    pub criteria: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateScore {
    pub id: String,
    pub probability: f32,
}

/// A loaded model, classified by what it can do.
pub enum Model {
    Detect(Arc<dyn Detect>),
    Decide(Arc<dyn Decide>),
    Generate(Arc<dyn Generate>),
}

impl Model {
    pub fn detect(value: impl Detect + 'static) -> Self {
        Model::Detect(Arc::new(value))
    }

    pub fn decide(value: impl Decide + 'static) -> Self {
        Model::Decide(Arc::new(value))
    }

    pub fn generate(value: impl Generate + 'static) -> Self {
        Model::Generate(Arc::new(value))
    }

    pub fn id(&self) -> &str {
        match self {
            Model::Detect(model) => model.id(),
            Model::Decide(model) => model.id(),
            Model::Generate(model) => model.id(),
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Model::Detect(_) => Kind::Detect,
            Model::Decide(_) => Kind::Decide,
            Model::Generate(_) => Kind::Generate,
        }
    }
}

/// An image in, boxes out. Owns its own size contract so the HTTP layer can
/// validate a raw RGB body before any preprocessing runs.
pub trait Detect: Send + Sync {
    fn id(&self) -> &str;

    /// `(width, height)` of the frame the model expects, in pixels.
    fn input_size(&self) -> (usize, usize);

    fn detect(&self, rgb: &[u8]) -> Result<Vec<Detection>, ServeError>;
}

/// A structured scoring question: state plus candidates in, probabilities out.
pub trait Decide: Send + Sync {
    fn id(&self) -> &str;

    fn decide(&self, request: &DecideRequest) -> Result<Vec<CandidateScore>, ServeError>;
}

/// Autoregressive text generation.
///
/// `emit` is a callback rather than an iterator because the decoder holds
/// `&mut` state internally: the handler decides whether the tokens are
/// appended to a `String` (JSON response) or pushed as SSE frames (streaming).
/// One implementation serves both shapes.
pub trait Generate: Send + Sync {
    fn id(&self) -> &str;

    /// Apply this model's chat template. Returns the prompt to run.
    fn render(&self, messages: &[ChatMessage]) -> Result<String, ServeError>;

    /// Produce up to `opts.max_tokens` tokens, calling `emit` with each
    /// decoded piece. Returning [`ControlFlow::Break`] stops early.
    fn generate(
        &self,
        prompt: &str,
        opts: &GenOpts,
        emit: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenReport, ServeError>;
}
