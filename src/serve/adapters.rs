//! Concrete adapters from [`crate::arch`] onto the serving traits.
//!
//! This is the only module in `serve` that names a network type. Swapping
//! YOLO26 for another detector, or NanoJev for another head, touches nothing
//! beyond these two structs.

use std::ops::ControlFlow;
use std::sync::Mutex;

use crate::arch::decision::{DecisionModel, QuestionType};
use crate::arch::llm::{Llama, LoadError};
use crate::arch::vision::yolo26::{Detection, Yolo26};
use crate::generate::{Sampler, generate_streaming};
use crate::tokenizer::BpeTokenizer;

use super::config::ChatTemplate;
use super::error::ServeError;
use super::model::{
    CandidateScore, ChatMessage, Decide, DecideRequest, Detect, GenOpts, GenReport, Generate,
};

/// Serves a [`Yolo26`] detector over raw interleaved RGB frames.
pub struct YoloDetect {
    id: String,
    model: Yolo26,
}

impl YoloDetect {
    pub fn new(id: impl Into<String>, model: Yolo26) -> Self {
        Self {
            id: id.into(),
            model,
        }
    }
}

impl Detect for YoloDetect {
    fn id(&self) -> &str {
        &self.id
    }

    fn input_size(&self) -> (usize, usize) {
        (self.model.input_width(), self.model.input_height())
    }

    fn detect(&self, rgb: &[u8]) -> Result<Vec<Detection>, ServeError> {
        let (width, height) = self.input_size();
        let expected = width * height * 3;
        if rgb.len() != expected {
            return Err(ServeError::bad_request(format!(
                "expected {expected} RGB bytes; a raw frame is exactly \
                 width*height*3 bytes, an image file needs \
                 `content-type: image/<fmt>`, and a URL needs \
                 `content-type: application/json` carrying image_url"
            )));
        }
        let input = self.model.preprocess_rgb(rgb, width, height);
        let output = self
            .model
            .infer_nchw(&input)
            .map_err(|error| ServeError::internal(format!("inference failed: {error}")))?;
        Ok(self.model.detections(&output))
    }
}

/// Serves a NanoJev [`DecisionModel`].
///
/// `decide` takes `&mut self` because the backbone owns the key/value cache,
/// so the model sits behind a lock. The lock is uncontended while a single
/// request runs, which costs nothing measurable.
pub struct NanoJevDecide {
    id: String,
    model: Mutex<DecisionModel>,
    tokenizer: BpeTokenizer,
}

impl NanoJevDecide {
    pub fn new(id: impl Into<String>, model: DecisionModel, tokenizer: BpeTokenizer) -> Self {
        Self {
            id: id.into(),
            model: Mutex::new(model),
            tokenizer,
        }
    }
}

impl Decide for NanoJevDecide {
    fn id(&self) -> &str {
        &self.id
    }

    fn decide(&self, request: &DecideRequest) -> Result<Vec<CandidateScore>, ServeError> {
        if request.kind == QuestionType::Boolean {
            // `DecisionModel::decide` builds a single candidate text for
            // booleans and then rejects it as below the two-candidate
            // minimum, so the path can never succeed. Fail with a clear
            // message instead of a confusing load error.
            return Err(ServeError::bad_request(
                "boolean questions are not supported yet",
            ));
        }
        let criteria: Vec<(&str, &str)> = request
            .criteria
            .iter()
            .map(|(id, description)| (id.as_str(), description.as_str()))
            .collect();

        let mut model = self
            .model
            .lock()
            .map_err(|_| ServeError::internal("decision model lock is poisoned"))?;
        let probabilities = model
            .decide(
                &request.state,
                request.kind,
                &request.instructions,
                &criteria,
                &self.tokenizer,
            )
            .map_err(decide_error)?;

        if probabilities.len() != request.criteria.len() {
            return Err(ServeError::internal(format!(
                "head returned {} probabilities for {} candidates",
                probabilities.len(),
                request.criteria.len()
            )));
        }
        Ok(request
            .criteria
            .iter()
            .zip(probabilities)
            .map(|((id, _), probability)| CandidateScore {
                id: id.clone(),
                probability,
            })
            .collect())
    }
}

/// How a failure out of the decision model maps onto an HTTP status.
///
/// The weights are loaded once at startup, so anything the model reports
/// while serving is a function of *this* request: too few candidates, or
/// state plus criteria that do not fit the context window. Those are the
/// caller's to fix, so they are 400s — reporting them as 500 blames the
/// server for the client's input. Anything else was not caused by the
/// request and stays a 500.
fn decide_error(error: LoadError) -> ServeError {
    match error {
        LoadError::BadConfig(message) => ServeError::bad_request(message),
        error => ServeError::internal(format!("decision failed: {error}")),
    }
}

/// Serves a [`Llama`] decoder over raw prompts and chat messages.
///
/// The model sits behind a lock because decoding owns the key/value cache.
/// The lock is uncontended while a single request runs.
pub struct LlamaGenerate {
    id: String,
    model: Mutex<Llama>,
    tokenizer: BpeTokenizer,
    template: Option<ChatTemplate>,
}

impl LlamaGenerate {
    pub fn new(
        id: impl Into<String>,
        model: Llama,
        tokenizer: BpeTokenizer,
        template: Option<ChatTemplate>,
    ) -> Self {
        Self {
            id: id.into(),
            model: Mutex::new(model),
            tokenizer,
            template,
        }
    }

    fn sampler(opts: &GenOpts) -> Sampler {
        if opts.temperature <= 0.0 {
            Sampler::greedy()
        } else {
            Sampler::temperature(opts.temperature, opts.seed.unwrap_or(0))
        }
    }
}

impl Generate for LlamaGenerate {
    fn id(&self) -> &str {
        &self.id
    }

    fn render(&self, messages: &[ChatMessage]) -> Result<String, ServeError> {
        let template = self.template.as_ref().ok_or_else(|| {
            ServeError::bad_request(format!(
                "model {:?} has no chat template; add \"chat_template\" to its manifest entry, \
                 or call /v1/completions with a raw prompt",
                self.id
            ))
        })?;
        template.render(messages)
    }

    fn generate(
        &self,
        prompt: &str,
        opts: &GenOpts,
        emit: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenReport, ServeError> {
        // Accepted for API compatibility, but nothing honours them: failing
        // loudly beats producing output that quietly ignores them.
        if opts.top_p != 1.0 {
            return Err(ServeError::bad_request(
                "top_p is not supported yet; omit it or send 1.0",
            ));
        }
        if !opts.stop.is_empty() {
            return Err(ServeError::bad_request(
                "stop sequences are not supported yet; use max_tokens instead",
            ));
        }
        let prompt_tokens = self.tokenizer.encode(prompt, false);
        if prompt_tokens.is_empty() {
            return Err(ServeError::bad_request(
                "prompt produced no tokens; the prompt must not be empty",
            ));
        }

        let mut sampler = Self::sampler(opts);
        let mut model = self
            .model
            .lock()
            .map_err(|_| ServeError::internal("language model lock is poisoned"))?;
        Ok(generate_streaming(
            &mut *model,
            &self.tokenizer,
            &mut sampler,
            &prompt_tokens,
            opts.max_tokens,
            emit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn a_request_that_does_not_fit_the_model_is_a_bad_request() {
        for message in [
            "need 2-255 candidates, have 1",
            "2780 leaf tokens exceed the 2048 context",
        ] {
            let error = decide_error(LoadError::BadConfig(message.to_string()));
            assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{message}");
            assert_eq!(error.message(), message, "keep the model's detail");
        }
    }

    #[test]
    fn a_failure_the_request_cannot_explain_stays_a_server_error() {
        let error = decide_error(LoadError::MissingTensor("q_proj".into()));
        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            error.message().starts_with("decision failed:"),
            "{}",
            error.message()
        );
    }
}
