//! `POST /decide`: a structured scoring question in, probabilities per
//! candidate out.
//!
//! This is the shape OpenAI's chat schema cannot carry — a decision head
//! emits scores, not prose — which is why the native surface exists
//! alongside the OpenAI one.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::sync::Arc;

use crate::arch::decision::QuestionType;
use crate::serve::AppState;
use crate::serve::error::{ServeError, json_response};
use crate::serve::model::DecideRequest;

#[derive(Debug, Deserialize)]
pub struct DecideBody {
    /// Disambiguates when several decision models are loaded.
    #[serde(default)]
    pub model: Option<String>,
    /// World state the question is asked about.
    pub state: String,
    /// `"boolean" | "choice" | "score"`. Omitted means `"choice"`.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub instructions: String,
    #[serde(default)]
    pub criteria: Vec<Criterion>,
}

#[derive(Debug, Deserialize)]
pub struct Criterion {
    pub id: String,
    pub description: String,
}

pub async fn decide(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<DecideBody>,
) -> Response {
    match decide_inner(&state, body) {
        Ok(payload) => json_response(StatusCode::OK, &payload),
        Err(error) => error.into_response(),
    }
}

fn decide_inner(state: &AppState, body: DecideBody) -> Result<String, ServeError> {
    let kind = match body.kind.as_deref() {
        None | Some("choice") => QuestionType::Choice,
        Some("score") => QuestionType::Score,
        Some("boolean") => QuestionType::Boolean,
        Some(other) => {
            return Err(ServeError::bad_request(format!(
                "type must be one of boolean, choice, score; got {other:?}"
            )));
        }
    };
    let model = state.registry.decide(body.model.as_deref())?;
    let request = DecideRequest {
        state: body.state,
        kind,
        instructions: body.instructions,
        criteria: body
            .criteria
            .into_iter()
            .map(|criterion| (criterion.id, criterion.description))
            .collect(),
    };
    let scores = model.decide(&request)?;

    let mut json = format!("{{\"model\":{:?},\"scores\":[", model.id());
    for (index, score) in scores.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"id\":{:?},\"probability\":{:.6}}}",
            score.id, score.probability
        ));
    }
    json.push_str("]}");
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::Registry;
    use crate::serve::model::{CandidateScore, Decide, Model};

    struct StubDecide {
        id: &'static str,
        fail_boolean: bool,
    }

    impl Decide for StubDecide {
        fn id(&self) -> &str {
            self.id
        }

        fn decide(&self, request: &DecideRequest) -> Result<Vec<CandidateScore>, ServeError> {
            if request.kind == QuestionType::Boolean {
                return Err(ServeError::bad_request(
                    "boolean questions are not supported yet",
                ));
            }
            if self.fail_boolean {
                unreachable!("stub flag is only read for boolean requests");
            }
            Ok(request
                .criteria
                .iter()
                .map(|(id, _)| CandidateScore {
                    id: id.clone(),
                    probability: 0.5,
                })
                .collect())
        }
    }

    fn state_with(model: StubDecide) -> AppState {
        let mut registry = Registry::new();
        registry.register(Model::decide(model));
        AppState::new(registry)
    }

    fn body(kind: Option<&str>, criteria: &[(&str, &str)]) -> DecideBody {
        DecideBody {
            model: None,
            state: "aim is centred".into(),
            kind: kind.map(str::to_string),
            instructions: "Choose the next action.".into(),
            criteria: criteria
                .iter()
                .map(|(id, description)| Criterion {
                    id: (*id).to_string(),
                    description: (*description).to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn choice_is_the_default_kind() {
        let state = state_with(StubDecide {
            id: "jev",
            fail_boolean: false,
        });
        let payload = decide_inner(&state, body(None, &[("left", "Move left")])).unwrap();
        assert!(
            payload.starts_with(r#"{"model":"jev","scores":["#),
            "{payload}"
        );
    }

    #[test]
    fn an_unknown_kind_is_rejected() {
        let state = state_with(StubDecide {
            id: "jev",
            fail_boolean: false,
        });
        let error = decide_inner(&state, body(Some("meh"), &[])).unwrap_err();
        assert_eq!(
            error.message(),
            r#"type must be one of boolean, choice, score; got "meh""#
        );
    }

    #[test]
    fn boolean_fails_loudly_instead_of_erroring_deep_in_the_head() {
        let state = state_with(StubDecide {
            id: "jev",
            fail_boolean: true,
        });
        let error = decide_inner(&state, body(Some("boolean"), &[("false", "x")])).unwrap_err();
        assert_eq!(error.message(), "boolean questions are not supported yet");
    }

    #[test]
    fn an_unknown_model_is_a_404() {
        let state = state_with(StubDecide {
            id: "jev",
            fail_boolean: false,
        });
        let mut request = body(None, &[("left", "Move left")]);
        request.model = Some("nope".into());
        let error = decide_inner(&state, request).unwrap_err();
        assert_eq!(error.status(), StatusCode::NOT_FOUND);
    }
}
