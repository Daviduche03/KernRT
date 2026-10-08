//! `GET /v1/models`: the generate models, and nothing else.
//!
//! The OpenAI surface has no meaning for a detector or a decision head —
//! they answer on `/infer` and `/decide` — so listing them here would invite
//! a client to call `/v1/completions` with an image model's id.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;

use super::started;
use super::types::{ModelEntry, ModelList};
use crate::serve::AppState;
use crate::serve::error::{ServeError, json_response};
use crate::serve::model::Model;

pub async fn list(State(state): State<Arc<AppState>>) -> Response {
    match render(&state) {
        Ok(body) => json_response(StatusCode::OK, &body),
        Err(error) => error.openai(),
    }
}

fn render(state: &AppState) -> Result<String, ServeError> {
    let data = state
        .registry
        .models()
        .iter()
        .filter(|model| matches!(model.as_ref(), Model::Generate(_)))
        .map(|model| ModelEntry {
            id: model.id().to_string(),
            object: "model",
            created: started(),
            owned_by: "kernrt",
        })
        .collect();
    let list = ModelList {
        object: "list",
        data,
    };
    serde_json::to_string(&list)
        .map_err(|error| ServeError::internal(format!("cannot encode response: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::model::GenOpts;
    use crate::serve::{GenReport, Registry};

    struct StubGenerate(&'static str);

    impl crate::serve::Generate for StubGenerate {
        fn id(&self) -> &str {
            self.0
        }
        fn render(&self, _: &[crate::serve::ChatMessage]) -> Result<String, ServeError> {
            Ok(String::new())
        }
        fn generate(
            &self,
            _: &str,
            _: &GenOpts,
            _: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
        ) -> Result<GenReport, ServeError> {
            Ok(GenReport::default())
        }
    }

    fn state_with(models: Vec<crate::serve::Model>) -> AppState {
        let mut registry = Registry::new();
        for model in models {
            registry.register(model);
        }
        AppState::new(registry)
    }

    #[test]
    fn only_generate_models_are_listed() {
        let state = state_with(vec![
            crate::serve::Model::detect(Detector),
            crate::serve::Model::generate(StubGenerate("smollm2")),
        ]);
        let body = render(&state).unwrap();
        let list: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(list["object"], "list");
        let ids: Vec<&str> = list["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry["id"].as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["smollm2"],
            "detectors do not belong in /v1/models"
        );
    }

    #[test]
    fn a_model_entry_carries_the_fields_clients_read() {
        let state = state_with(vec![crate::serve::Model::generate(StubGenerate("m"))]);
        let list: serde_json::Value = serde_json::from_str(&render(&state).unwrap()).unwrap();
        let entry = &list["data"][0];
        assert_eq!(entry["object"], "model");
        assert_eq!(entry["owned_by"], "kernrt");
        assert!(
            entry["created"].as_u64().is_some(),
            "created must be a unix timestamp"
        );
    }

    struct Detector;

    impl crate::serve::Detect for Detector {
        fn id(&self) -> &str {
            "det"
        }
        fn input_size(&self) -> (usize, usize) {
            (1, 1)
        }
        fn detect(
            &self,
            _: &[u8],
        ) -> Result<Vec<crate::arch::vision::yolo26::Detection>, ServeError> {
            Ok(Vec::new())
        }
    }
}
