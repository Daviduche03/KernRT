//! Which model serves a request.
//!
//! Selection is by kind first (a route only ever reaches one kind), then by
//! `?model=<id>`. An omitted id is only legal when the answer is unambiguous.

use std::sync::Arc;

use super::error::ServeError;
use super::model::{Decide, Detect, Generate, Kind, Model};

#[derive(Clone, Default)]
pub struct Registry {
    models: Vec<Arc<Model>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, model: Model) -> &mut Self {
        self.models.push(Arc::new(model));
        self
    }

    pub fn models(&self) -> &[Arc<Model>] {
        &self.models
    }

    pub fn detect(&self, id: Option<&str>) -> Result<&Arc<dyn Detect>, ServeError> {
        self.select(Kind::Detect, id, |model| match model {
            Model::Detect(found) => Some(found),
            _ => None,
        })
    }

    pub fn decide(&self, id: Option<&str>) -> Result<&Arc<dyn Decide>, ServeError> {
        self.select(Kind::Decide, id, |model| match model {
            Model::Decide(found) => Some(found),
            _ => None,
        })
    }

    pub fn generate(&self, id: Option<&str>) -> Result<&Arc<dyn Generate>, ServeError> {
        self.select(Kind::Generate, id, |model| match model {
            Model::Generate(found) => Some(found),
            _ => None,
        })
    }

    /// One candidate of `kind`, optionally narrowed by `id`.
    fn select<T: ?Sized>(
        &self,
        kind: Kind,
        id: Option<&str>,
        get: impl Fn(&Model) -> Option<&Arc<T>>,
    ) -> Result<&Arc<T>, ServeError> {
        let mut matches = 0usize;
        let mut found: Option<&Arc<T>> = None;
        for model in self.models.iter().map(Arc::as_ref) {
            if model.kind() != kind {
                continue;
            }
            if id.is_some_and(|wanted| model.id() != wanted) {
                continue;
            }
            matches += 1;
            found = get(model);
        }
        if matches > 1 {
            return Err(ServeError::bad_request(format!(
                "{matches} {kind} models are loaded; pass ?model=<id>"
            )));
        }
        found.ok_or_else(|| match id {
            Some(wanted) => ServeError::not_found(format!("no {kind} model named {wanted}")),
            None => ServeError::not_found(format!("no {kind} models are loaded")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::vision::yolo26::Detection;
    use crate::serve::model::DecideRequest;

    struct StubDetect(&'static str);
    impl Detect for StubDetect {
        fn id(&self) -> &str {
            self.0
        }
        fn input_size(&self) -> (usize, usize) {
            (1, 1)
        }
        fn detect(&self, _: &[u8]) -> Result<Vec<Detection>, ServeError> {
            Ok(Vec::new())
        }
    }

    struct StubDecide(&'static str);
    impl Decide for StubDecide {
        fn id(&self) -> &str {
            self.0
        }
        fn decide(
            &self,
            _: &DecideRequest,
        ) -> Result<Vec<crate::serve::model::CandidateScore>, ServeError> {
            Ok(Vec::new())
        }
    }

    /// `Result::unwrap_err` needs `T: Debug`, which the trait objects are not.
    fn error<T>(result: Result<T, ServeError>) -> ServeError {
        result.err().expect("expected an error")
    }

    #[test]
    fn a_single_model_needs_no_id() {
        let mut registry = Registry::new();
        registry.register(Model::detect(StubDetect("yolo26n")));
        assert_eq!(registry.detect(None).unwrap().id(), "yolo26n");
    }

    #[test]
    fn two_models_are_ambiguous_without_an_id() {
        let mut registry = Registry::new();
        registry.register(Model::detect(StubDetect("a")));
        registry.register(Model::detect(StubDetect("b")));
        let error = error(registry.detect(None));
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message().contains("pass ?model="), "{}", error);
    }

    #[test]
    fn an_id_disambiguates() {
        let mut registry = Registry::new();
        registry.register(Model::detect(StubDetect("a")));
        registry.register(Model::detect(StubDetect("b")));
        assert_eq!(registry.detect(Some("b")).unwrap().id(), "b");
    }

    #[test]
    fn kinds_do_not_leak_across_routes() {
        let mut registry = Registry::new();
        registry.register(Model::decide(StubDecide("jev")));
        let error = error(registry.detect(None));
        assert_eq!(error.status(), axum::http::StatusCode::NOT_FOUND);
        assert_eq!(error.message(), "no detect models are loaded");
    }

    #[test]
    fn an_unknown_id_is_a_404() {
        let mut registry = Registry::new();
        registry.register(Model::detect(StubDetect("a")));
        let error = error(registry.detect(Some("nope")));
        assert_eq!(error.status(), axum::http::StatusCode::NOT_FOUND);
        assert_eq!(error.message(), "no detect model named nope");
    }
}
