//! The KernRT-native HTTP surface, driven in-process.
//!
//! No port is bound and no weight file is read: the tests register stub
//! models so what is under test is routing, query parsing, status codes and
//! body shapes. The numerics are gated separately by
//! `cargo run --release --example yolo26_server -- MODEL --check N`.
#![cfg(feature = "serve")]

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use kernrt::arch::decision::QuestionType;
use kernrt::arch::vision::yolo26::Detection;
use kernrt::serve::{
    AppState, CandidateScore, Decide, DecideRequest, Detect, Manifest, Model, Registry, ServeError,
};
use tower::ServiceExt;

struct StubDetect {
    id: &'static str,
    width: usize,
    height: usize,
}

impl Detect for StubDetect {
    fn id(&self) -> &str {
        self.id
    }

    fn input_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn detect(&self, rgb: &[u8]) -> Result<Vec<Detection>, ServeError> {
        let expected = self.width * self.height * 3;
        if rgb.len() != expected {
            return Err(ServeError::bad_request(format!(
                "expected {expected} RGB bytes"
            )));
        }
        Ok(vec![Detection {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            score: 0.5,
            class_id: 7,
        }])
    }
}

struct StubDecide {
    id: &'static str,
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
        Ok(request
            .criteria
            .iter()
            .enumerate()
            .map(|(index, (id, _))| CandidateScore {
                id: id.clone(),
                probability: 1.0 / (index + 1) as f32,
            })
            .collect())
    }
}

fn registry(models: Vec<Model>) -> Registry {
    let mut registry = Registry::new();
    for model in models {
        registry.register(model);
    }
    registry
}

fn detector_app() -> Router {
    let registry = registry(vec![Model::detect(StubDetect {
        id: "det",
        width: 2,
        height: 2,
    })]);
    kernrt::serve::router(AppState::new(registry))
}

fn both_app() -> Router {
    let registry = registry(vec![
        Model::detect(StubDetect {
            id: "det",
            width: 2,
            height: 2,
        }),
        Model::decide(StubDecide { id: "jev" }),
    ]);
    kernrt::serve::router(AppState::new(registry))
}

/// Four pixels, three channels: 12 bytes for the 2x2 stub detector.
const FRAME: &[u8] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, //
];

async fn send(app: Router, request: Request<Body>) -> (StatusCode, String, String) {
    let response = app.oneshot(request).await.expect("router panics");
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|value| value.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .expect("body reads");
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

fn post(path: &str, body: Body) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .body(body)
        .unwrap()
}

fn post_json(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::builder().uri(path).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn health_reports_ok_as_json() {
    let (status, content_type, body) = send(detector_app(), get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    assert_eq!(body, r#"{"status":"ok"}"#);
}

#[tokio::test]
async fn infer_returns_the_detection_payload() {
    let (status, content_type, body) =
        send(detector_app(), post("/infer", Body::from(FRAME))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    assert_eq!(
        body,
        r#"{"detections":[{"x1":1.0000,"y1":2.0000,"x2":3.0000,"y2":4.0000,"score":0.500000,"class":7}]}"#
    );
}

#[tokio::test]
async fn infer_accepts_a_model_query_and_ignores_it_when_unambiguous() {
    let expected = detector_app()
        .oneshot(post("/infer", Body::from(FRAME)))
        .await
        .unwrap();
    let expected = to_bytes(expected.into_body(), 1024 * 1024).await.unwrap();
    let (status, _, body) = send(detector_app(), post("/infer?model=det", Body::from(FRAME))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_bytes(), &expected[..]);
}

#[tokio::test]
async fn infer_rejects_a_body_of_the_wrong_length_with_the_original_message() {
    let (status, _, body) = send(detector_app(), post("/infer", Body::from("abc"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, r#"{"error":"expected 12 RGB bytes"}"#);
}

#[tokio::test]
async fn infer_reports_an_unknown_model_as_404() {
    let (status, _, body) =
        send(detector_app(), post("/infer?model=nope", Body::from(FRAME))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error":"no detect model named nope"}"#);
}

#[tokio::test]
async fn decide_returns_one_probability_per_candidate() {
    let request = post_json(
        "/decide",
        r#"{"state":"s","instructions":"i","criteria":[
            {"id":"left","description":"Move left."},
            {"id":"right","description":"Move right."}]}"#,
    );
    let (status, content_type, body) = send(both_app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    assert_eq!(
        body,
        r#"{"model":"jev","scores":[{"id":"left","probability":1.000000},{"id":"right","probability":0.500000}]}"#
    );
}

#[tokio::test]
async fn decide_defaults_to_choice() {
    let request = post_json(
        "/decide",
        r#"{"state":"s","instructions":"i","criteria":[{"id":"a","description":"d"}]}"#,
    );
    let (status, _, body) = send(both_app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains(r#"{"id":"a","probability":"#), "{body}");
}

#[tokio::test]
async fn decide_rejects_an_unknown_kind() {
    let request = post_json(
        "/decide",
        r#"{"type":"meh","state":"s","instructions":"i","criteria":[]}"#,
    );
    let (status, _, body) = send(both_app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body,
        r#"{"error":"type must be one of boolean, choice, score; got \"meh\""}"#
    );
}

#[tokio::test]
async fn decide_fails_loudly_on_boolean() {
    let request = post_json(
        "/decide",
        r#"{"type":"boolean","state":"s","instructions":"i","criteria":[]}"#,
    );
    let (status, _, body) = send(both_app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body,
        r#"{"error":"boolean questions are not supported yet"}"#
    );
}

#[tokio::test]
async fn decide_reports_an_unknown_model_as_404() {
    let request = post_json(
        "/decide",
        r#"{"model":"nope","state":"s","instructions":"i","criteria":[]}"#,
    );
    let (status, _, body) = send(both_app(), request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error":"no decide model named nope"}"#);
}

#[tokio::test]
async fn an_unknown_path_keeps_the_json_404_body() {
    let (status, _, body) = send(detector_app(), get("/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error":"not found"}"#);
}

#[tokio::test]
async fn routes_do_not_leak_between_kinds() {
    let (status, _, body) = send(both_app(), post("/infer?model=jev", Body::from(FRAME))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error":"no detect model named jev"}"#);
}

#[tokio::test]
async fn a_route_is_unreachable_when_no_model_of_that_kind_is_loaded() {
    let registry = registry(vec![Model::detect(StubDetect {
        id: "det",
        width: 2,
        height: 2,
    })]);
    let app = kernrt::serve::router(AppState::new(registry));
    let request = post_json(
        "/decide",
        r#"{"state":"s","instructions":"i","criteria":[]}"#,
    );
    let (status, _, body) = send(app, request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error":"no decide models are loaded"}"#);
}

/// The shipped manifest is the demo, so it must cover every kind and stay
/// loadable. Listing the pairs rather than a count keeps this useful when
/// another model is added.
#[test]
fn the_shipped_manifest_parses() {
    let manifest = Manifest::load("models.json").expect("models.json in the repo root");
    let kinds: Vec<(&str, &str)> = manifest
        .models
        .iter()
        .map(|entry| (entry.id.as_str(), entry.kind.as_str()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("yolo26n", "detect"),
            ("jev", "decide"),
            ("smollm2", "generate"),
        ],
        "one of every kind, so the demo exercises all three routes"
    );
    let chat = manifest.models.last().expect("smollm2 is listed");
    assert!(
        chat.chat_template.is_some(),
        "without a template /v1/chat/completions cannot answer"
    );
}

/// A one-pixel PNG: decoded, stretched to the model's 2x2, then detected.
/// Goes through the whole router, not just the loader.
#[tokio::test]
async fn infer_accepts_an_image_file_body() {
    let (status, _, body) = send(
        detector_app(),
        post_typed("/infer", "image/png", Body::from(png(1, 1))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains(r#""class":7"#), "{body}");
}

/// `data:` URIs are the second form OpenAI's `image_url` accepts, so a
/// client written against their vision API works unchanged.
#[tokio::test]
async fn infer_accepts_a_data_uri() {
    use base64::Engine;
    let payload = base64::engine::general_purpose::STANDARD.encode(png(1, 1));
    let request = post_json(
        "/infer",
        &format!(r#"{{"image_url":"data:image/png;base64,{payload}"}}"#),
    );
    let (status, _, body) = send(detector_app(), request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains(r#""class":7"#), "{body}");
}

#[tokio::test]
async fn a_json_body_without_image_url_says_which_field_is_missing() {
    let (status, _, body) = send(detector_app(), post_json("/infer", "{}")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("image_url"),
        "the message should name the field: {body}"
    );
}

/// Only http(s) and `data:` — nothing that could reach the local disk.
#[tokio::test]
async fn a_url_we_will_not_fetch_is_rejected_before_any_request() {
    for url in ["file:///etc/passwd", "/etc/passwd", "ftp://x/y"] {
        let request = post_json("/infer", &format!(r#"{{"image_url":"{url}"}}"#));
        let (status, _, body) = send(detector_app(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{url}: {body}");
        assert!(body.contains("http:// or https://"), "{url}: {body}");
    }
}

/// The fastest path must keep working exactly as it did.
#[tokio::test]
async fn a_raw_rgb_frame_still_works() {
    let (status, _, body) = send(detector_app(), post("/infer", Body::from(FRAME))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains(r#""x1":1.0000"#), "{body}");
}

fn post_typed(path: &str, content_type: &str, body: Body) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, content_type)
        .body(body)
        .unwrap()
}

/// A `width * height` RGB image as PNG, encoded by the crate under test.
fn png(width: u32, height: u32) -> Vec<u8> {
    use image::ImageEncoder;
    let pixels = vec![42u8; (width * height * 3) as usize];
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(&pixels, width, height, image::ExtendedColorType::Rgb8)
        .unwrap();
    bytes
}
