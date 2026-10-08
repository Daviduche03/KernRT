//! `POST /infer`: an image in, JSON detections out.
//!
//! Three input shapes, all ending in the same RGB frame the detector reads:
//!
//! * **raw RGB** — exactly `width * height * 3` interleaved bytes, with no
//!   `Content-Type`. This is the fastest path and what the benches post.
//! * **an image file** — `Content-Type: image/jpeg` (or png, webp, ...):
//!   decode, then resample to the model's input size.
//! * **JSON** — `Content-Type: application/json` with
//!   `{"image_url": "https://..."}`, or the same field holding a `data:`
//!   URI.
//!
//! Fetching and decoding belong here rather than behind a build flag: a
//! caller holding an image URL should not have to know how to decode it.
//!
//! The fetch is bounded (15 s, 32 MB, 5 redirects) and only accepts
//! `http`/`https`. It is **not** filtered for private addresses — an edge
//! server is usually expected to reach hosts on its own network. Expose
//! this beyond a trusted network and you have an SSRF hole.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::arch::vision::yolo26::Detection;
use crate::serve::AppState;
use crate::serve::error::{ServeError, json_response};
use crate::serve::image as frame_input;

/// How long a single `image_url` fetch may take.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
/// Redirects to follow before giving up.
const FETCH_REDIRECTS: usize = 5;

#[derive(Debug, Default, Deserialize)]
pub struct InferQuery {
    /// Disambiguates when several detectors are loaded.
    #[serde(default)]
    pub model: Option<String>,
}

/// The JSON body of `/infer`.
#[derive(Debug, Deserialize)]
struct ImageRequest {
    /// An `http(s)` URL, or a `data:[<mime>];base64,<payload>` URI.
    image_url: String,
}

pub async fn infer(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InferQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model = match state.registry.detect(query.model.as_deref()) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };
    let rgb = match load_frame(&headers, &body, model.input_size()).await {
        Ok(rgb) => rgb,
        Err(error) => return error.into_response(),
    };
    match model.detect(&rgb) {
        Ok(detections) => json_response(StatusCode::OK, &detections_json(&detections)),
        Err(error) => error.into_response(),
    }
}

/// Resolve the request body to RGB at `(width, height)`.
///
/// The raw branch hands the bytes straight through: sizing is the
/// detector's job there, and it already reports `expected N RGB bytes`.
async fn load_frame(
    headers: &HeaderMap,
    body: &[u8],
    (width, height): (usize, usize),
) -> Result<Vec<u8>, ServeError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    let frame = if content_type == "application/json" {
        let request: ImageRequest = serde_json::from_slice(body).map_err(|error| {
            // serde's message already names the field it wanted.
            ServeError::bad_request(format!("cannot parse the request body as JSON: {error}"))
        })?;
        resolve(&request.image_url).await?
    } else if content_type.starts_with("image/") {
        frame_input::decode(body)?
    } else {
        return Ok(body.to_vec());
    };
    Ok(frame_input::resize(frame, width, height).rgb)
}

/// Bytes for one `image_url`: a `data:` URI, or something to fetch.
async fn resolve(url: &str) -> Result<frame_input::Frame, ServeError> {
    if url.starts_with("data:") {
        return frame_input::from_data_uri(url);
    }
    frame_input::decode(&fetch(url).await?)
}

/// Download one image URL.
async fn fetch(url: &str) -> Result<Vec<u8>, ServeError> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(ServeError::bad_request(format!(
            "image_url must start with http:// or https://, or be a data: URI; got {url:?}"
        )));
    }
    let mut response = http_client()?
        .get(url)
        .send()
        .await
        .map_err(|error| ServeError::bad_request(format!("cannot fetch image_url: {error}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(ServeError::bad_request(format!(
            "image_url returned HTTP {status}"
        )));
    }

    // Read in chunks so a hostile or careless server cannot stream forever
    // before the size check ever runs.
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| ServeError::bad_request(format!("cannot read image_url body: {error}")))?
    {
        if bytes.len() + chunk.len() > frame_input::MAX_IMAGE_BYTES {
            return Err(ServeError::bad_request(format!(
                "image_url body exceeds the {} byte limit",
                frame_input::MAX_IMAGE_BYTES
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// One shared client, so every fetch reuses a connection pool instead of
/// paying for a TLS handshake per request.
fn http_client() -> Result<&'static reqwest::Client, ServeError> {
    static CLIENT: OnceLock<std::result::Result<reqwest::Client, String>> = OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::limited(FETCH_REDIRECTS))
            .build()
            .map_err(|error| error.to_string())
    });
    client
        .as_ref()
        .map_err(|error| ServeError::internal(format!("cannot build the HTTP client: {error}")))
}

/// The detection payload, formatted by hand so the bytes on the wire do not
/// drift when the JSON layer changes: `f32` through `serde_json` prints the
/// shortest round-trip form, which is not the `{:.4}` / `{:.6}` fixed width
/// callers already parse.
pub fn detections_json(detections: &[Detection]) -> String {
    let mut json = String::from("{\"detections\":[");
    for (index, d) in detections.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"x1\":{:.4},\"y1\":{:.4},\"x2\":{:.4},\"y2\":{:.4},\"score\":{:.6},\"class\":{}}}",
            d.x1, d.y1, d.x2, d.y2, d.score, d.class_id
        ));
    }
    json.push_str("]}");
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(score: f32, class_id: usize) -> Detection {
        Detection {
            x1: 1.5,
            y1: 2.25,
            x2: 3.125,
            y2: 4.0625,
            score,
            class_id,
        }
    }

    /// The wire format is a compatibility promise; fix-width formatting must
    /// not silently become shortest-round-trip.
    #[test]
    fn detections_json_keeps_fixed_width_fields() {
        let json = detections_json(&[detection(0.875, 3)]);
        assert_eq!(
            json,
            r#"{"detections":[{"x1":1.5000,"y1":2.2500,"x2":3.1250,"y2":4.0625,"score":0.875000,"class":3}]}"#
        );
    }

    #[test]
    fn detections_json_joins_multiple_rows_with_commas() {
        let json = detections_json(&[detection(0.5, 0), detection(0.25, 7)]);
        assert!(json.starts_with(r#"{"detections":[{"x1""#));
        assert_eq!(json.matches("},{").count(), 1);
        assert!(json.ends_with(r#","class":7}]}"#));
    }

    #[test]
    fn an_empty_frame_is_a_valid_payload() {
        assert_eq!(detections_json(&[]), r#"{"detections":[]}"#);
    }

    fn content_type(value: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, value.parse().unwrap());
        headers
    }

    #[tokio::test]
    async fn a_json_body_is_the_image_url_branch() {
        let headers = content_type("application/json");
        let error = load_frame(&headers, b"{}", (2, 2))
            .await
            .expect_err("an empty object has no image_url");
        assert!(
            error.message().contains("image_url"),
            "the message should name the field: {error}"
        );
    }

    #[tokio::test]
    async fn an_unsupported_scheme_is_rejected_before_any_request() {
        let headers = content_type("application/json");
        let body = br#"{"image_url":"file:///etc/passwd"}"#;
        let error = load_frame(&headers, body, (2, 2))
            .await
            .expect_err("only http(s) and data: are fetchable");
        assert!(error.message().contains("http:// or https://"), "{error}");
    }

    /// The most common mistake: `curl -d` sends
    /// `application/x-www-form-urlencoded`, so the JSON never gets parsed.
    #[tokio::test]
    async fn a_body_without_a_usable_content_type_is_taken_as_raw_rgb() {
        let headers = HeaderMap::new();
        let body = br#"{"image_url":"https://x/y.jpg"}"#;
        // Passed straight through — sizing is the detector's job, and the
        // detector will report the byte count it wanted.
        assert_eq!(load_frame(&headers, body, (2, 2)).await.unwrap(), body);
    }

    #[tokio::test]
    async fn image_content_types_decode_and_resample() {
        let mut bytes = Vec::new();
        {
            use image::ImageEncoder;
            image::codecs::png::PngEncoder::new(&mut bytes)
                .write_image(&[7u8; 3], 1, 1, image::ExtendedColorType::Rgb8)
                .unwrap();
        }
        let headers = content_type("image/png");
        let rgb = load_frame(&headers, &bytes, (2, 2))
            .await
            .expect("a one-pixel PNG decodes");
        assert_eq!(
            rgb,
            vec![7; 2 * 2 * 3],
            "the single pixel is stretched to 2x2"
        );
    }

    #[tokio::test]
    async fn image_bytes_with_a_wrong_content_type_still_decode() {
        let mut bytes = Vec::new();
        {
            use image::ImageEncoder;
            image::codecs::png::PngEncoder::new(&mut bytes)
                .write_image(&[9u8; 3], 1, 1, image::ExtendedColorType::Rgb8)
                .unwrap();
        }
        let headers = content_type("image/jpeg");
        let rgb = load_frame(&headers, &bytes, (1, 1))
            .await
            .expect("the magic bytes decide, not the header");
        assert_eq!(rgb, vec![9, 9, 9]);
    }

    #[tokio::test]
    async fn a_data_uri_reaches_the_detector_without_a_fetch() {
        let mut bytes = Vec::new();
        {
            use image::ImageEncoder;
            image::codecs::png::PngEncoder::new(&mut bytes)
                .write_image(&[1u8, 2, 3], 1, 1, image::ExtendedColorType::Rgb8)
                .unwrap();
        }
        use base64::Engine;
        let payload = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let headers = content_type("application/json");
        let body = format!(r#"{{"image_url":"data:image/png;base64,{payload}"}}"#);
        let rgb = load_frame(&headers, body.as_bytes(), (1, 1))
            .await
            .expect("the data URI decodes");
        assert_eq!(rgb, vec![1, 2, 3]);
    }
}
