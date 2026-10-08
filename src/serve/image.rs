//! Turning a request body into RGB the detector can read.
//!
//! [`super::model::Detect::detect`] takes interleaved RGB at the model's own
//! input size, so decoding, resampling and `data:` URIs all live here rather
//! than being pushed onto callers. The resample uses the same nearest-
//! neighbour mapping as `Yolo26::preprocess_rgb`, so a decoded JPEG and a
//! hand-posted RGB frame of the same image land on identical pixels.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use super::error::ServeError;

/// Ceiling on a body accepted as an image, encoded or fetched. A frame at
/// 8K RGB is ~100 MB; nothing a detector wants is near this.
pub const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// Ceiling on what the decoder may allocate. An 8000x6000 photo is ~144 MB
/// as RGB, so this has real headroom while still refusing a decompression
/// bomb outright rather than trying to find out how bad it is.
const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

/// A decoded image as interleaved 8-bit RGB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Decode JPEG/PNG/WebP/GIF/BMP/... bytes.
///
/// The format is guessed from the magic bytes, so a wrong `Content-Type` is
/// not fatal. Alpha is dropped rather than composited, matching what
/// `PIL.Image.convert("RGB")` does — the tooling these models were trained
/// and evaluated against.
pub fn decode(bytes: &[u8]) -> Result<Frame, ServeError> {
    if bytes.is_empty() {
        return Err(ServeError::bad_request("the image body is empty"));
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ServeError::bad_request(format!(
            "image is {} bytes; the limit is {MAX_IMAGE_BYTES}",
            bytes.len()
        )));
    }
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| ServeError::bad_request(format!("cannot read image: {error}")))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| ServeError::bad_request(format!("cannot decode image: {error}")))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    Ok(Frame {
        width: width as usize,
        height: height as usize,
        rgb: image.into_raw(),
    })
}

/// Nearest-neighbour stretch to `(width, height)`.
///
/// The mapping is deliberately identical to `Yolo26::preprocess_rgb`, which
/// is what the raw-RGB path runs afterwards. Two resamples would otherwise
/// drift by a pixel and make the same image disagree with itself.
pub fn resize(frame: Frame, width: usize, height: usize) -> Frame {
    if (frame.width, frame.height) == (width, height) {
        return frame;
    }
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        let sy = y * frame.height / height;
        for x in 0..width {
            let sx = x * frame.width / width;
            let src = (sy * frame.width + sx) * 3;
            let dst = (y * width + x) * 3;
            rgb[dst..dst + 3].copy_from_slice(&frame.rgb[src..src + 3]);
        }
    }
    Frame { width, height, rgb }
}

/// Decode a `data:[<mime>][;base64],<payload>` URI.
///
/// This is the second form OpenAI's `image_url` accepts, so a client written
/// against their vision API works unchanged.
pub fn from_data_uri(uri: &str) -> Result<Frame, ServeError> {
    let rest = uri.strip_prefix("data:").ok_or_else(|| {
        ServeError::bad_request("image_url must be an http(s) URL or a data: URI")
    })?;
    let (metadata, payload) = rest
        .split_once(',')
        .ok_or_else(|| ServeError::bad_request("a data: URI needs a comma before its payload"))?;
    if !metadata.ends_with(";base64") {
        return Err(ServeError::bad_request(
            "only base64 data: URIs are supported; got \
             `data:[<mime>],<payload>` without ;base64",
        ));
    }
    let bytes = STANDARD.decode(payload).map_err(|error| {
        ServeError::bad_request(format!("invalid base64 in the data: URI: {error}"))
    })?;
    decode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3x2 gradient, encoded as PNG by the crate under test's own decoder.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut bytes);
        use image::ImageEncoder;
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| [(i % 256) as u8, (i / 7 % 256) as u8, (i / 13 % 256) as u8])
            .collect();
        encoder
            .write_image(&pixels, width, height, image::ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    #[test]
    fn a_png_round_trips_to_rgb() {
        let frame = decode(&png(3, 2)).unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(frame.rgb.len(), 3 * 2 * 3);
        assert_eq!(frame.rgb[0..3], [0, 0, 0], "first pixel");
    }

    /// The format is guessed from the bytes, not from a header the client
    /// may have got wrong.
    #[test]
    fn decoding_does_not_need_the_right_content_type() {
        assert!(decode(&png(1, 1)).is_ok());
    }

    #[test]
    fn garbage_is_a_bad_request_not_a_panic() {
        let error = decode(b"not an image at all").unwrap_err();
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message().contains("cannot decode image"), "{error}");
    }

    #[test]
    fn an_empty_body_is_rejected_early() {
        let error = decode(b"").unwrap_err();
        assert_eq!(error.message(), "the image body is empty");
    }

    #[test]
    fn an_oversized_body_is_rejected_before_decoding() {
        let mut bytes = vec![0u8; MAX_IMAGE_BYTES + 1];
        bytes[0] = 0x89; // PNG magic, so it would otherwise try
        let error = decode(&bytes).unwrap_err();
        assert!(error.message().contains("the limit is"), "{error}");
    }

    /// Resize must land on exactly the pixels `preprocess_rgb` would pick,
    /// or the same image gives different boxes depending on how it arrived.
    #[test]
    fn resize_uses_the_preprocess_mapping() {
        let frame = Frame {
            width: 4,
            height: 4,
            rgb: (0..48).collect(),
        };
        let out = resize(frame, 2, 2);
        assert_eq!((out.width, out.height), (2, 2));
        // y=0 -> sy=0*4/2=0; y=1 -> sy=1*4/2=2. Same for x.
        assert_eq!(&out.rgb[0..3], &frame_or(0, 0));
        assert_eq!(&out.rgb[3..6], &frame_or(2, 0));
        assert_eq!(&out.rgb[6..9], &frame_or(0, 2));
        assert_eq!(&out.rgb[9..12], &frame_or(2, 2));

        fn frame_or(x: usize, y: usize) -> [u8; 3] {
            let i = (y * 4 + x) * 3;
            [i as u8, (i + 1) as u8, (i + 2) as u8]
        }
    }

    #[test]
    fn resizing_to_the_same_size_changes_nothing() {
        let frame = Frame {
            width: 2,
            height: 2,
            rgb: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };
        let expected = frame.clone();
        assert_eq!(resize(frame, 2, 2), expected);
    }

    #[test]
    fn a_data_uri_decodes_through_the_same_path() {
        let payload = STANDARD.encode(png(1, 1));
        let frame = from_data_uri(&format!("data:image/png;base64,{payload}")).unwrap();
        assert_eq!((frame.width, frame.height), (1, 1));
    }

    #[test]
    fn a_non_base64_data_uri_says_so() {
        let error = from_data_uri("data:text/plain,hello").unwrap_err();
        assert!(error.message().contains("only base64"), "{error}");
    }

    #[test]
    fn something_that_is_neither_url_nor_data_uri_is_rejected() {
        for uri in [
            "/etc/passwd",
            "ftp://x/y",
            "cat.jpg",
            "data:image/png;base64",
        ] {
            assert!(from_data_uri(uri).is_err(), "{uri} should not be accepted");
        }
    }
}
