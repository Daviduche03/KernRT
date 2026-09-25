//! Reader for the `safetensors` format.
//!
//! Layout:
//!
//! ```text
//! header_len   u64, little-endian
//! header       header_len bytes of UTF-8 JSON
//! data         the rest of the file
//! ```
//!
//! The JSON is an object mapping tensor name to a spec
//! `{"dtype": "F32", "shape": [..], "data_offsets": [start, end]}`, plus an
//! optional `"__metadata__"` entry that is not a tensor. Offsets are byte
//! ranges into `data`, so a tensor's bytes are `data[start..end]`, little-endian
//! and row-major.
//!
//! Only `F32`, `BF16` and `F16` are read; each is widened to `f32`, which is
//! what the kernels compute in. As in [`crate::weights`], the file is treated
//! as untrusted: every offset and shape is checked against the file size before
//! it is used to slice or allocate.

use crate::tensor::{Tensor, TensorError};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::path::Path;

/// Everything that can go wrong reading a safetensors file.
#[derive(Debug)]
pub enum SafeTensorsError {
    /// The filesystem read failed.
    Io(std::io::Error),
    /// The file is too small to contain the header it claims.
    HeaderTooShort { file_len: usize, needed: usize },
    /// The header was not valid JSON.
    InvalidJson(serde_json::Error),
    /// The header JSON parsed but did not have the expected structure.
    InvalidHeader(String),
    /// A tensor used a dtype this runtime does not read.
    UnsupportedDtype(String),
    /// A tensor's `[start, end)` range was inconsistent with its shape.
    InvalidOffsets { name: String, start: u64, end: u64 },
    /// A tensor's own shape/length validation failed.
    Tensor(TensorError),
}

impl fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeTensorsError::Io(e) => write!(f, "i/o error: {e}"),
            SafeTensorsError::HeaderTooShort { file_len, needed } => write!(
                f,
                "file of {file_len} bytes is too short for a {needed}-byte header"
            ),
            SafeTensorsError::InvalidJson(e) => write!(f, "invalid header JSON: {e}"),
            SafeTensorsError::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            SafeTensorsError::UnsupportedDtype(dtype) => {
                write!(
                    f,
                    "unsupported dtype {dtype:?} (F32, BF16 and F16 are read)"
                )
            }
            SafeTensorsError::InvalidOffsets { name, start, end } => write!(
                f,
                "tensor {name:?} has byte range {start}..{end} that does not match its shape"
            ),
            SafeTensorsError::Tensor(e) => write!(f, "invalid tensor: {e}"),
        }
    }
}

impl std::error::Error for SafeTensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SafeTensorsError::Io(e) => Some(e),
            SafeTensorsError::InvalidJson(e) => Some(e),
            SafeTensorsError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SafeTensorsError {
    fn from(e: std::io::Error) -> Self {
        SafeTensorsError::Io(e)
    }
}

impl From<serde_json::Error> for SafeTensorsError {
    fn from(e: serde_json::Error) -> Self {
        SafeTensorsError::InvalidJson(e)
    }
}

impl From<TensorError> for SafeTensorsError {
    fn from(e: TensorError) -> Self {
        SafeTensorsError::Tensor(e)
    }
}

/// Read every tensor from a safetensors file by mapping it.
///
/// The file is mapped read-only rather than read into a `Vec<u8>`. For a model
/// file this is 257 MB of allocation and copying that the mapping avoids: the
/// pages are faulted in on demand as the parser walks the header and the tensor
/// payloads, and the ones already resident are shared with the page cache rather
/// than duplicated. Tensors are still copied out into owned [`Tensor`]s, so the
/// mapping is dropped at the end of the call.
///
/// Falls back to reading the file when mapping fails, which happens on some
/// filesystems and on empty files.
pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, SafeTensorsError> {
    let path = path.as_ref();
    match unsafe { Mmap::map(&File::open(path)?) } {
        Ok(map) => read_from(&map),
        Err(_) => {
            let bytes = fs::read(path)?;
            read_from(&bytes)
        }
    }
}

/// Map a safetensors file and parse it without widening.
pub fn load_raw(path: impl AsRef<Path>) -> Result<HashMap<String, RawTensor>, SafeTensorsError> {
    let path = path.as_ref();
    match unsafe { Mmap::map(&File::open(path)?) } {
        Ok(map) => read_raw_from(&map),
        Err(_) => read_raw_from(&fs::read(path)?),
    }
}

/// Read every `F32` tensor from a safetensors file.
///
/// The whole file is read into memory; tensors are copied out of it as owned
/// [`Tensor`]s.
pub fn load_into_memory(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, Tensor>, SafeTensorsError> {
    let bytes = fs::read(path)?;
    read_from(&bytes)
}

/// Parse from a complete byte image. Split out from [`load`] so tests can build
/// malformed inputs in memory.
fn read_from(bytes: &[u8]) -> Result<HashMap<String, Tensor>, SafeTensorsError> {
    Ok(read_with(bytes, |dtype, raw, shape| {
        Ok(RawTensor::F32(Tensor::from_vec(dtype.decode(raw), &shape)?))
    })?
    .into_iter()
    .map(|(name, raw)| {
        // The widening reader never produces the compact variants.
        let tensor = match raw {
            RawTensor::F32(t) => t,
            RawTensor::Bf16 { bits, shape } => Tensor::from_vec(
                bits.iter()
                    .map(|&b| f32::from_bits(u32::from(b) << 16))
                    .collect(),
                &shape,
            )
            .expect("shape came from the file and was validated"),
            RawTensor::F16(t) => t,
        };
        (name, tensor)
    })
    .collect())
}

/// Parse raw tensors from a complete byte image, preserving bf16 bit patterns.
pub fn read_raw_from(bytes: &[u8]) -> Result<HashMap<String, RawTensor>, SafeTensorsError> {
    read_with(bytes, |dtype, raw, shape| dtype.decode_raw(raw, shape))
}

/// Shared parser. `decode` turns one tensor's byte range into a value; every
/// offset, dtype and shape is validated here before `decode` is called, so both
/// the widening and the raw reader are held to the same untrusted-input checks.
fn read_with(
    bytes: &[u8],
    decode: impl Fn(DType, &[u8], Vec<usize>) -> Result<RawTensor, SafeTensorsError>,
) -> Result<HashMap<String, RawTensor>, SafeTensorsError> {
    if bytes.len() < 8 {
        return Err(SafeTensorsError::HeaderTooShort {
            file_len: bytes.len(),
            needed: 8,
        });
    }
    let mut length_bytes = [0u8; 8];
    length_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(length_bytes) as usize;

    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .ok_or(SafeTensorsError::HeaderTooShort {
            file_len: bytes.len(),
            needed: 8 + header_len,
        })?;

    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])?;
    let data = &bytes[header_end..];
    let entries = header
        .as_object()
        .ok_or_else(|| SafeTensorsError::InvalidHeader("top level is not an object".into()))?;

    let mut tensors = HashMap::new();
    for (name, spec) in entries {
        // `__metadata__` is reserved by the format for string metadata.
        if name == "__metadata__" {
            continue;
        }
        let spec = spec.as_object().ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("tensor {name:?} is not an object"))
        })?;

        let dtype = spec.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("tensor {name:?} has no dtype"))
        })?;
        let dtype = DType::parse(dtype)?;

        let shape: Vec<usize> = spec
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("tensor {name:?} has no shape array"))
            })?
            .iter()
            .map(|dim| dim.as_u64().map(|d| d as usize))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("tensor {name:?} has a non-integer shape"))
            })?;

        let offsets = spec
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("tensor {name:?} has no data_offsets"))
            })?;
        let start = offsets.first().and_then(|v| v.as_u64());
        let end = offsets.get(1).and_then(|v| v.as_u64());
        let (start, end) = match (start, end) {
            (Some(start), Some(end)) => (start, end),
            _ => {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "tensor {name:?} needs two integer data_offsets"
                )));
            }
        };

        // The byte range must be ordered, inside the data buffer, and exactly
        // the size the shape implies. `numel` is checked so a huge shape cannot
        // overflow before the length comparison.
        let numel = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64));
        let expected = numel.and_then(|n| n.checked_mul(dtype.byte_width()));
        if start > end || end > data.len() as u64 || Some(end - start) != expected {
            return Err(SafeTensorsError::InvalidOffsets {
                name: name.clone(),
                start,
                end,
            });
        }

        let raw = &data[start as usize..end as usize];
        tensors.insert(name.clone(), decode(dtype, raw, shape)?);
    }
    Ok(tensors)
}

/// The element formats this reader understands. Everything widens to `f32`,
/// which is what the kernels compute in.
#[derive(Clone, Copy)]
enum DType {
    F32,
    Bf16,
    F16,
}

impl DType {
    fn parse(name: &str) -> Result<Self, SafeTensorsError> {
        match name {
            "F32" => Ok(DType::F32),
            "BF16" => Ok(DType::Bf16),
            "F16" => Ok(DType::F16),
            other => Err(SafeTensorsError::UnsupportedDtype(other.to_string())),
        }
    }

    fn byte_width(self) -> u64 {
        match self {
            DType::F32 => 4,
            DType::Bf16 | DType::F16 => 2,
        }
    }

    /// Widen a little-endian payload to `f32`.
    fn decode(self, raw: &[u8]) -> Vec<f32> {
        match self {
            DType::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::Bf16 => raw
                .chunks_exact(2)
                .map(|c| {
                    // bfloat16 is the top 16 bits of an f32.
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                })
                .collect(),
            DType::F16 => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32_local(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        }
    }
}

/// A tensor exactly as the file stores it, before any widening.
///
/// The widening loader turns every element into `f32`, which for a bf16 file
/// means allocating twice the payload, converting element by element, and then
/// having the model narrow it straight back to bf16. Carrying the raw bits lets
/// a bf16 model skip all of that: the payload is a `memcpy` into the shape the
/// kernels already want.
pub enum RawTensor {
    /// Little-endian `f32`, already in the form the kernels compute in.
    F32(Tensor),
    /// bf16 bit patterns as little-endian `u16`, `[rows, cols]` order.
    Bf16 { bits: Vec<u16>, shape: Vec<usize> },
    /// f16 bit patterns; nothing consumes these raw yet, so they widen.
    F16(Tensor),
}

impl RawTensor {
    /// Shape as stored, regardless of how the payload is held.
    pub fn shape(&self) -> &[usize] {
        match self {
            RawTensor::F32(t) | RawTensor::F16(t) => t.shape(),
            RawTensor::Bf16 { shape, .. } => shape,
        }
    }
}

impl DType {
    /// Decode to the compact form where the format has one, else widen.
    fn decode_raw(self, raw: &[u8], shape: Vec<usize>) -> Result<RawTensor, SafeTensorsError> {
        Ok(match self {
            DType::F32 => RawTensor::F32(Tensor::from_vec(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                &shape,
            )?),
            DType::Bf16 => RawTensor::Bf16 {
                bits: raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                shape,
            },
            DType::F16 => RawTensor::F16(Tensor::from_vec(self.decode(raw), &shape)?),
        })
    }
}

/// IEEE binary16 to binary32.
fn f16_to_f32_local(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let widened = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: shift the mantissa left until the implicit bit appears.
            let mut mantissa = mantissa;
            let mut exponent = 127 - 15 + 1;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13),
    };
    f32::from_bits(widened)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors image; `shape` is described by the data length.
    fn encode(tensors: &[(&str, &str, &[usize], &[f32])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for &(name, dtype, shape, values) in tensors {
            let start = data.len();
            for &v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn roundtrip_preserves_names_shapes_and_values() {
        let bytes = encode(&[
            ("fc1.weight", "F32", &[2, 3], &[1., 2., 3., 4., 5., 6.]),
            ("fc1.bias", "F32", &[2], &[-1.5, 2.5]),
        ]);
        let tensors = read_from(&bytes).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["fc1.weight"].shape(), &[2, 3]);
        assert_eq!(tensors["fc1.weight"].data(), &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(tensors["fc1.bias"].data(), &[-1.5, 2.5]);
    }

    #[test]
    fn skips_the_metadata_entry() {
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "__metadata__": {"format": "pt"},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&7.0f32.to_le_bytes());

        let tensors = read_from(&bytes).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["w"].data(), &[7.0]);
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let bytes = encode(&[("w", "I32", &[1], &[0.0])]);
        assert!(matches!(
            read_from(&bytes),
            Err(SafeTensorsError::UnsupportedDtype(d)) if d == "I32"
        ));
    }

    /// Build a one-tensor file from raw (already encoded) bytes.
    fn encode_raw(dtype: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
        let header = serde_json::json!({
            "w": {"dtype": dtype, "shape": shape, "data_offsets": [0, payload.len()]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn decodes_bfloat16_weights() {
        // 1.0, -2.0, 0.5 as little-endian bfloat16.
        let payload = [0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f];
        let tensors = read_from(&encode_raw("BF16", &[3], &payload)).unwrap();
        assert_eq!(tensors["w"].data(), &[1.0, -2.0, 0.5]);
    }

    #[test]
    fn decodes_float16_weights() {
        // 1.0, 0.5, -2.0, 0.0 as little-endian binary16.
        let payload = [0x00, 0x3c, 0x00, 0x38, 0x00, 0xc0, 0x00, 0x00];
        let tensors = read_from(&encode_raw("F16", &[4], &payload)).unwrap();
        assert_eq!(tensors["w"].data(), &[1.0, 0.5, -2.0, 0.0]);
    }

    #[test]
    fn rejects_header_longer_than_file() {
        let mut bytes = 999u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(matches!(
            read_from(&bytes),
            Err(SafeTensorsError::HeaderTooShort { .. })
        ));
    }

    #[test]
    fn rejects_offsets_that_do_not_match_shape() {
        // Two elements of shape, but only one element of payload.
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(matches!(
            read_from(&bytes),
            Err(SafeTensorsError::InvalidOffsets { .. })
        ));
    }

    #[test]
    fn rejects_offsets_past_end_of_file() {
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes()); // only 4 bytes of data
        assert!(matches!(
            read_from(&bytes),
            Err(SafeTensorsError::InvalidOffsets { .. })
        ));
    }

    #[test]
    fn rejects_invalid_json() {
        let header_bytes = b"not json";
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header_bytes);
        assert!(matches!(
            read_from(&bytes),
            Err(SafeTensorsError::InvalidJson(_))
        ));
    }
}
