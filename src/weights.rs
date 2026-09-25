//! On-disk parameter format and its reader/writer.
//!
//! Layout (all integers little-endian, so a file written on one machine reads
//! back on another regardless of host endianness):
//!
//! ```text
//! magic          [u8; 4] = "KRTW"
//! version        u32
//! tensor_count   u32
//! per tensor:
//!   name_len     u32
//!   name         name_len bytes of UTF-8
//!   rank         u32
//!   dims         rank * u32
//!   payload      product(dims) * f32, little-endian
//! ```
//!
//! The reader treats the file as untrusted. Every length in the header is
//! checked against the total file size *before* a buffer of that size is
//! allocated, so a corrupted or hostile header cannot turn into a multi-gigabyte
//! allocation. Duplicate names are rejected rather than silently resolved.

use crate::tensor::{Tensor, TensorError};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: [u8; 4] = *b"KRTW";
const VERSION: u32 = 1;

/// Everything that can go wrong reading or writing a weight file.
#[derive(Debug)]
pub enum WeightError {
    /// The underlying filesystem or stream operation failed.
    Io(io::Error),
    /// The file does not start with the expected magic bytes.
    BadMagic { found: [u8; 4] },
    /// The file was written by an incompatible version of the format.
    UnsupportedVersion { found: u32, supported: u32 },
    /// A declared length could not fit inside the file.
    ImplausibleLength {
        what: &'static str,
        declared: u64,
        available: u64,
    },
    /// A tensor name was not valid UTF-8.
    InvalidName(std::string::FromUtf8Error),
    /// The same tensor name appeared twice.
    DuplicateName(String),
    /// A value is too large to be represented in the on-disk format.
    TooLarge { what: &'static str },
    /// A tensor's own shape/length validation failed.
    Tensor(TensorError),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::Io(e) => write!(f, "i/o error: {e}"),
            WeightError::BadMagic { found } => {
                write!(f, "not a KernRT weight file (magic was {found:?})")
            }
            WeightError::UnsupportedVersion { found, supported } => write!(
                f,
                "weight format version {found} is not supported (this build reads version {supported})"
            ),
            WeightError::ImplausibleLength {
                what,
                declared,
                available,
            } => write!(
                f,
                "{what} of {declared} cannot fit in a file of {available} bytes"
            ),
            WeightError::InvalidName(e) => write!(f, "tensor name is not valid UTF-8: {e}"),
            WeightError::DuplicateName(name) => write!(f, "duplicate tensor name {name:?}"),
            WeightError::TooLarge { what } => {
                write!(f, "{what} is too large for the on-disk format")
            }
            WeightError::Tensor(e) => write!(f, "invalid tensor: {e}"),
        }
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WeightError::Io(e) => Some(e),
            WeightError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WeightError {
    fn from(e: io::Error) -> Self {
        WeightError::Io(e)
    }
}

impl From<TensorError> for WeightError {
    fn from(e: TensorError) -> Self {
        WeightError::Tensor(e)
    }
}

/// Write named tensors to `path`.
pub fn save(path: impl AsRef<Path>, tensors: &[(&str, &Tensor)]) -> Result<(), WeightError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_to(&mut writer, tensors)?;
    writer.flush()?;
    Ok(())
}

/// Read all named tensors from `path`.
pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, WeightError> {
    let file = File::open(path)?;
    let available = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    read_from(&mut reader, available)
}

/// Serialize to any writer. Split out from [`save`] so tests can round-trip
/// through an in-memory buffer instead of touching the filesystem.
fn write_to<W: Write>(w: &mut W, tensors: &[(&str, &Tensor)]) -> Result<(), WeightError> {
    w.write_all(&MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&to_u32(tensors.len() as u64, "tensor count")?.to_le_bytes())?;

    for &(name, tensor) in tensors {
        let name = name.as_bytes();
        let name_len = to_u32(name.len() as u64, "tensor name length")?;
        w.write_all(&name_len.to_le_bytes())?;
        w.write_all(name)?;

        let rank = to_u32(tensor.rank() as u64, "tensor rank")?;
        w.write_all(&rank.to_le_bytes())?;
        for &dim in tensor.shape() {
            w.write_all(&to_u32(dim as u64, "tensor dimension")?.to_le_bytes())?;
        }

        let mut payload = Vec::with_capacity(tensor.numel() * 4);
        for &value in tensor.data() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        w.write_all(&payload)?;
    }
    Ok(())
}

/// Parse tensors from a reader that reports `available` total bytes.
fn read_from<R: Read>(r: &mut R, available: u64) -> Result<HashMap<String, Tensor>, WeightError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(WeightError::BadMagic { found: magic });
    }

    let version = read_u32(r)?;
    if version != VERSION {
        return Err(WeightError::UnsupportedVersion {
            found: version,
            supported: VERSION,
        });
    }

    let count = read_u32(r)?;
    let mut tensors = HashMap::new();
    for _ in 0..count {
        let name_len = read_u32(r)?.into();
        fits_in_file("tensor name length", name_len, available)?;
        let mut name_bytes = vec![0u8; name_len as usize];
        r.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).map_err(WeightError::InvalidName)?;

        // Each dimension occupies 4 header bytes, so the rank is bounded by the
        // file size too. Checking it keeps `with_capacity` from trusting input.
        let rank = read_u32(r)?;
        fits_in_file("tensor rank", u64::from(rank) * 4, available)?;
        let mut dims = Vec::with_capacity(rank as usize);
        let mut numel: u64 = 1;
        for _ in 0..rank {
            let dim = read_u32(r)?;
            dims.push(dim as usize);
            numel = numel
                .checked_mul(u64::from(dim))
                .ok_or(WeightError::ImplausibleLength {
                    what: "element count",
                    declared: u64::MAX,
                    available,
                })?;
        }

        let payload_len = numel.checked_mul(4).unwrap_or(u64::MAX);
        fits_in_file("tensor payload", payload_len, available)?;
        let mut payload = vec![0u8; payload_len as usize];
        r.read_exact(&mut payload)?;
        let data: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tensor = Tensor::from_vec(data, &dims)?;

        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(WeightError::DuplicateName(name));
        }
    }
    Ok(tensors)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, WeightError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn to_u32(value: u64, what: &'static str) -> Result<u32, WeightError> {
    u32::try_from(value).map_err(|_| WeightError::TooLarge { what })
}

fn fits_in_file(what: &'static str, declared: u64, available: u64) -> Result<(), WeightError> {
    if declared > available {
        return Err(WeightError::ImplausibleLength {
            what,
            declared,
            available,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape).unwrap()
    }

    fn encode(tensors: &[(&str, &Tensor)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_to(&mut bytes, tensors).unwrap();
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<HashMap<String, Tensor>, WeightError> {
        read_from(&mut Cursor::new(bytes), bytes.len() as u64)
    }

    #[test]
    fn roundtrip_preserves_names_shapes_and_values() {
        let a = t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(vec![-1.5, 2.5], &[2]);
        let decoded = decode(&encode(&[("w", &a), ("b", &b)])).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded["w"].shape(), &[2, 3]);
        assert_eq!(decoded["w"].data(), a.data());
        assert_eq!(decoded["b"].shape(), &[2]);
        assert_eq!(decoded["b"].data(), b.data());
    }

    #[test]
    fn roundtrip_handles_a_scalar_tensor() {
        let s = t(vec![42.0], &[]);
        let decoded = decode(&encode(&[("s", &s)])).unwrap();
        assert_eq!(decoded["s"].shape(), &[] as &[usize]);
        assert_eq!(decoded["s"].data(), &[42.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode(&[]);
        bytes[0] = b'X';
        assert!(matches!(decode(&bytes), Err(WeightError::BadMagic { .. })));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = encode(&[]);
        bytes[4] = 99;
        assert!(matches!(
            decode(&bytes),
            Err(WeightError::UnsupportedVersion { found: 99, .. })
        ));
    }

    #[test]
    fn rejects_duplicate_names() {
        let a = t(vec![1.], &[1]);
        let err = decode(&encode(&[("a", &a), ("a", &a)])).unwrap_err();
        assert!(matches!(err, WeightError::DuplicateName(name) if name == "a"));
    }

    #[test]
    fn rejects_truncated_file() {
        let a = t(vec![1., 2., 3., 4.], &[4]);
        let mut bytes = encode(&[("a", &a)]);
        bytes.truncate(bytes.len() - 4);
        // The declared payload still fits the (truncated) file size, so the
        // length guard passes and the short read is what catches it.
        assert!(matches!(decode(&bytes), Err(WeightError::Io(_))));
    }

    #[test]
    fn rejects_a_hostile_element_count() {
        // A 22-byte file whose header claims two u32::MAX dimensions.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&1u32.to_le_bytes()); // name length
        bytes.push(b'a');
        bytes.extend_from_slice(&2u32.to_le_bytes()); // rank
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());

        assert!(matches!(
            decode(&bytes),
            Err(WeightError::ImplausibleLength { .. })
        ));
    }

    #[test]
    fn rejects_non_utf8_names() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&1u32.to_le_bytes()); // name length
        bytes.push(0xFF); // not valid UTF-8
        bytes.extend_from_slice(&0u32.to_le_bytes()); // rank

        assert!(matches!(decode(&bytes), Err(WeightError::InvalidName(_))));
    }
}
