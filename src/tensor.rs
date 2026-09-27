use std::fmt;

/// Errors produced by tensor construction and element access.
///
/// This is a plain `enum` with struct-like variants: each variant carries
/// exactly the data needed to explain the failure. No dependency on `anyhow`
/// or `thiserror`; `Display` + `std::error::Error` is the standard library's
/// way to make a type usable with `?` and error-reporting tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// `data.len()` did not equal the product of `shape`.
    DataLenMismatch {
        shape: Vec<usize>,
        expected: usize,
        got: usize,
    },
    /// An operation expected a specific number of dimensions.
    RankMismatch { expected: usize, got: usize },
    /// Two tensors had incompatible shapes for a binary operation.
    ShapeMismatch {
        op: &'static str,
        a: Vec<usize>,
        b: Vec<usize>,
    },
    /// An index was outside the tensor's bounds.
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::DataLenMismatch {
                shape,
                expected,
                got,
            } => write!(
                f,
                "data length {got} does not match shape {shape:?} (expected {expected} elements)"
            ),
            TensorError::RankMismatch { expected, got } => {
                write!(f, "expected rank {expected}, got rank {got}")
            }
            TensorError::ShapeMismatch { op, a, b } => {
                write!(f, "{op} shape mismatch: {a:?} vs {b:?}")
            }
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} out of bounds for shape {shape:?}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// A dense, contiguous, row-major tensor of `f32`.
///
/// Layout: element at index `[i0, i1, ..., ik]` is stored at flat offset
/// `i0 * d1*d2*...*dk + i1 * d2*...*dk + ... + ik`. The last dimension is the
/// fastest-varying ("row-major" / C order).
///
/// The fields are private so the invariant `data.len() == product(shape)` can
/// never be violated from outside this module.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Build a tensor from raw data, validating that the element count matches
    /// the shape. Takes ownership of `data` (no copy): the `Vec`'s heap buffer
    /// moves into the tensor. `shape` is borrowed as a slice so callers can pass
    /// either `&[2, 3]` or a `&Vec<usize>`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::DataLenMismatch {
                shape: shape.to_vec(),
                expected,
                got: data.len(),
            });
        }
        Ok(Tensor {
            data,
            shape: shape.to_vec(),
        })
    }

    /// Allocate a zero-filled tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    /// The tensor's dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements (equals `data().len()`).
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Read-only view of the underlying elements.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of the underlying elements.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Drop the shape and hand over the elements.
    ///
    /// For conversions that produce a buffer with a shape already known to the
    /// caller — extracting an f32 weight copy, for instance — this avoids
    /// allocating a second time to move the data out.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Translate a multi-dimensional index into a flat row-major offset.
    fn flat_index(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                got: index.len(),
            });
        }
        let mut offset = 0usize;
        for (axis, &idx) in index.iter().enumerate() {
            let dim = self.shape[axis];
            if idx >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            offset = offset * dim + idx;
        }
        Ok(offset)
    }

    /// Read a single element. Bounds-checked.
    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let offset = self.flat_index(index)?;
        Ok(self.data[offset])
    }

    /// Write a single element. Bounds-checked.
    pub fn set(&mut self, index: &[usize], value: f32) -> Result<(), TensorError> {
        let offset = self.flat_index(index)?;
        self.data[offset] = value;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_accepts_matching_len() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn from_vec_rejects_mismatched_len() {
        let err = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err();
        assert!(matches!(
            err,
            TensorError::DataLenMismatch {
                expected: 4,
                got: 3,
                ..
            }
        ));
    }

    #[test]
    fn zeros_has_expected_layout() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn get_set_roundtrip_and_bounds() {
        let mut t = Tensor::zeros(&[2, 3]);
        t.set(&[1, 2], 7.0).unwrap();
        assert_eq!(t.get(&[1, 2]).unwrap(), 7.0);
        assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);
        // Out-of-range element and wrong rank both fail.
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0]).is_err());
    }
}
