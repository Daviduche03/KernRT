//! Elementwise tensor operations.

use crate::tensor::{Tensor, TensorError};

/// Elementwise rectified linear unit: `max(0, x)`, preserving the shape.
///
/// Unlike the shape-combining ops this cannot fail, so it returns `Tensor`
/// rather than `Result`. It builds from `zeros` and fills in place: the same
/// allocation pattern serves every elementwise op, and the count of elements is
/// unchanged by construction, so no validation is needed.
pub fn relu(x: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(x.shape());
    for (out_v, &x_v) in out.data_mut().iter_mut().zip(x.data()) {
        *out_v = x_v.max(0.0);
    }
    out
}

/// Add a per-column bias to a rank-2 tensor: `x[m, n] + bias[n] -> [m, n]`.
///
/// This is the `+ b` of a dense layer. Only the trailing dimension is
/// broadcast, which is all a bias vector needs; general broadcasting is a much
/// larger feature with no caller here yet.
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
    if x.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            got: x.rank(),
        });
    }
    if bias.rank() != 1 {
        return Err(TensorError::RankMismatch {
            expected: 1,
            got: bias.rank(),
        });
    }
    let n = x.shape()[1];
    if bias.shape()[0] != n {
        return Err(TensorError::ShapeMismatch {
            op: "add_bias",
            a: x.shape().to_vec(),
            b: bias.shape().to_vec(),
        });
    }

    let mut out = Tensor::zeros(x.shape());
    if n == 0 {
        return Ok(out);
    }
    let bias = bias.data();
    for (out_row, x_row) in out.data_mut().chunks_mut(n).zip(x.data().chunks(n)) {
        for ((out_v, &x_v), &b_v) in out_row.iter_mut().zip(x_row).zip(bias) {
            *out_v = x_v + b_v;
        }
    }
    Ok(out)
}

/// Softmax over the last dimension.
///
/// Each row is shifted by its own maximum before exponentiating. That is not
/// cosmetic: `exp(1000)` overflows `f32` to infinity and the ratio becomes
/// `NaN`, while softmax is invariant to a constant shift, so `exp(x - max)`
/// computes the same function without ever overflowing for finite inputs.
/// A zero-length last dimension is returned unchanged (there is nothing to
/// normalize).
pub fn softmax(x: &Tensor) -> Tensor {
    let last = x.shape().last().copied().unwrap_or(0);
    let mut out = x.clone();
    if last == 0 {
        return out;
    }
    for row in out.data_mut().chunks_mut(last) {
        softmax_in_place(row);
    }
    out
}

/// In-place numerically-stable softmax over a slice. Shared by the
/// transformer's attention and the sampler.
pub(crate) fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}
