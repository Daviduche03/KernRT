//! A concrete, runnable model built from the ops.
//!
//! Deliberately not generic: there is one architecture (a two-layer perceptron
//! with ReLU hidden activation) and no trait objects or dynamic dispatch. When a
//! second architecture arrives, the duplicated parts will be obvious enough to
//! factor out then.

use crate::ops::{add_bias, matmul_tiled, matvec, relu, transpose};
use crate::tensor::{Tensor, TensorError};
use std::collections::HashMap;
use std::fmt;

/// Errors from constructing or running a model.
#[derive(Debug)]
pub enum ModelError {
    /// An underlying tensor op rejected the shapes it was given.
    Tensor(TensorError),
    /// A parameter the model needs is missing from the weight map.
    MissingTensor(String),
    /// Two layers' dimensions do not line up.
    LayerMismatch {
        from: &'static str,
        to: &'static str,
        out_features: usize,
        in_features: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Tensor(e) => write!(f, "{e}"),
            ModelError::MissingTensor(name) => {
                write!(f, "weight map is missing tensor {name:?}")
            }
            ModelError::LayerMismatch {
                from,
                to,
                out_features,
                in_features,
            } => write!(
                f,
                "{from} outputs {out_features} features but {to} expects {in_features}"
            ),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TensorError> for ModelError {
    fn from(e: TensorError) -> Self {
        ModelError::Tensor(e)
    }
}

/// A dense layer: `y = x @ weight + bias`.
///
/// `weight` is stored `[in_features, out_features]` — the layout `matmul`
/// consumes directly — and `bias` is `[out_features]`.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    /// Validate and adopt a weight/bias pair.
    pub fn new(weight: Tensor, bias: Tensor) -> Result<Self, ModelError> {
        if weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                got: weight.rank(),
            }
            .into());
        }
        if bias.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                got: bias.rank(),
            }
            .into());
        }
        let out_features = weight.shape()[1];
        if bias.shape()[0] != out_features {
            return Err(TensorError::ShapeMismatch {
                op: "linear",
                a: weight.shape().to_vec(),
                b: bias.shape().to_vec(),
            }
            .into());
        }
        Ok(Linear { weight, bias })
    }

    /// Input feature count (`weight` rows).
    pub fn in_features(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Output feature count (`weight` columns).
    pub fn out_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Apply to a batch: `[batch, in_features] -> [batch, out_features]`.
    ///
    /// A batch of one takes the [`matvec`] path: the tiled matmul would send
    /// every tile to the scalar fallback for a single row, which measured
    /// several times slower than streaming the weights once into a
    /// register-resident accumulator.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        let linear = if x.rank() == 2 && x.shape()[0] == 1 {
            matvec(&self.weight, x.data())?
        } else {
            matmul_tiled(x, &self.weight)?
        };
        Ok(add_bias(&linear, &self.bias)?)
    }
}

/// A two-layer perceptron: `relu(x @ W1 + b1) @ W2 + b2`.
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    /// Combine two layers, checking that `fc1` feeds `fc2`.
    pub fn new(fc1: Linear, fc2: Linear) -> Result<Self, ModelError> {
        if fc1.out_features() != fc2.in_features() {
            return Err(ModelError::LayerMismatch {
                from: "fc1",
                to: "fc2",
                out_features: fc1.out_features(),
                in_features: fc2.in_features(),
            });
        }
        Ok(Mlp { fc1, fc2 })
    }

    /// Parameter names this architecture expects in a weight map.
    pub const PARAMETERS: [&str; 4] = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];

    /// Build from a weight map, consuming it so each tensor is moved into the
    /// model rather than copied.
    pub fn from_weights(mut weights: HashMap<String, Tensor>) -> Result<Self, ModelError> {
        let mut take = |name: &str| {
            weights
                .remove(name)
                .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
        };
        let fc1 = Linear::new(take("fc1.weight")?, take("fc1.bias")?)?;
        let fc2 = Linear::new(take("fc2.weight")?, take("fc2.bias")?)?;
        Mlp::new(fc1, fc2)
    }

    /// Build from tensors read by [`crate::safetensors::load`].
    ///
    /// safetensors follows the PyTorch/MLX convention that `Linear.weight` is
    /// `[out_features, in_features]`, whereas this runtime's kernels want
    /// `[in_features, out_features]`. So each weight is transposed once here;
    /// names and biases already match. This is the one place the two layout
    /// conventions meet.
    pub fn from_safetensors(mut weights: HashMap<String, Tensor>) -> Result<Self, ModelError> {
        let mut take = |name: &str| {
            weights
                .remove(name)
                .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
        };
        let fc1 = Linear::new(transpose(&take("fc1.weight")?)?, take("fc1.bias")?)?;
        let fc2 = Linear::new(transpose(&take("fc2.weight")?)?, take("fc2.bias")?)?;
        Mlp::new(fc1, fc2)
    }

    /// Raw scores `[batch, classes]`; apply [`crate::ops::softmax`] for
    /// probabilities.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        let hidden = relu(&self.fc1.forward(x)?);
        self.fc2.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::softmax;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape).unwrap()
    }

    /// in=2, hidden=2, out=1 with hand-checkable numbers.
    fn tiny() -> Mlp {
        let fc1 = Linear::new(t(vec![1., 2., 3., -4.], &[2, 2]), t(vec![0., 0.], &[2])).unwrap();
        let fc2 = Linear::new(t(vec![1., 2.], &[2, 1]), t(vec![0.5], &[1])).unwrap();
        Mlp::new(fc1, fc2).unwrap()
    }

    #[test]
    fn linear_applies_weight_and_bias() {
        let layer =
            Linear::new(t(vec![1., 2., 3., 4.], &[2, 2]), t(vec![0.5, -0.5], &[2])).unwrap();
        let y = layer.forward(&t(vec![1., 2.], &[1, 2])).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
        assert_eq!(y.data(), &[7.5, 9.5]);
    }

    #[test]
    fn mlp_forward_matches_hand_computation() {
        // x = [1, 1]: fc1 -> [4, -2], relu -> [4, 0], fc2 -> 4*1 + 0*2 + 0.5.
        let logits = tiny().forward(&t(vec![1., 1.], &[1, 2])).unwrap();
        assert_eq!(logits.data(), &[4.5]);
    }

    #[test]
    fn mlp_forward_zeroes_preactivations_through_relu() {
        // x = [1, -1]: fc1 -> [1 - 3, 1*2 + (-1)(-4)] = [-2, 6], relu -> [0, 6],
        // fc2 -> 0*1 + 6*2 + 0.5.
        let logits = tiny().forward(&t(vec![1., -1.], &[1, 2])).unwrap();
        assert_eq!(logits.data(), &[12.5]);
    }

    #[test]
    fn softmax_of_logits_is_a_distribution() {
        let fc1 = Linear::new(t(vec![1., 2., 3., -4.], &[2, 2]), t(vec![0., 0.], &[2])).unwrap();
        let fc2 = Linear::new(t(vec![1., -5., 2., 3.], &[2, 2]), t(vec![0.5, -0.5], &[2])).unwrap();
        let model = Mlp::new(fc1, fc2).unwrap();

        let probs = softmax(&model.forward(&t(vec![1., 1.], &[1, 2])).unwrap());
        let sum: f32 = probs.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sums to {sum}");
        assert!(probs.data().iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn from_weights_builds_the_same_model() {
        let mut weights = HashMap::new();
        weights.insert("fc1.weight".to_string(), t(vec![1., 2., 3., -4.], &[2, 2]));
        weights.insert("fc1.bias".to_string(), t(vec![0., 0.], &[2]));
        weights.insert("fc2.weight".to_string(), t(vec![1., 2.], &[2, 1]));
        weights.insert("fc2.bias".to_string(), t(vec![0.5], &[1]));

        let model = Mlp::from_weights(weights).unwrap();
        let x = t(vec![1., 1.], &[1, 2]);
        assert_eq!(
            model.forward(&x).unwrap().data(),
            tiny().forward(&x).unwrap().data()
        );
    }

    #[test]
    fn missing_parameter_is_an_error() {
        let mut weights = HashMap::new();
        weights.insert("fc1.weight".to_string(), t(vec![1., 2., 3., -4.], &[2, 2]));
        assert!(matches!(
            Mlp::from_weights(weights),
            Err(ModelError::MissingTensor(_))
        ));
    }

    #[test]
    fn mismatched_layers_are_rejected() {
        let fc1 = Linear::new(t(vec![1., 2., 3., -4.], &[2, 2]), t(vec![0., 0.], &[2])).unwrap();
        let fc2 = Linear::new(t(vec![1., 2., 3.], &[3, 1]), t(vec![0.], &[1])).unwrap();
        assert!(matches!(
            Mlp::new(fc1, fc2),
            Err(ModelError::LayerMismatch { .. })
        ));
    }

    #[test]
    fn linear_rejects_mismatched_bias() {
        assert!(Linear::new(t(vec![1., 2., 3., 4.], &[2, 2]), t(vec![0.], &[1])).is_err());
    }

    /// safetensors stores weights `[out, in]`; the loader must transpose them
    /// back to the `[in, out]` the kernels want.
    #[test]
    fn from_safetensors_transposes_weights() {
        let mut weights = HashMap::new();
        weights.insert("fc1.weight".to_string(), t(vec![1., 3., 2., -4.], &[2, 2]));
        weights.insert("fc1.bias".to_string(), t(vec![0., 0.], &[2]));
        weights.insert("fc2.weight".to_string(), t(vec![1., 2.], &[1, 2]));
        weights.insert("fc2.bias".to_string(), t(vec![0.5], &[1]));

        let model = Mlp::from_safetensors(weights).unwrap();
        let x = t(vec![1., 1.], &[1, 2]);
        assert_eq!(
            model.forward(&x).unwrap().data(),
            tiny().forward(&x).unwrap().data()
        );
    }
}
