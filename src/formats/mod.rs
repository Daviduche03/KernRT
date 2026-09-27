//! Parameter-file readers.
//!
//! Each turns one on-disk layout into [`Tensor`](crate::tensor::Tensor)s and
//! knows nothing about what those tensors mean.

pub mod llama2c;
pub mod safetensors;
pub mod weights;
