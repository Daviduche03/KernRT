//! KernRT: a small, edge-first inference runtime.
//!
//! Layering, bottom to top: [`tensor`] holds the dense `f32` buffer type,
//! [`ops`] the numerical kernels (a scalar reference plus blocked and NEON
//! variants), [`weights`], [`safetensors`] and [`llama2c`] read parameter
//! files, [`tokenizer`] turns text into ids, and [`model`] / [`llama`] /
//! [`generate`] compose the ops into runnable networks.

pub mod generate;
pub mod gpt2_tokenizer;
pub mod llama;
pub mod llama2c;
pub mod model;
pub mod ops;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;
pub mod weights;
