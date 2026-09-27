//! KernRT: a small, edge-first inference runtime.
//!
//! Layering, bottom to top: [`tensor`] holds the dense `f32` buffer type,
//! [`ops`] the numerical kernels (a scalar reference plus blocked and NEON
//! variants), [`formats`] reads parameter files, [`tokenizer`] turns text into
//! ids, and [`arch`] composes the ops into runnable networks.

pub mod arch;
pub mod formats;
pub mod generate;
pub mod ops;
pub mod tensor;
pub mod tokenizer;

// The file move above is invisible to callers: these aliases keep every
// pre-existing path — `kernrt::llama`, `kernrt::model`, `kernrt::safetensors`,
// `kernrt::weights`, `kernrt::llama2c`, `kernrt::gpt2_tokenizer` — resolving.
pub use arch::llm::llama;
pub use arch::vision::mlp as model;
pub use formats::{llama2c, safetensors, weights};
pub use tokenizer::gpt2 as gpt2_tokenizer;
