//! KernRT: a small, edge-first inference runtime.
//!
//! Layering, bottom to top: [`tensor`] holds the dense `f32` buffer type,
//! [`ops`] the numerical kernels (a scalar reference plus blocked and NEON
//! variants), [`formats`] reads parameter files, [`tokenizer`] turns text into
//! ids, and [`arch`] composes the ops into runnable networks.
//!
//! Compute is pluggable: [`backends`] defines the kernel surface a network
//! runs on and ships a tuned CPU implementation plus a scalar reference used as
//! an oracle. Select one with `KERNRT_BACKEND=cpu|reference`, or per model in a
//! serve manifest.

pub mod arch;
pub mod backends;
pub mod formats;
pub mod generate;
pub mod ops;
#[cfg(feature = "serve")]
pub mod serve;
pub mod tensor;
pub mod tokenizer;
