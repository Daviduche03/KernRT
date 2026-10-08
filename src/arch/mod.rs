//! Runnable networks, grouped by family.
//!
//! Each family is a thin arrangement of [`crate::ops`] over weights read by
//! [`crate::formats`]. `llm` holds the autoregressive decoders; `vision` holds
//! the single-shot image models.

pub mod decision;
pub mod llm;
pub mod vision;
