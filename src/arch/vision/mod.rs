//! Single-shot image models: an image in, a prediction out.
//!
//! The dense classifier, the YOLO26 detector, and the opt-in profiler the
//! detector keeps out of its own file ([`bench`]).

pub(crate) mod bench;
pub mod mlp;
pub mod yolo26;
