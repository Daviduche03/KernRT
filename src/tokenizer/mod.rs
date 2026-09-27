//! Text tokenizers, one per scheme.
//!
//! [`Tokenizer`] is the llama2.c byte-pair encoder; [`Gpt2Tokenizer`] is the
//! HuggingFace byte-level BPE the SmolLM2 family uses. They share no code, so
//! each lives in its own submodule and only the types are re-exported here.

pub mod byte_pair;
pub mod gpt2;

pub use byte_pair::Tokenizer;
pub use gpt2::Gpt2Tokenizer;
