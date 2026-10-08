//! Text tokenizers, one per scheme.
//!
//! [`Tokenizer`] is the llama2.c byte-pair encoder; [`BpeTokenizer`] is the
//! HuggingFace byte-level BPE that the SmolLM2 and Qwen families ship. They
//! share no code, so each lives in its own submodule and only the types are
//! re-exported here.

pub mod bpe;
pub mod byte_pair;

pub use bpe::BpeTokenizer;
pub use byte_pair::Tokenizer;
