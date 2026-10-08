//! Autoregressive language models: a token in, logits for the next out.

pub mod hf;
pub mod llama;

pub use hf::LoadError;
pub use llama::Llama;

/// What the generation loop needs from a decoder.
///
/// Deliberately tiny: [`crate::generate`] can drive any model that scores a
/// token at a position and reports its context length, so it never names a
/// concrete architecture. A new decoder implements this and inherits the loop.
pub trait LanguageModel {
    /// Logits for the token that follows `token` at absolute position `pos`.
    ///
    /// Positions must be fed in order, since attention reads the key/value
    /// cache written by earlier calls.
    fn forward(&mut self, token: usize, pos: usize) -> &[f32];

    /// Maximum number of positions the model can attend over.
    fn context_len(&self) -> usize;
}
