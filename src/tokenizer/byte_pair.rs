//! The llama2.c tokenizer: a SentencePiece-style BPE with a small custom file
//! format.
//!
//! Layout:
//!
//! ```text
//! max_token_length  i32, little-endian
//! per token, vocab_size times:
//!   score           f32, little-endian (BPE merge priority)
//!   length          i32, little-endian
//!   bytes           `length` bytes of the piece, not NUL-terminated
//! ```
//!
//! The vocabulary size is not stored; it is taken from the model's config.
//! Encoding is the same greedy merge loop `run.c` uses, and decoding reproduces
//! its two quirks: a piece after BOS loses one leading space, and pieces shaped
//! like `<0x1b>` denote a single raw byte rather than text.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// A loaded tokenizer.
pub struct Tokenizer {
    vocab: Vec<String>,
    scores: Vec<f32>,
    index: HashMap<String, usize>,
    max_token_length: usize,
}

/// Everything that can go wrong reading a tokenizer file.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    /// A length field ran past the end of the file.
    TooShort {
        at: usize,
    },
    /// A piece was not valid UTF-8.
    InvalidUtf8 {
        index: usize,
    },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "i/o error: {e}"),
            TokenizerError::TooShort { at } => write!(f, "tokenizer file ends at byte {at}"),
            TokenizerError::InvalidUtf8 { index } => {
                write!(f, "token {index} is not valid UTF-8")
            }
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl Tokenizer {
    /// Read a tokenizer, taking the vocabulary size from the model config.
    pub fn load(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self, TokenizerError> {
        Self::from_bytes(&fs::read(path)?, vocab_size)
    }

    fn from_bytes(bytes: &[u8], vocab_size: usize) -> Result<Self, TokenizerError> {
        let mut at = 0usize;
        let max_token_length = take_i32(bytes, &mut at)? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let score = take_f32(bytes, &mut at)?;
            let len = take_i32(bytes, &mut at)? as usize;
            if at + len > bytes.len() {
                return Err(TokenizerError::TooShort { at });
            }
            let piece = String::from_utf8(bytes[at..at + len].to_vec())
                .map_err(|_| TokenizerError::InvalidUtf8 { index: i })?;
            at += len;
            vocab.push(piece);
            scores.push(score);
        }
        // Later duplicates win, which only matters for malformed files.
        let index = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        Ok(Tokenizer {
            vocab,
            scores,
            index,
            max_token_length,
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Longest piece length, used by `run.c`'s merge buffer.
    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }

    /// Encode text into token ids, prepending BOS and the dummy space prefix,
    /// then merging adjacent pairs greedily by score.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![1]; // BOS
        if !text.is_empty() {
            if let Some(&space) = self.index.get(" ") {
                tokens.push(space);
            }
        }

        // One token per codepoint, falling back to raw bytes.
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let encoded: &str = ch.encode_utf8(&mut buf);
            match self.index.get(encoded) {
                Some(&id) => tokens.push(id),
                None => {
                    for byte in encoded.as_bytes() {
                        tokens.push(*byte as usize + 3);
                    }
                }
            }
        }

        // Merge the highest-scoring adjacent pair until no pair merges.
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let joined = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                if let Some(&id) = self.index.get(&joined) {
                    let score = self.scores[id];
                    if best.is_none_or(|(_, _, best_score)| score > best_score) {
                        best = Some((i, id, score));
                    }
                }
            }
            match best {
                Some((at, id, _)) => {
                    tokens[at] = id;
                    tokens.remove(at + 1);
                }
                None => break,
            }
        }
        tokens
    }

    /// Decode one token to bytes, given the previous token for the
    /// leading-space rule.
    pub fn decode(&self, prev_token: usize, token: usize) -> Vec<u8> {
        let piece = &self.vocab[token];
        let piece = if prev_token == 1 {
            piece.strip_prefix(' ').unwrap_or(piece)
        } else {
            piece
        };
        match byte_piece(piece) {
            Some(byte) => vec![byte],
            None => piece.as_bytes().to_vec(),
        }
    }
}

/// Recognise the `<0x1b>` form used for raw bytes.
fn byte_piece(piece: &str) -> Option<u8> {
    let b = piece.as_bytes();
    if b.len() == 6 && b[0] == b'<' && b[1] == b'0' && b[2] == b'x' && b[5] == b'>' {
        let hi = (b[3] as char).to_digit(16)?;
        let lo = (b[4] as char).to_digit(16)?;
        Some((hi * 16 + lo) as u8)
    } else {
        None
    }
}

fn take_i32(bytes: &[u8], at: &mut usize) -> Result<i32, TokenizerError> {
    let end = *at + 4;
    if end > bytes.len() {
        return Err(TokenizerError::TooShort { at: *at });
    }
    let value = i32::from_le_bytes([bytes[*at], bytes[*at + 1], bytes[*at + 2], bytes[*at + 3]]);
    *at = end;
    Ok(value)
}

fn take_f32(bytes: &[u8], at: &mut usize) -> Result<f32, TokenizerError> {
    let end = *at + 4;
    if end > bytes.len() {
        return Err(TokenizerError::TooShort { at: *at });
    }
    let value = f32::from_le_bytes([bytes[*at], bytes[*at + 1], bytes[*at + 2], bytes[*at + 3]]);
    *at = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny vocabulary with a merge ("a"+"b" -> "ab") and a known space
    /// token, so the BPE loop has something to do.
    fn tiny() -> Tokenizer {
        let pieces: [(&str, f32); 7] = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
            (" ", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("ab", 5.0),
        ];
        let mut bytes = (2i32).to_le_bytes().to_vec(); // max_token_length
        for (piece, score) in pieces {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(piece.len() as i32).to_le_bytes());
            bytes.extend_from_slice(piece.as_bytes());
        }
        Tokenizer::from_bytes(&bytes, pieces.len()).unwrap()
    }

    #[test]
    fn encodes_bos_space_and_greedy_merge() {
        // [BOS, " ", "a", "b"] -> merge "a"+"b" -> [BOS, " ", "ab"]
        assert_eq!(tiny().encode("ab"), vec![1, 3, 6]);
    }

    #[test]
    fn falls_back_to_bytes_for_unknown_characters() {
        // Byte tokens are `byte + 3`, so the vocabulary must reach at least 259
        // for the fallback ids to be valid.
        let mut bytes = (2i32).to_le_bytes().to_vec();
        for i in 0..300 {
            let piece = format!("<{i}>");
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
            bytes.extend_from_slice(&(piece.len() as i32).to_le_bytes());
            bytes.extend_from_slice(piece.as_bytes());
        }
        let tokenizer = Tokenizer::from_bytes(&bytes, 300).unwrap();
        // 'Z' is 0x5A, and it is absent from the vocabulary.
        assert_eq!(tokenizer.encode("Z"), vec![1, 0x5A + 3]);
    }

    #[test]
    fn decode_strips_one_space_after_bos() {
        let tokenizer = tiny();
        assert_eq!(tokenizer.decode(1, 3), Vec::<u8>::new());
        assert_eq!(tokenizer.decode(4, 3), b" ".to_vec());
    }

    #[test]
    fn decode_expands_byte_pieces() {
        let mut bytes = 1i32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&6i32.to_le_bytes());
        bytes.extend_from_slice(b"<0x1b>");
        let tokenizer = Tokenizer::from_bytes(&bytes, 1).unwrap();
        assert_eq!(tokenizer.decode(0, 0), vec![0x1b]);
    }

    #[test]
    fn rejects_truncated_files() {
        let mut bytes = 0i32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&99i32.to_le_bytes()); // claims 99 bytes, has none
        assert!(matches!(
            Tokenizer::from_bytes(&bytes, 1),
            Err(TokenizerError::TooShort { .. })
        ));
    }
}
