//! A byte-level BPE tokenizer, the HuggingFace `tokenizer.json` format used by
//! the SmolLM2 and Qwen families.
//!
//! This is a different animal from the llama2.c tokenizer in
//! [`crate::tokenizer::byte_pair`]:
//!
//! * Bytes, not text, are the alphabet. Every one of the 256 bytes maps to a
//!   printable "byte-level" codepoint (space becomes `Ġ`), so any input is
//!   representable and there is no unknown token.
//! * Merges are ranked rather than scored: the pair with the *lowest* rank wins,
//!   and the vocabulary is not consulted for priority.
//! * Text is split into words first, and merges never cross a word boundary — a
//!   detail that changes the tokenization of numbers and punctuation. The split
//!   rule is model-specific: [`PreTokenizer::Gpt2`] and [`PreTokenizer::Qwen`]
//!   differ on things like whether `":\n"` is one word, so it is read from the
//!   file rather than assumed.
//!
//! The file format is HuggingFace's `tokenizer.json`, of which only the BPE
//! model, the added tokens and the pre-tokenizer settings are read.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// A loaded byte-level BPE tokenizer.
pub struct BpeTokenizer {
    /// Piece text for each id; the inverse of `vocab`.
    pieces: Vec<String>,
    vocab: HashMap<String, usize>,
    /// Merge rank, keyed by the two halves. A lower rank merges first.
    merges: HashMap<(String, String), usize>,
    /// Multi-character special tokens, matched before any BPE splitting.
    added: Vec<(String, usize)>,
    /// How text is split into words before BPE. The pattern differs per model
    /// family and changes the result, so it is read from the file rather than
    /// assumed.
    pretok: PreTokenizer,
    bos: usize,
    eos: usize,
}

/// The word-splitting rule a model declares in `pre_tokenizer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizer {
    /// GPT-2's `\p{N}+` splitter, used by SmolLM2 and friends.
    Gpt2,
    /// Qwen2/Qwen3's splitter, whose punctuation rule keeps trailing newlines
    /// and whose digit rule is single-character.
    Qwen,
}

/// Everything that can go wrong reading a `tokenizer.json`.
#[derive(Debug)]
pub enum BpeError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// A field the loader needs was missing or the wrong shape.
    BadFile(&'static str),
    /// A merge referred to a pair that is not a known piece.
    BadMerge(usize),
}

impl fmt::Display for BpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpeError::Io(e) => write!(f, "i/o error: {e}"),
            BpeError::Json(e) => write!(f, "tokenizer json error: {e}"),
            BpeError::BadFile(what) => write!(f, "tokenizer.json is missing {what}"),
            BpeError::BadMerge(at) => write!(f, "merge {at} is malformed"),
        }
    }
}

impl std::error::Error for BpeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BpeError::Io(e) => Some(e),
            BpeError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BpeError {
    fn from(e: std::io::Error) -> Self {
        BpeError::Io(e)
    }
}

impl From<serde_json::Error> for BpeError {
    fn from(e: serde_json::Error) -> Self {
        BpeError::Json(e)
    }
}

impl BpeTokenizer {
    /// Read a `tokenizer.json`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    fn from_json(text: &str) -> Result<Self, BpeError> {
        let root: Value = serde_json::from_str(text)?;

        let vocab_json = root
            .pointer("/model/vocab")
            .and_then(Value::as_object)
            .ok_or(BpeError::BadFile("model.vocab"))?;
        let mut pieces = vec![String::new(); vocab_json.len()];
        let mut vocab = HashMap::with_capacity(vocab_json.len());
        for (piece, id) in vocab_json {
            let id = id.as_u64().ok_or(BpeError::BadFile("a numeric vocab id"))? as usize;
            if id >= pieces.len() {
                return Err(BpeError::BadFile("vocab ids within range"));
            }
            pieces[id] = piece.clone();
            vocab.insert(piece.clone(), id);
        }

        let merges_json = root
            .pointer("/model/merges")
            .and_then(Value::as_array)
            .ok_or(BpeError::BadFile("model.merges"))?;
        let mut merges = HashMap::with_capacity(merges_json.len());
        for (rank, entry) in merges_json.iter().enumerate() {
            // Two spellings in the wild: GPT-2's `"first second"` string and
            // the newer `["first", "second"]` pair (Qwen and most recent
            // tokenizers).
            let pair = match entry {
                Value::String(s) => s
                    .split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or(BpeError::BadMerge(rank))?,
                Value::Array(pair) => {
                    let first = pair.first().and_then(Value::as_str);
                    let second = pair.get(1).and_then(Value::as_str);
                    match (first, second) {
                        (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                        _ => return Err(BpeError::BadMerge(rank)),
                    }
                }
                _ => return Err(BpeError::BadMerge(rank)),
            };
            merges.insert(pair, rank);
        }

        // Added tokens carry their own ids and may sort after the BPE vocab.
        let mut added = Vec::new();
        if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
            for entry in list {
                let (Some(piece), Some(id)) = (
                    entry.get("content").and_then(Value::as_str),
                    entry.get("id").and_then(Value::as_u64),
                ) else {
                    return Err(BpeError::BadFile("added_tokens entries"));
                };
                let id = id as usize;
                if id >= pieces.len() {
                    pieces.resize(id + 1, String::new());
                }
                pieces[id] = piece.to_string();
                added.push((piece.to_string(), id));
            }
        }
        // Longest first, so `<|im_start|>` is tried before `<|im_start|` prefixes.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let find = |name: &str, fallback: usize| -> usize {
            root.get(name)
                .and_then(Value::as_str)
                .and_then(|s| vocab.get(s).copied())
                .unwrap_or(fallback)
        };
        let bos = find("bos_token", 1);
        let eos = find("eos_token", 0);

        Ok(BpeTokenizer {
            pieces,
            vocab,
            merges,
            added,
            pretok: detect_pretokenizer(&root),
            bos,
            eos,
        })
    }

    /// Number of token ids.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Beginning-of-sequence id.
    pub fn bos(&self) -> usize {
        self.bos
    }

    /// End-of-sequence id.
    pub fn eos(&self) -> usize {
        self.eos
    }

    /// Override the end-of-sequence id.
    ///
    /// `tokenizer.json` has no top-level `eos_token` field, so Qwen-family
    /// tokenizers load with the fallback. The model config (`eos_token_id`)
    /// is the authority; callers that read it pass the value here.
    pub fn set_eos(&mut self, eos: usize) {
        self.eos = eos;
    }

    /// Override the beginning-of-sequence id, for the same reason as
    /// [`set_eos`](Self::set_eos).
    pub fn set_bos(&mut self, bos: usize) {
        self.bos = bos;
    }

    /// Encode `text`, prefixing `bos` when `add_bos` is set.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let mut tokens = Vec::new();
        if add_bos {
            tokens.push(self.bos);
        }
        // Added tokens are matched in the raw text and their surrounding pieces
        // are encoded normally.
        let mut rest = text;
        while !rest.is_empty() {
            match self
                .added
                .iter()
                .find(|(piece, _)| rest.starts_with(piece.as_str()))
            {
                Some((piece, id)) => {
                    tokens.push(*id);
                    rest = &rest[piece.len()..];
                }
                None => {
                    let end = self
                        .added
                        .iter()
                        .filter_map(|(piece, _)| rest.find(piece.as_str()))
                        .min()
                        .unwrap_or(rest.len());
                    let (head, tail) = rest.split_at(end);
                    self.encode_ordinary(head, &mut tokens);
                    rest = tail;
                }
            }
        }
        tokens
    }

    /// Encode text with no added tokens in it: split into words, then BPE each.
    fn encode_ordinary(&self, text: &str, tokens: &mut Vec<usize>) {
        if text.is_empty() {
            return;
        }
        let bytes = byte_level_encode(text);
        let words = match self.pretok {
            PreTokenizer::Gpt2 => split_words(&bytes),
            PreTokenizer::Qwen => split_words_qwen(&bytes),
        };
        for word in words {
            let pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            for piece in self.bpe(pieces) {
                // A piece that never appeared in the vocabulary would be a
                // malformed file; the base 256 byte pieces always exist.
                if let Some(&id) = self.vocab.get(&piece) {
                    tokens.push(id);
                }
            }
        }
    }

    /// Merge the lowest-ranked adjacent pair until no pair has a rank.
    fn bpe(&self, mut pieces: Vec<String>) -> Vec<String> {
        while pieces.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..pieces.len() - 1 {
                if let Some(&rank) = self.merges.get(&(pieces[i].clone(), pieces[i + 1].clone())) {
                    if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                        best = Some((i, rank));
                    }
                }
            }
            match best {
                Some((at, _)) => {
                    pieces[at] = format!("{}{}", pieces[at], pieces[at + 1]);
                    pieces.remove(at + 1);
                }
                None => break,
            }
        }
        pieces
    }

    /// Decode token ids back to bytes. Unlike the llama2.c tokenizer there is
    /// no leading-space rule: the space is part of the byte-level piece.
    pub fn decode(&self, tokens: &[usize]) -> Vec<u8> {
        let mut text = String::new();
        for &token in tokens {
            if let Some(piece) = self.pieces.get(token) {
                text.push_str(piece);
            }
        }
        byte_level_decode(&text)
    }

    /// Decode token ids straight to a `String`, replacing invalid bytes.
    pub fn decode_text(&self, tokens: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode(tokens)).into_owned()
    }

    /// Id of a piece, if it is in the vocabulary.
    pub fn token_id(&self, piece: &str) -> Option<usize> {
        self.vocab.get(piece).copied()
    }
}

/// Pick the pre-tokenizer from the file's declared split pattern.
///
/// Only two patterns are supported. Any other declared pattern falls back to
/// GPT-2, which is the default for models that split with `ByteLevel`'s own
/// regex (SmolLM2) or that declare no pre-tokenizer at all.
fn detect_pretokenizer(root: &Value) -> PreTokenizer {
    let pattern = root
        .pointer("/pre_tokenizer/pretokenizers/0/pattern/Regex")
        .and_then(Value::as_str)
        .or_else(|| {
            root.pointer("/pre_tokenizer/pattern/Regex")
                .and_then(Value::as_str)
        });
    match pattern {
        // Qwen2/Qwen3's splitter is recognisable by its single-digit rule and
        // the punctuation rule that absorbs trailing newlines.
        Some(p) if p.contains("\\p{N}") && !p.contains("\\p{N}+") => PreTokenizer::Qwen,
        _ => PreTokenizer::Gpt2,
    }
}

/// The GPT-2 pre-tokenizer pattern: contractions, letter runs, digit runs,
/// punctuation runs and whitespace runs. Merges never cross these boundaries,
/// which is why "1234" does not merge into a neighbouring word.
///
/// The pattern is `\s?\p{L}+|\s?\p{N}+|...`: a single leading space is part of
/// the following run, not a separate piece, so `"a b"` splits as `["a", "Ġb"]`.
/// The input is byte-level text, so it can contain the shifted multi-byte
/// markers; the splitter walks chars rather than raw byte offsets.
pub fn split_words(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let class = Class::of(chars[i]);
        if class == Class::Space || class == Class::OtherWhitespace {
            i = split_whitespace_run(&chars, i, &mut words);
            continue;
        }
        let start = i;
        while i < chars.len() && Class::of(chars[i]) == class {
            i += 1;
        }
        words.push(chars[start..i].iter().collect());
    }
    words
}

/// Emit the pieces covering the whitespace run at `at`, returning the index
/// just past everything consumed. The rules come from GPT-2's alternation
/// `\s?\p{L}+|\s?\p{N}+|...|\s+(?!\S)`:
///
/// * A run of spaces before a word gives up its final space to lead that word,
///   so `"a b"` is `["a", "Ġb"]` and `"a  b"` is `["a", "Ġ", "Ġb"]`.
/// * A run with a tab or newline in it turns into single-character pieces, and
///   its final character leads the word: `"\n\nx"` is `["Ċ", "Ċx"]`. When the
///   run instead ends in spaces, those spaces stay with the piece: `"\n    x"`
///   is `["ĊĠĠĠĠ", "x"]`... except that a single trailing space leads the word,
///   so `"\n x"` is `["Ċ", "Ġx"]`.
/// * A run with nothing after it is one piece: `"x   "` is `["x", "ĠĠĠĠ"]`.
fn split_whitespace_run(chars: &[char], at: usize, words: &mut Vec<String>) -> usize {
    let mut end = at;
    while end < chars.len()
        && matches!(Class::of(chars[end]), Class::Space | Class::OtherWhitespace)
    {
        end += 1;
    }
    if end == chars.len() {
        words.push(chars[at..end].iter().collect());
        return end;
    }
    let run = &chars[at..end];
    // Exactly one whitespace character leads the word that follows: for spaces
    // it is the last one (`"a  b"` -> `["a", "Ġ", "Ġb"]`), for a tab or newline
    // it is that character (`"\n\nx"` -> `["Ċ", "Ċx"]`, `"\n    x"` ->
    // `["ĊĠĠĠ", "Ġx"]`).
    let (whitespace, _) = run.split_at(run.len() - 1);
    if !whitespace.is_empty() {
        words.push(whitespace.iter().collect());
    }
    let mut i = end;
    let class = Class::of(chars[i]);
    while i < chars.len() && Class::of(chars[i]) == class {
        i += 1;
    }
    words.push(chars[end - 1..i].iter().collect());
    i
}

/// The Qwen2/Qwen3 pre-tokenizer pattern, applied to byte-level text:
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
///   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// The rules that differ from GPT-2 and change real tokenizations: a
/// punctuation run absorbs any trailing newlines (`":\n"` is one word, hence
/// one token), a digit run is split one digit at a time, and the word leader
/// may be any non-letter/non-digit/non-newline character, spaces included.
pub fn split_words_qwen(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        // 1. `(?i:'s|'t|'re|'ve|'m|'ll|'d)`.
        if let Some(len) = contraction_len(&chars, i) {
            i += len;
            words.push(chars[start..i].iter().collect());
            continue;
        }
        // 2. `[^\r\n\p{L}\p{N}]?\p{L}+`: a letter run, optionally led by one
        //    character that is not a letter, digit or newline (a space counts).
        let mut j = i;
        if !is_letter(chars[j]) && !is_number(chars[j]) && !is_crlf(chars[j]) {
            j += 1;
        }
        if j < chars.len() && is_letter(chars[j]) {
            while j < chars.len() && is_letter(chars[j]) {
                j += 1;
            }
            i = j;
            words.push(chars[start..i].iter().collect());
            continue;
        }
        // 3. `\p{N}`: exactly one digit.
        if is_number(chars[i]) {
            i += 1;
            words.push(chars[start..i].iter().collect());
            continue;
        }
        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`: an optional space, a punctuation run,
        //    then any trailing newlines.
        let mut j = i;
        if chars[j] == 'Ġ' {
            j += 1;
        }
        if j < chars.len() && is_punct(chars[j]) {
            while j < chars.len() && is_punct(chars[j]) {
                j += 1;
            }
            while j < chars.len() && is_crlf(chars[j]) {
                j += 1;
            }
            i = j;
            words.push(chars[start..i].iter().collect());
            continue;
        }
        // 5. `\s*[\r\n]+`: a newline run, ignoring any whitespace before it.
        let mut j = i;
        while j < chars.len() && is_whitespace(chars[j]) {
            j += 1;
        }
        // Rule 5 needs at least one newline in the run; then split *after* the
        // last newline, leaving any trailing spaces for the next match.
        let mut last_newline: Option<usize> = None;
        if j > i {
            for (offset, &ch) in chars[i..j].iter().enumerate() {
                if is_crlf(ch) {
                    last_newline = Some(i + offset);
                }
            }
        }
        if let Some(nl) = last_newline {
            i = nl + 1;
            words.push(chars[start..i].iter().collect());
            continue;
        }
        // 6/7. `\s+(?!\S)` then `\s+`: a whitespace run. When the run is
        // followed by a non-space character the regex gives up its last space
        // so that space can lead the next word (`"  x"` is `["Ġ", "Ġx"]`);
        // otherwise the whole run is one piece.
        let mut end = j;
        if j < chars.len() && !is_whitespace(chars[j]) && end - i > 1 {
            end -= 1;
        }
        i = end;
        words.push(chars[start..i].iter().collect());
    }
    words
}

/// Match one of `'s 't 're 've 'm 'll 'd` (case-insensitive) at `at`.
fn contraction_len(chars: &[char], at: usize) -> Option<usize> {
    if chars.get(at) != Some(&'\'') {
        return None;
    }
    let rest: String = chars[at + 1..].iter().collect::<String>().to_lowercase();
    for suffix in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        // `rest` begins after the quote, so compare against the suffix minus it.
        if rest.starts_with(&suffix[1..]) {
            return Some(suffix.len());
        }
    }
    None
}

/// The byte-level markers for carriage return (0x0D) and newline (0x0A).
fn is_crlf(ch: char) -> bool {
    matches!(ch, 'Ċ' | 'č')
}

fn is_letter(ch: char) -> bool {
    Class::of(ch) == Class::Letter
}

fn is_number(ch: char) -> bool {
    Class::of(ch) == Class::Decimal
}

fn is_whitespace(ch: char) -> bool {
    matches!(Class::of(ch), Class::Space | Class::OtherWhitespace)
}

/// Anything that is not a letter, digit or whitespace — Qwen's punctuation
/// class (the byte-level text keeps punctuation as itself).
fn is_punct(ch: char) -> bool {
    Class::of(ch) == Class::Other
}

/// The character classes the GPT-2 pattern treats as atomic runs.
#[derive(PartialEq, Clone, Copy)]
enum Class {
    Decimal,
    Letter,
    /// The `Ġ` marker only, which is the one class that leads a following run.
    Space,
    /// Tabs, newlines and the rest of the shifted whitespace bytes.
    OtherWhitespace,
    Other,
}

impl Class {
    fn of(ch: char) -> Self {
        // Byte-level text has already mapped space to the `Ġ` marker (U+0120),
        // so whitespace has to be recognised through that marker, not through
        // `char::is_whitespace`. Printable bytes otherwise map to themselves,
        // so digits and letters remain ASCII.
        if ch == 'Ġ' {
            // Byte `0x20` mapped to the `Ġ` marker; nothing else is a space.
            Class::Space
        } else if ch.is_ascii_digit() {
            Class::Decimal
        } else if ch.is_ascii_alphabetic() {
            Class::Letter
        } else if matches!(ch, 'ĉ' | 'Ċ' | 'ċ' | 'Č' | 'č') {
            // The shifted markers for tab (0x09), newline (0x0A), vertical tab,
            // form feed and carriage return.
            Class::OtherWhitespace
        } else {
            Class::Other
        }
    }
}

/// Map each input byte to its printable byte-level codepoint. Byte `b` is
/// shifted into a private range so that spaces and control characters become
/// ordinary characters that can appear in merge rules.
pub fn byte_level_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &byte in text.as_bytes() {
        // Both branches produce the same character; the shift range is never
        // ASCII, so only the branch that can exceed 0x7F needs the conversion.
        out.push(char::from_u32(PRINTABLE[byte as usize]).expect("valid codepoint"));
    }
    out
}

/// Inverse of [`byte_level_encode`].
pub fn byte_level_decode(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let code = ch as u32;
        match REVERSE.get(&code) {
            Some(&byte) => out.push(byte),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

/// Byte to byte-level codepoint table, built the way GPT-2 builds it. Printable
/// bytes map to *themselves*; only the 68 non-printable bytes are renumbered,
/// starting immediately after the 188 printable ones, which lands space (byte
/// `0x20`) on `0x120` — the `Ġ` marker.
const PRINTABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut next = 0u32;
    let mut i = 0;
    while i < 256 {
        let b = i as u32;
        let keep = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if keep {
            table[i] = b;
        } else {
            table[i] = 256 + next;
            next += 1;
        }
        i += 1;
    }
    table
};

/// Reverse lookup built from [`PRINTABLE`].
static REVERSE: std::sync::LazyLock<HashMap<u32, u8>> = std::sync::LazyLock::new(|| {
    let mut map = HashMap::with_capacity(256);
    for (byte, code) in PRINTABLE.iter().enumerate() {
        map.insert(*code, byte as u8);
    }
    map
});

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary in these tests is written in *byte-level* form. Letters
    /// map to themselves, so `a` is the piece for byte `a`; only non-printable
    /// bytes (space, control characters) are shifted, so a space is `Ġ`.
    fn tiny() -> BpeTokenizer {
        let mut vocab = serde_json::Map::new();
        let pieces = [
            "<|endoftext|>",
            "Ġ",
            "a",
            "b",
            "ab",
            "Ġa",
            "c",
            "d",
            "cd",
            "hello",
            "Ġc",
        ];
        for (id, piece) in pieces.iter().enumerate() {
            vocab.insert(piece.to_string(), serde_json::json!(id));
        }
        let json = serde_json::json!({
            "model": { "vocab": vocab, "merges": ["a b", "c d"] },
            "added_tokens": [{"content": "<|endoftext|>", "id": 0}],
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
        });
        BpeTokenizer::from_json(&json.to_string()).unwrap()
    }

    #[test]
    fn byte_level_round_trips_arbitrary_bytes() {
        for text in ["hello world", "Ġ Ġ", "café\n\ttab", "1234", "\u{fffd}"] {
            let encoded = byte_level_encode(text);
            assert_eq!(byte_level_decode(&encoded), text.as_bytes());
        }
    }

    /// Printable bytes keep their value; the 68 that do not are renumbered after
    /// them, which lands space on `Ġ` (`U+0120`).
    #[test]
    fn space_becomes_g_marker() {
        assert_eq!(byte_level_encode(" "), "Ġ");
        assert_eq!(byte_level_encode("ab"), "ab");
        assert_eq!(byte_level_encode("ab cd"), "abĠcd");
        assert_eq!(byte_level_decode("Ġ"), b" ");
    }

    #[test]
    fn merges_by_rank() {
        let tokenizer = tiny();
        // "ab" merges because its pair has rank 0, "cd" because it has rank 1.
        assert_eq!(tokenizer.encode("ab", false), vec![4]);
        assert_eq!(tokenizer.encode("cd", false), vec![8]);
    }

    #[test]
    fn adds_bos_when_asked() {
        assert_eq!(tiny().encode("ab", true), vec![0, 4]);
    }

    #[test]
    fn splits_digits_from_letters() {
        // "ab12" must not merge across the digit boundary.
        assert_eq!(split_words(&byte_level_encode("ab12")), vec!["ab", "12"]);
        assert_eq!(split_words(&byte_level_encode("a b")), vec!["a", "Ġb"]);
    }

    #[test]
    fn decodes_back_to_bytes() {
        let tokenizer = tiny();
        assert_eq!(tokenizer.decode(&[2, 3]), b"ab");
        assert_eq!(tokenizer.decode_text(&[1, 2]), " a");
    }

    #[test]
    fn matches_added_tokens_before_bpe() {
        let tokenizer = tiny();
        assert_eq!(tokenizer.encode("ab<|endoftext|>", false), vec![4, 0]);
    }

    #[test]
    fn detects_the_qwen_splitter_from_the_declared_pattern() {
        let qwen = serde_json::json!({
            "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
                { "type": "Split", "pattern": { "Regex":
                    "(?i:'s|'t)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" } },
                { "type": "ByteLevel" }
            ]}
        });
        assert_eq!(detect_pretokenizer(&qwen), PreTokenizer::Qwen);
        let gpt2 = serde_json::json!({
            "pre_tokenizer": { "type": "ByteLevel", "use_regex": true }
        });
        assert_eq!(detect_pretokenizer(&gpt2), PreTokenizer::Gpt2);
        assert_eq!(
            detect_pretokenizer(&serde_json::json!({})),
            PreTokenizer::Gpt2
        );
    }

    /// The Qwen splitter differs from GPT-2 exactly where its regex does: a
    /// punct run keeps trailing newlines, digits split one at a time, and a
    /// space can lead a word. These expected splits are the reference
    /// tokenizer's (`tokenizers` crate) for the same inputs.
    #[test]
    fn qwen_splits_like_the_reference() {
        let split = |s: &str| split_words_qwen(&byte_level_encode(s));
        // ":\n" stays one word, so it BPEs to a single token.
        assert_eq!(split("State:\n"), vec!["State", ":Ċ"]);
        assert_eq!(split("3/10.\n"), vec!["3", "/", "1", "0", ".Ċ"]);
        assert_eq!(split("ab12"), vec!["ab", "1", "2"]);
        assert_eq!(
            split("  leading and trailing   "),
            vec!["Ġ", "Ġleading", "Ġand", "Ġtrailing", "ĠĠĠ"]
        );
    }
}
