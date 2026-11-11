//! Prefill throughput: feed a prompt of `n` tokens in one call and report
//! tokens/second, so the batched path can be compared against the decode loop.
//!
//! ```text
//! cargo run --release --example prefill_rate -- 128
//! VECLIB_MAXIMUM_THREADS=1 cargo run --release --example prefill_rate -- 128
//! KERNRT_Q8_0X4=1 cargo run --release --example prefill_rate -- 128
//! ```

use kernrt::arch::llm::Lfm2;
use kernrt::arch::llm::Llama;
use kernrt::tokenizer::BpeTokenizer;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

/// The benchmarked model, Llama- or LFM2-flavoured.
enum Model {
    Llama(Llama),
    Lfm2(Lfm2),
}

impl Model {
    fn load(dir: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let is_lfm2 = config.contains("\"model_type\": \"lfm2\"")
            || config.contains("\"model_type\":\"lfm2\"");
        if is_lfm2 {
            Ok(Model::Lfm2(Lfm2::from_hf(dir)?))
        } else {
            Ok(Model::Llama(Llama::from_hf(dir)?))
        }
    }

    fn prefill(&mut self, prompt: &[usize], start: usize) {
        match self {
            Model::Llama(m) => {
                std::hint::black_box(m.prefill(prompt, start));
            }
            Model::Lfm2(m) => {
                std::hint::black_box(m.prefill(prompt, start));
            }
        }
    }

    fn storage(&self) -> &'static str {
        match self {
            Model::Llama(m) => m.storage(),
            Model::Lfm2(m) => m.storage(),
        }
    }

    fn seq_len(&self) -> usize {
        match self {
            Model::Llama(m) => m.config().seq_len,
            Model::Lfm2(m) => m.config().seq_len,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::var("KERNRT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2"));
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let mut model = Model::load(&dir)?;
    let tokenizer = BpeTokenizer::load(dir.join("tokenizer.json"))?;

    // Build a prompt of exactly `n` tokens by repeating a sentence, so the
    // prompt is real text rather than token ids the model never sees.
    let seed = "The quick brown fox jumps over the lazy dog. ";
    let mut prompt = Vec::new();
    while prompt.len() < n {
        prompt.extend(tokenizer.encode(seed, false));
    }
    prompt.truncate(n);
    if prompt.len() < 2 {
        return Err("prompt too short".into());
    }
    println!(
        "storage {} | prompt {} tokens | ctx {}",
        model.storage(),
        prompt.len(),
        model.seq_len()
    );

    // prefill(prompt, 0) always writes positions 0..n, so repeating it on one
    // model is idempotent for the KV cache and avoids paying load time.
    //
    // The first call is timed on its own: f32/bf16 prefill builds its widened
    // projection copies there, while native Q8_0x4 initializes its batch
    // scratch. Folding that work into a best-of-N would report a rate no
    // individual steady-state call ever achieves.
    let start = Instant::now();
    model.prefill(&prompt, 0);
    let first = start.elapsed().as_secs_f64();

    let mut samples: Vec<f64> = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        model.prefill(&prompt, 0);
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let toks = prompt.len() as f64;
    println!("first call (one-time prefill setup): {:.1} ms", first * 1e3);
    println!(
        "prefill: {:.1} ms best ({:.1} ms median) = {:.0} tok/s best, {:.0} median",
        min * 1e3,
        median * 1e3,
        toks / min,
        toks / median
    );
    Ok(())
}
