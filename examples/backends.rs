//! Run one prompt through every registered backend and compare.
//!
//! ```text
//! cargo run --release --example backends
//! cargo run --release --example backends -- "The capital of France is" 16
//! ```
//!
//! The point is not the numbers — the reference backend exists to be slow —
//! but that the token ids match. A backend that produces different tokens is
//! wrong regardless of how fast it is.

use kernrt::arch::llm::Llama;
use kernrt::backends;
use kernrt::tokenizer::BpeTokenizer;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    let tokenizer = BpeTokenizer::load(dir.join("tokenizer.json"))?;
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    let prompt_ids = tokenizer.encode(&prompt, false);
    println!(
        "prompt {prompt:?} ({} tokens), {steps} generated",
        prompt_ids.len()
    );
    println!(
        "KERNRT_BACKEND={:?}, available: {}",
        std::env::var("KERNRT_BACKEND").unwrap_or_else(|_| "cpu (default)".into()),
        backends::available().join(", ")
    );

    let mut baseline: Option<Vec<usize>> = None;
    for name in backends::available() {
        let mut model = Llama::from_hf_with(&dir, backends::open(name)?)?;

        // Prefill the prompt so the timed loop is pure decode.
        let mut logits = model.prefill(&prompt_ids, 0).to_vec();
        let start = Instant::now();
        let mut generated = Vec::with_capacity(steps);
        for _ in 0..steps {
            let next = argmax(&logits);
            generated.push(next);
            let pos = prompt_ids.len() + generated.len() - 1;
            logits = model.forward(next, pos).to_vec();
        }
        let elapsed = start.elapsed();
        let rate = steps as f64 / elapsed.as_secs_f64();

        let agrees = match &baseline {
            None => {
                baseline = Some(generated.clone());
                "baseline"
            }
            Some(want) if *want == generated => "matches",
            Some(_) => "DIFFERS",
        };
        println!(
            "{name:<10} storage={:<8} {:>8.1} tok/s  {agrees}",
            model.storage(),
            rate
        );
    }
    Ok(())
}
