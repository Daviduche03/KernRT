//! Benchmark one decode step with a one-token context (llama-bench's `tg1`).
//!
//! ```text
//! cargo run --release --example decode_rate
//! KERNRT_Q8_0X4=1 cargo run --release --example decode_rate
//! ```

use kernrt::gpt2_tokenizer::Gpt2Tokenizer;
use kernrt::llama::Llama;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    let mut model = Llama::from_hf(&dir)?;
    let tokenizer = Gpt2Tokenizer::load(dir.join("tokenizer.json"))?;
    let token = tokenizer.encode("The", false)[0];

    for _ in 0..5 {
        black_box(model.forward(token, 0));
    }
    let mut samples = Vec::with_capacity(300);
    for _ in 0..300 {
        let start = Instant::now();
        black_box(model.forward(token, 0));
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let p10 = samples[samples.len() / 10];
    let median = samples[samples.len() / 2];
    println!(
        "storage={} token={token} tg1: {:.1} tok/s p10 ({:.1} tok/s median)",
        model.storage(),
        1e6 / p10,
        1e6 / median
    );
    Ok(())
}
