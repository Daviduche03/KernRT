//! Benchmark one decode step with a one-token context (llama-bench's `tg1`).
//!
//! ```text
//! cargo run --release --example decode_rate
//! KERNRT_Q8_0X4=1 cargo run --release --example decode_rate
//! ```

use kernrt::arch::llm::Lfm2;
use kernrt::arch::llm::Llama;
use kernrt::tokenizer::BpeTokenizer;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

/// The benchmarked model, Llama- or LFM2-flavoured. Both expose the same
/// decode loop; the match below is the only place the driver differs.
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

    fn forward(&mut self, token: usize, pos: usize) {
        match self {
            Model::Llama(m) => {
                black_box(m.forward(token, pos));
            }
            Model::Lfm2(m) => {
                black_box(m.forward(token, pos));
            }
        }
    }

    fn storage(&self) -> &'static str {
        match self {
            Model::Llama(m) => m.storage(),
            Model::Lfm2(m) => m.storage(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::var("KERNRT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2"));
    let mut model = Model::load(&dir)?;
    let tokenizer = BpeTokenizer::load(dir.join("tokenizer.json"))?;
    let token = tokenizer.encode("The", false)[0];

    for _ in 0..5 {
        model.forward(token, 0);
    }
    let mut samples = Vec::with_capacity(300);
    for _ in 0..300 {
        let start = Instant::now();
        model.forward(token, 0);
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
