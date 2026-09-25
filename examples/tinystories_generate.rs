//! Generate a TinyStories sentence with a prompt, and check the greedy output
//! against `run.c`.
//!
//! ```text
//! cargo run --release --example tinystories_generate -- "Once upon a time" 40
//! cargo run --release --example tinystories_generate -- "Once upon a time" 40 0.8
//! ```

use kernrt::generate::{Sampler, generate};
use kernrt::llama::Llama;
use kernrt::tokenizer::Tokenizer;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const REFERENCE_PROMPT: &str = "Once upon a time";
const REFERENCE_STEPS: usize = 40;

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path = root.join("target/stories15M.bin");
    let tokenizer_path = root.join("target/tokenizer.bin");
    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!(
            "missing fixtures; see scripts/dump_logits.c\n  expected {}",
            model_path.display()
        );
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| REFERENCE_PROMPT.to_string());
    let steps: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(REFERENCE_STEPS);
    let temperature: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let mut model = Llama::load(&model_path)?;
    let tokenizer = Tokenizer::load(&tokenizer_path, model.config().vocab_size)?;
    let mut sampler = if temperature > 0.0 {
        Sampler::temperature(temperature, 0x1234_5678)
    } else {
        Sampler::greedy()
    };

    println!("prompt tokens: {:?}", tokenizer.encode(&prompt));
    let start = Instant::now();
    let text = generate(&mut model, &tokenizer, &mut sampler, &prompt, steps);
    let elapsed = start.elapsed();
    println!("{text}");

    if temperature <= 0.0 && prompt == REFERENCE_PROMPT && steps == REFERENCE_STEPS {
        let reference = root.join("target/ref_gen.txt");
        if reference.exists() {
            let want = std::fs::read_to_string(&reference)?;
            let matches = text.trim_end() == want.trim_end();
            println!("\n[matches run.c: {matches}]");
            if !matches {
                println!("run.c: {}", want.trim_end());
            }
        }
    }

    println!(
        "{:.1} us/token ({:.1} tok/s)",
        elapsed.as_secs_f64() * 1e6 / steps as f64,
        steps as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}
