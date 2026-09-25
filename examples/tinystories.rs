//! Run a real TinyStories transformer and compare against the llama2.c
//! reference logits.
//!
//! ```text
//! cargo run --release --example tinystories
//! ```
//!
//! Expects `target/stories15M.bin` and `target/ref_logits.txt`, produced by the
//! steps documented in `scripts/dump_logits.c`.

use kernrt::llama::Llama;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const TOKENS: [usize; 4] = [1, 299, 1234, 5678];

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path = root.join("target/stories15M.bin");
    let reference_path = root.join("target/ref_logits.txt");
    if !model_path.exists() || !reference_path.exists() {
        eprintln!(
            "missing fixtures; see scripts/dump_logits.c for how to build them\n  expected {}",
            model_path.display()
        );
        std::process::exit(1);
    }

    let mut model = Llama::load(&model_path)?;
    let c = model.config();
    println!(
        "loaded: dim {} hidden {} layers {} heads {}/{} vocab {} ctx {}",
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len
    );

    let mut logits = Vec::new();
    for (pos, &token) in TOKENS.iter().enumerate() {
        logits = black_box(model.forward(token, pos)).to_vec();
    }

    let reference: Vec<f32> = std::fs::read_to_string(&reference_path)?
        .lines()
        .map(|line| line.parse::<f32>())
        .collect::<Result<_, _>>()?;
    assert_eq!(logits.len(), reference.len(), "logit vector length");

    let mut max_diff = 0.0f32;
    let mut argmax_agree = true;
    for (got, want) in logits.iter().zip(&reference) {
        max_diff = max_diff.max((got - want).abs());
    }
    argmax_agree &= argmax(&logits) == argmax(&reference);

    println!(
        "logits: {} values, max |diff| vs run.c = {:e}, argmax {} vs {}",
        logits.len(),
        max_diff,
        argmax(&logits),
        argmax(&reference)
    );
    if !argmax_agree {
        eprintln!("argmax disagreement with the reference");
        std::process::exit(1);
    }

    // Decode latency: reuse the model, resetting positions.
    let iterations = 200;
    let start = Instant::now();
    for _ in 0..iterations {
        for (pos, &token) in TOKENS.iter().enumerate() {
            black_box(model.forward(token, pos));
        }
    }
    let per_token = start.elapsed().as_secs_f64() * 1e6 / (iterations * TOKENS.len()) as f64;
    println!("forward: {per_token:.1} us/token");
    Ok(())
}
