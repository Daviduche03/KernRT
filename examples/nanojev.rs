//! Run the NanoJev decision head on the README example, check it against the
//! CPU fp32 PyTorch oracle, and time repeated decisions.
//!
//! ```text
//! cargo run --release --example nanojev
//! ```
//!
//! The oracle file is produced by `scripts/nanojev_oracle.py`.

use kernrt::arch::decision::{DecisionModel, QuestionType};
use kernrt::tokenizer::BpeTokenizer;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const CHECKPOINT: &str = "target/nanojev";
const ORACLE: &str = "target/nanojev_oracle.json";
/// Leaves are a few hundred tokens; the checkpoint's 40960 positions would
/// allocate a ~9 GB key/value cache.
const SEQ_LEN: usize = 2048;
const BENCH_ITERS: usize = 50;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(CHECKPOINT);
    let load = Instant::now();
    let mut model = DecisionModel::from_checkpoint(&dir, SEQ_LEN)?;
    println!("load: {:?}", load.elapsed());

    let mut tok = BpeTokenizer::load(dir.join("tokenizer/tokenizer.json"))?;
    tok.set_eos(model.eos());

    let criteria = [
        ("left", "Move the aim left."),
        ("right", "Move the aim right."),
        ("shoot", "Fire."),
        ("noop", "Wait."),
    ];
    let decide = |model: &mut DecisionModel, tok: &BpeTokenizer| {
        model
            .decide(
                "The target is left of the aim.",
                QuestionType::Choice,
                "Choose the next action.",
                &criteria,
                tok,
            )
            .expect("decide")
    };

    // Warm up, then check against the oracle.
    decide(&mut model, &tok);
    let probs = decide(&mut model, &tok);

    let oracle: serde_json::Value = serde_json::from_slice(&std::fs::read(ORACLE)?)?;
    let want: Vec<f32> = oracle["example:action"]["probs"]
        .as_array()
        .expect("oracle probs")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    println!("\n  candidate        kernrt      oracle       diff");
    for ((id, _), (got, &w)) in criteria.iter().zip(probs.iter().zip(&want)) {
        println!("  {id:<14} {got:9.6} {w:11.6} {:>10.2e}", (got - w).abs());
    }
    let max = probs
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nmax |diff| = {max:.2e}");

    // Time the whole decision: tokenize + 4 backbone prefills + head.
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        decide(&mut model, &tok);
    }
    let per = start.elapsed() / BENCH_ITERS as u32;
    println!("\ndecision latency: {per:?} over {BENCH_ITERS} runs (single thread)");
    println!("hosted Jev API claim: ~150 ms median");
    Ok(())
}
