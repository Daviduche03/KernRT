//! Load SmolLM2-135M (HuggingFace safetensors, BF16, GQA) and compare against
//! the `mlx-lm` reference logits.
//!
//! ```text
//! cargo run --release --example smollm2
//! ```

use kernrt::llama::Llama;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// Cosine similarity between two logit vectors, which is the usual way to judge
/// whether two implementations agree on the *shape* of a distribution rather
/// than on individual values.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Rank of `token` in `logits`, 1-based.
fn rank_of(logits: &[f32], token: usize) -> usize {
    let want = logits[token];
    logits.iter().filter(|&&v| v > want).count() + 1
}

fn read_numbers<T: std::str::FromStr>(path: &PathBuf) -> Result<Vec<T>, Box<dyn Error>> {
    Ok(std::fs::read_to_string(path)?
        .split_whitespace()
        .map(|s| s.parse::<T>().map_err(|_| "parse"))
        .collect::<Result<_, _>>()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    let mut model = Llama::from_hf(&dir)?;
    let c = *model.config();
    println!(
        "loaded: dim {} hidden {} layers {} heads {}/{} vocab {} ctx {} rope_theta {} interleaved {}",
        c.dim,
        c.hidden_dim,
        c.n_layers,
        c.n_heads,
        c.n_kv_heads,
        c.vocab_size,
        c.seq_len,
        c.rope_theta,
        c.rope_interleaved
    );

    println!("weight storage: {}", model.storage());

    let ids: Vec<usize> = read_numbers(&dir.join("smollm2_ref_ids.txt"))?;
    let reference: Vec<f32> = read_numbers(&dir.join("smollm2_ref_logits.txt"))?;
    println!("token ids: {ids:?}");

    let mut logits = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        logits = black_box(model.forward(id, pos)).to_vec();
    }
    let max_diff = logits
        .iter()
        .zip(&reference)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);

    // The reference logits are computed by MLX in f32, so compare to f32
    // precision rather than the file's bf16 precision. Print with enough digits
    // to show the real error: `{:.4}` rounds our ~7e-6 to `0.0000`.
    let next = argmax(&reference);
    println!(
        "max |diff| vs mlx-lm = {max_diff:.2e}, argmax {} vs {} (ref token {} at rank {})",
        argmax(&logits),
        next,
        next,
        rank_of(&logits, next)
    );
    println!("cosine similarity = {:.8}", cosine(&logits, &reference));

    // Decode latency. The machine is shared and noisy, so report the minimum
    // (the noise floor) alongside the median rather than a plain mean, which
    // interference spikes inflate.
    let mut samples: Vec<f64> = Vec::with_capacity(300);
    for i in 0..300 {
        let id = ids[i % ids.len()];
        let pos = i % ids.len();
        let start = Instant::now();
        black_box(model.forward(id, pos));
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let min = samples[0];
    let median = samples[samples.len() / 2];
    println!(
        "forward: {min:.1} us/token best ({median:.1} us median), {:.1} tok/s best",
        1e6 / min
    );
    Ok(())
}
