//! Generate a deterministic MLP, save it, load it back, and time inference.
//!
//! ```text
//! cargo run --release --example mlp_infer
//! ```

use kernrt::model::Mlp;
use kernrt::ops::softmax;
use kernrt::tensor::Tensor;
use kernrt::weights;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const INPUT: usize = 784;
const HIDDEN: usize = 128;
const CLASSES: usize = 10;
const ITERATIONS: u32 = 1000;

/// Deterministic pseudo-random fill, so the example is reproducible without
/// pulling in the `rand` crate.
fn deterministic(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
        data.push(x);
    }
    Tensor::from_vec(data, shape).unwrap()
}

/// Scale every element in place — used to mimic `1/sqrt(fan_in)` weight init.
fn scaled(mut t: Tensor, factor: f32) -> Tensor {
    for v in t.data_mut() {
        *v *= factor;
    }
    t
}

fn main() -> Result<(), Box<dyn Error>> {
    // Build the parameters, then persist them exactly as a training script would.
    // Weights are scaled by `1/sqrt(fan_in)` the way a real initializer would be:
    // unscaled random weights over 784 inputs make the logits so wide that
    // softmax saturates to a one-hot vector regardless of the input.
    let params = [
        (
            "fc1.weight",
            scaled(
                deterministic(&[INPUT, HIDDEN], 1),
                1.0 / (INPUT as f32).sqrt(),
            ),
        ),
        ("fc1.bias", deterministic(&[HIDDEN], 2)),
        (
            "fc2.weight",
            scaled(
                deterministic(&[HIDDEN, CLASSES], 3),
                1.0 / (HIDDEN as f32).sqrt(),
            ),
        ),
        ("fc2.bias", deterministic(&[CLASSES], 4)),
    ];
    let refs: Vec<(&str, &Tensor)> = params.iter().map(|(n, t)| (*n, t)).collect();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/mnist_mlp.krt");
    weights::save(&path, &refs)?;
    println!(
        "wrote {} parameters to {}",
        refs.len(),
        path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&path)
            .display()
    );

    // What an inference process does: load the file and run it.
    let loaded = weights::load(&path)?;
    println!("loaded {} tensors", loaded.len());
    let model = Mlp::from_weights(loaded)?;

    let x = deterministic(&[1, INPUT], 7);
    let logits = model.forward(&x)?;
    let probs = softmax(&logits);
    println!("input[:4]  = {:?}", &x.data()[..4]);
    println!("logits[:4] = {:?}", &logits.data()[..4]);

    for _ in 0..ITERATIONS {
        black_box(model.forward(black_box(&x))?);
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(model.forward(black_box(&x))?);
    }
    let per_inference = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERATIONS);
    println!("forward: {per_inference:.2} us/inference");

    let (class, &p) = probs
        .data()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    println!("argmax class {class} with p = {p:.4}");
    Ok(())
}
