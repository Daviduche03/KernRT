//! Load the real MNIST MLP exported by `scripts/train_mnist_mlp.py` and check
//! that KernRT reproduces MLX's logits, then report accuracy and latency.
//!
//! ```text
//! python scripts/train_mnist_mlp.py
//! cargo run --release --example mnist_infer
//! ```

use kernrt::model::Mlp;
use kernrt::safetensors;
use kernrt::tensor::Tensor;
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

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path = root.join("target/mnist.safetensors");
    let test_path = root.join("target/mnist_test.safetensors");
    if !model_path.exists() || !test_path.exists() {
        eprintln!(
            "missing fixtures; run: python scripts/train_mnist_mlp.py\n  expected {}",
            model_path.display()
        );
        std::process::exit(1);
    }

    let model = Mlp::from_safetensors(safetensors::load(&model_path)?)?;
    let test = safetensors::load(&test_path)?;
    let input = &test["input"];
    let labels = &test["label"];
    let reference = &test["logits"];

    let samples = input.shape()[0];
    let features = input.shape()[1];
    let classes = reference.shape()[1];
    println!("model loaded; checking {samples} samples of {features} features against MLX");

    // Run each sample as a single-row forward, the edge-inference path, and
    // compare against the batched MLX reference.
    let mut ours_correct = 0usize;
    let mut reference_correct = 0usize;
    let mut max_logit_diff = 0.0f32;
    for i in 0..samples {
        let x = Tensor::from_vec(
            input.data()[i * features..(i + 1) * features].to_vec(),
            &[1, features],
        )?;
        let logits = model.forward(&x)?;
        let want = &reference.data()[i * classes..(i + 1) * classes];
        for (got, expected) in logits.data().iter().zip(want) {
            max_logit_diff = max_logit_diff.max((got - expected).abs());
        }
        let label = labels.data()[i] as usize;
        ours_correct += usize::from(argmax(logits.data()) == label);
        reference_correct += usize::from(argmax(want) == label);
    }
    println!("max |logit| difference vs MLX reference : {max_logit_diff:.3e}");
    println!(
        "accuracy  kernrt {:.2}%  reference {:.2}%",
        100.0 * ours_correct as f64 / samples as f64,
        100.0 * reference_correct as f64 / samples as f64,
    );

    // Latency: a warm single-sample forward.
    let x = Tensor::from_vec(input.data()[..features].to_vec(), &[1, features])?;
    let iterations = 2000;
    for _ in 0..100 {
        black_box(model.forward(black_box(&x))?);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(model.forward(black_box(&x))?);
    }
    let per_inference = start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations);
    println!("forward: {per_inference:.2} us/inference");
    Ok(())
}
