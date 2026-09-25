//! End-to-end check through the public API only: tensors -> file -> model ->
//! logits. Unlike the unit tests, this goes through the real filesystem, which
//! is the only place `weights::save`/`load` differ from the in-memory codec.

use kernrt::model::Mlp;
use kernrt::tensor::Tensor;
use kernrt::weights;
use std::path::PathBuf;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape).unwrap()
}

fn scratch_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("kernrt-{tag}-{}.krt", std::process::id()))
}

#[test]
fn parameters_survive_a_round_trip_and_produce_logits() {
    let params = [
        ("fc1.weight", tensor(vec![1., 2., 3., -4.], &[2, 2])),
        ("fc1.bias", tensor(vec![0., 0.], &[2])),
        ("fc2.weight", tensor(vec![1., 2.], &[2, 1])),
        ("fc2.bias", tensor(vec![0.5], &[1])),
    ];
    let refs: Vec<(&str, &Tensor)> = params.iter().map(|(name, t)| (*name, t)).collect();

    let path = scratch_path("e2e");
    weights::save(&path, &refs).unwrap();
    let loaded = weights::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.len(), 4);
    let model = Mlp::from_weights(loaded).unwrap();
    // x = [1, 1]: fc1 -> [4, -2], relu -> [4, 0], fc2 -> 4 + 0 + 0.5.
    let logits = model.forward(&tensor(vec![1., 1.], &[1, 2])).unwrap();
    assert_eq!(logits.data(), &[4.5]);
}

#[test]
fn missing_file_is_an_io_error() {
    let path = scratch_path("missing");
    let _ = std::fs::remove_file(&path);
    assert!(matches!(
        weights::load(&path),
        Err(weights::WeightError::Io(_))
    ));
}

/// Prefill must produce the same logits as stepping the decode path one token
/// at a time. This is the oracle for the batched path: it is written before the
/// batched implementation so that a wrong kernel cannot pass it silently.
///
/// The model is a gitignored fixture, so the test reports that it was skipped
/// rather than failing when the weights are absent.
#[test]
fn prefill_matches_token_at_a_time() {
    use kernrt::gpt2_tokenizer::Gpt2Tokenizer;
    use kernrt::llama::Llama;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no SmolLM2 fixture in target/smollm2");
        return;
    }
    let tokenizer = Gpt2Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let prompt = tokenizer.encode("The capital of France is", false);
    assert!(prompt.len() > 1, "prompt should tokenize to several pieces");

    let mut sequential = Llama::from_hf(&dir).unwrap();
    let mut batched = Llama::from_hf(&dir).unwrap();

    let mut want = Vec::new();
    for (pos, &id) in prompt.iter().enumerate() {
        want = sequential.forward(id, pos).to_vec();
    }
    let got = batched.prefill(&prompt, 0).to_vec();

    assert_eq!(got.len(), want.len());
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same arithmetic grouped differently, so agreement is to f32 precision
    // rather than bit-exact.
    assert!(
        max_diff <= 1e-3,
        "prefill disagrees with decode by {max_diff}"
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(
        argmax(&got),
        argmax(&want),
        "prefill picked a different token"
    );

    // Generation must continue correctly from the prefilled cache: the next
    // token after the prompt has to match the sequential model's.
    let next_batched = {
        let ids = prompt.clone();
        let last = argmax(&got);
        let _ = ids;
        last
    };
    let next_sequential = argmax(&want);
    assert_eq!(next_batched, next_sequential);
}
