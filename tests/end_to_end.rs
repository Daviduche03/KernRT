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
