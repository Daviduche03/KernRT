//! The backend seam, proven two ways.
//!
//! Every test here builds one tiny synthetic model, hands the same weights to
//! two backends, and requires them to agree. No fixture is needed, so these run
//! on a fresh clone — which matters, because a seam that only works on the
//! machine that has the weights is not a seam.
//!
//! The two properties being pinned:
//!
//! * the tuned NEON/Q8_0 path and the scalar reference agree, and
//! * batched prefill agrees with stepping the decode loop, on *both* backends.
//!   The second is the one that catches a backend that quietly special-cases
//!   the batch dimension.

use kernrt::arch::llm::llama::{Config, Llama};
use kernrt::backends::ModelWeights;
use kernrt::formats::llama2c::LayerWeights;
use kernrt::ops::Matrix;
use kernrt::tensor::Tensor;

/// A small model with every shape distinct enough to catch a transposition:
/// `kv_dim < q_dim`, a non-square MLP, and dimensions aligned so the quantized
/// storages are constructible (`in_dim` a multiple of 32, `out_dim` of 4).
fn config() -> Config {
    Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 32,
        seq_len: 64,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_interleaved: false,
        head_dim: 8,
    }
}

/// Deterministic pseudo-random fill in `[-0.5, 0.5)`.
fn fill(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
    Matrix::from_f32(Tensor::from_vec(fill(rows * cols, seed), &[rows, cols]).unwrap())
}

fn vector(len: usize, seed: u64) -> Tensor {
    Tensor::from_vec(fill(len, seed), &[len]).unwrap()
}

fn weights(config: &Config) -> ModelWeights {
    weights_at(config, "f32")
}

/// The same logical model in a chosen storage, so a test can walk every path
/// the CPU backend has: f32, row-major Q8_0 and the four-row interleaved
/// Q8_0x4.
fn weights_at(config: &Config, storage: &str) -> ModelWeights {
    let cast = |m: Matrix| match storage {
        "bf16" => Matrix::from_bf16_bits(m.to_bf16_bits(), m.out_dim(), m.in_dim()),
        "q8_0" => Matrix::from_f32_q8_0(m.to_f32_tensor()).expect("in_dim is 32-aligned"),
        "q8_0x4" => Matrix::from_f32_q8_0x4(m.to_f32_tensor()).expect("out_dim is 4-aligned"),
        _ => m,
    };
    let (dim, hidden, vocab) = (config.dim, config.hidden_dim, config.vocab_size);
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let layers = (0..config.n_layers)
        .map(|l| {
            let s = 100 * (l as u64 + 1);
            LayerWeights {
                rms_att: vector(dim, s + 1),
                wq: cast(matrix(q_dim, dim, s + 2)),
                wk: cast(matrix(kv_dim, dim, s + 3)),
                wv: cast(matrix(kv_dim, dim, s + 4)),
                wo: cast(matrix(dim, q_dim, s + 5)),
                rms_ffn: vector(dim, s + 6),
                w1: cast(matrix(hidden, dim, s + 7)),
                w2: cast(matrix(dim, hidden, s + 8)),
                w3: cast(matrix(hidden, dim, s + 9)),
                // Exercise the Qwen3 per-head norm path on both backends.
                rms_q: Some(vector(config.head_dim, s + 10)),
                rms_k: Some(vector(config.head_dim, s + 11)),
            }
        })
        .collect();
    ModelWeights {
        token_embedding: cast(matrix(vocab, dim, 1)),
        layers,
        rms_final: vector(dim, 2),
        // Untied, so the classifier is a distinct matrix rather than a view of
        // the embedding.
        wcls: Some(cast(matrix(vocab, dim, 3))),
    }
}

fn model(name: &str) -> Llama {
    let config = config();
    let weights = weights(&config);
    let backend = kernrt::backends::open(name).unwrap();
    Llama::with_backend(config, weights, backend)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit widths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The tuned path and the scalar oracle must agree on a plain decode, token by
/// token, because attention reads the cache each step and an error compounds.
#[test]
fn backends_agree_on_decode() {
    let tokens: Vec<usize> = vec![3, 17, 0, 29, 8, 8, 21];
    let mut cpu = model("cpu");
    let mut reference = model("reference");

    for (pos, &token) in tokens.iter().enumerate() {
        let a = cpu.forward(token, pos).to_vec();
        let b = reference.forward(token, pos).to_vec();
        let diff = max_diff(&a, &b);
        assert!(
            diff <= 1e-3,
            "backends diverge at position {pos}: max |diff| {diff}"
        );
        assert_eq!(
            argmax(&a),
            argmax(&b),
            "backends pick different tokens at position {pos}"
        );
    }
}

/// The batched prefill must match the decode loop on *both* backends. On the
/// tuned path that compares a blocked GEMM against a GEMV; on the reference it
/// compares two scalar loops that differ only in traversal order.
#[test]
fn backends_agree_on_batched_prefill() {
    let prompt: Vec<usize> = vec![5, 11, 2, 30, 7, 7, 19, 1, 24];

    for name in ["cpu", "reference"] {
        let mut batched = model(name);
        let got = batched.prefill(&prompt, 0).to_vec();

        let mut sequential = model(name);
        let mut want = Vec::new();
        for (pos, &token) in prompt.iter().enumerate() {
            want = sequential.forward(token, pos).to_vec();
        }

        let diff = max_diff(&got, &want);
        assert!(
            diff <= 1e-3,
            "{name}: prefill disagrees with decode by {diff}"
        );
        assert_eq!(argmax(&got), argmax(&want), "{name}: different token");
    }
}

/// A model built on one backend must be able to hand back the same answers it
/// started with, and the two backends must agree on a shared prefix — the shape
/// the decision model and prefix caching both use.
#[test]
fn backends_agree_on_shared_prefix() {
    let prefix = vec![4, 9, 14];
    let suffixes: [&[usize]; 3] = [&[20, 6], &[12, 12, 12, 12], &[31]];

    let cpu = model("cpu").prefill_shared(&prefix, &suffixes);
    let reference = model("reference").prefill_shared(&prefix, &suffixes);

    assert_eq!(cpu.len(), reference.len());
    for (i, (a, b)) in cpu.iter().zip(&reference).enumerate() {
        let diff = max_diff(a, b);
        assert!(diff <= 1e-3, "leaf {i} differs by {diff}");
    }
}

/// `open` is the switch, so an unknown name must fail loudly rather than
/// silently returning the CPU path.
#[test]
fn unknown_backend_is_an_error() {
    let error = match kernrt::backends::open("cuda") {
        Ok(_) => panic!("cuda should not resolve unless compiled in"),
        Err(error) => error,
    };
    assert_eq!(error.requested, "cuda");
    assert!(error.to_string().contains("available"));
}

/// The backend reports the storage it was handed, so a benchmark table can say
/// which path produced its number.
#[test]
fn storage_is_reported_by_the_backend() {
    let config = config();
    let mut backend = kernrt::backends::open("cpu").unwrap();
    backend.prepare(
        std::sync::Arc::new(weights(&config)),
        kernrt::backends::KvSpec {
            n_layers: config.n_layers,
            kv_dim: config.kv_dim(),
            seq_len: config.seq_len,
        },
    );
    assert_eq!(backend.storage(), "f32");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(
        kernrt::backends::open("reference").unwrap().name(),
        "reference"
    );
}

/// Every exact configuration must reproduce the decode loop: the native Q8_0x4
/// batched kernel, and the f32/bf16 widened path (where widening loses nothing
/// because pretrained weights are not quantized).
///
/// Q8_0 is included through its row-at-a-time fallback, which is what a model
/// too large for an f32 copy takes.
#[test]
fn cpu_prefill_matches_decode_when_exactness_is_promised() {
    use kernrt::backends::cpu::Cpu;

    let prompt: Vec<usize> = vec![5, 11, 2, 30, 7, 7, 19, 1, 24];
    // (storage, f32 copy). Q8_0 with a copy is excluded: `prefill` routes Q8_0
    // through the decode loop rather than the batched path, so the copy is only
    // reachable from `prefill_multi`/`prefill_shared`.
    let cases = [
        ("f32", true),
        ("f32", false),
        ("q8_0", false),
        ("q8_0x4", false),
    ];

    for (storage, f32_copy) in cases {
        let build = || {
            let backend = Cpu::new().with_prefill_f32(f32_copy);
            Llama::with_backend(config(), weights_at(&config(), storage), Box::new(backend))
        };
        let mut batched = build();
        let got = batched.prefill(&prompt, 0).to_vec();

        let mut sequential = build();
        let mut want = Vec::new();
        for (pos, &token) in prompt.iter().enumerate() {
            want = sequential.forward(token, pos).to_vec();
        }

        let diff = max_diff(&got, &want);
        assert!(
            diff <= 1e-3,
            "{storage} (f32 copy {f32_copy}): prefill differs from decode by {diff}"
        );
        assert_eq!(
            argmax(&got),
            argmax(&want),
            "{storage} (f32 copy {f32_copy}): different token"
        );
    }
}

/// The opt-in f32 path is not a different answer, it is a different *precision*.
///
/// It must agree with the scalar reference backend, which computes exactly the
/// same quantities (dequantized weights, exact activations) by a completely
/// independent route. That is what separates "2.1x faster and more accurate"
/// from "2.1x faster and wrong".
#[test]
fn cpu_f32_prefill_agrees_with_the_reference_backend() {
    use kernrt::backends::cpu::Cpu;

    let prompt: Vec<usize> = vec![5, 11, 2, 30, 7, 7, 19, 1, 24];

    let cpu = Llama::with_backend(
        config(),
        weights_at(&config(), "q8_0x4"),
        Box::new(Cpu::new().with_prefill_f32(true)),
    )
    .prefill(&prompt, 0)
    .to_vec();
    let reference = Llama::with_backend(
        config(),
        weights_at(&config(), "q8_0x4"),
        kernrt::backends::open("reference").unwrap(),
    )
    .prefill(&prompt, 0)
    .to_vec();

    let diff = max_diff(&cpu, &reference);
    // Two exact computations of the same product, summed in a different order.
    // Absolute agreement therefore tracks the logit scale, so the check is a
    // scale-free one: a structural error (a transposed or misindexed copy)
    // moves cosine by orders of magnitude more than reassociation can.
    let dot = cpu.iter().zip(&reference).map(|(a, b)| a * b).sum::<f32>();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm(&cpu) * norm(&reference));
    assert!(
        cosine >= 0.999,
        "f32 prefill differs from the scalar oracle: cosine {cosine}, max |diff| {diff}"
    );
    assert_eq!(argmax(&cpu), argmax(&reference), "different token");
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// The GPU backend, when one is compiled in and present.
///
/// Same standard as the others: decode, batched prefill and a shared prefix all
/// have to match the scalar oracle. The GPU is allowed to be slower — it is not
/// allowed to be different.
#[cfg(feature = "metal")]
#[test]
fn metal_matches_the_scalar_oracle() {
    let opened = kernrt::backends::open("metal");
    let Ok(backend) = opened else {
        // Distinguish "this machine has no GPU" from "the shader is broken":
        // a silently skipped test would hide a compile error forever.
        let error = match opened {
            Err(error) => error,
            Ok(_) => unreachable!(),
        };
        assert!(
            error.to_string().contains("no Metal device"),
            "Metal is present but failed to start: {error}"
        );
        eprintln!("skipping: no Metal device available");
        return;
    };
    let build_model = |name: &str| {
        Llama::with_backend(
            config(),
            // Metal stores weights as bf16; compare against the same storage so
            // the check is about the kernels, not about the format.
            weights_at(&config(), "bf16"),
            kernrt::backends::open(name).expect("checked above"),
        )
    };
    drop(backend);

    let prompt: Vec<usize> = vec![5, 11, 2, 30, 7, 7, 19, 1, 24];
    let mut gpu = build_model("metal");
    let mut oracle = build_model("reference");
    let mut want = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        let a = gpu.forward(token, pos).to_vec();
        want = oracle.forward(token, pos).to_vec();
        let diff = max_diff(&a, &want);
        assert!(diff <= 1e-3, "decode differs at {pos} by {diff}");
        assert_eq!(argmax(&a), argmax(&want), "different token at {pos}");
    }

    let got = gpu.prefill(&prompt, 0).to_vec();
    let diff = max_diff(&got, &want);
    assert!(diff <= 1e-3, "prefill differs from decode by {diff}");
    assert_eq!(
        argmax(&got),
        argmax(&want),
        "prefill picked a different token"
    );
}

/// Tiles larger than one threadgroup, in both dimensions.
///
/// The small model above puts every projection inside a single 64x64 tile, so
/// it never exercises the group grid, the tile strides or the boundary clamps.
/// This one does: a 70-row prompt against 256-wide projections.
#[cfg(feature = "metal")]
#[test]
fn metal_handles_multi_tile_shapes() {
    if let Err(error) = kernrt::backends::open("metal") {
        assert!(
            error.to_string().contains("no Metal device"),
            "Metal is present but failed to start: {error}"
        );
        eprintln!("skipping: no Metal device available");
        return;
    }
    let cfg = Config {
        dim: 128,
        hidden_dim: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 128,
        seq_len: 128,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_interleaved: false,
        head_dim: 32,
    };
    let prompt: Vec<usize> = (0..70).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();

    let got = Llama::with_backend(
        cfg,
        weights_at(&cfg, "bf16"),
        kernrt::backends::open("metal").unwrap(),
    )
    .prefill(&prompt, 0)
    .to_vec();
    let want = Llama::with_backend(
        cfg,
        weights_at(&cfg, "bf16"),
        kernrt::backends::open("reference").unwrap(),
    )
    .prefill(&prompt, 0)
    .to_vec();

    let diff = max_diff(&got, &want);
    assert!(diff <= 1e-3, "multi-tile prefill differs by {diff}");
    assert_eq!(
        argmax(&got),
        argmax(&want),
        "multi-tile picked a different token"
    );
}
