# KernRT

A hand-written inference runtime for Apple silicon, in Rust.

Most of the kernels are ours: NEON and `SDOT` matmuls, Q8_0 quantization, the
attention and norm passes. The point is to have a number attached to the
question "how far does hand-written code get against mature libraries?" rather
than an opinion. `PLAN.md` is where that argument is made, with measurements
and the negative results that shaped it.

Three compute backends:

* **cpu** — the tuned path. Hand-written kernels, four weight storages
  (f32, bf16, Q8_0, Q8_0x4), Accelerate's `sgemm` for the f32 prefill GEMM.
* **metal** — device-resident weights, activations and KV cache, with MSL
  kernels. macOS only, behind the `metal` feature.
* **reference** — scalar f32 loops with no SIMD and no packing. Slow on
  purpose: it is the oracle the other two are tested against.

## Numbers

M1, bf16 weights, CPU single-threaded. Reproduce with `examples/backends`.

Prefill, 128 tokens (tok/s):

| model | CPU | Metal | |
|---|---|---|---|
| SmolLM2-135M | 1212 | 1144 | 0.94× |
| SmolLM2-360M | 536 | 533 | 0.99× |
| **SmolLM2-1.7B** | **11** | **146** | **13.3×** |

Decode, one token (tok/s):

| model | CPU | Metal | |
|---|---|---|---|
| SmolLM2-135M | 75.5 | 47.5 | 0.63× |
| SmolLM2-360M | 26.2 | 24.8 | 0.95× |
| SmolLM2-1.7B | 11.0 | 8.9 | 0.81× |

Metal wins prefill at 1.7B because the CPU path builds a 6.4 GB f32 copy of the
weights there and swaps; Metal holds them once, at bf16. It still loses decode:
the CPU is already at the memory-bandwidth wall, and the GPU pays a dispatch and
a synchronisation per projection. Output is byte-identical on all three models.

Against llama.cpp on the same machine, single thread, CPU: decode 1.04–1.13×
ahead and prefill 1.23–1.27× ahead. `PLAN.md` has the full tables, including
where those claims were wrong before they were measured properly.

## Quick start

Fixtures are not in the repository — see [Model fixtures](#model-fixtures).

```sh
cargo build --release

# Every backend on one prompt; the reference one is the slow oracle.
cargo run --release --example backends

# The same, with the GPU backend compiled in (macOS).
cargo run --release --features metal --example backends

# Force one backend, and pick a model.
KERNRT_BACKEND=metal KERNRT_MODEL=target/smollm2_1_7b \
  cargo run --release --features metal --example backends
```

## Backends

Selection is `KERNRT_BACKEND=cpu|metal|reference`, an environment variable read
once at load. A serve manifest can pin one per model:

```json
{ "id": "smollm2", "kind": "generate", "path": "target/smollm2", "backend": "metal" }
```

`src/backends/mod.rs` defines the surface: every op takes `Buf` handles, and
`read()` is the only barrier. That is what lets a GPU backend encode a whole
step and synchronise once instead of once per projection.

Adding a backend is one file plus an arm in `backends::open`. The test in
`tests/backends.rs` builds a synthetic model with no fixture and requires every
backend to reproduce the scalar one on decode, batched prefill and a shared
prefix.

## Layout

```
src/tensor.rs        rank-2 f32 tensors, contiguous row-major
src/ops/             matmul, gemv, Q8_0 kernels, norms, softmax
src/formats/         llama2c, safetensors, .krt readers
src/tokenizer/       llama2.c BPE and HuggingFace byte-level BPE
src/backends/        cpu, metal, and the scalar reference oracle
src/arch/            runnable networks: llm (llama, hf), vision, decision
src/generate.rs      greedy and temperature sampling
src/serve/           optional axum layer (feature = "serve")
```

## Models

| model | format | loader |
|---|---|---|
| TinyStories | llama2.c | `arch::llm::Llama::load` |
| SmolLM2-135M / 360M / 1.7B | HF safetensors | `Llama::from_hf` |
| Qwen3-0.6B | HF safetensors | `Llama::from_hf` |
| NanoJev decision head | HF + head | `arch::decision` |
| MNIST MLP | `.krt` / safetensors | `arch::vision::mlp` |
| YOLO26 | ONNX (via `tract`) | `arch::vision::yolo26` |

Storage is chosen at load: `KERNRT_Q8_0=1`, `KERNRT_Q8_0X4=1`,
`KERNRT_ALL_F32=1`, and `KERNRT_PREFILL_F32=1` for the wider batched prefill
path. Defaults are bf16 for bf16 checkpoints and f32 otherwise.

## Model fixtures

Weights are gitignored, so a fresh clone builds and passes its tests but cannot
reproduce the numbers above or run the model examples. The fixture layout the
examples expect is `target/<model>/` containing `config.json`,
`model.safetensors` and `tokenizer.json`; the YOLO26 export is a top-level
`yolo26n.onnx`. `tests/end_to_end.rs` reports a skip rather than failing when a
fixture is absent.

`cargo test` runs without weights. `cargo test --features serve` runs the whole
suite, `--features metal,serve` adds the GPU backend.

## Serving

```sh
cargo run --release --features serve --example serve -- models.json
```

One process, one port, two API surfaces: OpenAI chat/completions for text
models, and a typed JSON API for what does not fit it (`POST /infer` for
detections, `POST /decide` for decision heads). See `src/serve/mod.rs`.

## What this is not

* **Not a serving system.** One sequence at a time: there is no continuous
  batching, no paged KV cache, and requests serialise. That is the largest
  remaining piece of the plan, not a tuning detail.
* **Not optimized for contested hardware.** The wins come from choosing a
  target almost nobody tunes for. On an H200 against cuBLAS the same approach
  would not transfer.
* **No GGUF loader.** llama.cpp's GGUFs are used for speed comparison only.

`PLAN.md` is the measurement log: what was built, what it cost, and the rules
the project earned the hard way.

## License

No license file yet. Until one is added, the default is all rights reserved.
