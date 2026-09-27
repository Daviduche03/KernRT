# KernRT — status and plan

A hand-written Rust inference runtime for Apple Silicon. No BLAS, no
Accelerate, no MLX: every kernel is ours, and the point is to see how far that
gets against mature libraries.

## What exists

One model family (Llama-2 style: TinyStories, SmolLM2-135M and SmolLM2-360M),
one device (CPU, NEON), one sequence at a time.

| piece | notes |
|---|---|
| `tensor.rs` | rank-2 f32 tensors, contiguous row-major |
| `ops.rs` | matmul (scalar/tiled/packed/NEON), gemv, matvec, Q8_0 kernels, norms, softmax, transpose |
| `llama.rs` | forward pass, GQA, RoPE (both layouts), KV cache |
| `llama2c.rs` | TinyStories checkpoint reader |
| `safetensors.rs` | HF reader, f32/f16/bf16 |
| `tokenizer.rs` | llama2.c BPE |
| `gpt2_tokenizer.rs` | HuggingFace byte-level BPE (SmolLM2) |
| `generate.rs` | greedy and temperature sampling |

Four weight storages behind `Matrix`: f32, bf16, Q8_0, Q8_0x4 (four rows
interleaved for lane-broadcast SDOT).

## Measured, single-thread M1

Decode, 200 tokens of generation:

| engine | storage | tok/s |
|---|---|---|
| llama.cpp | Q8_0 CPU | 298 |
| **KernRT** | **Q8_0** | **201** |
| llama.cpp | Q8_0 Metal | 172 |
| llama.cpp | F16 Metal | 142 |
| MLX | bf16 GPU | 133 |
| **KernRT** | **bf16** | **114** |
| **KernRT** | **f32** | **74** |
| MLX | f32 CPU | 49 |
| MLX | bf16 CPU | 46 |
| llama.cpp | F16 CPU | 18 |

So we already beat everything except llama.cpp's Q8_0 CPU kernel (1.5x), and
llama.cpp's F16 CPU path is degenerate (it widens weights per token).

### Q8_0x4, and how it scales (SmolLM2-135M and 360M)

`Q8_0x4` interleaves four output rows so one lane-broadcast SDOT updates four
row sums from a single activation load, mirroring ggml's `block_q8_0x4`.

Same harness, same metric, both models (single-thread M1):

| storage | 135M | 360M | slowdown | params |
|---|---|---|---|---|
| f32 | 76.3 | 27.5 | 2.77x | 2.67x |
| bf16 | 123.0 | 45.2 | 2.72x | 2.67x |
| Q8_0 row | 245.4 | 89.8 | 2.73x | 2.67x |
| **Q8_0x4** | **307.3** | **106.1** | 2.89x | 2.67x |

Throughput falls in proportion to parameter count on every storage: decode is
bandwidth-bound and stays that way at 2.7x the size, with no overhead creeping
in. The ratios are stable too — bf16 to Q8_0 is 1.99x at both sizes (exactly
the 2x that halving the bytes predicts), and Q8_0 to Q8_0x4 is 1.25x and 1.28x.

Against llama.cpp on matched 200-token generation, interleaved:

| model | KernRT Q8_0x4 | llama.cpp Q8_0 CPU |
|---|---|---|
| 135M | 174-199 | 143-172 |
| 360M | 71-95 (median ~89) | 79-84 (median ~81) |

Both load with no code changes: the config parser and byte-level BPE were
already generic, and the 360M's awkward `dim 960 / 15 heads / 5 KV heads`
shape needed no special-casing.

Prefill (128-token prompt): llama.cpp Q8_0 is 492 tok/s single-thread and
1328 with four threads. We do not have prefill yet; see below.

## The one bottleneck behind every remaining loss

The micro-kernel. Evidence:

* The packed-GEMM panel sweep is flat between 58 and 62 GFLOP/s for every
  combination of `MC`/`KC`/`NC`. Blocking is tuned out; what is left is the
  inner `MR x NR` (8x8) loop, which is plain Rust with `mul_add`.
* Reference points on the same machine: MLX CPU reaches ~440 GFLOP/s, and our
  own kernel hits 60. llama.cpp's prefill advantage is a better inner loop.
* Decode is bandwidth-bound but still short of ceiling: Q8_0 moves 151 MB per
  token in ~5 ms, about 30 GB/s, where the machine streams 46 GB/s. A wider
  inner loop closes some of that too.
* The 1.5x decode gap to llama.cpp was the same story: their
  `gemv_q8_0_4x4` processed four rows per activation load, ours one. Closed
  by `Q8_0x4` below; decode is now ahead of their CPU path on both models.

## Plan, in order

### 1. Micro-kernel: 8x8 to wider register blocking

Raise `MR` to 12 or 16 and `NR` to 12, hand-write the NEON inner loop with
explicit vector accumulators instead of relying on `mul_add` and the
auto-vectorizer, and keep the f32 accumulators in registers across the whole
`k` panel. Target: 200+ GFLOP/s, which would put prefill at llama.cpp's level
and decode comfortably ahead.

Measure with `benches/model.rs` gemv shapes and a prefill-shape sweep, not by
feel. The MR=8 -> 12 experiment in milestone 3 showed no gain for matmul, but
that was before packing; the packed panels change which loop is hot.

### 2. Rebuild prefill on top of it

The working version was measured at 885 ms for 128 tokens (144 tok/s) and
matched step-by-step decode to 4.8e-5 with identical argmax, but it widened
bf16 weights to f32 inside every projection: 211 conversions x 538 MB = 113 GB
of copying per prompt, which dominated everything else.

Rebuild as:

* `PrefillWeights`, built once, holding f32 copies of every weight.
* `prefill(tokens, start)` with `attention_batched` (causal over the batch).
* `gemm_nt_into` wrapping the flat scratch buffers for the GEMM.

Constraint: decode must not regress. Prefill gets its own buffers and does not
touch the decode path.

Then add a batched Q8_0 kernel so quantized models prefill too; for now Q8_0
falls back to repeated `forward`.

### 3. Quantized prefill

The batched GEMM is f32-only. A Q8_0 prefill needs its own blocked kernel
(int8 operand, f32 accumulate), which is also the chance to fix the decode
kernel's row-at-a-time structure with the lane layout that llama.cpp uses.

### 4. Then, for an actual runtime

* mmap weights instead of parsing safetensors through serde_json at every
  start (269 MB per launch today).
* a CLI: load a GGUF, prefill a prompt, stream tokens.
* multi-threaded decode. llama.cpp gains nothing from 4 threads (267 vs 298)
  because it is bandwidth-saturated; we are at 30 of 46 GB/s, so we might.

## Rules that this project has earned the hard way

* **Measure before optimizing.** The bf16 scale-width change looked obviously
  right and was 1.47x slower. The "we are at the bandwidth ceiling" claim came
  from a broken microbenchmark and was wrong twice.
* **Trust within-run A/B, not cross-run absolute numbers.** The machine's load
  varied from 1 to 50 during this work; numbers taken minutes apart are not
  comparable. Interleave, and report the minimum.
* **Small edits, test after each.** A 180-line text splice corrupted
  `llama.rs` and cost the working prefill. The failure mode is silent.
* **Verify against an oracle, never against intuition.** `run.c` for
  TinyStories (character-exact), mlx-lm f32 for SmolLM2 (3.9e-5, cosine
  1.00000000). Q8_0 matches the f32 greedy path for about 60 tokens before a
  near-tie flips, which is the honest accuracy of int8 here.
* **Compare like statistics with like.** Claiming a 30% generation overhead
  came from putting the *minimum* of 300 samples next to the *mean* of 200.
  Median matched median showed no gap at all. The minimum is a noise floor,
  not a typical value; only quote it against another minimum.
* **Greedy divergence is chaos, not accuracy.** bf16 diverged from mlx-lm 4x
  earlier than Q8_0x4 while having the *better* cosine, because whichever
  side wins a near-tie picks an unrelated continuation. Rank fidelity by
  cosine and first-token argmax, never by "tokens until the text differs".

## Known caveats

* Q8_0 is not bit-reproducible against the f32 path. bf16 is (cosine 1.0).
* Prefill exists only for f32/bf16, and is currently reverted and being
  rebuilt.
* No output-level validation against llama.cpp; we use its GGUFs for speed
  comparison only.
* Model fixtures live in `target/` and are gitignored, so a fresh clone has
  harnesses but no weights.
