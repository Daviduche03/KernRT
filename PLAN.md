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
The 1.7B extends this: bf16 decode is 11.0 against the 135M's 136, which is
12.4x for 12.6x the parameters.

Against llama.cpp on matched 200-token generation, interleaved:

| model | KernRT Q8_0x4 | llama.cpp Q8_0 CPU |
|---|---|---|
| 135M | 174-199 | 143-172 |
| 360M | 71-95 (median ~89) | 79-84 (median ~81) |

And on matched single-token decode (llama-bench `tg1`, which is what
`examples/decode_rate.rs` measures), all three models, run back to back:

| model | KernRT Q8_0x4 | llama.cpp Q8_0 CPU | ratio |
|---|---|---|---|
| 135M | 355 | 332 | 1.07x |
| 360M | 140 | 135 | 1.04x |
| 1.7B | 31.4 | 27.8 | 1.13x |

Decode is ahead at every size and the margin widens slightly with it, which is
the expected split: decode is a bandwidth problem and `Q8_0x4` moves half the
bytes a Q8_0 row does. Note the two tables do not agree on the size of the
margin — the 200-token numbers put 135M at 174-199 against 143-172, and `tg1`
puts it at 355 against 332. Decode rate depends on context length, so a
measurement of one is not a measurement of the other.

All three load with no code changes: the config parser and byte-level BPE were
already generic, and nothing needed special-casing for the 360M's
`dim 960 / 15 heads / 5 KV heads` or the 1.7B's `2048 / 32 / 32`. The 1.7B is
the first with `kv_mul == 1`, so its attention reads a distinct key per head
where the other two share one across three and five heads respectively.

Prefill (128-token prompt) on CPU, 4 threads, interleaved. llama.cpp is Q8_0
GGUF; ours is the best storage for the job, which is `q8_0x4` on the decode
loop — the batched path only serves f32 and bf16, and there it is slower than
the quantized decode loop (see below).

| model | ours, best | ours, batched | llama.cpp Q8_0 | ratio |
|---|---|---|---|---|
| 135M | 274 (q8_0x4) | 218 (bf16) | 1207 | 4.4x |
| 360M | 114 (q8_0x4) | 77 (bf16) | 512 | 4.5x |
| 1.7B | 30 (q8_0x4) | 15 (bf16) | 124 | 4.1x |

In GFLOP/s, which is what the kernel work should be judged on:

| | 135M | 360M | 1.7B |
|---|---|---|---|
| llama.cpp | 326 | 371 | 424 |
| ours, best (q8_0x4 loop) | 74 | 83 | 103 |
| ours, batched (f32 GEMM) | 59 | 56 | 52 |

llama.cpp's CPU prefill is faster in GFLOP/s at every size and gains with it.
So does our quantized gemv, which is the interesting half: it is flat per token
but the per-token cost amortizes better as the matrices grow, so the loop we
routed the quantized storages *away* from is the thing that scales. The f32
batched GEMM goes the other way, flat to slightly down.

The ratio to llama.cpp is therefore roughly constant at 4.1-4.5x, and the
kernel is the whole story — not model size. An earlier draft of this table said
8.3x at 1.7B, which was measuring the batched path's collapse (below) rather
than the gap we actually have to close.

An earlier version of this file reported "492 single-thread, 1328 four-thread"
without naming a model, and the two were crossed: 492 is the 360M four-thread
number and 1328 is the 135M one (re-measured here as 512 and 1207).

At 1.7B the f32-copy design stops paying for itself, and it is not the kernel
that says so. `PrefillWeights` is 6.4 GB of f32 against 3.4 GB of bf16 weights
and a 3.2 GB KV cache, on a 17 GB machine: the one-time copy took 18.5 s
against the whole 128-token prefill's 8.6 s, with swap growing past 8 GB. The
copy ends up larger than the thing it accelerates, and batching lands at 15
tok/s against the 30 tok/s of simply looping the decode path. Anything at this
size needs the batched path to read weights in their stored format instead of
converting them, which is the same conclusion item 1 reaches from the other
direction.

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
`k` panel.

Target: llama.cpp's CPU prefill sits at 326-371 GFLOP/s, so "200+" was the
wrong bar — it would still leave prefill about 2x behind. Aim at 350, which
would put prefill at their level and decode comfortably ahead.

Measure with `benches/model.rs` gemv shapes and a prefill-shape sweep, not by
feel. The MR=8 -> 12 experiment in milestone 3 showed no gain for matmul, but
that was before packing; the packed panels change which loop is hot.

### 2. Rebuild prefill on top of it — DONE, independent of item 1

Built as planned: `PrefillWeights` holding one f32 copy of every projection
built lazily on first use (~170 ms for 135M, once per session), flat `[n_tok,
width]` scratch, and `gemm_nt_into` over the residual stream. 128 tokens for
135M went from 1030 ms (124 tok/s) to 587 ms (218 tok/s).

One deviation: there is no `attention_batched`. The scores are causal, so row
`i` reads `start..=start+i` and the rows never interact; the existing per-row
`attention` is called in a loop with the score buffer reused. Batching it would
have computed and discarded the masked half.

The gate that matters, measured before writing any of it by timing the decode
loop at every storage — batching is not automatically a win:

| storage | decode loop | batched | chosen |
|---|---|---|---|
| f32 | 87 tok/s | 218 | batched |
| bf16 | 124 tok/s | 218 | batched |
| q8_0 | 232 tok/s | 220 | loop |
| q8_0x4 | 303 tok/s | 274 | loop |

`q8_0x4`'s gemv decompresses weights as it reads them and reaches ~82 GFLOP/s
against the batched f32 GEMM's ~55, so routing it through the batched path
would have been a 10% regression reported as progress. This is the thing item 1
changes: a GEMM that beats 82 GFLOP/s puts every storage on the batched path,
including the one that is fastest today.

Also measured: prefill throughput in GFLOP/s is flat across model size (59 at
135M, 56 at 360M), so the batched path currently buys nothing from scale.

Later: a batched Q8_0 kernel so the quantized storages prefill too. Until then
they keep the loop, which for `q8_0x4` is still the fastest prefill we have.

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
