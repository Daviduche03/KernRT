# KernRT — status and plan

A hand-written Rust inference runtime for Apple Silicon. No BLAS, no MLX:
every kernel is ours, and the point is to see how far that gets against mature
libraries.

One deviation, recorded rather than implied: `gemm_nt_into` (the f32/bf16
prefill GEMM) calls Accelerate's `cblas_sgemm` on macOS — commit `5a0e831` — so
those dense projections run on the AMX unit. Decode, the packed kernels and
every quantized path stay hand-written, and the portable kernel remains for
other targets.

## What exists

One model family (Llama-2 style: TinyStories, SmolLM2-135M and SmolLM2-360M),
one device (CPU, NEON), one sequence at a time.

Laid out by role: `ops` (kernels), `formats` (weight readers), `tokenizer`,
`arch` (runnable networks), on top of `tensor`. `arch` is split by family
(`llm`, `vision`); a decoder implements `arch::llm::LanguageModel`, which is
all the `generate` loop needs, so it never names a concrete architecture.

| piece | notes |
|---|---|
| `tensor.rs` | rank-2 f32 tensors, contiguous row-major |
| `ops/` | matmul (scalar/tiled/packed/NEON), gemv, Q8_0 kernels, norms, softmax, transpose |
| `arch/llm/llama.rs` | forward pass, GQA, RoPE (both layouts), KV cache |
| `arch/llm/hf.rs` | HF loader: `config.json` parser, storage policy, tensor extraction |
| `arch/decision.rs` | NanoJev decision model over a Qwen3 backbone |
| `arch/vision/mlp.rs` | dense classifier (MNIST) |
| `arch/vision/yolo26.rs` | YOLO26 ONNX loader, native conv/pool kernels, graph rewrites |
| `arch/vision/yolo26/tests.rs` | the 20 kernel/rewrite tests, out of the model file |
| `arch/vision/bench.rs` | opt-in timers and graph reports (`KERNRT_TIMERS`, `KERNRT_VERBOSE`) |
| `formats/llama2c.rs` | TinyStories checkpoint reader |
| `formats/safetensors.rs` | HF reader, f32/f16/bf16 |
| `formats/weights.rs` | `.krt` tensor bundle reader/writer |
| `tokenizer/byte_pair.rs` | llama2.c BPE |
| `tokenizer/bpe.rs` | HuggingFace byte-level BPE (SmolLM2 and Qwen) |
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

### Model families: Qwen3

The loader now generalizes past Llama/SmolLM2 in two places that Qwen3 needs
and they do not exercise:

* `head_dim` read from `config.json` instead of assumed to be
  `hidden_size / n_heads`. Qwen3-0.6B is `hidden 1024`, 16 heads, but a head
  width of 128, so `q_proj` is `[2048, 1024]` and the attention output is 2048
  wide before `o_proj` maps it back to 1024. `Config::q_dim()` carries this;
  the query and attention-output buffers are sized from it.
* per-head `q_norm`/`k_norm` (`qk_norm_per_head`), applied to each head's slice
  before RoPE. Loaded as optional tensors, so Llama/SmolLM2 keep their geometry
  and arithmetic untouched.

The tokenizer also needed the newer `["a", "b"]` merge-pair spelling that Qwen
uses alongside GPT-2's `"a b"` strings. Our Qwen3 tokenization is byte-exact
against the official `Qwen/Qwen3-0.6B` tokenizer (`[785, 6722, 315, 9625, 374]`
for "The capital of France is").

Verified on Qwen3-0.6B: decode argmax matches mlx-lm (`12095` = " Paris"),
cosine `0.99992`; batched prefill vs token-at-a-time is cosine `1.0000`, max
`|diff| 2.8e-5`. Single-thread M1, 128-token prefill / one-token decode
(`tg1`), freshly built:

| storage | prefill tok/s | decode tok/s |
|---|---|---|
| bf16 | 299 | 25.7 |
| q8_0 | 51 | 56.5 |
| q8_0x4 | 159 | 61.3 |

### NanoJev decision model

NanoJev (`C-Tianyu/NanoJev`) is a Jev-style decision model: a Qwen3-0.6B
backbone plus a small head, scoring a set of candidate answers in one forward
pass with **zero decoding**. `src/arch/decision.rs` implements it against the
same `Llama` backbone.

The prompt layout matches `predict_toy_decisions.py` exactly: a shared prefix
`State:\n...\nQuestion type: ...\nQuestion:\n...\n`, then one leaf per candidate
`prefix + "Candidate:\n{text}\nDecision:" + [eos]`. Each leaf runs through the
backbone; the **EOS-position** final-normed hidden is the verdict. The head is
`LayerNorm -> scalar` plus, for `choice`, `set_project([h, ln k]) -> 4-head
self-attention -> set_output`, and `tanh` around the residual. Booleans use
`[0, z]`; scores use the scalar path alone.

The backbone needed two loader changes: a nested `rope_parameters.rope_theta`
fallback (`backbone_config/config.json` has no top-level `rope_theta`), and a
`backbone.` tensor prefix that `Llama::from_raw_tensors` accepts after
renaming. `Llama::prefill_hidden` exposes the backbone's last hidden row, which
the logits path computes but discards.

**The tokenizer was silently wrong for Qwen.** The GPT-2 pre-tokenizer was
hardcoded, but Qwen declares a different Split regex in `tokenizer.json` where
e.g. `":\n"` is one word (one token, id 510) rather than two (`:` + `\n`). The
tokenizer now reads `pre_tokenizer` from the file and dispatches to
`split_words` (GPT-2, SmolLM2) or `split_words_qwen`. Validated ASCII-for-ASCII
against the reference `tokenizers` crate on the rule-sensitive cases
(punctuation+newline, single-digit runs, `<br/>`-style whitespace). Non-ASCII
letters (`\p{L}` beyond ASCII) remain a known gap in both splitters.

Verified on the README example vs a CPU fp32 PyTorch oracle that reuses the
reference `DecisionModel` and `prepare_examples`: probabilities match to
`1.8e-7` (`[0.3371, 0.3438, 0.2858, 0.0333]`), backbone hidden to `3e-5`.

All candidate leaves are now scored in **one batched forward**
(`Llama::prefill_multi`): each sequence gets a disjoint key/value cache range
and its RoPE positions restart at zero, so a single pass scores many
independent prompts without cross-talk while every weight is read once. This is
the reference's own shape, and it is also the substrate batched serving needs.

The leaves share everything but their last few tokens, so `decide` now splits
the prompt and runs **`Llama::prefill_shared(prefix, suffixes)`**: the prefix
is embedded, projected and cached once into slots `0..prefix.len()`, and every
suffix row reads it back through a second disjoint window
(`PrefillRow::att_shared`) instead of spelling out its own copy. Scores are
accumulated shared-range first — the order a leaf that repeated the prefix
would have produced — so the output matches `prefill_multi` over the
concatenated leaves, pinned by
`prefill_shared_matches_repeating_the_prefix`, and the oracle diff is
unchanged at **1.79e-7**. Attention gained the window for free:
`slot`/`att_start`/`att_shared` moved into `AttentionInput`, so the function
lost arguments rather than gaining one. For the README example the ~34-token
leaves (≈24-token shared prefix + ~10-token suffix) drop from 136 rows to ~64,
and the cache holds the prefix once instead of four times.

End-to-end decision latency, single-thread M1, f32 storage (host heavily
loaded, load 6-13, so treat these as upper bounds):

| workload | latency |
|---|---|
| one 34-token leaf prefill | 197 ms |
| 4 leaves, sequential prefills | ~1000 ms |
| 4 leaves, one batched forward | ~375 ms |
| 4 leaves, batched + shared prefix | **271-468 ms (best 271)** |
| one 136-token single sequence | 513 ms |

Batching is a **~2.7x** win and beats even a single long sequence, because the
four short leaves share one weight pass. q8_0x4 is now *slower* than f32
(~720 ms): its SDOT kernel loses to Accelerate's SGEMM once the rows are
batched, so the quantized path is a decode-only win.

**Prefix sharing is a further 1.3-2.5x**: 6 interleaved A/B pairs against the
pre-sharing binary (alternating which binary ran first, load 7-13) — shared
271-468 ms, unshared 449-679 ms, and every pair favoured sharing. The best
sample is **271 ms**, ~1.8x off Jev's ~150 ms claim (was 2.5x). What is left
is the raw 0.6B-param weight pass (2.4 GB f32, ~30 ms at bandwidth) plus
per-row norm/attention/GEMM overhead, i.e. PLAN items 1-2 (prefill
micro-kernel, quantized prefill) rather than anything decision-specific.

The low absolute decode versus SmolLM2 is the vocab (151936, a 3x wider tied
classifier matvec) plus the wider q/o under `head_dim 128`.

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

Prefill (128-token prompt) on CPU, interleaved, load 4-9,
`VECLIB_MAXIMUM_THREADS=1`, `-ngl 0`. Both sides take **20 samples and both
quote the fastest**: ours from `examples/prefill_rate`, llama.cpp from
`llama-bench -p 128 -r 20`. Two corrections to the table that used to be here.
It ran llama.cpp at four threads against our one; and it quoted our side from a
loaded machine, where the same harness is 2-6x slower — at 135M, load 4 gives
1202 tok/s, load 10 gives 618, and 1199 once the load clears.

Same storage, same thread count — our `q8_0x4` loop against llama.cpp's Q8_0:

| model | ours, 1 thread | llama.cpp, 1 thread | ratio |
|---|---|---|---|
| 135M | 607-609 | 494-495 | 1.23x ahead |
| 360M | 237-245 | 187-194 | 1.26x ahead |
| 1.7B | 56-57 | 43-45 | 1.27x ahead |

The same three sizes with llama.cpp on four threads: 1308-1370, 493-527, 119 —
**2.6-2.8x its own single-thread rate**. Decode does not scale like that, which
is why thread count distorts this table and not the decode table above.

The batched path has a matched llama.cpp counterpart only at 135M, where an
F16 GGUF is on disk (there is no 360M or 1.7B F16 to compare against):

| path | tok/s | GFLOP/s |
|---|---|---|
| ours, bf16 batched (Accelerate) | 1232-1239 | 331-333 |
| llama.cpp F16, 1 thread | 603-605 | 162-163 |
| llama.cpp F16, 4 threads | 665-839 | 179-226 |

F16 gains only 1.4x from four threads where Q8_0 gains 2.7x, which is what
`VECLIB_MAXIMUM_THREADS=1` holding the BLAS GEMM to one thread looks like while
the unaccelerated ops still scale. On that like-for-like AMX comparison we are
about 2x ahead, and the mechanism is probably not a better inner loop:
`PrefillWeights` widens every projection once and calls `sgemm` on the widened
copy, where a GGUF conversion happens per call.

In GFLOP/s (2 x params per token, this file's convention), fastest of 20:

| | 135M | 360M | 1.7B |
|---|---|---|---|
| ours, bf16 batched | 331-333 | 354-393 | not run (see below) |
| ours, q8_0x4 loop | 163-164 | 171-177 | 192-195 |
| llama.cpp, 1 thread, Q8_0 | 133 | 135-140 | 147-154 |
| llama.cpp, 1 thread, F16 | 162-163 | - | - |
| llama.cpp, 4 threads, Q8_0 | 352-369 | 357-381 | 407 |

So prefill is **ahead** of llama.cpp single-thread at every size on both
storages — 1.23-1.27x on Q8_0, 2.05x on F16 — and the batched path lands within
a few percent of their four-thread rate while using one thread. What this table
used to call a 4.1-4.5x deficit was two measurement errors compounding, four
threads against one and a loaded machine, not a gap in the kernel. The old
llama.cpp column happened to be close (1207 against 1308-1370) because its
four-thread rate is insensitive to moderate load; its column for *us* was not.
The fast path is gated by `prefill_matches_token_at_a_time` — max `|diff| <=
1e-3` against the token-at-a-time loop, matching argmax, and a decode step
*after* prefill that has to agree too — so the speed is measured, not assumed.

A note on method, because it cost a rewrite: `llama-bench -o json` reports
`samples_ts`, and taking `min` of it selects the **slowest** sample. The first
draft of this table did exactly that on one side and used our fastest on the
other, which is the "compare like statistics with like" failure the rules above
warn about, caught only because the spread was implausibly wide.

An earlier version of this file reported "492 single-thread, 1328 four-thread"
without naming a model, and the two were crossed: 492 is the 360M four-thread
number and 1328 is the 135M one (four-thread, fastest of 20 samples across two
runs: 493-527 and 1308-1370).

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

That target is stale, and the re-measured table above is why: 326-371 is
llama.cpp at *four* threads, while a single thread reaches 133-163 GFLOP/s and
our own batched path already sits at 331-393. What is left of item 1 is a
question, not a number: the packed GEMM still measures 58-62 GFLOP/s in
isolation (above), but prefill no longer runs on it — it runs on Accelerate for
f32/bf16 and on SDOT for `q8_0x4`. So item 1 is worth doing only if the goal is
dropping Accelerate, which is a project-ideal question; as a *performance* item
it is no longer where the loss is.

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

## YOLO26 detection path

The vision side now accepts a fixed-shape, NMS-free YOLO26 ONNX export and
keeps the output contract explicit: normalized NCHW input and detection rows
`x1,y1,x2,y2,score,class`. `src/arch/vision/yolo26.rs` owns preprocessing,
execution, and output decoding; `examples/yolo26_server.rs` exposes `/health`
and a raw-RGB `POST /infer` endpoint.

The reproducible comparison is `scripts/benchmark_yolo26.py`. It uses the same
fixed input, ten warmups, iteration count, and one CPU thread for raw Torch,
Ultralytics, and KernRT (including an explicit vecLib thread cap on macOS). On
the M1 test machine (YOLO26n, 640x640), the pure-Rust `tract` path measures
about 6.1 img/s (165 ms), versus 21.0 img/s raw Torch and 20.0 img/s
Ultralytics end-to-end in the latest 50-iteration run. The absolute numbers
vary with host load; the within-run optimization A/B is the meaningful result.
The native convolution, depthwise, and im2col paths below are now on by
default and move KernRT to about 19 img/s, which is the number to compare
against Torch from here on.

KernRT specializes the two fixed 2x nearest-neighbor `Resize` nodes into a
contiguous NCHW copy kernel. That moved the matched CPU path from 4.8--4.9 to
5.5--5.8 img/s (about 13--18% faster), with byte-identical detection JSON on
the same input. The rewrite is native CPU-only and does not change the
single-threaded execution contract.

Run the portable server with:

```text
cargo run --release --example yolo26_server -- yolo26n.onnx --serve 127.0.0.1:8080
```

Run the comparison with the local benchmark environment (`torch`,
`ultralytics`, and `numpy` installed):

```text
python scripts/benchmark_yolo26.py --weights yolo26n.pt --iterations 100
```

The next target was the convolution path. An 8-second `sample` of a
400-iteration run put 73% of samples inside tract's `OptMatMul`, but the AMX
arithmetic kernels were under 1% of them: the time went into `LazyIm2col::
panel_bytes`, `OutputStore::set_from_tile`, and the loop load/store around
them, which is CPU-side packing plus the strided NCHW write of every output
tile. `KernConv` replaces that with an im2col panel laid out as `[k, hw]` —
the same order as the OIHW weight — so both GEMM operands are NoTrans,
Accelerate reads both contiguously, and the result lands directly in NCHW with
no strided store. Padding is applied during the gather, so tract's separate
`Pad` node never exists. The rewrite runs after `prop_consts` and before
`declutter`, because `declutter` both folds the weights and decomposes `Conv`
into `Pad + LazyIm2col + OptMatMul` — which is why the earlier pointwise scan
ran after declutter and always found zero matches.

Interleaved A/B at load 4--5, 20 iterations per side: 168.81 ms (5.9 img/s)
with `KERNRT_NATIVE_CONV=0` versus 73.35 ms (13.6 img/s) with the rewrite on,
2.30x. It is on by default and `KERNRT_NATIVE_CONV=0` restores the tract path.
94 of the 102 convolutions move over; the other eight are rejected by
`KernConv::from_conv` on `conv.group != 1` and are the subject of the depthwise
rewrite below.

Two follow-up rewrites take what that paragraph called the next candidates.

`KERNRT_FAST_IM2COL` (on by default, `=0` for the A/B) tiles the gather over
output rows so a panel fits in L2 instead of streaming through it — a full
`hw` panel for the first layer is 44 MB against a 12 MB L2 — and skips the
gather entirely for the 1x1 stride-1 layers, where the panel would be a
byte-for-byte copy of the input and the input can go straight to the GEMM. 55
of the 102 convolutions qualify. Interleaved A/B, min of three rounds: 71.05
ms with `KERNRT_FAST_IM2COL=0` versus 64.68 ms with it on, 1.10x.

`KernDepthWise` (on by default, `KERNRT_NATIVE_DEPTHWISE=0` for the A/B) takes
the eight grouped convolutions. tract leaves them to `inner_loop_generic`, a
scalar zone visitor that yields one `(kernel index, input offset)` pair per
tap for every output element: 13.3% of the profile for 13 MFLOP. A depthwise
has no reduction axis, so the native kernel adds each tap's
`weight * <input row>` to the output row — with stride 1 across x that is one
contiguous shifted axpy, and the row it accumulates into never leaves L1
because the next output row overlaps it. Interleaved A/B at load 5, 300
iterations per side: 73.96 -> 68.81 ms and 65.32 -> 60.99 ms, about 1.07x in
both rounds; the node's profiler share drops from 13.3% to 4.8%. Grouping is
checked twice — `Conv::group` must not be 1 and must equal both channel
counts — because a grouped convolution that mixes channels would otherwise
produce plausible, wrong numbers.

`KERNRT_NATIVE_SILU` (on by default, `=0` for the A/B) fuses the graph's
`Sigmoid` then `Mul` pair into a single `KernSiLU` node. Those two nodes appear
87 times in the optimized graph — one per conv block, plus the decoupled head —
and tract gives them separate passes: `Sigmoid` writes a fresh tensor, `Mul`
(plus a `deep_clone` of the inputs for `OptBinUnicast`) writes another. The
rewrite finds a `Mul` whose non-constant operand is produced by a `Sigmoid` on
the other operand, only neutralises a `Sigmoid` when *all* its consumers are in
the fused set, and swaps `Sigmoid -> Identity`, `Mul -> KernSiLU`. The kernel is
one vector pass with a degree-7 Horner exponential over the same
`n = round(x*log2(e))` range reduction tract uses, plus the tail as scalars; the
output tensor is allocated with `Tensor::uninitialized_dt` and written in place
rather than built from a `Vec` and copied. Four 8-second `sample` runs (two per
side, three seconds of warm-up so model load is outside the window) put the
`Sigmoid`+`Mul` block plus their clones at 1208 samples on average with the
fusion off and 1150 with it on, while `_platform_memmove` falls from 552 to 406
— the fused node re-absorbs most of what the removed `deep_clone` and second
tensor freed up. Total top-of-stack samples fall 4172 -> 4137, and nine
interleaved CPU-time pairs agree: about 1% faster. That is real but smaller than
the block's 26% profiler share suggests, because tract's own NEON sigmoid was
already good and the fusion mostly removes copies rather than arithmetic. The
flag exists so the A/B is repeatable, not because it moves the headline number.

The next target needed a full call-tree walk, because the sorted top-of-stack
table only prints frames with five or more samples and that filter hid the GEMM
entirely: `cblas_sgemm` is 17% of the run, sitting under unsymbolised `???`
frames in `libBLAS.dylib`. Subtracting each node's children from its own count
so that the leaves sum to the root puts the run into buckets, and two
instruments added for this give the rates a profile cannot: `KERNRT_VERBOSE`
prints each native conv's panel bytes and FLOPs, and `KERNRT_TIMERS` accumulates
nanoseconds around the gather and the GEMM. Together they say the gather moves
155 MiB of panel per inference in 18.0 ms at 8.6 GiB/s, while the GEMM does
5.33 GFLOP in 10.2 ms; Accelerate measures 790-1300 GFLOP/s on this machine, so
the arithmetic was already cheap and the panel was the thing to fix.

Two changes to `KernConv` followed. `im2col` used to test padding, split the row
into zero/copy/zero, and recompute `lead`/`take`/`window` for every output row
even though they depend only on `kx`. It now derives the `oy` range whose input
row is inside the frame once per `(ky, kx)` and clamps it to the tile, so the
padding rows are two `fill` calls instead of one per row, the three quantities
leave the row loop, and an unpadded row is a bare copy — and when the layer is
stride 1 with `in_w == out_w`, which is every pad-1 3x3, consecutive output rows
read consecutive input rows and both strides match, so the whole tile collapses
into a single `copy_from_slice`. `KernConv::eval` then stopped building a zeroed
`Vec` and copying it into a Tensor: it takes an uninitialised `Tensor` and
writes through `Tensor::as_slice_mut`, which is sound for the same reason as
`KernSiLU` — the GEMM writes with `beta = 0` over every element before the bias
pass reads any of it. Interleaved A/B at load 5, three pairs of 100-inference
runs with the GEMM as the load control (9.59 -> 9.34 ms, flat):
59.97/61.15/61.46 ms against 52.43/52.02/55.05, so 60.0 -> 52.0 ms at the
minimum, 1.15x. The gather alone goes 18.01 -> 12.85 ms, 1.40x, 8.6 -> 12.1
GiB/s. What is left of the gather is panel traffic itself, so the next step is
not a better gather but not writing the panel at all.

Correctness is gated by `--check N`, which runs four fixed inputs through both
paths. Every native rewrite has to be off for the baseline and on for the
native run — a flag left on for both sides compares a path against itself — so
the harness sets all five on-by-default flags (`NATIVE_CONV`, `FAST_IM2COL`,
`NATIVE_DEPTHWISE`, `NATIVE_UPSAMPLE`, `NATIVE_SILU`) explicitly. Row order
legitimately differs: the graph picks its own top 300 rows and
tail scores tie inside f32 noise, so one row at score ~1.2e-5 can fall on
either side of the cutoff. The harness therefore pairs rows by box and class
rather than by position. Across the four inputs: box drift at most 4.9e-4 px,
score drift at most 6.6e-7, at most one tail row unmatched per side, and no
detection above score 1e-3 moved. `KernConv::run` is unit-tested against a
direct convolution for eight shapes (stride 1 and 2, padding 0 and 1, 1x1, no
bias, batch 2, non-square input), and `KernDepthWise::run` against a direct
depthwise for eight more (the same spread, plus a single-channel plane and a
width of six so the NEON tail is exercised). `silu` is checked against a
scalar oracle over 4099 random values plus the extremes, against `f32::exp` at
15 fixed points away from the range-reduction boundary, over every tail length
from 0 to 11, and for saturation at `+-f32::MAX`; the rewrite itself has tests
for the replaced case and for a `Sigmoid` shared with an unfused consumer,
which must be left alone.

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
* **A top-k output is a set, not a sequence.** Comparing two runs' 300 output
  rows index by index reported 634-pixel differences when both paths held the
  same detections in a different order. Pair rows by content (or compare
  sorted multisets) before calling a rewrite wrong, then gate on whether
  anything above the noise-level score moved.

## Known caveats

* Q8_0 is not bit-reproducible against the f32 path. bf16 is (cosine 1.0).
* Prefill exists only for f32/bf16, and is currently reverted and being
  rebuilt.
* No output-level validation against llama.cpp; we use its GGUFs for speed
  comparison only.
* Model fixtures live in `target/` and are gitignored, so a fresh clone has
  harnesses but no weights.
