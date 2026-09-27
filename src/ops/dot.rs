//! Dot-product primitives, one per storage width.
//!
//! These are what the `gemv` kernels in [`super::quant`] are built from. The
//! AArch64 paths are `cfg`-gated to intrinsics with a portable scalar twin, and
//! the two are tested against each other.

use super::quant::Q8_0_BLOCK;

/// Dot product of one Q8_0 weight row with a Q8_0 activation vector. The int8
/// products are summed exactly in integer arithmetic per block, so the only
/// rounding is the quantization itself (and one f32 multiply per block).
///
/// Weight scales arrive as f16 (the ggml/Q8_0 storage width) and are widened as
/// they are consumed; the activation scales are already f32.
#[inline]
pub(crate) fn dot_q8_0(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: `dotprod` implies the SDOT instruction this uses.
            unsafe { dot_q8_0_sdot(a, a_scales, b, b_scales) }
        } else {
            dot_q8_0_scalar(a, a_scales, b, b_scales)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_q8_0_scalar(a, a_scales, b, b_scales)
    }
}

/// Portable Q8_0 dot product. Integer accumulation is exact, so this agrees
/// with the SDOT kernel up to the f32 summation order.
pub(crate) fn dot_q8_0_scalar(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    let mut total = [0.0f32; 4];
    for (block, (&sa, &sb)) in a_scales.iter().zip(b_scales).enumerate() {
        let lo = block * Q8_0_BLOCK;
        let mut acc = 0i32;
        for (&x, &y) in a[lo..lo + Q8_0_BLOCK].iter().zip(&b[lo..lo + Q8_0_BLOCK]) {
            acc += i32::from(x) * i32::from(y);
        }
        total[block & 3] += (sa * sb) * acc as f32;
    }
    total[0] + total[1] + total[2] + total[3]
}

/// One `SDOT`: `acc[i] += a[4i..4i+4] . b[4i..4i+4]`, four int8 products per
/// int32 lane. Written as inline assembly because the `vdotq_s32` intrinsic is
/// still unstable in Rust 1.87 (`stdarch_neon_dotprod`).
///
/// SAFETY: the CPU must implement `dotprod`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot(
    acc: core::arch::aarch64::int32x4_t,
    a: core::arch::aarch64::int8x16_t,
    b: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    let mut acc = acc;
    // SAFETY: `dotprod` is the caller's contract; the operands are 128-bit
    // vectors resident in registers.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// `SDOT` with one 4-byte activation lane broadcast to all four dot lanes.
/// Each 4-byte group of `a` is a different output row, which is the key to the
/// four-row Q8_0 GEMV kernel: one activation lane updates four row sums.
///
/// SAFETY: `dotprod` must be available; `lane` is 0..=3.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn sdot_lane<const LANE: i32>(
    acc: core::arch::aarch64::int32x4_t,
    a: core::arch::aarch64::int8x16_t,
    b: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    let mut acc = acc;
    // SAFETY: the caller guarantees dotprod and the const generic is a valid
    // lane index. The assembler checks that the immediate is in range.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{lane}]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            lane = const LANE,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// NEON Q8_0 dot product using SDOT, which sums four int8 products into each of
/// four int32 lanes — sixteen multiply-adds per instruction. That is what makes
/// int8 worth the bookkeeping: the kernel issues far fewer instructions than the
/// f32 one *and* reads a quarter of the bytes.
///
/// The block scale is applied while the partial sums are still in vector lanes,
/// and the horizontal reduction is deferred to the end of the row. Reducing per
/// block instead costs more than the multiply-adds it computes: the block is
/// only 32 elements, so a `vaddvq` plus a serial float chain per block ends up
/// dominating.
///
/// SAFETY: the caller must guarantee `dotprod` is available, and that `a` and
/// `b` each hold at least `32 * a_scales.len()` values.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q8_0_sdot(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_n_f32, vld1q_s8,
    };

    // Everything below walks raw pointers. Indexing the slices each iteration
    // made LLVM emit a bounds check and re-derive the offsets per block, and the
    // resulting branches cost more than the multiply-adds: the kernel ran at
    // about half this machine's streaming bandwidth with them.
    let mut wp = a.as_ptr();
    let mut xp = b.as_ptr();
    let mut sp = a_scales.as_ptr();
    let mut sq = b_scales.as_ptr();
    let mut blocks = a_scales.len();

    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };

    // Four blocks (128 weight bytes) per trip, each into its own accumulator.
    // Fewer accumulators left both the FMA chain and the load units idle.
    while blocks >= 4 {
        // SAFETY: the caller guarantees `32 * blocks` values in each slice and
        // `blocks` counts the remaining blocks, so these reads are in range.
        let (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7) = unsafe {
            (
                vld1q_s8(wp),
                vld1q_s8(wp.add(16)),
                vld1q_s8(wp.add(32)),
                vld1q_s8(wp.add(48)),
                vld1q_s8(wp.add(64)),
                vld1q_s8(wp.add(80)),
                vld1q_s8(wp.add(96)),
                vld1q_s8(wp.add(112)),
                vld1q_s8(xp),
                vld1q_s8(xp.add(16)),
                vld1q_s8(xp.add(32)),
                vld1q_s8(xp.add(48)),
                vld1q_s8(xp.add(64)),
                vld1q_s8(xp.add(80)),
                vld1q_s8(xp.add(96)),
                vld1q_s8(xp.add(112)),
            )
        };
        // SAFETY: `dotprod` availability is the caller's contract. The two
        // halves of a block are independent SDOTs combined by a vector add,
        // rather than chaining the second through the first's accumulator,
        // which would serialize two multi-cycle operations per block.
        let d0 = unsafe { sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1) };
        let d1 = unsafe { sdot(sdot(vdupq_n_s32(0), w2, x2), w3, x3) };
        let d2 = unsafe { sdot(sdot(vdupq_n_s32(0), w4, x4), w5, x5) };
        let d3 = unsafe { sdot(sdot(vdupq_n_s32(0), w6, x6), w7, x7) };
        // SAFETY: scales are read straight from the caller's slices.
        let (s0, s1, s2, s3) = unsafe {
            (
                *sp * *sq,
                *sp.add(1) * *sq.add(1),
                *sp.add(2) * *sq.add(2),
                *sp.add(3) * *sq.add(3),
            )
        };
        // SAFETY: plain NEON arithmetic on values already in registers.
        unsafe {
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), s0);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(d1), s1);
            acc2 = vfmaq_n_f32(acc2, vcvtq_f32_s32(d2), s2);
            acc3 = vfmaq_n_f32(acc3, vcvtq_f32_s32(d3), s3);
            wp = wp.add(128);
            xp = xp.add(128);
            sp = sp.add(4);
            sq = sq.add(4);
        }
        blocks -= 4;
    }

    let mut acc = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
    while blocks > 0 {
        // SAFETY: as above; at least one whole block remains.
        let (w0, w1, x0, x1) = unsafe {
            (
                vld1q_s8(wp),
                vld1q_s8(wp.add(16)),
                vld1q_s8(xp),
                vld1q_s8(xp.add(16)),
            )
        };
        let d = unsafe { sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1) };
        // SAFETY: as above.
        let s = unsafe { *sp * *sq };
        acc = unsafe { vfmaq_n_f32(acc, vcvtq_f32_s32(d), s) };
        // SAFETY: as above.
        unsafe {
            wp = wp.add(32);
            xp = xp.add(32);
            sp = sp.add(1);
            sq = sq.add(1);
        }
        blocks -= 1;
    }

    // SAFETY: as above.
    unsafe { vaddvq_f32(acc) }
}

/// bfloat16 bit pattern to f32: the value is the top half, so shift it up.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Dot product with a bf16 weight row: half the bytes to load, widened in
/// registers before the multiply-add.
pub(crate) fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_bf16_scalar(a, b)
    }
}

/// Portable bf16 dot product with four independent chains.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn dot_bf16_scalar(a: &[u16], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= a.len() {
        acc[0] = bf16_to_f32(a[i]).mul_add(b[i], acc[0]);
        acc[1] = bf16_to_f32(a[i + 1]).mul_add(b[i + 1], acc[1]);
        acc[2] = bf16_to_f32(a[i + 2]).mul_add(b[i + 2], acc[2]);
        acc[3] = bf16_to_f32(a[i + 3]).mul_add(b[i + 3], acc[3]);
        i += 4;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < a.len() {
        sum = bf16_to_f32(a[i]).mul_add(b[i], sum);
        i += 1;
    }
    sum
}

/// NEON bf16 dot product.
///
/// The weights are stored as u16, so each load fetches 16 bytes = 8 bf16 values
/// and `vshll` widens them left by 16 — bf16 to f32 in a single instruction.
/// Four widen + four fma per 16 elements would be 12 instructions (the f32
/// kernel needs 8), and that extra instruction cost, not memory, is what limits
/// bf16. Loading at 128 bits and splitting with `vget_low`/`vget_high`, which
/// are pure register renames, drops it to 10; eight independent accumulators
/// hide the load and widen latencies behind each other.
///
/// SAFETY: NEON is architecturally mandatory on AArch64, the only target that
/// compiles this. Every load sits inside `i + 16 <= len` or the `i < len` tail.
#[cfg(target_arch = "aarch64")]
fn dot_bf16_neon(a: &[u16], b: &[f32]) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vget_high_u16, vget_low_u16, vld1q_f32,
        vld1q_u16, vreinterpretq_f32_u32, vshll_n_u16,
    };

    // SAFETY: see the function comment.
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        // A bf16 value is the top 16 bits of an f32, so widening a u16 lane left
        // by 16 bit positions *is* the conversion to f32.
        let widen = |lanes: core::arch::aarch64::uint16x4_t| {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(lanes))
        };
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 32 <= len {
            // One 16-byte load covers eight bf16 values, so four loads cover
            // this iteration's 32 elements exactly; the low/high splits are
            // free register renames, not instructions.
            let x0 = vld1q_u16(ap.add(i));
            let x1 = vld1q_u16(ap.add(i + 8));
            let x2 = vld1q_u16(ap.add(i + 16));
            let x3 = vld1q_u16(ap.add(i + 24));
            acc0 = vfmaq_f32(acc0, widen(vget_low_u16(x0)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, widen(vget_high_u16(x0)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, widen(vget_low_u16(x1)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, widen(vget_high_u16(x1)), vld1q_f32(bp.add(i + 12)));
            acc4 = vfmaq_f32(acc4, widen(vget_low_u16(x2)), vld1q_f32(bp.add(i + 16)));
            acc5 = vfmaq_f32(acc5, widen(vget_high_u16(x2)), vld1q_f32(bp.add(i + 20)));
            acc6 = vfmaq_f32(acc6, widen(vget_low_u16(x3)), vld1q_f32(bp.add(i + 24)));
            acc7 = vfmaq_f32(acc7, widen(vget_high_u16(x3)), vld1q_f32(bp.add(i + 28)));
            i += 32;
        }
        let acc45 = vaddq_f32(acc4, acc5);
        let acc67 = vaddq_f32(acc6, acc7);
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        sum += vaddvq_f32(vaddq_f32(acc45, acc67));
        while i < len {
            sum = bf16_to_f32(a[i]).mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

/// Dot product. On AArch64 this dispatches to the NEON kernel; elsewhere to the
/// portable scalar one.
///
/// Auto-vectorization does not help here: a dot product is a *reduction*, and
/// LLVM may not reassociate floating-point additions, so it cannot split one
/// accumulator into several vector lanes on its own. (That is why the matmul
/// vectorized but this did not.) Four independent accumulators are written out
/// by hand to expose the parallelism.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_scalar(a, b)
    }
}

/// Portable dot product with four independent chains.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= a.len() {
        acc[0] = a[i].mul_add(b[i], acc[0]);
        acc[1] = a[i + 1].mul_add(b[i + 1], acc[1]);
        acc[2] = a[i + 2].mul_add(b[i + 2], acc[2]);
        acc[3] = a[i + 3].mul_add(b[i + 3], acc[3]);
        i += 4;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < a.len() {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

/// NEON dot product: four vector accumulators, so sixteen elements retire per
/// iteration with four independent dependency chains.
///
/// Four was chosen by measurement: eight accumulators was no faster, which
/// says the reduction is limited by memory streaming rather than by multiply-add
/// throughput or load latency.
///
/// SAFETY: NEON is architecturally mandatory on AArch64, the only target that
/// compiles this. Every load sits inside `i + 16 <= len` or the `i < len` tail,
/// and AArch64 permits unaligned vector loads from ordinary memory.
#[cfg(target_arch = "aarch64")]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // SAFETY: see the function comment.
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= len {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            i += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < len {
            sum = ap.add(i).read().mul_add(bp.add(i).read(), sum);
            i += 1;
        }
        sum
    }
}
