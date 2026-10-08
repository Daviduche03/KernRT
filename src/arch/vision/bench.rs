//! Opt-in instrumentation for the vision kernels.
//!
//! `KERNRT_TIMERS=1` accumulates per-kernel nanoseconds and prints the
//! im2col gather's MiB/s and the GEMM's GFLOP/s every 50 inferences;
//! `KERNRT_VERBOSE` prints the graph reports. Nothing here runs unless an
//! environment flag asks for it, so it lives apart from the kernels.

use std::sync::atomic::{AtomicU64, Ordering};

use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

pub(crate) fn report_convs(model: &TypedModel, label: &str) {
    use tract_onnx::tract_core::ops::cnn::conv::Conv;
    let mut tally: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for node in &model.nodes {
        let Some(conv) = node.op_as::<Conv>() else {
            continue;
        };
        let nonconst = usize::from(
            node.inputs.len() < 3
                || !model
                    .outlet_fact(node.inputs[1])
                    .is_ok_and(|f| f.konst.is_some())
                || !model
                    .outlet_fact(node.inputs[2])
                    .is_ok_and(|f| f.konst.is_some()),
        );
        let key = format!(
            "k{:?} g{} s{} d{}",
            conv.pool_spec.kernel_shape,
            conv.group,
            conv.pool_spec.stride(0),
            conv.pool_spec.dilation(0),
        );
        let entry = tally.entry(key).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += nonconst;
    }
    let mut rows: Vec<_> = tally.iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    eprintln!("kernrt: {label}: conv breakdown (count, with_nonconst_weights):");
    for (key, (count, nonconst)) in rows {
        eprintln!("kernrt:   {count:4} x {key:32} nonconst={nonconst}");
    }
}

/// Opt-in per-kernel timers, enabled by `KERNRT_TIMERS=1`.
///
/// The profiler gives shares but not rates, and the two rates decide the fix:
/// the gather's bytes-per-second says whether it is at the machine's bandwidth
/// or just badly written, and the GEMM's FLOP-per-second says whether the
/// arithmetic still has headroom. Two `Instant::now()` calls per tile is a few
/// hundred nanoseconds per inference, so this is safe to leave in.
pub(crate) static TIMERS_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var_os("KERNRT_TIMERS").is_some_and(|value| value != "0")
});
pub(crate) static IM2COL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static IM2COL_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static GEMM_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GEMM_FLOPS: AtomicU64 = AtomicU64::new(0);
pub(crate) static BIAS_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static BIAS_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static SILU_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SILU_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static MAXPOOL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static REDUCE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static RUN_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INFERENCES: AtomicU64 = AtomicU64::new(0);

/// Start a timer segment, or `None` when `KERNRT_TIMERS` is off.
pub(crate) fn timer_start() -> Option<std::time::Instant> {
    TIMERS_ON.then(std::time::Instant::now)
}

/// Close a segment against one of the atomic accumulators.
pub(crate) fn timer_stop(started: Option<std::time::Instant>, ns: &AtomicU64) {
    if let Some(started) = started {
        ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Called once per inference. Every `report_every` inferences, print the
/// rates so a short `--bench` run is enough to read them.
pub(crate) fn timer_record_inference(report_every: u64) {
    if !*TIMERS_ON {
        return;
    }
    let done = INFERENCES.fetch_add(1, Ordering::Relaxed) + 1;
    if done % report_every != 0 {
        return;
    }
    let secs = done as f64;
    let run_ms = RUN_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let im2col_ms = IM2COL_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let gemm_ms = GEMM_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let bias_ms = BIAS_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let silu_ms = SILU_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let maxpool_ms = MAXPOOL_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let reduce_ms = REDUCE_NS.load(Ordering::Relaxed) as f64 / secs / 1e6;
    let silu_mib = SILU_BYTES.load(Ordering::Relaxed) as f64 / secs / (1024.0 * 1024.0);
    let mib = IM2COL_BYTES.load(Ordering::Relaxed) as f64 / secs / (1024.0 * 1024.0);
    let gflop = GEMM_FLOPS.load(Ordering::Relaxed) as f64 / secs / 1e9;
    eprintln!(
        "kernrt timers: {done} inf, run {run_ms:.2} ms | im2col {im2col_ms:.2} ms ({:.0}%) {mib:.1} MiB/{:.2} MiB/s | gemm {gemm_ms:.2} ms ({:.0}%) {gflop:.2} GFLOP/{:.0} GFLOPS | silu {silu_ms:.2} ms ({:.0}%) {silu_mib:.1} MiB/{:.1} MiB/s | maxpool {maxpool_ms:.2} ms ({:.0}%) | bias {bias_ms:.2} ms ({:.0}%) | reduce {reduce_ms:.2} ms ({:.0}%)",
        100.0 * im2col_ms / run_ms,
        mib / (im2col_ms / 1e3),
        100.0 * gemm_ms / run_ms,
        gflop / (gemm_ms / 1e3),
        100.0 * silu_ms / run_ms,
        silu_mib / (silu_ms / 1e3),
        100.0 * maxpool_ms / run_ms,
        100.0 * bias_ms / run_ms,
        100.0 * reduce_ms / run_ms,
    );
}

/// The matmuls that survived `rewrite_native_conv` — every one of them is a
/// layer the native convolution path does not cover, and the profile puts
/// tract's own packing and strided store behind them, so they are worth
/// naming rather than only counting.
pub(crate) fn report_stragglers(model: &TypedModel) {
    for node in &model.nodes {
        let name = node.op.name();
        if name != "OptMatMul" && name != "OptMatMulPack" && name != "LazyIm2col" {
            continue;
        }
        let facts = node
            .inputs
            .iter()
            .filter_map(|input| model.outlet_fact(*input).ok())
            .map(|fact| format!("{:?}", fact.shape))
            .collect::<Vec<_>>()
            .join(" ");
        let outputs = node
            .outputs
            .iter()
            .map(|output| format!("{:?}", output.fact.shape))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("kernrt: leftover {name} #{} {facts} -> {outputs}", node.id);
    }
}

pub(crate) fn report_nodes(model: &TypedModel, label: &str) {
    let mut tally: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for node in &model.nodes {
        let mut elems = 1usize;
        if let Some(fact) = node.outputs.first().map(|outlet| &outlet.fact) {
            for dim in fact.shape.iter() {
                elems = elems.saturating_mul(dim.to_usize().unwrap_or(1));
            }
        }
        let entry = tally.entry(node.op.name().to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += elems;
    }
    let mut rows: Vec<(&String, &(usize, usize))> = tally.iter().collect();
    rows.sort_by_key(|(_, (count, _))| *count);
    eprintln!("kernrt: {label}: {} nodes", model.nodes.len());
    for (op, (count, elems)) in rows {
        eprintln!("kernrt:   {count:4} x {op:44} out_elems={elems}");
    }
}

/// Input and output facts, plus the op's own parameters, for the structural
/// nodes that make up the detection head. `report_nodes` only counts them, and
/// a native rewrite of any of them needs the axis and the layout, not just the
/// element count.
pub(crate) fn report_head_ops(model: &TypedModel) {
    const WANTED: &[&str] = &[
        "Reduce<Max>",
        "Softmax",
        "MoveAxis",
        "Concat",
        "Slice",
        "Topk",
        "GatherElements",
        "Sigmoid",
    ];
    for node in &model.nodes {
        let name = node.op.name();
        if !WANTED.contains(&name.as_ref()) {
            continue;
        }
        let facts = node
            .inputs
            .iter()
            .filter_map(|input| model.outlet_fact(*input).ok())
            .map(|fact| format!("{:?}", fact.shape))
            .collect::<Vec<_>>()
            .join(" ");
        let outputs = node
            .outputs
            .iter()
            .map(|output| format!("{:?}", output.fact.shape))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "kernrt: head {name} #{} {facts} -> {outputs} {op:?}",
            node.id,
            op = node.op
        );
    }
}
