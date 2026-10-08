//! Serve and benchmark an Ultralytics YOLO26 NMS-free ONNX export.
//!
//! ```text
//! # Export once with Ultralytics:
//! yolo export model=yolo26n.pt format=onnx imgsz=640 nms=False simplify=True
//! # Benchmark the graph (model-only, fixed 640x640 input):
//! cargo run --release --example yolo26_server -- yolo26n.onnx --bench 100
//! # Verify the native convolution rewrite against the tract path:
//! cargo run --release --example yolo26_server -- yolo26n.onnx --check 3
//! ```
//!
//! This binary is deliberately free of the `serve` feature: `--bench` and
//! `--check` measure the graph and nothing else. HTTP serving lives in
//! `examples/serve.rs` behind `--features serve`.

use kernrt::arch::vision::yolo26::{Detection, Yolo26};
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

/// Pixel tolerance when pairing detections produced by the two paths.
const BOX_TOLERANCE: f32 = 1e-2;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(
        args.next()
            .ok_or("usage: yolo26_server MODEL.onnx [--bench N|--check N]")?,
    );
    let mut mode = None;
    while let Some(arg) = args.next() {
        mode = Some((arg, args.next()));
    }

    match mode {
        Some((flag, Some(value))) if flag == "--bench" => {
            let model = Yolo26::from_onnx(&model_path, 640, 640)?;
            benchmark(&model, value.parse()?)
        }
        Some((flag, Some(value))) if flag == "--check" => check(&model_path, value.parse()?),
        _ => Err("expected --bench N or --check N (serving moved to examples/serve.rs)".into()),
    }
}

/// Run the graph twice — with and without the native convolution rewrite — and
/// compare raw outputs and detections. Speed is worthless if the rewrite
/// changes what the model computes, so this gates every kernel change.
///
/// The comparison is deliberately not index-aligned: the graph picks its own
/// top 300 rows, and scores in the tail tie within f32 noise, so the two runs
/// list the same detections in a different order and can even keep a different
/// row where the top-300 cutoff falls inside a tie. Rows are paired by box and
/// class; the leftover rows must all sit at noise-level scores; and the score
/// distribution is compared as a multiset.
fn check(model_path: &std::path::Path, iterations: usize) -> Result<(), Box<dyn Error>> {
    // SAFETY: the harness is single-threaded and no other thread reads the
    // environment during this process, so setting the flags here is fine.
    //
    // Every native rewrite has to be off for the baseline and on for the
    // native run, or the check compares a path against itself.
    unsafe { std::env::set_var("KERNRT_NATIVE_CONV", "0") };
    unsafe { std::env::set_var("KERNRT_NATIVE_DEPTHWISE", "0") };
    unsafe { std::env::set_var("KERNRT_NATIVE_SILU", "0") };
    unsafe { std::env::set_var("KERNRT_NATIVE_UPSAMPLE", "0") };
    unsafe { std::env::set_var("KERNRT_NATIVE_MAXPOOL", "0") };
    unsafe { std::env::set_var("KERNRT_NATIVE_REDUCE", "0") };
    let baseline_model = Yolo26::from_onnx(model_path, 640, 640)?;

    unsafe { std::env::set_var("KERNRT_NATIVE_CONV", "1") };
    unsafe { std::env::set_var("KERNRT_NATIVE_DEPTHWISE", "1") };
    unsafe { std::env::set_var("KERNRT_NATIVE_SILU", "1") };
    unsafe { std::env::set_var("KERNRT_NATIVE_UPSAMPLE", "1") };
    unsafe { std::env::set_var("KERNRT_NATIVE_MAXPOOL", "1") };
    unsafe { std::env::set_var("KERNRT_NATIVE_REDUCE", "1") };
    let native_model = Yolo26::from_onnx(model_path, 640, 640)?;

    let samples = 3 * baseline_model.input_width() * baseline_model.input_height();
    let seeds = [
        0x6a09e667f3bcc909u64,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
    ];
    for (input_index, seed) in seeds.iter().enumerate() {
        let input = deterministic_input_seeded(samples, *seed);
        let want = baseline_model.infer_nchw(&input)?;
        let want_detections = baseline_model.detections(&want);
        let mut got = native_model.infer_nchw(&input)?;
        for _ in 0..iterations {
            got = native_model.infer_nchw(&input)?;
        }
        let got_detections = native_model.detections(&got);

        if want.len() != got.len() {
            return Err(format!("output length differs: {} vs {}", want.len(), got.len()).into());
        }
        let mut max_abs = 0.0f32;
        let mut above = [0usize; 4];
        for (w, g) in want.iter().zip(&got) {
            let abs = (w - g).abs();
            max_abs = max_abs.max(abs);
            for (slot, threshold) in [1e-4, 1e-2, 1.0, 10.0].iter().enumerate() {
                if abs > *threshold {
                    above[slot] += 1;
                }
            }
        }
        let mut sorted_want_raw = want.clone();
        let mut sorted_got_raw = got.clone();
        sorted_want_raw.sort_by(|a, b| a.total_cmp(b));
        sorted_got_raw.sort_by(|a, b| a.total_cmp(b));
        let sorted_max = sorted_want_raw
            .iter()
            .zip(&sorted_got_raw)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        println!(
            "kernrt check[{input_index}]: {} outputs: index-aligned max diff {max_abs:.3e} \
             (>1e-4 {} >1e-2 {} >1 {} >10 {}), sorted max diff {sorted_max:.3e}",
            want.len(),
            above[0],
            above[1],
            above[2],
            above[3]
        );
        println!(
            "kernrt check[{input_index}]: baseline detections={}, native detections={}",
            want_detections.len(),
            got_detections.len()
        );
        let mut score_bad = 0;
        let mut box_bad = 0;
        let mut worst_score = 0.0f32;
        for (w, g) in want_detections.iter().zip(&got_detections) {
            let dscore = (w.score - g.score).abs();
            let dbox = box_distance(w, g);
            if dscore > 1e-3 {
                score_bad += 1;
            }
            if dbox > 1e-2 {
                box_bad += 1;
                worst_score = worst_score.max(w.score.max(g.score));
            }
        }
        println!(
            "kernrt check[{input_index}]: index-aligned: score diff >1e-3 {score_bad}, \
             box diff >1e-2 {box_bad}, highest score among box mismatches {worst_score:.3e}"
        );

        // The graph picks its own top 300 rows, and scores in the low-confidence
        // tail tie within f32 noise, so the two runs can order the same rows
        // differently — and can even keep a different row where the cutoff falls
        // inside a tie. Pair the rows up by box and class first, then account for
        // the leftovers separately.
        let mut used = vec![false; got_detections.len()];
        let mut matched_want = vec![false; want_detections.len()];
        let mut good_pairs = 0usize;
        let mut good_box = 0.0f32;
        let mut good_score = 0.0f32;
        for (w_index, w) in want_detections.iter().enumerate() {
            let mut best: Option<(f32, usize)> = None;
            for (g_index, g) in got_detections.iter().enumerate() {
                if used[g_index] || g.class_id != w.class_id {
                    continue;
                }
                let distance = box_distance(w, g);
                if distance <= BOX_TOLERANCE && best.is_none_or(|(previous, _)| distance < previous)
                {
                    best = Some((distance, g_index));
                }
            }
            if let Some((distance, g_index)) = best {
                used[g_index] = true;
                matched_want[w_index] = true;
                good_pairs += 1;
                good_box = good_box.max(distance);
                good_score = good_score.max((w.score - got_detections[g_index].score).abs());
            }
        }

        // Rows the other path did not keep: the top-300 cutoff sitting inside a
        // tie is fine, but a detection anybody would act on must not go missing.
        let tail_want_max = want_detections
            .iter()
            .zip(&matched_want)
            .filter(|(_, matched)| !**matched)
            .map(|(detection, _)| detection.score)
            .fold(0.0f32, f32::max);
        let tail_got_max = got_detections
            .iter()
            .zip(&used)
            .filter(|(_, used)| !**used)
            .map(|(detection, _)| detection.score)
            .fold(0.0f32, f32::max);
        let tail_want_count = matched_want.iter().filter(|matched| !**matched).count();
        let tail_got_count = used.iter().filter(|unused| !**unused).count();
        for (detection, matched) in want_detections.iter().zip(&matched_want) {
            if !*matched {
                println!(
                    "kernrt check[{input_index}]: baseline-only row {detection:?} \
                     (native kept {tail_got_count} rows baseline did not)"
                );
            }
        }

        let mut sorted_want_scores: Vec<f32> = want_detections.iter().map(|d| d.score).collect();
        let mut sorted_got_scores: Vec<f32> = got_detections.iter().map(|d| d.score).collect();
        sorted_want_scores.sort_by(|a, b| a.total_cmp(b));
        sorted_got_scores.sort_by(|a, b| a.total_cmp(b));
        let score_multiset_max = sorted_want_scores
            .iter()
            .zip(&sorted_got_scores)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);

        println!(
            "kernrt check[{input_index}]: {good_pairs}/{} paired: box diff max {good_box:.3e}, \
             score diff max {good_score:.3e}; tail-only rows {tail_want_count} baseline / \
             {tail_got_count} native (max score {tail_want_max:.3e} / {tail_got_max:.3e}), \
             score multiset max diff {score_multiset_max:.3e}",
            want_detections.len()
        );

        if want_detections.len() != got_detections.len() {
            return Err(format!(
                "input {input_index}: detection count {} vs {}",
                want_detections.len(),
                got_detections.len()
            )
            .into());
        }
        if good_score > 1e-3 {
            return Err(format!(
                "input {input_index}: paired detections differ in score by {good_score:.3e}"
            )
            .into());
        }
        if want_detections.len() - good_pairs > 5 {
            return Err(format!(
                "input {input_index}: {} detections do not pair up",
                want_detections.len() - good_pairs
            )
            .into());
        }
        if tail_want_max > 1e-3 || tail_got_max > 1e-3 {
            return Err(format!(
                "input {input_index}: a detection with score {tail_want_max:.3e} / \
                 {tail_got_max:.3e} was dropped or invented"
            )
            .into());
        }
        if score_multiset_max > 1e-4 {
            return Err(format!(
                "input {input_index}: score multiset drifts by {score_multiset_max:.3e}"
            )
            .into());
        }
        if sorted_max > 1e-2 && want_detections.len() == good_pairs {
            return Err(
                format!("input {input_index}: raw outputs drift by {sorted_max:.3e}").into(),
            );
        }
    }
    println!(
        "kernrt check: outputs and detections match to tolerance across {} inputs",
        seeds.len()
    );
    Ok(())
}

fn box_distance(w: &Detection, g: &Detection) -> f32 {
    (w.x1 - g.x1)
        .abs()
        .max((w.y1 - g.y1).abs())
        .max((w.x2 - g.x2).abs())
        .max((w.y2 - g.y2).abs())
}

fn benchmark(model: &Yolo26, iterations: usize) -> Result<(), Box<dyn Error>> {
    let input = deterministic_input(3 * model.input_width() * model.input_height());
    for _ in 0..10 {
        black_box(model.infer_nchw(&input)?);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(model.infer_nchw(&input)?);
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "kernrt yolo26: {:.2} ms/inference ({:.1} img/s), iterations={iterations}",
        seconds * 1e3 / iterations as f64,
        iterations as f64 / seconds
    );
    Ok(())
}

fn deterministic_input(n: usize) -> Vec<f32> {
    deterministic_input_seeded(n, 0x6a09e667f3bcc909)
}

fn deterministic_input_seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        })
        .collect()
}
