//! Unit tests for the YOLO26 kernels and the graph rewrites.

use super::*;

/// Deterministic pseudo-random values in `-1..1`.
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Direct convolution, written the obvious way, as the oracle for `run`.
fn reference(input: &[f32], batch: usize, in_h: usize, in_w: usize, conv: &KernConv) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * conv.output_channels * conv.output_h * conv.output_w];
    for b in 0..batch {
        for oc in 0..conv.output_channels {
            for oy in 0..conv.output_h {
                for ox in 0..conv.output_w {
                    let mut acc = conv.bias.get(oc).copied().unwrap_or(0.0);
                    for ic in 0..conv.input_channels {
                        for ky in 0..conv.kernel_h {
                            for kx in 0..conv.kernel_w {
                                let iy = (oy * conv.stride_h + ky) as isize - conv.pad_h as isize;
                                let ix = (ox * conv.stride_w + kx) as isize - conv.pad_w as isize;
                                if iy < 0 || ix < 0 || iy >= in_h as isize || ix >= in_w as isize {
                                    continue;
                                }
                                let input_index =
                                    ((b * conv.input_channels + ic) * in_h + iy as usize) * in_w
                                        + ix as usize;
                                let weight_index =
                                    ((oc * conv.input_channels + ic) * conv.kernel_h + ky)
                                        * conv.kernel_w
                                        + kx;
                                acc += input[input_index] * conv.weights[weight_index];
                            }
                        }
                    }
                    let output_index =
                        ((b * conv.output_channels + oc) * conv.output_h + oy) * conv.output_w + ox;
                    out[output_index] = acc;
                }
            }
        }
    }
    out
}

fn assert_close_enough(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: output length");
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        let tolerance = 1e-4 + 1e-4 * w.abs();
        assert!(
            (g - w).abs() <= tolerance,
            "{label}: index {index} got {g}, want {w}"
        );
    }
}

fn assert_matches_reference(
    label: &str,
    batch: usize,
    in_h: usize,
    in_w: usize,
    conv: &KernConv,
    input: &[f32],
) {
    let got = conv.run(input, batch, in_h, in_w);
    let want = reference(input, batch, in_h, in_w, conv);
    assert_close_enough(label, &got, &want);
}

/// One convolution configuration to exercise.
struct Case {
    batch: usize,
    cin: usize,
    cout: usize,
    in_h: usize,
    in_w: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    with_bias: bool,
}

impl Case {
    /// Build the convolution and its NCHW input.
    fn build(&self) -> (KernConv, Vec<f32>) {
        let input = values(self.batch * self.cin * self.in_h * self.in_w, 0x1234);
        let output_h = (self.in_h + 2 * self.pad - self.kernel) / self.stride + 1;
        let output_w = (self.in_w + 2 * self.pad - self.kernel) / self.stride + 1;
        let reduction = self.cin * self.kernel * self.kernel;
        let conv = KernConv {
            weights: values(self.cout * self.cin * self.kernel * self.kernel, 0xabcd),
            bias: if self.with_bias {
                values(self.cout, 0xbeef)
            } else {
                Vec::new()
            },
            kernel_h: self.kernel,
            kernel_w: self.kernel,
            stride_h: self.stride,
            stride_w: self.stride,
            pad_h: self.pad,
            pad_w: self.pad,
            dilation_h: 1,
            dilation_w: 1,
            input_channels: self.cin,
            output_channels: self.cout,
            output_h,
            output_w,
            tile_rows: tile_rows_for(true, reduction, output_w, output_h),
            fast_im2col: true,
            apply_silu: false,
        };
        (conv, input)
    }
}

const CASES: [Case; 8] = [
    Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 2,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 3,
        cout: 6,
        in_h: 7,
        in_w: 7,
        kernel: 3,
        stride: 2,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 1,
        stride: 1,
        pad: 0,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 0,
        with_bias: true,
    },
    Case {
        batch: 2,
        cin: 3,
        cout: 5,
        in_h: 9,
        in_w: 6,
        kernel: 3,
        stride: 2,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: false,
    },
    Case {
        batch: 1,
        cin: 6,
        cout: 6,
        in_h: 16,
        in_w: 16,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    },
];

#[test]
fn kern_conv_matches_reference_across_shapes() {
    for (index, case) in CASES.iter().enumerate() {
        let (conv, input) = case.build();
        assert_matches_reference(
            &format!("case {index}"),
            case.batch,
            case.in_h,
            case.in_w,
            &conv,
            &input,
        );
    }
}

/// Every tile height a shape can have must agree with the untiled
/// reference. The panel width and the output column offset both change
/// with the tile height, so this is where an off-by-one shows up.
#[test]
fn kern_conv_matches_reference_at_every_tile_height() {
    for (index, case) in CASES.iter().enumerate() {
        let (conv, input) = case.build();
        let want = reference(&input, case.batch, case.in_h, case.in_w, &conv);
        for rows in 1..=conv.output_h {
            let got = conv.run_tiled(&input, case.batch, case.in_h, case.in_w, rows);
            assert_close_enough(&format!("case {index}, {rows} row(s)"), &got, &want);
        }
    }
}

/// A tile height the shape cannot divide evenly must still cover the last
/// partial band exactly once.
#[test]
fn kern_conv_tile_height_clamps_to_the_shape() {
    let case = Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    };
    let (conv, input) = case.build();
    assert_eq!(conv.output_h, 8);
    let want = reference(&input, 1, 8, 8, &conv);
    for rows in [conv.output_h + 3, usize::MAX] {
        let got = conv.run_tiled(&input, 1, 8, 8, rows);
        assert_close_enough(&format!("clamped from {rows}"), &got, &want);
    }
}

/// A 1x1 stride-1 pad-0 convolution has to be recognised as a gather that
/// copies its input, and the numeric tests alone would pass either way.
#[test]
fn kern_conv_skips_the_gather_when_the_panel_is_the_input() {
    for (index, case) in CASES.iter().enumerate() {
        let (mut conv, input) = case.build();
        let copy = case.kernel == 1 && case.stride == 1 && case.pad == 0;
        assert_eq!(conv.gather_is_a_copy(), copy, "case {index}");

        // Both im2col strategies have to agree with the oracle: the flag
        // picks between gathering and passing the input through, and the
        // unit tests would pass either way if one of them were wrong.
        let reduction = conv.input_channels * conv.kernel_h * conv.kernel_w;
        for fast in [true, false] {
            conv.fast_im2col = fast;
            conv.tile_rows = tile_rows_for(fast, reduction, conv.output_w, conv.output_h);
            let got = conv.run(&input, case.batch, case.in_h, case.in_w);
            let want = reference(&input, case.batch, case.in_h, case.in_w, &conv);
            assert_close_enough(&format!("case {index}, fast_im2col={fast}"), &got, &want);
        }
    }
}

#[test]
fn kern_conv_shapes_follow_the_onnx_formula() {
    // `from_conv` reads the output shape out of tract's `computed_padding`;
    // keep the test fixture honest against the same ONNX arithmetic.
    for case in CASES.iter() {
        let (conv, _) = case.build();
        assert_eq!(
            conv.output_h,
            (case.in_h + 2 * case.pad - case.kernel) / case.stride + 1,
            "output height formula"
        );
        assert_eq!(
            conv.output_w,
            (case.in_w + 2 * case.pad - case.kernel) / case.stride + 1
        );
        assert_eq!(
            conv.weights.len(),
            case.cout * case.cin * case.kernel * case.kernel
        );
        assert_eq!(conv.bias.len(), if case.with_bias { case.cout } else { 0 });
    }
}

impl Case {
    /// The depthwise convolution for a case with one kernel per channel.
    fn build_depthwise(&self) -> (KernDepthWise, Vec<f32>) {
        assert_eq!(
            self.cin, self.cout,
            "a depthwise case needs one input channel per output channel"
        );
        let input = values(self.batch * self.cin * self.in_h * self.in_w, 0x1234);
        let output_h = (self.in_h + 2 * self.pad - self.kernel) / self.stride + 1;
        let output_w = (self.in_w + 2 * self.pad - self.kernel) / self.stride + 1;
        let dw = KernDepthWise {
            weights: values(self.cin * self.kernel * self.kernel, 0xabcd),
            bias: if self.with_bias {
                values(self.cin, 0xbeef)
            } else {
                Vec::new()
            },
            kernel_h: self.kernel,
            kernel_w: self.kernel,
            stride_h: self.stride,
            stride_w: self.stride,
            pad_h: self.pad,
            pad_w: self.pad,
            dilation_h: 1,
            dilation_w: 1,
            channels: self.cin,
            output_h,
            output_w,
        };
        (dw, input)
    }
}

/// Channel counts that do not match cannot be depthwise, whatever the
/// grouping says: mixing channels is what a grouped convolution is for.
#[test]
fn depthwise_needs_one_output_channel_per_input_channel() {
    let params = |input_channels, output_channels| ConvParams {
        weights: vec![0.0; output_channels * input_channels * 9],
        bias: Vec::new(),
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        input_channels,
        output_channels,
        output_h: 8,
        output_w: 8,
    };
    assert!(params(4, 4).into_depthwise().is_some());
    assert!(params(4, 8).into_depthwise().is_none());
}

/// Widths that are not a multiple of four, an odd height, a batch of two,
/// and a kernel with no padding all take different paths through the
/// bounds handling.
const DEPTHWISE_CASES: [Case; 8] = [
    Case {
        batch: 1,
        cin: 4,
        cout: 4,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 4,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 2,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 2,
        cin: 3,
        cout: 3,
        in_h: 7,
        in_w: 7,
        kernel: 3,
        stride: 2,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 5,
        cout: 5,
        in_h: 8,
        in_w: 8,
        kernel: 1,
        stride: 1,
        pad: 0,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 4,
        cout: 4,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 0,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 3,
        cout: 3,
        in_h: 9,
        in_w: 6,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    },
    Case {
        batch: 1,
        cin: 6,
        cout: 6,
        in_h: 16,
        in_w: 16,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: false,
    },
    Case {
        batch: 1,
        cin: 1,
        cout: 1,
        in_h: 5,
        in_w: 5,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    },
];

/// Direct depthwise convolution, written the obvious way, as the oracle
/// for `KernDepthWise::run`.
fn depthwise_reference(
    input: &[f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    dw: &KernDepthWise,
) -> Vec<f32> {
    let taps = dw.kernel_h * dw.kernel_w;
    let mut out = vec![0.0f32; batch * dw.channels * dw.output_h * dw.output_w];
    for b in 0..batch {
        for c in 0..dw.channels {
            for oy in 0..dw.output_h {
                for ox in 0..dw.output_w {
                    let mut acc = dw.bias.get(c).copied().unwrap_or(0.0);
                    for ky in 0..dw.kernel_h {
                        for kx in 0..dw.kernel_w {
                            let iy = (oy * dw.stride_h + ky * dw.dilation_h) as isize
                                - dw.pad_h as isize;
                            let ix = (ox * dw.stride_w + kx * dw.dilation_w) as isize
                                - dw.pad_w as isize;
                            if !(0..in_h as isize).contains(&iy)
                                || !(0..in_w as isize).contains(&ix)
                            {
                                continue;
                            }
                            let input_index =
                                ((b * dw.channels + c) * in_h + iy as usize) * in_w + ix as usize;
                            let weight = dw.weights[c * taps + ky * dw.kernel_w + kx];
                            acc += input[input_index] * weight;
                        }
                    }
                    let output_index =
                        ((b * dw.channels + c) * dw.output_h + oy) * dw.output_w + ox;
                    out[output_index] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn kern_depthwise_matches_reference_across_shapes() {
    for (index, case) in DEPTHWISE_CASES.iter().enumerate() {
        let (dw, input) = case.build_depthwise();
        let got = dw.run(&input, case.batch, case.in_h, case.in_w);
        let want = depthwise_reference(&input, case.batch, case.in_h, case.in_w, &dw);
        assert_close_enough(&format!("depthwise case {index}"), &got, &want);
    }
}

/// The vector kernel rounds `x * log2(e)`, so it can land on the wrong
/// side of a `.5` boundary and pull `r` about one `ln2` away from where
/// the series was fitted. That caps the achievable accuracy, hence a much
/// wider tolerance than the convolution tests use.
#[test]
fn silu_matches_the_scalar_oracle() {
    let mut input = values(4099, 0x51_51_51);
    input.extend([
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-6, -1e-6, 10.0, -10.0, 88.0, -88.0, 1e30, -1e30,
    ]);
    let want: Vec<f32> = input.iter().map(|&x| x * sigmoid_scalar(x)).collect();
    let mut got = vec![0.0f32; input.len()];
    silu(&input, &mut got);
    for (index, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-4 + 1e-4 * w.abs(),
            "index {index}: got {g}, want {w}"
        );
    }
}

/// Every one of these has `x * log2(e)` comfortably away from a `.5`
/// boundary, so the reduction is exact and only the series truncation is
/// left.
#[test]
fn silu_is_accurate_away_from_the_exponent_boundary() {
    let input = [
        -7.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0,
    ];
    let want: Vec<f32> = input.iter().map(|&x| x * sigmoid_scalar(x)).collect();
    let mut got = vec![0.0f32; input.len()];
    silu(&input, &mut got);
    for (index, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-7 + 1e-6 * w.abs(),
            "index {index}: got {g}, want {w}"
        );
    }
}

/// Lengths below one vector, and one that is not a multiple of four,
/// all take the scalar tail.
#[test]
fn silu_handles_every_tail_length() {
    for len in 0..12 {
        let input = values(len, 0x7a11);
        let want: Vec<f32> = input.iter().map(|&x| x * sigmoid_scalar(x)).collect();
        let mut got = vec![0.0f32; len];
        silu(&input, &mut got);
        assert_eq!(got.len(), want.len(), "length {len}");
        for (index, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-4 + 1e-4 * w.abs(),
                "length {len}, index {index}: got {g}, want {w}"
            );
        }
    }
}

/// The exponent saturates rather than wrapping, so a large positive input
/// keeps its value and a large negative one collapses towards zero.
#[test]
fn silu_saturates_instead_of_overflowing() {
    let input = [f32::MAX, 1e30, 89.0, -89.0, -1e30, -f32::MAX];
    let mut got = vec![0.0f32; input.len()];
    silu(&input, &mut got);
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    assert_eq!(got[0], f32::MAX);
    assert_eq!(got[1], 1e30);
    assert_eq!(got[2], 89.0);
    assert_eq!(got[3], 0.0);
    assert!(got[4].abs() < 1e-20, "{}", got[4]);
    assert!(got[5].abs() < 1e-20, "{}", got[5]);
}

/// Build `Sigmoid(x)` feeding `Mul(x, sig)` — the shape the ONNX export
/// produces after `prop_consts`.
/// The in-place kernel exists only so a fused convolution can activate
/// where it wrote, so it has to agree with the out-of-place one exactly.
#[test]
fn silu_in_place_matches_the_out_of_place_kernel() {
    for len in 0..=64 {
        let input = values(len, 0x51_51);
        let mut expected = vec![0.0; len];
        silu(&input, &mut expected);
        let mut in_place = input;
        silu_in_place(&mut in_place);
        assert_eq!(in_place, expected, "length {len}");
    }
}

/// Bias plus SiLU inside the convolution has to land on the same numbers
/// as adding the bias over the whole activation and activating afterwards.
/// Exact agreement is not expected: a tile boundary that is not a multiple
/// of four sends the tail down the scalar path instead of the vector one,
/// and the two exponentials differ in the last place.
#[test]
fn apply_silu_equals_a_separate_activation() {
    let cases = [
        Case {
            batch: 1,
            cin: 3,
            cout: 5,
            in_h: 8,
            in_w: 8,
            kernel: 3,
            stride: 1,
            pad: 1,
            with_bias: true,
        },
        Case {
            batch: 2,
            cin: 4,
            cout: 6,
            in_h: 7,
            in_w: 9,
            kernel: 3,
            stride: 2,
            pad: 1,
            with_bias: true,
        },
    ];
    for (index, case) in cases.iter().enumerate() {
        let (mut conv, input) = case.build();
        let plain = conv.run(&input, case.batch, case.in_h, case.in_w);
        let want: Vec<f32> = plain.iter().map(|&v| v * sigmoid_scalar(v)).collect();
        conv.apply_silu = true;
        let got = conv.run(&input, case.batch, case.in_h, case.in_w);
        assert_close_enough(&format!("case {index}"), &got, &want);
    }
}

/// source -> conv -> silu -> tail, with the folded reader no longer on the
/// path and the tail pointed straight at the convolution.
fn conv_silu_model(second_reader: bool) -> TractResult<(TypedModel, usize, usize)> {
    use tract_onnx::tract_core::model::{InletId, OutletId};
    use tract_onnx::tract_core::ops::identity::Identity;

    let (conv, _) = Case {
        batch: 1,
        cin: 3,
        cout: 5,
        in_h: 6,
        in_w: 6,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    }
    .build();

    let mut model = TypedModel::default();
    let source = model.add_source("x", f32::fact(tvec!(1usize, 3, 6, 6)))?;
    let source_fact = model.outlet_fact(source)?.clone();
    let conv_output = conv.output_facts(&[&source_fact])?[0].clone();
    let conv_id = model.add_node("conv", conv, tvec!(conv_output))?;
    model.add_edge(source, InletId::new(conv_id, 0))?;
    let conv_fact = model.outlet_fact(OutletId::new(conv_id, 0))?.clone();

    let silu_id = model.add_node("silu", KernSiLU, tvec!(conv_fact.clone()))?;
    model.add_edge(OutletId::new(conv_id, 0), InletId::new(silu_id, 0))?;
    let tail_id = model.add_node("tail", Identity, tvec!(conv_fact.clone()))?;
    model.add_edge(OutletId::new(silu_id, 0), InletId::new(tail_id, 0))?;
    let mut outputs = vec![OutletId::new(tail_id, 0)];

    if second_reader {
        let other = model.add_node("other", Identity, tvec!(conv_fact))?;
        model.add_edge(OutletId::new(conv_id, 0), InletId::new(other, 0))?;
        outputs.push(OutletId::new(other, 0));
    }
    model.set_output_outlets(&outputs)?;
    Ok((model, conv_id, silu_id))
}

/// A convolution read only by its activation may take the activation over,
/// and the reader's consumer has to follow the rewiring.
#[test]
fn fuse_conv_silu_folds_a_sole_reader() -> TractResult<()> {
    let (mut model, conv, silu) = conv_silu_model(false)?;
    assert_eq!(fuse_conv_silu(&mut model)?, 1);
    let folded = model
        .node(conv)
        .op_as::<KernConv>()
        .expect("still a native convolution");
    assert!(folded.apply_silu);
    assert_eq!(model.node(silu).op.name(), "Identity");
    let tail = &model.node(model.outputs[0].node);
    assert_eq!(tail.inputs[0].node, conv, "tail reads the convolution");
    Ok(())
}

/// The activation keeps computing when its input is read by something
/// else, because the convolution's output cannot be activated in place
/// behind another reader's back.
#[test]
fn fuse_conv_silu_leaves_a_shared_output_alone() -> TractResult<()> {
    let (mut model, conv, silu) = conv_silu_model(true)?;
    assert_eq!(fuse_conv_silu(&mut model)?, 0);
    let op = model
        .node(conv)
        .op_as::<KernConv>()
        .expect("still a native convolution");
    assert!(!op.apply_silu);
    assert_eq!(model.node(silu).op.name(), "KernRTSiLU");
    Ok(())
}

fn silu_shaped_model(extra_consumer: bool) -> TractResult<(TypedModel, usize, usize)> {
    use tract_onnx::tract_core::model::{InletId, OutletId};
    use tract_onnx::tract_core::ops::binary::TypedBinOp;
    use tract_onnx::tract_core::ops::element_wise::ElementWiseOp;
    use tract_onnx::tract_core::ops::math::Mul;
    use tract_onnx::tract_core::ops::nn::Sigmoid;

    let mut model = TypedModel::default();
    let source = model.add_source("x", f32::fact(tvec!(1usize, 6)))?;
    let fact = model.outlet_fact(source)?.clone();
    let sig = model.add_node(
        "sig",
        ElementWiseOp(Box::new(Sigmoid {}), None),
        tvec!(fact.clone()),
    )?;
    model.add_edge(source, InletId::new(sig, 0))?;
    let mul = model.add_node("mul", TypedBinOp(Box::new(Mul), None), tvec!(fact.clone()))?;
    model.add_edge(source, InletId::new(mul, 0))?;
    model.add_edge(OutletId::new(sig, 0), InletId::new(mul, 1))?;
    if extra_consumer {
        let other = model.add_node(
            "sig2",
            ElementWiseOp(Box::new(Sigmoid {}), None),
            tvec!(fact),
        )?;
        model.add_edge(OutletId::new(sig, 0), InletId::new(other, 0))?;
        model.set_output_outlets(&[OutletId::new(mul, 0), OutletId::new(other, 0)])?;
    } else {
        model.set_output_outlets(&[OutletId::new(mul, 0)])?;
    }
    Ok((model, sig, mul))
}

#[test]
fn rewrite_silu_replaces_a_sigmoid_fed_multiply() -> TractResult<()> {
    let (mut model, sig, mul) = silu_shaped_model(false)?;
    assert_eq!(rewrite_silu(&mut model)?, 1);
    assert_eq!(model.node(sig).op.name(), "Identity");
    assert_eq!(model.node(mul).op.name(), "KernRTSiLU");

    let input = values(6, 0x51_51);
    let runnable = model.into_optimized()?.into_runnable()?;
    let outputs = runnable.run(tvec!(
        tract_ndarray::Array2::from_shape_vec((1, 6), input.clone())?
            .into_tensor()
            .into_tvalue()
    ))?;
    let got = outputs[0]
        .to_array_view::<f32>()?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let want: Vec<f32> = input.iter().map(|&x| x * sigmoid_scalar(x)).collect();
    assert_close_enough("fused SiLU", &got, &want);
    Ok(())
}

/// Neutralising a Sigmoid changes every consumer, so one that still has
/// another reader has to keep computing.
#[test]
fn rewrite_silu_leaves_a_shared_sigmoid_alone() -> TractResult<()> {
    let (mut model, sig, mul) = silu_shaped_model(true)?;
    assert_eq!(rewrite_silu(&mut model)?, 0);
    assert_eq!(model.node(sig).op.name(), "Sigmoid");
    assert_eq!(model.node(mul).op.name(), "Mul");
    Ok(())
}

/// Direct reduce over every window, written the obvious way, as the
/// oracle for the two-pass kernel.
fn maxpool_reference(
    input: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    pad: usize,
) -> Vec<f32> {
    let kernel = 2 * pad + 1;
    let mut output = vec![0.0f32; batch * channels * in_h * in_w];
    for b in 0..batch {
        for c in 0..channels {
            let src = &input[(b * channels + c) * in_h * in_w..];
            let dst = &mut output[(b * channels + c) * in_h * in_w..];
            for oy in 0..in_h {
                for ox in 0..in_w {
                    let mut best = f32::NEG_INFINITY;
                    for dy in 0..kernel {
                        for dx in 0..kernel {
                            let iy = oy as isize + dy as isize - pad as isize;
                            let ix = ox as isize + dx as isize - pad as isize;
                            if (0..in_h as isize).contains(&iy) && (0..in_w as isize).contains(&ix)
                            {
                                best = best.max(src[iy as usize * in_w + ix as usize]);
                            }
                        }
                    }
                    dst[oy * in_w + ox] = best;
                }
            }
        }
    }
    output
}

/// Odd sizes make windows straddle a corner, and a plane smaller than the
/// window makes every output the max of the whole plane.
#[test]
fn maxpool_matches_a_naive_oracle_across_shapes() {
    for (batch, channels, in_h, in_w, pad) in [
        (1usize, 3usize, 8usize, 8usize, 2usize),
        (2, 1, 6, 10, 2),
        (3, 4, 5, 7, 2),
        (1, 2, 3, 3, 2),
        (2, 5, 4, 4, 1),
    ] {
        let pool = KernMaxPool {
            channels,
            in_h,
            in_w,
            pad,
        };
        let input = values(batch * channels * in_h * in_w, 0x5157);
        let got = pool.run(&input, batch);
        let want = maxpool_reference(&input, batch, channels, in_h, in_w, pad);
        assert_eq!(got.len(), want.len(), "{channels}x{in_h}x{in_w} pad {pad}");
        for (index, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                *g,
                *w,
                "batch {batch} channel {channel} element {index} of {in_h}x{in_w}",
                channel = index / (in_h * in_w),
            );
        }
    }
}

/// A model holding one pool of the given kernel, stride and symmetric
/// padding, shaped the way the ONNX importer writes them.
fn maxpool_model(
    channels: usize,
    size: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> TractResult<TypedModel> {
    use tract_onnx::tract_core::model::{InletId, OutletId};
    use tract_onnx::tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec};
    use tract_onnx::tract_core::ops::nn::DataFormat;

    let mut model = TypedModel::default();
    let source = model.add_source("x", f32::fact(tvec![1usize, channels, size, size]))?;
    let fact = model.outlet_fact(source)?.clone();
    let spec = PoolSpec::new(
        DataFormat::NCHW,
        tvec![kernel, kernel],
        PaddingSpec::ExplicitOnnxPool(tvec![pad, pad], tvec![pad, pad], false),
        None,
        Some(tvec![stride, stride]),
        channels,
        channels,
    );
    let op = MaxPool::new(spec, None);
    let output = op.output_facts(&[&fact])?[0].clone();
    let pool = model.add_node("pool", op, tvec!(output))?;
    model.add_edge(source, InletId::new(pool, 0))?;
    model.set_output_outlets(&[OutletId::new(pool, 0)])?;
    Ok(model)
}

/// The centred 5x5 pool is the one the native kernel was written for.
#[test]
fn rewrite_native_maxpool_replaces_a_centred_pool() -> TractResult<()> {
    let mut model = maxpool_model(4, 8, 5, 1, 2)?;
    assert_eq!(rewrite_native_maxpool(&mut model)?, 1);
    let native = model
        .node(model.outputs[0].node)
        .op_as::<KernMaxPool>()
        .expect("still a native pool");
    assert_eq!(
        (native.channels, native.in_h, native.in_w, native.pad),
        (4, 8, 8, 2)
    );
    Ok(())
}

/// A strided pool neither keeps the input shape nor reads a centred
/// window, so the kernel would have to cover both at once.
#[test]
fn rewrite_native_maxpool_leaves_a_strided_pool_alone() -> TractResult<()> {
    let mut model = maxpool_model(4, 8, 5, 2, 2)?;
    assert_eq!(rewrite_native_maxpool(&mut model)?, 0);
    assert_eq!(model.node(model.outputs[0].node).op.name(), "MaxPool");
    Ok(())
}

/// A model holding one `Reduce<Max>` over `axes`.
fn reduce_max_model(shape: &[usize], axes: &[usize]) -> TractResult<TypedModel> {
    use tract_onnx::tract_core::model::{InletId, OutletId};
    use tract_onnx::tract_core::ops::nn::{Reduce, Reducer};

    let mut model = TypedModel::default();
    let source = model.add_source("x", f32::fact(shape.to_vec()))?;
    let fact = model.outlet_fact(source)?.clone();
    let op = Reduce {
        axes: axes.iter().copied().collect(),
        reducer: Reducer::Max,
    };
    let output = op.output_facts(&[&fact])?[0].clone();
    let node = model.add_node("reduce", op, tvec!(output))?;
    model.add_edge(source, InletId::new(node, 0))?;
    model.set_output_outlets(&[OutletId::new(node, 0)])?;
    Ok(model)
}

/// The reduction itself, against a scalar fold over the same rows.
#[test]
fn reduce_max_matches_a_naive_oracle_across_shapes() {
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 80),
        (7, 5),
        (8400, 80),
        (13, 3),
        (2, 1),
    ] {
        let input = values(
            rows * cols,
            0x9e37_79b9_u64 ^ (rows as u64) << 8 | cols as u64,
        );
        let native = KernReduceMax { rows, cols }.run(&input);
        let want: Vec<f32> = input
            .chunks_exact(cols)
            .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            .collect();
        assert_eq!(native.len(), rows, "rows={rows} cols={cols}");
        for (index, (got, want)) in native.iter().zip(&want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 + 1e-6 * want.abs(),
                "row {index} of {rows}x{cols}: got {got} want {want}"
            );
        }
    }
}

/// The shape the detection head actually uses: reduce the innermost axis.
#[test]
fn rewrite_native_reduce_max_replaces_a_last_axis_reduce() -> TractResult<()> {
    let mut model = reduce_max_model(&[1, 8400, 80], &[2])?;
    assert_eq!(rewrite_native_reduce_max(&mut model)?, 1);
    let native = model
        .node(model.outputs[0].node)
        .op_as::<KernReduceMax>()
        .expect("still a native reduce");
    assert_eq!((native.rows, native.cols), (8400, 80));
    Ok(())
}

/// Reducing an outer axis strides through the input, which is the case the
/// vector kernel cannot cover.
#[test]
fn rewrite_native_reduce_max_leaves_an_outer_axis_alone() -> TractResult<()> {
    let mut model = reduce_max_model(&[8400, 80], &[0])?;
    assert_eq!(rewrite_native_reduce_max(&mut model)?, 0);
    assert_eq!(model.node(model.outputs[0].node).op.name(), "Reduce<Max>");
    Ok(())
}

#[test]
fn tmp_im2col_span_debug() {
    let case = Case {
        batch: 1,
        cin: 4,
        cout: 8,
        in_h: 8,
        in_w: 8,
        kernel: 3,
        stride: 1,
        pad: 1,
        with_bias: true,
    };
    let (conv, input) = case.build();
    let rows = 8usize;
    let out_w = 8usize;
    let tile = rows * out_w;
    let mut panel = vec![0.0f32; conv.input_channels * conv.kernel_h * conv.kernel_w * tile];
    conv.im2col(&input, 8, 8, 0, rows, &mut panel);
    let mut bad = 0;
    for c in 0..conv.input_channels {
        for ky in 0..conv.kernel_h {
            for kx in 0..conv.kernel_w {
                let p = (c * conv.kernel_h + ky) * conv.kernel_w + kx;
                for oy in 0..rows {
                    for ox in 0..out_w {
                        let iy = oy as isize + ky as isize - 1;
                        let ix = ox as isize + kx as isize - 1;
                        let want = if iy < 0 || ix < 0 || iy >= 8 || ix >= 8 {
                            0.0
                        } else {
                            input[(c * 8 + iy as usize) * 8 + ix as usize]
                        };
                        let got = panel[p * tile + oy * out_w + ox];
                        if got != want {
                            if bad < 12 {
                                println!("c{c} ky{ky} kx{kx} oy{oy} ox{ox}: got {got} want {want}");
                            }
                            bad += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} im2col mismatches");
}
