//! Pluggable compute backends.
//!
//! Every operation that reads weights, or touches an activation, is a
//! [`Backend`] method. The driver in [`crate::arch::llm::llama`] owns the
//! schedule — which layer, which projection, which row attends where — and a
//! backend owns the memory and the kernels.
//!
//! # Buffers are handles, not slices
//!
//! Activations live in backend memory and cross the API as [`Buf`] handles.
//! That is what makes a GPU backend viable: with host slices every op would
//! have to hand data over, wait, and hand it back, which is one synchronisation
//! per projection per token — 210 of them for a Llama, at roughly 250us each.
//! With handles a backend can encode a whole step and synchronise once, and
//! [`read`](Backend::read) is the only place data has to come back.
//!
//! # Three ship today
//!
//! * [`cpu`] — the hand-written NEON/Q8_0 kernels and the f32-widened copies
//!   the batched GEMM consumes. This is the default.
//! * `metal` — device-resident weights and activations. Behind the `metal`
//!   feature, macOS only.
//! * [`reference`] — scalar f32 loops with no packing, no activation
//!   quantization and no SIMD. It is slow on purpose: it is the oracle the
//!   other backends are checked against.
//!
//! Selection is [`from_env`] (`KERNRT_BACKEND=cpu|metal|reference`) or [`open`].

use std::sync::Arc;

use crate::formats::llama2c::LayerWeights;
use crate::ops::Matrix;
use crate::tensor::Tensor;

pub mod cpu;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
pub mod reference;

/// A handle to activation memory owned by a backend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Buf(pub(crate) usize);

/// Which projection of a decoder layer a call refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    /// Query projection, `[q_dim, dim]`.
    Q,
    /// Key projection, `[kv_dim, dim]`.
    K,
    /// Value projection, `[kv_dim, dim]`.
    V,
    /// Attention output projection, `[dim, q_dim]`.
    O,
    /// SwiGLU gate projection, `[hidden_dim, dim]`.
    Gate,
    /// SwiGLU down projection, `[dim, hidden_dim]`.
    Down,
    /// SwiGLU up projection, `[hidden_dim, dim]`.
    Up,
    /// LFM2 short-conv input projection, `[3*dim, dim]`. Only live on hybrid
    /// models, on layers whose weights carry it.
    ConvIn,
    /// LFM2 short-conv output projection, `[dim, dim]`.
    ConvOut,
}

impl Proj {
    /// The logical weight this projection reads, or `None` when the layer
    /// does not carry it: attention projections on a convolution layer, or
    /// convolution projections on an attention layer.
    pub fn matrix_opt(self, layer: &LayerWeights) -> Option<&Matrix> {
        match self {
            Proj::Q => layer.wq.as_ref(),
            Proj::K => layer.wk.as_ref(),
            Proj::V => layer.wv.as_ref(),
            Proj::O => layer.wo.as_ref(),
            Proj::Gate => Some(&layer.w1),
            Proj::Down => Some(&layer.w2),
            Proj::Up => Some(&layer.w3),
            Proj::ConvIn => layer.conv_in.as_ref(),
            Proj::ConvOut => layer.conv_out.as_ref(),
        }
    }

    /// The logical weight this projection reads.
    ///
    /// Panics when the layer does not carry it, which means the driver
    /// dispatched a projection to the wrong layer kind.
    pub fn matrix(self, layer: &LayerWeights) -> &Matrix {
        self.matrix_opt(layer).unwrap_or_else(|| {
            panic!(
                "projection {self:?} is not live on this layer (is_attention={})",
                layer.is_attention
            )
        })
    }

    /// Dense index for backend-side arrays.
    pub fn index(self) -> usize {
        match self {
            Proj::Q => 0,
            Proj::K => 1,
            Proj::V => 2,
            Proj::O => 3,
            Proj::Gate => 4,
            Proj::Down => 5,
            Proj::Up => 6,
            Proj::ConvIn => 7,
            Proj::ConvOut => 8,
        }
    }

    /// All nine projections, in [`Proj::index`] order. Prepare paths iterate
    /// this and skip the ones a layer does not carry; the driver only ever
    /// dispatches the live ones.
    pub const ALL: [Proj; 9] = [
        Proj::Q,
        Proj::K,
        Proj::V,
        Proj::O,
        Proj::Gate,
        Proj::Down,
        Proj::Up,
        Proj::ConvIn,
        Proj::ConvOut,
    ];
}

/// Everything a backend needs from a loaded checkpoint.
pub struct ModelWeights {
    pub token_embedding: Matrix,
    pub layers: Vec<LayerWeights>,
    pub rms_final: Tensor,
    /// `None` when the classifier is tied to the token embedding.
    pub wcls: Option<Matrix>,
}

impl ModelWeights {
    /// The matrix the final projection reads: the untied head when present,
    /// otherwise the embedding table.
    pub fn classifier(&self) -> &Matrix {
        self.wcls.as_ref().unwrap_or(&self.token_embedding)
    }
}

/// RoPE parameters for one call. Positions are per row.
#[derive(Debug, Clone, Copy)]
pub struct RopeSpec {
    pub head_size: usize,
    pub theta: f32,
    /// `true` rotates adjacent pairs (llama2.c); `false` rotates the two halves
    /// of each head (HuggingFace Llama, SmolLM2, Qwen3).
    pub interleaved: bool,
}

/// Attention geometry for one call.
#[derive(Debug, Clone, Copy)]
pub struct AttnSpec {
    pub n_heads: usize,
    /// Query heads sharing each key/value head.
    pub kv_mul: usize,
    pub head_size: usize,
    pub kv_dim: usize,
    /// Stride between layers in the cache, in elements.
    pub seq_len: usize,
}

/// Geometry of the key/value cache a backend owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSpec {
    pub n_layers: usize,
    /// Elements per key or value vector, `n_kv_heads * head_size`.
    pub kv_dim: usize,
    /// Slots allocated per layer.
    pub seq_len: usize,
}

/// One query row's causal window.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    /// First slot of this row's own range.
    pub att_start: usize,
    /// Last slot of this row's own range, inclusive.
    pub slot: usize,
    /// Extra disjoint range also attended, `(start, end)` exclusive.
    pub shared: (usize, usize),
}

/// Failure to resolve a backend by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    pub requested: String,
    /// Set when the backend exists but refused to start — no GPU, a shader
    /// that failed to compile — rather than being unknown.
    pub detail: Option<String>,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(detail) => write!(f, "backend {:?} is unavailable: {detail}", self.requested),
            None => write!(
                f,
                "unknown backend {:?}; available: {}",
                self.requested,
                available().join(", ")
            ),
        }
    }
}

impl std::error::Error for BackendError {}

/// Names accepted by [`open`], in preference order.
pub fn available() -> &'static [&'static str] {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        &["cpu", "metal", "reference"]
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        &["cpu", "reference"]
    }
}

/// Build a backend by name.
pub fn open(name: &str) -> Result<Box<dyn Backend>, BackendError> {
    match name {
        "cpu" | "neon" | "default" => Ok(Box::new(cpu::Cpu::new())),
        "reference" | "ref" | "scalar" => Ok(Box::new(reference::Reference::new())),
        #[cfg(all(target_os = "macos", feature = "metal"))]
        "metal" | "gpu" => metal::Metal::new()
            .map(|m| Box::new(m) as Box<dyn Backend>)
            .map_err(|detail| BackendError {
                requested: name.to_string(),
                detail: Some(detail),
            }),
        other => Err(BackendError {
            requested: other.to_string(),
            detail: None,
        }),
    }
}

/// The backend selected by `KERNRT_BACKEND`, defaulting to the tuned CPU path.
pub fn from_env() -> Box<dyn Backend> {
    let name = std::env::var("KERNRT_BACKEND").unwrap_or_else(|_| "cpu".to_string());
    let name = name.trim();
    let name = if name.is_empty() { "cpu" } else { name };
    open(name).unwrap_or_else(|error| panic!("{error}"))
}

/// The compute surface a model driver needs.
///
/// All activation arguments are handles produced by [`alloc`](Backend::alloc)
/// or [`upload`](Backend::upload). Sizes are implicit in the handle; `rows`
/// says how many rows of a row-major `[rows, width]` buffer are live, matching
/// the driver's habit of sizing scratch for the longest prompt seen and reusing
/// it for shorter ones.
///
/// A backend may defer execution: nothing here requires the work to have
/// finished. [`read`](Backend::read) is the barrier.
pub trait Backend: Send {
    /// Short name, for logs and benchmark tables.
    fn name(&self) -> &'static str;

    /// The weight storage the model was loaded at.
    fn storage(&self) -> &'static str;

    /// Prepare weights, key/value cache and any derived layout. Called once.
    fn prepare(&mut self, weights: Arc<ModelWeights>, kv: KvSpec);

    /// Allocate `len` f32 elements, uninitialised.
    fn alloc(&mut self, len: usize) -> Buf;

    /// Copy host data into a new buffer.
    fn upload(&mut self, data: &[f32]) -> Buf;

    /// Copy `out.len()` elements of `src` back to the host, completing any
    /// deferred work first.
    fn read(&mut self, src: Buf, out: &mut [f32]);

    /// Copy one token's embedding row into `out` at element `offset`.
    fn embed(&mut self, token: usize, out: Buf, offset: usize);

    /// `dst[dst_offset..] = src[src_offset..]` over `len` elements.
    fn copy(&mut self, src: Buf, src_offset: usize, dst: Buf, dst_offset: usize, len: usize);

    /// `out = weight * x / sqrt(mean(x^2) + eps)` over `rows` rows of `dim`.
    fn rms_norm(&mut self, out: Buf, x: Buf, weight: Buf, rows: usize, dim: usize, eps: f32);

    /// Per-head RMS norm over `rows` rows of `width` elements each.
    fn qk_norm(
        &mut self,
        v: Buf,
        rows: usize,
        width: usize,
        head_size: usize,
        weight: Buf,
        eps: f32,
    );

    /// Rotary position embedding, `positions[i]` for row `i`.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &mut self,
        q: Buf,
        k: Buf,
        q_width: usize,
        kv_width: usize,
        rows: usize,
        positions: &[usize],
        spec: RopeSpec,
    );

    /// `y[rows, out] = x[rows, in] * w[out, in]^T` for one layer's projection.
    fn project(&mut self, layer: usize, proj: Proj, x: Buf, y: Buf, rows: usize);

    /// Write `rows` key/value rows into the cache, `slots[i]` for row `i`.
    ///
    /// Batched rather than one call per row: at 128 prompt tokens over 30
    /// layers the per-row form is 3840 dispatches, and their scheduling
    /// latency dominates the arithmetic.
    fn kv_write(&mut self, layer: usize, slots: &[usize], k: Buf, v: Buf, rows: usize);

    /// Multi-head causal attention for `rows` query rows in one call.
    ///
    /// `windows[i]` is row `i`'s causal window. `att` holds the scores,
    /// indexed `[row, head, seq_len]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        q: Buf,
        out: Buf,
        att: Buf,
        layer: usize,
        windows: &[Window],
        spec: AttnSpec,
    );

    /// `dst += src` over `len` elements.
    fn add_in_place(&mut self, dst: Buf, src: Buf, len: usize);

    /// `out[i] = a[a_off+i] * b[b_off+i]` over `len` elements.
    ///
    /// The offsets let a driver multiply slices of one buffer — the LFM2
    /// short conv multiplies the `B` and `x` thirds of its input projection
    /// in place, without materialising either.
    #[allow(clippy::too_many_arguments)]
    fn mul(
        &mut self,
        out: Buf,
        out_off: usize,
        a: Buf,
        a_off: usize,
        b: Buf,
        b_off: usize,
        len: usize,
    );

    /// Batched causal depthwise convolution for the LFM2 short conv.
    ///
    /// Row `r` of `x` holds `dim` elements; `kernel` holds `dim * k` taps in
    /// `[dim, k]` order. Output row `r` is the causal sum over the current
    /// and previous `k - 1` rows of `x`, zero-padded before row 0. The state
    /// ring the decode path keeps is *not* touched: the driver copies the
    /// last `k - 1` rows into it with [`copy`](Backend::copy).
    fn conv_prefill(&mut self, out: Buf, x: Buf, kernel: Buf, rows: usize, dim: usize, k: usize);

    /// Single-row decode through the LFM2 short conv.
    ///
    /// `state` holds the previous `k - 1` rows (`(k-1)*dim` elements, oldest
    /// first). The output is the causal sum over state and `x`; afterwards
    /// the state drops its oldest row and gains `x`. Reads and writes happen
    /// in that order, so an output buffer that aliases `x` still sees the
    /// pre-update history.
    fn conv_decode(&mut self, out: Buf, x: Buf, kernel: Buf, state: Buf, dim: usize, k: usize);

    /// `gate[i] = silu(gate[i]) * up[i]` over `len` elements.
    fn silu_mul(&mut self, gate: Buf, up: Buf, len: usize);

    /// Next-token logits: the classifier over one row.
    fn logits(&mut self, x: Buf, y: Buf);
}
