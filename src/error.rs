//! Error type for the `lfm` crate.
//!
//! Single `Error` enum (matches siglip2/egemma idiom). Style rules:
//! 1. Wrap, don't stringify external errors (use `Box<dyn Error + Send + Sync>`).
//! 2. `SmolStr` for runtime-built short strings.
//! 3. `&'static str` for fixed literals (outlet names, `InvalidRequest` reasons).
//! 4. `#[error(transparent)]` for already-self-describing wrapped errors.
//! 5. Named constructors when `From` would conflict (`Error::tokenizer`, `Error::llguidance`).

use smol_str::SmolStr;
use std::path::PathBuf;
use thiserror::Error;

/// Crate-level Result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Crate-level Error enum.
///
/// `#[non_exhaustive]` so adding new variants in a future minor release
/// (notably the streaming + tool-calling variants planned in §10 of the
/// design spec) is not a SemVer break — downstream `match` arms must
/// include a wildcard.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  // ===== Loading =====
  /// File not found at the given path.
  #[error("file not found: {0}")]
  NotFound(PathBuf),

  /// I/O error.
  #[error(transparent)]
  Io(#[from] std::io::Error),

  /// `ort` session-build or session-run failure.
  #[cfg(all(feature = "inference", ort_backend))]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// `tokenizers` load/use failure. Boxed to preserve the source chain.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Tokenizer(Box<dyn std::error::Error + Send + Sync>),

  // ===== Session validation =====
  /// ONNX session input/output dtype or presence mismatch.
  #[cfg(all(feature = "inference", ort_backend))]
  #[error("session contract mismatch on {input}: expected {expected}, got {got:?}")]
  SessionContractMismatch {
    /// Outlet name (input/output).
    input: &'static str,
    /// Human-readable expected dtype/shape.
    expected: &'static str,
    /// Actual ONNX element type.
    got: ort::value::TensorElementType,
  },

  /// ONNX session shape mismatch.
  #[cfg(all(feature = "inference", ort_backend))]
  #[error("session shape mismatch on {input}: expected {expected}, got {got:?}")]
  SessionShapeMismatch {
    /// Outlet name (input/output).
    input: &'static str,
    /// Human-readable expected shape.
    expected: &'static str,
    /// Actual shape vector.
    got: Vec<i64>,
  },

  /// Decoder cache layer count or sparse-index mismatch.
  #[cfg(all(feature = "inference", ort_backend))]
  #[error(
    "decoder cache mismatch: expected {expected_conv} conv + {expected_attn} attn, \
     got {got_conv} conv + {got_attn} attn"
  )]
  DecoderCacheMismatch {
    /// Expected number of conv-cache layers.
    expected_conv: usize,
    /// Expected number of attn-cache layers.
    expected_attn: usize,
    /// Actual number of conv-cache layers found in the session.
    got_conv: usize,
    /// Actual number of attn-cache layers found in the session.
    got_attn: usize,
  },

  // ===== Preprocessing =====
  /// Image decode failure (`feature = "decoders"`).
  #[cfg(feature = "decoders")]
  #[error(transparent)]
  ImageDecode(#[from] image::ImageError),

  /// Image too small for the current `ImageBudget`.
  #[error("image {w}x{h} too small for ImageBudget (need at least {min_w}x{min_h})")]
  ImageTooSmall {
    /// Source image width.
    w: u32,
    /// Source image height.
    h: u32,
    /// Minimum required width.
    min_w: u32,
    /// Minimum required height.
    min_h: u32,
  },

  /// Image dimensions are within per-axis limits but the worst-case
  /// decoded RGBA buffer would exceed the configured `max_alloc`.
  ///
  /// Caught at header time so embed.run + prompt expansion don't
  /// run for an image that the full decoder would reject anyway.
  #[error("image {w}x{h} would allocate ~{bytes} bytes when decoded (cap is {max_bytes} bytes)")]
  ImageDecodedBufferTooLarge {
    /// Source image width.
    w: u32,
    /// Source image height.
    h: u32,
    /// Worst-case decoded buffer size (raw_w × raw_h × 4 for RGBA).
    bytes: u64,
    /// Configured allocation cap (`decode_limits().max_alloc`).
    max_bytes: u64,
  },

  /// Tile-grid algorithm could not satisfy the budget.
  #[error("no valid tile grid for image {w}x{h} under budget {budget:?}")]
  TileGridImpossible {
    /// Source image width.
    w: u32,
    /// Source image height.
    h: u32,
    /// The image budget that produced no valid grid (for diagnostics).
    budget: crate::options::ImageBudget,
  },

  /// The one authoritative [`ImagePlan`](crate::preproc::ImagePlan) — the plan
  /// that rendered the prompt's image markers and sized admission control —
  /// disagrees with what the backend's own preprocessing actually produced for
  /// that image.
  ///
  /// Every backend re-derives its per-image layout from the pixels it really
  /// decoded; a disagreement means the `<image>`-token runs in the prompt and
  /// the vision feature rows would bind to different positions. Fail closed:
  /// the alternative is a silent misbinding whose total token count can still
  /// line up (e.g. a 4×2 grid's features spliced into a 2×4 marker layout).
  #[error(
    "image plan mismatch for image #{image}: {parameter} planned {planned}, preprocessing produced {produced} (prompt markers and vision features would bind to different positions)"
  )]
  ImagePlanMismatch {
    /// Zero-based index of the offending image in the request.
    image: usize,
    /// Which quantity disagreed (`"sub-image count"`, `"image tokens"`, …).
    parameter: &'static str,
    /// The value the authoritative plan carried.
    planned: usize,
    /// The value the backend's preprocessing produced.
    produced: usize,
  },

  /// A tiling parameter this build relies on disagrees with the MLX
  /// checkpoint's own value in `config.json`.
  ///
  /// The MLX path preprocesses through the checkpoint's own tiling
  /// (`Lfm2Vl::split_image`, driven by `config.json`), while the prompt's
  /// `<image>` markers and the admission gates are rendered from `lfm`'s
  /// [`ImageBudget`](crate::ImageBudget) plus its hardcoded patch/tile
  /// constants. If the two disagree the marker layout and the feature block
  /// disagree too, so the load is refused **by name** rather than silently
  /// producing a mismatched prompt.
  ///
  /// The default [`ImageBudget::new()`](crate::ImageBudget::new) expresses "no
  /// opinion" and adopts the checkpoint's tiling instead of being compared
  /// against it; any other budget must match exactly.
  #[error(
    "MLX checkpoint tiling mismatch on `{parameter}`: lfm has {lfm_value}, the checkpoint's config.json has {checkpoint_value}"
  )]
  MlxTilingMismatch {
    /// The disagreeing parameter's name.
    parameter: &'static str,
    /// The value `lfm` would use (from `ImageBudget` or a crate constant).
    lfm_value: SmolStr,
    /// The value the checkpoint's `config.json` carries.
    checkpoint_value: SmolStr,
  },

  // ===== Backend / checkpoint selection =====
  /// The checkpoint directory does not resolve to exactly one backend and no
  /// [`Options::with_backend`](crate::Options::with_backend) pin says which to
  /// use.
  ///
  /// Raised for genuinely undecidable layouts — notably a *partial* ONNX graph
  /// set sitting next to an MLX weight set. (A **complete** ONNX graph set next
  /// to MLX-format source assets is decided, not ambiguous: the graphs win, as
  /// documented on [`Engine::from_dir`](crate::Engine::from_dir), and
  /// [`Engine::backend`](crate::Engine::backend) reports the choice.)
  #[error("ambiguous checkpoint layout in {dir}: {detail}")]
  CheckpointLayoutAmbiguous {
    /// The checkpoint directory that could not be classified.
    dir: PathBuf,
    /// What made it ambiguous.
    detail: &'static str,
  },

  /// The directory holds an MLX checkpoint whose only weight file is in a
  /// format this build did not enable.
  ///
  /// Detected up front, so the load reports the disabled format instead of
  /// falling through to the ONNX path and failing on an unrelated missing
  /// graph.
  #[error(
    "checkpoint in {dir} carries MLX weights only in the `{format}` format, which this build of lfm does not enable — rebuild with the `{format}` feature, or supply weights in an enabled format"
  )]
  CheckpointFormatDisabled {
    /// The checkpoint directory.
    dir: PathBuf,
    /// The disabled weight format (`"npz"` / `"gguf"`).
    format: &'static str,
  },

  /// The directory looks like a checkpoint but is missing the files the
  /// selected backend needs.
  #[error("incomplete checkpoint in {dir}: {detail}")]
  CheckpointIncomplete {
    /// The checkpoint directory.
    dir: PathBuf,
    /// What is missing.
    detail: &'static str,
  },

  /// The backend pinned by [`Options::with_backend`](crate::Options::with_backend)
  /// cannot serve this checkpoint on this platform.
  ///
  /// Never silently substituted — a caller who asked for MLX and got ONNX
  /// would be running a different numerical path than requested.
  #[error("backend `{requested}` unavailable for this checkpoint: {reason}")]
  BackendUnavailable {
    /// The backend the caller pinned.
    requested: crate::options::BackendKind,
    /// Why it cannot be served.
    reason: &'static str,
  },

  // ===== Tokenization / template =====
  /// `<image>` placeholder count mismatch with input image count.
  #[error("expected {expected} <image> placeholder(s) in prompt, got {got}")]
  ImageTokenCountMismatch {
    /// Number of images supplied to the call.
    expected: usize,
    /// Number of `<image>` placeholders found in the rendered prompt.
    got: usize,
  },

  /// Tokenized prompt + `max_new_tokens` exceeds the model's context
  /// window (128 K). Caught after tokenization, before any embedding
  /// or decoder work — the model would either OOM the KV cache or run
  /// past its valid position-embedding range otherwise.
  #[error(
    "context length exceeded: prompt_tokens={prompt_tokens} + max_new_tokens={max_new_tokens} > model_context={model_context}"
  )]
  ContextLengthExceeded {
    /// Number of tokens in the rendered prompt (after image expansion).
    prompt_tokens: usize,
    /// Configured `RequestOptions::max_new_tokens`.
    max_new_tokens: usize,
    /// Model's `max_position_embeddings` from `config.json`.
    model_context: usize,
  },

  /// Pre-decode grid layout (used to render the prompt's image
  /// markers) differs from the post-decode grid layout (computed
  /// from the actually-decoded image dimensions). Raised inside the
  /// splice loop as a release-time defense-in-depth check; should
  /// never fire in practice now that `image_dimensions()` is
  /// EXIF-aware.
  #[error(
    "image grid layout mismatch: prompt rendered for {expected_rows}x{expected_cols} grid, decoded image yielded {actual_rows}x{actual_cols} (markers and vision features would bind to wrong spatial positions)"
  )]
  ImageGridLayoutMismatch {
    /// Grid rows used to render the prompt's image markers.
    expected_rows: usize,
    /// Grid cols used to render the prompt's image markers.
    expected_cols: usize,
    /// Grid rows from the actually-decoded image.
    actual_rows: usize,
    /// Grid cols from the actually-decoded image.
    actual_cols: usize,
  },

  // ===== Generation =====
  /// llguidance compile or matcher failure.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  LlGuidance(Box<dyn std::error::Error + Send + Sync>),

  /// llguidance produced an all-zero next-token mask.
  #[error("llguidance produced empty mask at step {step}: {state}")]
  LlGuidanceDeadEnd {
    /// Decode step at which the dead-end occurred.
    step: usize,
    /// Short debug string of the matcher state.
    state: SmolStr,
  },

  /// Sampler observed at least one non-finite logit after applying
  /// repetition_penalty (issue #2 C-001).
  /// Sources: ONNX model emitted NaN/Inf (numerical overflow,
  /// malformed export) OR repetition_penalty multiplication
  /// overflowed an extreme negative logit. Either case poisons
  /// argmax (`f32::total_cmp` orders NaN as the largest) and
  /// softmax (NaN spreads through `e^x` and the sum). Fail closed
  /// rather than emit garbage tokens silently.
  #[error(
    "sampler logits include non-finite value(s) (model output NaN/Inf, or penalty × negative logit overflow)"
  )]
  SamplerNonFinite,

  /// A model stage produced a NaN or infinite value before any sampler logic
  /// ran. Covers the embed_tokens / vision_encoder / decoder pipelines on
  /// **either** backend — e.g. a numerically broken ONNX export, a vendor
  /// execution provider with a bad kernel, a corrupted weight file, or an MLX
  /// kernel overflowing on a quantized checkpoint.
  ///
  /// Fail closed at the stage boundary so a single non-finite value can't
  /// silently propagate through the embedding splice + decoder attention into
  /// every subsequent step. In particular a lone `+inf` logit is rejected here
  /// rather than admitted: greedy would always pick it, and the temperature
  /// path's softmax degrades to a uniform distribution spread over *every*
  /// entry — including the `-inf` entries an llguidance mask used to forbid a
  /// token, so a schema-disallowed token could win the draw.
  ///
  /// The `stage` identifies which stage produced the bad output. Both backends
  /// raise the same variant for the same defect, so the ONNX and MLX roads
  /// stay behaviourally identical at this boundary.
  #[error("{stage} stage produced non-finite output (NaN/Inf)")]
  SessionNonFiniteOutput {
    /// Which stage: `"embed_tokens"`, `"vision_encoder"`, or `"decoder"`.
    stage: &'static str,
  },

  /// Generation hit `max_new_tokens` without EOS or schema-complete.
  ///
  /// Distinct from [`Error::Empty`], which fires when EOS arrives at
  /// step 0 (model declined to generate); `MaxTokensExceeded` fires when
  /// generation ran productively but didn't terminate before the cap.
  #[error("hit max_new_tokens={max} (schema_complete={schema_complete})")]
  MaxTokensExceeded {
    /// The max_new_tokens cap that was hit.
    max: usize,
    /// True if the schema was already in an accepting state at the cap.
    schema_complete: bool,
  },

  /// Detokenize produced invalid UTF-8.
  #[error("detokenize produced invalid UTF-8")]
  InvalidUtf8,

  /// Generation produced no output — EOS was the first token sampled,
  /// before any content was emitted. Distinct from
  /// [`Error::MaxTokensExceeded`], which fires when generation runs to
  /// the cap; `Empty` fires only when the model declined to generate
  /// at all.
  #[error("generation produced empty output")]
  Empty,

  // ===== MLX backend (Apple Silicon only) =====
  /// MLX (`mlxrs`) backend failure — checkpoint load, model construction, or a
  /// device-tensor op on the Apple-Silicon path.
  ///
  /// `mlxrs::Error` is a structured enum; it is stringified into this variant at
  /// the crate boundary (the same way the ONNX path's
  /// [`Tokenizer`](Self::Tokenizer) / [`LlGuidance`](Self::LlGuidance) wrap their
  /// sources) because lfm does not depend on `mlxrs` off macOS/arm64, so the
  /// concrete type cannot appear in the public `Error` shape on every platform.
  #[cfg(mlx_backend)]
  #[error("mlx backend: {0}")]
  Mlx(String),

  // ===== Configuration =====
  /// Invalid `RequestOptions`.
  #[error("invalid RequestOptions: {0}")]
  InvalidRequest(&'static str),

  /// Invalid `ImageBudget`.
  #[error("invalid ImageBudget: {0}")]
  InvalidBudget(&'static str),

  // ===== Task parse =====
  /// Forwarded from `llmtask::JsonParseError`.
  #[error(transparent)]
  Parse(#[from] llmtask::JsonParseError),
}

impl Error {
  /// Wrap any `Error + Send + Sync` source as a `Tokenizer` variant.
  /// Use at call-sites: `.map_err(Error::tokenizer)`.
  ///
  /// `#[allow(dead_code)]` is item-scoped (not impl-scoped) so future
  /// helpers added to `impl Error` still surface dead-code warnings.
  /// Tasks 6–13 will wire this constructor into runtime call-sites.
  #[allow(dead_code)]
  #[cfg(feature = "inference")]
  pub(crate) fn tokenizer<E>(e: E) -> Self
  where
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
  {
    Self::Tokenizer(e.into())
  }

  /// Wrap any `Error + Send + Sync` source as an `LlGuidance` variant.
  /// Use at call-sites: `.map_err(Error::llguidance)`.
  ///
  /// Same `#[allow(dead_code)]` rationale as [`Self::tokenizer`].
  #[allow(dead_code)]
  #[cfg(feature = "inference")]
  pub(crate) fn llguidance<E>(e: E) -> Self
  where
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
  {
    Self::LlGuidance(e.into())
  }

  /// Wrap an `mlxrs::Error` (any `Display` source) as the [`Mlx`](Self::Mlx)
  /// variant. Use at call-sites on the MLX path: `.map_err(Error::from_mlx)`.
  /// `mlxrs::Error` is structured but cannot appear in lfm's cross-platform
  /// `Error` shape (lfm does not depend on `mlxrs` off macOS/arm64), so it is
  /// stringified at the boundary.
  #[allow(dead_code)]
  #[cfg(mlx_backend)]
  pub(crate) fn from_mlx<E: std::fmt::Display>(e: E) -> Self {
    Self::Mlx(e.to_string())
  }

  /// Build an [`Mlx`](Self::Mlx) error from a static message (an MLX-path
  /// invariant with no source error).
  #[allow(dead_code)]
  #[cfg(mlx_backend)]
  pub(crate) fn mlx(msg: &'static str) -> Self {
    Self::Mlx(msg.to_string())
  }

  /// Build an [`Mlx`](Self::Mlx) error from an owned, runtime-built message
  /// (e.g. one naming an offending token id or a bad logits shape).
  #[allow(dead_code)]
  #[cfg(mlx_backend)]
  pub(crate) fn mlx_owned(msg: String) -> Self {
    Self::Mlx(msg)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use smol_str::SmolStr;

  #[test]
  fn ll_guidance_dead_end_state_inlined_smolstr() {
    let e = Error::LlGuidanceDeadEnd {
      step: 5,
      state: SmolStr::new_inline("regex stuck"),
    };
    let msg = format!("{e}");
    assert!(msg.contains("step 5"));
    assert!(msg.contains("regex stuck"));
  }

  #[test]
  fn invalid_request_uses_static_str() {
    fn classify(e: &Error) -> &'static str {
      match e {
        Error::InvalidRequest(s) => s,
        _ => "other",
      }
    }
    let e = Error::InvalidRequest("max_new_tokens must be > 0");
    assert_eq!(classify(&e), "max_new_tokens must be > 0");
  }

  #[cfg(feature = "inference")]
  #[test]
  fn tokenizer_constructor_round_trips_through_box_dyn() {
    let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad utf8");
    let e = Error::tokenizer(inner);
    let display = format!("{e}");
    assert!(display.contains("bad utf8"));
  }

  #[cfg(feature = "inference")]
  #[test]
  fn llguidance_constructor_round_trips_through_box_dyn() {
    let inner = std::io::Error::other("matcher exhausted");
    let e = Error::llguidance(inner);
    let display = format!("{e}");
    assert!(display.contains("matcher exhausted"));
  }

  #[test]
  fn from_io_works_via_question_mark() {
    fn inner() -> Result<()> {
      std::fs::read("/no/such/path/here").map(|_| ())?;
      Ok(())
    }
    let e = inner().unwrap_err();
    assert!(matches!(e, Error::Io(_)));
  }
}
