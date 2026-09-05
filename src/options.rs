//! Configuration types: the [`Options`] document and the tiers it is built from.
//!
//! # The document is layered, and every layer defaults
//!
//! [`Options`] is one flat document with two tiers. The **top tier** holds
//! what every backend shares — the sampler ([`RequestOptions`]) and the image
//! preprocessing budget ([`ImageBudget`]). The **engine tier** is a single
//! flattened field, [`BackendOptions`]: an internally tagged enum whose
//! `backend` key both names the road and selects that road's own knob struct
//! ([`AutoOptions`], `OrtOptions`, `MlxOptions`). Both tiers carry serde
//! defaults, so a partial table such as `{ "backend": "mlx" }` fills the rest
//! from each tier's own `new()` instead of being refused by the first missing
//! field's name.
//!
//! ```text
//! { "request": { .. }, "image_budget": { .. },   <- top tier: shared knobs
//!   "backend": "onnx",                           <- the tag: picks the road
//!   "thread": { .. }, "optimization_level": ".." }  <- that road's own knobs
//! ```
//!
//! # Nested tables default too, key by key
//!
//! Every nested, non-flattened table of the document — [`RequestOptions`],
//! [`ImageBudget`] and (under `backend = "onnx"`) `ThreadOptions` — is itself
//! `#[serde(default)]`. A table naming only one key, `{ "request": {
//! "max_new_tokens": 512 } }`, used to be refused by the *next* missing
//! field's name instead of the first; it now fills the rest from that type's
//! own `Default`. That `Default` is, by construction, EXACTLY the value the
//! enclosing tier already substitutes when the whole table is absent — one
//! source of truth, not a second copy of the numbers: [`ImageBudget::default`]
//! and `ThreadOptions::default` simply return `new()` (the absent-table value
//! was already `new()`/`None`+`None`), but [`RequestOptions::default`] returns
//! [`RequestOptions::deterministic`], **not** [`RequestOptions::new`] —
//! because `deterministic()`, not `new()`, is what [`Options::new`] was
//! already carrying for an absent `request` table. `new()` keeps its own
//! meaning throughout: the explicit, model-card-recommended constructor a
//! caller reaches for by name, never the type's ambient default.
//!
//! # What the shape refuses, and how it says so
//!
//! - **A road this build does not compile** is refused by the variant roster
//!   itself — ``unknown variant `mlx`, expected `auto` or `onnx` ``. The
//!   engine tiers are `cfg`-gated on their backend's presence (see
//!   `Cargo.toml`'s target tables and the `ort_backend` cfg), so the roster in
//!   that message *is* the list of roads this build can run.
//! - **A misspelled engine knob** is refused by the variant struct's
//!   `deny_unknown_fields` — ``unknown field `optimisation_level`, expected
//!   `thread` or `optimization_level` ``.
//! - **A misspelled *shared* knob** is refused by the same rule, because an
//!   unmatched top-tier key falls through the flatten into the variant struct.
//!   The refusal therefore names the *engine* tier's fields, not the top
//!   tier's: `{ "backend": "mlx", "reqeust": {} }` reports ``unknown field
//!   `reqeust`, there are no fields``. The key is refused, which is the point;
//!   the field list in the message belongs to the tier that caught it.
//! - **An engine knob under the wrong road** — `optimization_level` beside
//!   `backend = "mlx"` — is refused the same way, rather than accepted and
//!   ignored.
//!
//! Serde's flatten cannot supply a missing tag, so `backend` is a **required**
//! key of the document: name the road, or name `"auto"` to leave the choice to
//! the checkpoint layout. An absent `backend` is ``missing field `backend` ``.
//! The consumer formats are JSON, YAML and TOML.

#[cfg(all(feature = "inference", ort_backend))]
#[cfg_attr(
  docsrs,
  doc(cfg(any(
    all(
      target_arch = "x86_64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "x86_64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(target_os = "macos", target_arch = "aarch64", feature = "ort")
  )))
)]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// =========================================================================
// RequestOptions
// =========================================================================

/// Sampler configuration applied per call by `Engine::run` / `generate` /
/// their `_with` variants.
///
/// LFM2.5-VL uses **min_p sampling** (NOT top_p / top_k); fields reflect
/// the model card's recommended sampler. Two named presets ship out of
/// the box: `RequestOptions::new()` (model-card defaults) and
/// `RequestOptions::deterministic()` (greedy + retained repetition_penalty).
///
/// `#[serde(default)]`: a partial `request` table fills whatever key it
/// leaves out from [`Self::default`] — the same value an absent `request`
/// table already got from [`Options::new`]'s own field value. That value is
/// [`Self::deterministic`], **not** [`Self::new`]: the safer, bit-stable
/// preset is what a caller gets for free, and `new()` remains the explicit
/// constructor for a caller who wants the model card's own recommended
/// (non-deterministic) values. The two are documented as differing on
/// purpose; see `default_differs_from_new_on_purpose` for the pin.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct RequestOptions {
  temperature: f32,
  min_p: f32,
  repetition_penalty: f32,
  max_new_tokens: usize,
}

impl RequestOptions {
  /// Defaults from the LFM2.5-VL-450M model card text:
  /// `temperature=0.1`, `min_p=0.15`, `repetition_penalty=1.05`,
  /// `max_new_tokens=512`. Best output quality; not bit-stable.
  ///
  /// Source: <https://huggingface.co/LiquidAI/LFM2.5-VL-450M> §"Inference".
  pub const fn new() -> Self {
    Self {
      temperature: 0.1,
      min_p: 0.15,
      repetition_penalty: 1.05,
      max_new_tokens: 512,
    }
  }

  /// Indexing-safe greedy: `temperature=0.0`, `repetition_penalty=1.05`
  /// retained (greedy without it loops on small models). `min_p` is
  /// irrelevant under argmax.
  ///
  /// **Bit-stability caveat:** greedy is necessary but not sufficient.
  /// ORT bit-stability also requires `intra_threads=1`, `inter_threads=1`,
  /// and CPU-only EP. See `ThreadOptions` + EP feature flags.
  pub const fn deterministic() -> Self {
    Self {
      temperature: 0.0,
      min_p: 0.0,
      repetition_penalty: 1.05,
      max_new_tokens: 512,
    }
  }

  /// Returns the sampling temperature.
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }
  /// Returns the min-p sampling cutoff.
  pub const fn min_p(&self) -> f32 {
    self.min_p
  }
  /// Returns the repetition penalty (≥ 1.0 means penalty is applied).
  pub const fn repetition_penalty(&self) -> f32 {
    self.repetition_penalty
  }
  /// Returns the maximum number of new tokens to generate.
  pub const fn max_new_tokens(&self) -> usize {
    self.max_new_tokens
  }

  /// Returns a copy with the given temperature.
  ///
  /// **Validation note:** [`Self::validate`] accepts `0.0` (greedy
  /// decoding) and any value `>= 1e-3`. Subnormal positive
  /// temperatures (`(0.0, 1e-3)`) are rejected because `1/temp`
  /// overflows to `+inf` inside `apply_temperature`, poisoning
  /// softmax with NaN/Inf. NaN, infinity, and negative values are
  /// also rejected.
  pub const fn with_temperature(mut self, v: f32) -> Self {
    self.temperature = v;
    self
  }
  /// Returns a copy with the given min-p.
  pub const fn with_min_p(mut self, v: f32) -> Self {
    self.min_p = v;
    self
  }
  /// Returns a copy with the given repetition penalty.
  pub const fn with_repetition_penalty(mut self, v: f32) -> Self {
    self.repetition_penalty = v;
    self
  }
  /// Returns a copy with the given max_new_tokens cap.
  pub const fn with_max_new_tokens(mut self, v: usize) -> Self {
    self.max_new_tokens = v;
    self
  }

  /// Sets the temperature in place.
  pub fn set_temperature(&mut self, v: f32) -> &mut Self {
    self.temperature = v;
    self
  }
  /// Sets the min-p in place.
  pub fn set_min_p(&mut self, v: f32) -> &mut Self {
    self.min_p = v;
    self
  }
  /// Sets the repetition penalty in place.
  pub fn set_repetition_penalty(&mut self, v: f32) -> &mut Self {
    self.repetition_penalty = v;
    self
  }
  /// Sets the max_new_tokens cap in place.
  pub fn set_max_new_tokens(&mut self, v: usize) -> &mut Self {
    self.max_new_tokens = v;
    self
  }

  /// Validate per spec §13.2 #19. Returns `Error::InvalidRequest(reason)` on failure.
  /// `const fn` so callers can validate presets at compile time.
  ///
  /// Rejects NaN/infinite floats explicitly: NaN values pass numeric
  /// range comparisons (`NaN < 0.0` is false), then poison softmax /
  /// `partial_cmp` downstream and can panic the sampler. f32::is_nan
  /// is `const fn` since Rust 1.83.
  ///
  /// Also caps `max_new_tokens` at [`MAX_NEW_TOKENS_CAP`] (32 768) to
  /// prevent caller-controlled `Vec::with_capacity` allocations from
  /// driving the process to OOM before any model work begins. The
  /// underlying model has 128 K context, so 32 K of new tokens leaves
  /// generous headroom for prompt + image tokens.
  pub const fn validate(&self) -> Result<()> {
    if self.temperature.is_nan() || self.temperature.is_infinite() {
      return Err(Error::InvalidRequest("temperature must be finite"));
    }
    if self.temperature < 0.0 {
      return Err(Error::InvalidRequest("temperature must be >= 0.0"));
    }
    // A subnormal positive temperature (e.g., 1e-40) makes 1/temp
    // overflow to +inf inside apply_temperature; logits × inf
    // produces inf/NaN, softmax
    // returns a non-finite distribution, and sample_min_p's argmax
    // fallback (total_cmp) selects an arbitrary token — including
    // schema-disallowed ones in ConstrainedSampler. Treat
    // temperatures in (0, MIN_TEMPERATURE) as ill-conditioned;
    // callers wanting effectively-greedy should pass exactly 0.0.
    const MIN_TEMPERATURE: f32 = 1e-3;
    if self.temperature > 0.0 && self.temperature < MIN_TEMPERATURE {
      return Err(Error::InvalidRequest(
        "temperature must be either exactly 0.0 (greedy) or >= 1e-3 (1/temp would overflow for smaller positive values, poisoning softmax with NaN/inf)",
      ));
    }
    if self.min_p.is_nan() || self.min_p.is_infinite() {
      return Err(Error::InvalidRequest("min_p must be finite"));
    }
    if self.min_p < 0.0 || self.min_p > 1.0 {
      return Err(Error::InvalidRequest("min_p must be in [0.0, 1.0]"));
    }
    if self.repetition_penalty.is_nan() || self.repetition_penalty.is_infinite() {
      return Err(Error::InvalidRequest("repetition_penalty must be finite"));
    }
    if self.repetition_penalty < 1.0 {
      return Err(Error::InvalidRequest("repetition_penalty must be >= 1.0"));
    }
    // An unbounded repetition_penalty (e.g., f32::MAX) multiplied
    // against a typical negative logit
    // (e.g., -2.0) overflows to -inf. If the seen-token set covers
    // every still-finite logit, the post-penalty logits become
    // all-non-finite; sample_min_p's argmax fallback then returns
    // an arbitrary token (often 0/PAD), which can violate a
    // ConstrainedSampler mask. Cap penalty at a value far above
    // any realistic use (the LFM2.5-VL card recommends 1.05) but
    // small enough that overflow can't reach -inf for any valid
    // logit (typical model logit range is roughly [-50, 50]).
    if self.repetition_penalty > MAX_REPETITION_PENALTY {
      return Err(Error::InvalidRequest(
        "repetition_penalty must be <= 100.0 (penalty × negative logit could otherwise overflow to -inf and poison sampling)",
      ));
    }
    if self.max_new_tokens == 0 {
      return Err(Error::InvalidRequest("max_new_tokens must be > 0"));
    }
    if self.max_new_tokens > MAX_NEW_TOKENS_CAP {
      return Err(Error::InvalidRequest(
        "max_new_tokens must be <= 32768 (model context is 128K; this leaves headroom for prompt + image tokens and prevents OOM from oversized preallocation)",
      ));
    }
    Ok(())
  }
}

/// Hard upper bound on `RequestOptions::max_new_tokens`. The decode
/// loop preallocates `Vec::with_capacity(max_new_tokens)`; without this
/// cap, a misconfigured caller could drive a `usize::MAX` allocation
/// to OOM before any model work begins. 32 768 is generous (the model
/// has 128 K context) and bounds the output_ids allocation at ~128 KB.
pub const MAX_NEW_TOKENS_CAP: usize = 32_768;

/// Hard upper bound on `RequestOptions::repetition_penalty`. With
/// no upper bound, a request could set
/// `repetition_penalty = f32::MAX`, which multiplied against any
/// negative seen-token logit immediately overflows to -inf. If the
/// model emits multiple negative logits (the common case), the
/// post-penalty logit set can become all-non-finite, making
/// `sample_min_p`'s `total_cmp` argmax pick an arbitrary token —
/// including ones masked by a `ConstrainedSampler`. 100.0 is far
/// above any realistic use (the model card recommends 1.05) and
/// safely below the f32 overflow threshold for typical logit
/// magnitudes (a logit of −10 × 100 = −1000 is well finite).
pub const MAX_REPETITION_PENALTY: f32 = 100.0;

/// Maximum total context length supported by the model. Sourced from
/// the bundled `models/config.json` field `max_position_embeddings`.
/// `generate()` enforces `prompt_tokens + max_new_tokens <=
/// MODEL_CONTEXT_TOKENS` after tokenization (and before embedding /
/// decoder prefill) so an over-sized request fails fast instead of
/// running the model past its valid position-embedding range.
pub const MODEL_CONTEXT_TOKENS: usize = 128_000;

/// Returns [`Self::deterministic`], **not** [`Self::new`]. This is the value
/// an absent `request` table fills from ([`Options::new`] carries
/// `RequestOptions::deterministic()`, not `RequestOptions::new()`), and
/// `#[serde(default)]` on this struct reuses this same impl so a partial
/// table fills its missing keys with the identical value — one source of
/// truth for both the whole-table-absent and the some-keys-absent case.
/// `new()` is unaffected and stays the explicit model-card constructor.
impl Default for RequestOptions {
  fn default() -> Self {
    Self::deterministic()
  }
}

// =========================================================================
// ImageBudget
// =========================================================================

/// Per-image preprocessing budget. Note: `max_image_tokens` is **asymmetric
/// across paths** — it bounds the single-tile path's `smart_resize` and
/// the thumbnail's `smart_resize`, but does NOT bound the multi-tile
/// path's main-tile total (which is `rows × cols × 256`, capped only by
/// `max_tiles`). See spec §13.3 #14 for the full discussion.
// `Eq` dropped because `max_pixels_tolerance: f32` can't be Eq.
// ImageBudget isn't used as a HashMap/HashSet key anywhere in
// the workspace, so PartialEq alone is enough. Storing the
// tolerance as `(v * 100.0) as u32` would truncate cooperative-
// caller inputs like `2.067 → 2.06`, silently routing 723x724
// images to the multi-tile path when upstream Python's float
// threshold would have kept them
// single-tile — a real algorithmic-parity break.
//
// `#[serde(default)]`: a partial `image_budget` table fills whatever key it
// leaves out from `Self::default()`, which is `Self::new()` — the same value
// an absent `image_budget` table already got from `Options::new`. Unlike
// `RequestOptions`, `Default` and `new()` do not diverge here: there is only
// one candidate default, so `#[serde(default)]` only widens what a partial
// table accepts and changes no value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct ImageBudget {
  min_image_tokens: usize,
  max_image_tokens: usize,
  min_tiles: usize,
  max_tiles: usize,
  use_thumbnail: bool,
  /// Stored as `f32` to preserve the caller's tolerance bit-for-bit
  /// — upstream Python's `_is_image_too_large` uses an unrestricted
  /// float multiply, so any rounding here can flip the
  /// single-tile vs multi-tile routing on edge-case dimensions.
  max_pixels_tolerance: f32,
}

impl ImageBudget {
  /// `preprocessor_config.json` defaults: min=64 tokens, max=256 tokens,
  /// min=2 tiles, max=10 tiles, thumbnail on, max_pixels_tolerance=2.0.
  pub const fn new() -> Self {
    Self {
      min_image_tokens: 64,
      max_image_tokens: 256,
      min_tiles: 2,
      max_tiles: 10,
      use_thumbnail: true,
      max_pixels_tolerance: 2.0,
    }
  }

  /// Speed-optimized: `max_image_tokens=64`, `max_tiles=4`, no thumbnail.
  /// ~3-4× speedup at lower per-frame quality.
  pub const fn fast() -> Self {
    Self {
      min_image_tokens: 32,
      max_image_tokens: 64,
      min_tiles: 2,
      max_tiles: 4,
      use_thumbnail: false,
      max_pixels_tolerance: 2.0,
    }
  }

  /// Quality-optimized — currently identical to `new()`; kept as a
  /// named preset so future config changes don't silently re-tune the
  /// "I want best quality" call site.
  pub const fn quality() -> Self {
    Self::new()
  }

  /// Returns the minimum number of image tokens.
  pub const fn min_image_tokens(&self) -> usize {
    self.min_image_tokens
  }
  /// Returns the maximum number of image tokens.
  pub const fn max_image_tokens(&self) -> usize {
    self.max_image_tokens
  }
  /// Returns the minimum number of tiles.
  pub const fn min_tiles(&self) -> usize {
    self.min_tiles
  }
  /// Returns the maximum number of tiles.
  pub const fn max_tiles(&self) -> usize {
    self.max_tiles
  }
  /// Returns whether to include the thumbnail tile.
  pub const fn use_thumbnail(&self) -> bool {
    self.use_thumbnail
  }
  /// Returns the max-pixels tolerance factor (e.g. `2.0` means ≤2× over budget is acceptable).
  pub const fn max_pixels_tolerance(&self) -> f32 {
    self.max_pixels_tolerance
  }

  /// Conservative upper bound on the number of `<image>` tokens a
  /// SINGLE preprocessed image can contribute to the prompt under
  /// this budget. Used by `generate()` for pre-decode admission
  /// control: rejecting requests whose `images.len() *
  /// max_tokens_per_image() + max_new_tokens` already exceeds the
  /// model context, *before* paying for image decode + smart_resize +
  /// flatten_to_patches.
  ///
  /// Math:
  /// - Multi-tile path: `rows * cols * tokens_per_main_tile +
  ///   thumbnail_tokens`. `rows * cols ≤ max_tiles` and
  ///   `tokens_per_main_tile = (FULL_TILE_SIZE / TILE_PIXEL_UNIT)² =
  ///   (512/32)² = 256`. Thumbnail is bounded by `max_image_tokens`
  ///   (hard cap). Worst case: `max_tiles × 256 +
  ///   max_image_tokens`.
  /// - Single-tile path: bounded by `max_image_tokens` only.
  ///
  /// `max_tiles × 256 + max_image_tokens` is the strict upper bound
  /// of both paths. For the default budget (max_tiles=10,
  /// max_image_tokens=256): 2560 + 256 = 2816 tokens per image.
  pub const fn max_tokens_per_image(&self) -> usize {
    // tokens_per_main_tile = (FULL_TILE_SIZE / TILE_PIXEL_UNIT)² =
    // (512/32)² = 256. Hardcoded since both constants are model-fixed
    // and live in src/preproc/tile_grid.rs (not visible from here
    // without an awkward cross-module import).
    const TOKENS_PER_FULL_TILE: usize = 256;
    self.max_tiles * TOKENS_PER_FULL_TILE + self.max_image_tokens
  }

  /// Returns a copy with the given min_image_tokens.
  pub const fn with_min_image_tokens(mut self, v: usize) -> Self {
    self.min_image_tokens = v;
    self
  }
  /// Returns a copy with the given max_image_tokens.
  pub const fn with_max_image_tokens(mut self, v: usize) -> Self {
    self.max_image_tokens = v;
    self
  }
  /// Returns a copy with the given min_tiles.
  pub const fn with_min_tiles(mut self, v: usize) -> Self {
    self.min_tiles = v;
    self
  }
  /// Returns a copy with the given max_tiles.
  pub const fn with_max_tiles(mut self, v: usize) -> Self {
    self.max_tiles = v;
    self
  }
  /// Returns a copy with the given use_thumbnail flag.
  pub const fn with_use_thumbnail(mut self, v: bool) -> Self {
    self.use_thumbnail = v;
    self
  }
  /// Returns a copy with the given max_pixels_tolerance. Stored as
  /// `f32` exactly — no rounding to hundredths. (Earlier versions
  /// stored this as `(v * 100) as u32` to keep `Eq` working, but
  /// that truncated cooperative-caller inputs like `2.067 → 2.06`
  /// and silently mis-routed images near the multi-tile threshold.
  /// `ImageBudget` now uses `PartialEq` only, so the f32 storage
  /// is direct.) Use [`ImageBudget::validate`] to reject NaN/Inf
  /// before passing to the preprocessor.
  pub fn with_max_pixels_tolerance(mut self, v: f32) -> Self {
    self.max_pixels_tolerance = v;
    self
  }

  /// Sets the min_image_tokens in place.
  pub fn set_min_image_tokens(&mut self, v: usize) -> &mut Self {
    self.min_image_tokens = v;
    self
  }
  /// Sets the max_image_tokens in place.
  pub fn set_max_image_tokens(&mut self, v: usize) -> &mut Self {
    self.max_image_tokens = v;
    self
  }
  /// Sets the min_tiles in place.
  pub fn set_min_tiles(&mut self, v: usize) -> &mut Self {
    self.min_tiles = v;
    self
  }
  /// Sets the max_tiles in place.
  pub fn set_max_tiles(&mut self, v: usize) -> &mut Self {
    self.max_tiles = v;
    self
  }
  /// Sets the use_thumbnail flag in place.
  pub fn set_use_thumbnail(&mut self, v: bool) -> &mut Self {
    self.use_thumbnail = v;
    self
  }
  /// Sets max_pixels_tolerance in place. See [`Self::with_max_pixels_tolerance`].
  pub fn set_max_pixels_tolerance(&mut self, v: f32) -> &mut Self {
    self.max_pixels_tolerance = v;
    self
  }

  /// Validate per spec §13.2 #19. Returns `Error::InvalidBudget(reason)` on failure.
  /// `const fn` so callers can validate presets at compile time.
  pub const fn validate(&self) -> Result<()> {
    if self.min_image_tokens == 0 {
      return Err(Error::InvalidBudget("min_image_tokens must be > 0"));
    }
    if self.max_image_tokens < self.min_image_tokens {
      return Err(Error::InvalidBudget(
        "max_image_tokens must be >= min_image_tokens",
      ));
    }
    if self.min_tiles == 0 {
      return Err(Error::InvalidBudget("min_tiles must be > 0"));
    }
    if self.max_tiles < self.min_tiles {
      return Err(Error::InvalidBudget("max_tiles must be >= min_tiles"));
    }
    // The tokenizer ships with `<|img_row_R_col_C|>` markers only for
    // R, C ∈ [1, MAX_TOKENIZER_TILE_DIM]. A budget that produces a
    // grid dimension above this limit would emit markers that
    // tokenize as ordinary text (silent corruption of position-token
    // embeddings). Since `max_tiles` is the upper bound for both
    // grid_width and grid_height in `find_closest_aspect_ratio`,
    // capping `max_tiles` here is sufficient.
    if self.max_tiles > MAX_TOKENIZER_TILE_DIM {
      return Err(Error::InvalidBudget(
        "max_tiles must be <= 10 (bundled tokenizer's row/col marker grid is 10x10)",
      ));
    }
    // Cap min_image_tokens / max_image_tokens at MAX_IMAGE_TOKENS_CAP.
    // smart_resize derives its pixel budget from these (pixels =
    // tokens * 16² * 2² = tokens * 1024); without a cap, a
    // caller-controlled `with_max_image_tokens(usize::MAX)` would
    // skip the shrink branch and let any input through, then
    // flatten_to_patches would allocate a huge pixel_values tensor.
    // 1024 tokens = ~1024×1024 pixels — 4× the 256-token default and
    // generous headroom for any legitimate use.
    if self.max_image_tokens > MAX_IMAGE_TOKENS_CAP {
      return Err(Error::InvalidBudget(
        "max_image_tokens must be <= 1024 (4× the model default; protects against unbounded smart_resize / pixel_values allocation)",
      ));
    }
    if !self.max_pixels_tolerance.is_finite() || self.max_pixels_tolerance <= 0.0 {
      return Err(Error::InvalidBudget(
        "max_pixels_tolerance must be a finite, positive f32 (NaN/Inf/<=0 reject)",
      ));
    }
    Ok(())
  }
}

/// The bundled `tokenizer.json` ships `<|img_row_R_col_C|>` markers
/// for R, C ∈ [1, 10]. Any dimension above this would tokenize as
/// ordinary text rather than as a single position-marker token.
pub const MAX_TOKENIZER_TILE_DIM: usize = 10;

/// Hard upper bound on `ImageBudget::max_image_tokens` (and therefore
/// also `min_image_tokens` because validate enforces
/// `max_image_tokens >= min_image_tokens`). `smart_resize` derives a
/// pixel budget from this value (pixels = tokens × 16² × 2² =
/// tokens × 1024); 1024 tokens ≈ 1024×1024 pixels of image input,
/// which is 4× the 256-token model default. Above this, the
/// `pixel_values` allocation in `flatten_to_patches` becomes a
/// memory-DoS vector.
pub const MAX_IMAGE_TOKENS_CAP: usize = 1024;

impl Default for ImageBudget {
  fn default() -> Self {
    Self::new()
  }
}

// =========================================================================
// ThreadOptions
// =========================================================================

/// ORT thread configuration. Mirrors siglip2/egemma `ThreadOptions`.
///
/// `deny_unknown_fields` because this is a **nested** table of the [`Options`]
/// document and both its fields are `Option`: without it,
/// `{ "backend": "onnx", "thread": { "intra_thread": 1 } }` would deserialize
/// happily with both counts left at `None`, silently handing back ORT's
/// defaults for a misspelled determinism control. Nesting is why the enclosing
/// tier's own `deny_unknown_fields` cannot catch this — it does not recurse.
///
/// The counts are `u16` rather than `usize` so that an out-of-range request is
/// refused by serde's own integer range check, at the field, in every format —
/// not by a ceiling constant this crate would have to invent and enforce. See
/// the fields for why that is the honest bound.
///
/// `#[serde(default)]`: a partial `thread` table fills whatever key it leaves
/// out from `Self::default()`, which is `Self::new()` — `None`/`None`, the
/// same value an absent `thread` table already got from `OrtOptions::new`.
/// Both fields are `Option`, so a correctly-spelled-but-omitted key already
/// deserialized to `None` without this attribute; the attribute makes that
/// contract explicit at the container level instead of leaving it an
/// unstated consequence of every field happening to be `Option`, and is what
/// a fully-empty `{}` table (or an entirely absent one reached some other
/// way) resolves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct ThreadOptions {
  /// Both counts are `u16` so that "a value ORT can actually take" is a
  /// property of the type rather than of a validation hook someone can forget
  /// to call. `ort` forwards a thread count to the C API's signed `int`, so a
  /// `usize` above `i32::MAX` wraps silently and a merely large one asks for a
  /// per-session pool big enough to exhaust the process — and an `Engine`
  /// builds three sessions. `u16` makes both unrepresentable: every value
  /// converts to `i32`/`usize` losslessly and infallibly, so the session seam
  /// needs no cast and no `try_from`. The 65 535 ceiling costs no documented
  /// constant to justify — it is already far beyond any pool a session should
  /// ask for, on any machine this crate runs on.
  intra_threads: Option<u16>,
  inter_threads: Option<u16>,
}

impl ThreadOptions {
  /// `None`/`None` = let ort pick defaults.
  pub const fn new() -> Self {
    Self {
      intra_threads: None,
      inter_threads: None,
    }
  }

  /// Indexing-safe single-threaded — pair with `RequestOptions::deterministic()`
  /// for end-to-end bit-stability.
  pub const fn deterministic() -> Self {
    Self {
      intra_threads: Some(1),
      inter_threads: Some(1),
    }
  }

  /// Returns the intra-op thread count (None = ort default).
  pub const fn intra_threads(&self) -> Option<u16> {
    self.intra_threads
  }
  /// Returns the inter-op thread count (None = ort default). See
  /// [`Self::with_inter_threads`] for what a count above 1 implies.
  pub const fn inter_threads(&self) -> Option<u16> {
    self.inter_threads
  }

  /// Whether this table asks for ort's **parallel** execution mode.
  ///
  /// ort's inter-op thread pool exists only in parallel execution mode: in the
  /// default sequential mode `SetInterOpNumThreads` is accepted and ignored, so
  /// a session built without this would take an `inter_threads` setting and
  /// silently run single-graph-threaded anyway. Asking for more than one
  /// inter-op thread is therefore also asking for parallel execution, and
  /// `build_session` reads exactly this predicate.
  ///
  /// `None` and `Some(1)` are false: they keep the sequential mode, so
  /// [`Self::deterministic`] (intra 1 / inter 1) and every other existing
  /// recipe build byte-identically to before.
  pub const fn requires_parallel_execution(&self) -> bool {
    match self.inter_threads {
      Some(n) => n > 1,
      None => false,
    }
  }

  /// Returns a copy with the given intra-op thread count.
  pub const fn with_intra_threads(mut self, v: u16) -> Self {
    self.intra_threads = Some(v);
    self
  }
  /// Returns a copy with the given inter-op thread count.
  ///
  /// **This knob also selects an execution mode.** `inter_threads > 1` turns on
  /// ort's parallel execution mode, because that is the only mode in which an
  /// inter-op thread pool exists; `1` or unset keeps the sequential mode, where
  /// the knob has no meaning and ort ignores it. Parallel execution trades
  /// higher memory use for concurrency across independent graph branches, and
  /// it is not bit-stable — pair [`Self::deterministic`] with
  /// [`RequestOptions::deterministic`] when reproducibility matters.
  pub const fn with_inter_threads(mut self, v: u16) -> Self {
    self.inter_threads = Some(v);
    self
  }

  /// Sets the intra-op thread count in place.
  pub fn set_intra_threads(&mut self, v: u16) -> &mut Self {
    self.intra_threads = Some(v);
    self
  }
  /// Sets the inter-op thread count in place. See
  /// [`Self::with_inter_threads`] for the execution mode a count above 1
  /// implies.
  pub fn set_inter_threads(&mut self, v: u16) -> &mut Self {
    self.inter_threads = Some(v);
    self
  }
}

impl Default for ThreadOptions {
  fn default() -> Self {
    Self::new()
  }
}

// =========================================================================
// BackendKind
// =========================================================================

/// Which inference backend an [`Engine`](crate::Engine) runs on.
///
/// The set is closed and framework-owned: `lfm` compiles the ONNX Runtime
/// (`ort`) backend by default only on the targets `ort-sys` ships prebuilt
/// binaries for that this crate supports (Linux x86_64/aarch64-gnu, Windows
/// x86_64-msvc). On macOS/arm64, MLX (`mlxrs`) is the native road instead and
/// `ort` becomes opt-in behind the `ort` feature; every other target compiles
/// neither (see `Cargo.toml`). The MLX (`mlxrs`) Metal backend itself is
/// compiled **only** on macOS/arm64. It is `#[non_exhaustive]` so adding a
/// third backend later is not a SemVer break.
///
/// The type is used in both directions:
///
/// - as a **request** — [`Options::backend_kind`] reports the pin the
///   document's engine tier carries, and that pin is what
///   [`Engine::from_dir`](crate::Engine::from_dir) must select instead of
///   letting the checkpoint layout decide. A pin that cannot be honoured
///   (the directory holds the other format, or the platform has no such
///   backend) is [`Error::BackendUnavailable`],
///   never a silent substitution. Which pins are even spellable is the
///   [`BackendOptions`] roster's business: a backend this build did not
///   compile has no variant to name it with;
/// - as an **answer** — [`Engine::backend`](crate::Engine::backend) reports
///   which backend the constructed engine actually runs on, so auto-selection
///   is observable rather than opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// One vocabulary: the wire form is the same lower-case name `as_str` reports
// and the same one `BackendOptions` uses as its tag, so a backend is spelled
// identically in a document, in a diagnostic, and in `Engine::backend`.
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[non_exhaustive]
pub enum BackendKind {
  /// The ONNX Runtime (`ort`) backend. Reads the `onnx/*.onnx` graphs;
  /// available on every target `lfm` builds for.
  Onnx,
  /// The MLX (`mlxrs`) Metal backend. Reads an MLX checkpoint (`config.json`
  /// plus a safetensors / gguf / npz weight set); compiled only on
  /// macOS/arm64.
  Mlx,
}

impl BackendKind {
  /// The backend's lower-case name, as it appears in diagnostics.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Onnx => "onnx",
      Self::Mlx => "mlx",
    }
  }
}

impl core::fmt::Display for BackendKind {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.as_str())
  }
}

// =========================================================================
// Backend tiers: the per-engine knob structs
// =========================================================================

/// The knobs the auto-selected road carries: none.
///
/// `backend = "auto"` names no engine, so there is no engine-specific knob to
/// set — [`Engine::from_dir`](crate::Engine::from_dir) reads the checkpoint
/// layout and decides.
///
/// This is a zero-field **struct** rather than a bare unit variant on purpose.
/// Serde's internally tagged representation lets a unit variant absorb every
/// remaining key without complaint, so `{ "backend": "auto", "intra_threads":
/// 1 }` would deserialize happily and drop the misplaced key on the floor. A
/// struct carrying `deny_unknown_fields` refuses it by name instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct AutoOptions {}

impl AutoOptions {
  /// The only value this type has.
  pub const fn new() -> Self {
    Self {}
  }
}

impl Default for AutoOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// The ONNX Runtime road's own knobs.
///
/// Compiled with the ORT backend itself: `ort` is a mandatory dependency only
/// on the targets `ort-sys` ships prebuilt binaries for that this crate
/// defaults it on (Linux x86_64/aarch64-gnu, Windows x86_64/aarch64-msvc); on
/// aarch64-apple-darwin, MLX is the native road and `ort` is opt-in behind the `ort`
/// feature; every other target compiles neither (see `Cargo.toml`'s target
/// tables and the `ort_backend` cfg `build.rs` emits). On a build without it,
/// `onnx` is not in [`BackendOptions`]'s roster and a document naming it is
/// refused.
///
/// Both knobs are ORT's alone — the MLX road runs on Metal and has no CPU
/// thread budget or graph-optimization pass — which is why they live in this
/// tier rather than beside the sampler and the image budget. Setting them
/// therefore means naming the ONNX road: under `backend = "auto"` the ORT
/// session is built from [`OrtOptions::new`].
#[cfg(all(feature = "inference", ort_backend))]
#[cfg_attr(
  docsrs,
  doc(cfg(any(
    all(
      target_arch = "x86_64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "x86_64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(target_os = "macos", target_arch = "aarch64", feature = "ort")
  )))
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OrtOptions {
  thread: ThreadOptions,
  optimization_level: GraphOptLevelMirror,
}

#[cfg(all(feature = "inference", ort_backend))]
impl OrtOptions {
  /// Defaults: [`ThreadOptions::new`] (ort picks its own thread counts) and
  /// `GraphOptimizationLevel::Level1` — matching siglip2/egemma, because
  /// higher levels can subtly alter numerics.
  pub const fn new() -> Self {
    Self {
      thread: ThreadOptions::new(),
      optimization_level: GraphOptLevelMirror::Level1,
    }
  }

  /// Indexing-safe single-threaded ORT: [`ThreadOptions::deterministic`] at
  /// the same `Level1` optimization. Pair with
  /// [`RequestOptions::deterministic`] for end-to-end bit-stability (a
  /// CPU-only execution provider is the third requirement).
  pub const fn deterministic() -> Self {
    Self {
      thread: ThreadOptions::deterministic(),
      optimization_level: GraphOptLevelMirror::Level1,
    }
  }

  /// Returns a reference to the ORT thread configuration.
  pub const fn thread(&self) -> &ThreadOptions {
    &self.thread
  }

  /// Returns the ORT graph optimization level.
  pub fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level.into()
  }

  /// Returns a copy with the given thread configuration.
  pub const fn with_thread(mut self, t: ThreadOptions) -> Self {
    self.thread = t;
    self
  }

  /// Returns a copy with the given ORT graph optimization level.
  pub fn with_optimization_level(mut self, lvl: GraphOptimizationLevel) -> Self {
    self.optimization_level = lvl.into();
    self
  }

  /// Sets the thread configuration in place.
  pub fn set_thread(&mut self, t: ThreadOptions) -> &mut Self {
    self.thread = t;
    self
  }

  /// Sets the ORT graph optimization level in place.
  pub fn set_optimization_level(&mut self, lvl: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = lvl.into();
    self
  }
}

#[cfg(all(feature = "inference", ort_backend))]
impl Default for OrtOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// The MLX (Metal) road's own knobs: none yet.
///
/// Compiled only on macOS/arm64, where the `mlxrs` target dependency exists;
/// elsewhere `mlx` is not in [`BackendOptions`]'s roster and a document naming
/// it is refused by that roster.
///
/// The MLX backend takes its model dimensions, tiling and quantization scheme
/// from the checkpoint's own `config.json`, so it has no knob to expose today.
/// The tier is here anyway: it is the seat a future MLX knob lands in without
/// reshaping the document, and — being a struct with `deny_unknown_fields`
/// rather than a unit variant — it is what refuses an ORT knob (or a
/// misspelled shared one) written beside `backend = "mlx"`.
#[cfg(all(feature = "inference", mlx_backend))]
#[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct MlxOptions {}

#[cfg(all(feature = "inference", mlx_backend))]
impl MlxOptions {
  /// The only value this type has.
  pub const fn new() -> Self {
    Self {}
  }
}

#[cfg(all(feature = "inference", mlx_backend))]
impl Default for MlxOptions {
  fn default() -> Self {
    Self::new()
  }
}

// =========================================================================
// BackendOptions — the engine tier of the document
// =========================================================================

/// Which road runs, together with that road's own knobs — the engine tier of
/// the [`Options`] document, flattened into it.
///
/// Serialized as an **internally tagged** enum under the key `backend`, so one
/// key both names the road and selects the knob struct that road owns:
///
/// ```text
/// backend = "auto"                     # AutoOptions   — no engine knobs
/// backend = "onnx"                     # OrtOptions    — thread, optimization_level
/// backend = "mlx"                      # MlxOptions    — none yet
/// ```
///
/// The variants are `cfg`-gated on their backend's presence, so the roster
/// serde reports in `unknown variant ...` is exactly the set of roads this
/// build can run. `#[non_exhaustive]` because that set is platform- and
/// feature-dependent (and because a third backend later must not be a SemVer
/// break): match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "backend", rename_all = "lowercase"))]
#[non_exhaustive]
pub enum BackendOptions {
  /// Let [`Engine::from_dir`](crate::Engine::from_dir) choose from the
  /// checkpoint layout. The default.
  Auto(AutoOptions),
  /// Pin the ONNX Runtime road and carry its knobs.
  #[cfg(all(feature = "inference", ort_backend))]
  #[cfg_attr(
    docsrs,
    doc(cfg(any(
      all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "x86_64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(target_os = "macos", target_arch = "aarch64", feature = "ort")
    )))
  )]
  Onnx(OrtOptions),
  /// Pin the MLX (Metal) road and carry its knobs.
  #[cfg(all(feature = "inference", mlx_backend))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  Mlx(MlxOptions),
}

impl BackendOptions {
  /// Auto-selection: the checkpoint layout decides. The default.
  pub const fn auto() -> Self {
    Self::Auto(AutoOptions::new())
  }

  /// Pin the ONNX Runtime road with the given knobs.
  #[cfg(all(feature = "inference", ort_backend))]
  #[cfg_attr(
    docsrs,
    doc(cfg(any(
      all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "x86_64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(target_os = "macos", target_arch = "aarch64", feature = "ort")
    )))
  )]
  pub const fn onnx(opts: OrtOptions) -> Self {
    Self::Onnx(opts)
  }

  /// Pin the MLX (Metal) road with the given knobs.
  #[cfg(all(feature = "inference", mlx_backend))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub const fn mlx(opts: MlxOptions) -> Self {
    Self::Mlx(opts)
  }

  /// The pinned backend, or `None` when the road is left to the checkpoint
  /// layout.
  ///
  /// A pin that cannot be served — the directory holds only the other
  /// format's checkpoint — is
  /// [`Error::BackendUnavailable`], never a
  /// silent substitution. A pin on a road this build does not compile cannot
  /// be spelled at all: the variant is not there.
  pub const fn kind(&self) -> Option<BackendKind> {
    match self {
      Self::Auto(_) => None,
      #[cfg(all(feature = "inference", ort_backend))]
      Self::Onnx(_) => Some(BackendKind::Onnx),
      #[cfg(all(feature = "inference", mlx_backend))]
      Self::Mlx(_) => Some(BackendKind::Mlx),
    }
  }

  /// The ORT knobs this tier carries, or `None` when it does not name the
  /// ONNX road.
  #[cfg(all(feature = "inference", ort_backend))]
  #[cfg_attr(
    docsrs,
    doc(cfg(any(
      all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "x86_64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(target_os = "macos", target_arch = "aarch64", feature = "ort")
    )))
  )]
  pub const fn ort_options(&self) -> Option<&OrtOptions> {
    match self {
      Self::Onnx(o) => Some(o),
      _ => None,
    }
  }

  /// The MLX knobs this tier carries, or `None` when it does not name the MLX
  /// road.
  #[cfg(all(feature = "inference", mlx_backend))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub const fn mlx_options(&self) -> Option<&MlxOptions> {
    match self {
      Self::Mlx(m) => Some(m),
      _ => None,
    }
  }
}

impl Default for BackendOptions {
  fn default() -> Self {
    Self::auto()
  }
}

// =========================================================================
// Options (top-level)
// =========================================================================

/// Top-level engine configuration: the shared tier, plus the engine tier
/// flattened into the same document.
///
/// See the [module note](crate::options) for the document's shape, its
/// defaulting rules and the exact text of each refusal.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// Every tier defaults, so a partial document fills the rest from `new()`
// instead of being refused by the first missing field's name. The flatten
// below is why this struct cannot also carry `deny_unknown_fields` (serde
// forbids the pair); an unmatched key is refused by the engine tier's own
// `deny_unknown_fields` instead — see the module note.
#[cfg_attr(feature = "serde", serde(default))]
pub struct Options {
  request: RequestOptions,
  image_budget: ImageBudget,
  /// The engine tier. Flattened, so its `backend` tag and the selected
  /// road's knobs sit at the document's top level beside the shared ones.
  #[cfg_attr(feature = "serde", serde(flatten))]
  backend: BackendOptions,
}

impl Options {
  /// Defaults: `RequestOptions::deterministic()`, `ImageBudget::new()`, and
  /// [`BackendOptions::auto`] — the checkpoint layout picks the road, and
  /// that road runs on its own tier's defaults.
  pub const fn new() -> Self {
    Self {
      request: RequestOptions::deterministic(),
      image_budget: ImageBudget::new(),
      backend: BackendOptions::auto(),
    }
  }

  /// Returns a reference to the sampler configuration.
  pub const fn request(&self) -> &RequestOptions {
    &self.request
  }
  /// Returns a reference to the image preprocessing budget.
  pub const fn image_budget(&self) -> &ImageBudget {
    &self.image_budget
  }

  /// Returns a reference to the engine tier: the road and its own knobs.
  pub const fn backend(&self) -> &BackendOptions {
    &self.backend
  }

  /// The explicit backend pin, or `None` when the backend is auto-selected
  /// from the checkpoint layout (the default).
  pub const fn backend_kind(&self) -> Option<BackendKind> {
    self.backend.kind()
  }

  /// The ORT knobs this configuration runs the ONNX road with: the ones the
  /// engine tier carries when it names `onnx`, and [`OrtOptions::new`]
  /// otherwise — a document that leaves the road to the checkpoint layout
  /// cannot have tuned a road it did not name.
  #[cfg(all(feature = "inference", ort_backend))]
  #[cfg_attr(
    docsrs,
    doc(cfg(any(
      all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu"
      ),
      all(
        target_arch = "x86_64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(
        target_arch = "aarch64",
        target_vendor = "pc",
        target_os = "windows",
        target_env = "msvc"
      ),
      all(target_os = "macos", target_arch = "aarch64", feature = "ort")
    )))
  )]
  pub const fn effective_ort_options(&self) -> OrtOptions {
    match self.backend.ort_options() {
      Some(o) => *o,
      None => OrtOptions::new(),
    }
  }

  /// Returns a copy with the given sampler configuration.
  pub const fn with_request(mut self, r: RequestOptions) -> Self {
    self.request = r;
    self
  }
  /// Returns a copy with the given image budget.
  pub const fn with_image_budget(mut self, b: ImageBudget) -> Self {
    self.image_budget = b;
    self
  }

  /// Returns a copy carrying the given engine tier.
  ///
  /// This is how a road is pinned:
  /// `Options::new().with_backend(BackendOptions::mlx(MlxOptions::new()))`.
  /// A pin that cannot be served — the directory holds only the other
  /// format's checkpoint — is
  /// [`Error::BackendUnavailable`]. A pin is
  /// also how a directory that carries BOTH an ONNX graph set and an MLX
  /// weight set is resolved deliberately instead of by the default
  /// ONNX-graph-wins rule.
  pub const fn with_backend(mut self, backend: BackendOptions) -> Self {
    self.backend = backend;
    self
  }

  /// Returns a copy with the backend pin cleared (back to auto-selection).
  pub const fn with_auto_backend(mut self) -> Self {
    self.backend = BackendOptions::auto();
    self
  }

  /// Sets the request-options sub-config in place.
  pub fn set_request(&mut self, r: RequestOptions) -> &mut Self {
    self.request = r;
    self
  }
  /// Sets the image-budget sub-config in place.
  pub fn set_image_budget(&mut self, b: ImageBudget) -> &mut Self {
    self.image_budget = b;
    self
  }

  /// Sets the engine tier in place.
  pub fn set_backend(&mut self, backend: BackendOptions) -> &mut Self {
    self.backend = backend;
    self
  }
}

impl Default for Options {
  fn default() -> Self {
    Self::new()
  }
}

// =========================================================================
// GraphOptLevelMirror — serde-friendly mirror enum
// =========================================================================

/// Serde-friendly mirror of [`GraphOptimizationLevel`] (which doesn't
/// derive `Serialize`/`Deserialize` directly). Mirrors the siglip2/egemma
/// pattern.
#[cfg(all(feature = "inference", ort_backend))]
#[cfg_attr(
  docsrs,
  doc(cfg(any(
    all(
      target_arch = "x86_64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "unknown",
      target_os = "linux",
      target_env = "gnu"
    ),
    all(
      target_arch = "x86_64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(
      target_arch = "aarch64",
      target_vendor = "pc",
      target_os = "windows",
      target_env = "msvc"
    ),
    all(target_os = "macos", target_arch = "aarch64", feature = "ort")
  )))
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
enum GraphOptLevelMirror {
  /// Disable all optimizations.
  Disable,
  /// Basic optimizations only (default).
  Level1,
  /// Extended optimizations.
  Level2,
  /// Full layout optimization.
  Level3,
  /// All optimizations.
  All,
}

#[cfg(all(feature = "inference", ort_backend))]
impl From<GraphOptimizationLevel> for GraphOptLevelMirror {
  fn from(v: GraphOptimizationLevel) -> Self {
    match v {
      GraphOptimizationLevel::Disable => Self::Disable,
      GraphOptimizationLevel::Level1 => Self::Level1,
      GraphOptimizationLevel::Level2 => Self::Level2,
      GraphOptimizationLevel::Level3 => Self::Level3,
      GraphOptimizationLevel::All => Self::All,
      // `ort`'s `GraphOptimizationLevel` is `#[non_exhaustive]` (since
      // ort 2.0.0-rc.13): a match on it must handle variants a future
      // ort release could add. There is no honest silent fallback here
      // — mapping an unknown level to e.g. `Level3` would misreport the
      // optimization level this crate's numerics/bit-stability docs
      // promise (see `Options::optimization_level` / the deterministic
      // preset notes). Panic loudly instead of guessing; this can only
      // fire once `ort` ships a 6th variant, at which point
      // `GraphOptLevelMirror` needs a matching new variant added.
      other => unreachable!(
        "ort::GraphOptimizationLevel gained a variant ({other:?}) that GraphOptLevelMirror \
         does not mirror yet — ort is #[non_exhaustive] here; add the matching variant to \
         GraphOptLevelMirror"
      ),
    }
  }
}

#[cfg(all(feature = "inference", ort_backend))]
impl From<GraphOptLevelMirror> for GraphOptimizationLevel {
  fn from(v: GraphOptLevelMirror) -> Self {
    match v {
      GraphOptLevelMirror::Disable => Self::Disable,
      GraphOptLevelMirror::Level1 => Self::Level1,
      GraphOptLevelMirror::Level2 => Self::Level2,
      GraphOptLevelMirror::Level3 => Self::Level3,
      GraphOptLevelMirror::All => Self::All,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `backend` is a required key of the document. Serde's flatten cannot
  /// supply a missing tag for an internally tagged enum, so the tier that
  /// names the road has no absent form — `"auto"` is how a document says
  /// "let the checkpoint layout decide". The refusal names the key.
  #[test]
  #[cfg(feature = "serde")]
  fn options_document_requires_the_backend_key() {
    let json = serde_json::to_value(Options::new()).expect("serialize Options");
    let mut map = json.as_object().expect("object").clone();
    assert_eq!(
      map.remove("backend"),
      Some(serde_json::Value::String("auto".into())),
      "the default document names the auto road"
    );
    let err = serde_json::from_value::<Options>(serde_json::Value::Object(map))
      .expect_err("a document with no backend key must be refused");
    assert!(
      err.to_string().starts_with("missing field `backend`"),
      "unexpected refusal: {err}"
    );
  }

  /// The auto road round-trips, and carries no engine knobs.
  #[test]
  #[cfg(feature = "serde")]
  fn auto_backend_round_trips() {
    let opts = Options::new();
    let json = serde_json::to_string(&opts).expect("serialize");
    let back: Options = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, opts);
    assert_eq!(back.backend_kind(), None);
    assert_eq!(*back.backend(), BackendOptions::auto());
  }

  /// A pin round-trips through serde as the backend's own lower-case name —
  /// the same one `BackendKind::as_str` reports.
  #[test]
  #[cfg(all(feature = "serde", feature = "inference", mlx_backend))]
  fn mlx_pin_round_trips() {
    let opts = Options::new().with_backend(BackendOptions::mlx(MlxOptions::new()));
    let json = serde_json::to_string(&opts).expect("serialize");
    assert!(
      json.contains(r#""backend":"mlx""#),
      "the tag is the backend's own name: {json}"
    );
    let back: Options = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.backend_kind(), Some(BackendKind::Mlx));
    assert_eq!(back, opts);
  }

  /// The ONNX tier carries its own knobs, and they survive the round-trip
  /// flattened beside the shared ones.
  #[test]
  #[cfg(all(feature = "serde", feature = "inference", ort_backend))]
  fn onnx_tier_round_trips_with_its_knobs() {
    let opts = Options::new().with_backend(BackendOptions::onnx(OrtOptions::deterministic()));
    let json = serde_json::to_string(&opts).expect("serialize");
    let back: Options = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, opts);
    let ort = back.backend().ort_options().expect("the onnx tier");
    assert_eq!(ort.thread().intra_threads(), Some(1));
    // Under `auto` the ORT road runs on its own tier's defaults, not on the
    // knobs a document never named a road to carry.
    assert_eq!(
      Options::new().effective_ort_options(),
      OrtOptions::new(),
      "auto cannot have tuned a road it did not name"
    );
    assert_eq!(back.effective_ort_options(), OrtOptions::deterministic());
  }

  // ===== RequestOptions =====

  #[test]
  fn request_options_new_matches_model_card() {
    let r = RequestOptions::new();
    assert_eq!(r.temperature(), 0.1);
    assert_eq!(r.min_p(), 0.15);
    assert_eq!(r.repetition_penalty(), 1.05);
    assert_eq!(r.max_new_tokens(), 512);
  }

  #[test]
  fn request_options_deterministic_is_greedy() {
    let r = RequestOptions::deterministic();
    assert_eq!(r.temperature(), 0.0);
    assert_eq!(r.repetition_penalty(), 1.05);
  }

  /// `Default` and `new()` are documented as differing on purpose:
  /// `Default` (and therefore `#[serde(default)]`, and an absent `request`
  /// table) is the safer `deterministic()` preset, while `new()` remains the
  /// explicit model-card constructor. Pin that they actually do differ, so
  /// this doesn't silently become a distinction without a difference.
  #[test]
  fn default_differs_from_new_on_purpose() {
    assert_eq!(RequestOptions::default(), RequestOptions::deterministic());
    assert_ne!(RequestOptions::default(), RequestOptions::new());
    assert_eq!(Options::new().request(), &RequestOptions::default());

    // ImageBudget and ThreadOptions have no such split: their absent-table
    // value already was `new()`, so `Default` stays `new()` for both.
    assert_eq!(ImageBudget::default(), ImageBudget::new());
    assert_eq!(ThreadOptions::default(), ThreadOptions::new());
    assert_eq!(ThreadOptions::default().intra_threads(), None);
    assert_eq!(ThreadOptions::default().inter_threads(), None);
  }

  #[test]
  fn request_options_validate_rejects_bad_inputs() {
    assert!(
      RequestOptions::new()
        .with_max_new_tokens(0)
        .validate()
        .is_err()
    );
    assert!(
      RequestOptions::new()
        .with_temperature(-1.0)
        .validate()
        .is_err()
    );
    assert!(RequestOptions::new().with_min_p(2.0).validate().is_err());
    assert!(
      RequestOptions::new()
        .with_repetition_penalty(0.5)
        .validate()
        .is_err()
    );
  }

  #[test]
  fn request_options_validate_rejects_non_finite() {
    // Each non-finite value would otherwise pass the range checks
    // (NaN < 0.0 is false, etc.), poison softmax/probabilities, and
    // panic the sampler at partial_cmp(...).unwrap().
    for nan_temp in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      assert!(
        RequestOptions::new()
          .with_temperature(nan_temp)
          .validate()
          .is_err(),
        "temperature {nan_temp:?} must be rejected"
      );
    }
    for nan_min_p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      assert!(
        RequestOptions::new()
          .with_min_p(nan_min_p)
          .validate()
          .is_err(),
        "min_p {nan_min_p:?} must be rejected"
      );
    }
    for nan_rep in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      assert!(
        RequestOptions::new()
          .with_repetition_penalty(nan_rep)
          .validate()
          .is_err(),
        "repetition_penalty {nan_rep:?} must be rejected"
      );
    }
  }

  #[test]
  fn request_options_with_chains() {
    let r = RequestOptions::new()
      .with_temperature(0.3)
      .with_min_p(0.05)
      .with_repetition_penalty(1.10)
      .with_max_new_tokens(1024);
    assert_eq!(r.temperature(), 0.3);
    assert_eq!(r.max_new_tokens(), 1024);
  }

  #[test]
  fn request_options_validate_rejects_subnormal_positive_temperature() {
    // A tiny positive temperature like 1e-40 makes 1/temp overflow
    // to +inf, poisoning softmax with NaN/inf. Reject these as
    // ill-conditioned. Exactly 0.0 (greedy) and >= 1e-3 are both
    // fine.
    assert!(
      RequestOptions::new()
        .with_temperature(1e-40)
        .validate()
        .is_err()
    );
    assert!(
      RequestOptions::new()
        .with_temperature(1e-6)
        .validate()
        .is_err()
    );
    // Boundary: exactly 1e-3 is allowed.
    assert!(
      RequestOptions::new()
        .with_temperature(1e-3)
        .validate()
        .is_ok()
    );
    // Greedy (exactly 0.0) is allowed.
    assert!(
      RequestOptions::new()
        .with_temperature(0.0)
        .validate()
        .is_ok()
    );
  }

  #[test]
  fn request_options_validate_caps_repetition_penalty() {
    // An unbounded repetition_penalty (e.g. f32::MAX) lets penalty
    // × negative-logit overflow to -inf, making argmax/sample_min_p
    // pick an arbitrary masked token.
    let r = RequestOptions::new().with_repetition_penalty(MAX_REPETITION_PENALTY + 0.001);
    assert!(matches!(r.validate(), Err(Error::InvalidRequest(_))));
    let r_at = RequestOptions::new().with_repetition_penalty(MAX_REPETITION_PENALTY);
    assert!(r_at.validate().is_ok());
    let r_max = RequestOptions::new().with_repetition_penalty(f32::MAX);
    assert!(matches!(r_max.validate(), Err(Error::InvalidRequest(_))));
  }

  #[test]
  fn request_options_validate_caps_max_new_tokens() {
    // A usize::MAX max_new_tokens would drive Vec::with_capacity
    // to OOM before any model work. Cap at MAX_NEW_TOKENS_CAP so
    // validate fails fast.
    let r = RequestOptions::new().with_max_new_tokens(MAX_NEW_TOKENS_CAP + 1);
    assert!(matches!(r.validate(), Err(Error::InvalidRequest(_))));
    let r_ok = RequestOptions::new().with_max_new_tokens(MAX_NEW_TOKENS_CAP);
    assert!(r_ok.validate().is_ok());
  }

  // ===== ImageBudget =====

  // ===== ThreadOptions: the execution-mode rule =====

  /// `build_session` cannot be unit-tested without a real graph — it ends at
  /// `commit_from_file` — so the rule it reads lives here, where it can be.
  /// ort's inter-op pool exists only in parallel execution mode, so a count
  /// above 1 must select that mode or be inert; 1 and unset must NOT, or every
  /// existing recipe would silently change execution mode.
  #[test]
  fn inter_threads_above_one_asks_for_parallel_execution() {
    assert!(
      ThreadOptions::new()
        .with_inter_threads(2)
        .requires_parallel_execution()
    );
    assert!(
      ThreadOptions::new()
        .with_inter_threads(8)
        .requires_parallel_execution()
    );

    assert!(!ThreadOptions::new().requires_parallel_execution());
    assert!(
      !ThreadOptions::new()
        .with_inter_threads(1)
        .requires_parallel_execution()
    );
    // 0 is ort's "you choose" and stays sequential rather than turning the
    // mode on with no thread budget to run it.
    assert!(
      !ThreadOptions::new()
        .with_inter_threads(0)
        .requires_parallel_execution()
    );
    // The intra-op count never selects the execution mode.
    assert!(
      !ThreadOptions::new()
        .with_intra_threads(16)
        .requires_parallel_execution()
    );
  }

  /// The bit-stability preset must stay sequential: it is what the crate's
  /// determinism docs promise, and parallel execution is not bit-stable.
  #[test]
  fn the_deterministic_thread_preset_stays_sequential() {
    assert_eq!(ThreadOptions::deterministic().inter_threads(), Some(1));
    assert!(!ThreadOptions::deterministic().requires_parallel_execution());
    #[cfg(all(feature = "inference", ort_backend))]
    assert!(
      !OrtOptions::deterministic()
        .thread()
        .requires_parallel_execution()
    );
  }

  #[test]
  fn image_budget_new_matches_preprocessor_config() {
    let b = ImageBudget::new();
    assert_eq!(b.min_image_tokens(), 64);
    assert_eq!(b.max_image_tokens(), 256);
    assert_eq!(b.min_tiles(), 2);
    assert_eq!(b.max_tiles(), 10);
    assert!(b.use_thumbnail());
  }

  #[test]
  fn image_budget_fast_is_smaller() {
    let f = ImageBudget::fast();
    assert!(f.max_image_tokens() < ImageBudget::new().max_image_tokens());
    assert!(!f.use_thumbnail());
  }

  #[test]
  fn image_budget_validate_rejects_bad_inputs() {
    let mut b = ImageBudget::new();
    b.set_min_image_tokens(0);
    assert!(b.validate().is_err());
    let mut b2 = ImageBudget::new();
    b2.set_max_image_tokens(b2.min_image_tokens() - 1);
    assert!(b2.validate().is_err());
  }

  #[test]
  fn image_budget_max_tokens_per_image_default() {
    // Conservative per-image upper bound used for pre-decode
    // admission control. Default budget: max_tiles=10,
    // max_image_tokens=256 → 10 × 256 + 256 = 2816.
    assert_eq!(ImageBudget::new().max_tokens_per_image(), 2816);
  }

  #[test]
  fn image_budget_max_tokens_per_image_fast() {
    // Fast preset: max_tiles=4, max_image_tokens=64 → 4×256+64 = 1088.
    assert_eq!(ImageBudget::fast().max_tokens_per_image(), 1088);
  }

  #[test]
  fn image_budget_validate_caps_max_image_tokens() {
    // An unbounded max_image_tokens (e.g., usize::MAX) lets
    // smart_resize derive a multi-EB pixel budget; flatten_to_patches
    // would then allocate enormous pixel_values tensors. Cap at
    // MAX_IMAGE_TOKENS_CAP.
    let mut b = ImageBudget::new();
    b.set_max_image_tokens(MAX_IMAGE_TOKENS_CAP + 1);
    assert!(matches!(b.validate(), Err(Error::InvalidBudget(_))));
    let mut b_ok = ImageBudget::new();
    b_ok.set_max_image_tokens(MAX_IMAGE_TOKENS_CAP);
    assert!(b_ok.validate().is_ok());
    // min_image_tokens is implicitly bounded since validate enforces
    // max_image_tokens >= min_image_tokens.
    let mut b_min = ImageBudget::new();
    b_min.set_min_image_tokens(MAX_IMAGE_TOKENS_CAP + 1);
    b_min.set_max_image_tokens(MAX_IMAGE_TOKENS_CAP + 1);
    assert!(matches!(b_min.validate(), Err(Error::InvalidBudget(_))));
  }

  #[test]
  fn image_budget_validate_rejects_max_tiles_above_tokenizer_grid() {
    // Tokenizer ships row/col markers up to 10×10. Values above that
    // would silently corrupt position-token embeddings.
    let mut b = ImageBudget::new();
    b.set_max_tiles(MAX_TOKENIZER_TILE_DIM + 1);
    assert!(b.validate().is_err());
    let mut b2 = ImageBudget::new();
    b2.set_max_tiles(MAX_TOKENIZER_TILE_DIM);
    assert!(b2.validate().is_ok());
  }

  #[test]
  fn image_budget_max_pixels_tolerance_round_trip() {
    // Tolerance is stored as `f32` directly — round-trip is exact
    // for any finite float.
    let b = ImageBudget::new().with_max_pixels_tolerance(2.5);
    assert_eq!(b.max_pixels_tolerance(), 2.5);
    let mut b2 = ImageBudget::new();
    b2.set_max_pixels_tolerance(1.75);
    assert_eq!(b2.max_pixels_tolerance(), 1.75);
    // Default value (2.0) round-trips.
    assert_eq!(ImageBudget::new().max_pixels_tolerance(), 2.0);
  }

  #[test]
  fn image_budget_max_pixels_tolerance_preserves_sub_hundredth_precision() {
    // A tolerance like 2.067 would be truncated to 2.06 by a prior
    // `(v * 100.0) as u32` storage shape, silently routing a
    // 723x724 image to multi-tile when upstream Python's float
    // threshold would have kept it single-tile. The f32-direct
    // storage preserves it.
    let b = ImageBudget::new().with_max_pixels_tolerance(2.067);
    assert_eq!(b.max_pixels_tolerance(), 2.067);
  }

  #[test]
  fn image_budget_validate_rejects_non_finite_tolerance() {
    let mut b = ImageBudget::new();
    b.set_max_pixels_tolerance(f32::NAN);
    assert!(b.validate().is_err());
    b.set_max_pixels_tolerance(f32::INFINITY);
    assert!(b.validate().is_err());
    b.set_max_pixels_tolerance(0.0);
    assert!(b.validate().is_err());
    b.set_max_pixels_tolerance(-1.0);
    assert!(b.validate().is_err());
  }

  // ===== Send/Sync =====

  #[test]
  fn options_are_send_sync_copy_or_clone() {
    fn req<T: Send + Sync>() {}
    req::<RequestOptions>();
    req::<ImageBudget>();
    req::<ThreadOptions>();
    req::<BackendOptions>();
    req::<Options>();
  }
}
