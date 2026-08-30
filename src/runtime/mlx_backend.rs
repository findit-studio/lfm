//! MLX (mlxrs) inference backend for LFM2.5-VL — Apple-Silicon only.
//!
//! This is the macOS/arm64 alternative to the default `ort` (ONNX Runtime)
//! decoder/vision/embed path. It is compiled only on
//! `aarch64-apple-darwin` (and only under the `inference` + `decoders`
//! features the rest of the [`Backend`] seam lives under), because `mlxrs`
//! binds the Metal-backed MLX C++ runtime through `mlx-c` FFI and has no
//! other target. There is **no** `mlx` Cargo feature — the backend is
//! selected by platform + checkpoint shape (see Cargo.toml and
//! [`Engine`](crate::Engine)'s `from_dir`), and a caller can pin it
//! explicitly with [`Options::with_backend`](crate::Options::with_backend).
//!
//! # Design
//!
//! [`MlxBackend`] implements the same [`Backend`] trait the [`OrtBackend`]
//! does, so the [`generate`](crate::generate::generate) loop drives it with no
//! signature change. It owns the loaded [`Lfm2Vl`] model; the engine still owns
//! tokenization, the chat template, EOS handling, and the sampler — the MLX
//! backend only replaces the model-weight stages (text embed, per-image vision
//! encode + splice, and the decoder forward).
//!
//! Unlike the ORT path, the embeds and KV cache stay **on device**: the prompt
//! embeddings are an mlx [`Array`](mlxrs::Array) and the cache is the LFM2
//! heterogeneous per-layer [`KvCache`](mlxrs::lm::cache::KvCache) vector. Only
//! the decoder logits (`Vec<f32>` for the sampler) cross the host boundary.
//!
//! # Weight source
//!
//! The MLX backend consumes an **MLX-format checkpoint** (`config.json` + a
//! weight set), not the ONNX graphs. The loader reads `config.json`, then
//! delegates weight discovery to [`mlxrs::io::load_weights_from_dir`], which
//! auto-detects a sharded `model.safetensors.index.json`, a single
//! `model.safetensors`, a `*.gguf`, or a `*.npz` via the centralized `mlxrs`
//! loader, runs [`Lfm2Vl::sanitize`], and builds the model via
//! [`Lfm2Vl::from_weights`]. A quantized checkpoint (e.g. the
//! `LiquidAI/LFM2.5-VL-450M-MLX-8bit` export) is detected by the `mlxrs`
//! convention — the presence of any `<layer>.scales` sibling — and loaded with
//! the `(group_size, bits, mode)` parsed from the `config.json` `quantization`
//! block.
//!
//! The model dimensions + quantization scheme are always read from
//! `config.json`; the gguf path is a **weight load seam only** — its embedded
//! metadata is NOT mapped to a config, so a gguf checkpoint still requires a
//! `config.json` alongside it.
//!
//! The explicit-format constructors ([`MlxBackend::from_safetensors`], and —
//! under the matching feature — `from_npz` / `from_gguf`) take a **weight file
//! path** directly and read the sibling `config.json` from the file's parent
//! directory, for callers who already know the format and location.
//!
//! # Preprocessing and the one authoritative image plan
//!
//! The MLX path preprocesses with `mlxrs`'s **own** tiling
//! ([`Lfm2Vl::split_image`], driven by the checkpoint's `config.json`) rather
//! than `lfm`'s ONNX [`Preprocessor`] — the patch geometry, normalization, and
//! grid-token math are the model's, so the spliced image features bind to the
//! positions `get_input_embeddings` expects. Images are decoded EXIF-aware via
//! `lfm`'s existing
//! [`decode_with_orientation`](crate::preproc::decode_with_orientation) /
//! [`decode_bytes_with_orientation`](crate::preproc::decode_bytes_with_orientation),
//! then handed to `split_image` as interleaved RGB bytes.
//!
//! But the prompt's `<image>` / `<|img_row_R_col_C|>` / `<|img_thumbnail|>`
//! markers and the context-budget admission gates are rendered by `lfm`, from
//! `lfm`'s [`ImageBudget`]. If the two tilings disagree the markers and the
//! feature block disagree, and when the totals happen to coincide the features
//! bind to the wrong spatial positions with no error at all. Three gates make
//! that unrepresentable:
//!
//! 1. **Load time** — [`MlxBackend::effective_budget`] compares every tiling
//!    parameter `lfm` renders markers from against the checkpoint's own value
//!    and refuses **by name**
//!    ([`Error::MlxTilingMismatch`]). The default
//!    [`ImageBudget::new()`](crate::ImageBudget::new) expresses "no opinion"
//!    and adopts the checkpoint's tiling instead; the engine then renders every
//!    prompt from the checkpoint's numbers.
//! 2. **Plan time** — [`MlxBackend::plan_image`] builds the plan from `lfm`'s
//!    ported tiling under that budget, then ratifies it against the
//!    checkpoint's own planner ([`plan_tiles`]): grid, thumbnail presence and
//!    sub-image count must agree.
//! 3. **Execute time** — [`MlxBackend::prepare_prompt_embeds`] re-runs gate 2
//!    against the dimensions it actually DECODED (a path-backed image is opened
//!    once for header planning and again here, so the file may have changed in
//!    between), then reads back the real patch grid of every sub-image
//!    `split_image` produced and checks the per-sub-image and total
//!    `<image>`-token counts against the plan. Both halves are needed: a
//!    transposed grid leaves every count identical, and matching grids still
//!    say nothing about the patch grid a produced sub-image really carries.
//!
//! Any disagreement is [`Error::ImagePlanMismatch`], never a silent splice.

use std::path::Path;

use mlxrs::{
  Array, Dtype,
  lm::{cache::KvCache, model::Model as LmModel},
  vlm::models::lfm2_vl::{
    Lfm2Vl, Lfm2VlImageInputs, TilePlan, config::ModelConfig, num_image_tokens_from_patch_grid,
    plan_tiles,
  },
};
use smol_str::SmolStr;

use crate::{
  error::{Error, Result},
  options::{BackendKind, ImageBudget},
  preproc::{
    ImagePlan, Preprocessor,
    tile_grid::{DOWNSAMPLE_FACTOR, FULL_TILE_SIZE, PATCH_SIZE, pick_tile_grid},
  },
  runtime::{backend::Backend, checkpoint::MLX_CONFIG},
};

/// The per-layer quantization marker `mlxrs` (and mlx-lm / mlx-vlm) use: a
/// quantized `nn.Linear` / `nn.Embedding` stores its packed weight alongside a
/// sibling `<prefix>.scales` tensor. Its presence ANYWHERE in the loaded weight
/// map is the sole signal that the checkpoint is quantized — exactly the
/// convention [`Lfm2Vl::from_weights`] keys its per-layer dense-vs-quantized
/// choice on.
const QUANT_SCALES_SUFFIX: &str = ".scales";

/// Discriminate a dense from a quantized MLX checkpoint by the `mlxrs`
/// convention: a quantized checkpoint carries at least one `<layer>.scales`
/// sibling tensor (see [`QUANT_SCALES_SUFFIX`]); a dense one carries none. This
/// is the same `.scales`-presence signal `mlxrs` resolves per layer inside
/// [`Lfm2Vl::from_weights`] — checked here over the whole (sanitized) weight map
/// to pick the quantization-config thread.
fn weights_are_quantized(weights: &std::collections::HashMap<String, Array>) -> bool {
  weights.keys().any(|k| k.ends_with(QUANT_SCALES_SUFFIX))
}

/// The directory the sibling `config.json` (+ the engine's `tokenizer.json`) is
/// read from when an explicit-format constructor is handed a **weight file
/// path**: the file's parent directory, or the current directory (`.`) when
/// `weights` is a bare filename with no parent component (so
/// `from_safetensors("model.safetensors")` reads `./config.json`, not a config
/// at the filesystem root).
pub(crate) fn weights_parent(weights: &Path) -> &Path {
  weights
    .parent()
    .filter(|p| !p.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."))
}

/// Refuse **by name** when a value `lfm` relies on differs from the
/// checkpoint's own. Naming the parameter, both values, and their sources is
/// the whole point: a bare "tiling mismatch" leaves the caller to bisect their
/// `ImageBudget` against a `config.json` they may never have opened.
fn require_same<T>(parameter: &'static str, lfm_value: T, checkpoint_value: T) -> Result<()>
where
  T: PartialEq + std::fmt::Display,
{
  if lfm_value != checkpoint_value {
    return Err(Error::MlxTilingMismatch {
      parameter,
      lfm_value: SmolStr::new(lfm_value.to_string()),
      checkpoint_value: SmolStr::new(checkpoint_value.to_string()),
    });
  }
  Ok(())
}

/// Narrow a checkpoint's `i32` cardinality field to `usize`, naming the field
/// if it is negative (a `config.json` that cannot describe a real tiling).
fn config_count(field: &'static str, value: i32) -> Result<usize> {
  usize::try_from(value).map_err(|_| {
    Error::mlx_owned(format!(
      "checkpoint config.json `{field}` is negative ({value}); it must be a non-negative count"
    ))
  })
}

/// Refuse, by name, every checkpoint value `lfm` **bakes in** rather than reads.
///
/// These are not budget knobs: each one is compiled into this crate's token
/// math, its marker table, or its prompt rendering, so it cannot be adopted from
/// a checkpoint at runtime — a disagreement can only be refused.
///
/// - The preprocessing geometry (`patch_size`, `encoder_patch_size`,
///   `downsample_factor`, `tile_size`) and the `<image>` token id drive
///   `TileGrid`'s arithmetic and the tokenizer's marker table.
/// - `use_image_special_tokens` is upstream's switch for bracketing an image
///   block with `image_start` / `image_end` (`config.py:87`, default `true`).
///   `lfm` has no such switch: [`crate::chat_template::expand_image_placeholders`]
///   always emits `<|image_start|>` / `<|image_end|>` and
///   [`ImagePlan`] always budgets
///   [`IMAGE_BLOCK_WRAPPER_TOKENS`](crate::preproc::IMAGE_BLOCK_WRAPPER_TOKENS)
///   for them. A `false` checkpoint therefore gets a prompt carrying two tokens
///   its processor contract omits — and every count still lines up, because the
///   brackets are not `<image>` tokens and the plan budgets exactly what it
///   renders, so neither the plan-time nor the execute-time gate can see it.
///   Refusing by name is the choice over carrying the policy through
///   `ImagePlan` / marker rendering / structural admission: the brackets are
///   pinned identically on the ONNX road (the bundled `config.json` ships
///   `use_image_special_tokens: true`, and `from_dir` refuses a `false` one),
///   so honouring the flag would mean parameterizing the crate's one prompt
///   contract into two — the second of which no released checkpoint exercises
///   and no parity test can validate.
fn require_baked_in_contract(config: &ModelConfig) -> Result<()> {
  require_same(
    "vision_config.patch_size",
    i64::from(PATCH_SIZE),
    i64::from(config.vision_config.patch_size),
  )?;
  require_same(
    "encoder_patch_size",
    i64::from(PATCH_SIZE),
    i64::from(config.encoder_patch_size),
  )?;
  require_same(
    "downsample_factor",
    i64::from(DOWNSAMPLE_FACTOR),
    i64::from(config.downsample_factor),
  )?;
  require_same(
    "tile_size",
    i64::from(FULL_TILE_SIZE),
    i64::from(config.tile_size),
  )?;
  require_same(
    "image_token_index",
    i64::from(crate::chat_template::IMAGE_TOKEN_ID),
    i64::from(config.image_token_index),
  )?;
  require_same(
    "use_image_special_tokens",
    true,
    config.use_image_special_tokens,
  )
}

/// Express the checkpoint's own tiling parameters as an [`ImageBudget`] — the
/// vocabulary `lfm`'s prompt rendering and admission gates speak.
///
/// `do_image_splitting = false` has no direct counterpart in [`ImageBudget`];
/// upstream's `_preprocess` disables splitting by forcing
/// `min_tiles = max_tiles = 1`, and `mlxrs`'s [`plan_tiles`] treats that band
/// as equivalent, so the flag is folded into the tile band the same way.
fn checkpoint_budget(config: &ModelConfig) -> Result<ImageBudget> {
  let splitting = config.do_image_splitting && !(config.min_tiles == 1 && config.max_tiles == 1);
  let (min_tiles, max_tiles) = if splitting {
    (
      config_count("min_tiles", config.min_tiles)?,
      config_count("max_tiles", config.max_tiles)?,
    )
  } else {
    (1, 1)
  };
  let budget = ImageBudget::new()
    .with_min_image_tokens(config_count("min_image_tokens", config.min_image_tokens)?)
    .with_max_image_tokens(config_count("max_image_tokens", config.max_image_tokens)?)
    .with_min_tiles(min_tiles)
    .with_max_tiles(max_tiles)
    .with_use_thumbnail(config.use_thumbnail)
    .with_max_pixels_tolerance(config.max_pixels_tolerance);
  // A checkpoint whose tiling `lfm` cannot express — e.g. `max_tiles > 10`,
  // beyond the bundled tokenizer's `<|img_row_R_col_C|>` marker grid — is
  // rejected here rather than producing markers that tokenize as plain text.
  budget.validate()?;
  Ok(budget)
}

/// MLX-backed [`Backend`] for LFM2.5-VL. Owns the loaded [`Lfm2Vl`] model.
///
/// The model locates the `<image>`-token positions itself from its
/// `image_token_index` during `get_input_embeddings`, so no token id is held
/// here. `Lfm2Vl`'s `get_input_embeddings` / `forward_embeddings` take `&self`,
/// so this is immutable after construction; the [`Backend`] methods take `&mut
/// self` only to satisfy the trait (the ORT backend mutates its sessions).
pub(crate) struct MlxBackend {
  model: Lfm2Vl,
}

impl MlxBackend {
  /// Load an [`MlxBackend`] from an MLX checkpoint **directory** containing
  /// `config.json` and a weight set in any enabled format.
  ///
  /// Weight discovery is delegated to [`mlxrs::io::load_weights_from_dir`], which
  /// auto-detects a sharded `model.safetensors.index.json`, a single
  /// `model.safetensors`, a `*.gguf`, or a `*.npz` via the centralized `mlxrs`
  /// loader. For an exact known weight file path use [`Self::from_safetensors`]
  /// (or the feature-gated `from_npz` / `from_gguf`).
  ///
  /// # Errors
  /// - [`Error::Io`] if `config.json` cannot be read;
  /// - [`Error::Mlx`] if no weight set in any enabled format is present, or for
  ///   any `mlxrs` parse / load / construction failure (malformed config,
  ///   corrupt weights, weight/key mismatch).
  pub(crate) fn from_dir(dir: &Path) -> Result<Self> {
    Self::construct(dir, || {
      mlxrs::io::load_weights_from_dir(dir).map_err(Error::from_mlx)
    })
  }

  /// Load an [`MlxBackend`] from an **exact** `model.safetensors` file path. The
  /// `config.json` is read from the weight file's parent directory (see
  /// [`weights_parent`]).
  pub(crate) fn from_safetensors(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_safetensors(weights).map_err(Error::from_mlx)
    })
  }

  /// Load an [`MlxBackend`] from an **exact** `*.npz` file path. The
  /// `config.json` is read from the weight file's parent directory (see
  /// [`weights_parent`]).
  #[cfg(feature = "npz")]
  pub(crate) fn from_npz(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_npz(weights).map_err(Error::from_mlx)
    })
  }

  /// Load an [`MlxBackend`] from an **exact** `*.gguf` file path. The
  /// `config.json` is read from the weight file's parent directory (see
  /// [`weights_parent`]); the gguf's embedded metadata is NOT mapped to a
  /// config, so a sibling `config.json` is still required.
  #[cfg(feature = "gguf")]
  pub(crate) fn from_gguf(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_gguf(weights)
        .map(|(w, _meta)| w)
        .map_err(Error::from_mlx)
    })
  }

  /// Shared construction body for every MLX constructor: read + parse the
  /// `config.json` from `dir`, then load the weights via `load`, [`sanitize`
  /// them](Lfm2Vl::sanitize), discriminate dense vs quantized by the
  /// `.scales`-presence convention, and build the model via
  /// [`Lfm2Vl::from_weights`].
  ///
  /// `dir` is the directory the `config.json` lives in (the checkpoint dir for
  /// [`Self::from_dir`], the weight file's parent for the explicit-format
  /// constructors); `load` supplies the raw (pre-`sanitize`) weight map. The
  /// config read + parse + full [`ModelConfig::validate`] run BEFORE `load`, so a
  /// malformed config fails fast and never touches the weight file.
  fn construct(
    dir: &Path,
    load: impl FnOnce() -> Result<std::collections::HashMap<String, Array>>,
  ) -> Result<Self> {
    let config_path = dir.join(MLX_CONFIG);

    let config_json = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
    let config = ModelConfig::from_json(&config_json).map_err(Error::from_mlx)?;

    // Run the FULL `ModelConfig::validate` (it pins `model_type` /
    // `projector_hidden_act`, bounds the projector + patch-budget fields, checks
    // the tile / token orderings, and validates both tower configs — every
    // dimension / count structurally valid) BEFORE any weight is loaded, so a
    // malformed config fails fast with a typed error and never touches the
    // (expensive) weight file.
    config.validate().map_err(Error::from_mlx)?;

    let raw = load()?;
    let weights = Lfm2Vl::sanitize(raw).map_err(Error::from_mlx)?;

    // An MLX LFM2.5-VL checkpoint may be a QUANTIZED safetensors (the released
    // `LiquidAI/LFM2.5-VL-450M-MLX-8bit` export carries per-layer `.scales` /
    // `.biases`), not dense. Pick the quantization thread by the `mlxrs`
    // convention: any `<layer>.scales` sibling in the (sanitized) weight map is
    // the sole quantized signal. `parse_quantization` is the golden read-path
    // parser (it returns `None` for a dense config, so it is also the no-op for
    // the dense branch); `from_weights` then resolves the per-layer
    // dense-vs-quantized split off the same `.scales` markers.
    let model = if weights_are_quantized(&weights) {
      let quantization =
        mlxrs::lm::quant::parse_quantization(&config_json).map_err(Error::from_mlx)?;
      Lfm2Vl::from_weights(config, weights, quantization.as_ref()).map_err(Error::from_mlx)?
    } else {
      Lfm2Vl::from_weights(config, weights, None).map_err(Error::from_mlx)?
    };

    Ok(Self { model })
  }

  /// Refuse a checkpoint whose real positional limit differs from the
  /// [`MODEL_CONTEXT_TOKENS`](crate::options::MODEL_CONTEXT_TOKENS) constant
  /// `generate`'s admission gates trust.
  ///
  /// The MLX-road counterpart of the ONNX road's
  /// `validate_config_context_matches_bundled`, read from the loaded
  /// `ModelConfig` rather than re-parsing `config.json`, so it asserts against
  /// the very value the model was built with. A smaller-context export would
  /// otherwise load fine and then accept prompts up to 128 K that it cannot
  /// serve, failing late or generating with invalid position state.
  pub(crate) fn validate_context_limit(&self) -> Result<()> {
    let context = self.model.config().text_config.max_position_embeddings;
    require_same(
      "text_config.max_position_embeddings",
      i64::try_from(crate::options::MODEL_CONTEXT_TOKENS).unwrap_or(i64::MAX),
      i64::from(context),
    )
  }

  /// Reconcile the caller's [`ImageBudget`] with the checkpoint's own tiling
  /// and return the budget the engine must render every prompt from.
  ///
  /// Two classes of parameter, two policies:
  ///
  /// - **Values `lfm` bakes in** (`PATCH_SIZE`, `DOWNSAMPLE_FACTOR`,
  ///   `FULL_TILE_SIZE`, the `<image>` token id, and the
  ///   `use_image_special_tokens` bracketing policy) are compiled into
  ///   `TileGrid`'s token math, the tokenizer's marker table and the prompt
  ///   renderer — they cannot be adopted at runtime, so a checkpoint that
  ///   disagrees is always refused by name (see
  ///   [`require_baked_in_contract`]).
  /// - **Budget knobs** (`min`/`max_image_tokens`, `min`/`max_tiles`,
  ///   `use_thumbnail`, `max_pixels_tolerance`) are caller-tunable, but the MLX
  ///   path cannot honour them: `split_image` reads the checkpoint's
  ///   `config.json`, not this budget. So the default
  ///   [`ImageBudget::new()`](crate::ImageBudget::new) — "no opinion" — adopts
  ///   the checkpoint's values, and any OTHER budget must match the checkpoint
  ///   exactly or the load is refused by name. Refusing is the point: the
  ///   released `LiquidAI/LFM2.5-VL-450M-MLX-8bit` ships `use_thumbnail: false`,
  ///   so a caller passing `ImageBudget::fast()` (or any hand-tuned budget that
  ///   disagrees) previously got markers for one layout and features for
  ///   another.
  ///
  /// # Errors
  /// [`Error::MlxTilingMismatch`] naming the first disagreeing parameter;
  /// [`Error::InvalidBudget`] if the checkpoint's own tiling is one `lfm`
  /// cannot express (e.g. `max_tiles` past the tokenizer's 10×10 marker grid).
  pub(crate) fn effective_budget(&self, requested: &ImageBudget) -> Result<ImageBudget> {
    let config = self.model.config();

    // ── baked-in values: never adoptable ───────────────────────────────────
    require_baked_in_contract(config)?;

    let checkpoint = checkpoint_budget(config)?;

    // ── budget knobs: adopt the checkpoint's when the caller expressed no
    //    opinion, otherwise demand an exact match ────────────────────────────
    if *requested == ImageBudget::new() {
      return Ok(checkpoint);
    }
    require_same(
      "min_image_tokens",
      requested.min_image_tokens(),
      checkpoint.min_image_tokens(),
    )?;
    require_same(
      "max_image_tokens",
      requested.max_image_tokens(),
      checkpoint.max_image_tokens(),
    )?;
    require_same("min_tiles", requested.min_tiles(), checkpoint.min_tiles())?;
    require_same("max_tiles", requested.max_tiles(), checkpoint.max_tiles())?;
    require_same(
      "use_thumbnail",
      requested.use_thumbnail(),
      checkpoint.use_thumbnail(),
    )?;
    require_same(
      "max_pixels_tolerance",
      requested.max_pixels_tolerance(),
      checkpoint.max_pixels_tolerance(),
    )?;
    Ok(checkpoint)
  }

  /// Ask the checkpoint's OWN planner what `width`×`height` tiles into, and
  /// ratify `plan` — the plan the prompt's markers were rendered from — against
  /// the answer.
  ///
  /// Run at BOTH ends of the plan's life, which is the point of it being one
  /// method: at plan time against the image's header dimensions, and again at
  /// execute time against the dimensions actually decoded. The second call is
  /// what closes the re-read window — a path-backed image is opened once for
  /// header planning and again for decoding, so a file replaced in between
  /// (1920×1080 → 1080×1920) reaches the splice as the transposed grid. Its
  /// sub-image count and every per-sub-image token count are identical, so
  /// [`verify_sub_images`] cannot see it; only the grid comparison can.
  fn ratify_against_checkpoint(
    &self,
    index: usize,
    plan: &ImagePlan,
    width: u32,
    height: u32,
  ) -> Result<()> {
    let processor = self.model.processor_config().map_err(Error::from_mlx)?;
    let tiles = plan_tiles(height, width, &processor).map_err(Error::from_mlx)?;
    ratify_plan(index, plan, &tiles)
  }

  /// Build a `(1, seq)` i32 `input_ids` [`Array`] from host token ids, rejecting
  /// any id outside `i32` range with a typed error.
  ///
  /// LFM token ids are `i64` in the loop; mlx consumes `input_ids` as `i32` (the
  /// embedding gather index dtype). An id above `i32::MAX` (or below `i32::MIN`)
  /// would wrap to a different — possibly negative — gather index under an `as`
  /// cast and silently corrupt the embedding, so each id is converted with a
  /// checked [`i32::try_from`] and an offending id surfaces as [`Error::Mlx`].
  fn ids_array(token_ids: &[i64]) -> Result<Array> {
    let mut ids: Vec<i32> = Vec::with_capacity(token_ids.len());
    for &id in token_ids {
      ids.push(i32::try_from(id).map_err(|_| {
        Error::mlx_owned(format!("token id {id} out of i32 range for MLX input_ids"))
      })?);
    }
    let seq = ids.len();
    Array::from_slice::<i32>(&ids, &(1usize, seq)).map_err(Error::from_mlx)
  }
}

/// Ratify an [`ImagePlan`] built from `lfm`'s ported tiling against the
/// checkpoint's OWN planner.
///
/// Gate 2 of the three described in the [module docs](self). The load-time gate
/// already proved the two use identical parameters, and both are faithful ports
/// of the same HuggingFace `resize_and_split`; this asserts the ports actually
/// agree on THIS image before the prompt is rendered from the plan. Everything
/// [`TilePlan`] exposes is compared — split flag, grid, thumbnail presence, and
/// sub-image count.
fn ratify_plan(index: usize, plan: &ImagePlan, tiles: &TilePlan) -> Result<()> {
  let info = plan.placeholder();
  // `TilePlan::grid()` is `(grid_width, grid_height)`; `lfm`'s marker layout is
  // (rows, cols) = (grid_height, grid_width).
  let (grid_width, grid_height) = tiles.grid();
  for (parameter, planned, produced) in [
    ("tile-grid rows", info.rows(), grid_height as usize),
    ("tile-grid cols", info.cols(), grid_width as usize),
    (
      "thumbnail sub-images",
      usize::from(info.thumbnail_tokens().is_some()),
      usize::from(tiles.has_thumbnail()),
    ),
    (
      "multi-tile split",
      usize::from(info.rows() > 1 || info.cols() > 1),
      usize::from(tiles.is_split()),
    ),
  ] {
    if planned != produced {
      return Err(Error::ImagePlanMismatch {
        image: index,
        parameter,
        planned,
        produced,
      });
    }
  }
  let produced = config_count(
    "sub_image_count",
    tiles.sub_image_count().map_err(Error::from_mlx)? as i32,
  )?;
  if produced != plan.sub_images() {
    return Err(Error::ImagePlanMismatch {
      image: index,
      parameter: "sub-image count",
      planned: plan.sub_images(),
      produced,
    });
  }
  Ok(())
}

/// Check the sub-images `split_image` actually produced against the plan the
/// prompt's markers were rendered from.
///
/// Gate 3 of the three described in the [module docs](self), and the only one
/// that sees real pixels. Each sub-image's `<image>`-token count is recomputed
/// with `mlxrs`'s own [`num_image_tokens_from_patch_grid`] from the patch grid
/// the tile really carries, in the HF batch order `split_image` documents
/// (tiles row-major, then the thumbnail).
fn verify_sub_images(
  index: usize,
  plan: &ImagePlan,
  tiles: &[Lfm2VlImageInputs],
  downsample_factor: i32,
) -> Result<()> {
  if tiles.len() != plan.sub_images() {
    return Err(Error::ImagePlanMismatch {
      image: index,
      parameter: "sub-image count",
      planned: plan.sub_images(),
      produced: tiles.len(),
    });
  }
  let info = plan.placeholder();
  let last = tiles.len().saturating_sub(1);
  let mut produced_tokens = 0usize;
  for (position, tile) in tiles.iter().enumerate() {
    let (rows, cols) = tile.grid().map_err(Error::from_mlx)?;
    let tokens =
      num_image_tokens_from_patch_grid(rows, cols, downsample_factor).map_err(Error::from_mlx)?;
    let tokens = config_count("sub-image token count", tokens)?;
    // The thumbnail is the LAST sub-image when the plan carries one; every
    // other position is a main tile.
    let planned = match info.thumbnail_tokens() {
      Some(thumbnail) if position == last => thumbnail,
      _ => info.tokens_per_main_tile(),
    };
    if tokens != planned {
      return Err(Error::ImagePlanMismatch {
        image: index,
        parameter: "tokens per sub-image",
        planned,
        produced: tokens,
      });
    }
    produced_tokens = produced_tokens.saturating_add(tokens);
  }
  if produced_tokens != plan.image_tokens() {
    return Err(Error::ImagePlanMismatch {
      image: index,
      parameter: "image tokens",
      planned: plan.image_tokens(),
      produced: produced_tokens,
    });
  }
  Ok(())
}

impl Backend for MlxBackend {
  /// On-device prompt / per-token embeddings (an mlx `Array`); no host copy.
  type Embeds = Array;
  /// The LFM2 heterogeneous per-layer cache (`Lfm2Vl::make_cache`).
  type Cache = Vec<Box<dyn KvCache>>;

  fn kind(&self) -> BackendKind {
    BackendKind::Mlx
  }

  fn make_cache(&self) -> Result<Self::Cache> {
    Ok(self.model.make_cache())
  }

  /// Plan one image, then ratify the plan against the checkpoint's own planner.
  ///
  /// `preproc`'s budget is the **effective** budget the engine installed at
  /// load — either adopted from this checkpoint or proven equal to it by
  /// [`Self::effective_budget`] — so `lfm`'s ported tiling and the checkpoint's
  /// run on identical parameters. [`plan_tiles`] is then asked the same
  /// question and every field it exposes must agree.
  fn plan_image(
    &self,
    preproc: &Preprocessor,
    index: usize,
    width: u32,
    height: u32,
  ) -> Result<ImagePlan> {
    let grid = pick_tile_grid(width, height, preproc.budget())?;
    let plan = ImagePlan::from_placeholder(grid.to_placeholder_info());
    self.ratify_against_checkpoint(index, &plan, width, height)?;
    Ok(plan)
  }

  /// Build the full prompt `inputs_embeds` on device: embed the token ids and
  /// splice each image's NaFlex features in, using `mlxrs`'s OWN tiling.
  ///
  /// Each image is decoded EXIF-aware (`lfm`'s `decode_*_with_orientation`),
  /// projected to interleaved RGB bytes, tiled by
  /// [`split_image`](Lfm2Vl::split_image), and **checked against its
  /// [`ImagePlan`]** — the plan the prompt's `<image>` runs were rendered from
  /// — before any feature is spliced. `get_input_embeddings` then embeds the
  /// ids and merges the concatenated image features at the `<image>`-token
  /// positions it locates itself from the model's `image_token_index`, which is
  /// why `image_positions` is not consumed here: `generate` has already proven
  /// that count equals the plans' total.
  fn prepare_prompt_embeds(
    &mut self,
    _preproc: &Preprocessor,
    input_ids: &[i64],
    images: &[crate::ImageInput<'_>],
    plans: &[ImagePlan],
    _image_positions: &[usize],
  ) -> Result<Self::Embeds> {
    if plans.len() != images.len() {
      return Err(Error::ImagePlanMismatch {
        image: images.len(),
        parameter: "plans per image",
        planned: plans.len(),
        produced: images.len(),
      });
    }
    let downsample_factor = self.model.config().downsample_factor;
    // Collect every image's tiled NaFlex sub-image inputs, in image order, then
    // sub-image order — the same order the rendered marker layout lays its
    // `<image>`-token runs in.
    let mut all_inputs: Vec<Lfm2VlImageInputs> = Vec::new();
    for (index, (img, plan)) in images.iter().zip(plans.iter()).enumerate() {
      // Decode EXIF-aware via lfm's own decoders (reused, not reimplemented),
      // then project to the interleaved `width * height * 3` RGB bytes
      // `split_image` consumes. `decode_rgb` is the fallible (try_reserve)
      // extraction so a near-cap image surfaces a typed error rather than an
      // infallible-alloc abort.
      let decoded = match img {
        #[cfg(not(target_arch = "wasm32"))]
        crate::ImageInput::Path(p) => crate::preproc::decode_with_orientation(p)?,
        crate::ImageInput::Bytes(b) => crate::preproc::decode_bytes_with_orientation(b)?,
      };
      let (rgb, width, height) =
        mlxrs::vlm::image::decode_rgb(&decoded).map_err(Error::from_mlx)?;
      drop(decoded); // free the DynamicImage before tiling/encoding.
      // Re-derive the layout from the dimensions actually DECODED and ratify it
      // against the plan the markers were rendered from, before a single
      // feature is spliced. `verify_sub_images` below compares counts, which a
      // transposed grid (2×4 → 4×2, after a path-backed image was replaced
      // between the header read and this one) leaves untouched.
      self.ratify_against_checkpoint(index, plan, width, height)?;
      let tiles = self
        .model
        .split_image(&rgb, width, height)
        .map_err(Error::from_mlx)?;
      verify_sub_images(index, plan, &tiles, downsample_factor)?;
      all_inputs.extend(tiles);
    }

    let ids = Self::ids_array(input_ids)?;
    // `get_input_embeddings` embeds the ids AND splices the image features at
    // the `<image>`-token positions (the mask-driven `masked_scatter`),
    // returning the merged `(1, seq, hidden)` embeds. Stays on device.
    self
      .model
      .get_input_embeddings(&ids, &all_inputs)
      .map_err(Error::from_mlx)
  }

  /// Embed a single newly-sampled token id (no images) → `(1, 1, hidden)`.
  fn embed_one(&mut self, token_id: i64) -> Result<Self::Embeds> {
    let ids = Self::ids_array(&[token_id])?;
    self
      .model
      .get_input_embeddings(&ids, &[])
      .map_err(Error::from_mlx)
  }

  /// Decoder forward over `embeds` (the prompt at prefill, one token per decode
  /// step). Advances `cache` in place and returns the HOST logits for the
  /// sampler — the LAST sequence position only, matching the ORT
  /// `decoder_step`'s contract.
  ///
  /// `forward_embeddings` returns `(1, seq, vocab)` logits; the last position is
  /// sliced to `(vocab,)`, **cast to f32 before the host copy** (a quantized /
  /// f16 / bf16 checkpoint yields non-f32 logits and `to_vec::<f32>` is
  /// dtype-strict), `eval`'d, and read to a `Vec<f32>`. `seq_len` is unused: the
  /// embeds tensor already carries the sequence axis, and the slice reads the
  /// real last position from the logits shape.
  fn decoder_step(
    &mut self,
    cache: &mut Self::Cache,
    embeds: &Self::Embeds,
    _seq_len: usize,
  ) -> Result<Vec<f32>> {
    let logits = <Lfm2Vl as LmModel>::forward_embeddings(&self.model, embeds, cache)
      .map_err(Error::from_mlx)?;
    last_position_logits_f32(&logits)
  }
}

/// Slice the LAST sequence position out of a `(1, seq, vocab)` logits array, cast
/// it to f32, and read it to a host `Vec<f32>` for the sampler.
///
/// Matches the ORT `decoder_step`'s "last position only" semantics. The cast to
/// f32 BEFORE the host copy is load-bearing: a quantized / half-precision
/// checkpoint produces logits in its activation dtype, and `to_vec::<f32>` is
/// dtype-strict, so without the `astype` it would fail. `astype` produces a NEW
/// array, so the model's tensors are never mutated.
///
/// The row is then rejected if ANY entry is non-finite, exactly as the ORT
/// [`Decoder::step`](crate::runtime::decoder::Decoder::step) does. A raw decoder
/// row has no legitimate `±inf` or NaN — the `-inf`s the samplers work with are
/// masks THEY apply — so a non-finite entry here is a broken forward, and
/// admitting it lets greedy lock onto a `+inf` position and collapses the
/// temperature path's softmax to a uniform draw across every entry, mask
/// included.
///
/// # Errors
/// - [`Error::Mlx`] if the logits are not rank-3 `(1, seq, vocab)` with `seq >=
///   1`, or for any slice / astype / eval / read failure;
/// - [`Error::SessionNonFiniteOutput`] if any logit is NaN or infinite.
fn last_position_logits_f32(logits: &Array) -> Result<Vec<f32>> {
  let shape = logits.shape();
  if shape.len() != 3 || shape[0] != 1 || shape[1] < 1 {
    return Err(Error::mlx_owned(format!(
      "expected (1, seq>=1, vocab) decoder logits, got shape {shape:?}"
    )));
  }
  let seq = shape[1] as i32;
  let vocab = shape[2] as i32;
  // Slice `[0, seq-1, 0] .. [1, seq, vocab]` (stride 1) → `(1, 1, vocab)`.
  let last = mlxrs::ops::indexing::slice(logits, &[0, seq - 1, 0], &[1, seq, vocab], &[1, 1, 1])
    .map_err(Error::from_mlx)?;
  // `(1, 1, vocab)` → `(vocab,)`, cast to f32 BEFORE the dtype-strict host read.
  let mut row = mlxrs::ops::shape::reshape(&last, &(vocab as usize,))
    .map_err(Error::from_mlx)?
    .astype(Dtype::F32)
    .map_err(Error::from_mlx)?;
  row.eval().map_err(Error::from_mlx)?;
  let row = row.to_vec::<f32>().map_err(Error::from_mlx)?;
  if row.iter().any(|v| !v.is_finite()) {
    return Err(Error::SessionNonFiniteOutput { stage: "decoder" });
  }
  Ok(row)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{chat_template::ImagePlaceholderInfo, runtime::checkpoint::MLX_SAFETENSORS};

  /// `weights_are_quantized` flips on the FIRST `<layer>.scales` sibling and is
  /// `false` for a `.weight`/`.bias`-only (dense) map — even one carrying a
  /// `.biases` sibling (a dense bias is not the quant marker; only `.scales` is).
  #[test]
  fn quantized_discriminator_keys_on_scales() {
    let dummy = || Array::from_slice::<f32>(&[0.0], &(1usize,)).expect("1-elem array");
    let mut dense: std::collections::HashMap<String, Array> = std::collections::HashMap::new();
    dense.insert(
      "language_model.model.layers.0.q_proj.weight".to_string(),
      dummy(),
    );
    dense.insert(
      "language_model.model.layers.0.q_proj.bias".to_string(),
      dummy(),
    );
    dense.insert("multi_modal_projector.linear_1.biases".to_string(), dummy());
    assert!(
      !weights_are_quantized(&dense),
      "a `.weight`/`.bias`/`.biases`-only map must be classified dense"
    );

    let mut quant = dense;
    quant.insert(
      "language_model.model.layers.0.q_proj.scales".to_string(),
      dummy(),
    );
    assert!(
      weights_are_quantized(&quant),
      "any `.scales` sibling must flip the discriminator to quantized"
    );
  }

  /// A token id above `i32::MAX` is rejected with a typed [`Error::Mlx`] naming
  /// the offending id, rather than wrapping (via an `as` cast) to a different —
  /// possibly negative — MLX gather index. The boundary id `i32::MAX` itself is
  /// in range.
  #[test]
  fn ids_array_rejects_out_of_i32_range() {
    let bad = i64::from(i32::MAX) + 1;
    // `Array` is not `Debug`, so use `.err().unwrap_or_else(...)` rather than
    // `expect_err` (which would require `Debug` on the `Ok` type).
    let err = MlxBackend::ids_array(&[10, bad, 20])
      .err()
      .unwrap_or_else(|| panic!("an out-of-i32-range id must be rejected"));
    match err {
      Error::Mlx(msg) => assert!(
        msg.contains(&bad.to_string()),
        "error must name the offending id {bad}, got {msg:?}"
      ),
      other => panic!("expected Error::Mlx, got {other:?}"),
    }
    // The boundary id round-trips into a valid `(1, 1)` array.
    let ok = MlxBackend::ids_array(&[i64::from(i32::MAX)]).expect("i32::MAX id must be accepted");
    assert_eq!(ok.shape(), vec![1, 1]);
  }

  /// The decoder logits reader slices the LAST sequence position and casts to
  /// f32 — exercised on a half-precision (f16) logits array, which the
  /// dtype-strict `to_vec::<f32>` would reject without the `astype`. Builds a
  /// `(1, 3, 4)` f16 array whose last row is `[7, 8, 9, 10]` and asserts the
  /// extracted f32 row.
  #[test]
  fn last_position_logits_casts_half_precision_and_takes_last_row() {
    // 3 positions × 4-vocab, last row distinct from the earlier ones.
    let flat: Vec<f32> = vec![
      1.0, 2.0, 3.0, 4.0, // pos 0
      5.0, 6.0, 7.0, 8.0, // pos 1
      7.0, 8.0, 9.0, 10.0, // pos 2 (last)
    ];
    let dense = Array::from_slice::<f32>(&flat, &(1usize, 3usize, 4usize)).expect("build logits");
    let half = dense.astype(Dtype::F16).expect("cast to f16");
    let row = last_position_logits_f32(&half).expect("extract last-row f32 logits");
    assert_eq!(row, vec![7.0_f32, 8.0, 9.0, 10.0]);
  }

  /// A non-rank-3 logits array (or `seq == 0`) is a typed [`Error::Mlx`], not a
  /// panic — the boundary discipline the rest of the crate's model seams use.
  #[test]
  fn last_position_logits_rejects_bad_rank() {
    let rank2 = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2usize)).expect("rank-2");
    assert!(matches!(
      last_position_logits_f32(&rank2),
      Err(Error::Mlx(_))
    ));
  }

  /// A non-finite decoder row is rejected at the MLX stage boundary with the
  /// same [`Error::SessionNonFiniteOutput`] the ORT decoder raises — including
  /// the lone `+inf` case, which greedy would otherwise always pick and which
  /// collapses the temperature path's softmax onto a uniform draw over every
  /// entry (masked entries included).
  #[test]
  fn last_position_logits_rejects_non_finite_row() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let flat = vec![1.0_f32, 2.0, 3.0, bad];
      let logits =
        Array::from_slice::<f32>(&flat, &(1usize, 1usize, 4usize)).expect("build logits");
      assert!(
        matches!(
          last_position_logits_f32(&logits),
          Err(Error::SessionNonFiniteOutput { stage: "decoder" })
        ),
        "a {bad} logit must be rejected at the decoder boundary"
      );
    }
    // A fully finite row still passes.
    let ok = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(1usize, 1usize, 2usize)).expect("ok");
    assert_eq!(
      last_position_logits_f32(&ok).expect("finite row"),
      vec![1.0_f32, 2.0]
    );
  }

  /// [`weights_parent`] returns the file's parent directory for a path with a
  /// directory component, and the current directory (`.`) — never the filesystem
  /// root — for a bare filename, so an explicit-format constructor handed
  /// `"model.safetensors"` reads `./config.json`.
  #[test]
  fn weights_parent_resolves_parent_else_current_dir() {
    assert_eq!(
      weights_parent(Path::new("/ckpt/model.safetensors")),
      Path::new("/ckpt")
    );
    assert_eq!(
      weights_parent(Path::new("ckpt/model.npz")),
      Path::new("ckpt")
    );
    // A bare filename has an empty parent; it must map to `.`, not `""`.
    assert_eq!(weights_parent(Path::new(MLX_SAFETENSORS)), Path::new("."));
  }

  /// A `config.json` that PARSES but fails the full [`ModelConfig::validate`] on
  /// a NON-`hidden_size` field — here a wrong top-level `model_type` (validate
  /// pins it to `"lfm2-vl"` / `"lfm2_vl"`) — is rejected with a typed
  /// [`Error::Mlx`] BEFORE any weight is loaded. The temp dir holds ONLY the
  /// malformed `config.json` (no readable `model.safetensors`), so a `from_dir`
  /// that nonetheless errors proves the full config validation runs ahead of the
  /// (expensive) weight load, not after it in `from_weights`.
  #[test]
  fn from_dir_rejects_invalid_config_before_weight_load() {
    let dir = std::env::temp_dir().join(format!(
      "lfm_mlx_invalid_cfg_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    // Valid JSON with present (defaulted) tower objects — so it PARSES — but a
    // top-level `model_type` the validator rejects. The two tower configs are
    // required keys; an empty `{}` for each supplies them with their defaulted,
    // valid geometry, so the failure is on a field OTHER than the dims, and no
    // weight file is written in the dir.
    std::fs::write(
      dir.join(MLX_CONFIG),
      br#"{"model_type": "not_lfm2_vl", "text_config": {}, "vision_config": {}}"#,
    )
    .expect("write config.json");
    // `Lfm2Vl` (inside `MlxBackend`) is not `Debug`, so use `.err()` rather than
    // `expect_err`.
    let err = MlxBackend::from_dir(&dir)
      .err()
      .expect("an invalid model_type must be rejected before the weight load");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
      matches!(err, Error::Mlx(_)),
      "expected Error::Mlx for an invalid config, got {err:?}"
    );
  }

  /// The checkpoint's tiling maps onto an [`ImageBudget`] one-for-one, and
  /// `do_image_splitting: false` folds into the degenerate `min = max = 1` tile
  /// band — the encoding upstream's `_preprocess` uses and the one `lfm`'s
  /// `pick_tile_grid` reads as "splitting off".
  #[test]
  fn checkpoint_budget_mirrors_config_and_folds_splitting_flag() {
    let config = ModelConfig::from_json(
      r#"{"text_config": {}, "vision_config": {}, "min_tiles": 2, "max_tiles": 8,
          "min_image_tokens": 32, "max_image_tokens": 128, "use_thumbnail": false,
          "max_pixels_tolerance": 1.5}"#,
    )
    .expect("parse config");
    let budget = checkpoint_budget(&config).expect("budget from config");
    assert_eq!(budget.min_tiles(), 2);
    assert_eq!(budget.max_tiles(), 8);
    assert_eq!(budget.min_image_tokens(), 32);
    assert_eq!(budget.max_image_tokens(), 128);
    assert!(!budget.use_thumbnail());
    assert_eq!(budget.max_pixels_tolerance(), 1.5);

    let no_split = ModelConfig::from_json(
      r#"{"text_config": {}, "vision_config": {}, "do_image_splitting": false,
          "min_tiles": 2, "max_tiles": 8}"#,
    )
    .expect("parse config");
    let budget = checkpoint_budget(&no_split).expect("budget from config");
    assert_eq!(
      (budget.min_tiles(), budget.max_tiles()),
      (1, 1),
      "do_image_splitting=false must fold into the degenerate 1..=1 tile band"
    );
  }

  /// A checkpoint whose tile grid exceeds the bundled tokenizer's 10×10
  /// `<|img_row_R_col_C|>` marker table cannot be rendered by `lfm` at all, so
  /// it is rejected rather than silently emitting markers that tokenize as
  /// plain text.
  #[test]
  fn checkpoint_budget_rejects_unrenderable_tile_grid() {
    let config = ModelConfig::from_json(
      r#"{"text_config": {}, "vision_config": {}, "min_tiles": 2, "max_tiles": 32}"#,
    )
    .expect("parse config");
    assert!(matches!(
      checkpoint_budget(&config),
      Err(Error::InvalidBudget(_))
    ));
  }

  /// The plan ratification compares every field `TilePlan` exposes and names
  /// the first disagreement. `TilePlan` cannot be constructed directly from
  /// outside `mlxrs`, so drive it through the real `plan_tiles` and mismatch
  /// the plan side instead.
  #[test]
  fn ratify_plan_names_the_disagreeing_field() {
    let processor = mlxrs::vlm::models::lfm2_vl::Lfm2VlProcessorConfig::new(396, 2, 16, 1024)
      .expect("processor config")
      .with_tiling(true, 2, 10, true, 64, 256, 16, 512, 2.0)
      .expect("tiling");
    // A large image: `plan_tiles` splits it into a multi-tile grid.
    let tiles = plan_tiles(2048, 2048, &processor).expect("plan tiles");
    assert!(tiles.is_split(), "2048x2048 must split under this config");

    // A single-tile plan cannot ratify against a split TilePlan, and the error
    // must name which field disagreed.
    let single = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(1, 1, 64, None));
    match ratify_plan(3, &single, &tiles) {
      Err(Error::ImagePlanMismatch {
        image, parameter, ..
      }) => {
        assert_eq!(image, 3, "the error must name the offending image");
        assert!(
          parameter.contains("rows") || parameter.contains("cols"),
          "expected a grid-shaped parameter, got {parameter:?}"
        );
      }
      other => panic!("expected ImagePlanMismatch, got {other:?}"),
    }

    // The plan that matches the checkpoint's own layout ratifies cleanly.
    let (grid_width, grid_height) = tiles.grid();
    let matching = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(
      grid_height as usize,
      grid_width as usize,
      256,
      tiles.has_thumbnail().then_some(256),
    ));
    ratify_plan(0, &matching, &tiles).expect("the checkpoint's own layout must ratify");
  }

  /// The re-read defence: a landscape image and its portrait transpose tile
  /// into MIRRORED grids with the SAME sub-image count and the SAME per-tile
  /// token count, so every quantity [`verify_sub_images`] compares is
  /// identical — a 1920×1080 file replaced by a 1080×1920 one between the
  /// header read and the decode would splice row-major features from the new
  /// layout under markers rendered for the old. Only the grid comparison
  /// [`MlxBackend::ratify_against_checkpoint`] runs on the DECODED dimensions
  /// can refuse it, and it must name the offending axis.
  #[test]
  fn ratify_plan_catches_a_transposed_grid_with_equal_token_totals() {
    let processor = mlxrs::vlm::models::lfm2_vl::Lfm2VlProcessorConfig::new(396, 2, 16, 1024)
      .expect("processor config")
      .with_tiling(true, 2, 10, true, 64, 256, 16, 512, 2.0)
      .expect("tiling");

    // `plan_tiles(height, width, …)`: the planned image, then the transpose the
    // execute-time decode would see.
    let planned = plan_tiles(1080, 1920, &processor).expect("plan landscape");
    let decoded = plan_tiles(1920, 1080, &processor).expect("plan portrait");
    assert!(planned.is_split() && decoded.is_split());
    assert_eq!(
      planned.grid(),
      {
        let (w, h) = decoded.grid();
        (h, w)
      },
      "the two must be exact transposes, or this is not the equal-token case"
    );
    assert_ne!(
      planned.grid(),
      decoded.grid(),
      "a square grid would make the swap undetectable AND harmless — pick a non-square one"
    );
    assert_eq!(
      planned.sub_image_count().expect("planned sub-images"),
      decoded.sub_image_count().expect("decoded sub-images"),
      "the sub-image count must be equal, so the count gates cannot see the swap"
    );

    let (grid_width, grid_height) = planned.grid();
    let plan = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(
      grid_height as usize,
      grid_width as usize,
      256,
      planned.has_thumbnail().then_some(256),
    ));
    // Same dimensions → ratifies. Transposed dimensions → refused by axis.
    ratify_plan(0, &plan, &planned).expect("the planned layout must ratify");
    match ratify_plan(7, &plan, &decoded) {
      Err(Error::ImagePlanMismatch {
        image,
        parameter,
        planned: p,
        produced,
      }) => {
        assert_eq!(image, 7, "the error must name the offending image");
        assert!(
          parameter.contains("rows") || parameter.contains("cols"),
          "expected a grid axis, got {parameter:?}"
        );
        assert_ne!(p, produced, "the named values must actually differ");
      }
      other => panic!("a transposed grid must be refused, got {other:?}"),
    }
  }

  /// `use_image_special_tokens: false` is a checkpoint whose processor contract
  /// omits the `<|image_start|>` / `<|image_end|>` brackets this crate always
  /// renders and always budgets. Every count still matches — the brackets are
  /// not `<image>` tokens — so the plan-time and execute-time gates are blind to
  /// it; the load-time contract refuses it BY NAME instead. The default
  /// (`true`, and absent ⇒ `true`) passes, as do the geometry constants.
  #[test]
  fn baked_in_contract_refuses_disabled_image_special_tokens() {
    let parse = |json: &str| ModelConfig::from_json(json).expect("parse config");

    // A config with the crate's own geometry and the brackets turned OFF.
    let off = parse(
      r#"{"text_config": {}, "vision_config": {}, "image_token_index": 396,
          "use_image_special_tokens": false}"#,
    );
    match require_baked_in_contract(&off) {
      Err(Error::MlxTilingMismatch {
        parameter,
        lfm_value,
        checkpoint_value,
      }) => {
        assert_eq!(parameter, "use_image_special_tokens");
        assert_eq!(lfm_value.as_str(), "true");
        assert_eq!(checkpoint_value.as_str(), "false");
      }
      other => panic!("a false bracketing policy must be refused by name, got {other:?}"),
    }

    // Explicit `true` and an absent key (upstream's default) both pass.
    let on = parse(
      r#"{"text_config": {}, "vision_config": {}, "image_token_index": 396,
          "use_image_special_tokens": true}"#,
    );
    require_baked_in_contract(&on).expect("use_image_special_tokens=true must pass");
    let absent = parse(r#"{"text_config": {}, "vision_config": {}, "image_token_index": 396}"#);
    require_baked_in_contract(&absent).expect("an absent key defaults to true and must pass");

    // The geometry half still refuses by name.
    let bad_tile = parse(
      r#"{"text_config": {}, "vision_config": {}, "image_token_index": 396,
          "tile_size": 384}"#,
    );
    match require_baked_in_contract(&bad_tile) {
      Err(Error::MlxTilingMismatch { parameter, .. }) => assert_eq!(parameter, "tile_size"),
      other => panic!("expected a tile_size mismatch, got {other:?}"),
    }
  }
}
