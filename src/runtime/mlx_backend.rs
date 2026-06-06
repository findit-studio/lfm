//! MLX (mlxrs) inference backend for LFM2.5-VL — Apple-Silicon only.
//!
//! This is the macOS/arm64 alternative to the default `ort` (ONNX Runtime)
//! decoder/vision/embed path. It is compiled only on
//! `aarch64-apple-darwin` (and only under the `inference` + `decoders`
//! features the rest of the [`Backend`] seam lives under), because `mlxrs`
//! binds the Metal-backed MLX C++ runtime through `mlx-c` FFI and has no
//! other target. There is **no** `mlx` Cargo feature — the backend is
//! selected automatically by platform + checkpoint shape (see Cargo.toml and
//! [`Engine`](crate::Engine)'s `from_dir`).
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
//! # Preprocessing
//!
//! The MLX path uses `mlxrs`'s **own** NaFlex tiling/preprocessing
//! ([`Lfm2Vl::split_image`]) rather than lfm's ONNX [`Preprocessor`] — the
//! patch geometry, normalization, and grid-token math are the model's, so the
//! spliced image features bind to the positions `get_input_embeddings` expects.
//! Images are decoded EXIF-aware via lfm's existing
//! [`decode_with_orientation`](crate::preproc::decode_with_orientation) /
//! [`decode_bytes_with_orientation`](crate::preproc::decode_bytes_with_orientation),
//! then handed to `split_image` as interleaved RGB bytes.

use std::path::Path;

use mlxrs::{
  Array, Dtype,
  lm::{cache::KvCache, model::Model as LmModel},
  vlm::models::lfm2_vl::{Lfm2Vl, Lfm2VlImageInputs, config::ModelConfig},
};

use crate::{
  error::{Error, Result},
  preproc::Preprocessor,
  runtime::backend::Backend,
};

/// The MLX-format config file name inside a checkpoint directory (the `mlxrs`
/// checkpoint marker, paired with a weight file).
pub(crate) const MLX_CONFIG: &str = "config.json";

/// The MLX-format safetensors weights file name — the always-available baseline
/// weight format. Its presence (with [`MLX_CONFIG`], a weight set in any enabled
/// format, and the absence of the ONNX graphs) is one signal
/// [`Engine::from_dir`](crate::Engine) routes to the MLX backend on Apple
/// Silicon.
pub(crate) const MLX_SAFETENSORS: &str = "model.safetensors";

/// The legacy single-file safetensors weights name. Some older MLX checkpoints
/// ship their weights as `weights.safetensors` rather than `model.safetensors`;
/// [`mlxrs::io::load_weights_from_dir`] accepts it as a fallback tier, so its
/// presence (with [`MLX_CONFIG`]) also routes to the MLX backend.
pub(crate) const MLX_SAFETENSORS_LEGACY: &str = "weights.safetensors";

/// The sharded-checkpoint index file name. A multi-shard safetensors export
/// (`model-00001-of-0000N.safetensors` + …) ships a `model.safetensors.index.json`
/// weight map instead of a single `model.safetensors`; [`mlxrs::io::load_weights_from_dir`]
/// loads it, so its presence (with [`MLX_CONFIG`]) also routes to the MLX backend.
pub(crate) const MLX_SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// The per-layer quantization marker `mlxrs` (and mlx-lm / mlx-vlm) use: a
/// quantized `nn.Linear` / `nn.Embedding` stores its packed weight alongside a
/// sibling `<prefix>.scales` tensor. Its presence ANYWHERE in the loaded weight
/// map is the sole signal that the checkpoint is quantized — exactly the
/// convention [`Lfm2Vl::from_weights`] keys its per-layer dense-vs-quantized
/// choice on.
const QUANT_SCALES_SUFFIX: &str = ".scales";

/// Report whether `dir` holds an MLX weight set in any ENABLED format:
/// a sharded `model.safetensors.index.json` always; a single `model.safetensors`
/// (or the legacy `weights.safetensors`) always; a `*.npz` only under the `npz`
/// feature; a `*.gguf` only under the `gguf` feature. Mirrors the formats
/// [`mlxrs::io::load_weights_from_dir`] loads, so routing and loading agree on
/// which checkpoints count. A dir with only `model.npz` therefore routes to MLX
/// iff `npz` is on.
fn has_mlx_weights(dir: &Path) -> bool {
  if dir.join(MLX_SAFETENSORS_INDEX).is_file() {
    return true;
  }
  if dir.join(MLX_SAFETENSORS).is_file() {
    return true;
  }
  if dir.join(MLX_SAFETENSORS_LEGACY).is_file() {
    return true;
  }
  #[cfg(feature = "npz")]
  if has_extension(dir, "npz") {
    return true;
  }
  #[cfg(feature = "gguf")]
  if has_extension(dir, "gguf") {
    return true;
  }
  false
}

/// Whether `dir` contains at least one file with the given `extension`. Only
/// referenced from the `npz`/`gguf` arms of [`has_mlx_weights`], so it is
/// `cfg`-elided on a default (safetensors-only) build.
#[cfg(any(feature = "npz", feature = "gguf"))]
fn has_extension(dir: &Path, extension: &str) -> bool {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return false;
  };
  entries.flatten().any(|entry| {
    let path = entry.path();
    path.extension().and_then(|e| e.to_str()) == Some(extension) && path.is_file()
  })
}

/// Probe `dir` and report whether the MLX backend should load it: `true` iff it
/// contains an MLX `config.json` and a weight file in any enabled format (see
/// [`has_mlx_weights`]) AND **neither** of `required_onnx` (the ONNX graph file
/// name(s) the ORT path loads) is a file in `dir`.
///
/// `config.json` + `model.safetensors` are also the standard HuggingFace
/// source-asset names, so the required ONNX graph is the disambiguator: a
/// directory that ships those HF sources next to the `*.onnx` graphs the ORT
/// backend reads is an ONNX checkpoint and must route to ONNX. Because the MLX
/// backend never reads the `*.onnx` graphs while the ONNX backend does, the
/// presence of a required graph means "this is an ONNX checkpoint" and selects
/// ONNX. Pure filesystem existence checks (no file I/O): the winning constructor
/// does the real load and surfaces a typed error if the checkpoint is malformed.
pub(crate) fn prefer_mlx(dir: &Path, required_onnx: &[&str]) -> bool {
  dir.join(MLX_CONFIG).is_file()
    && has_mlx_weights(dir)
    && !required_onnx.iter().any(|onnx| dir.join(onnx).is_file())
}

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

impl Backend for MlxBackend {
  /// On-device prompt / per-token embeddings (an mlx `Array`); no host copy.
  type Embeds = Array;
  /// The LFM2 heterogeneous per-layer cache (`Lfm2Vl::make_cache`).
  type Cache = Vec<Box<dyn KvCache>>;

  fn make_cache(&self) -> Result<Self::Cache> {
    Ok(self.model.make_cache())
  }

  /// Build the full prompt `inputs_embeds` on device: embed the token ids and
  /// splice each image's NaFlex features in, using `mlxrs`'s OWN tiling /
  /// preprocessing.
  ///
  /// The ORT-specific `preproc` / `grids` / `image_positions` are intentionally
  /// **ignored** here: the MLX path drives `mlxrs`'s native NaFlex
  /// [`split_image`](Lfm2Vl::split_image) + the mask-driven
  /// `get_input_embeddings` splice (which locates the `<image>`-token positions
  /// itself from `input_ids` and the model's `image_token_index`). Each image is
  /// decoded EXIF-aware (lfm's `decode_*_with_orientation`), projected to
  /// interleaved RGB bytes, tiled by `split_image`, and the resulting per-tile
  /// [`Lfm2VlImageInputs`] are collected; `get_input_embeddings` then embeds the
  /// ids and merges the concatenated image features.
  fn prepare_prompt_embeds(
    &mut self,
    _preproc: &Preprocessor,
    input_ids: &[i64],
    images: &[crate::ImageInput<'_>],
    _grids: &[crate::preproc::TileGrid],
    _image_positions: &[usize],
  ) -> Result<Self::Embeds> {
    // Collect every image's tiled NaFlex sub-image inputs, in image order, then
    // sub-image order (the same order `expand_image_tokens` lays the
    // `<image>`-token runs the embed splices into).
    let mut all_inputs: Vec<Lfm2VlImageInputs> = Vec::new();
    for img in images {
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
      let tiles = self
        .model
        .split_image(&rgb, width, height)
        .map_err(Error::from_mlx)?;
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
/// # Errors
/// - [`Error::Mlx`] if the logits are not rank-3 `(1, seq, vocab)` with `seq >=
///   1`, or for any slice / astype / eval / read failure.
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
  row.to_vec::<f32>().map_err(Error::from_mlx)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A directory holding BOTH an MLX `config.json` and `model.safetensors`, with
  /// neither ONNX graph present, is an MLX checkpoint — `prefer_mlx` routes it to
  /// the MLX backend.
  #[test]
  fn prefer_mlx_true_for_mlx_checkpoint_dir() {
    let tmp = std::env::temp_dir().join(format!("lfm_mlx_probe_mlx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS), b"\0").expect("write model.safetensors");
    assert!(
      prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]),
      "config.json + model.safetensors present (no ONNX graphs) must select MLX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

  /// When the ONNX graphs are present, the directory is an ONNX checkpoint and
  /// routes to ONNX — even if it also carries `config.json` + `model.safetensors`
  /// (which double as the HuggingFace source-asset names). The ONNX graphs are
  /// the disambiguator, so `prefer_mlx` is `false`.
  #[test]
  fn prefer_mlx_false_when_onnx_graphs_present() {
    let tmp = std::env::temp_dir().join(format!("lfm_mlx_probe_onnx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS), b"\0").expect("write model.safetensors");
    std::fs::write(tmp.join("vision_encoder.onnx"), b"\0").expect("write vision onnx");
    std::fs::write(tmp.join("decoder_model_merged.onnx"), b"\0").expect("write decoder onnx");
    assert!(
      !prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]),
      "the ONNX graphs disambiguate: a dir carrying them must route to ONNX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

  /// A bare `config.json` without `model.safetensors` is NOT an MLX checkpoint —
  /// `prefer_mlx` is `false`.
  #[test]
  fn prefer_mlx_false_without_weights() {
    let tmp = std::env::temp_dir().join(format!("lfm_mlx_probe_noweights_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    assert!(
      !prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]),
      "a lone config.json (no model.safetensors) must NOT select MLX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

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

  /// Create a fresh temp dir for a format-detection test, named for `tag`.
  fn detect_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "lfm_mlx_detect_{tag}_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp detect dir");
    dir
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
    let dir = detect_dir("invalid_cfg");
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
    assert_eq!(
      weights_parent(Path::new("model.safetensors")),
      Path::new(".")
    );
  }

  /// A SHARDED MLX checkpoint — `config.json` + `model.safetensors.index.json`
  /// (the weight map for a multi-shard export) with NO single `model.safetensors`
  /// and no ONNX graph — is an MLX checkpoint: `prefer_mlx` routes it to MLX,
  /// because [`mlxrs::io::load_weights_from_dir`] loads the sharded layout via
  /// the index.
  #[test]
  fn prefer_mlx_true_for_sharded_index_only() {
    let tmp = detect_dir("probe_shard");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS_INDEX), b"{}").expect("write index.json");
    let routed = prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
      routed,
      "config.json + model.safetensors.index.json (sharded, no ONNX graph) must select MLX"
    );
  }

  /// A LEGACY single-file MLX checkpoint — `config.json` + `weights.safetensors`
  /// (the older single-file name) with NO `model.safetensors` and no ONNX graph
  /// — is an MLX checkpoint: `prefer_mlx` routes it to MLX, because
  /// [`mlxrs::io::load_weights_from_dir`] accepts `weights.safetensors` as a
  /// fallback tier.
  #[test]
  fn prefer_mlx_true_for_legacy_weights_safetensors() {
    let tmp = detect_dir("probe_legacy");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS_LEGACY), b"\0").expect("write weights.safetensors");
    let routed = prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
      routed,
      "config.json + weights.safetensors (legacy single-file, no ONNX graph) must select MLX"
    );
  }

  /// A dir with only `config.json` + `model.npz` (no safetensors, no ONNX graph)
  /// is recognized as an MLX checkpoint by `prefer_mlx` **iff** the `npz` feature
  /// is on — the routing widens to the same formats the loader accepts. Without
  /// `npz`, the `.npz` is not a recognized weight file and `prefer_mlx` is
  /// `false`.
  #[test]
  fn prefer_mlx_npz_only_routes_iff_npz_feature() {
    let tmp = detect_dir("probe_npz");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join("model.npz"), b"\0").expect("write model.npz");
    let routed = prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(
      routed,
      cfg!(feature = "npz"),
      "a config.json + model.npz dir must route to MLX iff the npz feature is on"
    );
  }

  /// Same contract for gguf: a `config.json` + `model.gguf` dir routes to MLX
  /// iff the `gguf` feature is on.
  #[test]
  fn prefer_mlx_gguf_only_routes_iff_gguf_feature() {
    let tmp = detect_dir("probe_gguf");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join("model.gguf"), b"\0").expect("write model.gguf");
    let routed = prefer_mlx(&tmp, &["vision_encoder.onnx", "decoder_model_merged.onnx"]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(
      routed,
      cfg!(feature = "gguf"),
      "a config.json + model.gguf dir must route to MLX iff the gguf feature is on"
    );
  }
}
