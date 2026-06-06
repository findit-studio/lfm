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
//! The MLX backend consumes an **MLX-format checkpoint** (`config.json` +
//! `model.safetensors`), not the ONNX graphs. The loader reads the safetensors,
//! runs [`Lfm2Vl::sanitize`], and builds the model via [`Lfm2Vl::from_weights`].
//! A quantized checkpoint (e.g. the `LiquidAI/LFM2.5-VL-450M-MLX-8bit` export)
//! is detected by the `mlxrs` convention — the presence of any `<layer>.scales`
//! sibling — and loaded with the `(group_size, bits, mode)` parsed from the
//! `config.json` `quantization` block.
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
/// checkpoint marker, paired with [`MLX_WEIGHTS`]).
pub(crate) const MLX_CONFIG: &str = "config.json";

/// The MLX-format weights file name. Its presence (with [`MLX_CONFIG`], and the
/// absence of the ONNX graphs) is the signal [`Engine::from_dir`](crate::Engine)
/// routes to the MLX backend on Apple Silicon.
pub(crate) const MLX_WEIGHTS: &str = "model.safetensors";

/// The per-layer quantization marker `mlxrs` (and mlx-lm / mlx-vlm) use: a
/// quantized `nn.Linear` / `nn.Embedding` stores its packed weight alongside a
/// sibling `<prefix>.scales` tensor. Its presence ANYWHERE in the loaded weight
/// map is the sole signal that the checkpoint is quantized — exactly the
/// convention [`Lfm2Vl::from_weights`] keys its per-layer dense-vs-quantized
/// choice on.
const QUANT_SCALES_SUFFIX: &str = ".scales";

/// Probe `dir` and report whether the MLX backend should load it: `true` iff it
/// contains an MLX `config.json` and a `model.safetensors` AND **neither** of
/// `required_onnx` (the ONNX graph file name(s) the ORT path loads) is a file in
/// `dir`.
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
    && dir.join(MLX_WEIGHTS).is_file()
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
  /// Load an [`MlxBackend`] from an MLX checkpoint directory containing
  /// `config.json` and `model.safetensors`.
  ///
  /// Parses the [`ModelConfig`] off `config.json`, loads + [`sanitize`s the
  /// weights](Lfm2Vl::sanitize), discriminates dense vs quantized by the
  /// `.scales`-presence convention, and constructs the model via
  /// [`Lfm2Vl::from_weights`] (threading the parsed `(group_size, bits, mode)`
  /// quantization for a quantized checkpoint, `None` for a dense one).
  ///
  /// # Errors
  /// - [`Error::Io`] if `config.json` / `model.safetensors` cannot be read;
  /// - [`Error::Mlx`] for any `mlxrs` parse / load / construction failure
  ///   (malformed config, corrupt safetensors, weight/key mismatch).
  pub(crate) fn from_dir(dir: &Path) -> Result<Self> {
    let config_path = dir.join(MLX_CONFIG);
    let weights_path = dir.join(MLX_WEIGHTS);

    let config_json = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
    let config = ModelConfig::from_json(&config_json).map_err(Error::from_mlx)?;

    let raw = mlxrs::io::load_safetensors(&weights_path).map_err(Error::from_mlx)?;
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
    std::fs::write(tmp.join(MLX_WEIGHTS), b"\0").expect("write model.safetensors");
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
    std::fs::write(tmp.join(MLX_WEIGHTS), b"\0").expect("write model.safetensors");
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
}
