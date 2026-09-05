//! Backend seam over the vision / embed / decoder / KV-cache stack.
//!
//! [`generate`](crate::generate::generate) drives the whole VLM pipeline
//! through a single [`Backend`] trait instead of three concrete ORT
//! component handles. The loop stays backend-agnostic: it owns
//! tokenization, `<image>`-token position discovery, the sampler,
//! detokenization, EOS handling, admission/DoS guards, and loop control.
//! Everything that touches model weights — text embedding, per-image
//! vision encoding + splice, and the decoder forward — goes through the
//! trait.
//!
//! The design constraint that shapes this seam: the prompt embeddings AND
//! the KV cache are **backend-internal associated types** ([`Backend::Embeds`]
//! and [`Backend::Cache`]). Only the decoder logits (`Vec<f32>`) and token
//! ids cross the abstraction boundary. For the ORT backend the embeds are a
//! host `Vec<f32>` buffer and the cache is the ONNX [`KvCache`]; a future
//! on-device backend can keep both as device tensors with no host
//! round-trip, because nothing outside the trait ever inspects them.

use crate::{
  error::Result,
  options::BackendKind,
  preproc::{ImagePlan, Preprocessor},
};

// The ORT-backed component wrappers — `ort` is mandatory on the ALLOW-listed
// targets (Linux x86_64/aarch64-gnu, Windows x86_64/aarch64-msvc) and optional on
// aarch64-apple-darwin behind the `ort` feature; every other target compiles
// neither. See the `ort_backend` cfg (build.rs) and the target tables in
// Cargo.toml.
#[cfg(ort_backend)]
use crate::runtime::{
  decoder::{Decoder, KvCache},
  embed_tokens::EmbedTokens,
  vision::VisionEncoder,
};

/// Embedding dimension for text and vision outputs (1024 for LFM2.5-VL).
#[cfg(ort_backend)]
const EMBED_DIM: usize = 1024;

/// Drives the model-weight stage of generation: text embedding, vision
/// encoding + splice, and the decoder forward.
///
/// The embeds and KV cache are associated types so each backend can hold
/// them in whatever form avoids host round-trips. Only logits and token
/// ids cross the boundary, so the [`generate`](crate::generate::generate)
/// loop never manipulates embedding buffers directly.
pub(crate) trait Backend {
  /// Prompt + per-token embeddings, in whatever form the backend prefers
  /// (host `Vec<f32>` for ORT; an on-device tensor for a future backend).
  type Embeds;

  /// Per-call KV / conv cache. Owned by the loop, threaded back into
  /// [`Backend::decoder_step`].
  type Cache;

  /// Which backend this is. Reported by
  /// [`Engine::backend`](crate::Engine::backend) so auto-selection is
  /// observable.
  fn kind(&self) -> BackendKind;

  /// Construct a fresh, zero-initialized cache for one generation call.
  fn make_cache(&self) -> Result<Self::Cache>;

  /// Produce the ONE authoritative [`ImagePlan`] for an image of the given
  /// (EXIF-corrected) header dimensions.
  ///
  /// The plan is made by the backend that will execute it, because the tiling
  /// that decides the marker layout must be the same tiling that produces the
  /// feature rows — for the ONNX backend that is `lfm`'s ported
  /// [`pick_tile_grid`](crate::preproc::tile_grid::pick_tile_grid); for the MLX
  /// backend it is additionally ratified against the checkpoint's own planner.
  /// The [`generate`](crate::generate::generate) loop then renders the prompt,
  /// runs admission control, and hands the very same plans back to
  /// [`prepare_prompt_embeds`](Self::prepare_prompt_embeds).
  ///
  /// `index` is the image's position in the request, used only to name the
  /// image in errors.
  fn plan_image(
    &self,
    preproc: &Preprocessor,
    index: usize,
    width: u32,
    height: u32,
  ) -> Result<ImagePlan>;

  /// Build the full prompt `inputs_embeds`: embed the token ids, encode
  /// each image, and splice the per-image vision embeds in at the given
  /// `<image>`-token positions.
  ///
  /// `image_positions` lists the indices (into `input_ids`) of the
  /// `<image>` placeholder tokens, in order. `plans` carries the authoritative
  /// [`ImagePlan`] per image (same order as `images`) — the plans the prompt's
  /// markers were rendered from. The backend MUST re-derive its layout from the
  /// pixels it actually decoded and reject any disagreement with
  /// [`Error::ImagePlanMismatch`](crate::Error::ImagePlanMismatch); a
  /// disagreement whose total token count happens to line up would otherwise
  /// bind features to the wrong positions silently. `preproc` decodes +
  /// patchifies each image one at a time so peak memory stays at O(1 image's
  /// pixel buffer).
  fn prepare_prompt_embeds(
    &mut self,
    preproc: &Preprocessor,
    input_ids: &[i64],
    images: &[crate::ImageInput<'_>],
    plans: &[ImagePlan],
    image_positions: &[usize],
  ) -> Result<Self::Embeds>;

  /// Embed a single newly-sampled token id for the decode loop.
  fn embed_one(&mut self, token_id: i64) -> Result<Self::Embeds>;

  /// Decoder forward over `embeds` (the prompt at prefill, one token per
  /// decode step). Advances `cache` in place and returns the HOST logits
  /// for the sampler. `seq_len` is the number of new positions this step.
  fn decoder_step(
    &mut self,
    cache: &mut Self::Cache,
    embeds: &Self::Embeds,
    seq_len: usize,
  ) -> Result<Vec<f32>>;
}

/// Host-side embedding buffer used by [`OrtBackend`]: a flat
/// `[positions × 1024]` `f32` slab in the ONNX `inputs_embeds` layout.
#[cfg(ort_backend)]
pub(crate) struct OrtEmbeds(Vec<f32>);

#[cfg(ort_backend)]
impl OrtEmbeds {
  /// Borrow the flat buffer for the decoder's `inputs_embeds` input.
  pub(crate) fn as_slice(&self) -> &[f32] {
    &self.0
  }
}

/// ONNX/`ort`-backed [`Backend`]. Owns the three ONNX component sessions
/// (vision encoder, embed-tokens, decoder); embeds are host `Vec<f32>`
/// and the cache is the ONNX [`KvCache`].
///
/// `ort` is mandatory on the ALLOW-listed targets (Linux x86_64/aarch64-gnu,
/// Windows x86_64/aarch64-msvc) and optional on aarch64-apple-darwin behind the `ort`
/// feature (see `Cargo.toml`) — hence the `ort_backend` gate on this whole
/// type.
#[cfg(ort_backend)]
pub(crate) struct OrtBackend {
  vision: VisionEncoder,
  embed: EmbedTokens,
  decoder: Decoder,
}

#[cfg(ort_backend)]
impl OrtBackend {
  /// Construct from the three ONNX component sessions.
  pub(crate) fn new(vision: VisionEncoder, embed: EmbedTokens, decoder: Decoder) -> Self {
    Self {
      vision,
      embed,
      decoder,
    }
  }
}

#[cfg(ort_backend)]
impl Backend for OrtBackend {
  type Embeds = OrtEmbeds;
  type Cache = KvCache;

  fn kind(&self) -> BackendKind {
    BackendKind::Onnx
  }

  fn make_cache(&self) -> Result<Self::Cache> {
    self.decoder.new_cache()
  }

  /// The ONNX road's plan is `lfm`'s own ported tiling under the engine's
  /// [`ImageBudget`](crate::ImageBudget) — the same
  /// [`pick_tile_grid`](crate::preproc::tile_grid::pick_tile_grid) the
  /// per-image [`Preprocessor::preprocess`] re-runs on the decoded pixels, so
  /// the plan and the execution are the same function of the same budget.
  fn plan_image(
    &self,
    preproc: &Preprocessor,
    _index: usize,
    width: u32,
    height: u32,
  ) -> Result<ImagePlan> {
    let grid = crate::preproc::tile_grid::pick_tile_grid(width, height, preproc.budget())?;
    Ok(ImagePlan::from_placeholder(grid.to_placeholder_info()))
  }

  fn prepare_prompt_embeds(
    &mut self,
    preproc: &Preprocessor,
    input_ids: &[i64],
    images: &[crate::ImageInput<'_>],
    plans: &[ImagePlan],
    image_positions: &[usize],
  ) -> Result<Self::Embeds> {
    let seq_len = input_ids.len();
    let mut text_embeds: Vec<f32> = self.embed.run(input_ids)?;
    debug_assert_eq!(text_embeds.len(), seq_len * EMBED_DIM);

    // For each image: preprocess (decode + smart_resize + flatten_to_patches)
    // → vision encode → splice → DROP pixel buffer at iteration end. The
    // per-image PreprocessedImage is local to each iteration and freed when
    // the loop body exits, so peak memory is O(1 image's pixel buffer)
    // instead of O(N).
    let mut pos_cursor: usize = 0;
    for (index, (img, plan)) in images.iter().zip(plans.iter()).enumerate() {
      // Decode + preprocess just this image. The decoded DynamicImage and
      // the resulting PreprocessedImage both go out of scope at the end
      // of this iteration, freeing their pixel buffers.
      let decoded = match img {
        #[cfg(not(target_arch = "wasm32"))]
        crate::ImageInput::Path(p) => crate::preproc::decode_with_orientation(p)?,
        crate::ImageInput::Bytes(b) => crate::preproc::decode_bytes_with_orientation(b)?,
      };
      let preprocessed_img = preproc.preprocess(&decoded)?;
      drop(decoded); // free the source RGB buffer before vision.run

      // The prompt was rendered from `plan` (computed from header dimensions,
      // EXIF-corrected by image_dimensions); preprocessed_img's layout comes
      // from the actually-decoded image. With the EXIF fix in image_dimensions
      // these must agree; if they don't, markers and features would bind to
      // wrong spatial positions even when total token counts happen to match
      // (e.g., a 4×2 layout vs 2×4 layout both have 8 main tiles + the same
      // thumbnail tokens).
      let expected_info = *plan.placeholder();
      let actual_info = preprocessed_img.to_placeholder_info();
      if expected_info != actual_info {
        return Err(crate::error::Error::ImageGridLayoutMismatch {
          expected_rows: expected_info.rows(),
          expected_cols: expected_info.cols(),
          actual_rows: actual_info.rows(),
          actual_cols: actual_info.cols(),
        });
      }
      // The layouts match field-for-field, so the token counts must too; check
      // it anyway, because this is the number the splice indexes with.
      if preprocessed_img.num_image_tokens() != plan.image_tokens() {
        return Err(crate::error::Error::ImagePlanMismatch {
          image: index,
          parameter: "image tokens",
          planned: plan.image_tokens(),
          produced: preprocessed_img.num_image_tokens(),
        });
      }
      let n_img_tokens = plan.image_tokens();
      let vision_embeds: Vec<f32> = self.vision.run(&preprocessed_img)?;
      drop(preprocessed_img); // free pixel_values before splicing

      // Vision encoder returns [num_image_tokens × 1024] flat.
      if vision_embeds.len() != n_img_tokens * EMBED_DIM {
        return Err(crate::error::Error::SessionShapeMismatch {
          input: "image_features",
          expected: "num_image_tokens * 1024",
          got: vec![vision_embeds.len() as i64],
        });
      }

      // Splice vision embedding for each image-token position.
      for k in 0..n_img_tokens {
        let tok_pos = image_positions[pos_cursor + k];
        let dst_start = tok_pos * EMBED_DIM;
        let src_start = k * EMBED_DIM;
        text_embeds[dst_start..dst_start + EMBED_DIM]
          .copy_from_slice(&vision_embeds[src_start..src_start + EMBED_DIM]);
      }
      pos_cursor += n_img_tokens;
    }

    Ok(OrtEmbeds(text_embeds))
  }

  fn embed_one(&mut self, token_id: i64) -> Result<Self::Embeds> {
    Ok(OrtEmbeds(self.embed.run(&[token_id])?))
  }

  fn decoder_step(
    &mut self,
    cache: &mut Self::Cache,
    embeds: &Self::Embeds,
    seq_len: usize,
  ) -> Result<Vec<f32>> {
    self.decoder.step(cache, embeds.as_slice(), seq_len)
  }
}

/// The backend [`Engine`](crate::Engine) drives generation through.
///
/// The ORT variant is compiled whenever `ort_backend` holds — mandatory on
/// the ALLOW-listed targets (Linux x86_64/aarch64-gnu, Windows x86_64/aarch64-msvc),
/// opt-in on aarch64-apple-darwin behind the `ort` feature (see `Cargo.toml`). On
/// Apple Silicon a second on-device
/// [`MlxBackend`](crate::runtime::mlx_backend::MlxBackend) arm is ALWAYS
/// compiled in and auto-selected by checkpoint shape (see
/// [`Engine::from_dir`](crate::Engine)); on the allow-listed targets and on
/// aarch64-apple-darwin, at least one of the two arms is present. Outside both
/// groups — e.g. `x86_64-apple-darwin` with the `inference` feature on —
/// `BackendImpl` has ZERO variants: `backend_available` (build.rs) is false
/// there, and the [`Backend`] impl below adds one `#[cfg(not(backend_available))]`
/// fallback arm per method for exactly that case (see the comment on
/// [`Backend::kind`] for why the fallback has to exist at all, not merely
/// document the gap). No `Engine` constructor ever actually reaches such a
/// value, though: `Engine::from_dir` and `require_backend_compiled`
/// (`engine.rs`) refuse with `Error::BackendUnavailable` first, under the
/// same condition. The enum implements [`Backend`] by
/// delegating to the active variant, so [`generate`](crate::generate::generate)
/// stays generic over [`Backend`]. The associated [`Backend::Embeds`] /
/// [`Backend::Cache`] types are themselves enums ([`EngineEmbeds`] /
/// [`EngineCache`]) so each variant keeps its embeds + cache in its own native
/// form — the ORT variant on the host, the MLX variant on-device — with no
/// cross-variant conversion.
///
/// An [`Engine`](crate::Engine) holds exactly one `BackendImpl` (never an array
/// or a hot-path collection of them), so the inter-variant size difference is
/// not a layout concern — hence the `large_enum_variant` allow when both
/// variants are compiled in. The larger MLX model is still boxed to keep the
/// moved-around enum small.
#[cfg_attr(all(mlx_backend, ort_backend), allow(clippy::large_enum_variant))]
pub(crate) enum BackendImpl {
  /// ONNX/`ort` backend.
  #[cfg(ort_backend)]
  Ort(OrtBackend),
  /// MLX (`mlxrs`) Metal backend — Apple Silicon only. Boxed because the loaded
  /// [`Lfm2Vl`](mlxrs::vlm::models::lfm2_vl::Lfm2Vl) model is much larger than
  /// the ORT variant's session handles; boxing keeps `BackendImpl` small (one
  /// `Engine` holds exactly one backend, so the indirection cost is negligible).
  #[cfg(mlx_backend)]
  Mlx(Box<crate::runtime::mlx_backend::MlxBackend>),
}

/// Per-variant embeds for [`BackendImpl`]. See [`BackendImpl`] for why the
/// associated embeds type is an enum.
pub(crate) enum EngineEmbeds {
  /// Host `Vec<f32>` embeds for the ORT backend.
  #[cfg(ort_backend)]
  Ort(OrtEmbeds),
  /// On-device mlx [`Array`](mlxrs::Array) embeds for the MLX backend.
  #[cfg(mlx_backend)]
  Mlx(mlxrs::Array),
}

/// Per-variant cache for [`BackendImpl`]. See [`BackendImpl`] for why the
/// associated cache type is an enum.
pub(crate) enum EngineCache {
  /// ONNX [`KvCache`] for the ORT backend.
  #[cfg(ort_backend)]
  Ort(KvCache),
  /// The LFM2 heterogeneous per-layer KV/conv cache for the MLX backend.
  #[cfg(mlx_backend)]
  Mlx(Vec<Box<dyn mlxrs::lm::cache::KvCache>>),
}

impl Backend for BackendImpl {
  type Embeds = EngineEmbeds;
  type Cache = EngineCache;

  fn kind(&self) -> BackendKind {
    match self {
      #[cfg(ort_backend)]
      Self::Ort(b) => b.kind(),
      #[cfg(mlx_backend)]
      Self::Mlx(b) => b.kind(),
      // `backend_available` (build.rs) is false on a target outside the ORT
      // allow-list that is also not aarch64-apple-darwin — e.g. `x86_64-apple-darwin`
      // with the `inference` feature on. The allow-list conversion (D8) made
      // this a real, reachable build configuration rather than a
      // hypothetical one: both arms above are `#[cfg]`-gated out and
      // `BackendImpl` has zero variants there.
      //
      // Rust still requires this match to be exhaustive, because a
      // REFERENCE is always considered inhabited regardless of the
      // referent's own inhabitedness (`rustc --explain E0004`: "references
      // are always considered inhabited") — `&BackendImpl` is never "empty"
      // to the exhaustiveness checker even when `BackendImpl` itself
      // provably is, so an empty `match self {}` here is refused as
      // non-exhaustive rather than accepted as dead code. `match *self {}`
      // dereferences first: `*self: BackendImpl` IS the provably-uninhabited
      // value type, so the empty match is accepted there — and unlike
      // `unreachable!()`, the compiler itself verifies this can never
      // execute rather than merely trusting an assertion. It cannot execute
      // in practice either way: every `Engine` constructor already refuses
      // with `Error::BackendUnavailable` before a `BackendImpl` value would
      // need to exist on such a target (see `engine.rs`'s `from_dir` and
      // `require_backend_compiled`, gated on the same `backend_available`
      // condition spelled as `not(ort_backend) && not(macos-aarch64)`).
      //
      // The other five methods below hit the same zero-arm case; each adds
      // its own `#[cfg(not(backend_available))]` fallback (an `Err` where
      // the return type allows one, since only this method's bare
      // `BackendKind` return forces the `match *self {}` form instead).
      #[cfg(not(backend_available))]
      _ => match *self {},
    }
  }

  fn make_cache(&self) -> Result<Self::Cache> {
    match self {
      #[cfg(ort_backend)]
      Self::Ort(b) => Ok(EngineCache::Ort(b.make_cache()?)),
      #[cfg(mlx_backend)]
      Self::Mlx(b) => Ok(EngineCache::Mlx(b.make_cache()?)),
      // See `kind` above for why this arm must exist at all.
      #[cfg(not(backend_available))]
      _ => Err(crate::error::Error::InvalidRequest(
        "no backend is compiled for this target (outside the ORT allow-list \
         and not aarch64-apple-darwin); unreachable in practice, since no `Engine` \
         can be constructed here either",
      )),
    }
  }

  // Every parameter but `self` is read only by the real arms below; on a
  // `not(backend_available)` build only the fallback arm compiles, and it
  // ignores them all (see `kind` above for why the arm exists at all).
  #[cfg_attr(not(backend_available), allow(unused_variables))]
  fn plan_image(
    &self,
    preproc: &Preprocessor,
    index: usize,
    width: u32,
    height: u32,
  ) -> Result<ImagePlan> {
    match self {
      #[cfg(ort_backend)]
      Self::Ort(b) => b.plan_image(preproc, index, width, height),
      #[cfg(mlx_backend)]
      Self::Mlx(b) => b.plan_image(preproc, index, width, height),
      // See `kind` above for why this arm must exist at all.
      #[cfg(not(backend_available))]
      _ => Err(crate::error::Error::InvalidRequest(
        "no backend is compiled for this target (outside the ORT allow-list \
         and not aarch64-apple-darwin); unreachable in practice, since no `Engine` \
         can be constructed here either",
      )),
    }
  }

  // See `plan_image` above: same reasoning, same fallback-only unused set.
  #[cfg_attr(not(backend_available), allow(unused_variables))]
  fn prepare_prompt_embeds(
    &mut self,
    preproc: &Preprocessor,
    input_ids: &[i64],
    images: &[crate::ImageInput<'_>],
    plans: &[ImagePlan],
    image_positions: &[usize],
  ) -> Result<Self::Embeds> {
    match self {
      #[cfg(ort_backend)]
      Self::Ort(b) => Ok(EngineEmbeds::Ort(b.prepare_prompt_embeds(
        preproc,
        input_ids,
        images,
        plans,
        image_positions,
      )?)),
      #[cfg(mlx_backend)]
      Self::Mlx(b) => Ok(EngineEmbeds::Mlx(b.prepare_prompt_embeds(
        preproc,
        input_ids,
        images,
        plans,
        image_positions,
      )?)),
      // See `kind` above for why this arm must exist at all.
      #[cfg(not(backend_available))]
      _ => Err(crate::error::Error::InvalidRequest(
        "no backend is compiled for this target (outside the ORT allow-list \
         and not aarch64-apple-darwin); unreachable in practice, since no `Engine` \
         can be constructed here either",
      )),
    }
  }

  // See `plan_image` above: same reasoning (`token_id` is the unused one here).
  #[cfg_attr(not(backend_available), allow(unused_variables))]
  fn embed_one(&mut self, token_id: i64) -> Result<Self::Embeds> {
    match self {
      #[cfg(ort_backend)]
      Self::Ort(b) => Ok(EngineEmbeds::Ort(b.embed_one(token_id)?)),
      #[cfg(mlx_backend)]
      Self::Mlx(b) => Ok(EngineEmbeds::Mlx(b.embed_one(token_id)?)),
      // See `kind` above for why this arm must exist at all.
      #[cfg(not(backend_available))]
      _ => Err(crate::error::Error::InvalidRequest(
        "no backend is compiled for this target (outside the ORT allow-list \
         and not aarch64-apple-darwin); unreachable in practice, since no `Engine` \
         can be constructed here either",
      )),
    }
  }

  // `self`, `cache` and `embeds` are all used as the match scrutinee tuple
  // regardless of which arms compile, but `seq_len` is read only by the real
  // arms; see `plan_image` above for the general reasoning.
  #[cfg_attr(not(backend_available), allow(unused_variables))]
  fn decoder_step(
    &mut self,
    cache: &mut Self::Cache,
    embeds: &Self::Embeds,
    seq_len: usize,
  ) -> Result<Vec<f32>> {
    match (self, cache, embeds) {
      #[cfg(ort_backend)]
      (Self::Ort(b), EngineCache::Ort(cache), EngineEmbeds::Ort(embeds)) => {
        b.decoder_step(cache, embeds, seq_len)
      }
      #[cfg(mlx_backend)]
      (Self::Mlx(b), EngineCache::Mlx(cache), EngineEmbeds::Mlx(embeds)) => {
        b.decoder_step(cache, embeds, seq_len)
      }
      // A cross-variant (backend, cache, embeds) mix is unreachable: the engine
      // builds the cache + embeds from the SAME backend variant it dispatches
      // on, so the tuple is always all-Ort or all-Mlx. The wildcard keeps the
      // match exhaustive when BOTH arms are compiled in (macOS/arm64 with the
      // `ort` feature on); everywhere else exactly one arm exists and is
      // exhaustive by itself, so the wildcard is compiled out (an always-taken
      // single arm plus this wildcard would be an unreachable-pattern error).
      #[cfg(all(mlx_backend, ort_backend))]
      _ => Err(crate::error::Error::InvalidRequest(
        "backend / cache / embeds variant mismatch (internal invariant violation)",
      )),
      // See `kind` above for why this arm must exist at all when NEITHER
      // backend is compiled: the tuple's components are all references, and
      // a reference is always considered inhabited regardless of its
      // referent, so the two real arms alone would leave this non-exhaustive
      // rather than dead code.
      #[cfg(not(backend_available))]
      (_, _, _) => Err(crate::error::Error::InvalidRequest(
        "no backend is compiled for this target (outside the ORT allow-list \
         and not aarch64-apple-darwin); unreachable in practice, since no `Engine` \
         can be constructed here either",
      )),
    }
  }
}
