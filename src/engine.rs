//! Public Engine API. Holds runtime sessions + tokenizer + lazy
//! ParserFactory for schema-constrained sampling.
//!
//! The Engine is single-threaded (`&mut self` on every mutating method)
//! because the underlying ORT sessions are not `Sync`. Multi-engine
//! pools are out of scope for v0.1.
//!
//! ## llguidance 1.7.x setup (verified in Task 13)
//!
//! **TokenizerEnv adapter:**
//! `toktrie_hf_tokenizers::ByteTokenizer::from_file(tokenizer_path)?`
//! loads the tokenizer from `tokenizer.json` and extracts a byte-per-token representation.
//! `.into_tok_env(None)?` turns it into a `toktrie::TokEnv` (`Arc<dyn TokenizerEnv>`).
//!
//! *Why `from_file` instead of `from_tokenizer`?* `lfm` uses `tokenizers` v0.23 while
//! `toktrie_hf_tokenizers` 1.7 depends on v0.21. The two `Tokenizer` types are
//! incompatible at the type level (different crate versions → different nominal types).
//! Re-reading from the same path via `from_file` avoids the version boundary.
//!
//! **Factory construction:**
//! `ParserFactory::new_simple(&tok_env)?` — uses `InferenceCapabilities::default()`
//! (no ff_tokens) and `SlicedBiasComputer::general_slices()` (standard regex slices).
//!
//! **Constraint construction per call:**
//! `TopLevelGrammar::from_json_schema(serde_json::from_str(schema_json)?)` builds the
//! grammar from the task's JSON schema `Value`.  `factory.create_parser(grammar)?`
//! returns a `TokenParser`; `Constraint::new(parser)` wraps it for the sampling loop.
//!
//! **Note on ff_tokens:** `InferenceCapabilities::default()` disables fast-forward tokens;
//! `ConstrainedSampler` handles the `sample_mask = None` case defensively (see sampler.rs).

use std::{
  path::{Path, PathBuf},
  sync::Arc,
};

use tokenizers::Tokenizer;

use crate::{
  ChatMessage, ContentPart, ImageInput,
  chat_template::{
    BOS, BOS_TOKEN_ID, EOS_TOKEN_ID, IM_END, IM_START, IM_START_TOKEN_ID, IMAGE_END,
    IMAGE_END_TOKEN_ID, IMAGE_START, IMAGE_START_TOKEN_ID, IMAGE_THUMBNAIL,
    IMAGE_THUMBNAIL_TOKEN_ID, IMAGE_TOKEN, IMAGE_TOKEN_ID, IMG_ROW_COL_BASE_ID,
  },
  error::{Error, Result},
  generate::{GenerateInputs, generate},
  options::{BackendKind, ImageBudget, Options, RequestOptions},
  preproc::{ImagePlan, Preprocessor},
  runtime::{
    backend::{Backend, BackendImpl},
    checkpoint::{self, CheckpointLayout},
    sampler::{ConstrainedSampler, FreeSampler},
  },
};

// The ORT-backed component wrappers — `ort` is mandatory everywhere except
// aarch64-macos, where it is optional behind the `ort` feature; see the
// `ort_backend` cfg (build.rs) and the target tables in Cargo.toml.
#[cfg(ort_backend)]
use crate::runtime::{
  backend::OrtBackend, decoder::Decoder, embed_tokens::EmbedTokens, vision::VisionEncoder,
};

use llguidance::{Constraint, ParserFactory, api::TopLevelGrammar};
use toktrie::TokEnv;

/// Public engine for LFM2.5-VL inference.
///
/// Construct via [`Engine::from_dir`] for the standard HuggingFace download
/// layout, or via [`Engine::from_paths`] for unusual file arrangements.
pub struct Engine {
  preproc: Preprocessor,
  /// Drives vision encode / text embed / decoder forward / KV cache.
  /// Phase 1 holds the ORT variant; the seam lets phase 2 add an
  /// on-device backend without touching the generation loop.
  backend: BackendImpl,
  tokenizer: Tokenizer,
  /// Bytes of `tokenizer.json` captured at construction. Storing
  /// a `tokenizer_path` and re-reading lazily inside
  /// `parser_factory()` would let a file replaced between Engine
  /// construction and the first schema-constrained `run` cause
  /// silent schema-vs-model mismatch — llguidance would mask token
  /// IDs from the new file while embedding/detokenization continued
  /// to use the originally-validated `Tokenizer`. Capturing the
  /// bytes once ties both loads to the same content.
  tokenizer_bytes: Vec<u8>,
  /// Cached ParserFactory; lazily initialized on first schema-constrained call.
  parser_factory: Option<Arc<ParserFactory>>,
  eos_token_id: u32,
  /// Per-call sampler seed; advances every `generate`/`run`. Initialized
  /// from system-time nanoseconds so two engines on the same machine
  /// don't return identical sequences.
  next_seed: u64,
}

impl Engine {
  /// Construct from a directory containing the ONNX model files.
  ///
  /// Expected layout (matches HuggingFace download):
  /// ```text
  /// {model_dir}/
  ///   onnx/
  ///     vision_encoder.onnx
  ///     embed_tokens.onnx
  ///     decoder_model_merged.onnx
  ///   tokenizer.json
  ///   preprocessor_config.json
  /// ```
  ///
  /// **Strict constructor.** Whichever backend the directory selects, the
  /// checkpoint is validated against what this build of `lfm` hardcodes before
  /// the constructor returns:
  ///
  /// - the supplied `tokenizer.json` must byte-match the bundled blob — a
  ///   custom tokenizer whose normal vocabulary drifts from what the embedding
  ///   table expects would silently corrupt every prompt;
  /// - `chat_template.jinja` must byte-match the bundled template the renderer
  ///   actually uses;
  /// - the model's real context limit must match
  ///   [`MODEL_CONTEXT_TOKENS`](crate::options::MODEL_CONTEXT_TOKENS), which the
  ///   admission gates trust;
  /// - the preprocessing must match. Both roads read the pixel arithmetic their
  ///   patchifier bakes in — normalization, rescale factor, resampling — from
  ///   `preprocessor_config.json`. ONNX additionally validates the tiling
  ///   geometry from that file; MLX validates the equivalent from the
  ///   checkpoint's own `config.json`, against those constants and against the
  ///   [`ImageBudget`] the prompt's markers are rendered from.
  ///
  /// Requires the `bundled` feature so the byte-compare references are
  /// available. The named escape hatches are [`Engine::from_paths`] (ONNX) and
  /// [`Engine::from_mlx_dir_unchecked`] and friends (MLX).
  ///
  /// # Backend selection
  ///
  /// The layout decides, and the decision is observable through
  /// [`Engine::backend`]. A **complete** `onnx/` graph set selects ONNX even
  /// when MLX-format assets sit beside it, because `config.json` +
  /// `model.safetensors` are also the standard HuggingFace source-asset names —
  /// an export shipping those next to its graphs is an ONNX checkpoint.
  /// [`Options::with_backend`] overrides that, and every layout that has no
  /// documented answer (a half-present graph set beside MLX weights, weights in
  /// a format this build disabled, an MLX checkpoint on a non-Apple-Silicon
  /// host) is a named error rather than a fall-through to an unrelated
  /// missing-graph failure.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_dir<P: AsRef<Path>>(model_dir: P, opts: Options) -> Result<Self> {
    let dir: PathBuf = model_dir.as_ref().to_path_buf();
    match select_backend(&dir, opts.backend())? {
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      BackendKind::Mlx => Self::from_mlx_dir(&dir, opts),
      // `select_backend` has already refused an MLX selection on a platform
      // that does not compile the backend, so this arm is unreachable there;
      // it keeps the match exhaustive across the platform-gated variant set.
      #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
      BackendKind::Mlx => Err(Error::BackendUnavailable {
        requested: BackendKind::Mlx,
        reason: "the MLX backend is compiled only on macOS/arm64 (aarch64-apple-darwin)",
      }),
      #[cfg(ort_backend)]
      _ => Self::from_onnx_checkpoint_dir(&dir, opts),
      // `select_backend` has already refused an ONNX selection on aarch64-macos
      // without the `ort` feature (via `require_backend_compiled`), so this arm
      // is unreachable there; it keeps the match exhaustive across the
      // platform-gated variant set, mirroring the Mlx arm above.
      #[cfg(not(ort_backend))]
      _ => Err(Error::BackendUnavailable {
        requested: BackendKind::Onnx,
        reason: "the ONNX (`ort`) backend is not compiled in on aarch64-apple-darwin without the `ort` feature — MLX is the native road here",
      }),
    }
  }

  /// The ONNX half of [`Engine::from_dir`]: the strict drift validations
  /// against the bundled assets, then the three graphs.
  #[cfg(all(feature = "bundled", ort_backend))]
  fn from_onnx_checkpoint_dir(dir: &Path, opts: Options) -> Result<Self> {
    // validate preprocessor_config.json
    // matches our hardcoded algorithm constants. A model directory
    // with compatible ONNX shapes but a drifted preprocessing config
    // (different tile_size, patch_size, normalization) would
    // otherwise produce wrong visual embeddings without a clear
    // load-time error. Only runs in from_dir / from_onnx_dir where
    // we have access to the model's config files; from_paths users
    // explicitly opted out of this check.
    validate_preprocessor_config(&dir.join("preprocessor_config.json"))?;
    // The tokenizer special-token contract verifies
    // BOS/IM_START/image/row-col IDs, but a tokenizer with
    // those IDs unchanged AND a drifted NORMAL vocabulary (different
    // BPE merges, swapped subword IDs, etc.) would still pass. Such
    // a tokenizer would encode the same text into different token
    // IDs that no longer match the model's embedding table, silently
    // corrupting every prompt. For from_dir (strict constructor),
    // require the supplied tokenizer.json to byte-match the bundled
    // blob. from_paths remains the unchecked escape hatch for
    // advanced callers pairing custom tokenizers with custom ONNX.
    // (No inner cfg gate needed — from_dir itself is gated on `bundled`.)
    validate_tokenizer_matches_bundled(&dir.join("tokenizer.json"))?;
    // validate the model directory's chat
    // template byte-equals our bundled jinja. The renderer always
    // uses BUNDLED_CHAT_TEMPLATE_JINJA at run-time; a model revision
    // can ship a byte-identical tokenizer.json yet a different chat
    // template (different role envelope, different image-block
    // wrapping) — and from_dir would silently load it while we
    // render with the wrong template, producing semantically wrong
    // prompts whose `<image>` count still happens to line up. The
    // file is required for from_dir; absence is treated the same
    // way as a content mismatch (use from_paths to bypass).
    validate_chat_template_matches_bundled(&dir.join("chat_template.jinja"))?;
    // validate the model's
    // text_config.max_position_embeddings matches the hard-coded
    // MODEL_CONTEXT_TOKENS used by generate's admission gates. A
    // model directory with the same tokenizer/template/preprocessor
    // but a smaller-context decoder export would otherwise load
    // successfully, and requests up to 128 K tokens would pass
    // admission then fail late or generate with invalid position
    // state. The same gate refuses a config.json that turns the image
    // brackets off, which this crate's renderer and ImagePlan both hardcode.
    // Same theme as the chat_template drift check.
    validate_config_contract_matches_bundled(&dir.join("config.json"))?;
    let onnx = dir.join("onnx");
    Self::from_paths(
      EnginePaths::new(
        onnx.join("vision_encoder.onnx"),
        onnx.join("embed_tokens.onnx"),
        onnx.join("decoder_model_merged.onnx"),
        dir.join("tokenizer.json"),
      ),
      opts,
    )
  }

  /// Construct from a directory that contains **only the ONNX files**, using
  /// the tokenizer + configs that were bundled into this crate at compile time.
  ///
  /// Use this when you've downloaded only the ONNX artifacts from the upstream
  /// HuggingFace repo and don't want to also fetch the tokenizer / JSON configs.
  /// The bundled `tokenizer.json` is written to a per-process temp file (required
  /// by `toktrie_hf_tokenizers::ByteTokenizer::from_file`) and reused across all
  /// schema-constrained calls within the same `Engine` instance.
  ///
  /// Expected ONNX directory layout:
  /// ```text
  /// {onnx_dir}/
  ///   vision_encoder.onnx
  ///   embed_tokens.onnx
  ///   decoder_model_merged.onnx
  /// ```
  ///
  /// The tokenizer written to the temp directory (`$TMPDIR/lfm-bundled-<PID>/`)
  /// is never explicitly deleted; the OS cleans it up on next boot (standard
  /// behaviour for `std::env::temp_dir()`).
  #[cfg(all(feature = "bundled", ort_backend))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(
      feature = "bundled",
      any(
        not(all(target_arch = "aarch64", target_os = "macos")),
        feature = "ort"
      )
    )))
  )]
  pub fn from_onnx_dir<P: AsRef<Path>>(onnx_dir: P, opts: Options) -> Result<Self> {
    let onnx = onnx_dir.as_ref();
    let tmp_tokenizer = write_bundled_tokenizer()?;
    Self::from_paths(
      EnginePaths::new(
        onnx.join("vision_encoder.onnx"),
        onnx.join("embed_tokens.onnx"),
        onnx.join("decoder_model_merged.onnx"),
        tmp_tokenizer,
      ),
      opts,
    )
  }

  /// Construct from explicit paths (for non-standard layouts).
  ///
  /// Always builds the ONNX/`ort` backend. The MLX (`mlxrs`) backend is reached
  /// only through the platform auto-routing in [`from_dir`](Self::from_dir) /
  /// [`from_mlx_dir`](Self::from_mlx_dir) — there is no user-facing backend knob.
  ///
  /// Requires `ort` to be compiled in: unconditional on every target except
  /// aarch64-macos, where it needs the `ort` feature (MLX is the native road
  /// there — see `Cargo.toml`).
  #[cfg(ort_backend)]
  #[cfg_attr(
    docsrs,
    doc(cfg(any(
      not(all(target_arch = "aarch64", target_os = "macos")),
      feature = "ort"
    )))
  )]
  pub fn from_paths(paths: EnginePaths, opts: Options) -> Result<Self> {
    // Validate budget BEFORE any expensive work. validate_image_tokenizer_contract
    // performs an O(max_tiles²) nested scan; without this guard, an invalid
    // budget like `with_max_tiles(usize::MAX)` would hang construction
    // indefinitely. Validate() also caps max_tiles at MAX_TOKENIZER_TILE_DIM=10
    // so the scan is provably bounded after this returns Ok.
    opts.image_budget().validate()?;
    let vision = VisionEncoder::from_path(paths.vision(), &opts)?;
    let embed = EmbedTokens::from_path(paths.embed(), &opts)?;
    let decoder = Decoder::from_path(paths.decoder(), &opts)?;
    Self::assemble(
      BackendImpl::Ort(OrtBackend::new(vision, embed, decoder)),
      paths.tokenizer(),
      *opts.image_budget(),
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an MLX checkpoint
  /// directory (`config.json` + a weight set + `tokenizer.json`).
  ///
  /// Apple-Silicon only; on every other platform this constructor does not
  /// exist (the `mlxrs` dependency is macOS/arm64-only). It is normally reached
  /// via [`from_dir`](Self::from_dir)'s auto-routing rather than called
  /// directly. The engine still owns tokenization, the chat template, EOS
  /// handling, and the sampler — the MLX backend replaces only the model-weight
  /// stages (text embed, vision encode + splice, decoder forward).
  ///
  /// **Strict constructor**, matching [`from_dir`](Self::from_dir): the
  /// directory's `tokenizer.json` must byte-match the bundled blob, its
  /// `chat_template.jinja` must byte-match the bundled template, and its
  /// `preprocessor_config.json` must declare the normalization, rescale factor
  /// and resampling the MLX processor bakes in — on top of the structural
  /// contract every MLX road enforces (see
  /// [`from_mlx_dir_unchecked`](Self::from_mlx_dir_unchecked)). Use the
  /// `_unchecked` door for a custom checkpoint.
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled")))
  )]
  pub fn from_mlx_dir<P: AsRef<Path>>(model_dir: P, opts: Options) -> Result<Self> {
    let dir = model_dir.as_ref();
    validate_mlx_checkpoint_identity(dir)?;
    Self::from_mlx_dir_unchecked(dir, opts)
  }

  /// Construct an MLX engine from a checkpoint directory **without** the
  /// bundled-identity validations.
  ///
  /// This is the named door for a custom MLX checkpoint: a fine-tune with its
  /// own tokenizer, a re-export whose `chat_template.jinja` differs from the one
  /// this crate renders with, or one whose `preprocessor_config.json` declares
  /// different normalization / rescale / resampling than the MLX processor
  /// bakes in. Skipping those checks means YOU are asserting that the
  /// checkpoint's vocabulary, prompt format and pixel arithmetic match what
  /// `lfm` and `mlxrs` apply; if they do not, prompts and vision inputs are
  /// corrupted silently.
  ///
  /// What is **not** skippable, because `lfm`'s own arithmetic depends on it and
  /// no assertion by the caller can make it safe:
  ///
  /// - the model's context limit must equal
  ///   [`MODEL_CONTEXT_TOKENS`](crate::options::MODEL_CONTEXT_TOKENS), which the
  ///   admission gates use unconditionally;
  /// - the checkpoint's patch size, downsample factor, tile size and `<image>`
  ///   token id must equal the constants baked into this crate's token math;
  /// - the checkpoint's tiling must be one the prompt's markers can express,
  ///   and must agree with a non-default [`ImageBudget`] (see
  ///   [`Error::MlxTilingMismatch`](crate::Error::MlxTilingMismatch)).
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub fn from_mlx_dir_unchecked<P: AsRef<Path>>(model_dir: P, opts: Options) -> Result<Self> {
    let dir = model_dir.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_dir(dir)?;
    Self::assemble_mlx(backend, &dir.join("tokenizer.json"), &opts)
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact**
  /// `model.safetensors` file path (Apple Silicon only).
  ///
  /// The explicit-format counterpart to [`from_mlx_dir`](Self::from_mlx_dir): use
  /// it when you already know the checkpoint is an MLX safetensors file and where
  /// it lives. The `config.json`, `chat_template.jinja` and `tokenizer.json` are
  /// read from the weight file's **parent directory** (`weights.parent()`).
  /// There is no ONNX fallback — this constructor always builds the MLX backend,
  /// and it is strict; see
  /// [`from_mlx_safetensors_unchecked`](Self::from_mlx_safetensors_unchecked).
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled")))
  )]
  pub fn from_mlx_safetensors<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    validate_mlx_checkpoint_identity(crate::runtime::mlx_backend::weights_parent(weights))?;
    Self::from_mlx_safetensors_unchecked(weights, opts)
  }

  /// [`from_mlx_safetensors`](Self::from_mlx_safetensors) without the
  /// bundled-identity validations — see
  /// [`from_mlx_dir_unchecked`](Self::from_mlx_dir_unchecked) for exactly what
  /// that does and does not skip.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub fn from_mlx_safetensors_unchecked<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_safetensors(weights)?;
    Self::assemble_mlx(
      backend,
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact** `*.npz`
  /// file path (Apple Silicon only). The sibling assets are read from the weight
  /// file's **parent directory** (`weights.parent()`).
  ///
  /// Explicit-format MLX constructor (see
  /// [`from_mlx_safetensors`](Self::from_mlx_safetensors)); always builds the MLX
  /// backend, no ONNX fallback, strict.
  #[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "npz",
    feature = "bundled"
  ))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(
      target_os = "macos",
      target_arch = "aarch64",
      feature = "npz",
      feature = "bundled"
    )))
  )]
  pub fn from_mlx_npz<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    validate_mlx_checkpoint_identity(crate::runtime::mlx_backend::weights_parent(weights))?;
    Self::from_mlx_npz_unchecked(weights, opts)
  }

  /// [`from_mlx_npz`](Self::from_mlx_npz) without the bundled-identity
  /// validations — see [`from_mlx_dir_unchecked`](Self::from_mlx_dir_unchecked).
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "npz"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "npz")))
  )]
  pub fn from_mlx_npz_unchecked<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_npz(weights)?;
    Self::assemble_mlx(
      backend,
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact** `*.gguf`
  /// file path (Apple Silicon only). The sibling assets are read from the weight
  /// file's **parent directory** (`weights.parent()`); the gguf's embedded
  /// metadata is NOT mapped to a config, so a sibling `config.json` is still
  /// required.
  ///
  /// Explicit-format MLX constructor (see
  /// [`from_mlx_safetensors`](Self::from_mlx_safetensors)); always builds the MLX
  /// backend, no ONNX fallback, strict.
  #[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "gguf",
    feature = "bundled"
  ))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(
      target_os = "macos",
      target_arch = "aarch64",
      feature = "gguf",
      feature = "bundled"
    )))
  )]
  pub fn from_mlx_gguf<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    validate_mlx_checkpoint_identity(crate::runtime::mlx_backend::weights_parent(weights))?;
    Self::from_mlx_gguf_unchecked(weights, opts)
  }

  /// [`from_mlx_gguf`](Self::from_mlx_gguf) without the bundled-identity
  /// validations — see [`from_mlx_dir_unchecked`](Self::from_mlx_dir_unchecked).
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "gguf"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "gguf")))
  )]
  pub fn from_mlx_gguf_unchecked<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_gguf(weights)?;
    Self::assemble_mlx(
      backend,
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// The MLX road's structural contract, then assembly.
  ///
  /// Runs on EVERY MLX constructor, strict or `_unchecked`: the context limit
  /// the admission gates trust, the preprocessing constants this crate's token
  /// math bakes in, and the reconciliation between the caller's
  /// [`ImageBudget`] and the checkpoint's own tiling. The budget that survives
  /// that reconciliation — the checkpoint's own, when the caller passed the
  /// default — is what the engine's [`Preprocessor`] renders every prompt from,
  /// so the markers and the features are planned by the same parameters.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  fn assemble_mlx(
    backend: crate::runtime::mlx_backend::MlxBackend,
    tokenizer_path: &Path,
    opts: &Options,
  ) -> Result<Self> {
    backend.validate_context_limit()?;
    let budget = backend.effective_budget(opts.image_budget())?;
    Self::assemble(BackendImpl::Mlx(Box::new(backend)), tokenizer_path, budget)
  }

  /// Shared engine assembly: load the tokenizer from `tokenizer_path`, validate
  /// its EOS + image special-token contract, and build the [`Engine`] around the
  /// already-constructed `backend`. Used by both the ONNX [`from_paths`] and the
  /// MLX [`from_mlx_dir`] paths so they share one tokenizer / EOS / parser-factory
  /// setup.
  ///
  /// `budget` is the **effective** image budget — the caller's on the ONNX road,
  /// and the one reconciled against the checkpoint on the MLX road. Everything
  /// downstream (marker rendering, admission floors, the backend's own planning)
  /// reads it from the [`Preprocessor`] built here, so there is exactly one
  /// budget in play per engine.
  ///
  /// The caller has already validated the image budget.
  fn assemble(backend: BackendImpl, tokenizer_path: &Path, budget: ImageBudget) -> Result<Self> {
    let preproc = Preprocessor::new(budget);
    // read bytes ONCE, then build the
    // `tokenizers::Tokenizer` from those exact bytes. The same
    // bytes are stored on the Engine and reused by the lazy
    // ParserFactory — guaranteeing the schema matcher and the
    // tokenizer/embedding stack agree, regardless of any later
    // file changes at `tokenizer_path`.
    let tokenizer_bytes = std::fs::read(tokenizer_path).map_err(Error::Io)?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(Error::tokenizer)?;

    // EOS token: LFM2.5-VL chat models use <|im_end|> (id 7).
    // IM_END / EOS_TOKEN_ID are consts in chat_template.rs; cross-check
    // detects tokenizer.json drift (model rev mismatch, custom tokenizer).
    let eos_token_id = tokenizer
      .token_to_id(IM_END)
      .ok_or(Error::InvalidRequest("tokenizer missing <|im_end|> token"))?;
    if eos_token_id != EOS_TOKEN_ID {
      return Err(Error::InvalidRequest(
        "tokenizer <|im_end|> token id differs from expected EOS_TOKEN_ID (7) — wrong tokenizer.json?",
      ));
    }

    // Validate every special token that expand_image_placeholders can
    // emit. Without this, a tokenizer that's missing
    // <|image_start|>/<|image_end|>/<|img_thumbnail|> or any
    // <|img_row_R_col_C|> marker reachable under max_tiles loads
    // successfully — then tokenization at run-time silently splits
    // those markers into byte-level tokens while the <image>-token
    // count still matches, corrupting position-token embeddings on
    // every multi-tile prompt with no error reported.
    validate_image_tokenizer_contract(&tokenizer, budget.max_tiles())?;

    let next_seed = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0xC0_FFEE);

    Ok(Self {
      preproc,
      backend,
      tokenizer,
      tokenizer_bytes,
      parser_factory: None,
      eos_token_id,
      next_seed,
    })
  }

  /// Which backend this engine actually runs on.
  ///
  /// Backend selection is decided by the checkpoint layout (see
  /// [`from_dir`](Self::from_dir)) or pinned by
  /// [`Options::with_backend`](crate::Options::with_backend); either way the
  /// outcome is reportable rather than opaque, which matters because the two
  /// backends are different numerical paths over the same model.
  pub fn backend(&self) -> BackendKind {
    self.backend.kind()
  }

  /// The **effective** image budget this engine renders prompts from.
  ///
  /// On the ONNX road that is the budget from [`Options`]. On the MLX road a
  /// default [`ImageBudget`] is replaced by the checkpoint's own tiling (see
  /// [`from_mlx_dir_unchecked`](Self::from_mlx_dir_unchecked)), so reading it
  /// back is the way to see what the model will actually do.
  pub fn image_budget(&self) -> &ImageBudget {
    self.preproc.budget()
  }

  /// The authoritative [`ImagePlan`] for each image, from header dimensions
  /// only — no full decode.
  ///
  /// This is the same plan `generate` / `run` will use: the marker layout, the
  /// `<image>`-token count, and the number of sub-images the backend's
  /// preprocessing must produce. Useful to size or reject a request before
  /// paying for inference, and to compare what two backends would do with the
  /// same image.
  pub fn plan_images(&self, images: &[ImageInput<'_>]) -> Result<Vec<ImagePlan>> {
    crate::generate::plan_images(&self.preproc, &self.backend, images)
  }

  /// Run everything up to and including the decoder prefill, and return the
  /// last-position logits — the model's next-token distribution for this
  /// prompt.
  ///
  /// The full admission control, image planning, prompt rendering and vision
  /// splice run exactly as they do in [`generate`](Self::generate); only the
  /// decode loop is skipped. The row is guaranteed finite (a non-finite one is
  /// [`Error::SessionNonFiniteOutput`](crate::Error::SessionNonFiniteOutput)).
  ///
  /// `req`'s sampler fields are unused — no token is drawn — but
  /// `max_new_tokens` still participates in the context-budget admission
  /// checks, so the returned distribution is one a matching
  /// [`generate`](Self::generate) call would really have sampled from.
  pub fn next_token_logits(
    &mut self,
    messages: &[ChatMessage],
    images: &[ImageInput<'_>],
    req: &RequestOptions,
  ) -> Result<Vec<f32>> {
    req.validate()?;
    let inputs = GenerateInputs::new(messages, images, req, self.eos_token_id);
    let prefilled =
      crate::generate::prefill(&self.preproc, &mut self.backend, &self.tokenizer, &inputs)?;
    Ok(prefilled.logits)
  }

  /// Free-form generation (no schema constraint).
  ///
  /// Uses an unconstrained sampler (greedy or min-p with repetition penalty).
  pub fn generate(
    &mut self,
    messages: &[ChatMessage],
    images: &[ImageInput<'_>],
    req: &RequestOptions,
  ) -> Result<String> {
    req.validate()?;
    let seed = self.draw_seed();
    let mut sampler = FreeSampler::new(*req, seed, self.tokenizer.get_vocab_size(true) as u32);
    generate(
      &self.preproc,
      &mut self.backend,
      &self.tokenizer,
      &mut sampler,
      GenerateInputs::new(messages, images, req, self.eos_token_id),
    )
  }

  /// Schema-constrained generation driven by a [`llmtask::Task`].
  ///
  /// 1. Builds the user message from the supplied images plus
  ///    `task.prompt()` — the caller does not pass `messages`. This
  ///    guarantees `task.prompt()` is always present, so the schema-
  ///    valid output reflects the task's grounding rules and not just
  ///    its JSON shape.
  /// 2. Compiles `task.schema()` into an llguidance `Constraint`.
  /// 3. Runs the generation loop with a constraint-driven sampler.
  /// 4. Passes the raw text to `task.parse(raw)` for typed deserialization.
  ///
  /// The `ParserFactory` is constructed once and cached across calls.
  pub fn run<T: llmtask::Task>(
    &mut self,
    task: &T,
    images: &[ImageInput<'_>],
    req: &RequestOptions,
  ) -> Result<T::Output>
  where
    Error: From<T::ParseError>,
  {
    req.validate()?;
    // preflight image-count bounds BEFORE
    // allocating one ContentPart per image. Without this, a request
    // with millions of ImageInput entries would force a giant
    // Vec<ContentPart> allocation before generate() could reject
    // via its own admission checks. Mirror those checks here on
    // stack-only state.
    if images.len().saturating_add(1) > crate::generate::MAX_TOTAL_CONTENT_PARTS {
      return Err(Error::InvalidRequest(
        "too many images per request (request-shape DoS guard)",
      ));
    }
    crate::generate::check_image_count_lower_bound(
      images.len(),
      self
        .preproc
        .budget()
        .min_image_tokens()
        .saturating_add(crate::preproc::IMAGE_BLOCK_WRAPPER_TOKENS),
      req.max_new_tokens(),
    )?;

    // Build a single user message: N image parts followed by the task
    // prompt text. This locks in the contract that task.prompt() is
    // always sent with the images — callers can't accidentally drop it.
    let mut parts: Vec<ContentPart> = Vec::with_capacity(images.len() + 1);
    for _ in 0..images.len() {
      parts.push(ContentPart::Image);
    }
    parts.push(ContentPart::Text(task.prompt().to_owned()));
    let messages = [ChatMessage::parts(
      smol_str::SmolStr::new_static("user"),
      parts,
    )];

    let factory = self.parser_factory()?;
    let constraint = build_constraint(&factory, &task.grammar())?;
    let seed = self.draw_seed();
    let mut sampler = ConstrainedSampler::new(
      constraint,
      *req,
      seed,
      self.tokenizer.get_vocab_size(true) as u32,
    );
    let text = generate(
      &self.preproc,
      &mut self.backend,
      &self.tokenizer,
      &mut sampler,
      GenerateInputs::new(&messages, images, req, self.eos_token_id),
    )?;
    task.parse(&text).map_err(Error::from)
  }

  // ===== internal =====

  /// Return the current seed and advance the counter. Each call to
  /// [`Engine::generate`] / [`Engine::run`] gets a distinct seed so
  /// non-greedy sampling doesn't replay an identical sequence.
  fn draw_seed(&mut self) -> u64 {
    let seed = self.next_seed;
    self.next_seed = self.next_seed.wrapping_add(1);
    seed
  }

  /// Lazily construct and cache the `ParserFactory`.
  ///
  /// The factory is wrapped in `Arc` so it can be shared across
  /// multiple `Constraint` instances across calls without cloning
  /// the heavy trie data.
  fn parser_factory(&mut self) -> Result<Arc<ParserFactory>> {
    if let Some(f) = &self.parser_factory {
      return Ok(f.clone());
    }
    let factory = build_parser_factory(&self.tokenizer_bytes)?;
    let arc = Arc::new(factory);
    self.parser_factory = Some(arc.clone());
    Ok(arc)
  }
}

// =========================================================================
// Backend selection
// =========================================================================

/// Decide which backend a model directory describes, honouring an explicit
/// [`Options::with_backend`](crate::Options::with_backend) pin.
///
/// Every outcome is either a backend or a named error. The three that used to
/// fall through to the ONNX path and die on a missing graph — a weight set in a
/// disabled format, a half-present graph set beside MLX weights, and an MLX
/// checkpoint on a host with no MLX backend — are now reported for what they
/// are.
fn select_backend(dir: &Path, pinned: Option<BackendKind>) -> Result<BackendKind> {
  // (default choice, MLX servable from this dir, ONNX servable from this dir)
  let (default_choice, mlx_available, onnx_available) = match checkpoint::detect(dir) {
    // A complete graph set is the documented disambiguator, so ONNX wins by
    // default — but when MLX assets sit alongside, a pin can select them.
    CheckpointLayout::Onnx { mlx_alongside } => (BackendKind::Onnx, mlx_alongside, true),
    CheckpointLayout::Mlx => (BackendKind::Mlx, true, false),
    CheckpointLayout::MlxFormatDisabled(format) => {
      return Err(Error::CheckpointFormatDisabled {
        dir: dir.to_path_buf(),
        format,
      });
    }
    CheckpointLayout::Incomplete(detail) => {
      return Err(Error::CheckpointIncomplete {
        dir: dir.to_path_buf(),
        detail,
      });
    }
    // An explicit pin is exactly the disambiguation this layout needs; without
    // one there is no defensible default, so say so instead of guessing.
    CheckpointLayout::Ambiguous(detail) => {
      let Some(kind) = pinned else {
        return Err(Error::CheckpointLayoutAmbiguous {
          dir: dir.to_path_buf(),
          detail,
        });
      };
      return require_backend_compiled(kind);
    }
  };

  match pinned {
    Some(BackendKind::Mlx) if !mlx_available => Err(Error::BackendUnavailable {
      requested: BackendKind::Mlx,
      reason: "the directory holds no MLX checkpoint (config.json plus a weight set in a format this build enables)",
    }),
    Some(BackendKind::Onnx) if !onnx_available => Err(Error::BackendUnavailable {
      requested: BackendKind::Onnx,
      reason: "the directory holds no complete ONNX graph set under onnx/",
    }),
    Some(kind) => require_backend_compiled(kind),
    None => require_backend_compiled(default_choice),
  }
}

/// Refuse a backend this build does not compile.
///
/// `mlxrs` is a macOS/arm64-only target dependency, so an MLX checkpoint opened
/// on any other host has no backend to run on. Naming that beats reporting a
/// missing `onnx/vision_encoder.onnx`, which is what a directory of MLX files
/// used to produce. Symmetrically, `ort` is optional on aarch64-macos (behind
/// the `ort` feature — MLX is the native road there), so an ONNX checkpoint
/// opened there without that feature is named too, rather than failing deep
/// inside a constructor that does not exist.
fn require_backend_compiled(kind: BackendKind) -> Result<BackendKind> {
  #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
  if matches!(kind, BackendKind::Mlx) {
    return Err(Error::BackendUnavailable {
      requested: BackendKind::Mlx,
      reason: "the MLX backend is compiled only on macOS/arm64 (aarch64-apple-darwin)",
    });
  }
  #[cfg(not(ort_backend))]
  if matches!(kind, BackendKind::Onnx) {
    return Err(Error::BackendUnavailable {
      requested: BackendKind::Onnx,
      reason: "the ONNX (`ort`) backend is not compiled in on aarch64-apple-darwin without the `ort` feature — MLX is the native road here",
    });
  }
  Ok(kind)
}

/// The waivable half of the MLX road's strict contract: the checkpoint's
/// `tokenizer.json` and `chat_template.jinja` must byte-match the blobs this
/// crate renders and tokenizes with, and its `preprocessor_config.json` must
/// declare the pixel arithmetic the MLX processor bakes in.
///
/// Runs on EVERY strict MLX constructor — [`Engine::from_mlx_dir`],
/// [`Engine::from_mlx_safetensors`], `Engine::from_mlx_npz` and
/// `Engine::from_mlx_gguf` all route through here — so the preprocessing
/// contract cannot be enforced on one door and skipped on another.
///
/// These are the assertions a caller can legitimately waive for a custom
/// checkpoint (hence the `_unchecked` constructors, which remain the only
/// escape); the structural contract in [`Engine::assemble_mlx`] cannot be
/// waived.
///
/// The preprocessing check belongs here rather than in `assemble_mlx` because
/// it is waivable in exactly the sense the other two are: a fine-tune may
/// legitimately have been trained under different normalization, and asserting
/// that is what `_unchecked` means. What it must not do is pass silently —
/// `mlxrs` hardcodes `image_mean = image_std = 0.5`, rescale `1/255` and
/// bilinear resampling (`Lfm2Vl::processor_config` never reads those from the
/// checkpoint), so a revision that changes any of them would feed
/// systematically wrong pixels to the vision tower with every count, grid and
/// dimension check still green. See [`check_preprocessing_pixel_contract`].
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled"))]
fn validate_mlx_checkpoint_identity(dir: &Path) -> Result<()> {
  validate_tokenizer_matches_bundled(&dir.join("tokenizer.json"))?;
  validate_chat_template_matches_bundled(&dir.join("chat_template.jinja"))?;
  validate_mlx_preprocessing_contract(&dir.join("preprocessor_config.json"))?;
  Ok(())
}

// =========================================================================
// Tokenizer contract validation
// =========================================================================

/// Validate that the tokenizer recognises every special token that
/// [`crate::chat_template::expand_image_placeholders`] can emit:
///
/// - `<image>` (with the bundled-tokenizer id as a cross-check)
/// - `<|image_start|>`, `<|image_end|>`, `<|img_thumbnail|>`
/// - `<|img_row_R_col_C|>` for every R, C in `[1, max_tiles]`
///
/// Run-time tokenization treats unknown special-token strings as raw
/// text and breaks them into byte-level pieces, while the
/// `<image>`-token count remains correct — silently corrupting
/// position-token embeddings on every multi-tile prompt. Catching
/// this at session-construction prevents the silent failure mode.
#[allow(dead_code)]
fn validate_image_tokenizer_contract(tokenizer: &Tokenizer, max_tiles: usize) -> Result<()> {
  // the embedding/decoder contract is
  // ID-based — a tokenizer with the same special-token STRINGS but
  // remapped IDs would pass a presence-only check, then embed
  // markers as the wrong tokens at run-time. Validate ID for every
  // structural special token the chat template / image expansion
  // can emit. The bundled tokenizer's IDs are listed in the const
  // table at the top of chat_template.rs.
  let id_check = |name_str: &str, expected: u32| -> Result<()> {
    let actual = tokenizer
      .token_to_id(name_str)
      .ok_or(Error::InvalidRequest(
        "tokenizer missing required special token — wrong tokenizer.json?",
      ))?;
    if actual != expected {
      return Err(Error::InvalidRequest(
        "tokenizer special-token id differs from expected — wrong tokenizer.json?",
      ));
    }
    Ok(())
  };
  id_check(BOS, BOS_TOKEN_ID)?;
  id_check(IM_START, IM_START_TOKEN_ID)?;
  id_check(IMAGE_TOKEN, IMAGE_TOKEN_ID)?;
  id_check(IMAGE_START, IMAGE_START_TOKEN_ID)?;
  id_check(IMAGE_END, IMAGE_END_TOKEN_ID)?;
  id_check(IMAGE_THUMBNAIL, IMAGE_THUMBNAIL_TOKEN_ID)?;

  // Per-tile row/col markers reachable under max_tiles. The candidate
  // grid search in find_closest_aspect_ratio enumerates (i, j) for
  // i, j in [1, max_tiles] (constrained by i*j <= max_tiles), so any
  // reachable marker has both indices in [1, max_tiles].
  // ImageBudget::validate caps max_tiles at MAX_TOKENIZER_TILE_DIM
  // (=10), so this is at most 100 lookups.
  //
  // Defense-in-depth: even though the caller (Engine::from_paths) is
  // expected to validate the budget first, refuse to scan past
  // MAX_TOKENIZER_TILE_DIM here to keep the loop provably bounded
  // even if a future caller forgets.
  if max_tiles > crate::options::MAX_TOKENIZER_TILE_DIM {
    return Err(Error::InvalidBudget(
      "max_tiles must be <= 10 (bundled tokenizer's row/col marker grid is 10x10)",
    ));
  }
  // Per-tile markers <|img_row_R_col_C|> for R, C in [1, max_tiles].
  // Bundled IDs are contiguous: IMG_ROW_COL_BASE_ID + (R-1)*10 + (C-1)
  // for R, C in [1, 10] (so ids 397..=496). We validate both presence
  // AND id so a tokenizer with same strings but remapped ids fails
  // construction.
  for r in 1..=max_tiles as u32 {
    for c in 1..=max_tiles as u32 {
      let marker = format!("<|img_row_{r}_col_{c}|>");
      let actual = tokenizer
        .token_to_id(&marker)
        .ok_or(Error::InvalidRequest(
          "tokenizer missing one or more <|img_row_R_col_C|> markers reachable under max_tiles — wrong tokenizer.json?",
        ))?;
      let expected = IMG_ROW_COL_BASE_ID + (r - 1) * 10 + (c - 1);
      if actual != expected {
        return Err(Error::InvalidRequest(
          "tokenizer <|img_row_R_col_C|> id differs from expected (IMG_ROW_COL_BASE_ID + (R-1)*10 + (C-1)) — wrong tokenizer.json?",
        ));
      }
    }
  }

  Ok(())
}

// =========================================================================
// Preprocessor-config drift detectors
// =========================================================================

/// Read + parse a checkpoint's `preprocessor_config.json`, failing closed when
/// it is absent.
///
/// The strict drift detectors are the whole point of these checks; letting a
/// missing file skip them defeats it. A stripped-down checkpoint directory
/// belongs on one of the named unchecked doors instead
/// ([`Engine::from_paths`] on the ONNX road,
/// `Engine::from_mlx_*_unchecked` on the MLX road).
#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn read_preprocessor_config(path: &Path) -> Result<serde_json::Value> {
  if !path.exists() {
    return Err(Error::InvalidRequest(
      "model directory missing preprocessor_config.json — use from_paths (ONNX) or from_mlx_*_unchecked (MLX) to bypass strict drift checks",
    ));
  }
  let raw = std::fs::read_to_string(path).map_err(Error::Io)?;
  serde_json::from_str(&raw)
    .map_err(|e| Error::tokenizer(format!("preprocessor_config.json parse failure: {e}")))
}

#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn cfg_u64(cfg: &serde_json::Value, key: &'static str) -> Result<u64> {
  cfg
    .get(key)
    .and_then(|v| v.as_u64())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing required integer field — wrong model revision?",
    ))
}

#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn cfg_bool(cfg: &serde_json::Value, key: &'static str) -> Result<bool> {
  cfg
    .get(key)
    .and_then(|v| v.as_bool())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing required boolean field — wrong model revision?",
    ))
}

#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn cfg_str<'a>(cfg: &'a serde_json::Value, key: &'static str) -> Result<&'a str> {
  cfg
    .get(key)
    .and_then(|v| v.as_str())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing required string field — wrong model revision?",
    ))
}

#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn cfg_f64(cfg: &serde_json::Value, key: &'static str) -> Result<f64> {
  cfg
    .get(key)
    .and_then(|v| v.as_f64())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing required number field — wrong model revision?",
    ))
}

#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn cfg_f32_array3(cfg: &serde_json::Value, key: &'static str) -> Result<[f32; 3]> {
  let arr = cfg
    .get(key)
    .and_then(|v| v.as_array())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing required [f32; 3] field — wrong model revision?",
    ))?;
  if arr.len() != 3 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json field is not a 3-element array — wrong model revision?",
    ));
  }
  let mut out = [0f32; 3];
  for (i, v) in arr.iter().enumerate() {
    out[i] = v.as_f64().ok_or(Error::InvalidRequest(
      "preprocessor_config.json array element is not a number — wrong model revision?",
    ))? as f32;
  }
  Ok(out)
}

/// The preprocessing arithmetic BOTH roads bake in rather than read.
///
/// Every value here is a compile-time constant of some pixel pipeline, so a
/// checkpoint revision that changes one is not something either road can
/// honour — it can only be refused:
///
/// - the ONNX road's [`flatten_to_patches`](crate::preproc) computes
///   `(b / 255) * 2 - 1`, i.e. rescale `1/255` then `(x - 0.5) / 0.5`, and
///   resizes with a PIL-compatible bilinear filter;
/// - the MLX road's `mlxrs` `tile_image` folds the same contract into
///   `x * (1/255)/std + (-mean/std)` with `image_mean = image_std = 0.5`
///   (`Lfm2VlProcessorConfig::new`'s defaults — `Lfm2Vl::processor_config`
///   never overrides them from the checkpoint) and resizes with
///   `ResizeFilter::Bilinear`.
///
/// So a revision shipping ImageNet normalization, a different rescale factor,
/// a non-bilinear `resample`, or `do_normalize: false` would load fine and feed
/// systematically wrong pixels to the vision tower, with every count, grid and
/// dimension check still passing. This is the version-skew class the strict
/// constructors exist to refuse, on either road.
///
/// Budget-tunable fields (min/max image_tokens, min/max tiles,
/// `max_pixels_tolerance`, `use_thumbnail`) are deliberately NOT checked here —
/// callers override them via `Options::image_budget()`, and the MLX road
/// reconciles them against the checkpoint's own `config.json` in
/// `MlxBackend::effective_budget`.
#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn check_preprocessing_pixel_contract(cfg: &serde_json::Value) -> Result<()> {
  for (key, expected) in [
    ("do_resize", true),
    ("do_rescale", true),
    ("do_normalize", true),
    ("do_pad", true),
  ] {
    if cfg_bool(cfg, key)? != expected {
      return Err(Error::InvalidRequest(
        "preprocessor_config.json boolean preprocessing flag differs from lfm crate hardcoded value — wrong model revision?",
      ));
    }
  }

  // data_format: the layout upstream feeds `convert_image_to_patches`, whose
  // final reshape collapses (patch, patch, C) — so both of this crate's
  // patchifiers emit HWC bytes per patch. `channels_last` would be a different
  // byte order for both.
  if cfg_str(cfg, "data_format")? != "channels_first" {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json data_format != channels_first — wrong model revision?",
    ));
  }

  // resample: 2 = PIL BILINEAR. The ONNX road resizes with
  // `fast_image_resize`'s Pillow-compatible bilinear convolution; the MLX road
  // with `mlxrs`'s `ResizeFilter::Bilinear`.
  if cfg_u64(cfg, "resample")? != 2 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json resample != 2 (BILINEAR) — wrong model revision?",
    ));
  }

  // rescale_factor: 1/255, the divisor both roads bake into their normalization.
  let rf = cfg_f64(cfg, "rescale_factor")?;
  if (rf - (1.0 / 255.0)).abs() > 1e-9 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json rescale_factor != 1/255 — wrong model revision?",
    ));
  }

  // Normalization: `(px/255)*2 - 1` is `(px/255 - 0.5) / 0.5`, so image_mean and
  // image_std must both be [0.5, 0.5, 0.5].
  for (key, expected) in [("image_mean", [0.5f32; 3]), ("image_std", [0.5f32; 3])] {
    let got = cfg_f32_array3(cfg, key)?;
    for (g, e) in got.iter().zip(expected.iter()) {
      if (g - e).abs() > 1e-4 {
        return Err(Error::InvalidRequest(
          "preprocessor_config.json image_mean/image_std differs from [0.5, 0.5, 0.5] (lfm crate hardcoded normalization) — wrong model revision?",
        ));
      }
    }
  }
  Ok(())
}

/// Validate the model's `preprocessor_config.json` for the **ONNX** road: the
/// shared pixel contract plus the tiling geometry this crate's ported
/// `pick_tile_grid` / `smart_resize` hardcode.
///
/// The geometry half is ONNX-only because the MLX road reads the same
/// quantities from the checkpoint's `config.json` instead — the loaded
/// `ModelConfig` is checked against the identical constants in
/// `MlxBackend::effective_budget`'s `require_baked_in_contract`, which is
/// stronger (it asserts against the values the model was actually built with),
/// and `do_image_splitting` is *honoured* there rather than baked, folded into
/// the tile band by `checkpoint_budget`.
///
/// Used only by `from_dir` (where the model directory has the config alongside
/// the ONNX files). `from_onnx_dir` uses bundled assets and our own constants
/// by construction, so no drift is possible.
// Its only non-test caller, `from_onnx_checkpoint_dir`, is additionally
// gated on `ort_backend` (the ONNX road needs `ort` compiled in); the
// function body itself needs neither `ort` nor `bundled`-gated symbols, so
// it stays present everywhere and is merely allowed dead when unused.
#[cfg_attr(not(all(feature = "bundled", ort_backend)), allow(dead_code))]
fn validate_preprocessor_config(path: &Path) -> Result<()> {
  let cfg = read_preprocessor_config(path)?;
  check_preprocessing_pixel_contract(&cfg)?;

  // Model-fixed dimensional constants.
  if cfg_u64(&cfg, "encoder_patch_size")? != crate::preproc::tile_grid::PATCH_SIZE as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json encoder_patch_size != 16 (lfm crate hardcoded) — wrong model revision?",
    ));
  }
  if cfg_u64(&cfg, "downsample_factor")? != crate::preproc::tile_grid::DOWNSAMPLE_FACTOR as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json downsample_factor != 2 (lfm crate hardcoded) — wrong model revision?",
    ));
  }
  if cfg_u64(&cfg, "tile_size")? != crate::preproc::tile_grid::FULL_TILE_SIZE as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json tile_size != 512 (lfm crate hardcoded) — wrong model revision?",
    ));
  }

  // do_image_splitting: the ONNX road derives splitting from the budget's tile
  // band, not from this flag, so a checkpoint that disables splitting would be
  // split anyway. (The MLX road honours it — see this function's doc.)
  if !cfg_bool(&cfg, "do_image_splitting")? {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json boolean preprocessing flag differs from lfm crate hardcoded value — wrong model revision?",
    ));
  }

  // size = {height: 512, width: 512}. Mirrors tile_size but checked separately
  // because some HF processors use `size` independently.
  let size = cfg
    .get("size")
    .and_then(|v| v.as_object())
    .ok_or(Error::InvalidRequest(
      "preprocessor_config.json missing size object — wrong model revision?",
    ))?;
  for (key, expected) in [("height", 512u64), ("width", 512u64)] {
    if size.get(key).and_then(|v| v.as_u64()) != Some(expected) {
      return Err(Error::InvalidRequest(
        "preprocessor_config.json size.{height,width} != 512 — wrong model revision?",
      ));
    }
  }
  Ok(())
}

/// Validate the **MLX** road's half of the preprocessing contract: the pixel
/// arithmetic `mlxrs` bakes in, read from the checkpoint's
/// `preprocessor_config.json`.
///
/// The MLX geometry (patch size, downsample factor, tile size, tile band,
/// thumbnail policy) is validated from the loaded `ModelConfig` instead, so
/// only the shared pixel contract is read from this file.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled"))]
fn validate_mlx_preprocessing_contract(path: &Path) -> Result<()> {
  let cfg = read_preprocessor_config(path)?;
  check_preprocessing_pixel_contract(&cfg)
}

// =========================================================================
// Tokenizer-bytes drift detector (bundled feature only)
// =========================================================================

/// Validate the two `config.json` values this crate bakes in rather than reads:
/// the model's real context limit and its image-bracketing policy.
///
/// - `text_config.max_position_embeddings` (or top-level
///   `max_position_embeddings`) must match our hard-coded
///   [`crate::options::MODEL_CONTEXT_TOKENS`]. finding 1: generate()'s
///   admission gates trust this constant; a model exported with a smaller
///   positional embedding range would pass byte-identical
///   tokenizer/template/preprocessor checks (the static assets) but quietly
///   accept prompts past its real limit and either fail late or produce invalid
///   position state.
/// - `use_image_special_tokens` must not be `false`. This crate's renderer
///   always brackets an image block with `<|image_start|>` / `<|image_end|>`
///   and [`ImagePlan`](crate::preproc::ImagePlan) always budgets those two
///   tokens; a checkpoint whose processor contract omits them would be handed a
///   prompt with two tokens it never saw in training, with every count still
///   matching. The key is optional and **absent means `true`**, matching
///   upstream's own default (`config.py:87`) and the MLX road's config parse —
///   only an explicit `false` is refused.
// Its only non-test caller, `from_onnx_checkpoint_dir`, is additionally
// gated on `ort_backend`; the body needs neither `ort` nor a `bundled`-gated
// symbol (it compares against the plain `MODEL_CONTEXT_TOKENS` constant), so
// — matching `validate_preprocessor_config` above — it stays present
// everywhere and is merely allowed dead when unused, rather than a hard
// presence gate that would also force every test call site to track
// `ort_backend`.
#[cfg_attr(not(all(feature = "bundled", ort_backend)), allow(dead_code))]
fn validate_config_contract_matches_bundled(path: &Path) -> Result<()> {
  if !path.exists() {
    return Err(Error::InvalidRequest(
      "model directory missing config.json — use from_paths to bypass strict context-length drift checks (advanced: requires matching ONNX embedding table)",
    ));
  }
  let supplied = std::fs::read(path).map_err(Error::Io)?;
  let v: serde_json::Value = serde_json::from_slice(&supplied)
    .map_err(|_| Error::InvalidRequest("config.json is not valid JSON"))?;
  // LFM2.5-VL's config nests text params under `text_config`; older
  // single-modality configs put `max_position_embeddings` at the
  // top level. Accept either layout.
  let max_pos = v
    .get("text_config")
    .and_then(|tc| tc.get("max_position_embeddings"))
    .or_else(|| v.get("max_position_embeddings"))
    .and_then(|n| n.as_u64())
    .ok_or(Error::InvalidRequest(
      "config.json missing text_config.max_position_embeddings (or top-level max_position_embeddings)",
    ))?;
  if max_pos != crate::options::MODEL_CONTEXT_TOKENS as u64 {
    return Err(Error::InvalidRequest(
      "config.json max_position_embeddings differs from crate's MODEL_CONTEXT_TOKENS (128_000) — admission gates would accept requests past the loaded model's real position limit",
    ));
  }
  // Absent ⇒ `true` (upstream's default), so only an explicit `false` — a
  // checkpoint whose processor contract omits the image brackets this crate
  // unconditionally renders and budgets — is refused.
  if v.get("use_image_special_tokens") == Some(&serde_json::Value::Bool(false)) {
    return Err(Error::InvalidRequest(
      "config.json use_image_special_tokens is false — this crate always brackets an image block with <|image_start|>/<|image_end|> and budgets both tokens, so the rendered prompt would carry two tokens the checkpoint's processor contract omits",
    ));
  }
  Ok(())
}

/// Validate the model directory's `chat_template.jinja` byte-equals
/// the bundled jinja used by the engine at render time. The
/// renderer uses `chat_template::BUNDLED_CHAT_TEMPLATE_JINJA`
/// regardless of what the model directory ships, so if a model
/// revision changes the template (e.g., role wrapper, image-block
/// layout) but keeps `tokenizer.json` byte-identical, `from_dir`
/// would silently accept the directory; the engine would render
/// with the bundled template; the resulting prompt would be
/// semantically wrong even though `<image>` token counts still
/// match. Fail closed.
///
/// Fail-closed on missing file: from_dir is the strict constructor.
/// Callers with stripped-down model directories should use
/// `from_paths` (which explicitly opts out) or `from_onnx_dir`
/// (which uses bundled assets, so no drift is possible).
#[cfg(feature = "bundled")]
fn validate_chat_template_matches_bundled(path: &Path) -> Result<()> {
  if !path.exists() {
    return Err(Error::InvalidRequest(
      "model directory missing chat_template.jinja — use from_paths to bypass strict prompt-template drift checks (advanced: requires matching ONNX embedding table)",
    ));
  }
  let supplied = std::fs::read(path).map_err(Error::Io)?;
  if !chat_templates_render_alike(&supplied, crate::bundled::CHAT_TEMPLATE_JINJA) {
    return Err(Error::InvalidRequest(
      "supplied chat_template.jinja does not render the same prompt as the bundled chat template — engine renders with bundled template; mismatched model template would produce semantically wrong prompts even when <image> counts line up",
    ));
  }
  Ok(())
}

/// Whether two Jinja chat templates render identical prompts.
///
/// This is a **drift detector**, not a checksum, but it is deliberately only
/// one step away from one: the sole normalization is the **leading**
/// comment/whitespace header, after which the two bodies must be byte-equal.
/// Every emitting construct is therefore still compared byte for byte, so a
/// changed role envelope, image-block wrapping, or literal is refused.
///
/// The motivating case is real: `LiquidAI/LFM2.5-VL-450M-MLX-8bit` ships the
/// same template as the ONNX export with a two-line
/// `{# <|tool_list_start|> detection hint for mlx_lm #}` header prepended. A
/// byte-equality check refuses that checkpoint over a comment — making the
/// strict MLX constructor unusable against the released weights while catching
/// nothing.
///
/// # Why only the leading header
///
/// Erasing every `{# … #}` span anywhere in the file is **not** sound without a
/// Jinja lexer, because `{#` only opens a comment in template-text context. A
/// template whose body reads `{{- "sys{# drift #}tem" -}}` renders
/// `sys{# drift #}tem`, but a blind span-strip rewrites it to the bundled
/// `{{- "system" -}}` and reports no drift — a checkpoint with a different
/// prompt contract passing the strict constructor, which is exactly what this
/// gate exists to prevent. At offset zero there is no such ambiguity: the lexer
/// starts in text context, so a leading `{#` really is a comment, and the
/// whitespace it leaves behind cannot reach the output because the bundled
/// body's first emitting construct is `{{- bos_token -}}`, whose `-` strips it.
///
/// (The alternative — comparing parser tokens — would need minijinja's
/// `unstable_machinery` feature, an explicitly unstable API this crate does not
/// enable; the prefix rule needs no lexer at all.)
///
/// Trailing whitespace is NOT trimmed: nothing guarantees it is stripped, so a
/// difference there is still drift. A comment anywhere past the header is drift
/// too — fail-closed, and `from_paths` / the `_unchecked` constructors remain
/// the named door for a checkpoint that legitimately carries one.
#[cfg(feature = "bundled")]
fn chat_templates_render_alike(supplied: &[u8], bundled: &[u8]) -> bool {
  strip_leading_comment_header(supplied) == strip_leading_comment_header(bundled)
}

/// Drop a template's leading run of Jinja comments and ASCII whitespace,
/// returning the body that follows.
///
/// Only complete `{# … #}` spans at the very start (separated by optional
/// whitespace) are consumed; the scan stops at the first byte that opens
/// neither. An unterminated `{#` is left verbatim rather than swallowing the
/// rest of the file: a template that cannot be tokenized is drift, and silently
/// discarding its tail would hide that.
#[cfg(feature = "bundled")]
fn strip_leading_comment_header(template: &[u8]) -> &[u8] {
  const OPEN: &[u8] = b"{#";
  const CLOSE: &[u8] = b"#}";
  let mut rest = template.trim_ascii_start();
  while let Some(body) = rest.strip_prefix(OPEN) {
    let Some(end) = find_subslice(body, CLOSE) else {
      break; // unterminated comment — keep the remainder verbatim
    };
    rest = body[end + CLOSE.len()..].trim_ascii_start();
  }
  rest
}

/// Index of the first occurrence of `needle` in `haystack`.
#[cfg(feature = "bundled")]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
  if needle.is_empty() || haystack.len() < needle.len() {
    return None;
  }
  haystack
    .windows(needle.len())
    .position(|window| window == needle)
}

/// Verify the supplied `tokenizer.json` byte-matches the bundled blob: a
/// tokenizer with the same special-token IDs but a drifted normal vocabulary
/// would pass the per-token contract check yet still encode normal text into
/// different IDs that don't match the embedding table — silent global prompt
/// corruption.
///
/// Called from `from_dir` (strict constructor) only; `from_paths` remains the
/// explicit escape hatch for callers intentionally pairing custom tokenizers
/// with custom ONNX.
#[cfg(feature = "bundled")]
fn validate_tokenizer_matches_bundled(path: &Path) -> Result<()> {
  let supplied = std::fs::read(path).map_err(Error::Io)?;
  if supplied != crate::bundled::TOKENIZER_JSON {
    return Err(Error::InvalidRequest(
      "supplied tokenizer.json bytes do not match the bundled tokenizer — use Engine::from_paths to bypass strict tokenizer-identity check (advanced: requires matching ONNX embedding table)",
    ));
  }
  Ok(())
}

// =========================================================================
// Bundled-tokenizer helper (bundled feature only)
// =========================================================================

/// Write the bundled `tokenizer.json` bytes to a content-addressed
/// temp file and return its path. hardening:
///
/// - **Content-addressed path**: `$TMPDIR/lfm-bundled-<sha256_first_16hex>/`
///   so different lfm versions (or different bundled bytes) get
///   distinct paths. PID-based paths could reuse a stale file from a
///   previous process whose PID got recycled.
/// - **Initialize once**: a per-process `OnceLock<PathBuf>` ensures
///   only one writer races for a given target path within this
///   process. Cross-process races are still possible — see atomic
///   write below.
/// - **Atomic write via tempfile + rename**: write to a sibling
///   `<random>.tmp` file then rename to the final name. Concurrent
///   readers see either no file or the fully-written file — never
///   a partial write.
/// - **Content verification on reuse**: if the target file already
///   exists, verify its bytes match the bundled blob before reusing.
///   Mismatch → rewrite (same atomic dance).
// Only `from_onnx_dir` (ONNX-only; the MLX road has no bundled-tokenizer
// door) calls this outside tests, so it needs `ort_backend` too.
#[cfg(all(feature = "bundled", ort_backend))]
fn write_bundled_tokenizer() -> Result<PathBuf> {
  use std::sync::Mutex;
  // hardening still had a TOCTOU
  // race between the OnceLock check and the temp-file write. Two
  // threads both calling Engine::from_onnx_dir simultaneously could
  // observe an empty cache and both write to the same
  // tokenizer.json.<PID>.tmp; one rename would then remove the
  // other's temp before the second rename ran, causing a spurious
  // failure.
  //
  // Fix: serialize the entire init under a Mutex<Option<PathBuf>>.
  // Fast path (cache hit) is just a lock + clone. Slow path holds
  // the lock during the FS work, but that's fine — Engine
  // construction is rare and the work is bounded (~5 MB write +
  // rename). Also add a thread-id to the temp filename as
  // belt-and-suspenders against any future concurrent code.
  static CACHE: Mutex<Option<PathBuf>> = Mutex::new(None);
  let mut guard = CACHE
    .lock()
    .expect("write_bundled_tokenizer mutex poisoned");
  if let Some(p) = guard.as_ref() {
    // on every cache hit, re-read the
    // file and verify it still matches the bundled blob. If a
    // process (ours or another) has modified the cached temp file
    // between calls, the old code would happily return the stale
    // path and from_paths would consume the tampered tokenizer
    // (its structural-ID validation can't catch normal-vocab
    // drift). Re-validation forces a rewrite on tamper.
    match std::fs::read(p) {
      Ok(existing) if existing == crate::bundled::TOKENIZER_JSON => return Ok(p.clone()),
      _ => {
        // Tampered or removed — drop the cache and fall through
        // to the rewrite path below.
        *guard = None;
      }
    }
  }

  // Content hash: 16 hex chars of FNV-1a over the bundled bytes
  // (not crypto, just enough entropy to namespace by content).
  let hash = simple_hash_hex(crate::bundled::TOKENIZER_JSON);
  let dir = std::env::temp_dir().join(format!("lfm-bundled-{hash}"));
  std::fs::create_dir_all(&dir).map_err(Error::Io)?;
  let path = dir.join("tokenizer.json");

  // If file already exists, verify content matches before reuse.
  let needs_write = match std::fs::read(&path) {
    Ok(existing) if existing == crate::bundled::TOKENIZER_JSON => false,
    Ok(_) => true,
    Err(_) => true,
  };
  if needs_write {
    // Per-thread + per-process unique temp filename. Even though
    // the Mutex serializes within a process, the thread id makes
    // cross-process collisions on the temp filename impossible.
    let tid = std::thread::current().id();
    let tmp = dir.join(format!(
      "tokenizer.json.{}.{:?}.tmp",
      std::process::id(),
      tid
    ));
    std::fs::write(&tmp, crate::bundled::TOKENIZER_JSON).map_err(Error::Io)?;
    // rename can fail on Windows if
    // destination already exists (another process won the race
    // between our needs_write check and our rename). Recover by
    // re-reading the destination — if its bytes match the bundled
    // blob, the other process produced a correct file and we can
    // accept it. If the rename fails for any other reason, or if
    // the bytes still don't match, propagate the original error.
    if let Err(rename_err) = std::fs::rename(&tmp, &path) {
      let _ = std::fs::remove_file(&tmp); // clean up our temp
      match std::fs::read(&path) {
        Ok(existing) if existing == crate::bundled::TOKENIZER_JSON => {
          // Another process beat us; their file is correct. Accept.
        }
        _ => return Err(Error::Io(rename_err)),
      }
    }
  }

  *guard = Some(path.clone());
  Ok(path)
}

/// Simple content hash producing a 16-char hex string. Not crypto;
/// just enough entropy to namespace bundled bytes by content.
///
/// Only called from `write_bundled_tokenizer`, so it carries the same gate.
#[cfg(all(feature = "bundled", ort_backend))]
fn simple_hash_hex(bytes: &[u8]) -> String {
  // FNV-1a 64-bit
  let mut h: u64 = 0xcbf29ce484222325;
  for &b in bytes {
    h ^= b as u64;
    h = h.wrapping_mul(0x100000001b3);
  }
  format!("{h:016x}")
}

/// Paths to the four model files used by [`Engine::from_paths`].
pub struct EnginePaths {
  /// Path to `vision_encoder.onnx`.
  vision: PathBuf,
  /// Path to `embed_tokens.onnx`.
  embed: PathBuf,
  /// Path to `decoder_model_merged.onnx`.
  decoder: PathBuf,
  /// Path to `tokenizer.json`.
  tokenizer: PathBuf,
}

impl EnginePaths {
  /// Construct a new `EnginePaths`.
  pub fn new(vision: PathBuf, embed: PathBuf, decoder: PathBuf, tokenizer: PathBuf) -> Self {
    Self {
      vision,
      embed,
      decoder,
      tokenizer,
    }
  }

  /// Path to `vision_encoder.onnx`.
  pub fn vision(&self) -> &PathBuf {
    &self.vision
  }

  /// Path to `embed_tokens.onnx`.
  pub fn embed(&self) -> &PathBuf {
    &self.embed
  }

  /// Path to `decoder_model_merged.onnx`.
  pub fn decoder(&self) -> &PathBuf {
    &self.decoder
  }

  /// Path to `tokenizer.json`.
  pub fn tokenizer(&self) -> &PathBuf {
    &self.tokenizer
  }

  /// Set the vision encoder path.
  pub fn set_vision(&mut self, vision: PathBuf) {
    self.vision = vision;
  }

  /// Set the embed tokens path.
  pub fn set_embed(&mut self, embed: PathBuf) {
    self.embed = embed;
  }

  /// Set the decoder path.
  pub fn set_decoder(&mut self, decoder: PathBuf) {
    self.decoder = decoder;
  }

  /// Set the tokenizer path.
  pub fn set_tokenizer(&mut self, tokenizer: PathBuf) {
    self.tokenizer = tokenizer;
  }

  /// Builder: set the vision encoder path (chainable).
  pub fn with_vision(mut self, vision: PathBuf) -> Self {
    self.vision = vision;
    self
  }

  /// Builder: set the embed tokens path (chainable).
  pub fn with_embed(mut self, embed: PathBuf) -> Self {
    self.embed = embed;
    self
  }

  /// Builder: set the decoder path (chainable).
  pub fn with_decoder(mut self, decoder: PathBuf) -> Self {
    self.decoder = decoder;
    self
  }

  /// Builder: set the tokenizer path (chainable).
  pub fn with_tokenizer(mut self, tokenizer: PathBuf) -> Self {
    self.tokenizer = tokenizer;
    self
  }
}

// =========================================================================
// llguidance wiring helpers (inference feature only)
// =========================================================================

/// Build a `ParserFactory` from the tokenizer JSON bytes.
///
/// Steps:
/// 1. `ByteTokenizer::from_json_bytes(bytes)` — loads from in-memory
///    bytes captured at Engine construction (eliminating any
///    path-reload TOCTOU). Uses `toktrie_hf_tokenizers`'s
///    own `tokenizers` dependency (v0.21), avoiding a type-incompatibility
///    with the `tokenizers` v0.23 used elsewhere in `lfm`. This is safe:
///    both versions read the same `tokenizer.json` format.
/// 2. `.into_tok_env(None)` — builds a `TokTrie` and wraps it in
///    `Arc<dyn TokenizerEnv>` (`TokEnv`).
/// 3. `ParserFactory::new_simple(&tok_env)` — compiles with
///    `InferenceCapabilities::default()` (ff_tokens disabled) and
///    `SlicedBiasComputer::general_slices()`.
fn build_parser_factory(tokenizer_bytes: &[u8]) -> Result<ParserFactory> {
  let byte_tok = toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(tokenizer_bytes)
    .map_err(Error::llguidance)?;
  let tok_env: TokEnv = byte_tok.into_tok_env(None).map_err(Error::llguidance)?;
  ParserFactory::new_simple(&tok_env).map_err(Error::llguidance)
}

/// Build a `Constraint` for one generation call from any
/// [`llmtask::Grammar`] variant.
///
/// llguidance natively supports JSON Schema, Lark, and Regex —
/// all three [`Grammar`] variants this crate's [`llmtask::Task`]
/// can produce. Each variant routes to its corresponding
/// `TopLevelGrammar` constructor.
fn build_constraint(factory: &ParserFactory, grammar: &llmtask::Grammar) -> Result<Constraint> {
  let top = match grammar {
    llmtask::Grammar::JsonSchema(schema) => TopLevelGrammar::from_json_schema(schema.clone()),
    llmtask::Grammar::Lark(src) => TopLevelGrammar::from_lark(src.to_string()),
    // Grammar::Regex wraps a private RegexGrammar with both the
    // source pattern and a default-options compiled regex —
    // forcing default options prevents `RegexBuilder::
    // case_insensitive(true)`-smuggled regexes from diverging
    // between local validation and engine constraint. Borrow the
    // source pattern via `pattern()` and hand it to llguidance.
    llmtask::Grammar::Regex(rg) => TopLevelGrammar::from_regex(rg.pattern()),
    // Grammar is #[non_exhaustive]; future variants (e.g., raw
    // CFG, GBNF) would land here. lfm via llguidance can support
    // most of them but they need a per-variant routing change.
    _ => {
      return Err(Error::InvalidRequest(
        "llmtask::Grammar variant unsupported by lfm — please open an issue (lfm uses llguidance and can extend support)",
      ));
    }
  };
  let parser = factory.create_parser(top).map_err(Error::llguidance)?;
  Ok(Constraint::new(parser))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn engine_paths_accessors() {
    let ep = EnginePaths::new(
      PathBuf::from("v.onnx"),
      PathBuf::from("e.onnx"),
      PathBuf::from("d.onnx"),
      PathBuf::from("t.json"),
    );
    assert_eq!(ep.vision(), &PathBuf::from("v.onnx"));
    assert_eq!(ep.tokenizer(), &PathBuf::from("t.json"));
  }

  #[test]
  fn validate_image_tokenizer_contract_caps_max_tiles() {
    // Defense-in-depth: even if a caller forgot ImageBudget::validate(),
    // the contract validator must refuse to scan past
    // MAX_TOKENIZER_TILE_DIM. Without this guard, max_tiles=usize::MAX
    // would loop ~∞ in the nested R×C scan and hang Engine
    // construction (a startup-DoS path).
    //
    // Exercise via the bundled tokenizer (available under `bundled` +
    // `ort_backend` — `write_bundled_tokenizer` is the ONNX-only bundled
    // tokenizer writer; on aarch64-macos without the `ort` feature this
    // block is skipped and the cap behavior goes untested here, covered
    // instead by the MLX-road strict-constructor tests).
    #[cfg(all(feature = "bundled", ort_backend))]
    {
      let path = write_bundled_tokenizer().expect("write bundled tokenizer");
      let tokenizer = Tokenizer::from_file(&path).expect("load tokenizer");
      let r =
        validate_image_tokenizer_contract(&tokenizer, crate::options::MAX_TOKENIZER_TILE_DIM + 1);
      assert!(
        matches!(r, Err(Error::InvalidBudget(_))),
        "must reject max_tiles above the cap, got {r:?}"
      );
      // Sanity: at the cap it succeeds.
      assert!(
        validate_image_tokenizer_contract(&tokenizer, crate::options::MAX_TOKENIZER_TILE_DIM)
          .is_ok()
      );
    }
  }

  #[test]
  #[cfg(feature = "bundled")]
  fn validate_tokenizer_matches_bundled_rejects_drift() {
    // a tokenizer with valid special-token IDs but
    // any drift in normal vocabulary must be rejected by the strict
    // constructor. Reproducer: write a 1-byte mutation of the
    // bundled tokenizer.json to a temp file and verify the helper
    // rejects it. (We can't easily craft a "valid JSON tokenizer
    // with one normal-token ID swapped" without a tokenizer-aware
    // mutation library, but ANY byte difference must trip the
    // byte-equality check, which is the whole point of the helper.)
    let dir = std::env::temp_dir().join(format!("lfm-test-drift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let drift_path = dir.join("tokenizer-drift.json");
    let mut bytes = crate::bundled::TOKENIZER_JSON.to_vec();
    // Mutate a single byte deep in the file (vocab section) so we
    // don't accidentally produce something that's still valid by
    // coincidence. Last byte is safest.
    let last = bytes.len() - 1;
    bytes[last] = bytes[last].wrapping_add(1);
    std::fs::write(&drift_path, &bytes).unwrap();
    let result = validate_tokenizer_matches_bundled(&drift_path);
    assert!(
      matches!(result, Err(Error::InvalidRequest(_))),
      "drifted tokenizer must be rejected, got {result:?}"
    );

    // Sanity: writing the unmodified bundled bytes passes.
    let ok_path = dir.join("tokenizer-ok.json");
    std::fs::write(&ok_path, crate::bundled::TOKENIZER_JSON).unwrap();
    assert!(validate_tokenizer_matches_bundled(&ok_path).is_ok());
  }

  #[test]
  #[cfg(feature = "bundled")]
  fn validate_config_contract_matches_bundled_accepts_correct_and_rejects_drift() {
    // from_dir's strict drift check for
    // text_config.max_position_embeddings vs MODEL_CONTEXT_TOKENS.
    let dir = std::env::temp_dir().join(format!("lfm-test-config-drift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Missing → reject.
    let missing = dir.join("config-missing.json");
    let _ = std::fs::remove_file(&missing);
    assert!(matches!(
      validate_config_contract_matches_bundled(&missing),
      Err(Error::InvalidRequest(_))
    ));

    // Wrong context → reject.
    let drift = dir.join("config-drift.json");
    std::fs::write(
      &drift,
      r#"{"text_config":{"max_position_embeddings":4096}}"#,
    )
    .unwrap();
    assert!(matches!(
      validate_config_contract_matches_bundled(&drift),
      Err(Error::InvalidRequest(_))
    ));

    // Correct nested layout → ok.
    let ok_nested = dir.join("config-ok-nested.json");
    std::fs::write(
      &ok_nested,
      r#"{"text_config":{"max_position_embeddings":128000}}"#,
    )
    .unwrap();
    assert!(validate_config_contract_matches_bundled(&ok_nested).is_ok());

    // Correct top-level layout → ok (older single-modality configs).
    let ok_flat = dir.join("config-ok-flat.json");
    std::fs::write(&ok_flat, r#"{"max_position_embeddings":128000}"#).unwrap();
    assert!(validate_config_contract_matches_bundled(&ok_flat).is_ok());

    // Bundled config.json → ok.
    let ok_bundled = dir.join("config-ok-bundled.json");
    std::fs::write(&ok_bundled, crate::bundled::CONFIG_JSON).unwrap();
    assert!(validate_config_contract_matches_bundled(&ok_bundled).is_ok());

    // Invalid JSON → reject.
    let bad_json = dir.join("config-bad.json");
    std::fs::write(&bad_json, b"{not json").unwrap();
    assert!(matches!(
      validate_config_contract_matches_bundled(&bad_json),
      Err(Error::InvalidRequest(_))
    ));
  }

  /// The ONNX road's twin of the MLX `use_image_special_tokens` contract: an
  /// otherwise-correct `config.json` that turns the image brackets OFF is
  /// refused, because the renderer emits `<|image_start|>` / `<|image_end|>`
  /// and `ImagePlan` budgets both unconditionally. An explicit `true`, and an
  /// absent key (upstream's default is `true`), both pass.
  #[test]
  #[cfg(feature = "bundled")]
  fn validate_config_contract_rejects_disabled_image_special_tokens() {
    let dir = std::env::temp_dir().join(format!("lfm-test-config-brackets-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, body: &str| {
      let p = dir.join(name);
      std::fs::write(&p, body).unwrap();
      p
    };

    let off = write(
      "config-brackets-off.json",
      r#"{"max_position_embeddings":128000,"use_image_special_tokens":false}"#,
    );
    assert!(
      matches!(
        validate_config_contract_matches_bundled(&off),
        Err(Error::InvalidRequest(_))
      ),
      "use_image_special_tokens=false must be refused"
    );

    let on = write(
      "config-brackets-on.json",
      r#"{"max_position_embeddings":128000,"use_image_special_tokens":true}"#,
    );
    assert!(validate_config_contract_matches_bundled(&on).is_ok());

    let absent = write(
      "config-brackets-absent.json",
      r#"{"max_position_embeddings":128000}"#,
    );
    assert!(
      validate_config_contract_matches_bundled(&absent).is_ok(),
      "an absent key means upstream's default (true) and must not be refused"
    );
  }

  #[test]
  #[cfg(feature = "bundled")]
  fn validate_chat_template_matches_bundled_rejects_drift_and_missing() {
    // from_dir's strict drift check for
    // the model directory's chat_template.jinja. A model rev that
    // changes the template (role envelope, image-block layout)
    // while keeping tokenizer.json byte-identical must be rejected.
    let dir = std::env::temp_dir().join(format!("lfm-test-tmpl-drift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Missing file → reject (fail-closed for strict constructor).
    let missing = dir.join("chat_template-missing.jinja");
    let _ = std::fs::remove_file(&missing);
    assert!(matches!(
      validate_chat_template_matches_bundled(&missing),
      Err(Error::InvalidRequest(_))
    ));

    // Drifted bytes → reject. Mutate the last byte.
    let drift = dir.join("chat_template-drift.jinja");
    let mut bytes = crate::bundled::CHAT_TEMPLATE_JINJA.to_vec();
    let last = bytes.len() - 1;
    bytes[last] = bytes[last].wrapping_add(1);
    std::fs::write(&drift, &bytes).unwrap();
    assert!(matches!(
      validate_chat_template_matches_bundled(&drift),
      Err(Error::InvalidRequest(_))
    ));

    // Sanity: bundled bytes pass.
    let ok = dir.join("chat_template-ok.jinja");
    std::fs::write(&ok, crate::bundled::CHAT_TEMPLATE_JINJA).unwrap();
    assert!(validate_chat_template_matches_bundled(&ok).is_ok());
  }

  /// The template check normalizes exactly one thing — the leading
  /// comment/whitespace header — and compares the rest byte for byte.
  ///
  /// The accepted case is the released `LiquidAI/LFM2.5-VL-450M-MLX-8bit`
  /// checkpoint, which ships the bundled template with a two-line Jinja comment
  /// prepended for mlx_lm's tool detection. A comment at offset zero is
  /// unambiguously a comment (the lexer starts in text context) and the
  /// whitespace it leaves behind is swallowed by the template's own
  /// `{{- bos_token -}}`.
  #[test]
  #[cfg(feature = "bundled")]
  fn chat_template_comment_header_is_not_drift_but_content_is() {
    let bundled = crate::bundled::CHAT_TEMPLATE_JINJA;

    // The real MLX-8bit header: a comment plus a blank line.
    let mut mlx_style = b"{# <|tool_list_start|> detection hint for mlx_lm #}\n\n".to_vec();
    mlx_style.extend_from_slice(bundled);
    assert!(
      chat_templates_render_alike(&mlx_style, bundled),
      "a prepended Jinja comment cannot change the rendered prompt and must not be reported as drift"
    );

    // Several stacked header comments, with whitespace between and around them.
    let mut stacked = b"\n  {# one #}\n{# two #}\t\n\n".to_vec();
    stacked.extend_from_slice(bundled);
    assert!(chat_templates_render_alike(&stacked, bundled));

    // Real content changes are still refused: a mutated literal…
    let mut mutated = bundled.to_vec();
    let last = mutated.len() - 1;
    mutated[last] = mutated[last].wrapping_add(1);
    assert!(!chat_templates_render_alike(&mutated, bundled));

    // …trailing whitespace, which nothing guarantees is trimmed away…
    let mut trailing = bundled.to_vec();
    trailing.push(b'\n');
    assert!(!chat_templates_render_alike(&trailing, bundled));

    // …and an unterminated comment, which leaves the template untokenizable.
    let mut unterminated = b"{# never closed\n".to_vec();
    unterminated.extend_from_slice(bundled);
    assert!(
      !chat_templates_render_alike(&unterminated, bundled),
      "an unterminated comment must not swallow the rest of the file"
    );
  }

  /// The counterexample that rules out stripping `{# … #}` spans anywhere in
  /// the file: `{#` opens a comment only in template-text context, so a span
  /// that sits INSIDE a quoted string literal is rendered output, not a
  /// comment.
  ///
  /// The drifted template below is the bundled one with `{# drift #}` inserted
  /// into the assistant-turn literal, so it renders
  /// `<|im_start|>assistant{# drift #}\n` where the bundled template renders
  /// `<|im_start|>assistant\n` — a different prompt contract. It is also,
  /// byte for byte, nothing but an inserted comment span (asserted below), so a
  /// blind span-strip normalizes it straight back onto the bundled bytes and
  /// reports no drift. Only the leading-header rule refuses it.
  #[test]
  #[cfg(feature = "bundled")]
  fn chat_template_comment_inside_a_string_literal_is_drift() {
    let bundled = crate::bundled::CHAT_TEMPLATE_JINJA;
    const SPAN: &[u8] = b"{# drift #}";
    // Insert inside the quoted literal of `{{- "<|im_start|>assistant\n" -}}`.
    let needle = b"<|im_start|>assistant";
    let at = find_subslice(bundled, needle).expect("bundled template renders an assistant turn")
      + needle.len();

    let mut drifted = bundled.to_vec();
    drifted.splice(at..at, SPAN.iter().copied());
    // The ONLY difference is an inserted comment span — i.e. exactly what a
    // general span-strip erases, and exactly why it cannot be trusted.
    let mut unspliced = drifted.clone();
    unspliced.drain(at..at + SPAN.len());
    assert_eq!(
      unspliced, bundled,
      "the counterexample must differ from the bundled template by the comment span alone"
    );

    assert!(
      !chat_templates_render_alike(&drifted, bundled),
      "a `{{# … #}}` inside a quoted string is rendered text, not a comment — it must be drift"
    );
  }
  // =======================================================================
  // Preprocessing-contract drift (both roads)
  // =======================================================================

  /// The mutations that make a checkpoint's declared pixel arithmetic disagree
  /// with what BOTH patchifiers bake in. Each is `(name, json patch applied to
  /// the bundled preprocessor_config)`.
  ///
  /// Every one of these leaves the tile grid, the sub-image count and every
  /// `<image>`-token total identical, so nothing downstream can notice: the
  /// vision tower simply receives systematically wrong numbers.
  #[cfg(feature = "bundled")]
  fn preprocessing_skew_cases() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
      ("image_mean", json!([0.485, 0.456, 0.406])),
      ("image_std", json!([0.229, 0.224, 0.225])),
      ("rescale_factor", json!(1.0 / 127.5)),
      ("resample", json!(3)),
      ("do_normalize", json!(false)),
      ("do_rescale", json!(false)),
      ("do_resize", json!(false)),
      ("do_pad", json!(false)),
      ("data_format", json!("channels_last")),
    ]
  }

  /// Apply one skew case to the bundled `preprocessor_config.json` and return
  /// the mutated bytes.
  #[cfg(feature = "bundled")]
  fn skewed_preprocessor_config(key: &str, value: &serde_json::Value) -> Vec<u8> {
    let mut cfg: serde_json::Value =
      serde_json::from_slice(crate::bundled::PREPROCESSOR_CONFIG_JSON)
        .expect("bundled preprocessor_config.json is valid JSON");
    cfg
      .as_object_mut()
      .expect("preprocessor_config.json is an object")
      .insert(key.to_string(), value.clone());
    serde_json::to_vec_pretty(&cfg).expect("serialize mutated config")
  }

  /// The shared pixel contract refuses every normalization / rescale /
  /// resampling skew, and accepts the bundled config unchanged.
  #[test]
  #[cfg(feature = "bundled")]
  fn preprocessing_pixel_contract_refuses_normalization_and_resampling_skew() {
    let clean: serde_json::Value =
      serde_json::from_slice(crate::bundled::PREPROCESSOR_CONFIG_JSON).unwrap();
    assert!(
      check_preprocessing_pixel_contract(&clean).is_ok(),
      "the bundled preprocessor_config.json must satisfy the contract it defines"
    );

    for (key, value) in preprocessing_skew_cases() {
      let mutated: serde_json::Value =
        serde_json::from_slice(&skewed_preprocessor_config(key, &value)).unwrap();
      let result = check_preprocessing_pixel_contract(&mutated);
      assert!(
        matches!(result, Err(Error::InvalidRequest(_))),
        "a checkpoint declaring {key} = {value} must be refused, got {result:?}"
      );
    }
  }

  /// A missing `preprocessor_config.json` fails closed rather than skipping the
  /// contract, and the failure names the unchecked doors.
  #[test]
  #[cfg(feature = "bundled")]
  fn missing_preprocessor_config_fails_closed() {
    let dir = std::env::temp_dir().join(format!("lfm-test-preproc-missing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let missing = dir.join("preprocessor_config.json");
    let _ = std::fs::remove_file(&missing);
    let result = read_preprocessor_config(&missing);
    assert!(
      matches!(&result, Err(Error::InvalidRequest(msg)) if msg.contains("unchecked")),
      "absence must be refused by name, got {result:?}"
    );
  }

  /// The ONNX strict road refuses the same skew — `from_dir`'s validator is
  /// built on the shared contract, so this is a regression guard on the
  /// composition, not a duplicate of the contract test.
  #[test]
  #[cfg(feature = "bundled")]
  fn onnx_preprocessor_validator_refuses_skew_and_accepts_bundled() {
    let dir = std::env::temp_dir().join(format!("lfm-test-preproc-onnx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let ok = dir.join("preprocessor_config.json");
    std::fs::write(&ok, crate::bundled::PREPROCESSOR_CONFIG_JSON).unwrap();
    assert!(validate_preprocessor_config(&ok).is_ok());

    for (key, value) in preprocessing_skew_cases() {
      let path = dir.join(format!("preprocessor_config-{key}.json"));
      std::fs::write(&path, skewed_preprocessor_config(key, &value)).unwrap();
      let result = validate_preprocessor_config(&path);
      assert!(
        matches!(result, Err(Error::InvalidRequest(_))),
        "ONNX strict road must refuse {key} = {value}, got {result:?}"
      );
    }
  }

  /// Every STRICT MLX constructor refuses a checkpoint whose declared
  /// preprocessing disagrees with what `mlxrs` bakes in — and `_unchecked`
  /// stays the only escape.
  ///
  /// The checkpoint directory is deliberately weightless: a strict constructor
  /// validates identity BEFORE it loads anything, so a refusal that names
  /// `preprocessor_config.json` proves the gate ran, and the clean directory's
  /// different failure (no `config.json` / no weights) proves the gate was
  /// passed rather than short-circuiting everything.
  #[test]
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "bundled"))]
  fn strict_mlx_constructors_refuse_preprocessing_skew() {
    let root = std::env::temp_dir().join(format!("lfm-test-mlx-preproc-{}", std::process::id()));

    let write_checkpoint = |name: &str, preprocessor: &[u8]| -> PathBuf {
      let dir = root.join(name);
      std::fs::create_dir_all(&dir).unwrap();
      std::fs::write(dir.join("tokenizer.json"), crate::bundled::TOKENIZER_JSON).unwrap();
      std::fs::write(
        dir.join("chat_template.jinja"),
        crate::bundled::CHAT_TEMPLATE_JINJA,
      )
      .unwrap();
      std::fs::write(dir.join("preprocessor_config.json"), preprocessor).unwrap();
      dir
    };

    let names_preprocessing = |e: &Error| e.to_string().contains("preprocessor_config.json");

    // ── the skewed checkpoints are refused, by name ──────────────────────
    for (key, value) in preprocessing_skew_cases() {
      let dir = write_checkpoint(key, &skewed_preprocessor_config(key, &value));

      let err = Engine::from_mlx_dir(&dir, Options::default())
        .err()
        .unwrap_or_else(|| panic!("from_mlx_dir must refuse {key} = {value}"));
      assert!(
        names_preprocessing(&err),
        "from_mlx_dir must refuse {key} = {value} by naming preprocessor_config.json, got {err}"
      );

      // The explicit-format doors read the sibling config from the weight
      // file's parent, so the same skew must stop them too.
      let weights = dir.join("model.safetensors");
      let err = Engine::from_mlx_safetensors(&weights, Options::default())
        .err()
        .unwrap_or_else(|| panic!("from_mlx_safetensors must refuse {key} = {value}"));
      assert!(
        names_preprocessing(&err),
        "from_mlx_safetensors must refuse {key} = {value}, got {err}"
      );

      #[cfg(feature = "npz")]
      {
        let err = Engine::from_mlx_npz(dir.join("weights.npz"), Options::default())
          .err()
          .unwrap_or_else(|| panic!("from_mlx_npz must refuse {key} = {value}"));
        assert!(names_preprocessing(&err), "from_mlx_npz: {err}");
      }
      #[cfg(feature = "gguf")]
      {
        let err = Engine::from_mlx_gguf(dir.join("weights.gguf"), Options::default())
          .err()
          .unwrap_or_else(|| panic!("from_mlx_gguf must refuse {key} = {value}"));
        assert!(names_preprocessing(&err), "from_mlx_gguf: {err}");
      }
    }

    // ── absence fails closed on the MLX road too ─────────────────────────
    let bare = root.join("no-preprocessor-config");
    std::fs::create_dir_all(&bare).unwrap();
    std::fs::write(bare.join("tokenizer.json"), crate::bundled::TOKENIZER_JSON).unwrap();
    std::fs::write(
      bare.join("chat_template.jinja"),
      crate::bundled::CHAT_TEMPLATE_JINJA,
    )
    .unwrap();
    let err = Engine::from_mlx_dir(&bare, Options::default())
      .err()
      .expect("a checkpoint without preprocessor_config.json must be refused");
    assert!(
      names_preprocessing(&err),
      "absence must be named, got {err}"
    );

    // ── a conforming checkpoint gets PAST the gate (and then fails on the
    //    weights it does not have), so the gate is not refusing everything ──
    let clean = write_checkpoint("clean", crate::bundled::PREPROCESSOR_CONFIG_JSON);
    let err = Engine::from_mlx_dir(&clean, Options::default())
      .err()
      .expect("a weightless checkpoint cannot load");
    assert!(
      !names_preprocessing(&err),
      "a conforming preprocessor_config.json must pass the gate; failure was {err}"
    );

    // ── `_unchecked` is the escape: it never consults the file at all ─────
    let skewed = write_checkpoint(
      "unchecked-escape",
      &skewed_preprocessor_config("image_mean", &serde_json::json!([0.1, 0.2, 0.3])),
    );
    let err = Engine::from_mlx_dir_unchecked(&skewed, Options::default())
      .err()
      .expect("a weightless checkpoint cannot load");
    assert!(
      !names_preprocessing(&err),
      "the unchecked door must not run the preprocessing gate; failure was {err}"
    );
  }
}
