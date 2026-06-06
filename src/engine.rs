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
  options::{Options, RequestOptions},
  preproc::Preprocessor,
  runtime::{
    backend::{BackendImpl, OrtBackend},
    decoder::Decoder,
    embed_tokens::EmbedTokens,
    sampler::{ConstrainedSampler, FreeSampler},
    vision::VisionEncoder,
  },
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
  /// **Strict constructor.** Validates `preprocessor_config.json`
  /// matches our hardcoded preprocessing constants AND validates the
  /// supplied `tokenizer.json` byte-matches the bundled blob — a
  /// custom tokenizer whose normal vocabulary drifts from what the
  /// embedding table expects would silently corrupt prompts.
  /// Requires the `bundled` feature so the
  /// byte-compare reference is available; without it, use
  /// [`Engine::from_paths`] (the unchecked escape hatch for advanced
  /// callers pairing custom tokenizers with custom ONNX).
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_dir<P: AsRef<Path>>(model_dir: P, opts: Options) -> Result<Self> {
    let dir: PathBuf = model_dir.as_ref().to_path_buf();

    // Apple-Silicon backend auto-selection: when the directory is an MLX
    // checkpoint (a `config.json` + `model.safetensors`, and NONE of the ORT
    // path's `onnx/*.onnx` graphs present) route to the MLX (`mlxrs`) Metal
    // backend. There is no user-facing knob — platform + checkpoint shape decide.
    // The ONNX-graph absence is the disambiguator: `config.json` +
    // `model.safetensors` are also the standard HuggingFace source-asset names,
    // and the ORT path reads the `onnx/*.onnx` graphs while the MLX path never
    // does, so a directory carrying those graphs is an ONNX checkpoint and falls
    // through to the ONNX path below. The MLX checkpoint's `config.json` is the
    // MLX-format config (NOT byte-equal to the bundled ONNX `config.json`), so
    // this branch MUST precede the ONNX-specific bundled drift validations, which
    // would otherwise reject it. Off macOS/arm64 this arm is cfg-compiled-out and
    // only the ONNX path exists.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if crate::runtime::mlx_backend::prefer_mlx(
      &dir,
      &["onnx/vision_encoder.onnx", "onnx/decoder_model_merged.onnx"],
    ) {
      return Self::from_mlx_dir(&dir, opts);
    }

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
    // state. Same theme as the chat_template drift check.
    validate_config_context_matches_bundled(&dir.join("config.json"))?;
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
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
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
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an MLX checkpoint
  /// directory (`config.json` + `model.safetensors` + `tokenizer.json`).
  ///
  /// Apple-Silicon only; on every other platform this constructor does not
  /// exist (the `mlxrs` dependency is macOS/arm64-only). It is normally reached
  /// via [`from_dir`](Self::from_dir)'s platform auto-routing rather than called
  /// directly. The engine still owns tokenization, the chat template, EOS
  /// handling, and the sampler — the MLX backend replaces only the model-weight
  /// stages (text embed, vision encode + splice, decoder forward). The
  /// directory's `tokenizer.json` is used (validated against the
  /// `expand_image_placeholders` special-token contract), and the model weights
  /// are loaded from `model.safetensors`.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub fn from_mlx_dir<P: AsRef<Path>>(model_dir: P, opts: Options) -> Result<Self> {
    let dir = model_dir.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_dir(dir)?;
    Self::assemble(
      BackendImpl::Mlx(Box::new(backend)),
      &dir.join("tokenizer.json"),
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact**
  /// `model.safetensors` file path (Apple Silicon only).
  ///
  /// The explicit-format counterpart to [`from_mlx_dir`](Self::from_mlx_dir): use
  /// it when you already know the checkpoint is an MLX safetensors file and where
  /// it lives. The `config.json` and the `tokenizer.json` are read from the
  /// weight file's **parent directory** (`weights.parent()`). There is no ONNX
  /// fallback — this constructor always builds the MLX backend.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
  pub fn from_mlx_safetensors<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_safetensors(weights)?;
    Self::assemble(
      BackendImpl::Mlx(Box::new(backend)),
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact** `*.npz`
  /// file path (Apple Silicon only). The `config.json` and the `tokenizer.json`
  /// are read from the weight file's **parent directory** (`weights.parent()`).
  ///
  /// Explicit-format MLX constructor (see
  /// [`from_mlx_safetensors`](Self::from_mlx_safetensors)); always builds the MLX
  /// backend, no ONNX fallback.
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "npz"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "npz")))
  )]
  pub fn from_mlx_npz<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_npz(weights)?;
    Self::assemble(
      BackendImpl::Mlx(Box::new(backend)),
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// Construct an MLX (`mlxrs`) Metal-backed engine from an **exact** `*.gguf`
  /// file path (Apple Silicon only). The `config.json` and the `tokenizer.json`
  /// are read from the weight file's **parent directory** (`weights.parent()`);
  /// the gguf's embedded metadata is NOT mapped to a config, so a sibling
  /// `config.json` is still required.
  ///
  /// Explicit-format MLX constructor (see
  /// [`from_mlx_safetensors`](Self::from_mlx_safetensors)); always builds the MLX
  /// backend, no ONNX fallback.
  #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "gguf"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(target_os = "macos", target_arch = "aarch64", feature = "gguf")))
  )]
  pub fn from_mlx_gguf<P: AsRef<Path>>(weights: P, opts: Options) -> Result<Self> {
    let weights = weights.as_ref();
    opts.image_budget().validate()?;
    let backend = crate::runtime::mlx_backend::MlxBackend::from_gguf(weights)?;
    Self::assemble(
      BackendImpl::Mlx(Box::new(backend)),
      &crate::runtime::mlx_backend::weights_parent(weights).join("tokenizer.json"),
      &opts,
    )
  }

  /// Shared engine assembly: load the tokenizer from `tokenizer_path`, validate
  /// its EOS + image special-token contract, and build the [`Engine`] around the
  /// already-constructed `backend`. Used by both the ONNX [`from_paths`] and the
  /// MLX [`from_mlx_dir`] paths so they share one tokenizer / EOS / parser-factory
  /// setup.
  ///
  /// The caller has already validated the image budget.
  fn assemble(backend: BackendImpl, tokenizer_path: &Path, opts: &Options) -> Result<Self> {
    let preproc = Preprocessor::new(*opts.image_budget());
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
    validate_image_tokenizer_contract(&tokenizer, opts.image_budget().max_tiles())?;

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
        .saturating_add(crate::generate::IMAGE_BLOCK_WRAPPER_TOKENS),
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
// Preprocessor-config drift detector ()
// =========================================================================

/// Validate the model's `preprocessor_config.json` matches our hardcoded
/// preprocessing-algorithm constants (patch_size, downsample_factor,
/// tile_size, image_mean, image_std). Drift between any of these and
/// the values our `flatten_to_patches` / `smart_resize` / `pick_tile_grid`
/// rely on would produce visually-wrong embeddings without an obvious
/// runtime error.
///
/// Budget-tunable fields (min/max image_tokens, min/max tiles,
/// max_pixels_tolerance, use_thumbnail) are deliberately NOT checked
/// here — callers can override them via `Options::image_budget()`.
///
/// Used only by `from_dir` (where the model directory has the config
/// alongside the ONNX files). `from_onnx_dir` uses bundled assets and
/// our own constants by construction, so no drift is possible.
#[cfg_attr(not(feature = "bundled"), allow(dead_code))]
fn validate_preprocessor_config(path: &Path) -> Result<()> {
  // fail closed on missing config. The
  // strict drift detector is the whole point of this check; allowing
  // its absence to skip validation defeats it. If a caller has a
  // stripped-down model directory without preprocessor_config.json,
  // they should use `from_paths` (which explicitly opts out) or
  // `from_onnx_dir` (which uses bundled assets).
  if !path.exists() {
    return Err(Error::InvalidRequest(
      "model directory missing preprocessor_config.json — use from_paths to bypass strict drift checks",
    ));
  }
  let raw = std::fs::read_to_string(path).map_err(Error::Io)?;
  let cfg: serde_json::Value = serde_json::from_str(&raw)
    .map_err(|e| Error::tokenizer(format!("preprocessor_config.json parse failure: {e}")))?;

  // Helpers
  let read_u64 = |key: &'static str| -> Result<u64> {
    cfg
      .get(key)
      .and_then(|v| v.as_u64())
      .ok_or(Error::InvalidRequest(
        "preprocessor_config.json missing required integer field — wrong model revision?",
      ))
  };
  let read_bool = |key: &'static str| -> Result<bool> {
    cfg
      .get(key)
      .and_then(|v| v.as_bool())
      .ok_or(Error::InvalidRequest(
        "preprocessor_config.json missing required boolean field — wrong model revision?",
      ))
  };
  let read_str = |key: &'static str| -> Result<&str> {
    cfg
      .get(key)
      .and_then(|v| v.as_str())
      .ok_or(Error::InvalidRequest(
        "preprocessor_config.json missing required string field — wrong model revision?",
      ))
  };
  let read_f64 = |key: &'static str| -> Result<f64> {
    cfg
      .get(key)
      .and_then(|v| v.as_f64())
      .ok_or(Error::InvalidRequest(
        "preprocessor_config.json missing required number field — wrong model revision?",
      ))
  };
  let read_f32_array3 = |key: &'static str| -> Result<[f32; 3]> {
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
  };

  // Round-20 baseline: model-fixed dimensional constants.
  if read_u64("encoder_patch_size")? != crate::preproc::tile_grid::PATCH_SIZE as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json encoder_patch_size != 16 (lfm crate hardcoded) — wrong model revision?",
    ));
  }
  if read_u64("downsample_factor")? != crate::preproc::tile_grid::DOWNSAMPLE_FACTOR as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json downsample_factor != 2 (lfm crate hardcoded) — wrong model revision?",
    ));
  }
  if read_u64("tile_size")? != crate::preproc::tile_grid::FULL_TILE_SIZE as u64 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json tile_size != 512 (lfm crate hardcoded) — wrong model revision?",
    ));
  }

  // also validate every preprocessing
  // semantic the Rust code hardcodes. Any of these flipped vs the
  // model's training-time config would produce wrong embeddings.
  for (key, expected) in [
    ("do_resize", true),
    ("do_rescale", true),
    ("do_normalize", true),
    ("do_pad", true),
    ("do_image_splitting", true),
  ] {
    if read_bool(key)? != expected {
      return Err(Error::InvalidRequest(
        "preprocessor_config.json boolean preprocessing flag differs from lfm crate hardcoded value — wrong model revision?",
      ));
    }
  }

  // data_format: must be channels_first. Our flatten_to_patches
  // produces (C, H, W) order — see PATCH_SIZE × PATCH_SIZE × 3 unfold.
  if read_str("data_format")? != "channels_first" {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json data_format != channels_first — wrong model revision?",
    ));
  }

  // resample: 2 = PIL BILINEAR. We use image::imageops::FilterType::Triangle
  // which is bilinear; matches.
  if read_u64("resample")? != 2 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json resample != 2 (BILINEAR) — wrong model revision?",
    ));
  }

  // rescale_factor: 1/255 = 0.003921568627... — our flatten_to_patches
  // computes (px / 255.0) * 2.0 - 1.0, where /255.0 IS the rescale.
  let rf = read_f64("rescale_factor")?;
  if (rf - (1.0 / 255.0)).abs() > 1e-9 {
    return Err(Error::InvalidRequest(
      "preprocessor_config.json rescale_factor != 1/255 — wrong model revision?",
    ));
  }

  // size = {height: 512, width: 512}. Mirrors tile_size but checked
  // separately because some HF processors use `size` independently.
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

  // Normalization: our flatten_to_patches does (px/255)*2 - 1, which
  // is equivalent to (px/255 - 0.5) / 0.5 = subtract 0.5 then divide
  // by 0.5. So image_mean and image_std must both be [0.5, 0.5, 0.5].
  for (key, expected) in [("image_mean", [0.5f32; 3]), ("image_std", [0.5f32; 3])] {
    let got = read_f32_array3(key)?;
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

// =========================================================================
// Tokenizer-bytes drift detector (bundled feature only)
// =========================================================================

/// Verify the supplied `tokenizer.json` byte-matches the bundled blob.
/// a tokenizer with the same special-token IDs
/// but a drifted normal vocabulary would pass the per-token contract
/// check yet still encode normal text into different IDs that don't
/// match the embedding table — silent global prompt corruption.
///
/// Called from `from_dir` (strict constructor) only; `from_paths`
/// remains the explicit escape hatch for callers intentionally
/// pairing custom tokenizers with custom ONNX.
/// Validate the model's `config.json` exposes a
/// `text_config.max_position_embeddings` (or top-level
/// `max_position_embeddings`) that matches our hard-coded
/// [`crate::options::MODEL_CONTEXT_TOKENS`].
/// finding 1: generate()'s admission gates trust this constant; a
/// model exported with a smaller positional embedding range would
/// pass byte-identical tokenizer/template/preprocessor checks
/// (the static assets) but quietly accept prompts past its real
/// limit and either fail late or produce invalid position state.
#[cfg(feature = "bundled")]
fn validate_config_context_matches_bundled(path: &Path) -> Result<()> {
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
  if supplied != crate::bundled::CHAT_TEMPLATE_JINJA {
    return Err(Error::InvalidRequest(
      "supplied chat_template.jinja bytes do not match the bundled chat template — engine renders with bundled template; mismatched model template would produce semantically wrong prompts even when <image> counts line up",
    ));
  }
  Ok(())
}

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
#[cfg(feature = "bundled")]
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
#[cfg(feature = "bundled")]
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
    // Exercise via the bundled tokenizer (always available under the
    // `bundled` feature, which gates this whole test file via the
    // `inference + decoders` mod gate that engine.rs lives under).
    #[cfg(feature = "bundled")]
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
  fn validate_config_context_matches_bundled_accepts_correct_and_rejects_drift() {
    // from_dir's strict drift check for
    // text_config.max_position_embeddings vs MODEL_CONTEXT_TOKENS.
    let dir = std::env::temp_dir().join(format!("lfm-test-config-drift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Missing → reject.
    let missing = dir.join("config-missing.json");
    let _ = std::fs::remove_file(&missing);
    assert!(matches!(
      validate_config_context_matches_bundled(&missing),
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
      validate_config_context_matches_bundled(&drift),
      Err(Error::InvalidRequest(_))
    ));

    // Correct nested layout → ok.
    let ok_nested = dir.join("config-ok-nested.json");
    std::fs::write(
      &ok_nested,
      r#"{"text_config":{"max_position_embeddings":128000}}"#,
    )
    .unwrap();
    assert!(validate_config_context_matches_bundled(&ok_nested).is_ok());

    // Correct top-level layout → ok (older single-modality configs).
    let ok_flat = dir.join("config-ok-flat.json");
    std::fs::write(&ok_flat, r#"{"max_position_embeddings":128000}"#).unwrap();
    assert!(validate_config_context_matches_bundled(&ok_flat).is_ok());

    // Bundled config.json → ok.
    let ok_bundled = dir.join("config-ok-bundled.json");
    std::fs::write(&ok_bundled, crate::bundled::CONFIG_JSON).unwrap();
    assert!(validate_config_context_matches_bundled(&ok_bundled).is_ok());

    // Invalid JSON → reject.
    let bad_json = dir.join("config-bad.json");
    std::fs::write(&bad_json, b"{not json").unwrap();
    assert!(matches!(
      validate_config_context_matches_bundled(&bad_json),
      Err(Error::InvalidRequest(_))
    ));
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
}
