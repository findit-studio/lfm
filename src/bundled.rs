//! Bundled tokenizer + small JSON configs, embedded at compile time when
//! the `bundled` feature is enabled. ONNX model files are NOT bundled
//! (vision_encoder ~86 MB, decoder ~350 MB; can't fit under crates.io's
//! 10 MB include limit). Users must supply the ONNX directory separately.
//!
//! Total bundled size budget: <10 MB (crates.io hard limit). The actual
//! payload is ~4.5 MB (dominated by `tokenizer.json` at ~4.5 MB).

#![cfg(feature = "bundled")]

/// Bundled `tokenizer.json` bytes (~4.5 MB).
///
/// LFM2.5-VL-450M tokenizer: Qwen2-style, 151 665-token vocabulary.
/// Consumed by [`crate::engine::Engine::from_onnx_dir`] via
/// `write_bundled_tokenizer()` at runtime, and by both strict roads'
/// checkpoint-identity validation (byte comparison) at load time.
///
/// `backend_available` (build.rs): every reader of this constant is reachable
/// only through a backend-specific constructor (the ONNX bundled door needs
/// `ort_backend`, the MLX strict-identity check needs aarch64-apple-darwin), so on a
/// target with neither — e.g. `x86_64-apple-darwin` with the `inference`
/// feature on — nothing reads it.
#[cfg(backend_available)]
pub(crate) const TOKENIZER_JSON: &[u8] = include_bytes!("../models/tokenizer.json");

/// Bundled `tokenizer_config.json` bytes.
///
/// Provided for downstream users that inspect model metadata at runtime.
#[allow(dead_code)]
pub(crate) const TOKENIZER_CONFIG_JSON: &[u8] = include_bytes!("../models/tokenizer_config.json");

/// Bundled `preprocessor_config.json` bytes.
///
/// Provided for downstream users that inspect model metadata at runtime.
#[allow(dead_code)]
pub(crate) const PREPROCESSOR_CONFIG_JSON: &[u8] =
  include_bytes!("../models/preprocessor_config.json");

/// Bundled `processor_config.json` bytes.
///
/// Provided for downstream users that inspect model metadata at runtime.
#[allow(dead_code)]
pub(crate) const PROCESSOR_CONFIG_JSON: &[u8] = include_bytes!("../models/processor_config.json");

/// Bundled `generation_config.json` bytes.
///
/// Provided for downstream users that inspect model metadata at runtime.
#[allow(dead_code)]
pub(crate) const GENERATION_CONFIG_JSON: &[u8] = include_bytes!("../models/generation_config.json");

/// Bundled `config.json` bytes.
///
/// Provided for downstream users that inspect model metadata at runtime.
#[allow(dead_code)]
pub(crate) const CONFIG_JSON: &[u8] = include_bytes!("../models/config.json");

/// Bundled `chat_template.jinja` bytes.
///
/// Used by `from_dir`'s strict prompt-contract drift check: a
/// model directory that ships a different chat template would
/// otherwise be accepted while the engine renders with the bundled
/// jinja, producing silent prompt-envelope drift. Only used for
/// the byte-equal cross-check; the actual runtime template is
/// `chat_template::BUNDLED_CHAT_TEMPLATE_JINJA`.
#[allow(dead_code)]
pub(crate) const CHAT_TEMPLATE_JINJA: &[u8] = include_bytes!("../models/chat_template.jinja");
