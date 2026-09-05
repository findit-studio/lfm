//! Rust ONNX/MLX inference for LiquidAI LFM2.5-VL (vision-language) models.
//! ONNX (`ort`) is the default backend on the targets `ort-sys` ships
//! prebuilt binaries for that this crate supports (Linux x86_64/aarch64-gnu,
//! Windows x86_64/aarch64-msvc); on aarch64-apple-darwin, MLX (`mlxrs`) is the
//! always-compiled native road and `ort` becomes opt-in behind the `ort`
//! feature; every other target builds with no ONNX backend (see
//! `Cargo.toml`).
//!
//! See `docs/superpowers/specs/2026-05-03-lfm-vlm-wrapper-design.md`
//! for the full design rationale.
//!
//! ## Model weights license
//!
//! This crate is dual-licensed under MIT OR Apache-2.0. **The model
//! weights it wraps are NOT** — LFM2.5-VL-450M ships under the LFM
//! Open License v1.0 (`lfm1.0`, see <https://www.liquid.ai/lfm-license>).
//! Verify your use case complies with Liquid AI's terms separately
//! from this crate's license.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

#[cfg(feature = "bundled")]
pub(crate) mod bundled;
pub mod chat_template;
// Engine depends on the generate module, which itself requires the
// `decoders` feature for image decoding. Gate Engine on the same set
// to prevent a non-buildable `--features inference` config.
#[cfg(all(feature = "inference", feature = "decoders"))]
mod engine;
pub mod error;
#[cfg(all(feature = "inference", feature = "decoders"))]
pub(crate) mod generate;
pub mod options;
pub mod preproc;
#[cfg(feature = "inference")]
pub(crate) mod runtime;
mod task;

pub use chat_template::{
  BOS, BOS_TOKEN_ID, EOS_TOKEN_ID, IM_END, IM_START, IMAGE_END, IMAGE_START, IMAGE_THUMBNAIL,
  IMAGE_TOKEN, IMAGE_TOKEN_ID, ImagePlaceholderInfo, PAD, PAD_TOKEN_ID, TOOL_CALL_END,
  TOOL_CALL_START, expand_image_placeholders,
};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use chat_template::{
  BUNDLED_CHAT_TEMPLATE_JINJA, ContentItem, Message, UserContent, apply_chat_template,
};
#[cfg(all(feature = "inference", feature = "decoders"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "inference", feature = "decoders"))))]
pub use engine::{Engine, EnginePaths};
pub use error::{Error, Result};
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
pub use options::GraphOptimizationLevel;
#[cfg(all(feature = "inference", mlx_backend))]
#[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
pub use options::MlxOptions;
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
pub use options::OrtOptions;
pub use options::{
  AutoOptions, BackendKind, BackendOptions, ImageBudget, Options, RequestOptions, ThreadOptions,
};
#[cfg(feature = "decoders")]
#[cfg_attr(docsrs, doc(cfg(feature = "decoders")))]
pub use preproc::decode_bytes_with_orientation;
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
#[cfg_attr(
  docsrs,
  doc(cfg(all(feature = "decoders", not(target_arch = "wasm32"))))
)]
pub use preproc::decode_with_orientation;
pub use preproc::{
  IMAGE_BLOCK_WRAPPER_TOKENS, ImagePlan, PreprocessedImage, Preprocessor, TileGrid,
};

// ===== Public chat types (Engine / Task 13 API) =====

/// One message in a multi-turn conversation.
///
/// The `role` field accepts `"system"`, `"user"`, and `"assistant"`.
/// `content` is either plain text or a mixed sequence of text + image
/// references (for multimodal user messages).
#[derive(Debug, Clone)]
pub struct ChatMessage {
  /// Role of the message author: `"system"`, `"user"`, or `"assistant"`.
  role: smol_str::SmolStr,
  /// Message content: plain text or multimodal parts.
  content: ChatContent,
}

impl ChatMessage {
  /// Construct a new `ChatMessage` with the given role and content.
  pub fn new(role: smol_str::SmolStr, content: ChatContent) -> Self {
    Self { role, content }
  }

  /// Construct a text-only message (convenience constructor).
  pub fn text(role: smol_str::SmolStr, text: impl Into<String>) -> Self {
    Self {
      role,
      content: ChatContent::Text(text.into()),
    }
  }

  /// Construct a multimodal message (convenience constructor).
  pub fn parts(role: smol_str::SmolStr, parts: Vec<ContentPart>) -> Self {
    Self {
      role,
      content: ChatContent::Parts(parts),
    }
  }

  /// Role of the message author: `"system"`, `"user"`, or `"assistant"`.
  pub fn role(&self) -> &smol_str::SmolStr {
    &self.role
  }

  /// Message content: plain text or multimodal parts.
  pub fn content(&self) -> &ChatContent {
    &self.content
  }

  /// Set the role.
  pub fn set_role(&mut self, role: smol_str::SmolStr) {
    self.role = role;
  }

  /// Set the content.
  pub fn set_content(&mut self, content: ChatContent) {
    self.content = content;
  }

  /// Builder: set the role (chainable).
  pub fn with_role(mut self, role: smol_str::SmolStr) -> Self {
    self.role = role;
    self
  }

  /// Builder: set the content (chainable).
  pub fn with_content(mut self, content: ChatContent) -> Self {
    self.content = content;
    self
  }
}

/// Content payload of a [`ChatMessage`].
#[derive(Debug, Clone)]
pub enum ChatContent {
  /// Plain text content.
  Text(String),
  /// Mixed text + image parts (for multimodal user messages).
  /// Parts are processed in order; each [`ContentPart::Image`] refers
  /// to the next image in `GenerateInputs::images` (by position across
  /// the entire message list, not per-message).
  Parts(Vec<ContentPart>),
}

/// One part inside a [`ChatContent::Parts`] multimodal message.
#[derive(Debug, Clone)]
pub enum ContentPart {
  /// A text fragment.
  Text(String),
  /// An image reference. The N-th `ContentPart::Image` across all
  /// messages corresponds to `GenerateInputs::images[N]`.
  Image,
}

/// An image supplied to `generate`: either a file path or raw bytes.
#[derive(Debug, Clone, Copy)]
pub enum ImageInput<'a> {
  /// Path to an image file on disk (EXIF orientation is applied).
  #[cfg(not(target_arch = "wasm32"))]
  Path(&'a std::path::Path),
  /// Raw encoded image bytes (EXIF orientation is applied).
  Bytes(&'a [u8]),
}

// ===== Task + image-analysis exports =====
//
// `ImageAnalysis` and `ImageAnalysisTask` moved to `llmtask` (0.3+) as the
// canonical cross-engine implementation (prompt, JSON Schema, parser); lfm no
// longer carries its own copy. Re-exported here so `lfm::ImageAnalysis` /
// `lfm::ImageAnalysisTask` stay valid import paths for existing callers —
// only the underlying field shape changed (`mood` → `emotion`, plus a new
// required `categories` field; see CHANGELOG).
pub use llmtask::{ImageAnalysis, image_analysis::ImageAnalysisTask};
pub use task::{JsonParseError, Task};
