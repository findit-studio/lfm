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
// `ImageAnalysis`, `ImageAnalysisTask`, `Extension` and `UnknownExtension`
// live in `llmtask` as the canonical cross-engine implementation (prompt,
// JSON Schema, parser); lfm carries no copy of its own. They are re-exported
// here so a consumer with no direct `llmtask` dependency (mediagraph mounts
// the roster through lfm) can name all four. Since llmtask 0.4 the task is
// built from a field roster: `ImageAnalysisTask::new()` asks for
// `description` and `tags` only, each other field is an `Extension` switched
// on with `with_extensions`, and `parse` holds the answer to the task's
// schema. Since llmtask 0.4.1 an `Extension` is named by its field's JSON key
// (`FromStr`, `TryFrom<&str>`, `Display`, and serde with the `serde`
// feature), and any other name is refused as `UnknownExtension` (see
// CHANGELOG). Since llmtask 0.5 an analysis says how its description ends
// (`DescriptionEnd`), settled by the decoder's account of each string field
// (`FieldEnd`, `FieldEnds`) that `Engine::run` hands `Task::parse_ended`.
pub use llmtask::{
  DescriptionEnd, ImageAnalysis,
  image_analysis::{Extension, ImageAnalysisTask, UnknownExtension},
};
pub use task::{FieldEnd, FieldEnds, JsonParseError, Task};

#[cfg(test)]
mod tests {
  use crate::{Extension, ImageAnalysisTask, JsonParseError, Task, UnknownExtension};

  /// The `required` list of `task`'s JSON Schema, in order.
  fn required(task: &ImageAnalysisTask) -> Vec<&str> {
    task.schema()["required"]
      .as_array()
      .expect("the schema's `required` is an array")
      .iter()
      .map(|name| name.as_str().expect("a required field name is a string"))
      .collect()
  }

  /// The keys of `task`'s schema `properties`, sorted.
  fn properties(task: &ImageAnalysisTask) -> Vec<&str> {
    let mut keys: Vec<&str> = task.schema()["properties"]
      .as_object()
      .expect("the schema's `properties` is an object")
      .keys()
      .map(String::as_str)
      .collect();
    keys.sort_unstable();
    keys
  }

  /// LAW: through lfm's own paths (`lfm::ImageAnalysisTask`,
  /// `lfm::Extension`, `lfm::JsonParseError`), the default task asks for
  /// exactly `description` and `tags`, both capped, and refuses an answer
  /// that carries a field it did not ask for; `with_extensions(Extension::ALL)`
  /// restores the ten-field schema.
  #[test]
  fn the_reexported_task_asks_for_its_roster() {
    let task = ImageAnalysisTask::new();
    assert_eq!(required(&task), ["description", "tags"]);
    assert_eq!(properties(&task), ["description", "tags"]);
    assert_eq!(task.schema()["additionalProperties"], false);
    assert_eq!(
      task.schema()["properties"]["description"]["maxLength"],
      ImageAnalysisTask::DEFAULT_DESCRIPTION_MAX_CHARS.get()
    );
    assert_eq!(
      task.schema()["properties"]["tags"]["maxItems"],
      ImageAnalysisTask::DEFAULT_TAGS_MAX_ITEMS.get()
    );
    for extension in Extension::ALL {
      assert!(
        !task.has_extension(extension),
        "{extension:?} must be off by default"
      );
    }
    assert_eq!(ImageAnalysisTask::default().schema(), task.schema());

    let analysis = task
      .parse(r#"{"description":"A person reads by a window.","tags":["reading"]}"#)
      .expect("the default task's own two fields parse");
    assert_eq!(analysis.description(), "A person reads by a window.");
    assert!(
      analysis.scene().is_empty(),
      "a field not asked for reads empty"
    );
    match task.parse(r#"{"description":"A person reads.","tags":["reading"],"scene":"library"}"#) {
      Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, ["scene"]),
      other => panic!("a field the task did not ask for must be UnknownFields, got {other:?}"),
    }

    let full = ImageAnalysisTask::new().with_extensions(Extension::ALL);
    let ten = [
      "scene",
      "description",
      "subjects",
      "objects",
      "actions",
      "emotion",
      "shot_type",
      "lighting",
      "tags",
      "categories",
    ];
    assert_eq!(required(&full), ten);
    let mut sorted = ten;
    sorted.sort_unstable();
    assert_eq!(properties(&full), sorted);
  }

  /// LAW: through lfm's paths, an extension is named by its field's JSON
  /// key. `"shot_type"` reads as `Extension::ShotType` through `FromStr` and
  /// `TryFrom<&str>`, and `Display` writes it back; every name in
  /// `Extension::NAMES` reads back, and the eight together ask for all ten
  /// fields. `"tags"`, which every task asks for and which is not an
  /// extension, is refused as `lfm::UnknownExtension` carrying the name.
  #[test]
  fn an_extension_is_named_by_its_json_key() {
    assert_eq!("shot_type".parse::<Extension>(), Ok(Extension::ShotType));
    assert_eq!(Extension::try_from("shot_type"), Ok(Extension::ShotType));
    assert_eq!(Extension::ShotType.to_string(), "shot_type");

    let refused: UnknownExtension = "tags"
      .parse::<Extension>()
      .expect_err("`tags` is not an extension");
    assert_eq!(refused.name(), "tags");
    assert_eq!(Extension::try_from("tags"), Err(refused));

    let named = Extension::NAMES.map(|name| {
      name
        .parse::<Extension>()
        .unwrap_or_else(|err| panic!("{name} must read back: {err}"))
    });
    assert_eq!(named, Extension::ALL);
    let full = ImageAnalysisTask::new().with_extensions(named);
    assert_eq!(required(&full).len(), 10);
  }
}
