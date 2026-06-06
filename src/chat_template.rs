//! Chat-template rendering for LFM2.5-VL.
//!
//! Two-step pipeline:
//! 1. [`apply_chat_template`] (cfg=inference) renders the upstream Jinja
//!    template via `minijinja`, producing a chat-formatted prompt with
//!    literal `<image>` placeholders (one per image content item).
//! 2. [`expand_image_placeholders`] walks those placeholders and
//!    substitutes the per-image structure (per spec §8.4):
//!    `<|image_start|>` + per-tile `<|img_row_R_col_C|>` +
//!    `<image>` × tokens_per_tile + optional thumbnail + `<|image_end|>`.
//!
//! Special token strings + IDs come from the tokenizer
//! (see `tests/fixtures/onnx_io_contract.json` for the cross-reference).

// ===== Special token constants =====

/// Beginning-of-text token (id 1).
pub const BOS: &str = "<|startoftext|>";
/// Chat-message start marker.
pub const IM_START: &str = "<|im_start|>";
/// Chat-message end marker (also used as EOS, id 7).
pub const IM_END: &str = "<|im_end|>";
/// Padding token (id 0).
pub const PAD: &str = "<|pad|>";
/// Image placeholder; expanded into the per-image structure (id 396).
pub const IMAGE_TOKEN: &str = "<image>";
/// Per-image start marker (wraps the per-tile expansion).
pub const IMAGE_START: &str = "<|image_start|>";
/// Per-image end marker.
pub const IMAGE_END: &str = "<|image_end|>";
/// Marks the thumbnail block within a multi-tile image expansion.
pub const IMAGE_THUMBNAIL: &str = "<|img_thumbnail|>";
/// Tool-call start marker (function-calling envelope).
pub const TOOL_CALL_START: &str = "<|tool_call_start|>";
/// Tool-call end marker.
pub const TOOL_CALL_END: &str = "<|tool_call_end|>";

/// `<|startoftext|>` token id.
pub const BOS_TOKEN_ID: u32 = 1;
/// `<|im_start|>` token id.
pub const IM_START_TOKEN_ID: u32 = 6;
/// `<|im_end|>` token id (also EOS).
pub const EOS_TOKEN_ID: u32 = 7;
/// `<|pad|>` token id.
pub const PAD_TOKEN_ID: u32 = 0;
/// `<|tool_call_start|>` token id.
pub const TOOL_CALL_START_TOKEN_ID: u32 = 10;
/// `<|tool_call_end|>` token id.
pub const TOOL_CALL_END_TOKEN_ID: u32 = 11;
/// `<image>` token id.
pub const IMAGE_TOKEN_ID: u32 = 396;
/// `<|img_thumbnail|>` token id.
pub const IMAGE_THUMBNAIL_TOKEN_ID: u32 = 497;
/// `<|image_start|>` token id.
pub const IMAGE_START_TOKEN_ID: u32 = 498;
/// `<|image_end|>` token id.
pub const IMAGE_END_TOKEN_ID: u32 = 499;
/// Base id for the `<|img_row_R_col_C|>` 10×10 grid. The full token
/// id is `IMG_ROW_COL_BASE_ID + (R-1) * 10 + (C-1)` for R, C ∈ [1, 10],
/// yielding ids 397..=496.
pub const IMG_ROW_COL_BASE_ID: u32 = 397;

/// Bundled Jinja source. Useful as a parity-test fixture and for downstream
/// callers who want to verify the template they're rendering against.
///
/// Gated on `feature = "inference"` so the wasm-friendly preprocessing
/// subset doesn't pay the ~3.8 KB string in its binary.
#[cfg(feature = "inference")]
pub const BUNDLED_CHAT_TEMPLATE_JINJA: &str = include_str!("../models/chat_template.jinja");

/// Walk a chat-formatted prompt and expand each literal `<image>`
/// placeholder into the per-image structure. The `Engine` calls this
/// AFTER `apply_chat_template` and BEFORE tokenization.
///
/// **Multi-tile path** (rows > 1 OR cols > 1):
/// `<|image_start|>` + (per-tile `<|img_row_{R+1}_col_{C+1}|>` + `<image>` × tokens_per_main_tile)
/// + (optional `<|img_thumbnail|>` + `<image>` × thumbnail_tokens) + `<|image_end|>`
///
/// **Single-tile path** (1×1 grid, no thumbnail):
/// `<|image_start|>` + `<image>` × num_image_tokens + `<|image_end|>`
///
/// Returns `Error::ImageTokenCountMismatch` if the count of `<image>`
/// placeholders in `prompt` doesn't match `images.len()`.
pub fn expand_image_placeholders(
  prompt: &str,
  images: &[ImagePlaceholderInfo],
) -> crate::error::Result<String> {
  let pieces: Vec<&str> = prompt.split(IMAGE_TOKEN).collect();
  let placeholder_count = pieces.len() - 1;
  if placeholder_count != images.len() {
    return Err(crate::error::Error::ImageTokenCountMismatch {
      expected: images.len(),
      got: placeholder_count,
    });
  }
  // 8 KB per image covers 2×2 multi-tile + thumbnail (~7.7 KB) without
  // reallocating; common production case stays single-allocation.
  let mut out = String::with_capacity(prompt.len() + 8 * 1024 * images.len());
  for (i, piece) in pieces.iter().enumerate() {
    out.push_str(piece);
    if i < images.len() {
      build_image_block(&mut out, &images[i]);
    }
  }
  Ok(out)
}

/// The grid-layout slice that `expand_image_placeholders` needs per image.
/// Decoupled from `PreprocessedImage` so this function compiles without
/// the `inference` feature (preproc/mod.rs gates on inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePlaceholderInfo {
  /// Main tile-grid rows (1 in single-tile path).
  rows: usize,
  /// Main tile-grid cols (1 in single-tile path).
  cols: usize,
  /// Tokens per main tile (256 in multi-tile, dynamic in single-tile).
  tokens_per_main_tile: usize,
  /// Tokens for the thumbnail tile (None when no thumbnail).
  thumbnail_tokens: Option<usize>,
}

impl ImagePlaceholderInfo {
  /// Construct a new `ImagePlaceholderInfo`.
  pub const fn new(
    rows: usize,
    cols: usize,
    tokens_per_main_tile: usize,
    thumbnail_tokens: Option<usize>,
  ) -> Self {
    Self {
      rows,
      cols,
      tokens_per_main_tile,
      thumbnail_tokens,
    }
  }

  /// Main tile-grid rows (1 in single-tile path).
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// Main tile-grid cols (1 in single-tile path).
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// Tokens per main tile (256 in multi-tile, dynamic in single-tile).
  pub const fn tokens_per_main_tile(&self) -> usize {
    self.tokens_per_main_tile
  }

  /// Tokens for the thumbnail tile (None when no thumbnail).
  pub const fn thumbnail_tokens(&self) -> Option<usize> {
    self.thumbnail_tokens
  }

  /// Set rows.
  pub fn set_rows(&mut self, rows: usize) {
    self.rows = rows;
  }

  /// Set cols.
  pub fn set_cols(&mut self, cols: usize) {
    self.cols = cols;
  }

  /// Set tokens per main tile.
  pub fn set_tokens_per_main_tile(&mut self, tokens_per_main_tile: usize) {
    self.tokens_per_main_tile = tokens_per_main_tile;
  }

  /// Set thumbnail tokens.
  pub fn set_thumbnail_tokens(&mut self, thumbnail_tokens: Option<usize>) {
    self.thumbnail_tokens = thumbnail_tokens;
  }

  /// Builder: set rows (chainable).
  pub const fn with_rows(mut self, rows: usize) -> Self {
    self.rows = rows;
    self
  }

  /// Builder: set cols (chainable).
  pub const fn with_cols(mut self, cols: usize) -> Self {
    self.cols = cols;
    self
  }

  /// Builder: set tokens per main tile (chainable).
  pub const fn with_tokens_per_main_tile(mut self, tokens_per_main_tile: usize) -> Self {
    self.tokens_per_main_tile = tokens_per_main_tile;
    self
  }

  /// Builder: set thumbnail tokens (chainable).
  pub const fn with_thumbnail_tokens(mut self, thumbnail_tokens: Option<usize>) -> Self {
    self.thumbnail_tokens = thumbnail_tokens;
    self
  }

  /// Total `<image>` tokens this image expands to.
  pub const fn num_image_tokens(&self) -> usize {
    self.rows * self.cols * self.tokens_per_main_tile
      + match self.thumbnail_tokens {
        Some(n) => n,
        None => 0,
      }
  }
}

fn build_image_block(out: &mut String, img: &ImagePlaceholderInfo) {
  out.push_str(IMAGE_START);
  if img.rows() > 1 || img.cols() > 1 {
    // Upstream marker emission: `for row in range(num_rows): for col in
    // range(num_cols)` where `num_rows = grid_width = ratio[0]` and
    // `num_cols = grid_height = ratio[1]` (see
    // transformers/models/lfm2_vl/processing_lfm2_vl.py:200-205 and
    // image_processing_lfm2_vl_fast.py:397 unpacking
    // `images, num_rows, num_cols = crop_image_to_patches(...)`).
    //
    // Our `TileGrid::cols` stores `grid_width` (width-direction tiles)
    // and `TileGrid::rows` stores `grid_height` (height-direction
    // tiles), so to match upstream's marker order we iterate the OUTER
    // loop over `cols` (= upstream's num_rows = grid_width) and the
    // INNER loop over `rows` (= upstream's num_cols = grid_height).
    //
    // Without this swap, a 1920×1080 image would emit a 2×4 marker
    // sequence while the model was trained against a 4×2 sequence —
    // wrong position-token embeddings on every non-square multi-tile
    // image, and `ImageTokenCountMismatch` would NOT detect it because
    // the tile count is identical.
    for outer in 0..img.cols() {
      for inner in 0..img.rows() {
        out.push_str("<|img_row_");
        push_usize(out, outer + 1);
        out.push_str("_col_");
        push_usize(out, inner + 1);
        out.push_str("|>");
        for _ in 0..img.tokens_per_main_tile() {
          out.push_str(IMAGE_TOKEN);
        }
      }
    }
    if let Some(thumb) = img.thumbnail_tokens() {
      out.push_str(IMAGE_THUMBNAIL);
      for _ in 0..thumb {
        out.push_str(IMAGE_TOKEN);
      }
    }
  } else {
    let total = img.num_image_tokens();
    for _ in 0..total {
      out.push_str(IMAGE_TOKEN);
    }
  }
  out.push_str(IMAGE_END);
}

fn push_usize(out: &mut String, n: usize) {
  use std::fmt::Write as _;
  let _ = write!(out, "{n}");
}

// ===== apply_chat_template (cfg=inference) =====

#[cfg(feature = "inference")]
mod render {
  use super::*;
  use serde::Serialize;
  use std::sync::OnceLock;

  /// Bundled template with `{%- generation -%}` / `{%- endgeneration -%}`
  /// tags stripped (minijinja doesn't recognize them; tokenizers-extension
  /// only). Computed once and cached for the process lifetime.
  fn stripped_template() -> &'static str {
    static CELL: OnceLock<String> = OnceLock::new();
    CELL.get_or_init(|| {
      BUNDLED_CHAT_TEMPLATE_JINJA
        .replace("{%- generation -%}", "")
        .replace("{%- endgeneration -%}", "")
    })
  }

  /// Top-level chat-template entry point. Renders the bundled Jinja
  /// template via `minijinja`. Returns the chat-formatted prompt with
  /// literal `<image>` placeholders for image content items.
  ///
  /// Callers handle image expansion separately via
  /// [`super::expand_image_placeholders`] once they know the per-image
  /// grid layout.
  ///
  /// The `{%- generation -%}` / `{%- endgeneration -%}` Jinja tags from
  /// the upstream tokenizers extension are stripped before rendering —
  /// minijinja doesn't recognize them, and our use case (one-shot
  /// rendering for prompt construction) doesn't need the
  /// generation-region tracking they enable.
  pub fn apply_chat_template(
    messages: &[Message<'_>],
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
  ) -> crate::error::Result<String> {
    use minijinja::{Environment, Value};

    let mut env = Environment::new();
    env.add_function(
      "strftime_now",
      |_fmt: String| -> std::result::Result<String, minijinja::Error> { Ok(today_yyyymmdd()) },
    );
    let tmpl = env
      .template_from_str(stripped_template())
      .map_err(crate::error::Error::tokenizer)?;

    // Canonicalize the tools object to sorted-key order before rendering. The
    // template emits `tools | tojson`, whose key order follows the
    // `serde_json::Value` object's backing map. That map is a sorted `BTreeMap`
    // by default but becomes an insertion-order `IndexMap` whenever ANY crate in
    // the dependency graph enables `serde_json`'s `preserve_order` feature
    // (Cargo unifies features globally — e.g. the macOS/arm64 `mlxrs` backend
    // dependency pulls it in). Rebuilding the value with keys inserted in sorted
    // order pins the rendered tool JSON to the same deterministic sorted output
    // on every platform regardless of that feature, so the prompt the model sees
    // does not silently depend on which other crates are linked.
    let canonical_tools = tools.map(canonicalize_json_keys);
    let ctx = Value::from_serialize(&RenderContext {
      bos_token: BOS,
      messages,
      tools: canonical_tools.as_ref(),
      add_generation_prompt,
    });
    tmpl.render(ctx).map_err(crate::error::Error::tokenizer)
  }

  /// Recursively rebuild a [`serde_json::Value`] with every object's keys
  /// inserted in sorted (lexicographic) order, so the serialized key order is
  /// deterministic regardless of whether `serde_json`'s `preserve_order` feature
  /// is active in the dependency graph (see [`apply_chat_template`]). Arrays
  /// recurse element-wise; scalars are returned unchanged.
  fn canonicalize_json_keys(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
      Value::Object(map) => {
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();
        let mut out = serde_json::Map::new();
        for k in keys {
          out.insert(k.clone(), canonicalize_json_keys(&map[k]));
        }
        Value::Object(out)
      }
      Value::Array(items) => Value::Array(items.iter().map(canonicalize_json_keys).collect()),
      other => other.clone(),
    }
  }

  #[derive(Serialize)]
  struct RenderContext<'a> {
    bos_token: &'a str,
    messages: &'a [Message<'a>],
    tools: Option<&'a serde_json::Value>,
    add_generation_prompt: bool,
  }

  /// YYYY-MM-DD using a tiny no-chrono date routine.
  /// Exposed under `#[cfg(test)]` so tests can resolve the `__DATE__`
  /// placeholder in fixtures without duplicating the date logic.
  #[cfg(test)]
  pub(super) fn today_yyyymmdd_for_test() -> String {
    today_yyyymmdd()
  }

  fn today_yyyymmdd() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs();
    let days = (secs / 86400) as i64;
    // Howard Hinnant's algo: days since 1970-01-01 → (Y, M, D).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y_base = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y_base + 1 } else { y_base };
    format!("{y:04}-{m:02}-{d:02}")
  }
}

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use render::apply_chat_template;

/// One message in a chat-template render call.
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message<'a> {
  /// System prompt.
  System {
    /// Plain-text system prompt content.
    content: &'a str,
  },
  /// User message — text-only or multimodal (image + text).
  User {
    /// User message content.
    content: UserContent<'a>,
  },
  /// Assistant response.
  Assistant {
    /// Plain-text assistant response.
    content: &'a str,
    /// Optional inline thinking block.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<&'a str>,
  },
}

/// User-message content variants.
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum UserContent<'a> {
  /// Plain-text user message.
  Text(&'a str),
  /// Multimodal content (image + text items, in order).
  Multimodal(Vec<ContentItem<'a>>),
}

/// One item inside a multimodal user message.
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentItem<'a> {
  /// An image placeholder. Rendered as the literal `<image>` token in
  /// the chat-formatted output; expanded later by
  /// [`expand_image_placeholders`].
  Image,
  /// A text item.
  Text {
    /// Text content.
    text: &'a str,
  },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn special_token_ids_match_tokenizer_json() {
    use serde_json::Value;
    let tok_raw = include_str!("../models/tokenizer.json");
    let tok: Value = serde_json::from_str(tok_raw).expect("tokenizer.json must be valid JSON");
    let added: &Vec<Value> = tok["added_tokens"].as_array().expect("added_tokens array");
    // Build a map of "token string" -> id for the special tokens we care about.
    let mut found: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for entry in added {
      if let (Some(id), Some(content)) = (entry["id"].as_u64(), entry["content"].as_str()) {
        found.insert(content.to_string(), id);
      }
    }
    // Verify our 4 ID constants match the tokenizer's actual IDs.
    assert_eq!(
      found.get(BOS).copied(),
      Some(BOS_TOKEN_ID as u64),
      "BOS_TOKEN_ID"
    );
    assert_eq!(
      found.get(IM_END).copied(),
      Some(EOS_TOKEN_ID as u64),
      "EOS_TOKEN_ID = {}",
      IM_END
    );
    assert_eq!(
      found.get(PAD).copied(),
      Some(PAD_TOKEN_ID as u64),
      "PAD_TOKEN_ID"
    );
    assert_eq!(
      found.get(IMAGE_TOKEN).copied(),
      Some(IMAGE_TOKEN_ID as u64),
      "IMAGE_TOKEN_ID"
    );
  }

  #[test]
  fn expand_count_mismatch() {
    let r = expand_image_placeholders("Hello <image>", &[]);
    assert!(matches!(
      r,
      Err(crate::error::Error::ImageTokenCountMismatch {
        expected: 0,
        got: 1
      })
    ));
  }

  #[test]
  fn expand_single_tile() {
    let info = ImagePlaceholderInfo::new(1, 1, 64, None);
    let out = expand_image_placeholders("X<image>Y", &[info]).unwrap();
    assert!(out.starts_with("X<|image_start|>"));
    assert!(out.ends_with("<|image_end|>Y"));
    assert_eq!(out.matches("<image>").count(), 64);
  }

  #[test]
  fn expand_multi_tile_with_thumbnail() {
    let info = ImagePlaceholderInfo::new(2, 2, 256, Some(64));
    let out = expand_image_placeholders("<image>", &[info]).unwrap();
    // 4 main tiles × 256 + 64 thumbnail = 1088 image tokens
    assert_eq!(out.matches("<image>").count(), 4 * 256 + 64);
    // Per-tile position tokens
    assert!(out.contains("<|img_row_1_col_1|>"));
    assert!(out.contains("<|img_row_1_col_2|>"));
    assert!(out.contains("<|img_row_2_col_1|>"));
    assert!(out.contains("<|img_row_2_col_2|>"));
    assert!(out.contains("<|img_thumbnail|>"));
  }

  #[test]
  fn expand_multi_image_preserves_order() {
    let a = ImagePlaceholderInfo::new(1, 1, 1, None);
    let b = ImagePlaceholderInfo::new(1, 1, 2, None);
    let out = expand_image_placeholders("A<image>B<image>C", &[a, b]).unwrap();
    // Expected: A<|image_start|><image><|image_end|>B<|image_start|><image><image><|image_end|>C
    assert_eq!(
      out,
      "A<|image_start|><image><|image_end|>B<|image_start|><image><image><|image_end|>C"
    );
  }

  #[test]
  fn expand_image_placeholders_matches_fixtures() {
    use serde_json::Value;
    let raw = include_str!("../tests/fixtures/image_expansion_cases.json");
    let cases: Value = serde_json::from_str(raw).expect("fixture must be valid JSON");
    let cases_arr = cases.as_array().expect("fixture must be an array");
    let mut failures: Vec<String> = Vec::new();
    for case in cases_arr {
      let name = case["name"].as_str().expect("name");
      let prompt = case["prompt"].as_str().expect("prompt");
      let expected = case["expected"].as_str().expect("expected");
      let info_v = &case["info"];
      let info = ImagePlaceholderInfo::new(
        info_v["rows"].as_u64().unwrap() as usize,
        info_v["cols"].as_u64().unwrap() as usize,
        info_v["tokens_per_main_tile"].as_u64().unwrap() as usize,
        info_v["thumbnail_tokens"].as_u64().map(|n| n as usize),
      );
      match expand_image_placeholders(prompt, &[info]) {
        Ok(actual) if actual == expected => {}
        Ok(actual) => failures.push(format!(
          "case {name}:\n  expected={expected}\n  actual  ={actual}"
        )),
        Err(e) => failures.push(format!("case {name}: error: {e}")),
      }
    }
    assert!(
      failures.is_empty(),
      "{} of {} expansion cases failed:\n{}",
      failures.len(),
      cases_arr.len(),
      failures.join("\n")
    );
  }

  #[cfg(feature = "inference")]
  #[test]
  fn apply_chat_template_matches_upstream_fixtures() {
    use serde_json::Value;
    let raw = include_str!("../tests/fixtures/chat_template_cases.json");
    let cases: Value = serde_json::from_str(raw).expect("fixture must be valid JSON");
    let cases_arr = cases.as_array().expect("fixture must be a JSON array");
    let mut failures: Vec<String> = Vec::new();
    for case in cases_arr {
      let name = case["name"].as_str().expect("each case has a name");
      let expected_raw = case["expected"]
        .as_str()
        .expect("each case has expected output");
      let messages = case["messages"].as_array().expect("messages is an array");
      let add_gen = case
        .get("add_generation_prompt")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
      let tools: Option<Value> = case.get("tools").cloned();

      // Convert each fixture message to a Message enum value. We allocate
      // small temporaries here because Message borrows from &'a str inputs.
      let owned_msgs: Vec<OwnedMsg> = messages.iter().map(OwnedMsg::from_value).collect();
      let msg_refs: Vec<Message<'_>> = owned_msgs.iter().map(OwnedMsg::as_ref).collect();

      let tools_ref = tools.as_ref();
      // Cases that embed a dynamic date use the __DATE__ sentinel. Replace it
      // with today's actual date before comparing (the renderer already uses
      // today's date, so both sides agree once the placeholder is resolved).
      let today = super::render::today_yyyymmdd_for_test();
      let expected = expected_raw.replace("__DATE__", &today);

      match apply_chat_template(&msg_refs, tools_ref, add_gen) {
                Ok(actual) if actual == expected => {}
                Ok(actual) => failures.push(format!(
                    "case {name}: actual differs from expected\n--- actual ---\n{actual}\n--- expected ---\n{expected}",
                )),
                Err(e) => failures.push(format!("case {name}: render failed: {e}")),
            }
    }
    assert!(
      failures.is_empty(),
      "{} of {} cases failed:\n{}",
      failures.len(),
      cases_arr.len(),
      failures.join("\n\n")
    );
  }

  // Owned mirror so test borrows have something to point at.
  #[cfg(feature = "inference")]
  enum OwnedMsg {
    System(String),
    UserText(String),
    UserMulti(Vec<OwnedItem>),
    Assistant {
      content: String,
      thinking: Option<String>,
    },
  }

  #[cfg(feature = "inference")]
  #[allow(dead_code)]
  enum OwnedItem {
    Image,
    Text(String),
  }

  #[cfg(feature = "inference")]
  impl OwnedMsg {
    fn from_value(v: &serde_json::Value) -> Self {
      let role = v["role"].as_str().unwrap_or("");
      match role {
        "system" => Self::System(v["content"].as_str().unwrap_or("").to_string()),
        "user" => match &v["content"] {
          serde_json::Value::String(s) => Self::UserText(s.clone()),
          serde_json::Value::Array(items) => Self::UserMulti(
            items
              .iter()
              .map(|i| match i["type"].as_str() {
                Some("image") => OwnedItem::Image,
                Some("text") => OwnedItem::Text(i["text"].as_str().unwrap_or("").to_string()),
                _ => OwnedItem::Text(String::new()),
              })
              .collect(),
          ),
          _ => Self::UserText(String::new()),
        },
        "assistant" => Self::Assistant {
          content: v["content"].as_str().unwrap_or("").to_string(),
          thinking: v.get("thinking").and_then(|t| t.as_str()).map(String::from),
        },
        _ => Self::UserText(String::new()),
      }
    }

    fn as_ref(&self) -> Message<'_> {
      match self {
        Self::System(c) => Message::System { content: c },
        Self::UserText(c) => Message::User {
          content: UserContent::Text(c),
        },
        Self::UserMulti(items) => Message::User {
          content: UserContent::Multimodal(
            items
              .iter()
              .map(|i| match i {
                OwnedItem::Image => ContentItem::Image,
                OwnedItem::Text(t) => ContentItem::Text { text: t },
              })
              .collect(),
          ),
        },
        Self::Assistant { content, thinking } => Message::Assistant {
          content,
          thinking: thinking.as_deref(),
        },
      }
    }
  }
}
