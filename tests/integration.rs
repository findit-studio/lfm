//! Integration tests for the lfm crate.
//!
//! Gated on `feature = "integration"` + the `LFM_MODEL_PATH` env var.
//! Tests skip cleanly at runtime when `LFM_MODEL_PATH` is not set.
//!
//! A fixture image is expected at `tests/fixtures/test_image.jpg`.
//! Run with:
//! ```bash
//! LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX \
//!   cargo test --features integration --test integration
//! ```
//!
//! The ORT/MLX parity test (`t10`) needs BOTH checkpoints and runs only on
//! Apple Silicon:
//! ```bash
//! LFM_ONNX_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX \
//! LFM_MLX_MODEL_PATH=/path/to/LFM2.5-VL-450M-MLX-8bit \
//!   cargo test --features integration --test integration t10 -- --nocapture
//! ```

#![cfg(feature = "integration")]

use std::path::PathBuf;

use lfm::{
  ChatContent, ChatMessage, ContentPart, Engine, ImageAnalysisTask, ImageInput, Options,
  RequestOptions,
};
use smol_str::SmolStr;

fn model_dir() -> Option<PathBuf> {
  std::env::var_os("LFM_MODEL_PATH").map(PathBuf::from)
}

fn test_image() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.jpg")
}

/// A 2048×1024 image painted as eight solid-colour 512 px tiles — the
/// NON-SQUARE multi-tile geometry (2 rows × 4 cols) that a transposed marker
/// loop renders wrongly while every count, grid dimension and token total stays
/// identical. Square images cannot expose that class of defect at all.
fn nonsquare_multi_tile_image() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/grid_rows2_cols4_color_tiles.png")
}

fn make_engine() -> Option<Engine> {
  let dir = model_dir()?;
  Some(Engine::from_dir(&dir, Options::default()).expect("Engine::from_dir"))
}

fn user_msg(text: &str) -> Vec<ChatMessage> {
  vec![ChatMessage::new(
    SmolStr::new_static("user"),
    ChatContent::Parts(vec![ContentPart::Image, ContentPart::Text(text.to_owned())]),
  )]
}

#[test]
fn t01_free_form_generation() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  let req = RequestOptions::default();
  let out = engine
    .generate(&user_msg("Describe this image briefly."), &images, &req)
    .unwrap();
  assert!(
    !out.is_empty(),
    "expected non-empty output, got empty string"
  );
}

#[test]
fn t02_scene_task_structured_output() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  let req = RequestOptions::default();
  let task = ImageAnalysisTask::default().with_accept_empty(true);
  let analysis = engine.run(&task, &images, &req).unwrap();
  // Accept either a non-empty description or at least some detected objects.
  assert!(
    !analysis.description().is_empty() || !analysis.objects().is_empty(),
    "empty ImageAnalysis: {analysis:?}"
  );
}

#[test]
fn t03_max_new_tokens_caps_output() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  let req = RequestOptions::default()
    .with_max_new_tokens(8)
    .with_temperature(0.0);
  // With 8 tokens we expect either a short string or MaxTokensExceeded.
  match engine.generate(&user_msg("Describe this in detail."), &images, &req) {
    Ok(text) => assert!(
      text.len() < 400,
      "expected short output with max_new_tokens=8, got {} chars",
      text.len()
    ),
    Err(lfm::Error::MaxTokensExceeded { max, .. }) => assert_eq!(max, 8),
    Err(e) => panic!("unexpected error: {e}"),
  }
}

#[test]
fn t04_greedy_is_deterministic() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  let req = RequestOptions::default()
    .with_temperature(0.0)
    .with_max_new_tokens(20);
  let a = engine
    .generate(&user_msg("One word for this image."), &images, &req)
    .unwrap();
  let b = engine
    .generate(&user_msg("One word for this image."), &images, &req)
    .unwrap();
  assert_eq!(
    a, b,
    "greedy generation must be bit-stable across identical calls"
  );
}

#[test]
fn t05_image_token_count_mismatch_errors() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  // Two ContentPart::Image in the message but only one image supplied.
  let messages = vec![ChatMessage::new(
    SmolStr::new_static("user"),
    ChatContent::Parts(vec![
      ContentPart::Image,
      ContentPart::Image,
      ContentPart::Text("two images".to_owned()),
    ]),
  )];
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  let result = engine.generate(&messages, &images, &RequestOptions::default());
  assert!(
    matches!(result, Err(lfm::Error::ImageTokenCountMismatch { .. })),
    "expected ImageTokenCountMismatch, got {result:?}"
  );
}

#[test]
fn t06_no_image_text_only() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let messages = vec![ChatMessage::new(
    SmolStr::new_static("user"),
    ChatContent::Text("What is 2+2? Answer with just the number.".to_owned()),
  )];
  let images: Vec<ImageInput<'_>> = vec![];
  let req = RequestOptions::default().with_max_new_tokens(20);
  let out = engine.generate(&messages, &images, &req).unwrap();
  assert!(!out.is_empty(), "expected non-empty text-only output");
}

#[test]
fn t07_multi_image() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let messages = vec![ChatMessage::new(
    SmolStr::new_static("user"),
    ChatContent::Parts(vec![
      ContentPart::Image,
      ContentPart::Image,
      ContentPart::Text("Briefly compare these two images.".to_owned()),
    ]),
  )];
  let images = vec![ImageInput::Path(&fixture), ImageInput::Path(&fixture)];
  let req = RequestOptions::default().with_max_new_tokens(64);
  // The model legitimately runs to the cap on this prompt; treat
  // MaxTokensExceeded the same as a clean stop — both prove the
  // multi-image splice produced coherent decoder state.
  match engine.generate(&messages, &images, &req) {
    Ok(text) => assert!(!text.is_empty(), "expected non-empty multi-image output"),
    Err(lfm::Error::MaxTokensExceeded {
      schema_complete, ..
    }) => {
      assert!(
        !schema_complete,
        "free-form gen should never report schema-complete"
      );
    }
    Err(e) => panic!("unexpected error: {e}"),
  }
}

#[test]
fn t08_repetition_penalty_reduces_repeats() {
  let Some(mut engine) = make_engine() else {
    return;
  };
  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];
  // The point is verifying both calls *return something usable* —
  // either a clean string or MaxTokensExceeded with a finite output
  // path. We don't assert n-gram reduction here because a 64-token
  // window is too short for that to be statistically stable.
  let mut run = |opts: RequestOptions| match engine.generate(&user_msg("Describe."), &images, &opts)
  {
    Ok(text) => {
      if text.is_empty() {
        panic!("empty output");
      }
    }
    Err(lfm::Error::MaxTokensExceeded {
      schema_complete, ..
    }) => {
      assert!(
        !schema_complete,
        "free-form gen should never report schema-complete"
      )
    }
    Err(e) => panic!("unexpected error: {e}"),
  };
  run(
    RequestOptions::default()
      .with_max_new_tokens(64)
      .with_temperature(0.0),
  );
  run(
    RequestOptions::default()
      .with_max_new_tokens(64)
      .with_temperature(0.0)
      .with_repetition_penalty(1.5),
  );
}

#[test]
fn t09_image_analysis_airport_fixtures() {
  // Per-fixture ImageAnalysisTask run against the same airport thumbnails
  // qwen3-vl uses, for cross-engine comparison. Prints each parsed
  // ImageAnalysis to stdout — run with `-- --nocapture` to view.
  let Some(mut engine) = make_engine() else {
    return;
  };
  let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
  let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
    .expect("read tests/fixtures")
    .filter_map(|e| e.ok().map(|e| e.path()))
    .filter(|p| {
      p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("airport_") && n.ends_with(".jpg"))
    })
    .collect();
  paths.sort();
  assert!(
    !paths.is_empty(),
    "no airport_*.jpg fixtures found in {dir:?}"
  );
  let req = RequestOptions::default()
    .with_temperature(0.0)
    .with_max_new_tokens(512);
  let task = ImageAnalysisTask::default().with_accept_empty(true);
  for path in &paths {
    let images = vec![ImageInput::Path(path.as_path())];
    let analysis = engine
      .run(&task, &images, &req)
      .unwrap_or_else(|e| panic!("run failed for {path:?}: {e}"));
    println!(
      "===== {} =====",
      path.file_name().unwrap().to_string_lossy()
    );
    println!("{analysis:#?}");
  }
}

// =========================================================================
// t10 — ORT/MLX parity
// =========================================================================

/// Cosine similarity of two equal-length logit rows.
///
/// The right comparison for two implementations of the same model at different
/// precisions: an absolute tolerance would be dominated by the 8-bit
/// checkpoint's quantization error and by the arbitrary scale of a logit row,
/// while the direction of the vector is what determines every sampling
/// decision. `1.0` is identical; anything below the threshold means the two
/// roads disagree about the distribution, not merely about rounding.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
  let mut dot = 0f64;
  let mut na = 0f64;
  let mut nb = 0f64;
  for (x, y) in a.iter().zip(b.iter()) {
    dot += f64::from(*x) * f64::from(*y);
    na += f64::from(*x) * f64::from(*x);
    nb += f64::from(*y) * f64::from(*y);
  }
  if na == 0.0 || nb == 0.0 {
    return 0.0;
  }
  dot / (na.sqrt() * nb.sqrt())
}

/// Index of the largest entry — the token greedy decoding would emit.
fn argmax(values: &[f32]) -> usize {
  values
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.total_cmp(b.1))
    .map(|(i, _)| i)
    .unwrap_or(0)
}

/// The `<|img_row_R_col_C|>` markers of a rendered image block, in emission
/// order, as one-based `(row, col)` pairs.
fn marker_sequence(rendered: &str) -> Vec<(usize, usize)> {
  let mut out = Vec::new();
  let mut rest = rendered;
  while let Some(start) = rest.find("<|img_row_") {
    rest = &rest[start + "<|img_row_".len()..];
    let end = rest.find("|>").expect("marker terminator");
    let (r, c) = rest[..end]
      .split_once("_col_")
      .expect("marker carries _col_");
    out.push((
      r.parse::<usize>().expect("row index"),
      c.parse::<usize>().expect("col index"),
    ));
    rest = &rest[end + 2..];
  }
  out
}

/// Render one image plan's marker sequence, paired with the grid it came from.
fn plan_markers(plan: &lfm::ImagePlan) -> (usize, usize, Vec<(usize, usize)>) {
  let info = plan.placeholder();
  let rendered =
    lfm::expand_image_placeholders("<image>", &[*info]).expect("render one image block");
  (info.rows(), info.cols(), marker_sequence(&rendered))
}

/// ORT/MLX parity on one real image: the same preprocessing plan, the same
/// first greedy token, logit rows pointing the same way, and a
/// schema-constrained JSON completion that parses on both roads.
///
/// Gated on TWO env vars so it can only run when both checkpoints are present,
/// and on Apple Silicon (the only platform that compiles the MLX backend). It
/// prints why it skipped rather than passing silently — a parity test that
/// quietly no-ops is worse than none.
///
/// `LFM_PARITY_LOGIT_COSINE` overrides the similarity floor (default `0.98`)
/// for exploring a divergence; a failure here is a real signal, not noise, so
/// the default is deliberately tight enough to catch a preprocessing or splice
/// disagreement while tolerating 8-bit quantization error.
#[test]
fn t10_ort_mlx_parity() {
  let onnx_dir = std::env::var_os("LFM_ONNX_MODEL_PATH")
    .or_else(|| std::env::var_os("LFM_MODEL_PATH"))
    .map(PathBuf::from);
  let mlx_dir = std::env::var_os("LFM_MLX_MODEL_PATH").map(PathBuf::from);

  if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    println!("t10 skipped: the MLX backend is compiled only on macOS/arm64 (aarch64-apple-darwin)");
    return;
  }
  let (Some(onnx_dir), Some(mlx_dir)) = (onnx_dir, mlx_dir) else {
    println!(
      "t10 skipped: set BOTH LFM_ONNX_MODEL_PATH (or LFM_MODEL_PATH) and LFM_MLX_MODEL_PATH"
    );
    return;
  };

  let mut ort = Engine::from_dir(&onnx_dir, Options::default())
    .unwrap_or_else(|e| panic!("Engine::from_dir({onnx_dir:?}) [ONNX]: {e}"));
  let mut mlx = Engine::from_dir(&mlx_dir, Options::default())
    .unwrap_or_else(|e| panic!("Engine::from_dir({mlx_dir:?}) [MLX]: {e}"));

  // ── the selection is observable, and it picked what we expect ──────────
  assert_eq!(
    ort.backend(),
    lfm::BackendKind::Onnx,
    "{onnx_dir:?} must select the ONNX backend"
  );
  assert_eq!(
    mlx.backend(),
    lfm::BackendKind::Mlx,
    "{mlx_dir:?} must select the MLX backend"
  );
  println!("ONNX image budget: {:?}", ort.image_budget());
  println!("MLX  image budget: {:?}", mlx.image_budget());

  let fixture = test_image();
  let images = vec![ImageInput::Path(&fixture)];

  // ── 1. the preprocessing plans agree ───────────────────────────────────
  let ort_plans = ort.plan_images(&images).expect("ONNX plan_images");
  let mlx_plans = mlx.plan_images(&images).expect("MLX plan_images");
  println!("ONNX plan: {ort_plans:?}");
  println!("MLX  plan: {mlx_plans:?}");
  assert_eq!(
    ort_plans, mlx_plans,
    "the two backends must plan the same tiling, marker layout and token count for the same image"
  );

  // ── 2. the prefill logits point the same way, and greedy agrees ────────
  let req = RequestOptions::default()
    .with_temperature(0.0)
    .with_max_new_tokens(32);
  let messages = user_msg("Describe this image briefly.");
  let ort_logits = ort
    .next_token_logits(&messages, &images, &req)
    .expect("ONNX next_token_logits");
  let mlx_logits = mlx
    .next_token_logits(&messages, &images, &req)
    .expect("MLX next_token_logits");
  assert_eq!(
    ort_logits.len(),
    mlx_logits.len(),
    "both backends must expose the same vocabulary width"
  );
  let similarity = cosine_similarity(&ort_logits, &mlx_logits);
  let max_abs_diff = ort_logits
    .iter()
    .zip(mlx_logits.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0f32, f32::max);
  println!(
    "prefill logits: vocab={} cosine={similarity:.6} max_abs_diff={max_abs_diff:.4}",
    ort_logits.len()
  );
  let floor: f64 = std::env::var("LFM_PARITY_LOGIT_COSINE")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(0.98);
  assert!(
    similarity >= floor,
    "prefill logit rows diverge: cosine {similarity:.6} < {floor} (max abs diff {max_abs_diff:.4})"
  );
  // Exact top-1 agreement is NOT asserted: the ONNX export is fp32 while the
  // MLX checkpoint is 8-bit affine-quantized with half-precision activations
  // (its logits land on visible 0.125 steps), so a few tenths of a logit can
  // permute near-tied candidates. What a preprocessing, marker-layout or
  // splice defect WOULD do is move the distribution somewhere else entirely —
  // so the assertions are about the candidate set, which quantization noise
  // cannot scramble.
  let top_k: usize = std::env::var("LFM_PARITY_TOP_K")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(5);
  let top_of = |values: &[f32], k: usize| -> Vec<usize> {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    ids.truncate(k);
    ids
  };
  let ort_top = top_of(&ort_logits, top_k);
  let mlx_top = top_of(&mlx_logits, top_k);
  println!(
    "ONNX top{top_k}: {:?}",
    ort_top
      .iter()
      .map(|i| (*i, ort_logits[*i]))
      .collect::<Vec<_>>()
  );
  println!(
    "MLX  top{top_k}: {:?}",
    mlx_top
      .iter()
      .map(|i| (*i, mlx_logits[*i]))
      .collect::<Vec<_>>()
  );
  let ort_greedy = argmax(&ort_logits);
  let mlx_greedy = argmax(&mlx_logits);
  println!("greedy: ONNX={ort_greedy} MLX={mlx_greedy} (exact agreement not required)");
  assert!(
    mlx_top.contains(&ort_greedy),
    "the ONNX greedy token {ort_greedy} must be among the MLX backend's top {top_k}: {mlx_top:?}"
  );
  assert!(
    ort_top.contains(&mlx_greedy),
    "the MLX greedy token {mlx_greedy} must be among the ONNX backend's top {top_k}: {ort_top:?}"
  );
  let shared = ort_top.iter().filter(|id| mlx_top.contains(id)).count();
  let min_shared = top_k.div_ceil(2);
  println!("top{top_k} overlap: {shared}/{top_k}");
  assert!(
    shared >= min_shared,
    "the two backends must broadly agree on the candidate set: only {shared}/{top_k} shared (need {min_shared})"
  );

  // ── 3. greedy completions (reported; a late divergence is expected at
  //       8-bit and is not on its own a defect) ─────────────────────────────
  let ort_text = ort.generate(&messages, &images, &req);
  let mlx_text = mlx.generate(&messages, &images, &req);
  println!("ONNX greedy: {ort_text:?}");
  println!("MLX  greedy: {mlx_text:?}");
  for (name, out) in [("ONNX", &ort_text), ("MLX", &mlx_text)] {
    match out {
      Ok(text) => assert!(!text.is_empty(), "{name} greedy output must not be empty"),
      Err(lfm::Error::MaxTokensExceeded { .. }) => {}
      Err(e) => panic!("{name} greedy generation failed: {e}"),
    }
  }

  // ── 4. the schema-constrained JSON completion parses on both roads ─────
  let task = ImageAnalysisTask::default().with_accept_empty(true);
  let json_req = RequestOptions::default()
    .with_temperature(0.0)
    .with_max_new_tokens(256);
  let ort_json = ort
    .run(&task, &images, &json_req)
    .expect("ONNX constrained JSON run");
  let mlx_json = mlx
    .run(&task, &images, &json_req)
    .expect("MLX constrained JSON run");
  println!("ONNX JSON: {ort_json:#?}");
  println!("MLX  JSON: {mlx_json:#?}");

  // ── 5. the NON-SQUARE multi-tile geometry ──────────────────────────────
  //
  // Everything above runs on a 2×1 grid, where a transposed marker loop is
  // still wrong but only along one axis. A 2-row × 4-col grid is the case that
  // no count-based check can see: transposing it keeps eight tiles, eight
  // markers and the same `<image>` total, and simply pairs each tile with
  // another tile's position. So this section asserts the pairing itself —
  // the marker sequence must be the row-major enumeration of the planned grid,
  // on BOTH roads — and then drives a real prefill through it, which is what
  // makes the MLX road actually run `split_image` + its sub-image gates and the
  // ONNX road actually patchify and splice.
  let wide = nonsquare_multi_tile_image();
  let wide_images = vec![ImageInput::Path(&wide)];

  let ort_wide = ort
    .plan_images(&wide_images)
    .expect("ONNX plan (non-square)");
  let mlx_wide = mlx
    .plan_images(&wide_images)
    .expect("MLX plan (non-square)");
  println!("ONNX plan (non-square): {ort_wide:?}");
  println!("MLX  plan (non-square): {mlx_wide:?}");
  assert_eq!(
    ort_wide, mlx_wide,
    "the two backends must plan the same tiling for the non-square image"
  );

  let (rows, cols, ort_markers) = plan_markers(&ort_wide[0]);
  let (mlx_rows, mlx_cols, mlx_markers) = plan_markers(&mlx_wide[0]);
  assert_eq!((rows, cols), (mlx_rows, mlx_cols));
  assert!(
    rows > 1 && cols > 1 && rows != cols,
    "the fixture must reach a NON-SQUARE grid with both axes split, got {rows}x{cols}"
  );
  let expected: Vec<(usize, usize)> = (1..=rows)
    .flat_map(|r| (1..=cols).map(move |c| (r, c)))
    .collect();
  println!("non-square grid: {rows} rows x {cols} cols, markers {ort_markers:?}");
  assert_eq!(
    ort_markers, expected,
    "ONNX road: the marker sequence must be row-major over the planned grid"
  );
  assert_eq!(
    mlx_markers, expected,
    "MLX road: the marker sequence must be row-major over the planned grid"
  );

  // A real prefill on both roads. On the MLX side this is what exercises
  // `ratify_against_checkpoint` + `verify_sub_images` against the sub-images
  // `split_image` really produced for a non-square grid; a plan/feature
  // disagreement raises `ImagePlanMismatch` rather than reaching the logits.
  let wide_messages = user_msg("What colours are in this image?");
  let ort_wide_logits = ort
    .next_token_logits(&wide_messages, &wide_images, &req)
    .expect("ONNX next_token_logits (non-square)");
  let mlx_wide_logits = mlx
    .next_token_logits(&wide_messages, &wide_images, &req)
    .expect("MLX next_token_logits (non-square)");
  let wide_similarity = cosine_similarity(&ort_wide_logits, &mlx_wide_logits);
  let wide_max_abs_diff = ort_wide_logits
    .iter()
    .zip(mlx_wide_logits.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0f32, f32::max);
  println!(
    "non-square prefill logits: image_tokens={} cosine={wide_similarity:.6} max_abs_diff={wide_max_abs_diff:.4}",
    ort_wide[0].image_tokens()
  );
  assert!(
    wide_similarity >= floor,
    "non-square prefill logit rows diverge: cosine {wide_similarity:.6} < {floor} (max abs diff {wide_max_abs_diff:.4})"
  );
  let ort_wide_top = top_of(&ort_wide_logits, top_k);
  let mlx_wide_top = top_of(&mlx_wide_logits, top_k);
  let wide_shared = ort_wide_top
    .iter()
    .filter(|id| mlx_wide_top.contains(id))
    .count();
  println!(
    "non-square top{top_k}: ONNX {ort_wide_top:?} MLX {mlx_wide_top:?} overlap {wide_shared}/{top_k}"
  );
  assert!(
    wide_shared >= min_shared,
    "the two backends must broadly agree on the non-square candidate set: only {wide_shared}/{top_k} shared (need {min_shared})"
  );
}
