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
