//! Demo: load LFM2.5-VL, analyze a scene image, print structured output.
//!
//! Usage:
//! ```bash
//! LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX \
//!   cargo run --example scene_analysis --features inference,decoders -- image.jpg
//! ```

#[cfg(feature = "inference")]
fn main() -> lfm::Result<()> {
  let model_dir =
    std::env::var("LFM_MODEL_PATH").expect("set LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX");
  let image_path = std::env::args()
    .nth(1)
    .expect("usage: scene_analysis <image>");

  let opts = lfm::Options::default();
  let mut engine = lfm::Engine::from_dir(&model_dir, opts)?;
  // The default task asks for `description` (one capped sentence) and
  // `tags` (a capped label array) only. This demo switches every other
  // field on, so it prints the full ten-field analysis.
  let task = lfm::ImageAnalysisTask::new().with_extensions(lfm::Extension::ALL);

  // Engine::run wires the task prompt internally — caller only supplies
  // images + the task instance.
  let images = vec![lfm::ImageInput::Path(std::path::Path::new(&image_path))];
  let req = lfm::RequestOptions::default();

  let analysis = engine.run(&task, &images, &req)?;
  println!("{:#?}", analysis);
  Ok(())
}

#[cfg(not(feature = "inference"))]
fn main() {
  eprintln!("scene_analysis requires the `inference` feature");
}
