# Changelog

All notable changes follow the format from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/).

## [0.1.2] — 2026-08-31

### Fixed

- Fixed a build break on a fresh `Cargo.lock`: `ort` has never cut a
  stable `2.0.0`, so — per Cargo's pre-release version-requirement rules
  — the bare `ort = "2.0.0-rc.12"` floor already admitted the newer
  `2.0.0-rc.13` (a requirement naming a pre-release matches later
  pre-releases of the same `[major, minor, patch]`, same as an ordinary
  caret floor; only a requirement that names no pre-release at all is
  restricted to stable releases). `rc.13` relocated the
  execution-provider types and marked `GraphOptimizationLevel`
  `#[non_exhaustive]`, so any resolve landing on `rc.13` — which every
  resolve did, since this crate does not commit a lockfile — failed to
  compile. The floor is now written against `2.0.0-rc.13` (same bare/caret
  form, so it keeps floating to future `rc.N` releases and the eventual
  stable `2.0.0` the way the old line did), and the code follows ort's
  rc.13 migration:
  - `ort::execution_providers::{CUDA,TensorRT,DirectML,ROCm,CoreML}ExecutionProvider`
    → `ort::ep::{CUDA,TensorRT,DirectML,ROCm,CoreML}`. rc.13 removed the
    deprecated compatibility aliases entirely (present but
    `#[deprecated]` through rc.12) and made each `ep` submodule
    compile-time gated on its own `ort/<name>` Cargo feature (previously
    the structs compiled unconditionally and only linking was
    feature-gated).
  - The `GraphOptimizationLevel` → internal `GraphOptLevelMirror`
    conversion gained a defensive wildcard arm to satisfy
    `#[non_exhaustive]`'s forward-compatibility requirement. The 5 known
    variants (`Disable`/`Level1`/`Level2`/`Level3`/`All`) are unchanged;
    the wildcard panics with a clear message rather than silently
    mis-mapping a level this crate's bit-stability documentation makes
    promises about, and can only fire once `ort` ships a 6th variant.

  `ort` 2.0.0-rc.13 also bundles ONNX Runtime 1.28 (up from 1.24 at
  rc.12) and, for `--features cuda` users, now ships CUDA 13 binaries
  only (upstream dropped CUDA 12 support).

### Added

- Opt-in `lax-feature-matching` feature (`ort/lax-feature-matching`), not
  implied by `cuda`/`tensorrt`/`directml`/`rocm`/`coreml`. `ort`
  2.0.0-rc.13 now hard-errors at link time when the enabled
  execution-provider features don't match one of its published
  prebuilt-binary bundles exactly (previously this silently fell back to
  CPU). No platform ships a bundle covering all five GPU backends at
  once, so `--all-features` (a compile-coverage flag — no real
  deployment enables every backend together) needs this feature to link.
  A real single-backend build (e.g. `--features cuda` alone) is
  unaffected and stays strict, matching upstream's intent.

## [0.1.0] — 2026-05-03

### Added

- Public `Engine` API for LiquidAI LFM2.5-VL-450M ONNX inference:
  - `Engine::from_dir(model_dir, opts)` — load from a directory containing
    the three ONNX graphs + `tokenizer.json`.
  - `Engine::from_paths(EnginePaths, opts)` — explicit per-graph path
    override.
  - `Engine::from_onnx_dir(onnx_dir, opts)` (`bundled` feature) — load from
    a directory containing **only the ONNX files**; the bundled tokenizer +
    JSON configs (~4.5 MB embedded via `include_bytes!`) are written to a
    per-process temp file and used in place of the missing on-disk files.
    ONNX model files are NOT bundled (vision_encoder ~86 MB, decoder ~350 MB).
  - `engine.generate(messages, images, req)` — free-form generation;
    returns the model's raw text output.
  - `engine.run(&task, messages, images, req)` — schema-constrained
    generation via any `vlm_tasks::Task` instance; returns `Task::Output`.
- Bundled `SceneTask` (wrapping `vlm_tasks::SceneAnalysis`) for structured
  scene analysis without any extra configuration.
- Public chat types: `ChatMessage`, `ChatContent`, `ContentPart`,
  `ImageInput`.
- Public configuration: `Options`, `RequestOptions`, `ImageBudget`,
  `ThreadOptions`, `GraphOptimizationLevel`.
- Wasm-friendly preprocessing subset under
  `--no-default-features --features decoders` (no `ort`, no `tokenizers`):
  `Preprocessor`, `TileGrid`, `PreprocessedImage`,
  `decode_bytes_with_orientation`.
- EXIF-aware image decoding: `decode_with_orientation` (native) and
  `decode_bytes_with_orientation` (all targets including wasm).
- Schema-constrained sampling via `llguidance` 1.7 token-mask filtering
  applied at each decode step.
- Hybrid KV+conv-state cache management for the LFM2 hybrid LM
  (10 conv-state layers + 6 KV-attn layers, sparse layer indices).
- Per-image vision-encoder dispatch (Phase 0 G6 contract: one image per
  encoder call; batched multi-image calls produce silently-wrong embeddings).
- Chat template rendering with `minijinja` 2: `apply_chat_template`,
  `expand_image_placeholders`, bundled Jinja2 source via `include_str!`.
- Examples:
  - `smoke` — free-form generation over one image.
  - `scene_analysis` — structured `SceneAnalysis` output.
  - `preprocess_only` — preprocessing-only (no inference, no-default-features).
  - `qwen_compare` — side-by-side LFM vs Qwen3-VL comparison
    (requires `--features comparison`).
- Benches: `bench_preproc`, `bench_tile_grid`, `bench_chat_template`.
- Integration test suite gated on `feature = "integration"` and the
  `LFM_MODEL_PATH` env var.
- Execution-provider gates: `cuda`, `tensorrt`, `directml`, `rocm`,
  `coreml` (all off by default; each implies `inference`).
- `serde` feature: `Serialize`/`Deserialize` on `Options`,
  `RequestOptions`, `ThreadOptions`, `ImageBudget`.

### Model weights

The crate wraps [LFM2.5-VL-450M-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX).
The weights ship under the [LFM Open License v1.0](https://www.liquid.ai/lfm-license)
— verify your use case complies with Liquid AI's terms separately from
this crate's MIT OR Apache-2.0 license.

[0.1.2]: https://github.com/findit-ai/lfm/releases/tag/v0.1.2
[0.1.0]: https://github.com/findit-ai/lfm/releases/tag/v0.1.0
