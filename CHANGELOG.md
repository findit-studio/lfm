# Changelog

All notable changes follow the format from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **MLX (`mlxrs`) Metal backend for Apple Silicon.** `Engine::from_dir` selects
  it when the directory is an MLX checkpoint (`config.json` plus a safetensors /
  sharded-index / `*.npz` / `*.gguf` weight set) and no complete ONNX graph set
  is present. Explicit-format constructors: `Engine::from_mlx_dir`,
  `from_mlx_safetensors`, and — under the matching feature — `from_mlx_npz` /
  `from_mlx_gguf`. New `npz` / `gguf` features widen which weight formats the
  backend accepts.
- **One authoritative image plan.** `ImagePlan` (in `lfm::preproc`) is produced
  once per image by the backend that will execute it, and is what the prompt's
  `<image>` / `<|img_row_R_col_C|>` / `<|img_thumbnail|>` markers, the
  context-budget admission gates, the backend's preprocessing, and the
  vision-feature splice all read. The backend re-derives its layout from the
  pixels it really decoded and raises `Error::ImagePlanMismatch` on any
  disagreement, so a marker layout can no longer diverge from the feature block
  silently. `Engine::plan_images` exposes the plans.
- **Observable backend selection.** `BackendKind`, `Options::with_backend` /
  `with_auto_backend` / `set_backend` / `backend` to pin a backend, and
  `Engine::backend` to report the one in use. `Engine::image_budget` reports the
  effective budget, which on the MLX road may be the checkpoint's own.
- `Engine::next_token_logits` — run admission, planning, prompt rendering and the
  decoder prefill, and return the (guaranteed finite) next-token logit row
  without sampling.
- Named checkpoint-layout errors: `Error::CheckpointLayoutAmbiguous`,
  `Error::CheckpointFormatDisabled`, `Error::CheckpointIncomplete`,
  `Error::BackendUnavailable`, `Error::MlxTilingMismatch`,
  `Error::ImagePlanMismatch`.
- `Engine::from_mlx_dir_unchecked` / `from_mlx_safetensors_unchecked` /
  `from_mlx_npz_unchecked` / `from_mlx_gguf_unchecked` — the named door for a
  custom MLX checkpoint, skipping the bundled tokenizer / chat-template identity
  checks (but not the structural contract).
- ORT/MLX parity integration test (`t10`), gated on `LFM_ONNX_MODEL_PATH` (or
  `LFM_MODEL_PATH`) **and** `LFM_MLX_MODEL_PATH`; it compares the preprocessing
  plans, the prefill logit rows, and the schema-constrained JSON completion, and
  prints why it skipped when either checkpoint is absent.

### Changed

- The MLX road now runs the same strict checkpoint validations as the ONNX road
  before an auto-routed constructor returns: tokenizer identity, chat-template
  identity, the model's real context limit, and the preprocessing geometry. A
  checkpoint whose tiling parameters disagree with a non-default `ImageBudget`
  is refused by name; a default `ImageBudget` adopts the checkpoint's own tiling
  instead.
- `chat_template.jinja` drift detection compares rendered-prompt equivalence
  rather than raw bytes: Jinja comments and leading whitespace (which the
  template's own `{{- bos_token -}}` strips) are normalized away, while every
  emitting construct is still compared byte for byte. The released
  `LiquidAI/LFM2.5-VL-450M-MLX-8bit` ships the bundled template with an added
  `{# … #}` header for mlx_lm, which byte-equality refused for no behavioural
  reason.

### Fixed

- A non-finite decoder logit row is rejected before either sampler. Only `-inf`
  is a value this crate writes on purpose (vocab-tail masking, the llguidance
  allow-mask, repetition-penalty overflow); a NaN or `+inf` arriving from the
  model is a broken forward. A lone `+inf` used to be admitted on the MLX road:
  greedy would pick it unconditionally, and the temperature path's softmax
  degraded to a uniform draw across the whole row — including the `-inf` entries
  a constraint mask had used to forbid a token, so a schema-disallowed token
  could win.
- A checkpoint whose only MLX weight file is in a format the build did not
  enable, a half-present ONNX graph set beside MLX weights, and an MLX
  checkpoint on a non-Apple-Silicon host are now reported as such instead of
  falling through to the ONNX path and failing on an unrelated missing graph.

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

[0.1.0]: https://github.com/findit-ai/lfm/releases/tag/v0.1.0
