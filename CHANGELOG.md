# Changelog

All notable changes follow the format from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-09-05

### Changed

- **`Options` is a layered document: the shared knobs on top, the selected
  engine's own knobs flattened in, and every tier defaults.** A table naming
  only a road — `{ "backend": "mlx" }` — used to be refused by the first
  missing field's name; each tier now fills from its own `new()`. The engine
  tier is the new `BackendOptions`, an internally tagged enum serialized under
  the key `backend`, so one key both names the road and selects that road's
  knob struct: `AutoOptions` (none), `OrtOptions` (`thread`,
  `optimization_level`) or `MlxOptions` (none yet). Its variants are
  `cfg`-gated on their backend's presence, so the roster serde prints in
  `unknown variant ...` is exactly the set of roads the build can run, and a
  document naming a road this build did not compile is refused by that roster
  rather than at load time. **Breaking**, in five places:
  - `Options::backend` returns `&BackendOptions` (the tier) instead of
    `Option<BackendKind>`; the pin alone is now `Options::backend_kind`.
  - `Options::with_backend` / `set_backend` take a `BackendOptions` —
    `Options::new().with_backend(BackendOptions::mlx(MlxOptions::new()))`.
    `with_auto_backend` is unchanged.
  - `Options::thread` / `with_thread` / `set_thread` and `optimization_level` /
    `with_optimization_level` / `set_optimization_level` moved to `OrtOptions`,
    which owns them: both are ONNX Runtime's alone, and the MLX road (Metal)
    silently ignored them. Setting them therefore means naming the ONNX road;
    under `backend = "auto"` the session is built from `OrtOptions::new()`.
    `OrtOptions::deterministic()` is the bit-stability preset that
    `Options::new().with_thread(ThreadOptions::deterministic())` used to be.
  - `backend` is a **required** key of the serde document: serde's flatten
    cannot supply a missing tag, so a document with no `backend` is
    ``missing field `backend` `` rather than auto-select. `"auto"` is how a
    document defers the road to the checkpoint layout — and it is a zero-field
    struct rather than a unit variant precisely so that a key written beside it
    is refused by name instead of silently absorbed.
  - `BackendKind`'s wire form is now lower-case (`"onnx"` / `"mlx"`, was
    `"Onnx"` / `"Mlx"`), the same name `BackendKind::as_str` reports and the
    same one the document's tag uses: one vocabulary in documents, diagnostics
    and `Engine::backend`.

  The top-level key set of a full ONNX document is unchanged
  (`request`, `image_budget`, `backend`, `thread`, `optimization_level`) —
  `thread` and `optimization_level` changed tiers, not depth, because the
  engine tier is flattened. Key order moved: `backend` now leads the knobs it
  owns. On a build without the ORT backend those two keys are absent, which is
  the point of the gating.

### Added

- **`ort` is no longer a dependency on `wasm32`.** The platform-default target
  row pulled it on every target that is not Apple Silicon macOS, wasm included,
  and a *target* dependency is resolved before features are — so
  `--no-default-features` could not opt out of it and `ort-sys` failed the build
  with "no prebuilt binaries available for target wasm32-unknown-unknown". The
  row's cfg now also excludes `wasm32`, and `build.rs` emits `ort_backend` under
  exactly the same predicate. wasm builds with no backend at all; an `Options`
  document there still parses, with `auto` as the only road in its roster.
- **`ThreadOptions`' counts are `Option<u16>`, not `Option<usize>`.** `ort`
  forwards a thread count to the C API's signed `int`, so a `usize` above
  `i32::MAX` wrapped silently and a merely large one asked for a per-session
  pool big enough to exhaust the process — three times over, since an `Engine`
  builds three sessions, and the new parallel-execution path forwarded every
  value above 1. `u16` makes both unrepresentable rather than merely rejected:
  the session seam converts with `usize::from`, infallibly and losslessly, so
  there is no ceiling constant, no validation hook to forget, and no cast. An
  out-of-range count is refused by serde's own range check (`invalid value:
  integer \`65536\`, expected u16`) in JSON, YAML and TOML alike. **Breaking**
  for Rust callers of `ThreadOptions`' accessors and setters; **the wire form of
  every sane document is unchanged** — `1` is `1` in both types, and only a
  count above 65 535 changes meaning, from silently wrapped to refused.
- **`inter_threads > 1` now selects ort's parallel execution mode**, via the new
  `ThreadOptions::requires_parallel_execution`. ort's inter-op thread pool
  exists only in that mode; in the default sequential mode
  `SetInterOpNumThreads` is accepted and ignored, so every ONNX session built
  before this took an `inter_threads` setting and silently ran
  single-graph-threaded anyway. `None` and `Some(1)` keep the sequential mode,
  so `ThreadOptions::deterministic()`, `OrtOptions::deterministic()` and every
  other existing recipe build byte-identically to before — parallel execution
  is not bit-stable, and nothing opts into it without asking for more than one
  inter-op thread.
- **Every nested options table refuses unknown keys.** `deny_unknown_fields`
  does not recurse, so a tier's strictness could not see inside `thread`,
  `request` or `image_budget`: `{ "backend": "onnx", "thread": {
  "intra_thread": 1 } }` deserialized happily with *both* thread counts left at
  `None`, handing back ORT's defaults for a misspelled determinism control.
  `RequestOptions`, `ImageBudget` and `ThreadOptions` now each carry
  `#[serde(deny_unknown_fields)]`. **Breaking** for a document that carried a
  stray key inside one of those tables; the required-field semantics of those
  tables are unchanged.
- **`BackendOptions`, `AutoOptions`, `OrtOptions` and `MlxOptions`** — the
  tiers above, exported from the crate root under their backends' `cfg`s.
  `Options::effective_ort_options` reports the ORT knobs a configuration
  actually runs the ONNX road with.
- **`tests/options_document.rs`** pins the document's shape: per-tier
  defaulting, the roster's refusal of an uncompiled road, `deny_unknown_fields`
  on each engine tier (including for a misspelled *shared* key, which falls
  through the flatten and is refused there), one refusal per nested table in
  both JSON and TOML, and the JSON and TOML document forms.

## [0.2.0] — 2026-08-31

### Changed

- **`llmtask` bumped `0.1` → `0.3`; lfm's own `ImageAnalysisTask` copy is
  removed.** `llmtask` 0.3 absorbed the canonical `ImageAnalysisTask`
  (prompt, JSON Schema, and parser) that `lfm` and `qwen3-vl` had each
  carried as byte-for-byte-equivalent copies since `llmtask` 0.1.
  `lfm::ImageAnalysis` and `lfm::ImageAnalysisTask` now re-export
  `llmtask`'s types directly — the public import paths are unchanged, but
  the underlying shape is **breaking**: `ImageAnalysis` moved from nine
  fields to ten, `mood` was renamed to `emotion`, and a new required
  `categories` field (broad content categories, coarser than `tags`) was
  added. The parser also gained duplicate-top-level-key refusal
  (`JsonParseError::DuplicateField`) and a dedicated
  `JsonParseError::UnknownFields` variant, both now inherited for free.
  This crate's local copy of the 28 parser-focused unit tests is removed
  (that coverage now lives upstream in `llmtask`'s own test suite); the
  engine-integration tests in `tests/integration.rs` that exercise
  `ImageAnalysisTask` against real ONNX/MLX inference are unaffected and
  stay here.

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
  checks and the declared preprocessing contract (but not the structural
  contract).
- ORT/MLX parity integration test (`t10`), gated on `LFM_ONNX_MODEL_PATH` (or
  `LFM_MODEL_PATH`) **and** `LFM_MLX_MODEL_PATH`; it compares the preprocessing
  plans, the prefill logit rows, and the schema-constrained JSON completion,
  repeats the plan comparison and a prefill on a NON-SQUARE multi-tile image
  (2 rows × 4 cols) while asserting each road's marker sequence is the row-major
  enumeration of its planned grid, and prints why it skipped when either
  checkpoint is absent.
- `ort` Cargo feature, aarch64-macos only: opts the ONNX (`ort`) backend back
  in there (off by `default` — MLX is the native road on Apple Silicon). Needed
  to run `Engine::from_paths` / `from_onnx_dir` / the ONNX arm of `from_dir`,
  or the `t10` parity test, on Apple Silicon. On every other target `ort` is
  unaffected by this feature — it stays the mandatory dependency it always was.

### Changed

- **`ort` is now optional on aarch64-macos**, gated behind the new `ort`
  feature; it remains a mandatory dependency on every other target, unchanged
  from before. `BackendKind::Onnx` on Apple Silicon without the feature is a
  named `Error::BackendUnavailable` (mirroring the existing MLX-off-Apple-Silicon
  error) rather than a build failure or a silent fallback — `Options::backend()`
  / `Engine::backend()` still report which backend actually ran.
- The MLX road now runs the same strict checkpoint validations as the ONNX road
  before an auto-routed constructor returns: tokenizer identity, chat-template
  identity, the model's real context limit, and the preprocessing geometry. A
  checkpoint whose tiling parameters disagree with a non-default `ImageBudget`
  is refused by name; a default `ImageBudget` adopts the checkpoint's own tiling
  instead.
- Both roads refuse a checkpoint whose `config.json` sets
  `use_image_special_tokens: false` — by name on the MLX road
  (`Error::MlxTilingMismatch`), as a strict-constructor drift error on the ONNX
  road. This crate always brackets an image block with `<|image_start|>` /
  `<|image_end|>` and always budgets both tokens, so such a checkpoint would be
  prompted with two tokens its processor contract omits while every token count
  still matched. An absent key means `true` (upstream's default) and is
  accepted.
- The MLX execute-time gate now re-derives the tile layout from the dimensions
  it actually decoded and ratifies it against the plan the prompt's markers were
  rendered from, before any feature is spliced. A path-backed image is opened
  once for header planning and again for decoding; a file replaced in between
  (1920×1080 → 1080×1920) yields a transposed grid whose sub-image count and
  per-sub-image token counts are identical, so the previous count-only gate
  admitted it and spliced row-major features under markers for the old layout.
- `chat_template.jinja` drift detection normalizes exactly one thing before an
  otherwise byte-exact comparison: the template's **leading** run of Jinja
  comments and whitespace. The released `LiquidAI/LFM2.5-VL-450M-MLX-8bit` ships
  the bundled template with a `{# … #}` header prepended for mlx_lm, which
  byte-equality refused for no behavioural reason; at offset zero a `{#`
  unambiguously opens a comment and the whitespace it leaves is swallowed by the
  template's own `{{- bos_token -}}`. Comments elsewhere are NOT normalized —
  `{#` opens a comment only in template-text context, so erasing every span
  would rewrite `{{- "sys{# drift #}tem" -}}` into the bundled
  `{{- "system" -}}` and pass a checkpoint that renders a different prompt.

### Fixed

- **Per-tile position markers are emitted row-major, matching the tiles.** The
  shared image-block renderer iterated the grid columns-outer / rows-inner while
  both feature producers return tiles row-major, so on every NON-SQUARE
  multi-tile image each tile was conditioned under another tile's
  `<|img_row_R_col_C|>` marker. Upstream is unambiguous:
  `image_processing_lfm2_vl.py:310` cuts tiles with
  `split_to_tiles(num_tiles_height=grid_height, num_tiles_width=grid_width)`
  (row-major, height outer — `image_transforms.py:815-836`); `:328` returns
  `(images, grid_width, grid_height)`, unpacked at `:418` as
  `images, num_cols, num_rows`, published at `:553-554` as `image_rows` /
  `image_cols`; and `processing_lfm2_vl.py:208-211` emits
  `for row in range(rows): for col in range(cols)`. Square grids are unaffected,
  which is why nothing caught it: the tile count, the `<image>` total, every
  `spatial_shapes` entry and the ORT/MLX parity run were identical either way.
  This was a defect on **both** roads — the renderer is shared, so the ONNX path
  is fixed by the same change. Covered by a colour-coded oracle test that paints
  tile `(r, c)` a distinct solid colour on a 2×4 and a 4×2 grid and asserts the
  sub-image at each position decodes to the position its marker names, by
  row-major expansion fixtures replacing the reversed ones, and by a non-square
  multi-tile image added to the ORT/MLX parity run.
- **Strict MLX constructors validate the preprocessing contract.** They checked
  `tokenizer.json` and `chat_template.jinja` only, while `mlxrs` hardcodes
  `image_mean = image_std = 0.5`, rescale `1/255` and bilinear resampling
  (`Lfm2Vl::processor_config` never reads them from the checkpoint). A revision
  shipping different normalization, a different rescale factor, or a
  non-bilinear `resample` therefore loaded cleanly and fed systematically wrong
  pixels to the vision tower with every count, grid and dimension check green —
  the version-skew class the ONNX strict constructor already refused.
  `Engine::from_mlx_dir`, `from_mlx_safetensors`, `from_mlx_npz` and
  `from_mlx_gguf` now all validate it (a missing `preprocessor_config.json` also
  fails closed); the `_unchecked` doors remain the only escape.
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

[0.3.0]: https://github.com/findit-studio/lfm/releases/tag/v0.3.0
[0.2.0]: https://github.com/findit-studio/lfm/releases/tag/v0.2.0
[0.1.2]: https://github.com/findit-studio/lfm/releases/tag/v0.1.2
[0.1.0]: https://github.com/findit-studio/lfm/releases/tag/v0.1.0
