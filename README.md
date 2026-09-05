<div align="center">
<h1>lfm</h1>
</div>
<div align="center">

Rust ONNX/MLX inference for [LiquidAI LFM2.5-VL][lfm-card] — a 450M-parameter vision-language model with schema-constrained sampling via [llguidance]. Implements the engine-agnostic [`llmtask::Task`] contract, so any `Task` written against `llmtask` runs through `lfm` unchanged.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/lfm-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Flfm" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/lfm/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-studio/lfm?style=for-the-badge&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-lfm-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/lfm?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/lfm?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCAxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTYgYy0xMi4xMTgtMTcuOTM3LTMyLjI2Mi0yOC42NDUtNTMuODgyLTI4LjY0NSBjLTM1LjgzMywwLTY0Ljk4NSwyOS4xNTItNjQuOTg1LDY0Ljk4NnMyOS4xNTIsNjQuOTg2LDY0Ljk4NSw2NC45ODZjMjIuMjgxLDAsNDIuNzU5LTExLjIxOCw1NC43NzgtMzAuMDA5IEMyMDAuMjA4LDE2OS4xNDcsMTk4Ljk4NSwxNjMuNTgyLDE5NC43OTgsMTYwLjkwM3oiLz4gPC9nPiA8L2c+Cg08L3N2Zz4=" height="22">

</div>

## Overview

`lfm` is the [LiquidAI LFM2.5-VL][lfm-card] inference engine on Rust + ONNX Runtime / MLX + llguidance:

- **[`Engine`]** — sync, single-threaded; built on `ort` 2.0, with an MLX (Metal) backend auto-selected on Apple Silicon. `Engine::run<T: Task<Value = serde_json::Value>>` accepts any [`llmtask::Task`] whose grammar is JSON Schema, Lark, or Regex. Schema-constrained sampling is enforced by [llguidance] token-mask filtering. `Engine::generate` is the unconstrained path for free-form text.
- **[`ImageAnalysisTask`]** — built-in image-analysis preset that produces the canonical [`llmtask::ImageAnalysis`] output type, sharing the schema and parser with [`qwen3-vl`].
- **Bundled assets** — the `bundled` feature ships LFM2.5-VL's tokenizer, chat template, and preprocessor configs as `include_bytes!`. `Engine::from_onnx_dir` then accepts a directory containing only the three ONNX graphs; no separate tokenizer download required.
- **Wasm-friendly preprocessing** — `preproc::Preprocessor`, `TileGrid`, and EXIF-aware decode helpers compile under `--no-default-features --features decoders` (no `ort`, no `tokenizers`).

[`Engine`]: https://docs.rs/lfm/latest/lfm/struct.Engine.html
[`ImageAnalysisTask`]: https://docs.rs/lfm/latest/lfm/struct.ImageAnalysisTask.html
[`llmtask::Task`]: https://docs.rs/llmtask/latest/llmtask/task/trait.Task.html
[`llmtask::ImageAnalysis`]: https://docs.rs/llmtask/latest/llmtask/image_analysis/struct.ImageAnalysis.html
[`qwen3-vl`]: https://docs.rs/qwen3-vl
[llguidance]: https://github.com/microsoft/llguidance

## Why an `llmtask`-driven engine?

A bespoke `lfm::Task` would force every prompt + schema + parser to be rewritten against the next inference engine. Implementing [`llmtask::Task`] instead means the same `Task` code targets `lfm` (llguidance), [`qwen3-vl`] (mistralrs), or any future `llmtask`-compatible backend without modification — only the hardware backend selection differs.

```text
                                ┌──────────────────────────┐
   YourTask: impl Task   ──▶    │   llmtask::Task contract │   ──▶  lfm / qwen3-vl / …
                                │     prompt + Grammar     │
                                │     parse → Output       │
                                └──────────────────────────┘
```

Because lfm's backend is llguidance, all three [`llmtask::Grammar`] variants (JSON Schema, Lark, Regex) are accepted — engines that only speak JSON Schema (e.g. `qwen3-vl`) reject the others via `UnsupportedGrammar`, and the caller can route to lfm.

[`llmtask::Grammar`]: https://docs.rs/llmtask/latest/llmtask/grammar/enum.Grammar.html

## Features

- **All three `Grammar` variants** — JSON Schema, Lark, and Regex are all native to llguidance, so any `llmtask::Task` runs through `Engine::run`. The HIR-anchored regex validator on the `Grammar` side matches engine semantics exactly (no substring vs. full-match drift).
- **Bundled tokenizer + configs (`bundled` feature, default)** — `Engine::from_onnx_dir` accepts an ONNX-only directory; tokenizer / chat template / preprocessor configs are embedded in the binary at compile time. `Engine::from_dir` is the strict constructor that byte-validates a supplied tokenizer + chat template against the bundled blobs to catch silent prompt-envelope drift.
- **Hybrid KV/conv-state cache decoder** — LFM2 architecture has 10 conv-state layers and 6 attention layers at sparse indices. `decoder.rs` manages the non-contiguous cache layout transparently.
- **Wasm-friendly preprocessing** — drop the `inference` and `bundled` defaults to get a pure-CPU image-preprocessing surface (`Preprocessor`, `TileGrid`, EXIF-aware decode) usable from `wasm32-unknown-unknown`.
- **GPU acceleration** — `cuda`, `tensorrt`, `directml`, `rocm`, `coreml` ORT execution providers gated behind feature flags. None are required for CPU inference.
- **Admission-control DoS guards** — bounded request shape (max messages, max content parts), text-size cap, image-count lower bound from `min_image_tokens`, header-time decoded-buffer cap, and a special-token denylist seeded from the live tokenizer's `added_vocabulary`. All run BEFORE any image decode or template render.

## Example

### From a HuggingFace download (tokenizer.json + configs in dir)

```rust,no_run
use lfm::{
    ChatContent, ChatMessage, ContentPart, Engine, ImageInput, Options,
    RequestOptions,
};
use smol_str::SmolStr;

fn main() -> lfm::Result<()> {
    let model_dir = std::env::var("LFM_MODEL_PATH")
        .expect("set LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX");

    let mut engine = Engine::from_dir(&model_dir, Options::default())?;

    let messages = vec![ChatMessage {
        role: SmolStr::new_static("user"),
        content: ChatContent::Parts(vec![
            ContentPart::Image,
            ContentPart::Text("Describe this image.".into()),
        ]),
    }];
    let images = vec![ImageInput::Path(std::path::Path::new("photo.jpg"))];

    let text = engine.generate(&messages, &images, &RequestOptions::default())?;
    println!("{text}");
    Ok(())
}
```

### ONNX-only dir + bundled tokenizer

If you've downloaded just the ONNX files (not `tokenizer.json` and the JSON configs), use `Engine::from_onnx_dir`. The tokenizer + configs are embedded in the binary and written to a temp file on first use.

```rust,no_run
use lfm::{Engine, Options, RequestOptions};

fn main() -> lfm::Result<()> {
    let onnx_dir = std::env::var("LFM_ONNX_PATH")
        .expect("set LFM_ONNX_PATH=/path/with/onnx-files-only");
    let mut engine = Engine::from_onnx_dir(onnx_dir, Options::default())?;
    // … same usage as Engine::from_dir
    # let _ = engine; let _ = RequestOptions::default();
    Ok(())
}
```

### Structured output via the `ImageAnalysisTask` preset

```rust,no_run
use lfm::{
    ChatContent, ChatMessage, ContentPart, Engine, ImageAnalysisTask, ImageInput,
    Options, RequestOptions, Task,
};
use smol_str::SmolStr;

fn main() -> lfm::Result<()> {
    let model_dir = std::env::var("LFM_MODEL_PATH").unwrap();
    let mut engine = Engine::from_dir(&model_dir, Options::default())?;
    let task = ImageAnalysisTask::default();

    let messages = vec![ChatMessage {
        role: SmolStr::new_static("user"),
        content: ChatContent::Parts(vec![
            ContentPart::Image,
            ContentPart::Text(task.prompt().to_owned()),
        ]),
    }];
    let images = vec![ImageInput::Path(std::path::Path::new("frame.jpg"))];

    let analysis = engine.run(&task, &messages, &images, &RequestOptions::default())?;
    println!("{analysis:#?}");
    Ok(())
}
```

## Installation

```toml
[dependencies]
lfm = "0.3"
```

Download the ONNX artifacts from [`LiquidAI/LFM2.5-VL-450M-ONNX`][lfm-card] and set `LFM_MODEL_PATH` to the directory containing them:

```text
vision_encoder.onnx
embed_tokens.onnx
decoder_model_merged.onnx
tokenizer.json   (optional — bundled if absent and `bundled` feature is on)
```

### Cargo features

Defaults: `["inference", "bundled", "decoders"]`.

| Feature       | Default | What it adds                                                                                                       |
| ------------- | :-----: | ------------------------------------------------------------------------------------------------------------------ |
| `inference`   |   yes   | Pulls `ort`, `tokenizers`, `llguidance`, `minijinja`. Activates `Engine`. Native targets only.                     |
| `bundled`     |   yes   | Embeds `tokenizer.json` + JSON configs (~4.5 MB) at compile time; adds `Engine::from_onnx_dir`. Implies `inference`. |
| `decoders`    |   yes   | Activates JPEG/PNG decoding via the `image` crate.                                                                 |
| `serde`       |   no    | `Serialize`/`Deserialize` on the `Options` document and its tiers (`RequestOptions`, `ImageBudget`, `BackendOptions`, `OrtOptions`, `MlxOptions`, `ThreadOptions`). |
| `cuda`        |   no    | NVIDIA GPUs (Linux / Windows). Requires CUDA toolkit + cuDNN. Implies `inference`.                                 |
| `tensorrt`    |   no    | NVIDIA, optimized inference. Falls back to CUDA, then CPU. Implies `inference`.                                    |
| `directml`    |   no    | Windows GPUs (any vendor) via DirectX 12. Implies `inference`.                                                     |
| `rocm`        |   no    | AMD GPUs (Linux). Requires ROCm SDK. Implies `inference`.                                                          |
| `coreml`      |   no    | macOS / iOS via Core ML (Neural Engine + GPU + Metal). Implies `inference`.                                        |
| `integration` |   no    | Enables the integration test (`tests/integration.rs`). Requires `LFM_MODEL_PATH`.                                  |

GPU execution-provider features are off by default — none are required for CPU inference, and each requires its vendor SDK at build time.

### Wasm / preprocessing-only build

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features decoders
```

The public surface under `--no-default-features --features decoders` is `preproc::Preprocessor`, `preproc::TileGrid`, `preproc::PreprocessedImage`, `preproc::decode_bytes_with_orientation`, `options::*`, and `error::*`.

## Architecture

Per-image vision encoding → text+image embedding splice → hybrid KV/conv cache decoder loop → optional schema-constrained sampling.

| Graph                       | Role                                                            | Size       |
| --------------------------- | --------------------------------------------------------------- | ---------- |
| `vision_encoder.onnx`       | SigLIP2 image encoder — single image per call                   | ~86M params |
| `embed_tokens.onnx`         | Token embedding lookup table                                    | —          |
| `decoder_model_merged.onnx` | LFM2 hybrid LM: 10 conv-state + 6 KV-attn layers (sparse cache) | ~350M params |

The decoder manages a sparse hybrid cache: conv-state layers store recurrent state (not KV pairs), so cache indices are non-contiguous. Schema-constrained sampling is handled by `llguidance` masking the logits at each step to enforce the `Grammar` from the `Task`.

**Multi-image note:** the vision encoder accepts one image per call. Batched multi-image calls produce silently-wrong embeddings — `Engine::generate`/`run` iterate per-image and concatenate the flat `image_features` outputs in source order.

## MSRV

Rust 1.95.

## License

`lfm` is dual-licensed under the [MIT license](LICENSE-MIT) and the [Apache License, Version 2.0](LICENSE-APACHE).

The LFM2.5-VL model weights this crate runs are governed by the [LFM Open License v1.0](https://www.liquid.ai/lfm-license). Verify your use case complies with Liquid AI's terms separately from this crate's license.

Copyright (c) 2026 FinDIT Studio authors.

[lfm-card]: https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX

[Github-url]: https://github.com/findit-studio/lfm/
[CI-url]: https://github.com/findit-studio/lfm/actions/workflows/ci.yml
[doc-url]: https://docs.rs/lfm
[crates-url]: https://crates.io/crates/lfm
[codecov-url]: https://app.codecov.io/gh/findit-studio/lfm/
