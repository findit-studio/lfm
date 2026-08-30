//! ORT-backed runtime modules. Gated on `feature = "inference"`.

// The Backend seam drives the per-image vision encode + splice, which
// decodes images via the `decoders`-gated helpers; gate it on the same
// feature set as `generate` / `Engine` (inference + decoders).
#[cfg(feature = "decoders")]
pub(crate) mod backend;
// Checkpoint-layout detection is the selector `Engine::from_dir` routes on. It
// is platform-independent (a non-macOS host must still be able to say "this is
// an MLX checkpoint and this platform has no MLX backend" rather than reporting
// a missing ONNX graph), so it lives beside the `backend` seam under the same
// gate rather than inside the macOS-only `mlx_backend`.
#[cfg(feature = "decoders")]
pub(crate) mod checkpoint;
pub(crate) mod decoder;
pub(crate) mod embed_tokens;
// The MLX (mlxrs) backend is the Apple-Silicon-only on-device alternative to
// the ORT path. It is compiled only on macOS/arm64 (where the `mlxrs` target
// dependency exists) and under the same `decoders` gate the `backend` seam
// lives under. There is intentionally no `mlx` Cargo feature — see Cargo.toml.
#[cfg(all(feature = "decoders", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod mlx_backend;
pub(crate) mod sampler;
pub(crate) mod session;
pub(crate) mod vision;
