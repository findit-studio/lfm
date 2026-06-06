//! ORT-backed runtime modules. Gated on `feature = "inference"`.

// The Backend seam drives the per-image vision encode + splice, which
// decodes images via the `decoders`-gated helpers; gate it on the same
// feature set as `generate` / `Engine` (inference + decoders).
#[cfg(feature = "decoders")]
pub(crate) mod backend;
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
