//! Runtime modules. The whole tree is gated on `feature = "inference"`; the
//! ORT-specific submodules (`decoder`, `embed_tokens`, `session`, `vision`,
//! and the `Ort` arm of `backend`) are additionally gated on `ort_backend` —
//! `ort` is mandatory on the ALLOW-listed targets (Linux x86_64/aarch64-gnu,
//! Windows x86_64/aarch64-msvc) and optional on aarch64-apple-darwin behind the `ort`
//! feature (MLX is the native road there); every other target compiles
//! neither.

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
// The ORT-backed component wrappers. `ort` is mandatory on the ALLOW-listed
// targets (Linux x86_64/aarch64-gnu, Windows x86_64/aarch64-msvc) and optional on
// aarch64-apple-darwin behind the `ort` feature; every other target compiles
// neither. See the `ort_backend` cfg emitted by build.rs and the target
// tables in Cargo.toml.
#[cfg(ort_backend)]
pub(crate) mod decoder;
#[cfg(ort_backend)]
pub(crate) mod embed_tokens;
// The MLX (mlxrs) backend is the Apple-Silicon-only on-device alternative to
// the ORT path. It is compiled only on `aarch64-apple-darwin` exactly — not
// `arm64e-apple-darwin`, which shares that target's (target_arch, target_os)
// cfg pair but has no `mlxrs` dependency row either (see build.rs's
// `mlx_backend` cfg and Cargo.toml's exact-triple target table) — and under
// the same `decoders` gate the `backend` seam lives under. There is
// intentionally no `mlx` Cargo feature — see Cargo.toml.
#[cfg(all(feature = "decoders", mlx_backend))]
pub(crate) mod mlx_backend;
pub(crate) mod sampler;
#[cfg(ort_backend)]
pub(crate) mod session;
#[cfg(ort_backend)]
pub(crate) mod vision;
