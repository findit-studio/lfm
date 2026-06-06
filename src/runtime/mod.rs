//! ORT-backed runtime modules. Gated on `feature = "inference"`.

// The Backend seam drives the per-image vision encode + splice, which
// decodes images via the `decoders`-gated helpers; gate it on the same
// feature set as `generate` / `Engine` (inference + decoders).
#[cfg(feature = "decoders")]
pub(crate) mod backend;
pub(crate) mod decoder;
pub(crate) mod embed_tokens;
pub(crate) mod sampler;
pub(crate) mod session;
pub(crate) mod vision;
