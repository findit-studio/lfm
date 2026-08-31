use std::env::{self, var};

fn main() {
  // Don't rerun this on changes other than build.rs, as we only depend on
  // the rustc version.
  println!("cargo:rerun-if-changed=build.rs");

  // Check for `--features=tarpaulin`.
  let tarpaulin = var("CARGO_FEATURE_TARPAULIN").is_ok();

  if tarpaulin {
    use_feature("tarpaulin");
  } else {
    // Always rerun if these env vars change.
    println!("cargo:rerun-if-env-changed=CARGO_TARPAULIN");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARPAULIN");

    // Detect tarpaulin by environment variable
    if env::var("CARGO_TARPAULIN").is_ok() || env::var("CARGO_CFG_TARPAULIN").is_ok() {
      use_feature("tarpaulin");
    }
  }

  // Rerun this script if any of our features or configuration flags change,
  // or if the toolchain we used for feature detection changes.
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TARPAULIN");

  // `ort` is a mandatory dependency on every target except aarch64-macos
  // (see the target tables in Cargo.toml), where it is an optional
  // dependency gated behind the `ort` feature — MLX is the native road
  // there. Emit one `ort_backend` cfg so the ORT-touching code
  // (runtime/{session,vision,embed_tokens,decoder}.rs, the `OrtBackend` arm
  // in runtime/backend.rs, `Engine::from_paths` and friends, and the
  // ort-typed `Error` / `Options` items) gates on a single condition instead
  // of repeating the platform+feature compound at every site. `check-cfg`
  // for it is declared in Cargo.toml's `[lints.rust]` (same as `tarpaulin`
  // above).
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ORT");
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
  let aarch64_macos = target_os == "macos" && target_arch == "aarch64";
  let ort_feature = env::var("CARGO_FEATURE_ORT").is_ok();
  if !aarch64_macos || ort_feature {
    use_feature("ort_backend");
  }
}

fn use_feature(feature: &str) {
  println!("cargo:rustc-cfg={}", feature);
}
