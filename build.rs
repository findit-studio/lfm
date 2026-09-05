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

  // `ort` is a mandatory dependency ONLY on the ALLOW-listed targets below
  // (see the target tables in Cargo.toml, which cites the `ort-sys`
  // prebuilt-binary roster this list was census'd from) and an optional one,
  // gated behind the `ort` feature, on `aarch64-apple-darwin` exactly — MLX
  // is the native road there. Every other target (`arm64e-apple-darwin`,
  // `x86_64-apple-darwin` and wasm32 included) has no `ort` dependency at
  // all: no backend by default, and no feature to opt in with either, since
  // `ort-sys` ships no prebuilt binary for them. Emit one `ort_backend` cfg
  // so the ORT-touching code (runtime/{session,vision,embed_tokens,decoder}.rs,
  // the `OrtBackend` arm in runtime/backend.rs, `Engine::from_paths` and
  // friends, and the ort-typed `Error` / `Options` items) gates on a single
  // condition instead of repeating the platform+feature compound at every
  // site. `check-cfg` for it is declared in Cargo.toml's `[lints.rust]`
  // (same as `tarpaulin` above).
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ORT");
  // Every target gate in this file compares the EXACT target triple
  // (`TARGET`, the full string Cargo always sets for build scripts — e.g.
  // "x86_64-unknown-linux-gnu" or "aarch64-apple-darwin"), never a
  // `(target_arch, target_os, target_env, ...)` cfg tuple. A cfg tuple does
  // not uniquely name a target: `aarch64-apple-darwin`'s own
  // `(target_arch, target_os)` pair — "aarch64", "macos" — is ALSO
  // `arm64e-apple-darwin`'s (`target_arch` normalizes Apple's
  // pointer-authentication ABI variant down to plain "aarch64"; `target_vendor`
  // is "apple" for both, so no cfg key distinguishes them). Enabling the
  // `ort` feature on `arm64e-apple-darwin` under an `(arch, os)`-based check
  // would wrongly turn `ort_backend` on and `ort-sys` — which, like every
  // check in this file, keys its prebuilt-binary roster on the exact target
  // string — has no distribution for that target either. The same reasoning
  // is why the ORT allow-list below is four exact strings rather than a
  // `(arch, os, env)` triple: this file's whole discipline is exact-target
  // matching, everywhere, because Cargo target cfg tuples are inherently
  // many-to-one over real target triples.
  let target = env::var("TARGET").unwrap_or_default();

  // The allow-list: exactly the targets `ort-sys` 2.0.0-rc.13 ships a
  // prebuilt binary for that this crate supports as a desktop/server
  // target (see Cargo.toml's matching exact-triple `[target.<triple>.dependencies]`
  // tables, which cite the `ort-sys` roster this list was census'd from).
  // `tests/options_document.rs`'s `ort_allow_list_matches_the_manifest_target_table`
  // parses those tables and asserts they name this same set; update both
  // together.
  const ORT_MANDATORY_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "aarch64-pc-windows-msvc",
  ];
  let ort_allow_listed = ORT_MANDATORY_TARGETS.contains(&target.as_str());

  // The ONE target MLX (`mlxrs`) compiles on: `aarch64-apple-darwin`
  // exactly, not `arm64e-apple-darwin` (`ort-sys` has no distribution for
  // it — see above — and this crate has never claimed `mlxrs` supports it
  // either). Cargo.toml's `mlxrs` and opt-in `ort` rows both live under one
  // `[target.aarch64-apple-darwin.dependencies]` table, matching this exact
  // string. `mlx_backend` is the ONE flag every MLX-touching product-code
  // site gates on — `runtime/mlx_backend.rs`'s inclusion, the `Mlx` variants
  // of `BackendImpl`/`EngineEmbeds`/`EngineCache`/`BackendOptions`, `MlxOptions`,
  // the `Error` variants `ort`'s absence would otherwise leave ungated, and
  // every `Engine::from_mlx_*` constructor — instead of repeating
  // `target_os = "macos", target_arch = "aarch64"` (and risking exactly the
  // arm64e gap above) at each site. `check-cfg` for it is declared in
  // Cargo.toml's `[lints.rust]` (same as `ort_backend`/`backend_available`).
  let mlx_backend = target == "aarch64-apple-darwin";
  if mlx_backend {
    use_feature("mlx_backend");
  }

  let ort_feature = env::var("CARGO_FEATURE_ORT").is_ok();
  let ort_backend = ort_allow_listed || (mlx_backend && ort_feature);
  if ort_backend {
    use_feature("ort_backend");
  }

  // `backend_available` is true exactly when this build compiles at least one
  // of the two `Backend` implementors (`OrtBackend` behind `ort_backend`,
  // `MlxBackend` behind `mlx_backend`). It exists because `BackendImpl`
  // (runtime/backend.rs) is a `pub(crate)` enum whose variants are each
  // individually `#[cfg]`-gated on exactly those two conditions: when NEITHER
  // holds (e.g. `x86_64-apple-darwin` with the `inference` feature on — the
  // allow-list conversion above made this a real, reachable build
  // configuration, not merely a hypothetical one), the enum has zero
  // variants. Rust's exhaustiveness checker treats a REFERENCE as always
  // inhabited regardless of the referent's own inhabitedness (`&BackendImpl`
  // is never itself "empty" to the borrow checker, only `BackendImpl` is),
  // so `impl Backend for BackendImpl`'s `match self { <zero cfg'd-in arms> }`
  // bodies would be refused as non-exhaustive (E0004) rather than silently
  // accepted as dead code. `runtime/backend.rs` adds one `#[cfg(not(backend_available))]`
  // fallback arm per method precisely so those matches keep exactly one arm
  // (real or fallback) under every build configuration. `engine.rs` uses the
  // same flag to give the "no backend at all" diagnostic its own accurate
  // text instead of reusing the `mlx_backend`-specific one. `check-cfg` for
  // it is declared in Cargo.toml's `[lints.rust]` (same as `ort_backend`).
  if ort_backend || mlx_backend {
    use_feature("backend_available");
  }
}

fn use_feature(feature: &str) {
  println!("cargo:rustc-cfg={}", feature);
}
