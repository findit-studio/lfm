//! The `Options` document's shape, tier by tier.
//!
//! `Options` is one flat document with two tiers: the shared knobs on top and
//! the selected engine's own knobs flattened in under the `backend` tag. What
//! that buys is pinned here — every tier defaults, an unnamed road is refused
//! by the variant roster, and a key that belongs to no tier is refused by name
//! rather than dropped on the floor.
//!
//! The consumer formats are JSON, YAML and TOML; JSON and TOML are exercised
//! directly (YAML shares serde's self-describing map path with JSON, and the
//! crate carries no YAML dependency to test through).

use lfm::{AutoOptions, BackendKind, BackendOptions, ImageBudget, Options, RequestOptions};

#[cfg(all(feature = "inference", ort_backend))]
use lfm::{OrtOptions, ThreadOptions};

#[cfg(all(feature = "inference", mlx_backend))]
use lfm::MlxOptions;

/// The whole `expected ...` clause serde prints for this build's roster of
/// roads. One arm per reachable configuration: the engine tiers are `cfg`-gated
/// on their backend's presence, so the roster is a property of the build — and
/// serde spells a two-name roster `expected `a` or `b`` rather than
/// `expected one of ...`, which only appears from three names up.
#[cfg(not(feature = "inference"))]
const EXPECTED_ROSTER: &str = "expected `auto`";
#[cfg(all(feature = "inference", ort_backend, mlx_backend))]
const EXPECTED_ROSTER: &str = "expected one of `auto`, `onnx`, `mlx`";
#[cfg(all(feature = "inference", ort_backend, not(mlx_backend)))]
const EXPECTED_ROSTER: &str = "expected `auto` or `onnx`";
// `ort_backend` is false both on aarch64-macos without the `ort` feature
// (MLX is the only compiled road there) and, since `ort` is now an
// ALLOW-list rather than a deny-list (see Cargo.toml's target tables), on
// any other target outside that allow-list — there both roads are absent and
// only `auto` remains. The two arms below split exactly those two cases.
#[cfg(all(feature = "inference", not(ort_backend), mlx_backend))]
const EXPECTED_ROSTER: &str = "expected `auto` or `mlx`";
#[cfg(all(feature = "inference", not(ort_backend), not(mlx_backend)))]
const EXPECTED_ROSTER: &str = "expected `auto`";

fn keys(v: &serde_json::Value) -> Vec<String> {
  let mut k: Vec<String> = v
    .as_object()
    .expect("the document is a JSON object")
    .keys()
    .cloned()
    .collect();
  k.sort();
  k
}

// =========================================================================
// Every tier defaults
// =========================================================================

/// The complaint this shape answers: a partial table naming only a road used
/// to be refused by the first missing field's name. Now each tier fills from
/// its own `new()` — the shared tier from `Options::new()`, the engine tier
/// from that engine's own default.
#[test]
fn a_partial_table_fills_from_each_tiers_own_defaults() {
  let auto: Options = serde_json::from_str(r#"{"backend":"auto"}"#).expect("partial auto table");
  assert_eq!(auto, Options::new());
  assert_eq!(*auto.request(), RequestOptions::deterministic());
  assert_eq!(*auto.image_budget(), ImageBudget::new());
  assert_eq!(*auto.backend(), BackendOptions::auto());
  assert_eq!(*auto.backend(), BackendOptions::Auto(AutoOptions::new()));

  // A shared knob may be given alone; the engine tier still defaults.
  let one_knob: Options = serde_json::from_str(r#"{"backend":"auto","image_budget":{"min_image_tokens":32,"max_image_tokens":64,"min_tiles":2,"max_tiles":4,"use_thumbnail":false,"max_pixels_tolerance":2.0}}"#)
    .expect("partial shared tier");
  assert_eq!(*one_knob.image_budget(), ImageBudget::fast());
  assert_eq!(*one_knob.request(), RequestOptions::deterministic());
}

/// The bug this pins: `{ "request": { "max_new_tokens": 512 } }` used to be
/// refused at `temperature` — the first field the partial table left out —
/// instead of filling it. Each field left out of a partial `request` table
/// now comes from the SAME value an entirely absent `request` table already
/// produces, asserted here by comparing against that absent-table document
/// rather than against hard-coded numbers: if `RequestOptions`'s absent-table
/// value ever changes, this test does not have to.
#[test]
fn a_partial_request_table_fills_missing_keys_from_the_absent_table_value() {
  let absent: Options = serde_json::from_str(r#"{"backend":"auto"}"#).expect("absent table");
  let partial: Options =
    serde_json::from_str(r#"{"backend":"auto","request":{"max_new_tokens":128}}"#)
      .expect("partial request table");

  assert_eq!(
    partial.request().max_new_tokens(),
    128,
    "the given key took"
  );
  assert_eq!(
    partial.request().temperature(),
    absent.request().temperature(),
    "an omitted key must fill from the same value the absent table uses"
  );
  assert_eq!(partial.request().min_p(), absent.request().min_p());
  assert_eq!(
    partial.request().repetition_penalty(),
    absent.request().repetition_penalty()
  );
}

/// Same shape, `image_budget`: `ImageBudget`'s absent-table value and its
/// `Default` do not diverge the way `RequestOptions`'s do, but the partial
/// table only works at all because of the same `#[serde(default)]` this PR
/// adds — pin it the same way, against the absent-table document.
#[test]
fn a_partial_image_budget_table_fills_missing_keys_from_the_absent_table_value() {
  let absent: Options = serde_json::from_str(r#"{"backend":"auto"}"#).expect("absent table");
  let partial: Options =
    serde_json::from_str(r#"{"backend":"auto","image_budget":{"max_tiles":4}}"#)
      .expect("partial image_budget table");

  assert_eq!(partial.image_budget().max_tiles(), 4, "the given key took");
  assert_eq!(
    partial.image_budget().min_image_tokens(),
    absent.image_budget().min_image_tokens()
  );
  assert_eq!(
    partial.image_budget().max_image_tokens(),
    absent.image_budget().max_image_tokens()
  );
  assert_eq!(
    partial.image_budget().min_tiles(),
    absent.image_budget().min_tiles()
  );
  assert_eq!(
    partial.image_budget().use_thumbnail(),
    absent.image_budget().use_thumbnail()
  );
  assert_eq!(
    partial.image_budget().max_pixels_tolerance(),
    absent.image_budget().max_pixels_tolerance()
  );
}

/// A FULL `request` table — every key given explicitly, none left to
/// `#[serde(default)]` — must still round-trip exactly, unaffected by the
/// new attribute.
#[test]
fn a_full_request_table_round_trips() {
  let json = r#"{"backend":"auto","request":{"temperature":0.3,"min_p":0.05,"repetition_penalty":1.1,"max_new_tokens":1024}}"#;
  let opts: Options = serde_json::from_str(json).expect("full request table");
  assert_eq!(opts.request().temperature(), 0.3);
  assert_eq!(opts.request().min_p(), 0.05);
  assert_eq!(opts.request().repetition_penalty(), 1.1);
  assert_eq!(opts.request().max_new_tokens(), 1024);

  let back_json = serde_json::to_string(&opts).expect("serialize");
  let back: Options = serde_json::from_str(&back_json).expect("round-trip");
  assert_eq!(back, opts);
}

#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn a_partial_onnx_table_fills_the_ort_tier() {
  let opts: Options = serde_json::from_str(r#"{"backend":"onnx"}"#).expect("partial onnx table");
  assert_eq!(opts.backend_kind(), Some(BackendKind::Onnx));
  assert_eq!(
    opts.backend().ort_options().copied(),
    Some(OrtOptions::new())
  );
  assert_eq!(
    opts,
    Options::new().with_backend(BackendOptions::onnx(OrtOptions::new()))
  );

  // One engine knob given, the rest of that tier defaulted.
  let tuned: Options =
    serde_json::from_str(r#"{"backend":"onnx","thread":{"intra_threads":1,"inter_threads":1}}"#)
      .expect("partial ort tier");
  let ort = tuned.backend().ort_options().expect("the onnx tier");
  assert_eq!(*ort.thread(), ThreadOptions::deterministic());
  assert_eq!(*ort, OrtOptions::deterministic());
}

#[cfg(all(feature = "inference", mlx_backend))]
#[test]
fn a_partial_mlx_table_fills_the_mlx_tier() {
  let opts: Options = serde_json::from_str(r#"{"backend":"mlx"}"#).expect("partial mlx table");
  assert_eq!(opts.backend_kind(), Some(BackendKind::Mlx));
  assert_eq!(
    opts.backend().mlx_options().copied(),
    Some(MlxOptions::new())
  );
  assert_eq!(
    opts,
    Options::new().with_backend(BackendOptions::mlx(MlxOptions::new()))
  );
}

// =========================================================================
// The roster is the refusal
// =========================================================================

/// A road that is not in this build's roster is refused by serde's own
/// unknown-variant message, and the roster it prints is exactly the set of
/// roads this build compiled — which is why the engine tiers are `cfg`-gated
/// on their backend's presence rather than always present and empty.
#[test]
fn an_uncompiled_road_is_refused_by_the_variant_roster() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"trt"}"#)
    .expect_err("a road that is not in the roster must be refused");
  let expected = format!("unknown variant `trt`, {EXPECTED_ROSTER}");
  assert!(
    err.to_string().starts_with(&expected),
    "\n  expected: {expected}\n  actual:   {err}"
  );
}

/// On aarch64-macos without the `ort` feature, MLX is the native road and the
/// ONNX one is not compiled — so `onnx` is not merely unavailable at load
/// time, it is not a spellable value of the document.
#[cfg(all(feature = "inference", not(ort_backend), mlx_backend))]
#[test]
fn onnx_is_not_in_the_roster_without_the_ort_feature() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"onnx"}"#)
    .expect_err("this build compiles no ONNX road");
  assert!(
    err
      .to_string()
      .starts_with("unknown variant `onnx`, expected `auto` or `mlx`"),
    "unexpected refusal: {err}"
  );
}

/// Off Apple Silicon there is no `mlxrs`, so `mlx` is not a spellable value.
#[cfg(all(feature = "inference", not(mlx_backend)))]
#[test]
fn mlx_is_not_in_the_roster_off_apple_silicon() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"mlx"}"#)
    .expect_err("this build compiles no MLX road");
  assert!(
    err.to_string().starts_with("unknown variant `mlx`"),
    "unexpected refusal: {err}"
  );
}

// =========================================================================
// A key that belongs to no tier is refused, never dropped
// =========================================================================

/// A misspelled engine knob is refused by the variant struct's
/// `deny_unknown_fields`, which lists that engine's fields.
#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn a_misspelled_engine_knob_is_refused() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"onnx","optimisation_level":"level1"}"#)
    .expect_err("a misspelled ORT knob must be refused");
  assert!(
    err
      .to_string()
      .starts_with("unknown field `optimisation_level`, expected `thread` or `optimization_level`"),
    "unexpected refusal: {err}"
  );
}

/// A misspelled *shared* knob falls through the flatten into the engine tier
/// and is refused there. The message therefore lists the engine tier's fields
/// rather than the shared tier's — the key is refused, which is the point.
#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn a_misspelled_shared_knob_falls_through_and_is_refused() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"onnx","reqeust":{}}"#)
    .expect_err("a misspelled shared knob must be refused");
  assert!(
    err
      .to_string()
      .starts_with("unknown field `reqeust`, expected `thread` or `optimization_level`"),
    "unexpected refusal: {err}"
  );
}

/// An engine knob written beside the *other* road is refused rather than
/// accepted and silently ignored.
#[cfg(all(feature = "inference", mlx_backend))]
#[test]
fn an_ort_knob_under_the_mlx_road_is_refused() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"mlx","optimization_level":"level1"}"#)
    .expect_err("an ORT knob has no meaning on the MLX road");
  assert!(
    err
      .to_string()
      .starts_with("unknown field `optimization_level`, there are no fields"),
    "unexpected refusal: {err}"
  );
}

/// `auto` names no engine, so it accepts no engine knob — and, because its
/// tier is a zero-field struct with `deny_unknown_fields` rather than a bare
/// unit variant, it refuses one by name. A unit variant would have absorbed
/// the key silently: that is the trap this shape is built to avoid.
#[test]
fn the_auto_road_refuses_a_stray_key_instead_of_absorbing_it() {
  let err = serde_json::from_str::<Options>(r#"{"backend":"auto","intra_threads":1}"#)
    .expect_err("auto carries no engine knobs");
  assert!(
    err
      .to_string()
      .starts_with("unknown field `intra_threads`, there are no fields"),
    "unexpected refusal: {err}"
  );
}

/// Serde's flatten cannot supply a missing tag: naming the road is mandatory,
/// and `"auto"` is how a document defers the choice to the checkpoint layout.
#[test]
fn the_backend_key_is_required() {
  for doc in [
    r#"{}"#,
    r#"{"request":{"temperature":0.0,"min_p":0.0,"repetition_penalty":1.05,"max_new_tokens":512}}"#,
  ] {
    let err = serde_json::from_str::<Options>(doc).expect_err("the road must be named");
    assert!(
      err.to_string().starts_with("missing field `backend`"),
      "unexpected refusal for {doc}: {err}"
    );
  }
}

// =========================================================================
// Document form: what the bytes look like, per road
// =========================================================================

/// The auto road's document carries the shared tier and the tag, nothing else.
#[test]
fn the_auto_document_is_the_shared_tier_plus_the_tag() {
  let value = serde_json::to_value(Options::new()).expect("serialize");
  assert_eq!(keys(&value), ["backend", "image_budget", "request"]);
  assert_eq!(value["backend"], serde_json::json!("auto"));
  let back: Options = serde_json::from_value(value).expect("round-trip");
  assert_eq!(back, Options::new());
}

/// The ONNX road's document: the shared tier, the tag, then that road's own
/// knobs flattened in beside them — the same top-level key set the pre-tier
/// document had, with `backend` now naming the tier that owns `thread` and
/// `optimization_level`.
#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn the_onnx_document_form_is_pinned() {
  let opts = Options::new().with_backend(BackendOptions::onnx(OrtOptions::new()));
  let json = serde_json::to_string(&opts).expect("serialize");
  let expected = format!(
    r#"{{"request":{},"image_budget":{},"backend":"onnx","thread":{},"optimization_level":"level1"}}"#,
    serde_json::to_string(&RequestOptions::deterministic()).expect("request"),
    serde_json::to_string(&ImageBudget::new()).expect("image_budget"),
    serde_json::to_string(&ThreadOptions::new()).expect("thread"),
  );
  assert_eq!(json, expected);
  assert_eq!(
    keys(&serde_json::to_value(opts).expect("value")),
    [
      "backend",
      "image_budget",
      "optimization_level",
      "request",
      "thread"
    ]
  );
  let back: Options = serde_json::from_str(&json).expect("round-trip");
  assert_eq!(back, opts);
}

/// The MLX road's document: the shared tier and the tag. The MLX tier has no
/// knobs yet, so it contributes no keys.
#[cfg(all(feature = "inference", mlx_backend))]
#[test]
fn the_mlx_document_form_is_pinned() {
  let opts = Options::new().with_backend(BackendOptions::mlx(MlxOptions::new()));
  let json = serde_json::to_string(&opts).expect("serialize");
  let expected = format!(
    r#"{{"request":{},"image_budget":{},"backend":"mlx"}}"#,
    serde_json::to_string(&RequestOptions::deterministic()).expect("request"),
    serde_json::to_string(&ImageBudget::new()).expect("image_budget"),
  );
  assert_eq!(json, expected);
  let back: Options = serde_json::from_str(&json).expect("round-trip");
  assert_eq!(back, opts);
}

/// TOML is a consumer format too: the flattened tag is emitted as a scalar
/// before the shared tier's tables (TOML forbids the other order), and a full
/// document round-trips through it.
#[test]
fn the_document_round_trips_through_toml() {
  #[allow(unused_mut)]
  let mut roads = vec![Options::new()];
  #[cfg(all(feature = "inference", ort_backend))]
  roads.push(Options::new().with_backend(BackendOptions::onnx(OrtOptions::deterministic())));
  #[cfg(all(feature = "inference", mlx_backend))]
  roads.push(Options::new().with_backend(BackendOptions::mlx(MlxOptions::new())));

  for opts in roads {
    let text = toml::to_string(&opts).expect("serialize to TOML");
    assert!(
      text.starts_with("backend = "),
      "the tag must precede the shared tier's tables:\n{text}"
    );
    let back: Options = toml::from_str(&text).expect("round-trip through TOML");
    assert_eq!(back, opts);
  }
}

/// A partial TOML table fills from each tier's defaults, exactly as the JSON
/// one does.
#[test]
fn a_partial_toml_table_fills_from_each_tiers_defaults() {
  let opts: Options = toml::from_str("backend = \"auto\"\n").expect("partial TOML table");
  assert_eq!(opts, Options::new());
}

// =========================================================================
// One vocabulary
// =========================================================================

/// A backend is spelled the same way in the document, in `BackendKind`'s wire
/// form, and in the diagnostics `as_str` feeds.
#[test]
fn the_wire_name_is_the_backends_own_name() {
  for kind in [BackendKind::Onnx, BackendKind::Mlx] {
    let wire = serde_json::to_string(&kind).expect("serialize BackendKind");
    assert_eq!(wire, format!(r#""{}""#, kind.as_str()));
    assert_eq!(wire, format!(r#""{kind}""#));
  }
}

// =========================================================================
// Strictness does not recurse: every nested table carries its own
// =========================================================================

/// A tier's `deny_unknown_fields` sees only its own direct fields, never
/// inside a nested table. So every nested, non-flattened table of the document
/// carries the attribute itself — otherwise a misspelled key in one of them is
/// dropped and the caller silently gets the default it never asked for.
fn refused_by_both_formats(json: &str, toml_text: &str, expected: &str) {
  let json_err = serde_json::from_str::<Options>(json).expect_err("JSON must refuse");
  assert!(
    json_err.to_string().starts_with(expected),
    "\n  expected: {expected}\n  JSON:     {json_err}"
  );
  let toml_err = toml::from_str::<Options>(toml_text).expect_err("TOML must refuse");
  assert!(
    toml_err.to_string().contains(expected),
    "\n  expected: {expected}\n  TOML:     {toml_err}"
  );
}

/// The exact document the reviewer named: both `ThreadOptions` fields are
/// `Option`, so without the attribute this deserializes with *both* counts
/// silently left at `None`.
#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn a_misspelled_key_in_the_thread_table_is_refused() {
  refused_by_both_formats(
    r#"{"backend":"onnx","thread":{"intra_thread":1}}"#,
    "backend = \"onnx\"\n\n[thread]\nintra_thread = 1\n",
    "unknown field `intra_thread`, expected `intra_threads` or `inter_threads`",
  );
}

/// Same class, the sampler's table.
#[test]
fn a_misspelled_key_in_the_request_table_is_refused() {
  refused_by_both_formats(
    r#"{"backend":"auto","request":{"temperature":0.0,"min_p":0.0,"repetition_penalty":1.05,"max_new_tokens":512,"temperatur":0.5}}"#,
    "backend = \"auto\"\n\n[request]\ntemperature = 0.0\nmin_p = 0.0\nrepetition_penalty = 1.05\nmax_new_tokens = 512\ntemperatur = 0.5\n",
    "unknown field `temperatur`, expected one of `temperature`, `min_p`, `repetition_penalty`, `max_new_tokens`",
  );
}

/// Same class, the image budget's table.
#[test]
fn a_misspelled_key_in_the_image_budget_table_is_refused() {
  refused_by_both_formats(
    r#"{"backend":"auto","image_budget":{"min_image_tokens":64,"max_image_tokens":256,"min_tiles":2,"max_tiles":10,"use_thumbnail":true,"max_pixels_tolerance":2.0,"min_tile":1}}"#,
    "backend = \"auto\"\n\n[image_budget]\nmin_image_tokens = 64\nmax_image_tokens = 256\nmin_tiles = 2\nmax_tiles = 10\nuse_thumbnail = true\nmax_pixels_tolerance = 2.0\nmin_tile = 1\n",
    "unknown field `min_tile`, expected one of `min_image_tokens`, `max_image_tokens`, `min_tiles`, `max_tiles`, `use_thumbnail`, `max_pixels_tolerance`",
  );
}

/// The thread counts are `u16`, so "a value ORT can actually take" is a
/// property of the type: an out-of-range count is refused by serde's own
/// integer range check, in every format, with no ceiling constant for this
/// crate to invent, document and enforce. 65 535 is accepted; one more is not.
#[cfg(all(feature = "inference", ort_backend))]
#[test]
fn a_thread_count_beyond_the_fields_type_is_refused() {
  let accepted: Options =
    serde_json::from_str(r#"{"backend":"onnx","thread":{"intra_threads":65535}}"#)
      .expect("the largest representable count is accepted");
  assert_eq!(
    accepted
      .backend()
      .ort_options()
      .expect("the onnx tier")
      .thread()
      .intra_threads(),
    Some(u16::MAX)
  );

  let json_err =
    serde_json::from_str::<Options>(r#"{"backend":"onnx","thread":{"intra_threads":65536}}"#)
      .expect_err("JSON must refuse a count the type cannot hold");
  assert!(
    json_err
      .to_string()
      .contains("invalid value: integer `65536`"),
    "JSON: {json_err}"
  );
  let toml_err =
    toml::from_str::<Options>("backend = \"onnx\"\n\n[thread]\nintra_threads = 65536\n")
      .expect_err("TOML must refuse a count the type cannot hold");
  // Same refusal, same words. TOML's span points at the document's start
  // rather than at the key, because the flatten buffers the whole map before
  // the engine tier ever sees it — the refusal is precise about the value and
  // the type it must fit, not about where it was written. Neither format names
  // the field here; do not "fix" this by expecting one to.
  assert!(
    toml_err
      .to_string()
      .contains("invalid value: integer `65536`, expected u16"),
    "TOML: {toml_err}"
  );

  // The same bound applies to the inter-op count, which is the one that now
  // also selects an execution mode.
  assert!(
    serde_json::from_str::<Options>(r#"{"backend":"onnx","thread":{"inter_threads":65536}}"#)
      .is_err()
  );
}

// =========================================================================
// The ORT/MLX platform gates: Cargo.toml and build.rs name the same targets
// =========================================================================

/// `Cargo.toml`'s target tables and `build.rs`'s target checks are
/// independent spellings of the same EXACT target-triple sets, on BOTH
/// sides of the ORT/MLX split (D9, USER-SEALED 2026-09-06:
/// `aarch64-pc-windows-msvc` joined the mandatory allow-list; two later
/// rounds moved first the mandatory side, then the `aarch64-apple-darwin`
/// opt-in side, from `(arch, os, env)` `cfg(...)` predicates to exact
/// triples — several real, distinct targets share a cfg tuple with an
/// intended one while sitting outside `ort-sys`'s prebuilt-binary roster:
/// `aarch64_be-unknown-linux-gnu`, `aarch64-oe-linux-gnu`,
/// `aarch64-uwp-windows-msvc`, `x86_64-win7-windows-msvc` on the mandatory
/// side, and `arm64e-apple-darwin` — sharing `aarch64-apple-darwin`'s own
/// `(target_arch, target_os)` pair, with `target_vendor` "apple" on both —
/// on the opt-in side). This test parses Cargo.toml's actual TOML structure
/// (not text-scanned — a target-table key is just a string to the parser,
/// `cfg(...)` syntax included, so only structure tells a mandatory row from
/// the `aarch64-apple-darwin` optional one, via the `optional` key) and
/// build.rs's literal target strings, and asserts each side's sets are
/// string-equal: if either file is ever edited to add, drop, or misspell a
/// target without updating the other, this fails instead of silently
/// reintroducing the ort-sys build failure these exact-triple gates exist
/// to prevent.
#[test]
fn ort_allow_list_matches_the_manifest_target_table() {
  use std::collections::BTreeSet;

  const ALLOW_LIST: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "aarch64-pc-windows-msvc",
  ];
  let expected: BTreeSet<&str> = ALLOW_LIST.iter().copied().collect();

  // ----- Cargo.toml: parse it, don't scan its text -----
  // `toml::Table` (not `toml::Value`), because a `Value`'s own `FromStr`
  // parses a single inline value expression, not a whole multi-table
  // document — a document's root is itself a table, which is exactly what
  // `Table::from_str` (aliasing `toml::from_str`) parses.
  let manifest_text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
    .expect("read Cargo.toml");
  let manifest: toml::Table = manifest_text.parse().expect("parse Cargo.toml as TOML");
  let target_table = manifest
    .get("target")
    .and_then(toml::Value::as_table)
    .expect("Cargo.toml has a [target] table");

  // A row is the mandatory allow-list's if it carries an `ort` dependency
  // that is NOT `optional = true` — that is exactly what distinguishes the
  // four exact-triple rows from the aarch64-macos opt-in row (whose key,
  // being `cfg(...)` syntax, is otherwise just another string to a TOML
  // parser; nothing about the KEY shape marks it optional).
  fn ort_dep_is_mandatory(deps: &toml::Value) -> bool {
    let Some(ort) = deps.get("ort") else {
      return false;
    };
    match ort {
      toml::Value::Table(t) => t.get("optional").and_then(toml::Value::as_bool) != Some(true),
      // A bare version-string dependency (`ort = "2"`) is never optional.
      _ => true,
    }
  }

  let manifest_mandatory: BTreeSet<&str> = target_table
    .iter()
    .filter(|(_, table)| table.get("dependencies").is_some_and(ort_dep_is_mandatory))
    .map(|(key, _)| key.as_str())
    .collect();
  assert_eq!(
    manifest_mandatory, expected,
    "Cargo.toml's mandatory (non-optional) `ort` target tables do not name \
     the allow-list — got {manifest_mandatory:?}, expected {expected:?}. \
     Each allow-listed target needs its own exact-triple \
     `[target.<triple>.dependencies]` table (not a `cfg(...)` one — see the \
     comment above the four rows in Cargo.toml)."
  );

  // ----- build.rs: extract ORT_MANDATORY_TARGETS's literal strings -----
  let build_rs = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
    .expect("read build.rs");
  let marker = "const ORT_MANDATORY_TARGETS: &[&str] = &[";
  let start = build_rs
    .find(marker)
    .expect("build.rs defines ORT_MANDATORY_TARGETS")
    + marker.len();
  let end = build_rs[start..]
    .find("];")
    .expect("ORT_MANDATORY_TARGETS's array literal is closed with `];`");
  let array_body = &build_rs[start..start + end];
  let build_rs_targets: BTreeSet<&str> = array_body
    .split(',')
    .map(str::trim)
    .filter(|s| !s.is_empty())
    .map(|s| {
      s.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or_else(|| panic!("ORT_MANDATORY_TARGETS entry is not a plain string literal: {s}"))
    })
    .collect();
  assert_eq!(
    build_rs_targets, expected,
    "build.rs's ORT_MANDATORY_TARGETS does not name the allow-list — got \
     {build_rs_targets:?}, expected {expected:?}."
  );

  // ----- The aarch64-apple-darwin OPT-IN row: exact triple, not cfg(...) -----
  // Same class of gap Codex found on the mandatory side, relocated to the
  // MLX carve-out: `cfg(target_arch = "aarch64", target_os = "macos")` also
  // matches `arm64e-apple-darwin` (Apple's pointer-authentication ABI
  // variant — `target_arch` normalizes it to plain "aarch64", and
  // `target_vendor` is "apple" for both, so no cfg key tells them apart),
  // for which `ort-sys` has no distribution either. Both `ort` (optional)
  // and `mlxrs` belong under one exact-triple table for the same reason.
  const MLX_TARGET: &str = "aarch64-apple-darwin";
  assert!(
    !expected.contains(MLX_TARGET),
    "{MLX_TARGET} must not also be one of the mandatory ORT_MANDATORY_TARGETS \
     — it is the opt-in row, not an allow-list entry"
  );

  let mlx_row = target_table
    .get(MLX_TARGET)
    .and_then(toml::Value::as_table)
    .unwrap_or_else(|| {
      panic!(
        "Cargo.toml has no exact-triple [target.{MLX_TARGET}.dependencies] \
       table — the MLX/ORT opt-in row must be a bare exact-triple key, not \
       a `cfg(...)` one"
      )
    });
  let mlx_deps = mlx_row
    .get("dependencies")
    .and_then(toml::Value::as_table)
    .unwrap_or_else(|| panic!("[target.{MLX_TARGET}.dependencies] has no dependencies table"));
  assert!(
    mlx_deps.contains_key("mlxrs"),
    "[target.{MLX_TARGET}.dependencies] does not carry `mlxrs`"
  );
  let ort_optional_flag = mlx_deps
    .get("ort")
    .and_then(toml::Value::as_table)
    .and_then(|t| t.get("optional"))
    .and_then(toml::Value::as_bool);
  assert_eq!(
    ort_optional_flag,
    Some(true),
    "[target.{MLX_TARGET}.dependencies].ort must be `optional = true` (it \
     is the opt-in row, distinct from the mandatory allow-list rows above)"
  );

  // build.rs's `mlx_backend` must compare against this SAME exact string.
  let mlx_marker = "let mlx_backend = ";
  let mlx_start = build_rs
    .find(mlx_marker)
    .expect("build.rs defines mlx_backend")
    + mlx_marker.len();
  let mlx_end = build_rs[mlx_start..]
    .find(';')
    .expect("the mlx_backend assignment ends with `;`");
  let mlx_expr: String = build_rs[mlx_start..mlx_start + mlx_end]
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ");
  assert_eq!(
    mlx_expr,
    format!("target == {MLX_TARGET:?}"),
    "build.rs's `mlx_backend` does not compare TARGET against the exact \
     triple {MLX_TARGET:?} — got `{mlx_expr}`"
  );
}
