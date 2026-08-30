//! Checkpoint-layout detection — the observable half of backend selection.
//!
//! [`Engine::from_dir`](crate::Engine) has to answer one question before it can
//! load anything: *which backend does this directory describe?* That answer used
//! to be a bare `bool`, which left three states unrepresentable — a directory
//! whose only MLX weight file is in a format this build disabled, a directory
//! with a half-present ONNX graph set beside an MLX weight set, and a directory
//! that is recognizably an MLX checkpoint on a platform with no MLX backend.
//! All three fell through to the ONNX path and surfaced as an unrelated
//! "missing `onnx/vision_encoder.onnx`".
//!
//! [`detect`] returns the classification instead, so each of those becomes a
//! named error at the point of decision, and the decision itself is reportable
//! through [`Engine::backend`](crate::Engine::backend).
//!
//! Detection is pure filesystem existence checks — no file is opened. The
//! winning constructor does the real load and surfaces a typed error if the
//! checkpoint is malformed.

use std::path::Path;

/// The ONNX graphs [`Engine::from_dir`](crate::Engine) loads, relative to the
/// model directory. Their presence is the documented disambiguator between an
/// ONNX checkpoint and an MLX one, because `config.json` + `model.safetensors`
/// are ALSO the standard HuggingFace source-asset names: an export that ships
/// those next to its graphs is an ONNX checkpoint.
pub(crate) const REQUIRED_ONNX_GRAPHS: [&str; 3] = [
  "onnx/vision_encoder.onnx",
  "onnx/embed_tokens.onnx",
  "onnx/decoder_model_merged.onnx",
];

/// The MLX-format config file name inside a checkpoint directory (the `mlxrs`
/// checkpoint marker, paired with a weight file).
pub(crate) const MLX_CONFIG: &str = "config.json";

/// The MLX-format safetensors weights file name — the always-available baseline
/// weight format.
pub(crate) const MLX_SAFETENSORS: &str = "model.safetensors";

/// The legacy single-file safetensors weights name. Some older MLX checkpoints
/// ship their weights as `weights.safetensors` rather than `model.safetensors`;
/// `mlxrs::io::load_weights_from_dir` accepts it as a fallback tier.
pub(crate) const MLX_SAFETENSORS_LEGACY: &str = "weights.safetensors";

/// The sharded-checkpoint index file name. A multi-shard safetensors export
/// ships a `model.safetensors.index.json` weight map instead of a single
/// `model.safetensors`.
pub(crate) const MLX_SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// What a checkpoint directory describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CheckpointLayout {
  /// A complete ONNX graph set (or nothing recognizable at all, so that the
  /// ONNX path keeps naming the missing graph as it always has).
  Onnx {
    /// Whether an MLX checkpoint's assets sit alongside the graphs. The graphs
    /// still win by default, but this is what makes
    /// [`Options::with_backend`](crate::Options::with_backend) able to pick MLX
    /// out of the same directory instead of the choice being silent.
    mlx_alongside: bool,
  },
  /// An MLX checkpoint: `config.json` plus a weight set in an enabled format,
  /// with no complete ONNX graph set.
  Mlx,
  /// The directory cannot be classified.
  Ambiguous(&'static str),
  /// An MLX checkpoint whose only weight file is in a format this build did
  /// not enable (`"npz"` / `"gguf"`).
  MlxFormatDisabled(&'static str),
  /// Recognizably a checkpoint, but missing what either backend needs.
  Incomplete(&'static str),
}

/// Which MLX weight formats a directory offers, relative to what this build
/// enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxWeights {
  /// No MLX weight file of any format.
  None,
  /// At least one weight file in a format `mlxrs::io::load_weights_from_dir`
  /// can load in this build.
  Enabled,
  /// Weight files exist, but only in formats this build disabled.
  DisabledOnly(&'static str),
}

/// Whether `dir` contains at least one file with the given `extension`.
fn has_extension(dir: &Path, extension: &str) -> bool {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return false;
  };
  entries.flatten().any(|entry| {
    let path = entry.path();
    path.extension().and_then(|e| e.to_str()) == Some(extension) && path.is_file()
  })
}

/// Classify `dir`'s MLX weight set against the formats this build enabled.
///
/// The safetensors tiers (sharded index, `model.safetensors`, the legacy
/// `weights.safetensors`) are always available; `npz` / `gguf` are behind their
/// Cargo features and mirror exactly what `mlxrs::io::load_weights_from_dir`
/// will accept, so routing and loading agree on which checkpoints count.
///
/// The enabled formats are tested BEFORE the disabled ones so a directory
/// carrying both an unusable `*.npz` and a usable `*.gguf` reports `Enabled`
/// rather than the first-seen disabled format.
fn mlx_weights(dir: &Path) -> MlxWeights {
  if dir.join(MLX_SAFETENSORS_INDEX).is_file()
    || dir.join(MLX_SAFETENSORS).is_file()
    || dir.join(MLX_SAFETENSORS_LEGACY).is_file()
  {
    return MlxWeights::Enabled;
  }
  let npz = has_extension(dir, "npz");
  let gguf = has_extension(dir, "gguf");
  if (npz && cfg!(feature = "npz")) || (gguf && cfg!(feature = "gguf")) {
    MlxWeights::Enabled
  } else if npz {
    MlxWeights::DisabledOnly("npz")
  } else if gguf {
    MlxWeights::DisabledOnly("gguf")
  } else {
    MlxWeights::None
  }
}

/// Classify a model directory.
pub(crate) fn detect(dir: &Path) -> CheckpointLayout {
  let onnx_found = REQUIRED_ONNX_GRAPHS
    .iter()
    .filter(|graph| dir.join(graph).is_file())
    .count();
  let onnx_complete = onnx_found == REQUIRED_ONNX_GRAPHS.len();
  let onnx_partial = onnx_found > 0 && !onnx_complete;
  let has_config = dir.join(MLX_CONFIG).is_file();
  let weights = mlx_weights(dir);
  let mlx_checkpoint = has_config && weights == MlxWeights::Enabled;

  if onnx_complete {
    return CheckpointLayout::Onnx {
      mlx_alongside: mlx_checkpoint,
    };
  }
  if onnx_partial {
    // Half an ONNX graph set beside an MLX weight set has no documented
    // answer: the graph-presence disambiguator assumes a COMPLETE graph set,
    // and picking either backend would run a checkpoint the caller did not
    // describe. Previously this routed to ONNX and died on the missing graph.
    if has_config && weights != MlxWeights::None {
      return CheckpointLayout::Ambiguous(
        "an incomplete ONNX graph set sits beside an MLX checkpoint — pin one with Options::with_backend, or remove the stray files",
      );
    }
    // Incomplete ONNX with nothing else: unchanged: the ONNX path names the
    // graph it could not open.
    return CheckpointLayout::Onnx {
      mlx_alongside: false,
    };
  }
  if mlx_checkpoint {
    return CheckpointLayout::Mlx;
  }
  if has_config {
    return match weights {
      MlxWeights::DisabledOnly(format) => CheckpointLayout::MlxFormatDisabled(format),
      MlxWeights::None => CheckpointLayout::Incomplete(
        "config.json is present but the directory holds neither an MLX weight file nor the ONNX graphs",
      ),
      MlxWeights::Enabled => CheckpointLayout::Mlx,
    };
  }
  if weights == MlxWeights::Enabled {
    return CheckpointLayout::Incomplete(
      "an MLX weight file is present but config.json is missing, and the ONNX graphs were not found",
    );
  }
  // Nothing recognizable — keep the historical error surface and let the ONNX
  // path report the graph it could not open.
  CheckpointLayout::Onnx {
    mlx_alongside: false,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Create a fresh, uniquely named temp dir for one layout case.
  fn layout_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "lfm_layout_{tag}_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp layout dir");
    dir
  }

  /// Touch a (possibly nested) file under `dir`.
  fn touch(dir: &Path, rel: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
      std::fs::create_dir_all(parent).expect("create parent dir");
    }
    std::fs::write(&path, b"\0").expect("write file");
  }

  #[test]
  fn complete_onnx_graph_set_selects_onnx() {
    let dir = layout_dir("onnx");
    for graph in REQUIRED_ONNX_GRAPHS {
      touch(&dir, graph);
    }
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
      layout,
      CheckpointLayout::Onnx {
        mlx_alongside: false
      }
    );
  }

  /// The HuggingFace source assets (`config.json` + `model.safetensors`) beside
  /// a COMPLETE ONNX graph set is a decided case, not an ambiguous one — the
  /// graphs win — but the MLX assets are reported so an explicit backend pin
  /// can select them instead.
  #[test]
  fn onnx_graphs_beside_mlx_assets_reports_both() {
    let dir = layout_dir("dual");
    for graph in REQUIRED_ONNX_GRAPHS {
      touch(&dir, graph);
    }
    touch(&dir, MLX_CONFIG);
    touch(&dir, MLX_SAFETENSORS);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
      layout,
      CheckpointLayout::Onnx {
        mlx_alongside: true
      }
    );
  }

  #[test]
  fn mlx_checkpoint_selects_mlx() {
    let dir = layout_dir("mlx");
    touch(&dir, MLX_CONFIG);
    touch(&dir, MLX_SAFETENSORS);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(layout, CheckpointLayout::Mlx);
  }

  /// Sharded and legacy single-file safetensors layouts are MLX checkpoints
  /// too — `mlxrs::io::load_weights_from_dir` loads both tiers.
  #[test]
  fn sharded_and_legacy_safetensors_select_mlx() {
    for weight in [MLX_SAFETENSORS_INDEX, MLX_SAFETENSORS_LEGACY] {
      let dir = layout_dir("mlxtier");
      touch(&dir, MLX_CONFIG);
      touch(&dir, weight);
      let layout = detect(&dir);
      let _ = std::fs::remove_dir_all(&dir);
      assert_eq!(layout, CheckpointLayout::Mlx, "weight file {weight}");
    }
  }

  /// A partial ONNX graph set beside an MLX weight set has no documented
  /// answer. Previously this routed to ONNX and failed on the missing graph.
  #[test]
  fn partial_onnx_beside_mlx_is_ambiguous() {
    let dir = layout_dir("partial");
    touch(&dir, REQUIRED_ONNX_GRAPHS[0]);
    touch(&dir, MLX_CONFIG);
    touch(&dir, MLX_SAFETENSORS);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
      matches!(layout, CheckpointLayout::Ambiguous(_)),
      "expected Ambiguous, got {layout:?}"
    );
  }

  /// An incomplete ONNX set on its own keeps the historical surface: the ONNX
  /// path names the graph it could not open.
  #[test]
  fn partial_onnx_alone_still_routes_to_onnx() {
    let dir = layout_dir("partial_alone");
    touch(&dir, REQUIRED_ONNX_GRAPHS[0]);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
      layout,
      CheckpointLayout::Onnx {
        mlx_alongside: false
      }
    );
  }

  /// `npz` / `gguf` are recognized as MLX weight formats iff their feature is
  /// on; when off, the checkpoint is reported as format-disabled rather than
  /// falling through to an unrelated ONNX error.
  #[test]
  fn disabled_weight_format_is_named() {
    for (file, format, enabled) in [
      ("model.npz", "npz", cfg!(feature = "npz")),
      ("model.gguf", "gguf", cfg!(feature = "gguf")),
    ] {
      let dir = layout_dir("fmt");
      touch(&dir, MLX_CONFIG);
      touch(&dir, file);
      let layout = detect(&dir);
      let _ = std::fs::remove_dir_all(&dir);
      if enabled {
        assert_eq!(layout, CheckpointLayout::Mlx, "{file} with feature on");
      } else {
        assert_eq!(
          layout,
          CheckpointLayout::MlxFormatDisabled(format),
          "{file} with feature off"
        );
      }
    }
  }

  /// A lone `config.json` is an incomplete checkpoint, named as such.
  #[test]
  fn config_without_weights_is_incomplete() {
    let dir = layout_dir("noweights");
    touch(&dir, MLX_CONFIG);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
      matches!(layout, CheckpointLayout::Incomplete(_)),
      "expected Incomplete, got {layout:?}"
    );
  }

  /// An MLX weight file without `config.json` is incomplete too — the MLX
  /// loader needs the config for every format, gguf included.
  #[test]
  fn weights_without_config_is_incomplete() {
    let dir = layout_dir("noconfig");
    touch(&dir, MLX_SAFETENSORS);
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
      matches!(layout, CheckpointLayout::Incomplete(_)),
      "expected Incomplete, got {layout:?}"
    );
  }

  /// An empty directory keeps the historical behaviour: route to ONNX so the
  /// error names the graph that could not be opened.
  #[test]
  fn empty_dir_routes_to_onnx() {
    let dir = layout_dir("empty");
    let layout = detect(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
      layout,
      CheckpointLayout::Onnx {
        mlx_alongside: false
      }
    );
  }
}
