//! ORT session building + strict input/output validation.
//!
//! Per spec §8.5: ONNX I/O contract (tests/fixtures/onnx_io_contract.json).

use std::path::Path;

use ort::{
  session::Session,
  value::{Outlet, TensorElementType, ValueType},
};

use crate::{
  error::{Error, Result},
  options::Options,
};

/// Build an ORT session from a path with the given options.
///
/// Wires `optimization_level` and thread counts from the ONNX road's own tier
/// of `Options` — `Options::effective_ort_options`, which is that tier when the
/// document pinned `backend = "onnx"` and `OrtOptions::new()` when it left the
/// road to the checkpoint layout. An `inter_threads` above 1 also selects ort's
/// parallel execution mode, without which that count would be inert. EP
/// registration (cuda/tensorrt/etc.) is feature-gated below per spec §5.3
/// EP-feature pattern.
#[allow(dead_code)]
pub(crate) fn build_session(graph: &Path, opts: &Options) -> Result<Session> {
  if !graph.exists() {
    return Err(Error::NotFound(graph.to_path_buf()));
  }
  let ort_opts = opts.effective_ort_options();
  let level = ort_opts.optimization_level();
  let parallel = ort_opts.thread().requires_parallel_execution();
  // Session::builder() returns ort::Result<SessionBuilder>.
  // with_* methods return BuilderResult = Result<SessionBuilder, Error<SessionBuilder>>.
  // Error<SessionBuilder> converts to ort::Error (Error<()>) via From.
  let mut builder = Session::builder()
    .map_err(Error::Ort)?
    .with_optimization_level(level)
    .map_err(|e| Error::Ort(ort::Error::from(e)))?;

  // ort's inter-op thread pool exists only in parallel execution mode; in the
  // default sequential mode `SetInterOpNumThreads` is accepted and ignored. An
  // `inter_threads` above 1 is therefore a request for that mode too, or the
  // knob would be inert — see `ThreadOptions::requires_parallel_execution`,
  // which is where the rule and its unit test live (this function cannot be
  // tested without a real graph: it ends at `commit_from_file`).
  if parallel {
    builder = builder
      .with_parallel_execution(true)
      .map_err(|e| Error::Ort(ort::Error::from(e)))?;
  }

  // `usize::from` and not a cast: `ThreadOptions` stores both counts as `u16`
  // precisely so this seam is infallible and lossless. ort forwards the count
  // to the C API's signed `int`, where a `usize` above `i32::MAX` would wrap
  // and a merely huge one would ask for a pool that can exhaust the process —
  // an `Engine` builds three sessions. A `u16` cannot express either, so there
  // is nothing to validate or recover from here.
  if let Some(t) = ort_opts.thread().intra_threads() {
    builder = builder
      .with_intra_threads(usize::from(t))
      .map_err(|e| Error::Ort(ort::Error::from(e)))?;
  }
  if let Some(t) = ort_opts.thread().inter_threads() {
    builder = builder
      .with_inter_threads(usize::from(t))
      .map_err(|e| Error::Ort(ort::Error::from(e)))?;
  }

  // register the requested execution provider
  // per Cargo feature. Without this, enabling `--features cuda` (etc.)
  // turned on the underlying ort EP support but never told the session
  // to use it, so workloads silently ran on CPU.
  //
  // ort 2.0's `with_execution_providers` takes an iterable of
  // `ExecutionProviderDispatch`; we register at most one per feature.
  // Multiple GPU features compiled in together stack in declaration
  // order — the first whose runtime is available wins.
  #[allow(unused_mut)]
  let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
  #[cfg(feature = "cuda")]
  {
    eps.push(ort::ep::CUDA::default().build());
  }
  #[cfg(feature = "tensorrt")]
  {
    eps.push(ort::ep::TensorRT::default().build());
  }
  #[cfg(feature = "directml")]
  {
    eps.push(ort::ep::DirectML::default().build());
  }
  #[cfg(feature = "rocm")]
  {
    eps.push(ort::ep::ROCm::default().build());
  }
  #[cfg(feature = "coreml")]
  {
    eps.push(ort::ep::CoreML::default().build());
  }
  if !eps.is_empty() {
    builder = builder
      .with_execution_providers(eps)
      .map_err(|e| Error::Ort(ort::Error::from(e)))?;
  }

  let session = builder.commit_from_file(graph).map_err(Error::Ort)?;
  Ok(session)
}

/// Verify an outlet matches the expected dtype + shape.
///
/// `expected_shape` semantics:
/// - `-1` means "this axis MUST be dynamic in the graph". A static
///   dim there is rejected.
/// - any other value means "exact match (or `-1` ok)". The graph may
///   bake a concrete dim or declare it dynamic; both work at runtime.
///
/// Mirrors siglip2's `check_outlet` exactly.
#[allow(dead_code)]
pub(crate) fn check_outlet(
  outlets: &[Outlet],
  name: &'static str,
  expected_dtype: TensorElementType,
  expected_shape: &[i64],
) -> Result<()> {
  let outlet = outlets
    .iter()
    .find(|o| o.name() == name)
    .ok_or(Error::SessionShapeMismatch {
      input: name,
      expected: "outlet present in session",
      got: vec![],
    })?;

  match outlet.dtype() {
    ValueType::Tensor { ty, shape, .. } => {
      if *ty != expected_dtype {
        return Err(Error::SessionContractMismatch {
          input: name,
          expected: "matching tensor dtype",
          got: *ty,
        });
      }
      let actual: &[i64] = shape;
      if actual.len() != expected_shape.len() {
        return Err(Error::SessionShapeMismatch {
          input: name,
          expected: "matching tensor rank",
          got: actual.to_vec(),
        });
      }
      for (i, &want) in expected_shape.iter().enumerate() {
        let got = actual[i];
        if want == -1 {
          // Expected dynamic axis. The graph MUST declare it dynamic —
          // a static dim here would fail at runtime with variable batch sizes.
          if got != -1 {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "dynamic axis required",
              got: actual.to_vec(),
            });
          }
        } else {
          // Expected concrete dim. Graph may match exactly or declare
          // the axis dynamic (-1) — both work at runtime.
          if got != -1 && got != want {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "matching static dim",
              got: actual.to_vec(),
            });
          }
        }
      }
      Ok(())
    }
    _ => Err(Error::SessionShapeMismatch {
      input: name,
      expected: "tensor",
      got: vec![],
    }),
  }
}

/// Validate the vision encoder session against the contract.
/// pixel_values is PRE-PATCHIFIED [batch, num_patches, 768] (not image-shaped).
#[allow(dead_code)]
pub(crate) fn validate_vision_session(s: &Session) -> Result<()> {
  check_outlet(
    s.inputs(),
    "pixel_values",
    TensorElementType::Float32,
    &[-1, -1, 768],
  )?;
  check_outlet(
    s.inputs(),
    "pixel_attention_mask",
    TensorElementType::Int64,
    &[-1, -1],
  )?;
  check_outlet(
    s.inputs(),
    "spatial_shapes",
    TensorElementType::Int64,
    &[-1, 2],
  )?;
  // Output: rank 2 [num_image_tokens, 1024]. NOT rank 3.
  check_outlet(
    s.outputs(),
    "image_features",
    TensorElementType::Float32,
    &[-1, 1024],
  )?;
  Ok(())
}

/// Validate the embed_tokens session.
#[allow(dead_code)]
pub(crate) fn validate_embed_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(), "input_ids", TensorElementType::Int64, &[-1, -1])?;
  check_outlet(
    s.outputs(),
    "inputs_embeds",
    TensorElementType::Float32,
    &[-1, -1, 1024],
  )?;
  Ok(())
}

/// Validate the decoder session.
/// decoder has NO `position_ids` input.
/// cache uses sparse layer indices
/// (conv at [0,1,3,4,6,7,9,11,13,15], attn at [2,5,8,10,12,14] × {key,value}).
#[allow(dead_code)]
pub(crate) fn validate_decoder_session(s: &Session) -> Result<()> {
  check_outlet(
    s.inputs(),
    "inputs_embeds",
    TensorElementType::Float32,
    &[-1, -1, 1024],
  )?;
  check_outlet(
    s.inputs(),
    "attention_mask",
    TensorElementType::Int64,
    &[-1, -1],
  )?;

  // actively REJECT position_ids if
  // present. Decoder::step does not pass position_ids; an ONNX export
  // that requires it would silently fail at first session.run with an
  // opaque ORT error. Catch it at construction.
  if s.inputs().iter().any(|o| o.name() == "position_ids") {
    return Err(Error::SessionShapeMismatch {
      input: "position_ids",
      expected: "must NOT be a required input (Decoder::step doesn't pass it)",
      got: vec![],
    });
  }

  let cache = collect_cache_inputs(s.inputs())?;
  if cache.conv.len() != 10 || cache.attn.len() != 12 {
    return Err(Error::DecoderCacheMismatch {
      expected_conv: 10,
      expected_attn: 12,
      got_conv: cache.conv.len(),
      got_attn: cache.attn.len(),
    });
  }
  // Sparse-index check: collect indices from discovered names, verify
  // they exactly match the expected sets.
  const EXPECTED_CONV: &[u32] = &[0, 1, 3, 4, 6, 7, 9, 11, 13, 15];
  const EXPECTED_ATTN: &[u32] = &[2, 5, 8, 10, 12, 14];
  let mut conv_indices: Vec<u32> = cache
    .conv
    .iter()
    .filter_map(|n| parse_conv_index(n))
    .collect();
  conv_indices.sort_unstable();
  if conv_indices != EXPECTED_CONV {
    return Err(Error::SessionShapeMismatch {
      input: "past_conv.*",
      expected: "sparse indices [0,1,3,4,6,7,9,11,13,15]",
      got: conv_indices.into_iter().map(i64::from).collect(),
    });
  }
  let mut attn_indices: Vec<u32> = cache
    .attn
    .iter()
    .filter_map(|n| parse_attn_index(n))
    .collect();
  attn_indices.sort_unstable();
  attn_indices.dedup();
  if attn_indices != EXPECTED_ATTN {
    return Err(Error::SessionShapeMismatch {
      input: "past_key_values.*.{key,value}",
      expected: "sparse indices [2,5,8,10,12,14]",
      got: attn_indices.into_iter().map(i64::from).collect(),
    });
  }

  // validate dtype + shape for EACH
  // past_* cache input AND its corresponding present_* output. The
  // fix already required present_* to exist for every past_*;
  // this adds the dtype/shape contract so an ONNX export with same
  // names but changed dimensions (e.g., a different head dim, or
  // float16 instead of float32) fails at construction instead of at
  // first decode-step with an opaque ORT shape error.
  //
  // Conv cache: shape [1, 1024, 3], dtype f32 ().
  for name in &cache.conv {
    let owned: &'static str = leak_static(name);
    check_outlet(s.inputs(), owned, TensorElementType::Float32, &[1, 1024, 3])?;
    let present = format!(
      "present_conv.{}",
      parse_conv_index(name).unwrap_or(u32::MAX)
    );
    let present_owned: &'static str = leak_static(&present);
    check_outlet(
      s.outputs(),
      present_owned,
      TensorElementType::Float32,
      &[1, 1024, 3],
    )?;
  }
  // Attn cache: shape [1, 8, past_len, 64], dtype f32 ().
  // past_len is dynamic (-1) on inputs; present is also dynamic since
  // it's past_len + seq.
  for name in &cache.attn {
    let owned: &'static str = leak_static(name);
    check_outlet(
      s.inputs(),
      owned,
      TensorElementType::Float32,
      &[1, 8, -1, 64],
    )?;
    // present_X.key / present_X.value derived from past_key_values.X.{key,value}
    if let Some(rest) = name.strip_prefix("past_key_values.") {
      let present = format!("present.{rest}");
      let present_owned: &'static str = leak_static(&present);
      check_outlet(
        s.outputs(),
        present_owned,
        TensorElementType::Float32,
        &[1, 8, -1, 64],
      )?;
    }
  }

  check_outlet(
    s.outputs(),
    "logits",
    TensorElementType::Float32,
    &[-1, -1, 65536],
  )?;
  Ok(())
}

/// Leak a `String` to obtain a `&'static str` for `check_outlet`'s
/// `name: &'static str` parameter. Called O(layer count) times per
/// session construction (≤22 outlets × 2 = 44 leaks); the leaked
/// memory persists for the process lifetime and is bounded.
fn leak_static(s: &str) -> &'static str {
  Box::leak(s.to_string().into_boxed_str())
}

/// Cache input names grouped by kind, discovered at session-build time.
#[allow(dead_code)]
pub(crate) struct CacheInputs {
  pub(crate) conv: Vec<String>,
  pub(crate) attn: Vec<String>,
}

#[allow(dead_code)]
pub(crate) fn collect_cache_inputs(outlets: &[Outlet]) -> Result<CacheInputs> {
  let mut conv = Vec::new();
  let mut attn = Vec::new();
  for o in outlets {
    let n = o.name();
    if n.starts_with("past_conv.") {
      conv.push(n.to_string());
    } else if n.starts_with("past_key_values.") {
      attn.push(n.to_string());
    }
  }
  Ok(CacheInputs { conv, attn })
}

fn parse_conv_index(name: &str) -> Option<u32> {
  name.strip_prefix("past_conv.")?.parse().ok()
}

#[allow(dead_code)]
fn parse_attn_index(name: &str) -> Option<u32> {
  let rest = name.strip_prefix("past_key_values.")?;
  let dot = rest.find('.')?;
  rest[..dot].parse().ok()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_conv_index_works() {
    assert_eq!(parse_conv_index("past_conv.0"), Some(0));
    assert_eq!(parse_conv_index("past_conv.15"), Some(15));
    assert_eq!(parse_conv_index("past_kv.0"), None);
    assert_eq!(parse_conv_index("past_conv."), None); // empty index
    assert_eq!(parse_conv_index("past_conv.foo"), None); // non-numeric
  }

  #[test]
  fn parse_attn_index_works() {
    assert_eq!(parse_attn_index("past_key_values.2.key"), Some(2));
    assert_eq!(parse_attn_index("past_key_values.14.value"), Some(14));
    assert_eq!(parse_attn_index("past_conv.0"), None);
    assert_eq!(parse_attn_index("past_key_values.2"), None); // no .key/.value suffix
  }

  // Note: validators that require a real ort::Session are tested at the
  // integration level (Task 15) — they need actual ONNX files. The
  // shape-discovery + sparse-index sorting logic here is testable via
  // string parsing, which we cover above.
}
