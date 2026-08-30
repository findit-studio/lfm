//! Token sampler — `FreeSampler` for unconstrained sampling
//! (greedy/min_p with repetition penalty), `ConstrainedSampler`
//! for llguidance schema-driven sampling.
//!
//! # llguidance API discovery (1.7.3)
//!
//! - `Constraint::compute_mask(&mut self) -> anyhow::Result<&StepResult>`
//!   Returns a reference to the internal `StepResult` (alias for
//!   `Branch<SimpleVob>`).  The caller MUST call `commit_token` afterwards.
//!
//! - `StepResult` = `Branch<SimpleVob>` from `toktrie`:
//!   - `is_stop() -> bool` — no sampling mask AND no splices → schema done.
//!   - `sample_mask: Option<SimpleVob>` — `Some` when sampling is required.
//!   - `unconditional_splice() -> Option<&Splice>` — ff_tokens path (unused
//!     here; we never enable `ff_tokens` in `InferenceCapabilities`).
//!
//! - `SimpleVob::is_allowed(tok: TokenId) -> bool` — O(1) bit-vector lookup.
//!   Used by our `apply_mask` to set disallowed logits to `f32::NEG_INFINITY`.
//!   (`SimpleVob::apply_to` sets *allowed* tokens to 0.0 — wrong polarity for
//!   us; we need to set disallowed tokens to −∞ while preserving the LLM
//!   logit magnitudes on allowed tokens.)
//!
//! - `Constraint::commit_token(&mut self, sampled_token: Option<TokenId>)
//!     -> anyhow::Result<CommitResult>`
//!   Advances the grammar state.  Pass `Some(id)` when a mask was present.
//!
//! - `ParserFactory` (factory.rs) — compiles grammars and caches tokenizer
//!   state; passed `TokEnv` / `TokenizerEnv` (trait from `toktrie`).  Used in
//!   Task 13 (engine.rs) — not wired here.
//!
//! - No `rand` dep needed; we carry a self-contained xoshiro128+ RNG.

use std::collections::HashSet;

#[cfg(feature = "inference")]
use llguidance::Constraint;
use smol_str::SmolStr;

use crate::{
  error::{Error, Result},
  options::RequestOptions,
};

// =============================================================================
// Public-facing types
// =============================================================================

/// Decision returned by a sampler at each step.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum SampleResult {
  /// Continue generating with this token.
  Token(u32),
  /// Schema reached an accepting state BEFORE this step — no token
  /// sampled, no further tokens needed. (`ConstrainedSampler` only.)
  SchemaComplete,
  /// This token was sampled AND it completes the schema. The caller
  /// must include the token in the output but stop generation
  /// afterward without re-entering the loop. (`ConstrainedSampler`
  /// only.) Without this distinction, a constrained run whose final
  /// JSON token lands at index `max_new_tokens - 1` would discard a
  /// valid completion as `MaxTokensExceeded`, since the next loop
  /// iteration that would have surfaced `SchemaComplete` never runs.
  TokenAndComplete(u32),
}

// =============================================================================
// Sampler trait
// =============================================================================

/// Common sampler interface.
#[allow(dead_code)]
pub(crate) trait Sampler {
  /// Sample a token from `logits`. `logits` is a flat slice of length
  /// `vocab_size`. `step` is the 0-based decode step (used for diagnostic
  /// errors only). `seen_tokens` are previously-emitted tokens (for
  /// repetition penalty). Returns either a token id or `SchemaComplete`.
  fn sample(
    &mut self,
    logits: &mut [f32],
    seen_tokens: &HashSet<u32>,
    step: usize,
  ) -> Result<SampleResult>;
}

// =============================================================================
// FreeSampler
// =============================================================================

/// Unconstrained sampler with greedy/min_p + repetition penalty.
///
/// **Vocab masking** (): the decoder produces
/// 65 536 logits but the bundled tokenizer only defines token IDs
/// through 64 399. IDs in [64 400, 65 535] are decoder-only padding
/// with no string representation; sampling one would either silently
/// truncate the output (detokenize skips it) or feed garbage back
/// into the decoder. `vocab_size` caps the sample-able range —
/// logits at indices ≥ `vocab_size` are masked to -inf before any
/// sampling decision. ConstrainedSampler doesn't need this because
/// llguidance's allow-mask already excludes the tail IDs.
#[allow(dead_code)]
pub(crate) struct FreeSampler {
  opts: RequestOptions,
  rng: SmallRng,
  /// Tokenizer's actual vocab size (e.g. 64 400 for the bundled
  /// LFM2.5-VL tokenizer). Logits beyond this are decoder-only
  /// padding and must not be sampled.
  vocab_size: u32,
}

impl FreeSampler {
  #[allow(dead_code)]
  pub(crate) fn new(opts: RequestOptions, seed: u64, vocab_size: u32) -> Self {
    Self {
      opts,
      rng: SmallRng::seed_from_u64(seed),
      vocab_size,
    }
  }
}

impl Sampler for FreeSampler {
  fn sample(
    &mut self,
    logits: &mut [f32],
    seen_tokens: &HashSet<u32>,
    _step: usize,
  ) -> Result<SampleResult> {
    // mask logits beyond the tokenizer's vocab size.
    // Decoder produces 65 536 logits; tokenizer only defines IDs up
    // to vocab_size-1. Cap the sample-able range so non-decodable
    // tokens never win greedy/min-p draws.
    let cap = (self.vocab_size as usize).min(logits.len());
    for v in logits.iter_mut().skip(cap) {
      *v = f32::NEG_INFINITY;
    }
    apply_repetition_penalty(logits, seen_tokens, self.opts.repetition_penalty());
    // Issue #2 C-001 + three-part numeric
    // safety check, restricted to the valid vocab range so the
    // intentional -Inf masking of the [vocab_size, logits.len())
    // tail (above) doesn't trip the guard.
    //
    // (a) Single NaN in the valid range → reject. NaN poisons
    //     softmax (e^NaN = NaN spreads through the sum) and biases
    //     argmax (f32::total_cmp orders NaN as the largest, so a
    //     NaN's position would always win greedy). Source can be
    //     model output (numerical overflow / malformed export) or
    //     repetition_penalty * NaN (validation rejects NaN penalty
    //     but defense-in-depth).
    // (b) Single +Inf in the valid range → reject. `-Inf` is the
    //     masking vocabulary this crate writes on purpose (the vocab
    //     tail above, `apply_mask`'s llguidance allow-mask, a penalty
    //     overflowing a large negative logit), but `+Inf` is never
    //     written by us: it can only arrive from a broken forward.
    //     Admitting one makes greedy pick its position unconditionally,
    //     and under temperature it makes `softmax`'s maximum
    //     non-finite — whose safe fallback spreads probability
    //     UNIFORMLY over the whole row, including the `-Inf` entries a
    //     constraint mask used to forbid a token, so `sample_min_p`
    //     could return a schema-disallowed id.
    // (c) Every valid logit -Inf → reject. Penalty × extreme-
    //     negative logit can overflow every candidate to -Inf, so
    //     sample_min_p's argmax fallback would pick id 0 (which a
    //     ConstrainedSampler mask might forbid).
    let valid = &logits[..cap];
    if valid.iter().any(|&v| v.is_nan() || v == f32::INFINITY) {
      return Err(Error::SamplerNonFinite);
    }
    if valid.iter().all(|&v| !v.is_finite()) {
      return Err(Error::SamplerNonFinite);
    }
    if self.opts.temperature() <= 0.0 {
      // Greedy.
      let id = argmax(logits);
      return Ok(SampleResult::Token(id));
    }
    apply_temperature(logits, self.opts.temperature());
    let probs = softmax(logits);
    let id = sample_min_p(&probs, self.opts.min_p(), &mut self.rng);
    Ok(SampleResult::Token(id))
  }
}

// =============================================================================
// ConstrainedSampler
// =============================================================================

/// Schema-constrained sampler driven by llguidance.
#[cfg(feature = "inference")]
#[allow(dead_code)]
pub(crate) struct ConstrainedSampler {
  inner: FreeSampler,
  constraint: Constraint,
}

#[cfg(feature = "inference")]
impl ConstrainedSampler {
  #[allow(dead_code)]
  pub(crate) fn new(
    constraint: Constraint,
    opts: RequestOptions,
    seed: u64,
    vocab_size: u32,
  ) -> Self {
    Self {
      // The inner FreeSampler also masks logits ≥ vocab_size, but the
      // ConstrainedSampler's own apply_mask runs first using llguidance's
      // SimpleVob (which already excludes the unused tail IDs). The
      // double-masking is cheap and defensive.
      inner: FreeSampler::new(opts, seed, vocab_size),
      constraint,
    }
  }
}

#[cfg(feature = "inference")]
impl Sampler for ConstrainedSampler {
  fn sample(
    &mut self,
    logits: &mut [f32],
    seen_tokens: &HashSet<u32>,
    step: usize,
  ) -> Result<SampleResult> {
    // 1) Ask llguidance for the allowed-token mask.
    //    Returns &StepResult = &Branch<SimpleVob>.
    let step_result = self.constraint.compute_mask().map_err(Error::llguidance)?;

    // 2) Check if the schema has accepted (stop state: no mask, no splices).
    if step_result.is_stop() {
      return Ok(SampleResult::SchemaComplete);
    }

    // 3) If there is no sample_mask but the result is not a stop, it means
    //    an unconditional splice (ff_tokens). We don't enable ff_tokens in
    //    InferenceCapabilities, so this branch is defensive only.
    let mask = match &step_result.sample_mask {
      Some(m) => m,
      None => {
        // Unconditional splice with no mask — commit with None and
        // treat as a schema-complete signal so the caller stops.
        self
          .constraint
          .commit_token(None)
          .map_err(Error::llguidance)?;
        return Ok(SampleResult::SchemaComplete);
      }
    };

    // 4) Apply mask to logits: set disallowed token logits to −∞.
    //    We deliberately avoid SimpleVob::apply_to, which sets *allowed*
    //    tokens to 0.0 (wrong polarity — would destroy logit magnitudes).
    //    `mask.len()` is the bit-vec capacity (number of token slots);
    //    if this is smaller than `logits.len()`, the tail is also masked
    //    out — out-of-range token ids must not be sample-able.
    apply_mask(logits, mask);
    if logits.iter().all(|&v| !v.is_finite()) {
      return Err(Error::LlGuidanceDeadEnd {
        step,
        state: SmolStr::new_inline("empty mask"),
      });
    }

    // 5) Sample from masked distribution via the underlying FreeSampler.
    let inner_decision = self.inner.sample(logits, seen_tokens, step)?;
    let id = match inner_decision {
      SampleResult::Token(id) => id,
      SampleResult::SchemaComplete | SampleResult::TokenAndComplete(_) => {
        // FreeSampler never emits SchemaComplete or TokenAndComplete.
        return Ok(inner_decision);
      }
    };

    // 6) Commit the chosen token to advance llguidance's state machine.
    //
    // `CommitResult.stop` ONLY reports stop when the *previous*
    // compute_mask was already in stop state (see llguidance 1.7.3
    // src/constraint.rs:207-208 doc comment). After sampling the
    // final token, commit_token sets `pending_stop = true` internally
    // (line 258-260) and saves a `StepResult::splice(...)` whose
    // `is_stop()` is false — so `commit.stop` would be false here
    // even when the schema is now complete.
    //
    // The reliable post-commit signal is `has_pending_stop()`. Using
    // it ensures `TokenAndComplete` fires for the boundary case
    // documented above (final JSON token at index max_new_tokens-1).
    let _commit = self
      .constraint
      .commit_token(Some(id))
      .map_err(Error::llguidance)?;

    if self.constraint.has_pending_stop() {
      Ok(SampleResult::TokenAndComplete(id))
    } else {
      Ok(SampleResult::Token(id))
    }
  }
}

// =============================================================================
// Helpers
// =============================================================================

#[allow(dead_code)]
fn apply_repetition_penalty(logits: &mut [f32], seen: &HashSet<u32>, penalty: f32) {
  if penalty == 1.0 {
    return;
  }
  for &tok in seen {
    let i = tok as usize;
    if i >= logits.len() {
      continue;
    }
    let v = logits[i];
    // Hugging Face symmetric formulation:
    // positive logit → divide (make less likely);
    // negative logit → multiply (push further negative).
    logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
  }
}

#[allow(dead_code)]
fn apply_temperature(logits: &mut [f32], temp: f32) {
  if temp == 1.0 {
    return;
  }
  let inv = 1.0 / temp;
  for v in logits.iter_mut() {
    *v *= inv;
  }
}

#[allow(dead_code)]
fn softmax(logits: &[f32]) -> Vec<f32> {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  // If `max` is itself non-finite, every `logits[i]` is non-finite
  // too (or the slice is empty). The sampler's `apply_mask` +
  // post-penalty all-non-finite check should already have rejected
  // this case before softmax runs, but compute a safe uniform
  // distribution here as defense-in-depth — the previous
  // `(v - max).exp()` would produce `NaN.exp() = NaN` for every
  // entry, and the downstream argmax would silently pick an
  // arbitrary token (often id 0, which a `ConstrainedSampler` mask
  // might forbid).
  if !max.is_finite() {
    let n = logits.len();
    if n == 0 {
      return Vec::new();
    }
    let p = 1.0_f32 / n as f32;
    return vec![p; n];
  }
  let mut out: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
  let sum: f32 = out.iter().sum();
  if sum > 0.0 {
    for v in out.iter_mut() {
      *v /= sum;
    }
  }
  out
}

#[allow(dead_code)]
fn argmax(logits: &[f32]) -> u32 {
  let mut best_i = 0u32;
  let mut best_v = f32::NEG_INFINITY;
  for (i, &v) in logits.iter().enumerate() {
    if v > best_v {
      best_v = v;
      best_i = i as u32;
    }
  }
  best_i
}

#[allow(dead_code)]
fn sample_min_p(probs: &[f32], min_p: f32, rng: &mut SmallRng) -> u32 {
  let p_max = probs.iter().copied().fold(0.0f32, f32::max);
  let threshold = min_p * p_max;
  // also exclude zero-probability entries
  // unconditionally. With min_p=0 the threshold is 0; the inclusive
  // `>= threshold` would otherwise keep entries with p == 0, including
  // tokens that ConstrainedSampler::apply_mask set to -inf logit
  // (softmax → 0). If gen_f32() returns 0.0, the cumulative `r <= cum`
  // check would then select the first such zero-prob entry — which
  // could be a schema-disallowed token, breaking llguidance's
  // guarantees and producing invalid structured output.
  let filtered: Vec<(u32, f32)> = probs
    .iter()
    .enumerate()
    .filter_map(|(i, &p)| (p >= threshold && p > 0.0).then_some((i as u32, p)))
    .collect();
  if filtered.is_empty() {
    // Fallback: argmax. Use total_cmp so non-finite probabilities (NaN
    // from a poisoned softmax, ±inf from an overflowing logit) sort
    // deterministically instead of panicking on `partial_cmp(...).unwrap()`.
    // RequestOptions::validate already rejects NaN/inf inputs, but this
    // is the last line of defense — softmax can still produce NaN if
    // every input logit is -inf (e.g., a llguidance mask that disallows
    // every token at this step).
    return probs
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .map(|(i, _)| i as u32)
      .unwrap_or(0);
  }
  let total: f32 = filtered.iter().map(|(_, p)| *p).sum();
  let r: f32 = rng.gen_f32() * total;
  let mut cum = 0.0f32;
  for &(id, p) in &filtered {
    cum += p;
    if r <= cum {
      return id;
    }
  }
  filtered.last().unwrap().0
}

/// Set logits[i] = −∞ for every token id NOT in the llguidance allow-mask.
/// The mask's bit-vec capacity (`SimpleVob::len`) defines the valid token
/// range; logits past that range are also set to −∞ so out-of-range ids
/// cannot be sampled even when the model's vocab over-counts the
/// constrained vocabulary.
#[cfg(feature = "inference")]
#[allow(dead_code)]
fn apply_mask(logits: &mut [f32], mask: &llguidance::toktrie::SimpleVob) {
  let mask_len = mask.len();
  for (i, logit) in logits.iter_mut().enumerate().take(mask_len) {
    if !mask.is_allowed(i as u32) {
      *logit = f32::NEG_INFINITY;
    }
  }
  for v in logits.iter_mut().skip(mask_len) {
    *v = f32::NEG_INFINITY;
  }
}

// =============================================================================
// Self-contained xoshiro128-like RNG — no rand dep, no alloc per draw
// =============================================================================

/// Two-word LFSR-based RNG (xoshiro-style). Field 0 = "s1" state word,
/// field 1 = "s0" state word (in xoshiro notation).
#[allow(dead_code)]
struct SmallRng {
  s1: u64,
  s0: u64,
}

#[allow(dead_code)]
impl SmallRng {
  fn seed_from_u64(seed: u64) -> Self {
    let a = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let b = a.wrapping_mul(0xBF58476D1CE4E5B9);
    Self {
      s1: a | 1,
      s0: b | 1,
    }
  }

  fn next_u64(&mut self) -> u64 {
    // xorshift128+ step (Vigna). the
    // previous implementation had a bug where `self.s0 = prev_s0`
    // wrote the OLD self.s0 back to self.s0 (a no-op), leaving one
    // of the two state words frozen forever. The correct Vigna
    // transition copies the OLD s1 ("y") into s0, then mixes a
    // mutated copy of s0 ("x") with y to produce the new s1.
    let mut x = self.s0;
    let y = self.s1;
    self.s0 = y; // FIX: state s0 advances to OLD s1, not OLD s0
    x ^= x << 23;
    self.s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    self.s1.wrapping_add(y)
  }

  /// Generate a float in [0, 1) using the upper 24 bits of a u64.
  fn gen_f32(&mut self) -> f32 {
    let bits = self.next_u64() >> 40; // 24 bits for f32 mantissa precision
    (bits as f32) / ((1u64 << 24) as f32)
  }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::options::RequestOptions;
  use std::collections::HashSet;

  #[test]
  fn argmax_picks_largest() {
    let logits = vec![0.1, 0.5, 0.2, 1.5, 0.0];
    assert_eq!(argmax(&logits), 3);
  }

  #[test]
  fn softmax_sums_to_one() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let probs = softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
  }

  #[test]
  fn softmax_returns_uniform_for_all_non_finite_input() {
    // The sampler's all-non-finite check should have rejected this
    // case before softmax was called, but if it ever slips through
    // (e.g., a future path that bypasses the sampler), softmax must
    // not return NaN — that would poison every downstream argmax /
    // min_p decision and silently pick token id 0. A uniform 1/n
    // distribution is the safe degenerate output.
    let logits = vec![f32::NEG_INFINITY; 7];
    let probs = softmax(&logits);
    assert_eq!(probs.len(), 7);
    let expected = 1.0 / 7.0;
    for p in &probs {
      assert!(p.is_finite(), "softmax must never return NaN/Inf");
      assert!((p - expected).abs() < 1e-6);
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Empty input is a no-op (no positions to score).
    let empty: Vec<f32> = Vec::new();
    assert!(softmax(&empty).is_empty());

    // Mixed -inf + NaN also collapses to uniform — `max` is NaN
    // (any NaN poisons the fold), `is_finite()` is false, so we
    // take the safe path.
    let logits = vec![f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY];
    let probs = softmax(&logits);
    assert_eq!(probs.len(), 3);
    for p in &probs {
      assert!(p.is_finite());
    }
  }

  #[test]
  fn repetition_penalty_lowers_seen_positive_logits() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let mut seen = HashSet::new();
    seen.insert(1u32);
    apply_repetition_penalty(&mut logits, &seen, 2.0);
    assert_eq!(logits, vec![1.0, 1.0, 3.0]);
  }

  #[test]
  fn repetition_penalty_amplifies_seen_negative_logits() {
    let mut logits = vec![-1.0, -2.0, -3.0];
    let mut seen = HashSet::new();
    seen.insert(1u32);
    apply_repetition_penalty(&mut logits, &seen, 2.0);
    assert_eq!(logits, vec![-1.0, -4.0, -3.0]);
  }

  #[test]
  fn free_sampler_greedy_picks_argmax() {
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    // vocab_size larger than logits.len(): no masking applied.
    let mut sampler = FreeSampler::new(opts, 42, 65_536);
    let mut logits = vec![0.1f32, 0.5, 0.2, 1.5, 0.0];
    let result = sampler.sample(&mut logits, &HashSet::new(), 0).unwrap();
    assert!(matches!(result, SampleResult::Token(3)));
  }

  #[test]
  fn free_sampler_masks_logits_beyond_vocab_size() {
    // decoder produces 65 536 logits but
    // tokenizer only defines IDs through 64 399. Sampling an ID in
    // [64 400, 65 535] would produce non-decodable output. Verify
    // FreeSampler caps the sample-able range.
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    // vocab_size = 5: only IDs 0..4 are valid; ID 7 is decoder-only.
    let mut sampler = FreeSampler::new(opts, 42, 5);
    // Logit at index 7 is the highest; without masking, greedy would
    // pick it. With masking, ID 7 is -inf so we pick the next best.
    let mut logits = vec![0.1f32, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 99.0];
    let result = sampler.sample(&mut logits, &HashSet::new(), 0).unwrap();
    let id = match result {
      SampleResult::Token(id) => id,
      _ => panic!("expected Token, got {result:?}"),
    };
    assert!(id < 5, "FreeSampler picked masked id {id} (vocab_size=5)");
    assert_eq!(
      id, 3,
      "expected id=3 (logit 1.0); masked logit at 7 (99.0) should be -inf"
    );
  }

  #[test]
  fn free_sampler_errors_on_all_non_finite_post_penalty() {
    // if every logit becomes -inf after
    // repetition_penalty, sampling must fail closed instead of
    // letting argmax/sample_min_p return an arbitrary id (which a
    // ConstrainedSampler mask might forbid). Construct a degenerate
    // input: all logits are -inf already, and the seen set covers
    // them so apply_repetition_penalty leaves them at -inf.
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    let mut sampler = FreeSampler::new(opts, 42, 65_536);
    let mut logits = vec![f32::NEG_INFINITY; 8];
    let seen: HashSet<u32> = (0..8).collect();
    let result = sampler.sample(&mut logits, &seen, 0);
    assert!(matches!(result, Err(Error::SamplerNonFinite)));
  }

  #[test]
  fn free_sampler_errors_on_single_nan_logit_from_model() {
    // Issue #2 C-001: a SINGLE NaN logit from the model output (not
    // all of them) is the dangerous case — argmax with NaN returns
    // the NaN's position; softmax with NaN poisons every output. The
    // previous .all() check missed this. Reproducer: 7 finite logits
    // + 1 NaN.
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    let mut sampler = FreeSampler::new(opts, 42, 65_536);
    let mut logits = vec![0.1f32, 0.5, 0.2, 1.0, 0.0, 0.3, 0.4, f32::NAN];
    let result = sampler.sample(&mut logits, &HashSet::new(), 0);
    assert!(
      matches!(result, Err(Error::SamplerNonFinite)),
      "single-NaN logit must reject (issue #2 C-001 regression)"
    );
  }

  /// A lone `+Inf` in the valid vocab range must reject. The ORT decoder has
  /// always refused a non-finite row at its session boundary; the MLX decoder
  /// did not, so this is the sampler-side half of making the two backends
  /// behave identically. Greedy would otherwise pick the `+Inf` position
  /// unconditionally.
  #[test]
  fn free_sampler_errors_on_single_positive_inf_logit() {
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    let mut sampler = FreeSampler::new(opts, 42, 65_536);
    let mut logits = vec![0.1f32, 0.5, 0.2, 1.0, f32::INFINITY, 0.3];
    assert!(
      matches!(
        sampler.sample(&mut logits, &HashSet::new(), 0),
        Err(Error::SamplerNonFinite)
      ),
      "a lone +Inf logit must reject rather than win greedy"
    );
  }

  /// The constrained-decoding failure a `+Inf` causes, and the fix, in one
  /// test.
  ///
  /// `apply_mask` writes `-Inf` on every id an llguidance allow-mask forbids.
  /// With a `+Inf` also present, `softmax`'s maximum is non-finite, so its
  /// safe fallback returns a UNIFORM distribution over the whole row — masked
  /// ids included — and `sample_min_p` will happily draw one of them. The
  /// sampler now rejects the row before reaching softmax at all.
  #[test]
  fn positive_inf_would_let_a_masked_token_be_drawn_and_is_rejected() {
    // ids 0, 1, 3 forbidden by the constraint; id 2 carries the broken +Inf.
    let masked = [
      f32::NEG_INFINITY,
      f32::NEG_INFINITY,
      f32::INFINITY,
      f32::NEG_INFINITY,
    ];

    // The hazard: softmax over this row is uniform across ALL four ids, so a
    // draw can land on a forbidden one.
    let probs = softmax(&masked);
    assert!(
      probs.iter().all(|p| (p - 0.25).abs() < 1e-6),
      "softmax over a +Inf row spreads uniformly across masked entries: {probs:?}"
    );
    let mut rng = SmallRng::seed_from_u64(7);
    let mut drew_masked = false;
    for _ in 0..256 {
      if sample_min_p(&probs, 0.0, &mut rng) != 2 {
        drew_masked = true;
        break;
      }
    }
    assert!(
      drew_masked,
      "the uniform fallback must be able to draw a masked id — that is the bug being closed"
    );

    // The fix: the row never reaches softmax.
    let opts = RequestOptions::default()
      .with_temperature(0.8)
      .with_repetition_penalty(1.0);
    let mut sampler = FreeSampler::new(opts, 42, 4);
    let mut logits = masked.to_vec();
    assert!(matches!(
      sampler.sample(&mut logits, &HashSet::new(), 0),
      Err(Error::SamplerNonFinite)
    ));
  }

  #[test]
  fn free_sampler_allows_neg_inf_in_valid_range() {
    // -Inf in the valid range is legitimate: it's how the
    // vocab/llguidance/penalty masking semantically marks a token
    // as "do not pick." As long as at least one logit remains
    // finite, sampling proceeds normally. Verifies the all(-Inf)
    // check doesn't mis-fire on partial -Inf.
    let opts = RequestOptions::default()
      .with_temperature(0.0)
      .with_repetition_penalty(1.05);
    let mut sampler = FreeSampler::new(opts, 42, 65_536);
    let mut logits = vec![
      f32::NEG_INFINITY,
      0.5,
      f32::NEG_INFINITY,
      1.5,
      f32::NEG_INFINITY,
    ];
    let result = sampler.sample(&mut logits, &HashSet::new(), 0).unwrap();
    assert!(
      matches!(result, SampleResult::Token(3)),
      "argmax picks the largest finite logit (1.5 at index 3)"
    );
  }

  #[test]
  fn rng_both_state_words_advance() {
    // the previous implementation froze
    // one state word, producing biased correlated draws. Verify
    // both s0 and s1 actually change between consecutive draws.
    let mut rng = SmallRng::seed_from_u64(0x1234_5678_9ABC_DEF0);
    let initial_s0 = rng.s0;
    let initial_s1 = rng.s1;
    let _ = rng.next_u64();
    assert_ne!(rng.s0, initial_s0, "s0 must advance after next_u64");
    assert_ne!(rng.s1, initial_s1, "s1 must advance after next_u64");
  }

  #[test]
  fn sample_min_p_excludes_zero_prob_tokens_at_min_p_zero() {
    // with min_p=0 and a probability vector
    // containing zeros (typical for ConstrainedSampler after apply_mask
    // sets disallowed token logits to -inf, softmax → 0), the previous
    // filter `p >= threshold` with threshold=0 included those zeros.
    // If gen_f32() returned 0.0, the cumulative `r <= cum` check would
    // select the FIRST zero-prob entry — a schema-disallowed token.
    //
    // Verify: token 0 has probability 0 (masked), token 1 has 0.6, token
    // 2 has 0.4. With min_p=0 we must NEVER pick token 0 regardless of
    // RNG state.
    let probs = [0.0f32, 0.6, 0.4];
    // Construct an RNG state that yields very small floats first to
    // simulate the worst-case r ≈ 0.
    let mut rng = SmallRng::seed_from_u64(0);
    for _ in 0..1000 {
      let id = sample_min_p(&probs, 0.0, &mut rng);
      assert_ne!(
        id, 0,
        "sample_min_p must never select a zero-probability token even when min_p=0"
      );
    }
  }

  #[test]
  fn rng_produces_non_constant_outputs() {
    // Stronger sanity check: 1024 consecutive draws must not all
    // be identical (the broken impl would have produced a short
    // cycle since half the state was frozen). Take 1024 draws and
    // assert at least 1000 unique values — well above any
    // collision rate for an unbiased 64-bit RNG.
    let mut rng = SmallRng::seed_from_u64(42);
    let mut seen = HashSet::new();
    for _ in 0..1024 {
      seen.insert(rng.next_u64());
    }
    assert!(
      seen.len() > 1000,
      "RNG produced only {} unique values across 1024 draws — state likely frozen",
      seen.len()
    );
  }
}
