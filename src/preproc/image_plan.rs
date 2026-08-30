//! The ONE authoritative per-image plan.
//!
//! Every stage of a multimodal request that has an opinion about an image's
//! layout — the prompt's `<image>` / `<|img_row_R_col_C|>` / `<|img_thumbnail|>`
//! markers, the context-budget admission gates, the backend's own pixel
//! preprocessing, and the vision-feature splice — reads that opinion from a
//! single [`ImagePlan`] produced once, up front, from the image's header
//! dimensions.
//!
//! # Why one plan
//!
//! The failure this prevents is silent, not loud. `lfm` renders the prompt from
//! its own [`ImageBudget`](crate::ImageBudget) tiling, but a backend
//! preprocesses with whatever tiling *it* implements — on the MLX road that is
//! the checkpoint's own `config.json`-driven `split_image`. When the two
//! disagree, the `<image>`-token runs in the prompt and the produced feature
//! rows disagree too. Sometimes that surfaces as a length mismatch; sometimes
//! the totals coincide (a 4×2 grid and a 2×4 grid have the same tile count) and
//! the features simply bind to the wrong spatial positions, with no error at
//! all and quietly wrong conditioning.
//!
//! So the plan is produced by the backend that will execute it
//! ([`Backend::plan_image`](crate::runtime::backend::Backend::plan_image)),
//! consumed by everything downstream, and then **checked against reality**: the
//! backend re-derives the layout from the pixels it actually decoded and raises
//! [`Error::ImagePlanMismatch`](crate::Error::ImagePlanMismatch) if it differs.
//! On the MLX road a third gate runs at load time — the checkpoint's tiling
//! parameters must equal the ones `lfm` renders markers from, or the load is
//! refused by name
//! ([`Error::MlxTilingMismatch`](crate::Error::MlxTilingMismatch)).

use crate::chat_template::ImagePlaceholderInfo;

/// Per-image structural tokens the chat template always emits around an image
/// block, independent of the grid: `<|image_start|>` + `<|image_end|>`.
pub const IMAGE_BLOCK_WRAPPER_TOKENS: usize = 2;

/// One image's authoritative layout decision.
///
/// Built from an [`ImagePlaceholderInfo`] — the marker layout — plus the token
/// counts derived from it, so the markers, the admission arithmetic, and the
/// splice arithmetic can never drift apart: they are the same numbers.
///
/// See the [module docs](self) for why a single plan exists at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePlan {
  /// The marker layout the chat template renders for this image.
  placeholder: ImagePlaceholderInfo,
  /// Total `<image>` placeholder tokens this image expands to.
  image_tokens: usize,
  /// Total non-`<image>` structural tokens the block adds (wrapper + row/col
  /// markers + the optional thumbnail marker).
  structural_tokens: usize,
  /// Number of sub-images the backend's preprocessing must produce (main tiles
  /// plus the thumbnail when present), in marker order.
  sub_images: usize,
}

impl ImagePlan {
  /// Derive the plan from a marker layout.
  ///
  /// Token arithmetic is saturating throughout: an absurd grid produced by a
  /// hostile budget must surface as a rejected admission check, never as a
  /// wrapped count that under-reports the prompt size.
  pub const fn from_placeholder(placeholder: ImagePlaceholderInfo) -> Self {
    let rows = placeholder.rows();
    let cols = placeholder.cols();
    let tiles = rows.saturating_mul(cols);
    let main = tiles.saturating_mul(placeholder.tokens_per_main_tile());
    let (thumb_tokens, thumb_count) = match placeholder.thumbnail_tokens() {
      Some(n) => (n, 1usize),
      None => (0usize, 0usize),
    };
    // Structural tokens: the block wrapper always; for a multi-tile layout also
    // one `<|img_row_R_col_C|>` per main tile and one `<|img_thumbnail|>` when
    // a thumbnail is rendered. A 1×1 grid is the single-tile format, which the
    // template emits without any positional marker.
    let structural_tokens = if rows > 1 || cols > 1 {
      IMAGE_BLOCK_WRAPPER_TOKENS
        .saturating_add(tiles)
        .saturating_add(thumb_count)
    } else {
      IMAGE_BLOCK_WRAPPER_TOKENS
    };
    Self {
      placeholder,
      image_tokens: main.saturating_add(thumb_tokens),
      structural_tokens,
      sub_images: tiles.saturating_add(thumb_count),
    }
  }

  /// The marker layout the chat template renders for this image.
  pub const fn placeholder(&self) -> &ImagePlaceholderInfo {
    &self.placeholder
  }

  /// Total `<image>` placeholder tokens this image expands to — equivalently,
  /// the number of vision feature rows its preprocessing must produce.
  pub const fn image_tokens(&self) -> usize {
    self.image_tokens
  }

  /// Total non-`<image>` structural tokens the rendered image block adds.
  pub const fn structural_tokens(&self) -> usize {
    self.structural_tokens
  }

  /// Number of sub-images (main tiles, then the thumbnail when present) the
  /// backend's preprocessing must produce, in marker order.
  pub const fn sub_images(&self) -> usize {
    self.sub_images
  }

  /// Every token this image contributes to the prompt: `<image>` placeholders
  /// plus structural markers.
  pub const fn total_tokens(&self) -> usize {
    self.image_tokens.saturating_add(self.structural_tokens)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A single-tile image: no positional markers, no thumbnail, one sub-image,
  /// and the block wrapper is the whole structural cost.
  #[test]
  fn single_tile_plan() {
    let plan = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(1, 1, 64, None));
    assert_eq!(plan.image_tokens(), 64);
    assert_eq!(plan.structural_tokens(), IMAGE_BLOCK_WRAPPER_TOKENS);
    assert_eq!(plan.sub_images(), 1);
    assert_eq!(plan.total_tokens(), 64 + 2);
  }

  /// A multi-tile image with a thumbnail: `rows*cols` positional markers, one
  /// `<|img_thumbnail|>`, and `rows*cols + 1` sub-images in marker order.
  #[test]
  fn multi_tile_plan_with_thumbnail() {
    let plan = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(2, 3, 256, Some(64)));
    assert_eq!(plan.image_tokens(), 6 * 256 + 64);
    assert_eq!(plan.structural_tokens(), 2 + 6 + 1);
    assert_eq!(plan.sub_images(), 7);
  }

  /// Multi-tile without a thumbnail: markers but no thumbnail marker, and the
  /// sub-image count is exactly the tile count.
  #[test]
  fn multi_tile_plan_without_thumbnail() {
    let plan = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(2, 3, 256, None));
    assert_eq!(plan.image_tokens(), 6 * 256);
    assert_eq!(plan.structural_tokens(), 2 + 6);
    assert_eq!(plan.sub_images(), 6);
  }

  /// Count the structural (`<|…|>`) tokens in a rendered image block. Every
  /// structural marker the template emits — `<|image_start|>`, `<|image_end|>`,
  /// `<|img_row_R_col_C|>`, `<|img_thumbnail|>` — is exactly one `<|…|>` run,
  /// and the `<image>` placeholder is not one, so counting the opener counts
  /// them all.
  fn structural_markers(rendered: &str) -> usize {
    rendered.matches("<|").count()
  }

  /// The plan's numbers ARE the rendered prompt's numbers.
  ///
  /// This is the invariant the whole one-plan design exists to hold: the
  /// `<image>` count the backend must produce feature rows for equals the
  /// `<image>` count the template actually emitted, and the structural count
  /// admission control budgeted for equals the markers actually emitted. Swept
  /// across the default and `fast()` budgets (which differ in tile band,
  /// token band and thumbnail) and across square, landscape, portrait, and
  /// small-enough-to-stay-single-tile geometries.
  #[test]
  fn plan_counts_match_the_rendered_marker_layout() {
    use crate::{
      chat_template::{IMAGE_TOKEN, expand_image_placeholders},
      options::ImageBudget,
      preproc::tile_grid::pick_tile_grid,
    };

    for (budget_name, budget) in [
      ("default", ImageBudget::new()),
      ("fast", ImageBudget::fast()),
    ] {
      for (width, height) in [(1024u32, 1024u32), (1600, 900), (640, 1280), (200, 200)] {
        let grid = pick_tile_grid(width, height, &budget)
          .unwrap_or_else(|e| panic!("{budget_name} budget, {width}x{height}: {e}"));
        let plan = ImagePlan::from_placeholder(grid.to_placeholder_info());
        let rendered = expand_image_placeholders("<image>", &[*plan.placeholder()])
          .expect("render one image block");
        assert_eq!(
          rendered.matches(IMAGE_TOKEN).count(),
          plan.image_tokens(),
          "{budget_name} budget, {width}x{height}: rendered <image> count must equal the plan's feature-row count"
        );
        assert_eq!(
          structural_markers(&rendered),
          plan.structural_tokens(),
          "{budget_name} budget, {width}x{height}: rendered marker count must equal the plan's structural budget"
        );
      }
    }
  }

  /// The same invariant across a multi-image request, where a per-image drift
  /// would be masked by the totals if the plans were not summed the way
  /// admission control sums them.
  #[test]
  fn multi_image_plan_counts_match_the_rendered_prompt() {
    use crate::{
      chat_template::{IMAGE_TOKEN, expand_image_placeholders},
      options::ImageBudget,
      preproc::tile_grid::pick_tile_grid,
    };

    let budget = ImageBudget::new();
    // Deliberately mixed geometries: a large square (multi-tile), a wide
    // landscape, and a small image that stays single-tile.
    let plans: Vec<ImagePlan> = [(1024u32, 1024u32), (1600, 500), (200, 200)]
      .into_iter()
      .map(|(w, h)| {
        ImagePlan::from_placeholder(
          pick_tile_grid(w, h, &budget)
            .expect("grid")
            .to_placeholder_info(),
        )
      })
      .collect();
    let infos: Vec<_> = plans.iter().map(|p| *p.placeholder()).collect();
    let rendered = expand_image_placeholders("a<image>b<image>c<image>d", &infos)
      .expect("render three image blocks");

    let planned_images: usize = plans.iter().map(ImagePlan::image_tokens).sum();
    let planned_structural: usize = plans.iter().map(ImagePlan::structural_tokens).sum();
    assert_eq!(rendered.matches(IMAGE_TOKEN).count(), planned_images);
    assert_eq!(structural_markers(&rendered), planned_structural);
    // The plans must not all be the same layout, or the test would pass on a
    // degenerate case.
    assert!(
      plans.iter().any(|p| p.sub_images() > 1),
      "at least one geometry must reach the multi-tile path"
    );
    assert!(
      plans.iter().any(|p| p.sub_images() == 1),
      "at least one geometry must stay single-tile"
    );
  }

  /// The counts saturate rather than wrap. A wrapped total would under-report
  /// the prompt size and slip past the context-budget admission gate — the one
  /// place these numbers are load-bearing for safety.
  #[test]
  fn token_arithmetic_saturates() {
    let plan = ImagePlan::from_placeholder(ImagePlaceholderInfo::new(
      usize::MAX,
      usize::MAX,
      usize::MAX,
      None,
    ));
    assert_eq!(plan.image_tokens(), usize::MAX);
    assert_eq!(plan.total_tokens(), usize::MAX);
    assert_eq!(plan.sub_images(), usize::MAX);
  }
}
