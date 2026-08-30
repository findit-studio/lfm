//! Image preprocessing for LFM2.5-VL. Wasm-compatible.
//!
//! Per spec §6.4: vision_encoder.onnx takes
//! pre-patchified `[N_batch, num_patches, 768]` (NOT image-shaped).
//! Patch-level padding via attention_mask, NOT pixel-level.

#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
use std::path::Path;

use image::DynamicImage;

use crate::{
  error::{Error, Result},
  options::ImageBudget,
};

pub mod image_plan;
pub use image_plan::{IMAGE_BLOCK_WRAPPER_TOKENS, ImagePlan};
pub mod tile_grid;
pub use tile_grid::TileGrid;
mod target_ratios;
use tile_grid::{PATCH_SIZE, TILE_PIXEL_UNIT};

/// Image preprocessor for LFM2.5-VL.
#[derive(Debug, Clone, Copy)]
pub struct Preprocessor {
  budget: ImageBudget,
}

impl Preprocessor {
  /// Construct with the given image budget.
  pub fn new(budget: ImageBudget) -> Self {
    Self { budget }
  }

  /// Returns the budget this preprocessor was constructed with.
  pub fn budget(&self) -> &ImageBudget {
    &self.budget
  }

  /// Single-image preprocess.
  pub fn preprocess(&self, image: &DynamicImage) -> Result<PreprocessedImage> {
    self.budget.validate()?;
    let grid = tile_grid::pick_tile_grid(image.width(), image.height(), &self.budget)?;
    flatten_to_patches(image, &grid)
  }

  /// Multi-image convenience.
  pub fn preprocess_batch(&self, images: &[DynamicImage]) -> Result<Vec<PreprocessedImage>> {
    images.iter().map(|i| self.preprocess(i)).collect()
  }

  /// Path-based convenience with EXIF orientation correction.
  #[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "decoders", not(target_arch = "wasm32"))))
  )]
  pub fn preprocess_path(&self, path: &Path) -> Result<PreprocessedImage> {
    let img = decode_with_orientation(path)?;
    self.preprocess(&img)
  }
}

/// Output of `Preprocessor::preprocess` — directly fed to `vision_encoder.run`.
///
/// LAYOUT:
/// - `pixel_values`: `[N_batch, num_patches, 768]` flattened (NOT image-shaped).
///   768 = 16² × 3 = patch_size² × channels.
/// - `pixel_attention_mask`: `[N_batch, num_patches]` — 1 = valid, 0 = padded.
/// - `spatial_shapes`: `[N_batch, 2]` — (h_patches, w_patches) per entry.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
  pixel_values: Vec<f32>,
  pixel_attention_mask: Vec<i64>,
  spatial_shapes: Vec<i64>,
  batch_size: usize,
  patches_per_entry: usize,
  rows: u32,
  cols: u32,
  main_tile_h: u32,
  main_tile_w: u32,
  thumbnail_size: Option<(u32, u32)>,
  tokens_per_main_tile: usize,
  thumbnail_tokens: Option<usize>,
}

impl PreprocessedImage {
  /// Pre-patchified pixel values `[N_batch * num_patches * 768]`.
  pub fn pixel_values(&self) -> &[f32] {
    &self.pixel_values
  }

  /// Per-patch attention mask `[N_batch * num_patches]` (1=valid, 0=padded).
  pub fn pixel_attention_mask(&self) -> &[i64] {
    &self.pixel_attention_mask
  }

  /// Per-entry (h_patches, w_patches) `[N_batch * 2]`.
  pub fn spatial_shapes(&self) -> &[i64] {
    &self.spatial_shapes
  }

  /// Number of batch entries (= number of tiles incl. thumbnail).
  pub fn batch_size(&self) -> usize {
    self.batch_size
  }

  /// Per-batch-entry padded num_patches (the tensor's second dim).
  pub fn patches_per_entry(&self) -> usize {
    self.patches_per_entry
  }

  /// Total number of tiles (main + thumbnail).
  pub fn num_tiles(&self) -> usize {
    (self.rows * self.cols) as usize + usize::from(self.thumbnail_size.is_some())
  }

  /// Main tile-grid rows (1 in single-tile path).
  pub fn rows(&self) -> usize {
    self.rows as usize
  }

  /// Main tile-grid cols (1 in single-tile path).
  pub fn cols(&self) -> usize {
    self.cols as usize
  }

  /// (h, w) of one main tile.
  pub fn main_tile_size(&self) -> (usize, usize) {
    (self.main_tile_h as usize, self.main_tile_w as usize)
  }

  /// (h, w) of the thumbnail tile if present.
  pub fn thumbnail_size(&self) -> Option<(usize, usize)> {
    self.thumbnail_size.map(|(h, w)| (h as usize, w as usize))
  }

  /// Tokens per main tile (256 in multi-tile path; dynamic in single-tile).
  pub fn tokens_per_main_tile(&self) -> usize {
    self.tokens_per_main_tile
  }

  /// Tokens for the thumbnail tile (None when no thumbnail).
  pub fn thumbnail_tokens(&self) -> Option<usize> {
    self.thumbnail_tokens
  }

  /// Total `<image>` tokens in the chat template after expansion.
  pub fn num_image_tokens(&self) -> usize {
    (self.rows as usize) * (self.cols as usize) * self.tokens_per_main_tile
      + self.thumbnail_tokens.unwrap_or(0)
  }

  /// Build an [`crate::chat_template::ImagePlaceholderInfo`] for use with
  /// `expand_image_placeholders` — bridges this preproc output to the
  /// chat-template module's grid-info struct.
  pub fn to_placeholder_info(&self) -> crate::chat_template::ImagePlaceholderInfo {
    crate::chat_template::ImagePlaceholderInfo::new(
      self.rows as usize,
      self.cols as usize,
      self.tokens_per_main_tile,
      self.thumbnail_tokens,
    )
  }
}

/// Decode an image from disk applying EXIF orientation. Mirrors siglip2 idiom.
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
#[cfg_attr(
  docsrs,
  doc(cfg(all(feature = "decoders", not(target_arch = "wasm32"))))
)]
pub fn decode_with_orientation(path: &Path) -> Result<DynamicImage> {
  use image::{ImageDecoder, ImageReader};
  let mut decoder = ImageReader::open(path)?
    .with_guessed_format()?
    .into_decoder()?;
  // Cap source dims + alloc BEFORE the full decode so a
  // decompression-bomb header (e.g., a 100k×100k PNG that
  // would allocate >30 GB of RGB) fails fast instead of OOMing the
  // process. set_limits() is strict for width/height — exceeding either
  // returns Err(LimitError) without allocating the buffer.
  decoder.set_limits(decode_limits())?;
  let orientation = decoder.orientation()?;
  let mut img = DynamicImage::from_decoder(decoder)?;
  img.apply_orientation(orientation);
  Ok(img)
}

/// In-memory variant of [`decode_with_orientation`].
#[cfg(feature = "decoders")]
#[cfg_attr(docsrs, doc(cfg(feature = "decoders")))]
pub fn decode_bytes_with_orientation(bytes: &[u8]) -> Result<DynamicImage> {
  use image::{ImageDecoder, ImageReader};
  use std::io::Cursor;
  let mut decoder = ImageReader::new(Cursor::new(bytes))
    .with_guessed_format()?
    .into_decoder()?;
  // Same decompression-bomb cap as the path-based path; see comment
  // in decode_with_orientation above.
  decoder.set_limits(decode_limits())?;
  let orientation = decoder.orientation()?;
  let mut img = DynamicImage::from_decoder(decoder)?;
  img.apply_orientation(orientation);
  Ok(img)
}

/// Strict resource limits for image decoding. Caps source width and
/// height at 16 384 px each (4× 4K, generous for legitimate use) and
/// total decoder allocation at 256 MiB (half the image-crate default
/// of 512 MiB).
///
/// 16384² × 4 bytes ≈ 1 GiB raw RGBA — well above max_alloc, so the
/// width/height check fires first for square decompression bombs.
/// For asymmetric bombs (e.g., 1×1_000_000) the height limit catches
/// them before max_alloc would.
#[cfg(feature = "decoders")]
fn decode_limits() -> image::Limits {
  let mut limits = image::Limits::default();
  limits.max_image_width = Some(16_384);
  limits.max_image_height = Some(16_384);
  limits.max_alloc = Some(256 * 1024 * 1024);
  limits
}

/// Same source-dim caps as [`decode_limits`], exposed for the
/// header-only `image_dimensions` path in `generate.rs` — the
/// header-read must reject decompression bombs at the same
/// threshold the full decode does.
// Only `generate.rs` consumes this, and generate.rs is gated on
// `inference + decoders`. Under `--features decoders` (without
// `inference`), the function is dead code and `-D dead_code`
// would fail clippy. Gate accordingly.
#[cfg(all(feature = "decoders", feature = "inference"))]
pub(crate) fn header_decode_limits() -> image::Limits {
  decode_limits()
}

/// PIL-compatible bilinear-with-antialias resize. Upstream
/// `Lfm2VlImageProcessorFast` resizes via torchvision
/// `F.resize(..., interpolation=BILINEAR, antialias=True)`, which
/// matches PIL's `Image.resize(..., Image.BILINEAR)` — a separable
/// triangle filter with kernel SUPPORT scaled by max(2/ratio, 2)
/// for each downscaled axis (i.e., a low-pass prefilter that the
/// `image` crate's `FilterType::Triangle` does NOT apply).
///
/// `fast_image_resize`'s `Convolution(FilterType::Bilinear)` is
/// explicitly designed to match Pillow's bilinear (the same target
/// torchvision aims for). Using it removes the silent algorithmic
/// divergence between our preprocessing and the upstream encoder's
/// training distribution.
///
/// Returns an `RgbImage` of size `(dst_w, dst_h)`. Errors only on
/// structurally-impossible conditions (buffer mis-sizing); for our
/// callers (validated `RgbImage` source, validated dst dims) the
/// failure path is unreachable but we still propagate rather than
/// panic.
fn pil_bilinear_resize(src: &image::RgbImage, dst_w: u32, dst_h: u32) -> Result<image::RgbImage> {
  use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image as FirImage,
  };

  let (sw, sh) = (src.width(), src.height());
  let src_fir = FirImage::from_vec_u8(sw, sh, src.as_raw().to_vec(), PixelType::U8x3)
    .map_err(|_| Error::InvalidRequest("pil_bilinear_resize: source buffer mis-sized"))?;
  let mut dst_fir = FirImage::new(dst_w, dst_h, PixelType::U8x3);
  let mut resizer = Resizer::new();
  let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
  resizer
    .resize(&src_fir, &mut dst_fir, &opts)
    .map_err(|_| Error::InvalidRequest("pil_bilinear_resize: resize failed"))?;
  image::RgbImage::from_raw(dst_w, dst_h, dst_fir.into_vec()).ok_or(Error::InvalidRequest(
    "pil_bilinear_resize: output buffer mis-sized",
  ))
}

/// Convert source image into the patch-flattened tensor layout
/// `vision_encoder.onnx` expects.
fn flatten_to_patches(src: &DynamicImage, grid: &TileGrid) -> Result<PreprocessedImage> {
  use image::imageops;

  // 1. Resize source to (cols × tile_w, rows × tile_h).
  let target_w = grid.cols() * grid.tile_w();
  let target_h = grid.rows() * grid.tile_h();
  let src_rgb = src.to_rgb8();
  let resized = if src_rgb.width() == target_w && src_rgb.height() == target_h {
    src_rgb.clone()
  } else {
    pil_bilinear_resize(&src_rgb, target_w, target_h)?
  };

  // 2. Build per-tile RGB blocks (row-major).
  let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(grid.num_tiles());
  for r in 0..grid.rows() {
    for c in 0..grid.cols() {
      let crop = imageops::crop_imm(
        &resized,
        c * grid.tile_w(),
        r * grid.tile_h(),
        grid.tile_w(),
        grid.tile_h(),
      )
      .to_image();
      tiles.push(crop);
    }
  }

  // 3. Append thumbnail (if present) — smart-resize the WHOLE source.
  if let Some((th, tw)) = grid.thumbnail() {
    let thumb = pil_bilinear_resize(&src_rgb, tw, th)?;
    tiles.push(thumb);
  }

  // 4. Per-tile: 16×16 RGB patches → flatten to 768-vec each → normalize px/255 → 2*px-1.
  // Pad each batch entry to `max_h * max_w` patches (per-axis maxes
  // across the batch, NOT max(h*w) per entry). The vision encoder's
  // SigLIP2 NaFlex `pos_embed` reduces `spatial_shapes` with
  // `ReduceMax(axis=0)` per axis to choose the Resize target
  // `(max_h, max_w)`; it then pads the resulting `[max_h * max_w,
  // dim]` positions out to `pixel_values.shape[1]` by repeating the
  // first-position embedding. So `pixel_values.shape[1]` must equal
  // `max_h * max_w` (or be larger), otherwise the position-embedding
  // tensor and the patch-embedding tensor disagree on axis 1 and the
  // first Add inside the encoder fails to broadcast.
  let max_h = tiles
    .iter()
    .map(|t| (t.height() / PATCH_SIZE) as usize)
    .max()
    .unwrap_or(0);
  let max_w = tiles
    .iter()
    .map(|t| (t.width() / PATCH_SIZE) as usize)
    .max()
    .unwrap_or(0);
  let max_patches = max_h * max_w;
  let n_batch = tiles.len();
  let mut pixel_values = vec![0f32; n_batch * max_patches * 768];
  let mut attn_mask = vec![0i64; n_batch * max_patches];
  let mut spatial = Vec::with_capacity(n_batch * 2);

  // Scratch buffer: one tile's worth of u8 bytes in patchified order (py, px, dy, dx, ch).
  //
  // Past reviews flagged this as a possible HWC-vs-CHW mismatch
  // ("preprocessor_config says channels_first but we emit HWC").
  // FALSE POSITIVE — explained:
  //
  // The "channels_first" data_format in preprocessor_config.json refers
  // to the LAYOUT OF THE RESIZED IMAGE FED TO `convert_image_to_patches`
  // — i.e., torch tensor shape (B, C, H, W). Inside that function,
  // upstream does:
  //   patched = images.reshape(B, C, n_h, ps, n_w, ps)
  //   patched = patched.permute(0, 2, 4, 3, 5, 1)  # → (B, n_h, n_w, ps, ps, C)
  //   patched = patched.reshape(B, n_h * n_w, -1)
  // The final `.reshape(..., -1)` collapses (ps, ps, C) into 768 bytes
  // in HWC order (last dim is C). So the actual ENCODER input is HWC
  // per-patch despite the upstream pipeline starting from a CHW image.
  //
  // Our `(dy, dx, ch)` byte order IS HWC and IS what upstream produces.
  // The multi_image_ordering_proof.json fixture captures upstream
  // pixel_values bit-for-bit and our code passes it.
  //
  // Strategy: access tiles via `as_raw()` (avoids get_pixel's per-call bounds check overhead),
  // assemble bytes in patch-traversal order into a contiguous u8 scratch buffer, then run a
  // single flat iterator-zip to convert u8 → f32 — the form the compiler vectorizes best.
  let max_tile_pixels = tiles
    .iter()
    .map(|t| (t.width() * t.height()) as usize * 3)
    .max()
    .unwrap_or(0);
  let mut raw_patch_bytes: Vec<u8> = Vec::with_capacity(max_tile_pixels);

  for (i, tile) in tiles.iter().enumerate() {
    let (tw, th) = (tile.width(), tile.height());
    let h_patches = th / PATCH_SIZE;
    let w_patches = tw / PATCH_SIZE;
    spatial.push(h_patches as i64);
    spatial.push(w_patches as i64);

    // Access raw pixel buffer: row-major (y * width + x) * 3, channels R/G/B.
    // Using the raw slice eliminates per-pixel bounds checks from get_pixel.
    let raw: &[u8] = tile.as_raw();
    let stride = tw as usize * 3; // bytes per row

    let n_valid = (h_patches * w_patches) as usize;
    raw_patch_bytes.clear();
    // Preserve same traversal order as old impl: outer (py, px), inner (dy, dx, ch).
    for py in 0..h_patches as usize {
      for px in 0..w_patches as usize {
        for dy in 0..PATCH_SIZE as usize {
          let row_start = (py * PATCH_SIZE as usize + dy) * stride + px * PATCH_SIZE as usize * 3;
          // Push the 16 pixels (48 bytes) of this row-within-patch in one extend.
          raw_patch_bytes.extend_from_slice(&raw[row_start..row_start + PATCH_SIZE as usize * 3]);
        }
      }
    }

    // Normalize: single flat u8 → f32 iterator, no division/modulo in the hot path.
    // Each patch's 768 bytes are contiguous in raw_patch_bytes at patch_idx*768, and
    // the destination in pixel_values is also at dst_base + patch_idx*768.
    let dst_base = i * max_patches * 768;
    let dst = &mut pixel_values[dst_base..dst_base + n_valid * 768];
    for (dst_el, &b) in dst.iter_mut().zip(raw_patch_bytes.iter()) {
      *dst_el = (b as f32 / 255.0) * 2.0 - 1.0;
    }
    // Mark valid patches in the attention mask.
    for p in 0..n_valid {
      attn_mask[i * max_patches + p] = 1;
    }
  }

  let tokens_per_main =
    ((grid.tile_h() / TILE_PIXEL_UNIT) * (grid.tile_w() / TILE_PIXEL_UNIT)) as usize;
  let thumbnail_tokens = grid
    .thumbnail()
    .map(|(th, tw)| ((th / TILE_PIXEL_UNIT) * (tw / TILE_PIXEL_UNIT)) as usize);

  Ok(PreprocessedImage {
    pixel_values,
    pixel_attention_mask: attn_mask,
    spatial_shapes: spatial,
    batch_size: n_batch,
    patches_per_entry: max_patches,
    rows: grid.rows(),
    cols: grid.cols(),
    main_tile_h: grid.tile_h(),
    main_tile_w: grid.tile_w(),
    thumbnail_size: grid.thumbnail(),
    tokens_per_main_tile: tokens_per_main,
    thumbnail_tokens,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::{ImageBuffer, Rgb};

  #[test]
  fn preprocess_small_square_succeeds() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([128, 128, 128])));
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    assert!(out.batch_size() >= 1);
    assert!(out.num_image_tokens() > 0);
  }

  #[test]
  fn preprocess_large_square_routes_multi_tile() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1024, 1024, Rgb([128, 128, 128])));
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    assert!(out.num_tiles() >= 4);
    assert_eq!(out.tokens_per_main_tile(), 256);
  }

  #[test]
  fn pixel_values_normalized_minus_one_to_one() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([255, 0, 0])));
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    let pv = out.pixel_values();
    assert!((pv[0] - 1.0).abs() < 1e-5); // R = 255 → 1.0
    assert!((pv[1] + 1.0).abs() < 1e-5); // G = 0 → -1.0
    assert!((pv[2] + 1.0).abs() < 1e-5); // B = 0 → -1.0
  }

  #[test]
  fn batch_preserves_order() {
    let p = Preprocessor::new(ImageBudget::new());
    let red = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([255, 0, 0])));
    let blue = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([0, 0, 255])));
    let batch = p.preprocess_batch(&[red, blue]).unwrap();
    assert_eq!(batch.len(), 2);
    // First image's first pixel R should be 1.0 (red); second's B should be 1.0 (blue).
    assert!((batch[0].pixel_values()[0] - 1.0).abs() < 1e-5);
    assert!((batch[1].pixel_values()[2] - 1.0).abs() < 1e-5);
  }

  /// N5 byte-identity test: ensures the vectorized flat-u8→f32 normalization
  /// produces bit-for-bit identical output to the scalar formula `(b as f32 / 255.0) * 2.0 - 1.0`
  /// applied to raw pixel bytes in patchified traversal order.
  #[test]
  fn normalization_byte_identical_to_scalar_reference() {
    // Use a non-uniform image so different pixel values are exercised.
    // Pattern: x selects R (0–255 gradient), y selects G (0–255 gradient), B=128 constant.
    let w: u32 = 64;
    let h: u32 = 64;
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| {
      Rgb([((x * 4) % 256) as u8, ((y * 4) % 256) as u8, 128u8])
    }));

    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    let pv = out.pixel_values();

    // Reconstruct the same patchified traversal as flatten_to_patches to get expected values.
    // This is the scalar reference: collect bytes in patch order, then apply the formula.
    use image::imageops;
    let budget = ImageBudget::new();
    let grid = tile_grid::pick_tile_grid(w, h, &budget).unwrap();
    let target_w = grid.cols() * grid.tile_w();
    let target_h = grid.rows() * grid.tile_h();
    let src_rgb = img.to_rgb8();
    let resized = if src_rgb.width() == target_w && src_rgb.height() == target_h {
      src_rgb.clone()
    } else {
      pil_bilinear_resize(&src_rgb, target_w, target_h).unwrap()
    };

    let mut expected: Vec<f32> = Vec::with_capacity(pv.len());
    // Compute max_patches (same as flatten_to_patches).
    let h_patches_main = grid.tile_h() / tile_grid::PATCH_SIZE;
    let w_patches_main = grid.tile_w() / tile_grid::PATCH_SIZE;
    let main_patches = (h_patches_main * w_patches_main) as usize;
    let thumb_patches = grid
      .thumbnail()
      .map(|(th, tw)| ((th / tile_grid::PATCH_SIZE) * (tw / tile_grid::PATCH_SIZE)) as usize)
      .unwrap_or(0);
    // All main tiles have the same patch count; thumbnail (if any) may differ.
    let max_patches = if grid.thumbnail().is_some() {
      main_patches.max(thumb_patches)
    } else {
      main_patches
    };

    // Build expected: main tiles.
    for r in 0..grid.rows() {
      for c in 0..grid.cols() {
        let crop = imageops::crop_imm(
          &resized,
          c * grid.tile_w(),
          r * grid.tile_h(),
          grid.tile_w(),
          grid.tile_h(),
        )
        .to_image();
        let mut patch_vals = vec![0f32; max_patches * 768];
        for py in 0..h_patches_main {
          for px in 0..w_patches_main {
            let pidx = (py * w_patches_main + px) as usize;
            for dy in 0..tile_grid::PATCH_SIZE {
              for dx in 0..tile_grid::PATCH_SIZE {
                let pix = crop.get_pixel(
                  px * tile_grid::PATCH_SIZE + dx,
                  py * tile_grid::PATCH_SIZE + dy,
                );
                for ch in 0..3usize {
                  let k = dy * tile_grid::PATCH_SIZE * 3 + dx * 3 + ch as u32;
                  patch_vals[pidx * 768 + k as usize] = (pix[ch] as f32 / 255.0) * 2.0 - 1.0;
                }
              }
            }
          }
        }
        expected.extend_from_slice(&patch_vals);
      }
    }
    // Thumbnail (if any).
    if let Some((th, tw)) = grid.thumbnail() {
      let thumb = pil_bilinear_resize(&src_rgb, tw, th).unwrap();
      let th_h_patches = th / tile_grid::PATCH_SIZE;
      let th_w_patches = tw / tile_grid::PATCH_SIZE;
      let mut patch_vals = vec![0f32; max_patches * 768];
      for py in 0..th_h_patches {
        for px in 0..th_w_patches {
          let pidx = (py * th_w_patches + px) as usize;
          for dy in 0..tile_grid::PATCH_SIZE {
            for dx in 0..tile_grid::PATCH_SIZE {
              let pix = thumb.get_pixel(
                px * tile_grid::PATCH_SIZE + dx,
                py * tile_grid::PATCH_SIZE + dy,
              );
              for ch in 0..3usize {
                let k = dy * tile_grid::PATCH_SIZE * 3 + dx * 3 + ch as u32;
                patch_vals[pidx * 768 + k as usize] = (pix[ch] as f32 / 255.0) * 2.0 - 1.0;
              }
            }
          }
        }
      }
      expected.extend_from_slice(&patch_vals);
    }

    assert_eq!(pv.len(), expected.len(), "pixel_values length mismatch");
    for (idx, (&got, &exp)) in pv.iter().zip(expected.iter()).enumerate() {
      assert_eq!(
        got.to_bits(),
        exp.to_bits(),
        "pixel_values[{idx}] mismatch: got {got} vs ref {exp}"
      );
    }
  }

  #[test]
  fn to_placeholder_info_round_trip() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1024, 1024, Rgb([128, 128, 128])));
    let p = Preprocessor::new(ImageBudget::new());
    let pre = p.preprocess(&img).unwrap();
    let info = pre.to_placeholder_info();
    assert_eq!(info.rows(), pre.rows());
    assert_eq!(info.cols(), pre.cols());
    assert_eq!(info.tokens_per_main_tile(), pre.tokens_per_main_tile());
    assert_eq!(info.num_image_tokens(), pre.num_image_tokens());
  }
}
