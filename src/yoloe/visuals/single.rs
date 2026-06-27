//! Single-image [`Visuals`] — the official-style SAVPE input tensor.

use candle_core::{Device, Tensor};

use super::merge::merge_visual_prompt_masks;
use crate::yoloe::prompt::visual::{
    Visual, VisualKind, scaled_mask_dim, validate_visual_prompt_source,
};

/// The merged, class-sorted SAVPE-input tensor for a single image.
///
/// Carries shape `[1, classes, mask_h, mask_w]`, with same-class prompt masks
/// combined via element-wise `maximum` and classes ordered by ascending class
/// id. Construct it with [`Visuals::from_boxes`] or [`Visuals::from_masks`].
#[derive(Debug, Clone)]
pub struct Visuals {
    /// The wrapped `[1, classes, mask_h, mask_w]` tensor.
    pub tensor: Tensor,
}

impl Visuals {
    /// Builds official-style visual masks from box prompts.
    ///
    /// `source_size` is the original image size. `scale_factor` maps
    /// source-image coordinates to the visual-mask feature map, matching
    /// Ultralytics `LoadVisualPrompt.make_mask`. Each box is rasterized to a
    /// filled rectangle, then same-class prompts are merged and classes are
    /// sorted by numeric class id.
    pub fn from_boxes(
        prompts: &[Visual],
        source_size: crate::ImageSize,
        scale_factor: f32,
        device: &Device,
    ) -> crate::Result<Self> {
        validate_visual_prompt_source(prompts, VisualKind::Box)?;
        let (mask_h, mask_w) = visual_prompt_mask_size(source_size, scale_factor)?;
        let mut masks = vec![0f32; prompts.len() * mask_h * mask_w];
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let x1 = (prompt.xyxy[0] * scale_factor).ceil().max(0.0) as usize;
            let y1 = (prompt.xyxy[1] * scale_factor).ceil().max(0.0) as usize;
            let x2 = (prompt.xyxy[2] * scale_factor).ceil().max(0.0) as usize;
            let y2 = (prompt.xyxy[3] * scale_factor).ceil().max(0.0) as usize;
            for y in y1.min(mask_h)..y2.min(mask_h) {
                let row = prompt_index * mask_h * mask_w + y * mask_w;
                for x in x1.min(mask_w)..x2.min(mask_w) {
                    masks[row + x] = 1.0;
                }
            }
        }
        let masks = Tensor::from_vec(masks, (1, prompts.len(), mask_h, mask_w), device)?;
        merge_visual_prompt_masks(prompts, &masks)
    }

    /// Builds official-style visual masks from segmentation prompt masks.
    ///
    /// `prompt_masks` may have shape `[prompts, height, width]` or
    /// `[1, prompts, height, width]`. `scale_factor` resizes masks to the
    /// visual feature-map resolution with nearest-neighbor sampling, then
    /// same-class masks are merged and classes are sorted by class id.
    pub fn from_masks(
        prompts: &[Visual],
        prompt_masks: &Tensor,
        scale_factor: f32,
    ) -> crate::Result<Self> {
        validate_visual_prompt_source(prompts, VisualKind::Mask)?;
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt scale_factor must be positive and finite".to_string(),
            ));
        }
        let masks = match prompt_masks.dims() {
            [prompts_len, height, width] => {
                prompt_masks.reshape((1, *prompts_len, *height, *width))?
            }
            [1, _, _, _] => prompt_masks.clone(),
            dims => {
                return Err(crate::Error::InvalidTensor(format!(
                    "YOLOE source prompt masks must have shape [prompts, height, width] or [1, prompts, height, width], got {dims:?}"
                )));
            }
        };
        let (_, _, height, width) = masks.dims4()?;
        let mask_h = scaled_mask_dim(height, scale_factor)?;
        let mask_w = scaled_mask_dim(width, scale_factor)?;
        let masks = if (height, width) == (mask_h, mask_w) {
            masks
        } else {
            masks.upsample_nearest2d(mask_h, mask_w)?
        };
        merge_visual_prompt_masks(prompts, &masks)
    }
}

fn visual_prompt_mask_size(
    source_size: crate::ImageSize,
    scale_factor: f32,
) -> crate::Result<(usize, usize)> {
    if source_size.width == 0 || source_size.height == 0 {
        return Err(crate::Error::InvalidConfig(
            "YOLOE visual prompt source size must be non-zero".to_string(),
        ));
    }
    Ok((
        scaled_mask_dim(source_size.height, scale_factor)?,
        scaled_mask_dim(source_size.width, scale_factor)?,
    ))
}
