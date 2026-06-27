//! Shared letterbox helpers for YOLOE visual-prompt mask construction.
//!
//! Box and mask visual prompts are expressed in source-image coordinates, but
//! the official SAVPE encoder consumes feature-scale masks. These helpers
//! remap prompts through the same rect letterbox used for inference, so the
//! detect and segment networks share one mask-building path that returns
//! [`Visuals`].

use candle_core::{DType, Tensor};

use crate::yoloe::prompt::visual::{Visual, VisualKind};
use crate::yoloe::visuals::Visuals;
use crate::yoloe::visuals::merge::merge_visual_prompt_masks;

/// Builds box-prompt [`Visuals`] in letterbox coordinates.
///
/// Each prompt's source-image xyxy is mapped through the letterbox transform,
/// then rasterized to `[1, classes, feature_h, feature_w]`. Returns an error
/// when any prompt is not a [`VisualKind::Box`], matching official
/// `LoadVisualPrompt.make_mask`.
pub(crate) fn box_masks_for_letterbox(
    prompts: &[Visual],
    letterbox: &crate::model::LetterboxInfo,
    feature_h: usize,
    feature_w: usize,
    input: &Tensor,
) -> crate::Result<Visuals> {
    if prompts.iter().any(|prompt| prompt.kind != VisualKind::Box) {
        return Err(crate::Error::InvalidConfig(
            "YOLOE online visual-prompt inference currently accepts box prompts".to_string(),
        ));
    }
    let scale_x = feature_w as f32 / letterbox.model_width as f32;
    let scale_y = feature_h as f32 / letterbox.model_height as f32;
    if (scale_x - scale_y).abs() > 1.0e-6 {
        return Err(crate::Error::InvalidTensor(format!(
            "YOLOE visual-prompt feature scale differs by axis: x={scale_x}, y={scale_y}"
        )));
    }
    let mapped = prompts
        .iter()
        .map(|prompt| {
            Visual::from_box(
                prompt.class_id,
                [
                    prompt.xyxy[0] * letterbox.scale + letterbox.pad_x,
                    prompt.xyxy[1] * letterbox.scale + letterbox.pad_y,
                    prompt.xyxy[2] * letterbox.scale + letterbox.pad_x,
                    prompt.xyxy[3] * letterbox.scale + letterbox.pad_y,
                ],
            )
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let masks = Visuals::from_boxes(
        &mapped,
        crate::ImageSize::new(letterbox.model_width, letterbox.model_height),
        scale_x,
        input.device(),
    )?;
    Ok(masks)
}

/// Builds mask-prompt [`Visuals`] in letterbox coordinates.
///
/// `source_masks` (`[prompts, H, W]` or `[1, prompts, H, W]`) is letterboxed
/// and resampled to `[1, classes, feature_h, feature_w]`. Returns an error
/// when any prompt is not a [`VisualKind::Mask`].
pub(crate) fn mask_prompts_for_letterbox(
    prompts: &[Visual],
    source_masks: &Tensor,
    letterbox: &crate::model::LetterboxInfo,
    feature_h: usize,
    feature_w: usize,
) -> crate::Result<Visuals> {
    crate::yoloe::prompt::visual::validate_visual_prompt_source(prompts, VisualKind::Mask)?;
    let source_masks = source_masks_3d(source_masks)?;
    let (_, source_h, source_w) = source_masks.dims3()?;
    let letterboxed = letterbox_source_masks(&source_masks, letterbox, source_h, source_w)?;
    let rescaled = rescale_to_feature(&letterboxed, letterbox, feature_h, feature_w)?;
    merge_visual_prompt_masks(prompts, &rescaled)
}

/// Upsamples the letterboxed masks to the feature resolution, or returns them
/// unchanged when already at feature scale.
fn rescale_to_feature(
    letterboxed: &Tensor,
    letterbox: &crate::model::LetterboxInfo,
    feature_h: usize,
    feature_w: usize,
) -> crate::Result<Tensor> {
    let scale_h = feature_h as f32 / letterbox.model_height as f32;
    let scale_w = feature_w as f32 / letterbox.model_width as f32;
    if (scale_h - scale_w).abs() > 1.0e-6 {
        return Err(crate::Error::InvalidTensor(format!(
            "YOLOE visual-prompt feature scale differs by axis: h={scale_h}, w={scale_w}"
        )));
    }
    let (_, _, height, width) = letterboxed.dims4()?;
    let mask_h = crate::yoloe::prompt::visual::scaled_mask_dim(height, scale_h)?;
    let mask_w = crate::yoloe::prompt::visual::scaled_mask_dim(width, scale_w)?;
    if (height, width) == (mask_h, mask_w) {
        Ok(letterboxed.clone())
    } else {
        letterboxed
            .upsample_nearest2d(mask_h, mask_w)
            .map_err(Into::into)
    }
}

fn source_masks_3d(source_masks: &Tensor) -> crate::Result<Tensor> {
    match source_masks.dims() {
        [_, _, _] => source_masks.to_dtype(DType::F32).map_err(Into::into),
        [1, prompts, height, width] => source_masks
            .reshape((*prompts, *height, *width))?
            .to_dtype(DType::F32)
            .map_err(Into::into),
        dims => Err(crate::Error::InvalidTensor(format!(
            "YOLOE source visual prompt masks must have shape [prompts, H, W] or [1, prompts, H, W], got {dims:?}"
        ))),
    }
}

fn letterbox_source_masks(
    masks: &Tensor,
    letterbox: &crate::model::LetterboxInfo,
    source_h: usize,
    source_w: usize,
) -> crate::Result<Tensor> {
    let prompts = masks.dim(0)?;
    let values = masks.to_vec3::<f32>()?;
    let new_w = (source_w as f32 * letterbox.scale).round().max(1.0) as usize;
    let new_h = (source_h as f32 * letterbox.scale).round().max(1.0) as usize;
    let pad_x = letterbox.pad_x.floor() as usize;
    let pad_y = letterbox.pad_y.floor() as usize;
    let plane = letterbox.model_height * letterbox.model_width;
    let mut out = vec![0.0f32; prompts * plane];
    for p in 0..prompts {
        for y in 0..new_h {
            let (y0, y1, fy) = resize_axis(y, source_h, new_h);
            for x in 0..new_w {
                let (x0, x1, fx) = resize_axis(x, source_w, new_w);
                let value = bilinear(
                    values[p][y0][x0],
                    values[p][y0][x1],
                    values[p][y1][x0],
                    values[p][y1][x1],
                    fx,
                    fy,
                );
                out[p * plane + (y + pad_y) * letterbox.model_width + x + pad_x] = value;
            }
        }
    }
    Tensor::from_vec(
        out,
        (1, prompts, letterbox.model_height, letterbox.model_width),
        masks.device(),
    )
    .map_err(Into::into)
}

fn resize_axis(dst: usize, source: usize, target: usize) -> (usize, usize, f32) {
    let pos = (dst as f32 + 0.5) * source as f32 / target as f32 - 0.5;
    let start = pos.floor().max(0.0) as usize;
    let end = (start + 1).min(source - 1);
    (start, end, (pos - start as f32).max(0.0))
}

fn bilinear(v00: f32, v01: f32, v10: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}
