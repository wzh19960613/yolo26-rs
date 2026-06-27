//! Affine scale/translate applied to segmentation masks via nearest resampling.

use super::affine::{AffinePlan, Placement, place_resized};
use super::mask_resample::nearest_resample;
use candle_core::Tensor;

/// Applies an image-space affine `plan` to a `[1, C, mask_h, mask_w]` mask.
///
/// The mask is nearest-resampled by the plan scale (labels preserved, no
/// bilinear blending) and placed on the mask canvas at the plan translation
/// scaled to mask resolution (`dx / ratio_x`, `dy / ratio_y`), padding the
/// background with 0. An identity plan returns the mask unchanged. This keeps
/// the mask aligned with [`apply_affine_image`](super::affine::apply_affine_image).
pub(crate) fn apply_affine_mask(
    mask: &Tensor,
    plan: AffinePlan,
    image_h: usize,
    image_w: usize,
) -> crate::Result<Tensor> {
    let dims = mask.dims();
    let (mh, mw) = (dims[2], dims[3]);
    let ratio_h = image_h as f32 / mh as f32;
    let ratio_w = image_w as f32 / mw as f32;
    let mask_plan = AffinePlan {
        s_w: plan.s_w,
        s_h: plan.s_h,
        dx: plan.dx / ratio_w,
        dy: plan.dy / ratio_h,
    };
    if mask_plan.s_w == 1.0 && mask_plan.s_h == 1.0 && mask_plan.dx == 0.0 && mask_plan.dy == 0.0 {
        return Ok(mask.clone());
    }
    let rmh = ((mh as f32) * mask_plan.s_h).round().max(1.0) as usize;
    let rmw = ((mw as f32) * mask_plan.s_w).round().max(1.0) as usize;
    let resized = nearest_resample(mask, rmh, rmw)?;
    let top_src = (-(mask_plan.dy)).max(0.0) as usize;
    let left_src = (-(mask_plan.dx)).max(0.0) as usize;
    let top_dst = mask_plan.dy.max(0.0) as usize;
    let left_dst = mask_plan.dx.max(0.0) as usize;
    place_resized(
        &resized,
        Placement {
            canvas_h: mh,
            canvas_w: mw,
            top_dst,
            left_dst,
            top_src,
            left_src,
            pad_value: 0.0,
        },
    )
}
