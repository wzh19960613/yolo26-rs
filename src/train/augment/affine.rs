//! Random scale + translation for letterboxed model-space images.
//!
//! `candle` 0.10 has no affine `grid_sample`, so rotation/shear/perspective
//! (defaulted to zero upstream) are not modeled. Scale is realized with
//! `interpolate2d`, then the resized image is placed onto a constant-pad canvas
//! whose placement offset doubles as translation: oversized results are
//! randomly cropped, undersized results are randomly padded.

use candle_core::{Device, Tensor};

use super::SeededRng;

/// Letterbox pad value used when an undersized image does not cover the canvas.
pub(crate) const PAD_VALUE: f32 = 114.0 / 255.0;

/// Affine map `x' = x * s_w + dx`, `y' = y * s_h + dy` shared by image and boxes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AffinePlan {
    pub(crate) s_w: f32,
    pub(crate) s_h: f32,
    pub(crate) dx: f32,
    pub(crate) dy: f32,
}

/// Placement of a resized tensor inside a padded canvas.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    pub(crate) canvas_h: usize,
    pub(crate) canvas_w: usize,
    pub(crate) top_dst: usize,
    pub(crate) left_dst: usize,
    pub(crate) top_src: usize,
    pub(crate) left_src: usize,
    pub(crate) pad_value: f32,
}

/// Samples an affine plan from `rng` for a `(height, width)` canvas.
pub(crate) fn affine_plan(
    rng: &mut SeededRng,
    height: usize,
    width: usize,
    scale_gain: f32,
    translate_gain: f32,
) -> AffinePlan {
    let s = if scale_gain > 0.0 {
        rng.uniform(1.0 - scale_gain, 1.0 + scale_gain)
    } else {
        1.0
    };
    let rh = ((height as f32) * s).round().max(1.0) as usize;
    let rw = ((width as f32) * s).round().max(1.0) as usize;
    let translate = translate_gain.max(0.0);
    let tx = rng.uniform(0.5 - translate, 0.5 + translate) * width as f32;
    let ty = rng.uniform(0.5 - translate, 0.5 + translate) * height as f32;
    AffinePlan {
        s_w: rw as f32 / width as f32,
        s_h: rh as f32 / height as f32,
        dx: tx - rw as f32 * 0.5,
        dy: ty - rh as f32 * 0.5,
    }
}

/// Applies a plan to the image, returning a `[1, C, H, W]` tensor.
pub(crate) fn apply_affine_image(
    image: &Tensor,
    plan: AffinePlan,
    height: usize,
    width: usize,
) -> crate::Result<Tensor> {
    if plan.s_w == 1.0 && plan.s_h == 1.0 && plan.dx == 0.0 && plan.dy == 0.0 {
        return Ok(image.clone());
    }
    let rh = ((height as f32) * plan.s_h).round().max(1.0) as usize;
    let rw = ((width as f32) * plan.s_w).round().max(1.0) as usize;
    let resized = image.interpolate2d(rh, rw)?;
    let dx = plan.dx.round();
    let dy = plan.dy.round();
    let top_src = (-dy).max(0.0) as usize;
    let left_src = (-dx).max(0.0) as usize;
    let top_dst = dy.max(0.0) as usize;
    let left_dst = dx.max(0.0) as usize;
    place_resized(
        &resized,
        Placement {
            canvas_h: height,
            canvas_w: width,
            top_dst,
            left_dst,
            top_src,
            left_src,
            pad_value: PAD_VALUE,
        },
    )
}

/// Applies a plan to one `xyxy` box in place.
pub(crate) fn affine_box(box_xyxy: &mut [f32; 4], plan: AffinePlan) {
    box_xyxy[0] = box_xyxy[0] * plan.s_w + plan.dx;
    box_xyxy[2] = box_xyxy[2] * plan.s_w + plan.dx;
    box_xyxy[1] = box_xyxy[1] * plan.s_h + plan.dy;
    box_xyxy[3] = box_xyxy[3] * plan.s_h + plan.dy;
}

/// Places `resized` onto a constant-pad canvas of `(1, C, canvas_h, canvas_w)`.
pub(crate) fn place_resized(resized: &Tensor, placement: Placement) -> crate::Result<Tensor> {
    let channels = resized.dim(1)?;
    let rh = resized.dim(2)?;
    let rw = resized.dim(3)?;
    let copy_h = (rh - placement.top_src).min(placement.canvas_h.saturating_sub(placement.top_dst));
    let copy_w =
        (rw - placement.left_src).min(placement.canvas_w.saturating_sub(placement.left_dst));
    let device = resized.device();
    if copy_h == 0 || copy_w == 0 {
        return Ok(Tensor::full(
            placement.pad_value,
            (1, channels, placement.canvas_h, placement.canvas_w),
            device,
        )?);
    }
    let crop =
        resized
            .narrow(2, placement.top_src, copy_h)?
            .narrow(3, placement.left_src, copy_w)?;
    let block = block_with_horizontal_pad(
        &crop,
        channels,
        copy_h,
        placement.canvas_w,
        placement.left_dst,
        placement.pad_value,
    )?;
    let result = vertical_pad(&block, channels, copy_h, placement, device)?;
    Ok(result.contiguous()?)
}

fn block_with_horizontal_pad(
    crop: &Tensor,
    channels: usize,
    copy_h: usize,
    canvas_w: usize,
    left_dst: usize,
    pad_value: f32,
) -> crate::Result<Tensor> {
    let device = crop.device();
    let copy_w = crop.dim(3)?;
    let right = canvas_w - copy_w - left_dst;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left_dst > 0 {
        parts.push(Tensor::full(
            pad_value,
            (1, channels, copy_h, left_dst),
            device,
        )?);
    }
    parts.push(crop.clone());
    if right > 0 {
        parts.push(Tensor::full(
            pad_value,
            (1, channels, copy_h, right),
            device,
        )?);
    }
    Ok(Tensor::cat(&parts, 3)?)
}

fn vertical_pad(
    block: &Tensor,
    channels: usize,
    copy_h: usize,
    placement: Placement,
    device: &Device,
) -> crate::Result<Tensor> {
    let bottom = placement.canvas_h - placement.top_dst - copy_h;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if placement.top_dst > 0 {
        parts.push(Tensor::full(
            placement.pad_value,
            (1, channels, placement.top_dst, placement.canvas_w),
            device,
        )?);
    }
    parts.push(block.clone());
    if bottom > 0 {
        parts.push(Tensor::full(
            placement.pad_value,
            (1, channels, bottom, placement.canvas_w),
            device,
        )?);
    }
    Ok(Tensor::cat(&parts, 2)?)
}
