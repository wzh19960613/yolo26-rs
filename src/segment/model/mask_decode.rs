//! Prototype matmul and per-instance mask tensor assembly.

use candle_core::{DType, Tensor};

use crate::model::LetterboxInfo;
use crate::{BBox, Result};

use crate::segment::Mask;

use super::mask_crop::{crop_source_mask_logits, fill_outside_bbox};

pub(super) struct MaskDecodeBase<'a> {
    pub(super) proto_shape: (usize, usize),
    pub(super) channels: usize,
    pub(super) coefficients: &'a [f32],
    pub(super) bbox: BBox,
    pub(super) letterbox: &'a LetterboxInfo,
}

pub(super) struct CroppedMaskSpec<'a> {
    pub(super) base: MaskDecodeBase<'a>,
    pub(super) crop_origin: (usize, usize),
    pub(super) content_size: (usize, usize),
}

pub(super) struct NativeMaskSpec<'a> {
    pub(super) base: MaskDecodeBase<'a>,
    pub(super) image_size: (u32, u32),
}

pub(super) fn instance_mask_tensor(proto_2d: &Tensor, spec: CroppedMaskSpec<'_>) -> Result<Mask> {
    let (proto_h, proto_w) = spec.base.proto_shape;
    let (crop_x, crop_y) = spec.crop_origin;
    let (content_w, content_h) = spec.content_size;
    let device = proto_2d.device();
    let coeff_t = Tensor::from_vec(
        spec.base.coefficients.to_vec(),
        (1, spec.base.channels),
        device,
    )?
    .to_dtype(proto_2d.dtype())?;
    let logits = coeff_t.matmul(proto_2d)?;
    let logits = logits.reshape((1, 1, proto_h, proto_w))?;
    let logits = logits
        .narrow(3, crop_x, content_w)?
        .narrow(2, crop_y, content_h)?;

    let mut logits = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    let scale_x = content_w as f32 / spec.base.letterbox.content_width();
    let scale_y = content_h as f32 / spec.base.letterbox.content_height();
    let bx_min = (spec.base.bbox.x_min * scale_x).max(0.0) as usize;
    let by_min = (spec.base.bbox.y_min * scale_y).max(0.0) as usize;
    let bx_max = (spec.base.bbox.x_max * scale_x).min(content_w as f32) as usize;
    let by_max = (spec.base.bbox.y_max * scale_y).min(content_h as f32) as usize;

    fill_outside_bbox(
        &mut logits,
        content_w,
        content_h,
        (bx_min, by_min, bx_max, by_max),
    );

    Mask::new(content_w as u16, content_h as u16, logits)
}

pub(super) fn instance_mask_tensor_native(
    proto_2d: &Tensor,
    spec: NativeMaskSpec<'_>,
) -> Result<Mask> {
    let (proto_h, proto_w) = spec.base.proto_shape;
    let (image_width, image_height) = spec.image_size;
    let width = image_width as usize;
    let height = image_height as usize;
    let device = proto_2d.device();
    let coeff_t = Tensor::from_vec(
        spec.base.coefficients.to_vec(),
        (1, spec.base.channels),
        device,
    )?
    .to_dtype(proto_2d.dtype())?;
    let logits = coeff_t
        .matmul(proto_2d)?
        .reshape((1, 1, proto_h, proto_w))?;
    let logits = scale_mask_logits_to_source(
        logits,
        (proto_w, proto_h),
        (width, height),
        spec.base.letterbox,
    )?;
    let mut logits = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    crop_source_mask_logits(&mut logits, width, height, spec.base.bbox);
    Mask::new(image_width as u16, image_height as u16, logits)
}

fn scale_mask_logits_to_source(
    logits: Tensor,
    (proto_w, proto_h): (usize, usize),
    (image_width, image_height): (usize, usize),
    letterbox: &LetterboxInfo,
) -> Result<Tensor> {
    use super::mask_crop::{crop_end, crop_start};
    let pad_x = letterbox.feature_pad_x(proto_w);
    let pad_y = letterbox.feature_pad_y(proto_h);

    let left = crop_start(pad_x).min(proto_w);
    let top = crop_start(pad_y).min(proto_h);
    let right = proto_w.saturating_sub(crop_end(pad_x).min(proto_w));
    let bottom = proto_h.saturating_sub(crop_end(pad_y).min(proto_h));
    if right <= left || bottom <= top {
        return Err(crate::Error::InvalidTensor(
            "mask padding removed the entire prototype map".to_string(),
        ));
    }

    let cropped = logits
        .narrow(3, left, right - left)?
        .narrow(2, top, bottom - top)?;
    Ok(cropped.upsample_bilinear2d(image_height, image_width, false)?)
}
