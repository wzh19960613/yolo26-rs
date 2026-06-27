//! Classification training image preprocessing.
//!
//! Hand-rolled bilinear resize + center-crop + normalize to [0,1] producing a
//! `(1, 3, H, W)` tensor for the classification training path. Semantic mask
//! decoding and YOLO detection label parsing live in their own modules
//! ([`crate::train::eval::semantic_mask_decode`] and [`crate::train::dataset::detection_label`]).

use super::*;

/// Letterbox-free classification preprocess: bilinear resize to cover the
/// target, center-crop to exact target dims, then scale to [0,1].
pub(crate) fn classify_train_preprocess(
    image: &crate::Image,
    target: ImageSize,
    dtype: DType,
    device: &Device,
) -> crate::Result<Tensor> {
    if target.width == 0 || target.height == 0 {
        return Err(crate::Error::InvalidConfig(
            "classification target dimensions must be greater than zero".to_string(),
        ));
    }

    let src_w = image.width as usize;
    let src_h = image.height as usize;
    let scale = f32::max(
        target.width as f32 / src_w as f32,
        target.height as f32 / src_h as f32,
    );
    let resized_w = (src_w as f32 * scale).round().max(target.width as f32) as usize;
    let resized_h = (src_h as f32 * scale).round().max(target.height as f32) as usize;
    let crop_x = (resized_w - target.width) as f32 * 0.5;
    let crop_y = (resized_h - target.height) as f32 * 0.5;

    let plane = target.width * target.height;
    let mut chw = vec![0.0f32; 3 * plane];

    for dst_y in 0..target.height {
        let resized_y = dst_y as f32 + crop_y;
        let src_y = (resized_y + 0.5) * src_h as f32 / resized_h as f32 - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        let row0 = y0 * src_w * 3;
        let row1 = y1 * src_w * 3;
        let data = &image.data;

        for dst_x in 0..target.width {
            let resized_x = dst_x as f32 + crop_x;
            let src_x = (resized_x + 0.5) * src_w as f32 / resized_w as f32 - 0.5;
            let x0 = src_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = (src_x - x0 as f32).max(0.0);

            let i00 = row0 + x0 * 3;
            let i01 = row0 + x1 * 3;
            let i10 = row1 + x0 * 3;
            let i11 = row1 + x1 * 3;
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w01 = fx * (1.0 - fy);
            let w10 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let out = dst_y * target.width + dst_x;
            for c in 0..3 {
                let value = data[i00 + c] as f32 * w00
                    + data[i01 + c] as f32 * w01
                    + data[i10 + c] as f32 * w10
                    + data[i11 + c] as f32 * w11;
                chw[c * plane + out] = value / 255.0;
            }
        }
    }

    Ok(Tensor::from_vec(chw, (1, 3, target.height, target.width), device)?.to_dtype(dtype)?)
}
