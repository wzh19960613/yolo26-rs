//! Semantic segmentation mask decoding helpers.
//!
//! Reads a grayscale class-mask image and resamples it into a dense class-id
//! tensor aligned with the model's letterboxed output grid.

use super::*;

/// Decodes a semantic class-mask image file into a grayscale image.
pub(crate) fn read_semantic_mask(path: &Path) -> crate::Result<image::GrayImage> {
    Ok(image::open(path)
        .map_err(|err| {
            crate::Error::InvalidImage(format!(
                "failed to decode semantic mask {}: {err}",
                path.display()
            ))
        })?
        .to_luma8())
}

/// Resamples a grayscale class mask into a dense class-id tensor at
/// `output_size`, mapping through the letterbox geometry back to source space.
pub(crate) fn semantic_class_map_from_mask(
    mask: &image::GrayImage,
    source_size: (usize, usize),
    letterbox: &crate::model::LetterboxInfo,
    output_size: ImageSize,
    device: &Device,
) -> crate::Result<Tensor> {
    let (source_w, source_h) = source_size;
    if source_w == 0 || source_h == 0 || mask.width() == 0 || mask.height() == 0 {
        return Err(crate::Error::InvalidImage(
            "semantic source image and mask dimensions must be greater than zero".to_string(),
        ));
    }

    let mask_w = mask.width() as usize;
    let mask_h = mask.height() as usize;
    let mut class_ids = vec![0u32; output_size.width * output_size.height];
    for y in 0..output_size.height {
        let model_y = (y as f32 + 0.5) * letterbox.model_height as f32 / output_size.height as f32;
        let source_y = letterbox.to_source_y(model_y);
        if !(0.0..source_h as f32).contains(&source_y) {
            continue;
        }
        let mask_y = (source_y * mask_h as f32 / source_h as f32)
            .floor()
            .clamp(0.0, (mask_h - 1) as f32) as u32;

        for x in 0..output_size.width {
            let model_x =
                (x as f32 + 0.5) * letterbox.model_width as f32 / output_size.width as f32;
            let source_x = letterbox.to_source_x(model_x);
            if !(0.0..source_w as f32).contains(&source_x) {
                continue;
            }
            let mask_x = (source_x * mask_w as f32 / source_w as f32)
                .floor()
                .clamp(0.0, (mask_w - 1) as f32) as u32;
            class_ids[y * output_size.width + x] = mask.get_pixel(mask_x, mask_y).0[0] as u32;
        }
    }

    Tensor::from_vec(
        class_ids,
        (1, output_size.height, output_size.width),
        device,
    )
    .map_err(Into::into)
}
