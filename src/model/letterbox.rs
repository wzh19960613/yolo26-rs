use candle_core::{DType, Device, Tensor};

use super::ImageSize;
use crate::BBox;
use crate::{Image, Result};

/// Default square YOLO26 model input size.
pub const MODEL_INPUT_SIZE: usize = 640;
const LETTERBOX_PAD_VALUE: f32 = 114.0 / 255.0;

/// Geometry used to map model-space coordinates back to the source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxInfo {
    /// Resize scale applied to the source image.
    pub scale: f32,
    /// Horizontal padding added in model coordinates.
    pub pad_x: f32,
    /// Vertical padding added in model coordinates.
    pub pad_y: f32,
    /// Width of the model input tensor.
    pub model_width: usize,
    /// Height of the model input tensor.
    pub model_height: usize,
}

impl LetterboxInfo {
    /// Transforms a model-space x coordinate to source-image coordinates.
    pub fn to_source_x(self, x: f32) -> f32 {
        (x - self.pad_x) / self.scale
    }

    /// Transforms a model-space y coordinate to source-image coordinates.
    pub fn to_source_y(self, y: f32) -> f32 {
        (y - self.pad_y) / self.scale
    }

    /// Transforms a model-space size (width or height) to source-image size.
    pub fn to_source_size(self, s: f32) -> f32 {
        s / self.scale
    }

    /// Maps model-space `[x1, y1, x2, y2]` to a clamped source-image bounding box.
    pub fn xyxy_to_source_bbox(&self, xyxy: &[f32], image_width: u32, image_height: u32) -> BBox {
        BBox::from_xyxy(
            self.to_source_x(xyxy[0]),
            self.to_source_y(xyxy[1]),
            self.to_source_x(xyxy[2]),
            self.to_source_y(xyxy[3]),
        )
        .clamp(image_width, image_height)
    }

    /// Source-image content width (excluding letterbox padding) in source pixels.
    pub fn content_width(&self) -> f32 {
        (self.model_width as f32 - 2.0 * self.pad_x) / self.scale
    }

    /// Source-image content height (excluding letterbox padding) in source pixels.
    pub fn content_height(&self) -> f32 {
        (self.model_height as f32 - 2.0 * self.pad_y) / self.scale
    }

    /// Horizontal padding expressed in feature-map coordinates.
    pub(crate) fn feature_pad_x(&self, feature_w: usize) -> f32 {
        self.pad_x * feature_w as f32 / self.model_width as f32
    }

    /// Vertical padding expressed in feature-map coordinates.
    pub(crate) fn feature_pad_y(&self, feature_h: usize) -> f32 {
        self.pad_y * feature_h as f32 / self.model_height as f32
    }

    /// Maps source-image width to feature-map width.
    pub(crate) fn source_to_feature_w(&self, source_w: u32, feature_w: usize) -> usize {
        (source_w as f32 * self.scale * feature_w as f32 / self.model_width as f32).round() as usize
    }

    /// Maps source-image height to feature-map height.
    pub(crate) fn source_to_feature_h(&self, source_h: u32, feature_h: usize) -> usize {
        (source_h as f32 * self.scale * feature_h as f32 / self.model_height as f32).round()
            as usize
    }
}

/// Resizes and pads an image into a normalized NCHW tensor.
pub fn letterbox(
    image: &Image,
    target: ImageSize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, LetterboxInfo)> {
    if target.width == 0 || target.height == 0 {
        return Err(crate::Error::InvalidConfig(
            "letterbox target dimensions must be greater than zero".to_string(),
        ));
    }

    let src_w = image.width as usize;
    let src_h = image.height as usize;
    let scale = f32::min(
        target.width as f32 / src_w as f32,
        target.height as f32 / src_h as f32,
    );
    let new_w = (src_w as f32 * scale).round().max(1.0) as usize;
    let new_h = (src_h as f32 * scale).round().max(1.0) as usize;
    let pad_x = (target.width as f32 - new_w as f32) / 2.0;
    let pad_y = (target.height as f32 - new_h as f32) / 2.0;
    let pad_x_i = pad_x.floor() as usize;
    let pad_y_i = pad_y.floor() as usize;

    let plane = target.width * target.height;
    let mut chw = vec![LETTERBOX_PAD_VALUE; 3 * plane];

    for dst_y in 0..new_h {
        let src_y = (dst_y as f32 + 0.5) * src_h as f32 / new_h as f32 - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        // Precompute row base offsets (3 bytes/pixel) to avoid re-indexing per pixel.
        let row0 = y0 * src_w * 3;
        let row1 = y1 * src_w * 3;
        let data = &image.data;

        for dst_x in 0..new_w {
            let src_x = (dst_x as f32 + 0.5) * src_w as f32 / new_w as f32 - 0.5;
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

            let out = (dst_y + pad_y_i) * target.width + dst_x + pad_x_i;
            for c in 0..3 {
                let value = data[i00 + c] as f32 * w00
                    + data[i01 + c] as f32 * w01
                    + data[i10 + c] as f32 * w10
                    + data[i11 + c] as f32 * w11;
                chw[c * plane + out] = value / 255.0;
            }
        }
    }

    let tensor =
        Tensor::from_vec(chw, (1, 3, target.height, target.width), device)?.to_dtype(dtype)?;
    Ok((
        tensor,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
            model_width: target.width,
            model_height: target.height,
        },
    ))
}

pub(crate) fn letterbox_with_canvas(
    image: &Image,
    resize: ImageSize,
    canvas: ImageSize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, LetterboxInfo)> {
    if resize.width == 0 || resize.height == 0 || canvas.width == 0 || canvas.height == 0 {
        return Err(crate::Error::InvalidConfig(
            "letterbox resize and canvas dimensions must be greater than zero".to_string(),
        ));
    }

    let src_w = image.width as usize;
    let src_h = image.height as usize;
    let scale = f32::min(
        resize.width as f32 / src_w as f32,
        resize.height as f32 / src_h as f32,
    );
    let new_w = (src_w as f32 * scale).round().max(1.0) as usize;
    let new_h = (src_h as f32 * scale).round().max(1.0) as usize;
    if new_w > canvas.width || new_h > canvas.height {
        return Err(crate::Error::InvalidConfig(format!(
            "letterbox canvas {}x{} is smaller than resized image {}x{}",
            canvas.width, canvas.height, new_w, new_h
        )));
    }
    let pad_x = ((canvas.width - new_w) as f32 / 2.0 - 0.1).round().max(0.0);
    let pad_y = ((canvas.height - new_h) as f32 / 2.0 - 0.1)
        .round()
        .max(0.0);
    let pad_x_i = pad_x.floor() as usize;
    let pad_y_i = pad_y.floor() as usize;

    let plane = canvas.width * canvas.height;
    let mut chw = vec![LETTERBOX_PAD_VALUE; 3 * plane];

    for dst_y in 0..new_h {
        let src_y = (dst_y as f32 + 0.5) * src_h as f32 / new_h as f32 - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        let row0 = y0 * src_w * 3;
        let row1 = y1 * src_w * 3;
        let data = &image.data;

        for dst_x in 0..new_w {
            let src_x = (dst_x as f32 + 0.5) * src_w as f32 / new_w as f32 - 0.5;
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

            let out = (dst_y + pad_y_i) * canvas.width + dst_x + pad_x_i;
            for c in 0..3 {
                let value = data[i00 + c] as f32 * w00
                    + data[i01 + c] as f32 * w01
                    + data[i10 + c] as f32 * w10
                    + data[i11 + c] as f32 * w11;
                chw[c * plane + out] = value / 255.0;
            }
        }
    }

    let tensor =
        Tensor::from_vec(chw, (1, 3, canvas.height, canvas.width), device)?.to_dtype(dtype)?;
    Ok((
        tensor,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
            model_width: canvas.width,
            model_height: canvas.height,
        },
    ))
}
