use candle_core::{DType, Device, Tensor};

use crate::{Image, Result};

use super::{ImageSize, LetterboxInfo};

const PAD_VALUE: f32 = 114.0 / 255.0;

pub(crate) fn letterbox_rect(
    image: &Image,
    target: ImageSize,
    stride: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, LetterboxInfo)> {
    if stride == 0 {
        return Err(crate::Error::InvalidConfig(
            "letterbox stride must be greater than zero".to_string(),
        ));
    }
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
    let pad_w = (target.width - new_w) % stride;
    let pad_h = (target.height - new_h) % stride;
    let model_w = new_w + pad_w;
    let model_h = new_h + pad_h;
    let pad_x = top_left_pad(pad_w);
    let pad_y = top_left_pad(pad_h);
    let chw = resize_into_chw(image, new_w, new_h, model_w, model_h, pad_x, pad_y);
    let tensor = Tensor::from_vec(chw, (1, 3, model_h, model_w), device)?.to_dtype(dtype)?;
    Ok((
        tensor,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
            model_width: model_w,
            model_height: model_h,
        },
    ))
}

fn top_left_pad(pad: usize) -> f32 {
    (pad as f32 / 2.0 - 0.1).round().max(0.0)
}

fn resize_into_chw(
    image: &Image,
    new_w: usize,
    new_h: usize,
    model_w: usize,
    model_h: usize,
    pad_x: f32,
    pad_y: f32,
) -> Vec<f32> {
    let src_w = image.width as usize;
    let src_h = image.height as usize;
    let plane = model_w * model_h;
    let mut chw = vec![PAD_VALUE; 3 * plane];
    let pad_x_i = pad_x.floor() as usize;
    let pad_y_i = pad_y.floor() as usize;
    let mut target = ResizeTarget {
        image,
        chw: &mut chw,
        model_w,
        plane,
    };
    for dst_y in 0..new_h {
        let src_y = (dst_y as f32 + 0.5) * src_h as f32 / new_h as f32 - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        for dst_x in 0..new_w {
            let src_x = (dst_x as f32 + 0.5) * src_w as f32 / new_w as f32 - 0.5;
            let x0 = src_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = (src_x - x0 as f32).max(0.0);
            target.write_pixel(
                dst_x + pad_x_i,
                dst_y + pad_y_i,
                BilinearSample {
                    xs: [x0, x1],
                    ys: [y0, y1],
                    fx,
                    fy,
                },
            );
        }
    }
    chw
}

struct ResizeTarget<'a> {
    image: &'a Image,
    chw: &'a mut [f32],
    model_w: usize,
    plane: usize,
}

struct BilinearSample {
    xs: [usize; 2],
    ys: [usize; 2],
    fx: f32,
    fy: f32,
}

impl ResizeTarget<'_> {
    fn write_pixel(&mut self, dst_x: usize, dst_y: usize, sample: BilinearSample) {
        let src_w = self.image.width as usize;
        let data = &self.image.data;
        let row0 = sample.ys[0] * src_w * 3;
        let row1 = sample.ys[1] * src_w * 3;
        let i00 = row0 + sample.xs[0] * 3;
        let i01 = row0 + sample.xs[1] * 3;
        let i10 = row1 + sample.xs[0] * 3;
        let i11 = row1 + sample.xs[1] * 3;
        let w00 = (1.0 - sample.fx) * (1.0 - sample.fy);
        let w01 = sample.fx * (1.0 - sample.fy);
        let w10 = (1.0 - sample.fx) * sample.fy;
        let w11 = sample.fx * sample.fy;
        let out = dst_y * self.model_w + dst_x;
        for c in 0..3 {
            let value = data[i00 + c] as f32 * w00
                + data[i01 + c] as f32 * w01
                + data[i10 + c] as f32 * w10
                + data[i11 + c] as f32 * w11;
            self.chw[c * self.plane + out] = value / 255.0;
        }
    }
}
