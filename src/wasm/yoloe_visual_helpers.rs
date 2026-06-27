//! Shared prompt-building helpers for the YOLOE visual wasm entry points,
//! extracted to keep [`super::yoloe_visual`] under the 200-line limit.

use candle_core::Tensor;
use wasm_bindgen::prelude::*;

use crate::yoloe::prompt::Visual;

use super::js_error;

/// Builds one [`Visual`] per 4-tuple in `box_prompts`, assigning sequential
/// class ids `0..n`. Validates that each box has positive width and height.
pub(super) fn build_box_prompts(box_prompts: &[f32]) -> Result<Vec<Visual>, JsValue> {
    if !box_prompts.len().is_multiple_of(4) {
        return Err(js_error(format!(
            "box_prompts length {} is not a multiple of 4",
            box_prompts.len()
        )));
    }
    let mut prompts = Vec::with_capacity(box_prompts.len() / 4);
    for (class_id, chunk) in box_prompts.chunks_exact(4).enumerate() {
        let xyxy = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let visual =
            Visual::from_box(class_id as u32, xyxy).map_err(|err| js_error(err.to_string()))?;
        prompts.push(visual);
    }
    if prompts.is_empty() {
        return Err(js_error(
            "YOLOE visual prompts require at least one exemplar",
        ));
    }
    Ok(prompts)
}

/// Splits a flat `[n_prompts * mask_h * mask_w]` byte buffer into one
/// [`Visual`] per prompt (with xyxy derived from each mask's nonzero extent)
/// and a single `[n_prompts, mask_h, mask_w]` F32 tensor (1.0 inside / 0.0
/// outside) suitable for `VisualSource::Masks`.
pub(super) fn build_mask_prompts(
    mask_data: &[u8],
    mask_w: u32,
    mask_h: u32,
    n_prompts: usize,
) -> Result<(Vec<Visual>, Tensor), JsValue> {
    let area = mask_w as usize * mask_h as usize;
    if mask_data.len() != n_prompts * area {
        return Err(js_error(format!(
            "mask_data length {} != n_prompts {} * mask_w {} * mask_h {} (= {})",
            mask_data.len(),
            n_prompts,
            mask_w,
            mask_h,
            n_prompts * area,
        )));
    }
    if n_prompts == 0 {
        return Err(js_error("YOLOE mask prompts require at least one exemplar"));
    }

    let mut prompts = Vec::with_capacity(n_prompts);
    let mut flat_f32 = Vec::with_capacity(mask_data.len());
    for class_id in 0..n_prompts {
        let slice = &mask_data[class_id * area..(class_id + 1) * area];
        let mut x_min = mask_w as f32;
        let mut y_min = mask_h as f32;
        let mut x_max = 0.0_f32;
        let mut y_max = 0.0_f32;
        let mut any = false;
        for (i, &v) in slice.iter().enumerate() {
            let on = v != 0;
            flat_f32.push(if on { 1.0 } else { 0.0 });
            if on {
                any = true;
                let x = (i % mask_w as usize) as f32;
                let y = (i / mask_w as usize) as f32;
                if x < x_min {
                    x_min = x;
                }
                if x > x_max {
                    x_max = x;
                }
                if y < y_min {
                    y_min = y;
                }
                if y > y_max {
                    y_max = y;
                }
            }
        }
        if !any {
            return Err(js_error(format!(
                "YOLOE mask prompt {class_id} is empty (no nonzero pixels)"
            )));
        }
        let xyxy = [x_min, y_min, x_max + 1.0, y_max + 1.0];
        let visual =
            Visual::from_mask(class_id as u32, xyxy).map_err(|err| js_error(err.to_string()))?;
        prompts.push(visual);
    }

    let mask_tensor = Tensor::from_vec(
        flat_f32,
        (n_prompts, mask_h as usize, mask_w as usize),
        &candle_core::Device::Cpu,
    )
    .map_err(|err| js_error(err.to_string()))?;
    Ok((prompts, mask_tensor))
}
