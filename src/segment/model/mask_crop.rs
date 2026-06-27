//! Bounding-box constrained mask cropping helpers.

use crate::BBox;

pub(super) fn crop_start(pad: f32) -> usize {
    (pad - 0.1).round().max(0.0) as usize
}

pub(super) fn crop_end(pad: f32) -> usize {
    (pad + 0.1).round().max(0.0) as usize
}

fn fill_neg_inf(logits: &mut [f32], base: usize, from: usize, to: usize) {
    logits[base + from..base + to].fill(f32::NEG_INFINITY);
}

pub(super) fn fill_outside_bbox(
    logits: &mut [f32],
    stride: usize,
    rows: usize,
    (x_min, y_min, x_max, y_max): (usize, usize, usize, usize),
) {
    for row in 0..y_min {
        fill_neg_inf(logits, row * stride, 0, stride);
    }
    for row in y_max..rows {
        fill_neg_inf(logits, row * stride, 0, stride);
    }
    for row in y_min..y_max {
        fill_neg_inf(logits, row * stride, 0, x_min);
        fill_neg_inf(logits, row * stride, x_max, stride);
    }
}

pub(super) fn crop_source_mask_logits(logits: &mut [f32], width: usize, height: usize, bbox: BBox) {
    let x_min = bbox.x_min.round().clamp(0.0, width as f32) as usize;
    let y_min = bbox.y_min.round().clamp(0.0, height as f32) as usize;
    let x_max = bbox.x_max.round().clamp(0.0, width as f32) as usize;
    let y_max = bbox.y_max.round().clamp(0.0, height as f32) as usize;

    fill_outside_bbox(logits, width, height, (x_min, y_min, x_max, y_max));
}
