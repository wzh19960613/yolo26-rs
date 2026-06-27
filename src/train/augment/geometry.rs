//! Pure box-coordinate transforms shared by all geometric augmentations.
//!
//! Boxes are `xyxy` in model-input pixel coordinates `[0, width] x [0, height]`.

/// Mirrors a box horizontally across a canvas of `width` pixels.
pub(crate) fn flip_horizontal(box_xyxy: &mut [f32; 4], width: f32) {
    let new_x_min = width - box_xyxy[2];
    let new_x_max = width - box_xyxy[0];
    box_xyxy[0] = new_x_min;
    box_xyxy[2] = new_x_max;
}

/// Mirrors a box vertically across a canvas of `height` pixels.
pub(crate) fn flip_vertical(box_xyxy: &mut [f32; 4], height: f32) {
    let new_y_min = height - box_xyxy[3];
    let new_y_max = height - box_xyxy[1];
    box_xyxy[1] = new_y_min;
    box_xyxy[3] = new_y_max;
}

/// Clamps a box to `[0, width] x [0, height]`.
///
/// Returns `false` when the clamped box has non-positive area so the caller can
/// drop the now-empty object.
pub(crate) fn clamp_to_canvas(box_xyxy: &mut [f32; 4], width: f32, height: f32) -> bool {
    box_xyxy[0] = box_xyxy[0].clamp(0.0, width);
    box_xyxy[2] = box_xyxy[2].clamp(0.0, width);
    box_xyxy[1] = box_xyxy[1].clamp(0.0, height);
    box_xyxy[3] = box_xyxy[3].clamp(0.0, height);
    box_xyxy[2] > box_xyxy[0] && box_xyxy[3] > box_xyxy[1]
}
