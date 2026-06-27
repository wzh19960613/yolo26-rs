//! Per-task affine (scale + translate) target transforms.
//!
//! The affine plan `x' = x * s_w + dx`, `y' = y * s_h + dy` is applied to every
//! spatial target so the target stays aligned with the affined image. Detection
//! boxes use the shared box transform; segmentation masks are nearest-resampled
//! and placed on the mask canvas; pose keypoints scale/translate like box
//! coordinates; OBB boxes transform while the angle is invariant under the
//! modeled uniform-scale + translate affine (rotation/shear default to zero).

use candle_core::Tensor;

use super::affine::AffinePlan;
use super::affine::affine_box;
use super::affine_mask::apply_affine_mask;
use super::geometry::clamp_to_canvas;
use super::{DetectionTargets, ObbTargets, PoseTargets, SegmentationTargets, Target};
use crate::train::SegmentationMaskEncoding;

const DETECTION_AREA_THRESHOLD: f32 = 0.10;
const SEGMENT_AREA_THRESHOLD: f32 = 0.01;

/// Applies an affine `plan` to a training target, keeping it aligned with the
/// affined image. Non-spatial targets (classification, dense) pass through.
pub(crate) fn affine_target(
    target: Target,
    plan: AffinePlan,
    width: f32,
    height: f32,
) -> crate::Result<Target> {
    Ok(match target {
        Target::Detection(t) => Target::Detection(
            affine_detection(t, plan, width, height, DETECTION_AREA_THRESHOLD)?.targets,
        ),
        Target::Segmentation(t) => {
            Target::Segmentation(affine_segmentation(t, plan, width, height)?)
        }
        Target::Pose(t) => Target::Pose(affine_pose(t, plan, width, height)?),
        Target::Obb(t) => Target::Obb(affine_obb(t, plan, width, height)?),
        Target::Semantic { class_map } => Target::Semantic {
            class_map: affine_semantic(class_map, plan, height as usize, width as usize)?,
        },
        other => other,
    })
}

/// Applies an image-space affine `plan` to a semantic class map.
///
/// The `[batch, h, w]` class map is treated as a per-batch mask stack, nearest-
/// resampled (class ids preserved) and placed on the canvas with the plan
/// translation scaled to class-map resolution, padding the background with 0.
fn affine_semantic(
    class_map: Tensor,
    plan: AffinePlan,
    image_h: usize,
    image_w: usize,
) -> crate::Result<Tensor> {
    let (batch, h, w) = class_map.dims3()?;
    let dtype = class_map.dtype();
    let as4 = class_map.reshape((batch, 1, h, w))?;
    let mut out = Vec::with_capacity(batch * h * w);
    for b in 0..batch {
        // Run the affine in f32 (place_resized pads with an f32 constant), then
        // restore the class-map dtype at the end so integer class ids survive.
        let plane = as4.narrow(0, b, 1)?.to_dtype(candle_core::DType::F32)?;
        let placed = super::affine_mask::apply_affine_mask(&plane, plan, image_h, image_w)?;
        out.extend(placed.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(Tensor::from_vec(out, (batch, h, w), class_map.device())?.to_dtype(dtype)?)
}

fn affine_detection(
    detection: DetectionTargets,
    plan: AffinePlan,
    width: f32,
    height: f32,
    area_threshold: f32,
) -> crate::Result<AffineDetectionResult> {
    let template = detection.boxes_xyxy.clone();
    let mut boxes = detection
        .boxes_xyxy
        .to_dtype(candle_core::DType::F32)?
        .to_vec3::<f32>()?;
    let classes = detection.class_ids.to_vec2::<u32>()?;
    let mut valid = detection
        .valid
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?;
    for b in 0..boxes.len() {
        for obj in 0..boxes[b].len() {
            if valid[b][obj] <= 0.0 {
                continue;
            }
            let original_box = [
                boxes[b][obj][0],
                boxes[b][obj][1],
                boxes[b][obj][2],
                boxes[b][obj][3],
            ];
            let mut box_xyxy = [
                boxes[b][obj][0],
                boxes[b][obj][1],
                boxes[b][obj][2],
                boxes[b][obj][3],
            ];
            affine_box(&mut box_xyxy, plan);
            let on_canvas = clamp_to_canvas(&mut box_xyxy, width, height);
            if !on_canvas
                || !box_candidate(original_box, box_xyxy, plan.s_w, plan.s_h, area_threshold)
            {
                valid[b][obj] = 0.0;
                boxes[b][obj] = box_xyxy.to_vec();
            } else {
                boxes[b][obj] = box_xyxy.to_vec();
            }
        }
    }
    let targets = DetectionTargets::new(
        rebuild(&template, flat_boxes(&boxes))?,
        rebuild_class(&template, classes)?,
        rebuild_valid(&template, valid)?,
    )?;
    Ok(AffineDetectionResult { targets })
}

fn affine_segmentation(
    targets: SegmentationTargets,
    plan: AffinePlan,
    width: f32,
    height: f32,
) -> crate::Result<SegmentationTargets> {
    let mut masks = apply_affine_mask(&targets.masks, plan, height as usize, width as usize)?;
    let detection = affine_detection(
        targets.detection,
        plan,
        width,
        height,
        SEGMENT_AREA_THRESHOLD,
    )?;
    masks = zero_invalid_masks(&masks, targets.mask_encoding, &detection.targets.valid)?;
    SegmentationTargets::new_with_mask_encoding(detection.targets, masks, targets.mask_encoding)
}

fn affine_pose(
    targets: PoseTargets,
    plan: AffinePlan,
    width: f32,
    height: f32,
) -> crate::Result<PoseTargets> {
    let keypoints = affine_keypoints(&targets.keypoints, plan)?;
    let detection = affine_detection(
        targets.detection,
        plan,
        width,
        height,
        DETECTION_AREA_THRESHOLD,
    )?;
    Ok(PoseTargets {
        detection: detection.targets,
        keypoints,
        visibility: targets.visibility,
        flip_indices: targets.flip_indices,
    })
}

fn affine_obb(
    targets: ObbTargets,
    plan: AffinePlan,
    width: f32,
    height: f32,
) -> crate::Result<ObbTargets> {
    // Uniform scale + translate (rotation/shear are zero by default) leaves the
    // oriented-box angle invariant; only the underlying box geometry transforms.
    let detection = affine_detection(
        targets.detection,
        plan,
        width,
        height,
        DETECTION_AREA_THRESHOLD,
    )?;
    ObbTargets::new(detection.targets, targets.angles)
}

struct AffineDetectionResult {
    targets: DetectionTargets,
}

fn box_candidate(
    original: [f32; 4],
    transformed: [f32; 4],
    scale_w: f32,
    scale_h: f32,
    area_threshold: f32,
) -> bool {
    let w1 = (original[2] - original[0]).max(0.0) * scale_w.abs();
    let h1 = (original[3] - original[1]).max(0.0) * scale_h.abs();
    let w2 = transformed[2] - transformed[0];
    let h2 = transformed[3] - transformed[1];
    if w2 <= 2.0 || h2 <= 2.0 {
        return false;
    }
    let area_ratio = w2 * h2 / (w1 * h1 + 1e-16);
    let aspect = (w2 / (h2 + 1e-16)).max(h2 / (w2 + 1e-16));
    area_ratio > area_threshold && aspect < 100.0
}

fn zero_invalid_masks(
    masks: &Tensor,
    encoding: SegmentationMaskEncoding,
    valid: &Tensor,
) -> crate::Result<Tensor> {
    let dims = masks.dims();
    let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let mut data = masks
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let valid = valid.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
    let pixels = height * width;
    match encoding {
        SegmentationMaskEncoding::PerInstance => {
            for (b, valid_row) in valid.iter().enumerate().take(batch) {
                for obj in 0..channels {
                    if valid_row.get(obj).copied().unwrap_or(0.0) > 0.0 {
                        continue;
                    }
                    let start = (b * channels + obj) * pixels;
                    data[start..start + pixels].fill(0.0);
                }
            }
        }
        SegmentationMaskEncoding::Overlap => {
            for (b, valid_row) in valid.iter().enumerate().take(batch) {
                let start = b * pixels;
                let end = start + pixels;
                for pixel in &mut data[start..end] {
                    let object_idx = *pixel as usize;
                    if object_idx == 0 {
                        continue;
                    }
                    let keep = valid_row.get(object_idx - 1).copied().unwrap_or(0.0) > 0.0;
                    if !keep {
                        *pixel = 0.0;
                    }
                }
            }
        }
    }
    Tensor::from_vec(data, (batch, channels, height, width), masks.device())?
        .to_dtype(masks.dtype())
        .map_err(Into::into)
}

fn affine_keypoints(keypoints: &Tensor, plan: AffinePlan) -> crate::Result<Tensor> {
    let shape = keypoints.dims();
    let mut flat = keypoints
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for i in 0..flat.len() / 2 {
        flat[2 * i] = flat[2 * i] * plan.s_w + plan.dx;
        flat[2 * i + 1] = flat[2 * i + 1] * plan.s_h + plan.dy;
    }
    Ok(Tensor::from_vec(flat, shape, keypoints.device())?.to_dtype(keypoints.dtype())?)
}

fn flat_boxes(boxes: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let mut flat = Vec::new();
    for batch in boxes {
        for object in batch {
            flat.extend_from_slice(object);
        }
    }
    flat
}

fn rebuild(template: &Tensor, flat: Vec<f32>) -> crate::Result<Tensor> {
    let tensor =
        Tensor::from_vec(flat, template.dims(), template.device()).map_err(crate::Error::from)?;
    tensor
        .to_dtype(template.dtype())
        .map_err(crate::Error::from)
}

fn rebuild_class(template: &Tensor, classes: Vec<Vec<u32>>) -> crate::Result<Tensor> {
    let dims = (template.dim(0)?, template.dim(1)?);
    let tensor =
        Tensor::from_vec(classes.concat(), dims, template.device()).map_err(crate::Error::from)?;
    tensor
        .to_dtype(candle_core::DType::U32)
        .map_err(crate::Error::from)
}

fn rebuild_valid(template: &Tensor, valid: Vec<Vec<f32>>) -> crate::Result<Tensor> {
    let dims = (template.dim(0)?, template.dim(1)?);
    let tensor =
        Tensor::from_vec(valid.concat(), dims, template.device()).map_err(crate::Error::from)?;
    tensor
        .to_dtype(candle_core::DType::F32)
        .map_err(crate::Error::from)
}
