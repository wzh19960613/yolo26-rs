//! Target-aware flipping for box-bearing tasks and semantic segmentation.
//!
//! HSV is image-only and therefore task-agnostic; geometric flips must also
//! transform each task's spatial target. Detection boxes share one helper;
//! segmentation masks, pose keypoints and OBB angles get task-specific flips.
//! Pose keypoints additionally swap left/right pairs via [`flip_indices`] before
//! mirroring x, matching the official Ultralytics `flip_idx`.

use candle_core::Tensor;

use super::geometry::{clamp_to_canvas, flip_horizontal, flip_vertical};
use super::{DetectionTargets, ObbTargets, PoseTargets, SegmentationTargets, Target};

/// Flips a target horizontally (`do_lr`) and/or vertically (`do_ud`).
pub(crate) fn flip_target(
    target: Target,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
) -> crate::Result<Target> {
    if !do_lr && !do_ud {
        return Ok(target);
    }
    Ok(match target {
        Target::Detection(t) => Target::Detection(flip_detection(t, do_lr, do_ud, width, height)?),
        Target::Segmentation(t) => {
            Target::Segmentation(flip_segmentation(t, do_lr, do_ud, width, height)?)
        }
        Target::Pose(t) => Target::Pose(flip_pose(t, do_lr, do_ud, width, height)?),
        Target::Obb(t) => Target::Obb(flip_obb(t, do_lr, do_ud, width, height)?),
        Target::Semantic { class_map } => {
            let mut m = class_map;
            if do_lr {
                m = m.flip(&[2])?;
            }
            if do_ud {
                m = m.flip(&[1])?;
            }
            Target::Semantic { class_map: m }
        }
        other => other,
    })
}

fn flip_detection(
    detection: DetectionTargets,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
) -> crate::Result<DetectionTargets> {
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
            let mut box_xyxy = [
                boxes[b][obj][0],
                boxes[b][obj][1],
                boxes[b][obj][2],
                boxes[b][obj][3],
            ];
            if do_lr {
                flip_horizontal(&mut box_xyxy, width);
            }
            if do_ud {
                flip_vertical(&mut box_xyxy, height);
            }
            if clamp_to_canvas(&mut box_xyxy, width, height) {
                boxes[b][obj] = box_xyxy.to_vec();
            } else {
                valid[b][obj] = 0.0;
            }
        }
    }
    DetectionTargets::new(
        rebuild(&template, flat_boxes(&boxes))?,
        rebuild_class(&template, classes.concat())?,
        rebuild_valid(&template, valid.concat())?,
    )
}

fn flip_segmentation(
    targets: SegmentationTargets,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
) -> crate::Result<SegmentationTargets> {
    let mut masks = targets.masks;
    if do_lr {
        masks = masks.flip(&[3])?;
    }
    if do_ud {
        masks = masks.flip(&[2])?;
    }
    let detection = flip_detection(targets.detection, do_lr, do_ud, width, height)?;
    SegmentationTargets::new_with_mask_encoding(detection, masks, targets.mask_encoding)
}

fn flip_pose(
    targets: PoseTargets,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
) -> crate::Result<PoseTargets> {
    let num_kp = targets.keypoints.dim(targets.keypoints.rank() - 2)?;
    let default_perm = super::flip_indices::flip_indices(num_kp);
    let perm = targets.flip_indices.as_ref().unwrap_or(&default_perm);
    let keypoints =
        super::flip_indices::flip_keypoints(&targets.keypoints, do_lr, do_ud, width, height, perm)?;
    let detection = flip_detection(targets.detection, do_lr, do_ud, width, height)?;
    Ok(PoseTargets {
        detection,
        keypoints,
        visibility: targets.visibility,
        flip_indices: targets.flip_indices,
    })
}

fn flip_obb(
    targets: ObbTargets,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
) -> crate::Result<ObbTargets> {
    let mut angles = targets
        .angles
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for a in &mut angles {
        if do_lr {
            *a = std::f32::consts::PI - *a;
        }
        if do_ud {
            *a = -*a;
        }
    }
    let template = targets.angles.clone();
    let angles =
        Tensor::from_vec(angles, template.dims(), template.device())?.to_dtype(template.dtype());
    let detection = flip_detection(targets.detection, do_lr, do_ud, width, height)?;
    ObbTargets::new(detection, angles?)
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

fn rebuild_class(template: &Tensor, flat: Vec<u32>) -> crate::Result<Tensor> {
    let dims = template.dims();
    let tensor = Tensor::from_vec(flat, (dims[0], dims[1]), template.device())
        .map_err(crate::Error::from)?;
    tensor
        .to_dtype(candle_core::DType::U32)
        .map_err(crate::Error::from)
}

fn rebuild_valid(template: &Tensor, flat: Vec<f32>) -> crate::Result<Tensor> {
    let dims = template.dims();
    let tensor = Tensor::from_vec(flat, (dims[0], dims[1]), template.device())
        .map_err(crate::Error::from)?;
    tensor
        .to_dtype(candle_core::DType::F32)
        .map_err(crate::Error::from)
}
