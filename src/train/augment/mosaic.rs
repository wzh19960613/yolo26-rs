//! Four-image mosaic composition for box-bearing tasks.
//!
//! Each input image is resized to half the canvas and placed in a quadrant;
//! each box/keypoint is scaled by 0.5 and shifted to its quadrant, while OBB
//! angles stay invariant and per-instance masks are nearest-resampled. Boxes,
//! classes and the task-specific extras are merged and truncated to the
//! per-sample `max_objects` slot count.

use candle_core::Tensor;

use super::affine::AffinePlan;
use super::mosaic_target::MosaicExtras;
use super::{DetectionTargets, Sample, Target};

/// Composes four box-bearing-task samples into one 2x2 mosaic.
pub(crate) fn compose_mosaic(samples: [&Sample; 4]) -> crate::Result<Sample> {
    let height = samples[0].input.dim(2)?;
    let width = samples[0].input.dim(3)?;
    let max_objects = mosaic_max_objects(&samples[0].target)?;
    let half_h = height / 2;
    let half_w = width / 2;
    let offsets = [
        (0.0_f32, 0.0_f32),
        (0.0, half_w as f32),
        (half_h as f32, 0.0),
        (half_h as f32, half_w as f32),
    ];
    let device = samples[0].input.device().clone();

    let mut tiles: Vec<Tensor> = Vec::with_capacity(4);
    let mut all_boxes: Vec<[f32; 4]> = Vec::new();
    let mut all_classes: Vec<u32> = Vec::new();
    let mut extras = MosaicExtras::new(&samples[0].target)?;
    for (slot, sample) in samples.iter().enumerate() {
        tiles.push(sample.input.interpolate2d(half_h, half_w)?);
        let plan = AffinePlan {
            s_w: 0.5,
            s_h: 0.5,
            dx: offsets[slot].1,
            dy: offsets[slot].0,
        };
        extras.collect_sample(
            sample,
            plan,
            width as f32,
            height as f32,
            &mut all_boxes,
            &mut all_classes,
        )?;
    }

    let take = all_boxes.len().min(max_objects);
    let detection = build_detection(&all_boxes, &all_classes, take, max_objects, &device)?;
    let image = build_image(&tiles)?;
    let target = extras.finish(detection, take, &device)?;
    Ok(Sample {
        input: image,
        target,
    })
}

/// Returns the per-sample object slot count for the task target.
pub(crate) fn mosaic_max_objects(target: &Target) -> crate::Result<usize> {
    let boxes = match target {
        Target::Detection(t) => &t.boxes_xyxy,
        Target::Segmentation(t) => &t.detection.boxes_xyxy,
        Target::Pose(t) => &t.detection.boxes_xyxy,
        Target::Obb(t) => &t.detection.boxes_xyxy,
        _ => {
            return Err(crate::Error::InvalidConfig(
                "mosaic composition requires a box-bearing task sample".into(),
            ));
        }
    };
    Ok(boxes.dim(1)?)
}

pub(crate) fn build_detection(
    boxes: &[[f32; 4]],
    classes: &[u32],
    take: usize,
    max_objects: usize,
    device: &candle_core::Device,
) -> crate::Result<DetectionTargets> {
    let mut flat_boxes = vec![0.0f32; max_objects * 4];
    let mut flat_classes = vec![0u32; max_objects];
    let mut flat_valid = vec![0.0f32; max_objects];
    for i in 0..take {
        flat_boxes[i * 4..i * 4 + 4].copy_from_slice(&boxes[i]);
        flat_classes[i] = classes[i];
        flat_valid[i] = 1.0;
    }
    let boxes_xyxy = Tensor::from_vec(flat_boxes, (1, max_objects, 4), device)?;
    let class_ids = Tensor::from_vec(flat_classes, (1, max_objects), device)?;
    let valid = Tensor::from_vec(flat_valid, (1, max_objects), device)?;
    DetectionTargets::new(boxes_xyxy, class_ids, valid)
}

fn build_image(tiles: &[Tensor]) -> crate::Result<Tensor> {
    let top = Tensor::cat(&[tiles[0].clone(), tiles[1].clone()], 3)?;
    let bottom = Tensor::cat(&[tiles[2].clone(), tiles[3].clone()], 3)?;
    Tensor::cat(&[top, bottom], 2).map_err(Into::into)
}

/// Returns the detection target shared by every box-bearing task.
pub(crate) fn detection_of(target: &Target) -> crate::Result<&DetectionTargets> {
    match target {
        Target::Detection(t) => Ok(t),
        Target::Segmentation(t) => Ok(&t.detection),
        Target::Pose(t) => Ok(&t.detection),
        Target::Obb(t) => Ok(&t.detection),
        other => Err(crate::Error::InvalidConfig(format!(
            "expected a box-bearing task target, got {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

/// Returns `true` when the target can be composed by mosaic/mixup.
///
/// Detection, pose and OBB are always composable; per-instance segmentation
/// masks are composable, but overlap-encoded instance-index maps cannot be
/// concatenated across samples and are skipped.
pub(crate) fn compose_supported(target: &Target) -> bool {
    use crate::train::SegmentationMaskEncoding;
    match target {
        Target::Detection(_) | Target::Pose(_) | Target::Obb(_) => true,
        Target::Segmentation(t) => t.mask_encoding == SegmentationMaskEncoding::PerInstance,
        _ => false,
    }
}
