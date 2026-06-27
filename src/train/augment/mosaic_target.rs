//! Per-task extra-data accumulation for mosaic/mixup composition.
//!
//! Mosaic and mixup merge multiple samples' targets. Boxes/classes and the
//! task-specific extras (pose keypoints+visibility, OBB angles, per-instance
//! segmentation masks) are collected together per valid object so that an
//! object dropped by canvas clamping is dropped from boxes and extras alike,
//! keeping the merged target internally consistent.

use candle_core::{DType, Tensor};

use super::affine::AffinePlan;
use super::affine::affine_box;
use super::affine_mask::apply_affine_mask;
use super::geometry::clamp_to_canvas;
use super::{DetectionTargets, ObbTargets, PoseTargets, Sample, SegmentationTargets, Target};
use crate::train::SegmentationMaskEncoding;

/// Accumulator for the task-specific extras produced while composing samples.
pub(crate) enum MosaicExtras {
    /// Plain detection: no extra data.
    Detection,
    /// Pose keypoints (`[num_kp, 2]` per object) and visibility (`[num_kp]`).
    Pose {
        /// Flattened keypoint coordinates, appended per valid object.
        keypoints: Vec<f32>,
        /// Flattened visibility, appended per valid object.
        visibility: Vec<f32>,
        /// Keypoint count per object.
        num_kp: usize,
    },
    /// OBB angles (one per object, invariant under scale/translate).
    Obb {
        /// Angles, appended per valid object.
        angles: Vec<f32>,
    },
    /// Per-instance segmentation mask planes.
    Model {
        /// Flattened mask planes, appended per valid object.
        masks: Vec<f32>,
        /// Mask height.
        mh: usize,
        /// Mask width.
        mw: usize,
    },
}

impl MosaicExtras {
    /// Creates an empty accumulator matching `primary`'s task.
    pub(crate) fn new(primary: &Target) -> crate::Result<Self> {
        Ok(match primary {
            Target::Detection(_) => MosaicExtras::Detection,
            Target::Pose(t) => MosaicExtras::Pose {
                keypoints: Vec::new(),
                visibility: Vec::new(),
                num_kp: t.keypoints.dim(2)?,
            },
            Target::Obb(_) => MosaicExtras::Obb { angles: Vec::new() },
            Target::Segmentation(t) => match t.mask_encoding {
                SegmentationMaskEncoding::PerInstance => MosaicExtras::Model {
                    masks: Vec::new(),
                    mh: t.masks.dim(2)?,
                    mw: t.masks.dim(3)?,
                },
                SegmentationMaskEncoding::Overlap => {
                    return Err(crate::Error::InvalidConfig(
                        "mosaic/mixup on overlap-encoded segmentation masks is unsupported".into(),
                    ));
                }
            },
            other => {
                return Err(crate::Error::InvalidConfig(format!(
                    "mosaic/mixup requires a box-bearing task target, got {:?}",
                    std::mem::discriminant(other)
                )));
            }
        })
    }

    /// Collects transformed boxes/classes and the task-specific extras for every
    /// valid, on-canvas object in one sample.
    pub(crate) fn collect_sample(
        &mut self,
        sample: &Sample,
        plan: AffinePlan,
        width: f32,
        height: f32,
        all_boxes: &mut Vec<[f32; 4]>,
        all_classes: &mut Vec<u32>,
    ) -> crate::Result<()> {
        let det = super::mosaic::detection_of(&sample.target)?;
        let boxes = det.boxes_xyxy.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let classes = det.class_ids.to_vec2::<u32>()?;
        let valid = det.valid.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for obj in 0..boxes[0].len() {
            if valid[0][obj] <= 0.0 {
                continue;
            }
            let row = &boxes[0][obj];
            let mut box_xyxy = [row[0], row[1], row[2], row[3]];
            affine_box(&mut box_xyxy, plan);
            if !clamp_to_canvas(&mut box_xyxy, width, height) {
                continue;
            }
            all_boxes.push(box_xyxy);
            all_classes.push(classes[0][obj]);
            self.collect_extra(sample, obj, plan, height as usize, width as usize)?;
        }
        Ok(())
    }

    fn collect_extra(
        &mut self,
        sample: &Sample,
        obj: usize,
        plan: AffinePlan,
        height: usize,
        width: usize,
    ) -> crate::Result<()> {
        match (self, &sample.target) {
            (MosaicExtras::Detection, _) => Ok(()),
            (
                MosaicExtras::Pose {
                    keypoints,
                    visibility,
                    num_kp,
                },
                Target::Pose(t),
            ) => {
                let kp_flat = t
                    .keypoints
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let vis = t.visibility.to_dtype(DType::F32)?.to_vec3::<f32>()?;
                let stride = *num_kp * 2;
                for k in 0..*num_kp {
                    keypoints.push(kp_flat[obj * stride + k * 2] * plan.s_w + plan.dx);
                    keypoints.push(kp_flat[obj * stride + k * 2 + 1] * plan.s_h + plan.dy);
                    visibility.push(vis[0][obj][k]);
                }
                Ok(())
            }
            (MosaicExtras::Obb { angles }, Target::Obb(t)) => {
                let ang = t.angles.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                angles.push(ang[0][obj]);
                Ok(())
            }
            (MosaicExtras::Model { masks, mh, mw }, Target::Segmentation(t)) => {
                let masks_flat = t
                    .masks
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let plane_size = *mh * *mw;
                let plane = masks_flat[obj * plane_size..(obj + 1) * plane_size].to_vec();
                let plane = Tensor::from_vec(plane, (1, 1, *mh, *mw), t.masks.device())?;
                let placed = apply_affine_mask(&plane, plan, height, width)?;
                masks.extend(
                    placed
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Assembles the final training target from the merged detection target and
    /// the first `take` accumulated extras.
    pub(crate) fn finish(
        self,
        detection: DetectionTargets,
        take: usize,
        device: &candle_core::Device,
    ) -> crate::Result<Target> {
        let max_objects = detection.boxes_xyxy.dim(1)?;
        match self {
            MosaicExtras::Detection => Ok(Target::Detection(detection)),
            MosaicExtras::Pose {
                keypoints,
                visibility,
                num_kp,
            } => {
                let mut kp_flat = vec![0.0f32; max_objects * num_kp * 2];
                let mut vis_flat = vec![0.0f32; max_objects * num_kp];
                let per_obj = num_kp * 2;
                for i in 0..take.min(max_objects) {
                    let src = i * per_obj;
                    kp_flat[src..src + per_obj].copy_from_slice(&keypoints[src..src + per_obj]);
                    vis_flat[i * num_kp..(i + 1) * num_kp]
                        .copy_from_slice(&visibility[i * num_kp..(i + 1) * num_kp]);
                }
                let keypoints = Tensor::from_vec(kp_flat, (1, max_objects, num_kp, 2), device)?;
                let visibility = Tensor::from_vec(vis_flat, (1, max_objects, num_kp), device)?;
                Ok(Target::Pose(PoseTargets {
                    detection,
                    keypoints,
                    visibility,
                    flip_indices: None,
                }))
            }
            MosaicExtras::Obb { angles } => {
                let mut flat = vec![0.0f32; max_objects];
                for (i, a) in angles.into_iter().take(take.min(max_objects)).enumerate() {
                    flat[i] = a;
                }
                let angles = Tensor::from_vec(flat, (1, max_objects), device)?;
                Ok(Target::Obb(ObbTargets::new(detection, angles)?))
            }
            MosaicExtras::Model { masks, mh, mw } => {
                let plane = mh * mw;
                let mut flat = vec![0.0f32; max_objects * plane];
                for i in 0..take.min(max_objects) {
                    let src = i * plane;
                    flat[src..src + plane].copy_from_slice(&masks[src..src + plane]);
                }
                let masks = Tensor::from_vec(flat, (1, max_objects, mh, mw), device)?;
                Ok(Target::Segmentation(SegmentationTargets::new(
                    detection, masks,
                )?))
            }
        }
    }
}
