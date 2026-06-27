use super::*;

/// Collates single-image segmentation samples into one batch.
pub fn collate_segmentation_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "segmentation sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut box_refs = Vec::with_capacity(samples.len());
    let mut class_refs = Vec::with_capacity(samples.len());
    let mut valid_refs = Vec::with_capacity(samples.len());
    let mut mask_refs = Vec::with_capacity(samples.len());
    let mut target_shape = None;
    let mut mask_shape = None;
    let mut mask_encoding = None;
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all segmentation sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Segmentation(targets) = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_segmentation_samples only accepts segmentation targets".to_string(),
            ));
        };
        let shape = targets.detection.boxes_xyxy.dims().to_vec();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
            return Err(crate::Error::InvalidTensor(format!(
                "segmentation detection boxes must have shape [1, objects, 4], got {shape:?}"
            )));
        }
        if let Some(expected) = &target_shape {
            if expected != &shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all segmentation detection targets must share shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            target_shape = Some(shape);
        }
        let masks_shape = targets.masks.dims().to_vec();
        if masks_shape.len() != 4 || masks_shape[0] != 1 {
            return Err(crate::Error::InvalidTensor(format!(
                "segmentation masks must have shape [1, objects_or_overlap, H, W], got {masks_shape:?}"
            )));
        }
        if let Some(expected) = mask_encoding {
            if expected != targets.mask_encoding {
                return Err(crate::Error::InvalidTensor(
                    "all segmentation masks must use the same encoding".to_string(),
                ));
            }
        } else {
            mask_encoding = Some(targets.mask_encoding);
        }
        if let Some(expected) = &mask_shape {
            if expected != &masks_shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all segmentation masks must share shape {expected:?}, got {masks_shape:?}"
                )));
            }
        } else {
            mask_shape = Some(masks_shape);
        }
        input_refs.push(&sample.input);
        box_refs.push(&targets.detection.boxes_xyxy);
        class_refs.push(&targets.detection.class_ids);
        valid_refs.push(&targets.detection.valid);
        mask_refs.push(&targets.masks);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let boxes_xyxy = Tensor::cat(&box_refs, 0)?;
    let class_ids = Tensor::cat(&class_refs, 0)?;
    let valid = Tensor::cat(&valid_refs, 0)?;
    let masks = Tensor::cat(&mask_refs, 0)?;
    let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
    Ok(Sample {
        input,
        target: Target::Segmentation(SegmentationTargets::new_with_mask_encoding(
            detection,
            masks,
            mask_encoding.unwrap_or(SegmentationMaskEncoding::PerInstance),
        )?),
    })
}

/// Collates single-image pose samples into one batch.
pub fn collate_pose_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "pose sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut box_refs = Vec::with_capacity(samples.len());
    let mut class_refs = Vec::with_capacity(samples.len());
    let mut valid_refs = Vec::with_capacity(samples.len());
    let mut keypoint_refs = Vec::with_capacity(samples.len());
    let mut visibility_refs = Vec::with_capacity(samples.len());
    let mut target_shape = None;
    let mut keypoint_shape = None;
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all pose sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Pose(targets) = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_pose_samples only accepts pose targets".to_string(),
            ));
        };
        let shape = targets.detection.boxes_xyxy.dims().to_vec();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
            return Err(crate::Error::InvalidTensor(format!(
                "pose detection boxes must have shape [1, objects, 4], got {shape:?}"
            )));
        }
        if let Some(expected) = &target_shape {
            if expected != &shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all pose detection targets must share shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            target_shape = Some(shape);
        }
        let kpt_shape = targets.keypoints.dims().to_vec();
        if kpt_shape.len() != 4 || kpt_shape[0] != 1 || kpt_shape[3] != 2 {
            return Err(crate::Error::InvalidTensor(format!(
                "pose keypoints must have shape [1, objects, keypoints, 2], got {kpt_shape:?}"
            )));
        }
        if let Some(expected) = &keypoint_shape {
            if expected != &kpt_shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all pose keypoints must share shape {expected:?}, got {kpt_shape:?}"
                )));
            }
        } else {
            keypoint_shape = Some(kpt_shape);
        }
        input_refs.push(&sample.input);
        box_refs.push(&targets.detection.boxes_xyxy);
        class_refs.push(&targets.detection.class_ids);
        valid_refs.push(&targets.detection.valid);
        keypoint_refs.push(&targets.keypoints);
        visibility_refs.push(&targets.visibility);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let boxes_xyxy = Tensor::cat(&box_refs, 0)?;
    let class_ids = Tensor::cat(&class_refs, 0)?;
    let valid = Tensor::cat(&valid_refs, 0)?;
    let keypoints = Tensor::cat(&keypoint_refs, 0)?;
    let visibility = Tensor::cat(&visibility_refs, 0)?;
    let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
    Ok(Sample {
        input,
        target: Target::Pose(PoseTargets::new(detection, keypoints, visibility)?),
    })
}
