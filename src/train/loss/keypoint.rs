use super::*;

pub(crate) struct KeypointLossReport {
    pub(crate) loss: Tensor,
    pub(crate) pose_loss: Tensor,
    pub(crate) kobj_loss: Option<Tensor>,
}

pub(crate) fn keypoint_loss_report(
    keypoints: &Tensor,
    detect: &DenseDetectionOutput,
    targets: &PoseTargets,
    built: &BuiltDetectionTargets,
    config: DetectionLossConfig,
) -> crate::Result<KeypointLossReport> {
    let (batch, channels, anchors_len) = keypoints.dims3()?;
    if detect.anchors.dim(2)? != anchors_len || detect.stride_tensor.dim(2)? != anchors_len {
        return Err(crate::Error::InvalidTensor(
            "pose keypoint anchors do not match detection anchors".to_string(),
        ));
    }
    if targets.detection.boxes_xyxy.dim(0)? != batch {
        return Err(crate::Error::InvalidTensor(
            "pose target batch does not match model output".to_string(),
        ));
    }
    let keypoints_count = targets.keypoints.dim(2)?;
    if keypoints_count == 0 || channels % keypoints_count != 0 {
        return Err(crate::Error::InvalidTensor(format!(
            "pose keypoint channels {channels} are not divisible by target keypoints {keypoints_count}"
        )));
    }
    let keypoint_dims = channels / keypoints_count;
    if keypoint_dims < 2 {
        return Err(crate::Error::InvalidTensor(
            "pose keypoint output must contain at least x/y channels".to_string(),
        ));
    }

    let pred_xy = decode_keypoint_xy(
        keypoints,
        &detect.anchors,
        &detect.stride_tensor,
        keypoints_count,
        keypoint_dims,
    )?;
    let assigned =
        build_assigned_keypoint_targets(targets, &built.target_gt_idx, batch, anchors_len)?;
    let target_xy = assigned.xy.to_dtype(pred_xy.dtype())?;
    let position_mask = assigned.position_mask.to_dtype(pred_xy.dtype())?;
    let xy_diff = (&pred_xy - &target_xy)?;
    let position_loss = xy_diff
        .sqr()?
        .broadcast_mul(&position_mask)?
        .sum_all()?
        .broadcast_div(&Tensor::new(
            assigned.position_count.max(1.0) as f32,
            pred_xy.device(),
        )?)?;

    let pose_loss = (position_loss * config.pose_weight)?;
    let mut total = pose_loss.clone();
    let mut kobj_loss = None;
    if keypoint_dims >= 3 {
        let visibility_logits =
            keypoint_visibility_logits(keypoints, keypoints_count, keypoint_dims)?;
        let target_visibility = assigned.visibility.to_dtype(visibility_logits.dtype())?;
        let visibility_mask = assigned
            .visibility_mask
            .to_dtype(visibility_logits.dtype())?;
        let visibility_loss = bce_with_logits_elementwise(&visibility_logits, &target_visibility)?
            .broadcast_mul(&visibility_mask)?
            .sum_all()?
            .broadcast_div(&Tensor::new(
                assigned.visibility_count.max(1.0) as f32,
                visibility_logits.device(),
            )?)?;
        let weighted_visibility = (visibility_loss * config.keypoint_objectness_weight)?;
        total = (total + weighted_visibility.clone())?;
        kobj_loss = Some(weighted_visibility);
    }
    Ok(KeypointLossReport {
        loss: total,
        pose_loss,
        kobj_loss,
    })
}
