use super::*;

pub(crate) fn decode_keypoint_xy(
    keypoints: &Tensor,
    anchors: &Tensor,
    strides: &Tensor,
    keypoints_count: usize,
    keypoint_dims: usize,
) -> crate::Result<Tensor> {
    let anchor_x = anchors.narrow(1, 0, 1)?;
    let anchor_y = anchors.narrow(1, 1, 1)?;
    let mut decoded = Vec::with_capacity(keypoints_count * 2);
    for keypoint_idx in 0..keypoints_count {
        let base = keypoint_idx * keypoint_dims;
        let x = keypoints
            .narrow(1, base, 1)?
            .broadcast_add(&anchor_x)?
            .broadcast_mul(strides)?;
        let y = keypoints
            .narrow(1, base + 1, 1)?
            .broadcast_add(&anchor_y)?
            .broadcast_mul(strides)?;
        decoded.push(x);
        decoded.push(y);
    }
    let refs: Vec<&Tensor> = decoded.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

pub(crate) fn keypoint_visibility_logits(
    keypoints: &Tensor,
    keypoints_count: usize,
    keypoint_dims: usize,
) -> crate::Result<Tensor> {
    let mut channels = Vec::with_capacity(keypoints_count);
    for keypoint_idx in 0..keypoints_count {
        channels.push(keypoints.narrow(1, keypoint_idx * keypoint_dims + 2, 1)?);
    }
    let refs: Vec<&Tensor> = channels.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

pub(crate) struct AssignedKeypointTargets {
    pub(crate) xy: Tensor,
    pub(crate) position_mask: Tensor,
    pub(crate) visibility: Tensor,
    pub(crate) visibility_mask: Tensor,
    pub(crate) position_count: f64,
    pub(crate) visibility_count: f64,
}

pub(crate) fn build_assigned_keypoint_targets(
    targets: &PoseTargets,
    target_gt_idx: &[usize],
    batch: usize,
    anchors_len: usize,
) -> crate::Result<AssignedKeypointTargets> {
    let objects = targets.detection.boxes_xyxy.dim(1)?;
    let keypoints_count = targets.keypoints.dim(2)?;
    let keypoint_data = targets
        .keypoints
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let visibility_data = targets
        .visibility
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut xy = vec![0f32; batch * keypoints_count * 2 * anchors_len];
    let mut position_mask = vec![0f32; batch * keypoints_count * 2 * anchors_len];
    let mut visibility = vec![0f32; batch * keypoints_count * anchors_len];
    let mut visibility_mask = vec![0f32; batch * keypoints_count * anchors_len];
    let mut position_count = 0f64;
    let mut visibility_count = 0f64;

    for b in 0..batch {
        for anchor_idx in 0..anchors_len {
            let obj = target_gt_idx[b * anchors_len + anchor_idx];
            if obj == usize::MAX || obj >= objects {
                continue;
            }
            for keypoint_idx in 0..keypoints_count {
                let src_vis = (b * objects + obj) * keypoints_count + keypoint_idx;
                let visible = if visibility_data[src_vis] > 0.0 {
                    1.0
                } else {
                    0.0
                };
                let visibility_dst =
                    (b * keypoints_count + keypoint_idx) * anchors_len + anchor_idx;
                visibility[visibility_dst] = visible;
                visibility_mask[visibility_dst] = 1.0;
                visibility_count += 1.0;

                let src_xy = ((b * objects + obj) * keypoints_count + keypoint_idx) * 2;
                let x_channel = keypoint_idx * 2;
                let y_channel = x_channel + 1;
                let x_dst = (b * keypoints_count * 2 + x_channel) * anchors_len + anchor_idx;
                let y_dst = (b * keypoints_count * 2 + y_channel) * anchors_len + anchor_idx;
                xy[x_dst] = keypoint_data[src_xy];
                xy[y_dst] = keypoint_data[src_xy + 1];
                if visible > 0.0 {
                    position_mask[x_dst] = 1.0;
                    position_mask[y_dst] = 1.0;
                    position_count += 2.0;
                }
            }
        }
    }

    let device = targets.keypoints.device();
    Ok(AssignedKeypointTargets {
        xy: Tensor::from_vec(xy, (batch, keypoints_count * 2, anchors_len), device)?,
        position_mask: Tensor::from_vec(
            position_mask,
            (batch, keypoints_count * 2, anchors_len),
            device,
        )?,
        visibility: Tensor::from_vec(visibility, (batch, keypoints_count, anchors_len), device)?,
        visibility_mask: Tensor::from_vec(
            visibility_mask,
            (batch, keypoints_count, anchors_len),
            device,
        )?,
        position_count,
        visibility_count,
    })
}

pub(crate) fn obb_angle_loss(
    angles: &Tensor,
    targets: &ObbTargets,
    built: &BuiltDetectionTargets,
) -> crate::Result<Tensor> {
    let (batch, angle_channels, anchors_len) = angles.dims3()?;
    if angle_channels != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "obb angle output must have shape [batch, 1, anchors], got {:?}",
            angles.dims()
        )));
    }
    if targets.detection.boxes_xyxy.dim(0)? != batch {
        return Err(crate::Error::InvalidTensor(
            "obb target batch does not match model output".to_string(),
        ));
    }
    let assigned =
        build_assigned_angle_targets(&targets.angles, &built.target_gt_idx, batch, anchors_len)?;
    let target_angles = assigned.angles.to_dtype(angles.dtype())?;
    let angle_mask = assigned.mask.to_dtype(angles.dtype())?;
    let diff = (angles - &target_angles)?;
    let one = Tensor::new(1f32, angles.device())?.to_dtype(angles.dtype())?;
    let periodic = diff.cos()?.neg()?.broadcast_add(&one)?;
    Ok(periodic
        .broadcast_mul(&angle_mask)?
        .sum_all()?
        .broadcast_div(&Tensor::new(
            assigned.count.max(1.0) as f32,
            angles.device(),
        )?)?)
}

pub(crate) struct AssignedAngleTargets {
    pub(crate) angles: Tensor,
    pub(crate) mask: Tensor,
    pub(crate) count: f64,
}
