use candle_core::{DType, Tensor};

use super::{
    MapAccumulator, Output, Target, batch_image_predictions_from_tensors, decode_detection_boxes,
    decode_keypoint_xy, pose_oks::pose_tp_matrix,
};

pub(crate) fn update_pose_map(
    output: &Output,
    target: &Target,
    confidence_threshold: f32,
    max_detections: usize,
    single_class: bool,
    map: Option<&mut MapAccumulator>,
) -> crate::Result<()> {
    let (Some(acc), Some((detect, keypoints)), Target::Pose(targets)) =
        (map, pose_parts(output), target)
    else {
        return Ok(());
    };
    let pred_xyxy = decode_detection_boxes(detect)?.to_dtype(DType::F32)?;
    let pred_scores = candle_nn::ops::sigmoid(&detect.scores)?.to_dtype(DType::F32)?;
    let mut batch_preds = batch_image_predictions_from_tensors(
        &pred_xyxy,
        &pred_scores,
        confidence_threshold,
        0.7,
        max_detections,
    )?;
    let keypoints_count = targets.keypoints.dim(2)?;
    let keypoint_channels = keypoints.dim(1)?;
    if keypoints_count == 0 || keypoint_channels % keypoints_count != 0 {
        return Err(crate::Error::InvalidTensor(format!(
            "pose keypoint channels {keypoint_channels} are not divisible by target keypoints {keypoints_count}"
        )));
    }
    let keypoint_dims = keypoint_channels / keypoints_count;
    if keypoint_dims < 2 {
        return Err(crate::Error::InvalidTensor(format!(
            "pose keypoint output requires at least x/y dims per keypoint, got {keypoint_dims}"
        )));
    }
    let pred_keypoints = decode_keypoint_xy(
        keypoints,
        &detect.anchors,
        &detect.stride_tensor,
        keypoints_count,
        keypoint_dims,
    )?
    .to_dtype(DType::F32)?
    .to_vec3::<f32>()?;
    let target_boxes = targets
        .detection
        .boxes_xyxy
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let target_classes = targets
        .detection
        .class_ids
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let target_valid = targets
        .detection
        .valid
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let target_keypoints = targets
        .keypoints
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let target_visibility = targets.visibility.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let objects = targets.keypoints.dim(1)?;
    for b in 0..batch_preds.len() {
        let preds = &mut batch_preds[b];
        if single_class {
            for pred in preds.iter_mut() {
                pred.class_id = 0;
            }
        }
        let tp = pose_tp_matrix(
            preds,
            &pred_keypoints[b],
            &target_boxes[b],
            &target_classes[b],
            &target_valid[b],
            &target_keypoints[b * objects * keypoints_count * 2..],
            objects,
            keypoints_count,
            &target_visibility[b],
        );
        acc.add_image_with_tp(
            preds,
            &target_boxes[b],
            &target_classes[b],
            &target_valid[b],
            &tp,
        );
    }
    Ok(())
}

fn pose_parts(output: &Output) -> Option<(&super::DenseDetectionOutput, &Tensor)> {
    match output {
        Output::Pose { detect, keypoints } => Some((detect, keypoints)),
        Output::PoseE2e {
            one_to_one_detect,
            one_to_one_keypoints,
            ..
        } => Some((one_to_one_detect, one_to_one_keypoints)),
        _ => None,
    }
}
