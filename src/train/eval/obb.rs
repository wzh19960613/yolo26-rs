use candle_core::{DType, Tensor};

use crate::network::head::anchors::dist2rbox_xywh;

use super::{
    DetectionEvalMetrics, MapAccumulator, Output, Target,
    batch_image_predictions_with_tensor_scores,
};
use crate::train::eval::obb_match::{match_obb_targets, obb_tp_matrix, pred_rbox_xyxy_channels};

pub(crate) fn obb_eval_metrics(
    output: &Output,
    target: &Target,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    single_class: bool,
    mut map: Option<&mut MapAccumulator>,
) -> crate::Result<Option<DetectionEvalMetrics>> {
    let (Some((detect, angles)), Target::Obb(targets)) = (obb_parts(output), target) else {
        return Ok(None);
    };
    let pred_rboxes = decode_obb_rboxes(detect, angles)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let pred_scores = candle_nn::ops::sigmoid(&detect.scores)?.to_dtype(DType::F32)?;
    let anchors_len = detect.boxes.dim(2)?;
    let pred_xyxy = pred_rboxes
        .iter()
        .map(|batch_rboxes| pred_rbox_xyxy_channels(batch_rboxes, anchors_len))
        .collect::<Vec<_>>();
    let mut batch_preds = batch_image_predictions_with_tensor_scores(
        &pred_xyxy,
        &pred_scores,
        confidence_threshold,
        iou_threshold,
        max_detections,
    )?;
    let target_rboxes = targets
        .rboxes_xywhr
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
    let mut metrics = DetectionEvalMetrics {
        matched_targets: 0,
        total_targets: 0,
        predictions: 0,
    };
    for b in 0..pred_rboxes.len() {
        let preds = &mut batch_preds[b];
        if single_class {
            for pred in preds.iter_mut() {
                pred.class_id = 0;
            }
        }
        let tp = obb_tp_matrix(
            preds,
            &pred_rboxes[b],
            &target_rboxes[b],
            &target_classes[b],
            &target_valid[b],
        );
        if let Some(acc) = map.as_deref_mut() {
            acc.add_image_with_tp(
                preds,
                &target_boxes[b],
                &target_classes[b],
                &target_valid[b],
                &tp,
            );
        }
        let matched = match_obb_targets(
            preds,
            &pred_rboxes[b],
            &target_rboxes[b],
            &target_classes[b],
            &target_valid[b],
            iou_threshold,
        );
        metrics.predictions += preds.len();
        metrics.matched_targets += matched.0;
        metrics.total_targets += matched.1;
    }
    Ok(Some(metrics))
}

fn obb_parts(output: &Output) -> Option<(&super::DenseDetectionOutput, &Tensor)> {
    match output {
        Output::Obb { detect, angles } => Some((detect, angles)),
        Output::ObbE2e {
            one_to_one_detect,
            one_to_one_angles,
            ..
        } => Some((one_to_one_detect, one_to_one_angles)),
        _ => None,
    }
}

fn decode_obb_rboxes(
    detect: &super::DenseDetectionOutput,
    angles: &Tensor,
) -> crate::Result<Tensor> {
    let xywh = dist2rbox_xywh(&detect.boxes, angles, &detect.anchors)?
        .broadcast_mul(&detect.stride_tensor)?;
    Ok(Tensor::cat(&[&xywh, angles], 1)?)
}
