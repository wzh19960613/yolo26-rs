use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "eval dispatcher passes independent validation knobs and optional accumulators"
)]
pub(crate) fn detection_eval_metrics(
    output: &Output,
    postprocess: Option<&EvalPostprocessOutput>,
    target: &Target,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    single_class: bool,
    mut map: Option<&mut MapAccumulator>,
) -> crate::Result<Option<DetectionEvalMetrics>> {
    let Some(detect) = output_detection(output) else {
        return Ok(None);
    };
    let Some(targets) = target_detection(target) else {
        return Ok(None);
    };
    let mut batch_preds = match postprocess {
        Some(EvalPostprocessOutput::Detect { predictions })
        | Some(EvalPostprocessOutput::Segment { predictions, .. }) => {
            batch_image_predictions_from_topk_tensor(
                predictions,
                confidence_threshold,
                iou_threshold,
                max_detections,
            )?
        }
        None => {
            let pred_xyxy = decode_detection_boxes(detect)?.to_dtype(DType::F32)?;
            let pred_scores = candle_nn::ops::sigmoid(&detect.scores)?.to_dtype(DType::F32)?;
            batch_image_predictions_from_tensors(
                &pred_xyxy,
                &pred_scores,
                confidence_threshold,
                iou_threshold,
                max_detections,
            )?
        }
    };
    let target_boxes = targets.boxes_xyxy.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let target_classes = targets.class_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let target_valid = targets.valid.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut metrics = DetectionEvalMetrics {
        matched_targets: 0,
        total_targets: 0,
        predictions: 0,
    };
    for b in 0..batch_preds.len() {
        let preds = &mut batch_preds[b];
        if single_class {
            for pred in preds.iter_mut() {
                pred.class_id = 0;
            }
        }
        let matched = match_targets(
            preds.as_mut_slice(),
            &target_boxes[b],
            &target_classes[b],
            &target_valid[b],
            iou_threshold,
        );
        if let Some(acc) = map.as_deref_mut() {
            acc.add_image(
                preds,
                &target_boxes[b],
                &target_classes[b],
                &target_valid[b],
            );
        }
        metrics.predictions += preds.len();
        metrics.matched_targets += matched.0;
        metrics.total_targets += matched.1;
    }
    Ok(Some(metrics))
}

fn match_targets(
    preds: &mut [EvalPrediction],
    target_boxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
    iou_threshold: f32,
) -> (usize, usize) {
    let mut used = vec![false; preds.len()];
    let mut matched_targets = 0usize;
    let mut total_targets = 0usize;
    for obj in 0..target_boxes.len() {
        if target_valid[obj] <= 0.0 {
            continue;
        }
        let gt = [
            target_boxes[obj][0],
            target_boxes[obj][1],
            target_boxes[obj][2],
            target_boxes[obj][3],
        ];
        if gt[2] <= gt[0] || gt[3] <= gt[1] {
            continue;
        }
        total_targets += 1;
        let mut best_pred = None;
        let mut best_iou = 0.0f32;
        for (pred_idx, pred) in preds.iter().enumerate() {
            if used[pred_idx] || pred.class_id != target_classes[obj] {
                continue;
            }
            let iou = xyxy_iou_scalar(pred.xyxy, gt);
            if iou >= iou_threshold && iou > best_iou {
                best_iou = iou;
                best_pred = Some(pred_idx);
            }
        }
        if let Some(pred_idx) = best_pred {
            used[pred_idx] = true;
            matched_targets += 1;
        }
    }
    (matched_targets, total_targets)
}

fn output_detection(output: &Output) -> Option<&DenseDetectionOutput> {
    match output {
        Output::Detect(detect)
        | Output::DetectE2e {
            one_to_one: detect, ..
        }
        | Output::SegmentE2e {
            one_to_one_detect: detect,
            ..
        }
        | Output::Segment { detect, .. }
        | Output::PoseE2e {
            one_to_one_detect: detect,
            ..
        }
        | Output::Pose { detect, .. }
        | Output::ObbE2e {
            one_to_one_detect: detect,
            ..
        }
        | Output::Obb { detect, .. } => Some(detect),
        _ => None,
    }
}
