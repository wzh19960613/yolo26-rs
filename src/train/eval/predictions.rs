use candle_core::{DType, Tensor};

use super::{EvalPrediction, xyxy_iou_scalar};

pub(crate) fn image_predictions(
    pred_xyxy: &[Vec<f32>],
    pred_scores: &[Vec<f32>],
    anchors_len: usize,
    classes: usize,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<EvalPrediction> {
    let scores_by_anchor = (0..anchors_len)
        .map(|anchor| {
            (0..classes)
                .map(|class_id| pred_scores[class_id][anchor])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    official_topk_predictions(
        pred_xyxy,
        &scores_by_anchor,
        confidence_threshold,
        iou_threshold,
        max_detections,
    )
}

pub(crate) fn batch_image_predictions_from_tensors(
    pred_xyxy: &Tensor,
    pred_scores: &Tensor,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> crate::Result<Vec<Vec<EvalPrediction>>> {
    let (batch, coords, anchors_len) = pred_xyxy.dims3()?;
    if coords != 4 {
        return Err(crate::Error::InvalidTensor(format!(
            "decoded detection boxes must have shape [batch, 4, anchors], got {:?}",
            pred_xyxy.dims()
        )));
    }
    let (score_batch, _classes, score_anchors) = pred_scores.dims3()?;
    if score_batch != batch || score_anchors != anchors_len {
        return Err(crate::Error::InvalidTensor(format!(
            "detection scores must match decoded boxes, got boxes {:?}, scores {:?}",
            pred_xyxy.dims(),
            pred_scores.dims()
        )));
    }
    let boxes = pred_xyxy.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    batch_image_predictions_with_tensor_scores(
        &boxes,
        pred_scores,
        confidence_threshold,
        iou_threshold,
        max_detections,
    )
}

pub(crate) fn batch_image_predictions_with_tensor_scores(
    boxes: &[Vec<Vec<f32>>],
    pred_scores: &Tensor,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> crate::Result<Vec<Vec<EvalPrediction>>> {
    let batch = boxes.len();
    let anchors_len = boxes
        .first()
        .and_then(|batch_boxes| batch_boxes.first())
        .map(Vec::len)
        .unwrap_or(0);
    let (score_batch, _classes, score_anchors) = pred_scores.dims3()?;
    if score_batch != batch || score_anchors != anchors_len {
        return Err(crate::Error::InvalidTensor(format!(
            "detection scores must match decoded boxes, got boxes batch {batch} anchors {anchors_len}, scores {:?}",
            pred_scores.dims()
        )));
    }
    let scores = pred_scores.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let classes = scores[b].len();
        let scores_by_anchor = (0..anchors_len)
            .map(|anchor| {
                (0..classes)
                    .map(|class_id| scores[b][class_id][anchor])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        out.push(official_topk_predictions(
            &boxes[b],
            &scores_by_anchor,
            confidence_threshold,
            iou_threshold,
            max_detections,
        ));
    }
    Ok(out)
}

pub(crate) fn batch_image_predictions_from_topk_tensor(
    predictions: &Tensor,
    confidence_threshold: f32,
    _iou_threshold: f32,
    max_detections: usize,
) -> crate::Result<Vec<Vec<EvalPrediction>>> {
    let (batch, rows, cols) = predictions.dims3()?;
    if cols < 6 {
        return Err(crate::Error::InvalidTensor(format!(
            "top-k predictions must have at least 6 columns, got {:?}",
            predictions.dims()
        )));
    }
    let rows_to_keep = rows.min(max_detections);
    let data = predictions.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut out = Vec::with_capacity(batch);
    for batch_rows in data.iter().take(batch) {
        let mut preds = Vec::new();
        for (row_idx, row) in batch_rows.iter().take(rows_to_keep).enumerate() {
            let score = row[4];
            if score <= confidence_threshold {
                continue;
            }
            preds.push(EvalPrediction {
                class_id: row[5].max(0.0) as u32,
                score,
                xyxy: [row[0], row[1], row[2], row[3]],
                anchor_idx: row_idx,
            });
        }
        out.push(preds);
    }
    Ok(out)
}

fn official_topk_predictions(
    boxes: &[Vec<f32>],
    scores_by_anchor: &[Vec<f32>],
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<EvalPrediction> {
    let anchors_len = scores_by_anchor.len();
    if max_detections == 0 || anchors_len == 0 {
        return Vec::new();
    }
    let mut candidates = Vec::with_capacity(anchors_len * scores_by_anchor[0].len());
    for anchor_idx in 0..anchors_len {
        for (class_id, score) in scores_by_anchor[anchor_idx].iter().copied().enumerate() {
            if score > confidence_threshold {
                candidates.push(EvalPrediction {
                    class_id: class_id as u32,
                    score,
                    xyxy: [
                        boxes[0][anchor_idx],
                        boxes[1][anchor_idx],
                        boxes[2][anchor_idx],
                        boxes[3][anchor_idx],
                    ],
                    anchor_idx,
                });
            }
        }
    }
    class_aware_nms(candidates, iou_threshold, max_detections)
}

fn class_aware_nms(
    mut candidates: Vec<EvalPrediction>,
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<EvalPrediction> {
    candidates.sort_unstable_by(|a, b| compare_score_desc(a.score, b.score));
    let mut keep: Vec<EvalPrediction> = Vec::with_capacity(max_detections.min(candidates.len()));
    'candidate: for candidate in candidates {
        if keep.len() == max_detections {
            break;
        }
        for kept in &keep {
            if kept.class_id == candidate.class_id
                && xyxy_iou_scalar(kept.xyxy, candidate.xyxy) > iou_threshold
            {
                continue 'candidate;
            }
        }
        keep.push(candidate);
    }
    keep
}

fn compare_score_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}
