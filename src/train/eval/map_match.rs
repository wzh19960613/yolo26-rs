use super::{EvalPrediction, MAP_IOU_THRESHOLDS, N_THRESHOLDS, xyxy_iou_scalar};

/// Builds the `[N_pred][N_THRESHOLDS]` true-positive matrix for one image.
pub(crate) fn image_tp_matrix(
    preds: &[EvalPrediction],
    target_boxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
) -> Vec<[bool; N_THRESHOLDS]> {
    let np = preds.len();
    let mut tp = vec![[false; N_THRESHOLDS]; np];
    let ng = target_boxes.len();
    for (tj, &thr) in MAP_IOU_THRESHOLDS.iter().enumerate() {
        let mut cands: Vec<(usize, usize, f32)> = Vec::new();
        for (pi, pred) in preds.iter().enumerate().take(np) {
            for gi in 0..ng {
                if target_valid[gi] <= 0.0 || pred.class_id != target_classes[gi] {
                    continue;
                }
                let g = read_box(&target_boxes[gi]);
                if g[2] <= g[0] || g[3] <= g[1] {
                    continue;
                }
                let iou = xyxy_iou_scalar(pred.xyxy, g);
                if iou >= thr {
                    cands.push((pi, gi, iou));
                }
            }
        }
        mark_official_matches(&mut tp, tj, np, ng, cands);
    }
    tp
}

pub(crate) fn read_box(b: &[f32]) -> [f32; 4] {
    [b[0], b[1], b[2], b[3]]
}

pub(crate) fn mark_official_matches(
    tp: &mut [[bool; N_THRESHOLDS]],
    threshold_idx: usize,
    preds: usize,
    targets: usize,
    mut cands: Vec<(usize, usize, f32)>,
) {
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_pred = vec![false; preds];
    let mut pred_unique = Vec::new();
    for cand in cands {
        if used_pred[cand.0] {
            continue;
        }
        used_pred[cand.0] = true;
        pred_unique.push(cand);
    }

    // NumPy's `np.unique(matches[:, 1], return_index=True)[1]` returns the
    // retained indices ordered by detection index, not by descending IoU. The
    // following GT de-dup therefore keeps the earliest/highest-confidence
    // detection row for each GT among detections that survived the IoU sort.
    pred_unique.sort_by_key(|cand| cand.0);
    let mut used_gt = vec![false; targets];
    for (pi, gi, _) in pred_unique {
        if used_gt[gi] {
            continue;
        }
        tp[pi][threshold_idx] = true;
        used_gt[gi] = true;
    }
}
