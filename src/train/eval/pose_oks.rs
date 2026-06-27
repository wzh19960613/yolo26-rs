use super::{EvalPrediction, MAP_IOU_THRESHOLDS, N_THRESHOLDS, read_box};

const COCO_OKS_SIGMA: [f32; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

#[expect(
    clippy::too_many_arguments,
    reason = "OKS matching consumes prediction, target, area, visibility, and threshold views"
)]
pub(crate) fn pose_tp_matrix(
    preds: &[EvalPrediction],
    pred_keypoints: &[Vec<f32>],
    target_boxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
    target_keypoints: &[f32],
    objects: usize,
    keypoints_count: usize,
    target_visibility: &[Vec<f32>],
) -> Vec<[bool; N_THRESHOLDS]> {
    let mut tp = vec![[false; N_THRESHOLDS]; preds.len()];
    for (ti, &thr) in MAP_IOU_THRESHOLDS.iter().enumerate() {
        let mut cands = Vec::new();
        for (pi, pred) in preds.iter().enumerate() {
            for gi in 0..target_boxes.len().min(objects) {
                if target_valid[gi] <= 0.0 || pred.class_id != target_classes[gi] {
                    continue;
                }
                let oks = oks_score(
                    pred_keypoints,
                    pred.anchor_idx,
                    &target_keypoints[gi * keypoints_count * 2..],
                    keypoints_count,
                    &target_visibility[gi],
                    area53(read_box(&target_boxes[gi])),
                );
                if oks >= thr {
                    cands.push((pi, gi, oks));
                }
            }
        }
        greedy_match(&mut tp, ti, cands, preds.len(), target_boxes.len());
    }
    tp
}

fn greedy_match(
    tp: &mut [[bool; N_THRESHOLDS]],
    threshold_index: usize,
    mut cands: Vec<(usize, usize, f32)>,
    pred_count: usize,
    target_count: usize,
) {
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut used_pred = vec![false; pred_count];
    let mut used_gt = vec![false; target_count];
    for (pi, gi, _) in cands {
        if used_pred[pi] || used_gt[gi] {
            continue;
        }
        tp[pi][threshold_index] = true;
        used_pred[pi] = true;
        used_gt[gi] = true;
    }
}

fn oks_score(
    pred_keypoints: &[Vec<f32>],
    anchor_idx: usize,
    gt_keypoints: &[f32],
    keypoints_count: usize,
    gt_visibility: &[f32],
    area: f32,
) -> f32 {
    let sigmas = oks_sigmas(keypoints_count);
    let mut sum = 0.0f32;
    let mut visible = 0.0f32;
    for k in 0..keypoints_count {
        if gt_visibility[k] <= 0.0 {
            continue;
        }
        let dx = gt_keypoints[k * 2] - pred_keypoints[k * 2][anchor_idx];
        let dy = gt_keypoints[k * 2 + 1] - pred_keypoints[k * 2 + 1][anchor_idx];
        let denom = (2.0 * sigmas[k]).powi(2) * (area + 1e-7) * 2.0;
        sum += (-(dx * dx + dy * dy) / denom).exp();
        visible += 1.0;
    }
    if visible > 0.0 {
        sum / (visible + 1e-7)
    } else {
        0.0
    }
}

fn oks_sigmas(nk: usize) -> Vec<f32> {
    if nk == COCO_OKS_SIGMA.len() {
        COCO_OKS_SIGMA.to_vec()
    } else {
        vec![1.0 / nk.max(1) as f32; nk]
    }
}

fn area53(xyxy: [f32; 4]) -> f32 {
    ((xyxy[2] - xyxy[0]).max(0.0) * (xyxy[3] - xyxy[1]).max(0.0)) * 0.53
}
