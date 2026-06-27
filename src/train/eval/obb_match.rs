use super::{EvalPrediction, MAP_IOU_THRESHOLDS, N_THRESHOLDS, obb_geometry};

pub(crate) fn pred_rbox_xyxy_channels(
    pred_rboxes: &[Vec<f32>],
    anchors_len: usize,
) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; anchors_len]; 4];
    for anchor_idx in 0..anchors_len {
        let xyxy = obb_geometry::rbox_xyxy(pred_rbox(pred_rboxes, anchor_idx));
        for (c, out_channel) in out.iter_mut().enumerate().take(4) {
            out_channel[anchor_idx] = xyxy[c];
        }
    }
    out
}

pub(crate) fn obb_tp_matrix(
    preds: &[EvalPrediction],
    pred_rboxes: &[Vec<f32>],
    target_rboxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
) -> Vec<[bool; N_THRESHOLDS]> {
    let mut tp = vec![[false; N_THRESHOLDS]; preds.len()];
    for (ti, &thr) in MAP_IOU_THRESHOLDS.iter().enumerate() {
        let cands = obb_candidates(
            preds,
            pred_rboxes,
            target_rboxes,
            target_classes,
            target_valid,
            thr,
        );
        greedy_match(&mut tp, ti, cands, preds.len(), target_rboxes.len());
    }
    tp
}

pub(crate) fn match_obb_targets(
    preds: &[EvalPrediction],
    pred_rboxes: &[Vec<f32>],
    target_rboxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
    threshold: f32,
) -> (usize, usize) {
    let total = target_rboxes
        .iter()
        .zip(target_valid)
        .filter(|(rbox, valid)| **valid > 0.0 && valid_rbox(read_target_rbox(rbox)))
        .count();
    let cands = obb_candidates(
        preds,
        pred_rboxes,
        target_rboxes,
        target_classes,
        target_valid,
        threshold,
    );
    (greedy_count(cands, preds.len(), target_rboxes.len()), total)
}

fn obb_candidates(
    preds: &[EvalPrediction],
    pred_rboxes: &[Vec<f32>],
    target_rboxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let mut cands = Vec::new();
    for (pi, pred) in preds.iter().enumerate() {
        let pred_rbox = pred_rbox(pred_rboxes, pred.anchor_idx);
        for gi in 0..target_rboxes.len() {
            let gt = read_target_rbox(&target_rboxes[gi]);
            if target_valid[gi] <= 0.0 || pred.class_id != target_classes[gi] || !valid_rbox(gt) {
                continue;
            }
            let iou = obb_geometry::probiou(gt, pred_rbox);
            if iou >= threshold {
                cands.push((pi, gi, iou));
            }
        }
    }
    cands
}

fn greedy_match(
    tp: &mut [[bool; N_THRESHOLDS]],
    threshold_index: usize,
    cands: Vec<(usize, usize, f32)>,
    pred_count: usize,
    target_count: usize,
) {
    for (pi, _, _) in greedy_pairs(cands, pred_count, target_count) {
        tp[pi][threshold_index] = true;
    }
}

fn greedy_count(cands: Vec<(usize, usize, f32)>, pred_count: usize, target_count: usize) -> usize {
    greedy_pairs(cands, pred_count, target_count).len()
}

fn greedy_pairs(
    mut cands: Vec<(usize, usize, f32)>,
    pred_count: usize,
    target_count: usize,
) -> Vec<(usize, usize, f32)> {
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut used_pred = vec![false; pred_count];
    let mut used_gt = vec![false; target_count];
    let mut pairs = Vec::new();
    for (pi, gi, iou) in cands {
        if used_pred[pi] || used_gt[gi] {
            continue;
        }
        used_pred[pi] = true;
        used_gt[gi] = true;
        pairs.push((pi, gi, iou));
    }
    pairs
}

fn pred_rbox(pred_rboxes: &[Vec<f32>], anchor_idx: usize) -> [f32; 5] {
    [
        pred_rboxes[0][anchor_idx],
        pred_rboxes[1][anchor_idx],
        pred_rboxes[2][anchor_idx],
        pred_rboxes[3][anchor_idx],
        pred_rboxes[4][anchor_idx],
    ]
}

fn read_target_rbox(rbox: &[f32]) -> [f32; 5] {
    [rbox[0], rbox[1], rbox[2], rbox[3], rbox[4]]
}

fn valid_rbox([_, _, w, h, _]: [f32; 5]) -> bool {
    w > 0.0 && h > 0.0
}
