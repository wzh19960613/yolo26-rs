use super::{EvalPrediction, MAP_IOU_THRESHOLDS, N_THRESHOLDS, mark_official_matches, read_box};

pub(crate) fn mask_tp_matrix(
    preds: &[EvalPrediction],
    pred_masks: &[Vec<f32>],
    target_boxes: &[Vec<f32>],
    target_classes: &[u32],
    target_valid: &[f32],
    gt_masks: &[Vec<f32>],
) -> Vec<[bool; N_THRESHOLDS]> {
    let mut tp = vec![[false; N_THRESHOLDS]; preds.len()];
    for (tj, &thr) in MAP_IOU_THRESHOLDS.iter().enumerate() {
        let mut cands = Vec::new();
        for (pi, pred) in preds.iter().enumerate() {
            for gi in 0..target_boxes.len() {
                if target_valid[gi] <= 0.0 || pred.class_id != target_classes[gi] {
                    continue;
                }
                let gt = read_box(&target_boxes[gi]);
                if gt[2] <= gt[0] || gt[3] <= gt[1] {
                    continue;
                }
                let iou = binary_mask_iou(&gt_masks[gi], &pred_masks[pi]);
                if iou >= thr {
                    cands.push((pi, gi, iou));
                }
            }
        }
        mark_official_matches(&mut tp, tj, preds.len(), target_boxes.len(), cands);
    }
    tp
}

fn binary_mask_iou(a: &[f32], b: &[f32]) -> f32 {
    let (mut inter, mut area_a, mut area_b) = (0.0f32, 0.0f32, 0.0f32);
    for (&av, &bv) in a.iter().zip(b) {
        let aa = av > 0.5;
        let bb = bv > 0.5;
        area_a += aa as u8 as f32;
        area_b += bb as u8 as f32;
        inter += (aa && bb) as u8 as f32;
    }
    inter / (area_a + area_b - inter + 1e-7)
}
