#![allow(clippy::items_after_test_module)]

use super::*;

pub(crate) fn build_assigned_angle_targets(
    angles: &Tensor,
    target_gt_idx: &[usize],
    batch: usize,
    anchors_len: usize,
) -> crate::Result<AssignedAngleTargets> {
    let objects = angles.dim(1)?;
    let angle_data = angles
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut assigned = vec![0f32; batch * anchors_len];
    let mut mask = vec![0f32; batch * anchors_len];
    let mut count = 0f64;
    for b in 0..batch {
        for anchor_idx in 0..anchors_len {
            let obj = target_gt_idx[b * anchors_len + anchor_idx];
            if obj == usize::MAX || obj >= objects {
                continue;
            }
            let dst = b * anchors_len + anchor_idx;
            assigned[dst] = angle_data[b * objects + obj];
            mask[dst] = 1.0;
            count += 1.0;
        }
    }
    let device = angles.device();
    Ok(AssignedAngleTargets {
        angles: Tensor::from_vec(assigned, (batch, 1, anchors_len), device)?,
        mask: Tensor::from_vec(mask, (batch, 1, anchors_len), device)?,
        count,
    })
}

#[derive(Clone, Copy)]

pub(crate) struct AssignmentCandidate {
    pub(crate) anchor_idx: usize,
    pub(crate) metric: f32,
    pub(crate) overlap: f32,
}

#[expect(
    clippy::too_many_arguments,
    reason = "task-aligned assignment computes one candidate set from decomposed target/prediction views"
)]
pub(crate) fn task_aligned_candidates(
    batch_idx: usize,
    class_id: usize,
    xyxy: &[f32],
    anchors: &[Vec<f32>],
    strides: &[f32],
    pred_scores: &[Vec<Vec<f32>>],
    pred_xyxy: &[Vec<Vec<f32>>],
    config: DetectionLossConfig,
) -> Vec<AssignmentCandidate> {
    // Match the official `TaskAlignedAssigner.get_box_metrics`: compute the
    // alignment metric for EVERY anchor (no spatial pre-filter). IoU naturally
    // down-weights far-away anchors; the earlier `stal_candidate_bounds`
    // pre-filter could drop anchors the official keeps, changing the topk set.
    let _ = strides;
    let mut candidates = Vec::new();
    for (idx, anchor) in anchors.iter().enumerate() {
        let _ = anchor;
        let pred = [
            pred_xyxy[batch_idx][0][idx],
            pred_xyxy[batch_idx][1][idx],
            pred_xyxy[batch_idx][2][idx],
            pred_xyxy[batch_idx][3][idx],
        ];
        let iou = box_ciou_xyxy(pred, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]).max(0.0);
        let score = pred_scores[batch_idx][class_id][idx].clamp(0.0, 1.0);
        let metric = score.powf(config.tal_alpha) * iou.powf(config.tal_beta);
        candidates.push(AssignmentCandidate {
            anchor_idx: idx,
            metric,
            overlap: iou,
        });
    }
    candidates
}

fn box_ciou_xyxy(a: [f32; 4], b: [f32; 4]) -> f32 {
    const EPS: f32 = 1e-7;
    let inter_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let inter_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = inter_w * inter_h;
    let aw = a[2] - a[0];
    let ah = a[3] - a[1] + EPS;
    let bw = b[2] - b[0];
    let bh = b[3] - b[1] + EPS;
    let union = aw * ah + bw * bh - inter + EPS;
    let iou = inter / union;

    let cw = a[2].max(b[2]) - a[0].min(b[0]);
    let ch = a[3].max(b[3]) - a[1].min(b[1]);
    let c2 = cw * cw + ch * ch + EPS;
    let dx = (b[0] + b[2]) - (a[0] + a[2]);
    let dy = (b[1] + b[3]) - (a[1] + a[3]);
    let rho2 = (dx * dx + dy * dy) * 0.25;

    let v = (4.0 / std::f32::consts::PI.powi(2)) * ((bw / bh).atan() - (aw / ah).atan()).powi(2);
    let alpha = v / (v - iou + 1.0 + EPS);
    iou - (rho2 / c2 + v * alpha)
}

pub(crate) fn decode_detection_boxes(output: &DenseDetectionOutput) -> crate::Result<Tensor> {
    let lt = output.boxes.narrow(1, 0, 2)?;
    let rb = output.boxes.narrow(1, 2, 2)?;
    let x1y1 = output.anchors.broadcast_sub(&lt)?;
    let x2y2 = output.anchors.broadcast_add(&rb)?;
    Ok(Tensor::cat(&[&x1y1, &x2y2], 1)?.broadcast_mul(&output.stride_tensor)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_detection_boxes_matches_dist2bbox_without_clamping() -> crate::Result<()> {
        let device = Device::Cpu;
        let output = DenseDetectionOutput {
            boxes: Tensor::from_vec(vec![-1.0f32, -2.0, -3.0, -4.0], (1, 4, 1), &device)?,
            scores: Tensor::from_vec(vec![0.0f32], (1, 1, 1), &device)?,
            anchors: Tensor::from_vec(vec![5.0f32, 6.0], (1, 2, 1), &device)?,
            stride_tensor: Tensor::from_vec(vec![8.0f32], (1, 1, 1), &device)?,
        };

        let decoded = decode_detection_boxes(&output)?.to_vec3::<f32>()?;

        assert_eq!(decoded[0][0][0], 48.0);
        assert_eq!(decoded[0][1][0], 64.0);
        assert_eq!(decoded[0][2][0], 16.0);
        assert_eq!(decoded[0][3][0], 16.0);
        Ok(())
    }

    #[test]
    fn bce_with_logits_is_non_negative_for_saturated_logits() -> crate::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![-100.0f32, 100.0, 0.0, 0.0], (1, 4), &device)?;
        let target = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], (1, 4), &device)?;

        let losses = bce_with_logits_elementwise(&logits, &target)?.to_vec2::<f32>()?;

        for value in losses[0].iter().copied() {
            assert!(value >= 0.0, "BCE loss must be non-negative, got {value}");
        }
        assert!(losses[0][0] < 1e-4);
        assert!(losses[0][1] < 1e-4);
        assert!((losses[0][2] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((losses[0][3] - std::f32::consts::LN_2).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn task_aligned_topk_is_applied_after_gt_geometry_filter() {
        let xyxy = [10.0f32, 10.0, 20.0, 20.0];
        let anchors = vec![vec![50.0f32, 50.0], vec![15.0, 15.0]];
        let strides = vec![1.0f32, 1.0];
        let pred_scores = vec![vec![vec![0.99f32, 0.5]]];
        let pred_xyxy = vec![vec![
            vec![10.0f32, 12.0],
            vec![10.0f32, 12.0],
            vec![20.0f32, 18.0],
            vec![20.0f32, 18.0],
        ]];
        let config = DetectionLossConfig {
            tal_topk: 1,
            ..Default::default()
        };

        let candidates = task_aligned_candidates(
            0,
            0,
            &xyxy,
            &anchors,
            &strides,
            &pred_scores,
            &pred_xyxy,
            config,
        );
        let pending = candidates
            .into_iter()
            .map(
                |candidate| crate::train::eval::detection_assignment::PendingDetectionAssignment {
                    batch_idx: 0,
                    object_idx: 0,
                    class_id: 0,
                    anchor_idx: candidate.anchor_idx,
                    metric: candidate.metric,
                    overlap: candidate.overlap,
                    max_metric: 0.0,
                    max_overlap: 0.0,
                },
            )
            .collect::<Vec<_>>();
        let geometry = crate::train::eval::detection_assignment::Geometry {
            gt_boxes: vec![xyxy],
            anchor_centers_pixel: vec![(50.0, 50.0), (15.0, 15.0)],
            max_objects: 1,
        };

        let result = crate::train::eval::detection_assignment::resolve_detection_assignments(
            &pending, 1, 2, 1, config, &geometry,
        );

        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].anchor_idx, 1);
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn xyxy_iou(pred: &Tensor, target: &Tensor) -> crate::Result<Tensor> {
    let px1 = pred.narrow(1, 0, 1)?;
    let py1 = pred.narrow(1, 1, 1)?;
    let px2 = pred.narrow(1, 2, 1)?;
    let py2 = pred.narrow(1, 3, 1)?;
    let tx1 = target.narrow(1, 0, 1)?;
    let ty1 = target.narrow(1, 1, 1)?;
    let tx2 = target.narrow(1, 2, 1)?;
    let ty2 = target.narrow(1, 3, 1)?;

    let inter_w = (px2.minimum(&tx2)? - px1.maximum(&tx1)?)?.maximum(0f32)?;
    let inter_h = (py2.minimum(&ty2)? - py1.maximum(&ty1)?)?.maximum(0f32)?;
    let inter = (inter_w * inter_h)?;
    let pred_area = (px2 - px1)?
        .maximum(0f32)?
        .broadcast_mul(&(py2 - py1)?.maximum(0f32)?)?;
    let target_area = (tx2 - tx1)?
        .maximum(0f32)?
        .broadcast_mul(&(ty2 - ty1)?.maximum(0f32)?)?;
    let union = ((pred_area + target_area)? - &inter)?.maximum(1e-7f32)?;
    Ok(inter.broadcast_div(&union)?)
}

pub(crate) fn bce_with_logits_sum(logits: &Tensor, target: &Tensor) -> crate::Result<Tensor> {
    bce_with_logits_elementwise(logits, target)?
        .sum_all()
        .map_err(Into::into)
}

pub(crate) fn bce_with_logits_elementwise(
    logits: &Tensor,
    target: &Tensor,
) -> crate::Result<Tensor> {
    let one = Tensor::new(1f32, logits.device())?.to_dtype(logits.dtype())?;
    let max_logits = logits.maximum(0f32)?;
    let target_term = logits.broadcast_mul(target)?;
    let log_term = logits.abs()?.neg()?.exp()?.broadcast_add(&one)?.log()?;
    max_logits
        .broadcast_sub(&target_term)?
        .broadcast_add(&log_term)
        .map_err(Into::into)
}
