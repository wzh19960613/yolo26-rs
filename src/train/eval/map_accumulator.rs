//! Dataset-level mAP accumulation matching Ultralytics DetMetrics.

use std::collections::HashMap;

use super::{EvalPrediction, compute_ap};

/// Official COCO IoU thresholds `[0.5, 0.55, ..., 0.95]` for mAP@0.5:0.95.
pub const MAP_IOU_THRESHOLDS: [f32; 10] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];

pub(crate) const N_THRESHOLDS: usize = MAP_IOU_THRESHOLDS.len();

/// Final detection-style mAP summary.
///
/// `map50` is mean AP at IoU 0.5 and `map50_95` is mean AP averaged over the ten
/// official IoU thresholds `[0.5:0.95]`, both averaged over classes that have at
/// least one ground-truth object (matching Ultralytics `ap_per_class`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapReport {
    /// Mean Average Precision at IoU 0.5, averaged over classes with GTs.
    pub map50: f32,
    /// Mean Average Precision at IoU 0.75, averaged over classes with GTs.
    pub map75: f32,
    /// Mean Average Precision averaged over `[0.5:0.95]`, averaged over classes.
    pub map50_95: f32,
    /// Precision at the max-F1 confidence threshold (pooled over classes, IoU 0.5).
    pub precision: f32,
    /// Recall at the max-F1 confidence threshold (pooled over classes, IoU 0.5).
    pub recall: f32,
}

impl MapReport {
    /// Official Ultralytics model fitness (`[0, 0, 0, 1]` weights over
    /// `[P, R, mAP50, mAP50-95]`), i.e. mAP@0.5:0.95.
    pub fn fitness(&self) -> f32 {
        self.map50_95
    }
}

/// One accumulated detection with its per-threshold true-positive flags.
#[derive(Clone, Copy)]
struct DetectionRecord {
    class: u32,
    score: f32,
    tp: [bool; N_THRESHOLDS],
}

/// Accumulates per-image detections and ground truths to compute mAP.
///
/// Feed one image at a time via [`MapAccumulator::add_image`] (predictions must
/// already be confidence-filtered, sorted by descending score and truncated to
/// `max_detections`, matching the existing detection-style eval pipeline), then
/// call [`MapAccumulator::finalize`] once to produce a [`MapReport`].
#[derive(Default)]
pub(crate) struct MapAccumulator {
    records: Vec<DetectionRecord>,
    gt_counts: HashMap<u32, usize>,
}

impl MapAccumulator {
    /// Creates an empty accumulator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds one image's predictions and ground truths.
    pub(crate) fn add_image(
        &mut self,
        preds: &[EvalPrediction],
        target_boxes: &[Vec<f32>],
        target_classes: &[u32],
        target_valid: &[f32],
    ) {
        let tp = super::image_tp_matrix(preds, target_boxes, target_classes, target_valid);
        self.add_image_with_tp(preds, target_boxes, target_classes, target_valid, &tp);
    }

    pub(crate) fn add_image_with_tp(
        &mut self,
        preds: &[EvalPrediction],
        target_boxes: &[Vec<f32>],
        target_classes: &[u32],
        target_valid: &[f32],
        tp: &[[bool; N_THRESHOLDS]],
    ) {
        self.add_gt_counts(target_boxes, target_classes, target_valid);
        for (pi, pred) in preds.iter().enumerate() {
            self.records.push(DetectionRecord {
                class: pred.class_id,
                score: pred.score,
                tp: tp.get(pi).copied().unwrap_or([false; N_THRESHOLDS]),
            });
        }
    }

    fn add_gt_counts(
        &mut self,
        target_boxes: &[Vec<f32>],
        target_classes: &[u32],
        target_valid: &[f32],
    ) {
        for obj in 0..target_boxes.len() {
            if target_valid[obj] <= 0.0 {
                continue;
            }
            let g = super::read_box(&target_boxes[obj]);
            if g[2] > g[0] && g[3] > g[1] {
                *self.gt_counts.entry(target_classes[obj]).or_insert(0) += 1;
            }
        }
    }

    /// Computes mAP50 and mAP@0.5:0.95 over the accumulated dataset.
    pub(crate) fn finalize(&self) -> MapReport {
        if self.gt_counts.is_empty() {
            return MapReport {
                map50: 0.0,
                map75: 0.0,
                map50_95: 0.0,
                precision: 0.0,
                recall: 0.0,
            };
        }
        let mut ap_sum50 = 0.0f64;
        let mut ap_sum75 = 0.0f64;
        let mut ap_sum_all = 0.0f64;
        for (&class, &n_l) in &self.gt_counts {
            if n_l == 0 {
                continue;
            }
            let mut recs: Vec<&DetectionRecord> =
                self.records.iter().filter(|r| r.class == class).collect();
            if recs.is_empty() {
                continue;
            }
            recs.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut ap_row = [0.0f64; N_THRESHOLDS];
            for (j, ap) in ap_row.iter_mut().enumerate().take(N_THRESHOLDS) {
                let (recall, precision) = class_pr_curve(&recs, j, n_l);
                *ap = compute_ap(&recall, &precision) as f64;
            }
            ap_sum50 += ap_row[0];
            ap_sum75 += ap_row[5];
            ap_sum_all += ap_row.iter().sum::<f64>() / N_THRESHOLDS as f64;
        }
        let nc = self.gt_counts.len() as f64;
        let total_gt = self.gt_counts.values().sum::<usize>() as f32;
        let (precision, recall) = pooled_max_f1_pr(&self.records, total_gt);
        MapReport {
            map50: (ap_sum50 / nc) as f32,
            map75: (ap_sum75 / nc) as f32,
            map50_95: (ap_sum_all / nc) as f32,
            precision,
            recall,
        }
    }
}

/// Returns the (precision, recall) at the max-F1 point of the pooled PR curve
/// at IoU 0.5, matching the official `box(P, R)` reporting point.
///
/// Simplified vs the official pipeline (EVAL-03): detections are pooled across
/// all classes and the *raw* cumulative PR curve is used — no 1000-point
/// interpolation, no 0.1 smoothing, no per-class averaging — before the max-F1
/// point is picked. The value is for report readability only; [`MapReport::fitness`]
/// uses `map50_95` and does not depend on it.
fn pooled_max_f1_pr(records: &[DetectionRecord], total_gt: f32) -> (f32, f32) {
    if records.is_empty() || total_gt <= 0.0 {
        return (0.0, 0.0);
    }
    let mut sorted = records.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut tp, mut fp) = (0.0f32, 0.0f32);
    let (mut best_f1, mut best_p, mut best_r) = (-1.0f32, 0.0, 0.0);
    for r in sorted {
        if r.tp[0] {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let denom = tp + fp;
        if denom <= 0.0 {
            continue;
        }
        let (precision, recall) = (tp / denom, tp / (total_gt + 1e-16));
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        if f1 > best_f1 {
            best_f1 = f1;
            best_p = precision;
            best_r = recall;
        }
    }
    (best_p, best_r)
}

/// Cumulative precision/recall curves for one class at one IoU threshold.
fn class_pr_curve(
    recs: &[&DetectionRecord],
    threshold_index: usize,
    n_labels: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut recall = Vec::with_capacity(recs.len());
    let mut precision = Vec::with_capacity(recs.len());
    let mut tpc = 0.0f64;
    let mut fpc = 0.0f64;
    let n_l = n_labels as f64;
    for rec in recs {
        if rec.tp[threshold_index] {
            tpc += 1.0;
        } else {
            fpc += 1.0;
        }
        recall.push((tpc / (n_l + 1e-16)) as f32);
        precision.push((tpc / (tpc + fpc)) as f32);
    }
    (recall, precision)
}
