//! Output-buffer construction for detection target building.
//!
//! Extracted from [`crate::train::loss::build_detection_targets`]: given the resolved
//! assignments and per-object stats, fills the official-format target buffers
//! (ltrb, xyxy, scores, foreground mask, gt index) and packs them into a
//! [`BuiltDetectionTargets`].

use candle_core::Tensor;

use super::*;
use crate::train::eval::detection_assignment::{ObjectStats, PendingDetectionAssignment};

/// Per-object normalized assignment metric, pairing a resolved assignment with
/// its `mask_pos`-scoped normalized value (matching the official
/// `align_metric * pos_overlaps / (pos_align_metrics + eps)` term).
#[derive(Clone, Copy)]
pub(super) struct NormalizedAssignment {
    /// The resolved positive assignment.
    pub(super) assignment: PendingDetectionAssignment,
    /// Normalized metric for this assignment under its object's positive stats.
    pub(super) per_obj_norm: f32,
}

/// Writes the official detection target buffers from resolved assignments.
///
/// Mirrors the official `TaskAlignedAssigner` target packing: per-anchor amax
/// of the normalized metric writes `target_scores`, the ltrb/xyxy buffers store
/// the matched GT box per anchor, and the foreground mask / gt index record
/// which anchor matched which object.
#[expect(
    clippy::too_many_arguments,
    reason = "target buffer construction receives precomputed assignment views and output dimensions"
)]
pub(super) fn build_target_buffers(
    output: &DenseDetectionOutput,
    assignments: &[PendingDetectionAssignment],
    pos_stats: &[ObjectStats],
    boxes: &[Vec<Vec<f32>>],
    anchors: &[Vec<f32>],
    strides: &[f32],
    batch: usize,
    classes: usize,
    anchors_len: usize,
    max_objects: usize,
    image_width: f32,
    image_height: f32,
) -> crate::Result<BuiltDetectionTargets> {
    let normalized = normalize_assignments(assignments, pos_stats, max_objects);
    let batch_anchor_norm = per_anchor_amax(&normalized, batch, anchors_len);

    let mut target_ltrb = vec![0f32; batch * 4 * anchors_len];
    let mut target_xyxy = vec![0f32; batch * 4 * anchors_len];
    let mut target_scores = vec![0f32; batch * classes * anchors_len];
    let mut foreground_mask = vec![0f32; batch * anchors_len];
    let mut target_gt_idx = vec![usize::MAX; batch * anchors_len];

    for n in &normalized {
        let assignment = &n.assignment;
        let xyxy = &boxes[assignment.batch_idx][assignment.object_idx];
        let norm_key = assignment.batch_idx * anchors_len + assignment.anchor_idx;
        // Official: target_score = norm_align_metric (per-batch-per-anchor amax
        // already taken); no per-object clamp to [0,1].
        let target_score = batch_anchor_norm[norm_key];
        let stride = strides[assignment.anchor_idx].max(f32::EPSILON);
        let ax = anchors[assignment.anchor_idx][0];
        let ay = anchors[assignment.anchor_idx][1];
        let base = assignment.batch_idx * 4 * anchors_len + assignment.anchor_idx;
        target_ltrb[base] = (ax - xyxy[0] / stride).max(0.0);
        target_ltrb[base + anchors_len] = (ay - xyxy[1] / stride).max(0.0);
        target_ltrb[base + 2 * anchors_len] = (xyxy[2] / stride - ax).max(0.0);
        target_ltrb[base + 3 * anchors_len] = (xyxy[3] / stride - ay).max(0.0);
        target_xyxy[base] = xyxy[0];
        target_xyxy[base + anchors_len] = xyxy[1];
        target_xyxy[base + 2 * anchors_len] = xyxy[2];
        target_xyxy[base + 3 * anchors_len] = xyxy[3];
        let score_idx = assignment.batch_idx * classes * anchors_len
            + assignment.class_id * anchors_len
            + assignment.anchor_idx;
        // First assignment to this (batch, anchor, class) writes the score;
        // target_score is the per-anchor amax, so write once.
        if target_scores[score_idx] == 0.0 {
            target_scores[score_idx] = target_score;
        }
        foreground_mask[assignment.batch_idx * anchors_len + assignment.anchor_idx] = 1.0;
        target_gt_idx[assignment.batch_idx * anchors_len + assignment.anchor_idx] =
            assignment.object_idx;
    }

    let foreground_count = foreground_mask.iter().filter(|value| **value > 0.0).count() as f64;
    let target_scores_sum = target_scores.iter().map(|value| *value as f64).sum::<f64>();
    let device = output.boxes.device();
    Ok(BuiltDetectionTargets {
        target_ltrb: Tensor::from_vec(target_ltrb, (batch, 4, anchors_len), device)?
            .to_dtype(output.boxes.dtype())?,
        target_xyxy: Tensor::from_vec(target_xyxy, (batch, 4, anchors_len), device)?
            .to_dtype(output.boxes.dtype())?,
        target_scores: Tensor::from_vec(target_scores, (batch, classes, anchors_len), device)?
            .to_dtype(output.scores.dtype())?,
        foreground_mask: Tensor::from_vec(foreground_mask, (batch, 1, anchors_len), device)?
            .to_dtype(output.boxes.dtype())?,
        foreground_count,
        target_scores_sum: target_scores_sum.max(1.0),
        target_gt_idx,
        image_width,
        image_height,
    })
}

/// Computes the per-object normalized metric for each assignment, scoped to the
/// `mask_pos` positive subset stats.
fn normalize_assignments(
    assignments: &[PendingDetectionAssignment],
    pos_stats: &[ObjectStats],
    max_objects: usize,
) -> Vec<NormalizedAssignment> {
    let mut normalized = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        let stats = pos_stats[assignment.batch_idx * max_objects + assignment.object_idx];
        let per_obj_norm = if stats.pos_metric > f32::EPSILON {
            assignment.metric * stats.pos_overlap / stats.pos_metric
        } else {
            0.0
        };
        normalized.push(NormalizedAssignment {
            assignment: *assignment,
            per_obj_norm,
        });
    }
    normalized
}

/// Reduces per-object normalized metrics to the per-anchor maximum, matching
/// the official `.amax(-2)` over objects within each batch.
fn per_anchor_amax(
    normalized: &[NormalizedAssignment],
    batch: usize,
    anchors_len: usize,
) -> Vec<f32> {
    let mut batch_anchor_norm = vec![0f32; batch * anchors_len];
    for n in normalized {
        let key = n.assignment.batch_idx * anchors_len + n.assignment.anchor_idx;
        if n.per_obj_norm > batch_anchor_norm[key] {
            batch_anchor_norm[key] = n.per_obj_norm;
        }
    }
    batch_anchor_norm
}
