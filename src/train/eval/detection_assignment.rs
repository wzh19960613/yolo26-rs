use super::*;

pub(crate) use crate::train::eval::detection_geometry::Geometry;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PendingDetectionAssignment {
    pub(crate) batch_idx: usize,
    pub(crate) object_idx: usize,
    pub(crate) class_id: usize,
    pub(crate) anchor_idx: usize,
    pub(crate) metric: f32,
    pub(crate) overlap: f32,
    pub(crate) max_metric: f32,
    /// Maximum IoU among the object's in-GT candidate anchors.
    pub(crate) max_overlap: f32,
}

/// Per-object statistics computed on the final `mask_pos` positive subset,
/// matching the official `TaskAlignedAssigner._forward`'s
/// `pos_overlaps = (overlaps * mask_pos).amax(-1)` and
/// `pos_align_metrics = (align_metric * mask_pos).amax(-1)`.
#[derive(Clone, Copy, Default)]
pub(crate) struct ObjectStats {
    /// `(overlaps * mask_pos).amax`: max IoU of this object's positive anchors.
    pub pos_overlap: f32,
    /// `(align_metric * mask_pos).amax`: max align metric of positive anchors.
    pub pos_metric: f32,
}

/// Resolved assignments paired with the per-object `mask_pos`-scoped stats
/// needed for the official `target_scores` normalization.
pub(crate) struct AssignmentResult {
    /// Final positive assignments after conflict resolution + topk2 trim.
    pub assignments: Vec<PendingDetectionAssignment>,
    /// Per-object stats `[(batch, object)]` from the `mask_pos` subset.
    pub stats: Vec<ObjectStats>,
}

/// Resolves task-aligned detection assignments, mirroring the official
/// `TaskAlignedAssigner._forward` pipeline (see module-level doc in
/// `build_detection_targets`). Returns both the final positive assignments
/// and the per-object `mask_pos`-scoped `pos_overlap` / `pos_metric` used by
/// `target_scores` normalization (`norm_align_metric`).
pub(crate) fn resolve_detection_assignments(
    assignments: &[PendingDetectionAssignment],
    batch: usize,
    anchors_len: usize,
    max_objects: usize,
    config: DetectionLossConfig,
    geo: &Geometry,
) -> AssignmentResult {
    let primary = config.tal_primary_topk();
    let secondary = config.tal_secondary_topk();
    // Per-object candidate indices into `assignments`.
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); batch * max_objects];
    for (idx, a) in assignments.iter().enumerate() {
        candidates[a.batch_idx * max_objects + a.object_idx].push(idx);
    }

    // Phase 1: select_candidates_in_gts + select_topk_candidates.
    // pos_mask[(b, obj)] = topk ∩ in_gt. The official count>1 filter lives
    // inside a single GT's scattered top-k indices; because `assignments`
    // contains each anchor once per GT, cross-GT conflicts must be resolved by
    // IoU argmax below, not removed here.
    let mut pos_mask: Vec<Vec<bool>> = vec![vec![false; anchors_len]; batch * max_objects];
    for (obj_key, idxs) in candidates.iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let batch_idx = obj_key / max_objects;
        let object_idx = obj_key % max_objects;
        let mut inside: Vec<usize> = idxs
            .iter()
            .copied()
            .filter(|&i| geo.anchor_in_gt(batch_idx, object_idx, assignments[i].anchor_idx))
            .collect();
        inside.sort_by(|&a, &b| {
            assignments[b]
                .metric
                .partial_cmp(&assignments[a].metric)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| assignments[a].anchor_idx.cmp(&assignments[b].anchor_idx))
        });
        for &i in inside.iter().take(primary) {
            pos_mask[obj_key][assignments[i].anchor_idx] = true;
        }
    }

    // Phase 2: conflict resolution (overlap-argmax) + topk2 trim. When one
    // anchor is claimed by multiple GTs, official Ultralytics chooses the GT
    // with max overlap across all valid GTs for that anchor, then applies the
    // optional secondary top-k on the updated mask.
    let fg_object =
        resolve_anchor_conflicts(assignments, &pos_mask, batch, anchors_len, max_objects);
    let mut final_by_object: Vec<Vec<usize>> = vec![Vec::new(); batch * max_objects];
    for (key, winning) in fg_object.iter().enumerate() {
        let Some(object_idx) = winning else {
            continue;
        };
        let object_idx = *object_idx;
        let batch_idx = key / anchors_len;
        let anchor_idx = key % anchors_len;
        final_by_object[batch_idx * max_objects + object_idx].push(anchor_idx);
    }
    let mut out: Vec<PendingDetectionAssignment> = Vec::new();
    for (obj_key, anchors_chosen) in final_by_object.iter().enumerate() {
        if anchors_chosen.is_empty() {
            continue;
        }
        let batch_idx = obj_key / max_objects;
        let object_idx = obj_key % max_objects;
        let mut chosen: Vec<usize> = anchors_chosen.clone();
        if secondary < primary && chosen.len() > secondary {
            chosen.sort_by(|&a, &b| {
                let ma = metric_of(assignments, obj_key, a, max_objects);
                let mb = metric_of(assignments, obj_key, b, max_objects);
                mb.partial_cmp(&ma)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            chosen.truncate(secondary);
        }
        for anchor_idx in chosen {
            if let Some(assignment) =
                find_assignment(assignments, batch_idx, object_idx, anchor_idx)
            {
                out.push(*assignment);
            }
        }
    }
    let stats = compute_object_stats(&out, batch, max_objects);
    AssignmentResult {
        assignments: out,
        stats,
    }
}

fn resolve_anchor_conflicts(
    assignments: &[PendingDetectionAssignment],
    pos_mask: &[Vec<bool>],
    batch: usize,
    anchors_len: usize,
    max_objects: usize,
) -> Vec<Option<usize>> {
    let mut fg_object: Vec<Option<usize>> = vec![None; batch * anchors_len];
    for batch_idx in 0..batch {
        for anchor_idx in 0..anchors_len {
            let mut claim_count = 0usize;
            let mut only_claim = None;
            for object_idx in 0..max_objects {
                if pos_mask[batch_idx * max_objects + object_idx][anchor_idx] {
                    claim_count += 1;
                    only_claim = Some(object_idx);
                }
            }
            let winner = match claim_count {
                0 => None,
                1 => only_claim,
                _ => max_overlap_object(assignments, batch_idx, anchor_idx),
            };
            fg_object[batch_idx * anchors_len + anchor_idx] = winner;
        }
    }
    fg_object
}

fn max_overlap_object(
    assignments: &[PendingDetectionAssignment],
    batch_idx: usize,
    anchor_idx: usize,
) -> Option<usize> {
    assignments
        .iter()
        .filter(|a| a.batch_idx == batch_idx && a.anchor_idx == anchor_idx)
        .max_by(|a, b| {
            a.overlap
                .partial_cmp(&b.overlap)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.object_idx.cmp(&a.object_idx))
        })
        .map(|a| a.object_idx)
}

fn compute_object_stats(
    assignments: &[PendingDetectionAssignment],
    batch: usize,
    max_objects: usize,
) -> Vec<ObjectStats> {
    let mut stats = vec![ObjectStats::default(); batch * max_objects];
    for assignment in assignments {
        let key = assignment.batch_idx * max_objects + assignment.object_idx;
        stats[key].pos_overlap = stats[key].pos_overlap.max(assignment.overlap);
        stats[key].pos_metric = stats[key].pos_metric.max(assignment.metric);
    }
    stats
}

fn find_assignment(
    assignments: &[PendingDetectionAssignment],
    batch_idx: usize,
    object_idx: usize,
    anchor_idx: usize,
) -> Option<&PendingDetectionAssignment> {
    assignments.iter().find(|a| {
        a.batch_idx == batch_idx && a.object_idx == object_idx && a.anchor_idx == anchor_idx
    })
}

fn metric_of(
    assignments: &[PendingDetectionAssignment],
    obj_key: usize,
    anchor_idx: usize,
    max_objects: usize,
) -> f32 {
    assignments
        .iter()
        .find(|a| a.batch_idx * max_objects + a.object_idx == obj_key && a.anchor_idx == anchor_idx)
        .map(|a| a.metric)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assignment(
        object_idx: usize,
        anchor_idx: usize,
        metric: f32,
        overlap: f32,
    ) -> PendingDetectionAssignment {
        PendingDetectionAssignment {
            batch_idx: 0,
            object_idx,
            class_id: 0,
            anchor_idx,
            metric,
            overlap,
            max_metric: 0.0,
            max_overlap: 0.0,
        }
    }

    #[test]
    fn conflict_resolution_keeps_highest_overlap_gt() {
        let assignments = vec![assignment(0, 0, 0.9, 0.2), assignment(1, 0, 0.8, 0.7)];
        let geometry = Geometry {
            gt_boxes: vec![[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
            anchor_centers_pixel: vec![(5.0, 5.0)],
            max_objects: 2,
        };

        let result = resolve_detection_assignments(
            &assignments,
            1,
            1,
            2,
            DetectionLossConfig {
                tal_topk: 1,
                ..Default::default()
            },
            &geometry,
        );

        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].object_idx, 1);
    }

    #[test]
    fn normalization_stats_are_computed_after_secondary_topk() {
        let assignments = vec![assignment(0, 0, 0.9, 0.4), assignment(0, 1, 0.8, 0.99)];
        let geometry = Geometry {
            gt_boxes: vec![[0.0, 0.0, 20.0, 20.0]],
            anchor_centers_pixel: vec![(5.0, 5.0), (6.0, 6.0)],
            max_objects: 1,
        };

        let result = resolve_detection_assignments(
            &assignments,
            1,
            2,
            1,
            DetectionLossConfig {
                tal_topk: 2,
                tal_topk2: Some(1),
                ..Default::default()
            },
            &geometry,
        );

        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].anchor_idx, 0);
        assert!((result.stats[0].pos_metric - 0.9).abs() < 1e-6);
        assert!((result.stats[0].pos_overlap - 0.4).abs() < 1e-6);
    }
}
