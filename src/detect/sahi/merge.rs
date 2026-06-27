//! Detection merging across slices (NMS / greedy NMM).

use crate::BBox;

use super::options::{MatchMetric, MergeStrategy, Options};

use crate::detect::Prediction;

/// Merges detections using the configured strategy.
pub fn merge_detections(mut detections: Vec<Prediction>, options: &Options) -> Vec<Prediction> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    match options.merge_strategy {
        MergeStrategy::Nms => nms(detections, options),
        MergeStrategy::GreedyNmm => greedy_nmm(detections, options),
    }
}

fn nms(detections: Vec<Prediction>, options: &Options) -> Vec<Prediction> {
    let n = detections.len();
    let mut suppressed = vec![false; n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            if same_merge_group(&detections[i], &detections[j], options)
                && overlap(detections[i].bbox, detections[j].bbox, options)
            {
                suppressed[j] = true;
            }
        }
    }

    detections
        .into_iter()
        .zip(suppressed.iter())
        .filter(|&(_, &s)| !s)
        .map(|(d, _)| d)
        .collect()
}

fn greedy_nmm(detections: Vec<Prediction>, options: &Options) -> Vec<Prediction> {
    let n = detections.len();
    let mut suppressed = vec![false; n];
    let mut keep_to_merge_list: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            if same_merge_group(&detections[i], &detections[j], options)
                && overlap(detections[i].bbox, detections[j].bbox, options)
            {
                keep_to_merge_list[i].push(j);
                suppressed[j] = true;
            }
        }
    }

    let mut result = Vec::new();
    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let mut merged = detections[i];
        for &j in &keep_to_merge_list[i] {
            merged = merge_pair(merged, &detections[j]);
        }
        result.push(merged);
    }

    result.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

fn same_merge_group(a: &Prediction, b: &Prediction, options: &Options) -> bool {
    options.class_agnostic || a.class_id == b.class_id
}

fn overlap(a: BBox, b: BBox, options: &Options) -> bool {
    let value = match options.match_metric {
        MatchMetric::Iou => a.iou(b),
        MatchMetric::Ios => a.ios(b),
    };
    value >= options.match_threshold
}

/// Merges two predictions: bbox union + max score.
fn merge_pair(keeper: Prediction, candidate: &Prediction) -> Prediction {
    let bbox = BBox {
        x_min: keeper.bbox.x_min.min(candidate.bbox.x_min),
        y_min: keeper.bbox.y_min.min(candidate.bbox.y_min),
        x_max: keeper.bbox.x_max.max(candidate.bbox.x_max),
        y_max: keeper.bbox.y_max.max(candidate.bbox.y_max),
    };
    Prediction {
        bbox,
        confidence: keeper.confidence.max(candidate.confidence),
        class_id: match keeper.confidence >= candidate.confidence {
            true => keeper.class_id,
            false => candidate.class_id,
        },
    }
}
