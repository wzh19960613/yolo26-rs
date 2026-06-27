use super::{DetectionEvalMetrics, EvalLoopReport};

impl DetectionEvalMetrics {
    /// Returns the F1 score derived from precision and recall.
    pub fn f1(&self) -> f32 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall <= f32::EPSILON {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

impl EvalLoopReport {
    /// Returns native validation fitness for best-checkpoint selection.
    ///
    /// Detection-style tasks use the official Ultralytics fitness
    /// `mAP@0.5:0.95`; instance segmentation adds box and mask fitness, and
    /// pose adds box and OKS pose fitness. Classification uses top-1/top-5
    /// mean. Other tasks use negative mean loss as a finite fallback, matching
    /// the official trainer's loss fallback when no validator fitness is available.
    pub fn fitness(&self) -> f32 {
        if let (Some(map), Some(mask_map)) = (self.map, self.mask_map) {
            map.fitness() + mask_map.fitness()
        } else if let (Some(map), Some(pose_map)) = (self.map, self.pose_map) {
            map.fitness() + pose_map.fitness()
        } else if let Some(map) = self.map {
            map.fitness()
        } else if let Some(classification) = self.classification {
            (classification.accuracy() + classification.top5_accuracy()) / 2.0
        } else if let Some(semantic) = &self.semantic {
            semantic.fitness()
        } else if let Some(detection) = self.detection {
            detection.f1()
        } else {
            -self.mean_loss
        }
    }
}
