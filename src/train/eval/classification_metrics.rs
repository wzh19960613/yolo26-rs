use super::*;

/// Classification evaluation metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassificationEvalMetrics {
    /// Correct top-1 predictions.
    pub correct: usize,
    /// Samples where the target is among the top-5 predicted classes.
    pub top5_correct: usize,
    /// Total evaluated samples.
    pub total: usize,
}

impl ClassificationEvalMetrics {
    /// Returns top-1 accuracy in `[0, 1]`.
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    /// Returns top-5 accuracy in `[0, 1]`.
    pub fn top5_accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.top5_correct as f32 / self.total as f32
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.correct += other.correct;
        self.top5_correct += other.top5_correct;
        self.total += other.total;
    }
}

/// Detection-style evaluation metrics.
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct DetectionEvalMetrics {
    /// Matched ground-truth targets.
    pub matched_targets: usize,
    /// Total valid ground-truth targets.
    pub total_targets: usize,
    /// Predictions retained after confidence filtering.
    pub predictions: usize,
}

impl DetectionEvalMetrics {
    /// Returns precision in `[0, 1]`.
    pub fn precision(&self) -> f32 {
        if self.predictions == 0 {
            0.0
        } else {
            self.matched_targets as f32 / self.predictions as f32
        }
    }

    /// Returns recall in `[0, 1]`.
    pub fn recall(&self) -> f32 {
        if self.total_targets == 0 {
            0.0
        } else {
            self.matched_targets as f32 / self.total_targets as f32
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.matched_targets += other.matched_targets;
        self.total_targets += other.total_targets;
        self.predictions += other.predictions;
    }
}

/// Evaluation result for one batch.
#[derive(Debug, Clone, PartialEq)]

pub struct EvalReport {
    /// Scalar supervised loss.
    pub loss: f32,
    /// Component-level supervised loss.
    pub components: LossComponents,
    /// Batch size in samples.
    pub samples: usize,
    /// Classification metrics when evaluating classification outputs.
    pub classification: Option<ClassificationEvalMetrics>,
    /// Detection-style metrics when evaluating dense detection outputs.
    pub detection: Option<DetectionEvalMetrics>,
}

/// Configuration for a dataset evaluation loop.
#[derive(Debug, Clone, PartialEq)]

pub struct EvalLoopConfig {
    /// Batch size in samples.
    pub batch_size: usize,
    /// Optional number of batches to evaluate.
    pub steps: Option<usize>,
    /// Maximum detection predictions retained per image for validation metrics.
    pub max_detections: usize,
    /// Confidence threshold for retained detection-style validation predictions.
    pub confidence_threshold: f32,
    /// IoU threshold used to match detection-style validation predictions.
    pub iou_threshold: f32,
    /// Dataset sample order derived from Ultralytics `seed` and `deterministic`.
    pub sample_order: super::SampleOrder,
    /// Detection-style loss gains and task-aligned assignment settings.
    pub loss_config: super::DetectionLossConfig,
    /// Optional class filtering/remapping applied before collation.
    pub class_filter: Option<super::ClassFilter>,
}

impl Default for EvalLoopConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            steps: None,
            max_detections: 300,
            confidence_threshold: 0.001,
            iou_threshold: 0.7,
            sample_order: super::SampleOrder::default(),
            loss_config: super::DetectionLossConfig::default(),
            class_filter: None,
        }
    }
}

impl EvalLoopConfig {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        if self.batch_size == 0 {
            return Err(crate::Error::InvalidConfig(
                "evaluation batch_size must be greater than zero".to_string(),
            ));
        }
        if matches!(self.steps, Some(0)) {
            return Err(crate::Error::InvalidConfig(
                "evaluation steps must be greater than zero".to_string(),
            ));
        }
        if self.max_detections == 0 {
            return Err(crate::Error::InvalidConfig(
                "evaluation max_detections must be greater than zero".to_string(),
            ));
        }
        if !(self.confidence_threshold.is_finite() && self.confidence_threshold >= 0.0) {
            return Err(crate::Error::InvalidConfig(
                "evaluation confidence_threshold must be finite and non-negative".to_string(),
            ));
        }
        if !(self.iou_threshold.is_finite() && self.iou_threshold >= 0.0) {
            return Err(crate::Error::InvalidConfig(
                "evaluation iou_threshold must be finite and non-negative".to_string(),
            ));
        }
        self.loss_config.validate()?;
        if matches!(self.class_filter.as_ref(), Some(filter) if !filter.is_enabled()) {
            return Err(crate::Error::InvalidConfig(
                "evaluation class_filter must be enabled when set".to_string(),
            ));
        }
        Ok(())
    }
}

/// Dataset evaluation loop summary.
#[derive(Debug, Clone, PartialEq)]

pub struct EvalLoopReport {
    /// Total evaluated batches.
    pub total_steps: usize,
    /// Total evaluated samples.
    pub total_samples: usize,
    /// Mean supervised loss across batches.
    pub mean_loss: f32,
    /// Last batch loss.
    pub last_loss: f32,
    /// Mean component-level losses across evaluated batches.
    pub mean_components: LossComponents,
    /// Last batch component-level losses.
    pub last_components: LossComponents,
    /// Classification metrics when evaluating classification outputs.
    pub classification: Option<ClassificationEvalMetrics>,
    /// Detection-style metrics when evaluating dense detection outputs.
    pub detection: Option<DetectionEvalMetrics>,
    /// Official mAP@0.5 and mAP@0.5:0.95 when evaluating detection-style outputs.
    pub map: Option<MapReport>,
    /// Official mask mAP when evaluating instance-segmentation outputs.
    pub mask_map: Option<MapReport>,
    /// Official pose OKS mAP when evaluating pose/keypoint outputs.
    pub pose_map: Option<MapReport>,
    /// Semantic mIoU when evaluating semantic segmentation outputs (EVAL-05).
    pub semantic: Option<SemanticReport>,
}
