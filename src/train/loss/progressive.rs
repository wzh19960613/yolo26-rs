use super::*;

/// Official YOLO26 progressive one-to-many / one-to-one loss weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressiveLossSchedule {
    /// Total loss weight budget split between the two heads.
    pub total: f64,
    /// Initial one-to-many loss weight before epoch-end updates.
    pub initial_one_to_many: f64,
    /// Final one-to-many loss weight after the decay completes.
    pub final_one_to_many: f64,
}

/// Concrete progressive loss weights for one training epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressiveLossWeights {
    /// Weight applied to the one-to-many detection loss.
    pub one_to_many: f64,
    /// Weight applied to the one-to-one detection loss.
    pub one_to_one: f64,
}

impl ProgressiveLossSchedule {
    /// Creates the default YOLO26 end-to-end progressive loss schedule.
    pub const fn yolo26() -> Self {
        Self {
            total: 1.0,
            initial_one_to_many: 0.8,
            final_one_to_many: 0.1,
        }
    }

    /// Validates that all weights are finite and form a usable schedule.
    pub fn validate(self) -> crate::Result<()> {
        if !self.total.is_finite()
            || !self.initial_one_to_many.is_finite()
            || !self.final_one_to_many.is_finite()
            || self.total <= 0.0
            || self.initial_one_to_many < 0.0
            || self.final_one_to_many < 0.0
            || self.initial_one_to_many > self.total
            || self.final_one_to_many > self.total
        {
            return Err(crate::Error::InvalidConfig(
                "progressive loss weights must be finite and within the total weight".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns weights after `completed_epochs` epoch-end updates.
    pub fn weights_after_epochs(
        self,
        completed_epochs: usize,
        epochs: usize,
    ) -> crate::Result<ProgressiveLossWeights> {
        self.validate()?;
        let denom = epochs.saturating_sub(1).max(1) as f64;
        let progress = (completed_epochs as f64 / denom).clamp(0.0, 1.0);
        let one_to_many = (1.0 - progress) * (self.initial_one_to_many - self.final_one_to_many)
            + self.final_one_to_many;
        Ok(ProgressiveLossWeights {
            one_to_many,
            one_to_one: (self.total - one_to_many).max(0.0),
        })
    }
}

impl Default for ProgressiveLossSchedule {
    fn default() -> Self {
        Self::yolo26()
    }
}

pub(crate) fn progressive_detection_loss_report(
    one_to_many: &DenseDetectionOutput,
    one_to_one: &DenseDetectionOutput,
    targets: &DetectionTargets,
    weights: ProgressiveLossWeights,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let one_to_many_report = detection_loss_report(
        one_to_many,
        targets,
        config.with_yolo26_one_to_many_assignment(),
    )?;
    let one_to_one_report = detection_loss_report(
        one_to_one,
        targets,
        config.with_yolo26_one_to_one_assignment(),
    )?;
    let weighted_many = (one_to_many_report.loss * weights.one_to_many)?;
    let weighted_one = (one_to_one_report.loss * weights.one_to_one)?;
    Ok(LossTensorReport {
        loss: (weighted_many + weighted_one)?,
        // The official `E2ELoss.__call__` returns `loss_one2one[1]` as the
        // detached component vector that gets logged as `train/{box,cls,dfl}_loss`.
        // Report the one-to-one components so the logged metrics match
        // Ultralytics `results.csv` rather than the blended training target.
        components: one_to_one_report.components,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "progressive segmentation loss blends one-to-many and one-to-one head outputs"
)]
pub(crate) fn progressive_segmentation_loss_report(
    one_to_many_detect: &DenseDetectionOutput,
    one_to_many_masks: &Tensor,
    one_to_one_detect: &DenseDetectionOutput,
    one_to_one_masks: &Tensor,
    proto: &Tensor,
    semantic_logits: Option<&Tensor>,
    targets: &SegmentationTargets,
    weights: ProgressiveLossWeights,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let one_to_many_report = segmentation_loss_report(
        one_to_many_detect,
        one_to_many_masks,
        proto,
        semantic_logits,
        targets,
        config.with_yolo26_one_to_many_assignment(),
    )?;
    let detached_proto = proto.detach();
    let detached_semantic = semantic_logits.map(Tensor::detach);
    let one_to_one_report = segmentation_loss_report(
        one_to_one_detect,
        one_to_one_masks,
        &detached_proto,
        detached_semantic.as_ref(),
        targets,
        config.with_yolo26_one_to_one_assignment(),
    )?;
    let weighted_many = (one_to_many_report.loss * weights.one_to_many)?;
    let weighted_one = (one_to_one_report.loss * weights.one_to_one)?;
    Ok(LossTensorReport {
        loss: (weighted_many + weighted_one)?,
        components: one_to_one_report.components,
    })
}

pub(crate) fn progressive_pose_loss_report(
    one_to_many_detect: &DenseDetectionOutput,
    one_to_many_keypoints: &Tensor,
    one_to_one_detect: &DenseDetectionOutput,
    one_to_one_keypoints: &Tensor,
    targets: &PoseTargets,
    weights: ProgressiveLossWeights,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let one_to_many_report = pose_loss_report(
        one_to_many_detect,
        one_to_many_keypoints,
        targets,
        config.with_yolo26_one_to_many_assignment(),
    )?;
    let one_to_one_report = pose_loss_report(
        one_to_one_detect,
        one_to_one_keypoints,
        targets,
        config.with_yolo26_one_to_one_assignment(),
    )?;
    weighted_progressive_report(one_to_many_report, one_to_one_report, weights)
}

pub(crate) fn progressive_obb_loss_report(
    one_to_many_detect: &DenseDetectionOutput,
    one_to_many_angles: &Tensor,
    one_to_one_detect: &DenseDetectionOutput,
    one_to_one_angles: &Tensor,
    targets: &ObbTargets,
    weights: ProgressiveLossWeights,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let one_to_many_report = obb_loss_report(
        one_to_many_detect,
        one_to_many_angles,
        targets,
        config.with_yolo26_one_to_many_assignment(),
    )?;
    let one_to_one_report = obb_loss_report(
        one_to_one_detect,
        one_to_one_angles,
        targets,
        config.with_yolo26_one_to_one_assignment(),
    )?;
    weighted_progressive_report(one_to_many_report, one_to_one_report, weights)
}

fn weighted_progressive_report(
    one_to_many_report: LossTensorReport,
    one_to_one_report: LossTensorReport,
    weights: ProgressiveLossWeights,
) -> crate::Result<LossTensorReport> {
    let weighted_many = (one_to_many_report.loss * weights.one_to_many)?;
    let weighted_one = (one_to_one_report.loss * weights.one_to_one)?;
    Ok(LossTensorReport {
        loss: (weighted_many + weighted_one)?,
        components: blend_tensor_components(
            one_to_many_report.components,
            one_to_one_report.components,
            weights.one_to_many,
            weights.one_to_one,
        )?,
    })
}
