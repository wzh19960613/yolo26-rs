use super::*;

pub(crate) fn target_detection(target: &Target) -> Option<&DetectionTargets> {
    match target {
        Target::Detection(targets) => Some(targets),
        Target::Segmentation(targets) => Some(&targets.detection),
        Target::Pose(targets) => Some(&targets.detection),
        Target::Obb(targets) => Some(&targets.detection),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]

pub(crate) struct EvalPrediction {
    pub(crate) class_id: u32,
    pub(crate) score: f32,
    pub(crate) xyxy: [f32; 4],
    pub(crate) anchor_idx: usize,
}

pub(crate) fn xyxy_iou_scalar(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let inter_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = (area_a + area_b - inter).max(1e-7);
    inter / union
}

#[derive(Default)]

pub struct LossTensorComponents {
    /// Classification cross-entropy (classification task).
    pub classification_loss: Option<Tensor>,
    /// Weighted box regression loss.
    pub box_loss: Option<Tensor>,
    /// Weighted class confidence loss.
    pub cls_loss: Option<Tensor>,
    /// Weighted distribution-focal / normalized-L1 box loss.
    pub dfl_loss: Option<Tensor>,
    /// Weighted instance mask loss.
    pub mask_loss: Option<Tensor>,
    /// Weighted pose keypoint loss.
    pub pose_loss: Option<Tensor>,
    /// Weighted keypoint visibility loss.
    pub kobj_loss: Option<Tensor>,
    /// Weighted OBB angle loss.
    pub angle_loss: Option<Tensor>,
    /// Weighted semantic segmentation loss.
    pub semantic_loss: Option<Tensor>,
    /// Smoke-test raw activation loss (debug only).
    pub smoke_loss: Option<Tensor>,
}

/// A supervised loss result with the scalar total and per-component tensors.
///
/// Exposed so external tooling (e.g. loss-vs-official comparison harnesses) can
/// read the individual box/cls/dfl tensors that produced a training loss.
pub struct LossTensorReport {
    /// Total weighted loss tensor.
    pub loss: Tensor,
    /// Per-component weighted loss tensors (one-to-one branch when the task uses
    /// the YOLO26 progressive E2E loss, matching the official `E2ELoss` logging).
    pub components: LossTensorComponents,
}

impl LossTensorReport {
    pub(crate) fn scalar_components(&self) -> crate::Result<LossComponents> {
        let classification_loss = scalar_optional_loss(&self.components.classification_loss)?;
        let box_loss = scalar_optional_loss(&self.components.box_loss)?;
        let cls_loss = scalar_optional_loss(&self.components.cls_loss)?;
        let dfl_loss = scalar_optional_loss(&self.components.dfl_loss)?;
        let mask_loss = scalar_optional_loss(&self.components.mask_loss)?;
        let pose_loss = scalar_optional_loss(&self.components.pose_loss)?;
        let kobj_loss = scalar_optional_loss(&self.components.kobj_loss)?;
        let angle_loss = scalar_optional_loss(&self.components.angle_loss)?;
        let semantic_loss = scalar_optional_loss(&self.components.semantic_loss)?;
        let smoke_loss = scalar_optional_loss(&self.components.smoke_loss)?;
        let mut component_total = 0f32;
        let mut component_count = 0usize;
        for value in [
            classification_loss,
            box_loss,
            cls_loss,
            dfl_loss,
            mask_loss,
            pose_loss,
            kobj_loss,
            angle_loss,
            semantic_loss,
            smoke_loss,
        ]
        .into_iter()
        .flatten()
        {
            component_total += value;
            component_count += 1;
        }
        Ok(LossComponents {
            total: if component_count > 0 {
                component_total
            } else {
                scalar_loss_value(&self.loss)?
            },
            classification_loss,
            box_loss,
            cls_loss,
            dfl_loss,
            mask_loss,
            pose_loss,
            kobj_loss,
            angle_loss,
            semantic_loss,
            smoke_loss,
        })
    }
}

/// Scalar loss components recorded for Ultralytics-style training logs.
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct LossComponents {
    /// Total scalar loss used for optimization or evaluation.
    pub total: f32,
    /// Classification cross-entropy for classification models.
    pub classification_loss: Option<f32>,
    /// Weighted box regression loss.
    pub box_loss: Option<f32>,
    /// Weighted class loss for detection-style heads.
    pub cls_loss: Option<f32>,
    /// Weighted distance/DFL-style box distribution loss.
    pub dfl_loss: Option<f32>,
    /// Instance mask loss for segmentation models.
    pub mask_loss: Option<f32>,
    /// Keypoint location loss for pose models.
    pub pose_loss: Option<f32>,
    /// Keypoint objectness/visibility loss for pose models.
    pub kobj_loss: Option<f32>,
    /// Periodic angle loss for OBB models.
    pub angle_loss: Option<f32>,
    /// Semantic segmentation loss.
    pub semantic_loss: Option<f32>,
    /// Smoke-test raw-output regularization loss.
    pub smoke_loss: Option<f32>,
}

impl LossComponents {
    fn from_total(total: f32) -> Self {
        Self {
            total,
            classification_loss: None,
            box_loss: None,
            cls_loss: None,
            dfl_loss: None,
            mask_loss: None,
            pose_loss: None,
            kobj_loss: None,
            angle_loss: None,
            semantic_loss: None,
            smoke_loss: None,
        }
    }
}

impl Default for LossComponents {
    fn default() -> Self {
        Self::from_total(0.0)
    }
}

#[derive(Debug, Clone, Copy, Default)]

pub(crate) struct OptionalLossAccumulator {
    pub(crate) sum: f64,
    pub(crate) count: usize,
}
