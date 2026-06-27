//! Official-aligned losses for the trainable YOLOE segmentation network.
//!
//! The shared box+cls+dfl+mask supervision reuses the existing
//! [`segmentation_loss_report`](crate::train::segmentation_loss_report) by adapting
//! the YOLOE head output into a [`DenseDetectionOutput`]. The contrastive class
//! BCE IS the YOLOE text/visual prompt alignment loss (the BNContrastive scores
//! are differentiated directly). Prompt-free mode feeds dense LRPC boxes/scores
//! into the same loss rather than adding a separate objectness term.

use candle_core::Tensor;

use crate::train::{DenseDetectionOutput, SegmentationTargets, segmentation_loss_report};

use super::output::Output;

/// Report returned by [`segmentation_loss`].
#[derive(Debug, Clone)]
pub struct LossReport {
    /// Total weighted loss to backpropagate.
    pub loss: Tensor,
    /// Shared detection (box/cls/dfl) + mask loss.
    pub base_loss: Tensor,
}

/// Computes the YOLOE segmentation loss for the active prompt mode.
///
/// Text/visual prompt modes pass BNContrastive logits in `output.scores`; the
/// prompt-free path passes LRPC vocabulary logits in the same field.
pub fn segmentation_loss(
    output: &Output,
    targets: &SegmentationTargets,
    config: &crate::train::DetectionLossConfig,
) -> crate::Result<LossReport> {
    let dense = DenseDetectionOutput {
        boxes: output.boxes.clone(),
        scores: output.scores.clone(),
        anchors: output.anchors.clone(),
        stride_tensor: output.stride_tensor.clone(),
    };
    let base =
        segmentation_loss_report(&dense, &output.masks, &output.proto, None, targets, *config)?;
    Ok(LossReport {
        loss: base.loss.clone(),
        base_loss: base.loss,
    })
}
