use super::*;

/// Computes a supervised loss for supported task outputs.
pub fn supervised_loss(output: &Output, target: &Target) -> crate::Result<Tensor> {
    Ok(supervised_loss_report(output, target)?.loss)
}

/// Computes a supervised loss for supported task outputs, returning both the
/// total loss tensor and the per-component (box/cls/dfl/...) weighted tensors.
pub fn supervised_loss_report(output: &Output, target: &Target) -> crate::Result<LossTensorReport> {
    supervised_loss_report_with_config(output, target, DetectionLossConfig::default())
}

pub(crate) fn supervised_loss_report_with_config(
    output: &Output,
    target: &Target,
    loss_config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    supervised_loss_report_with_progressive_weights(
        output,
        target,
        ProgressiveLossSchedule::yolo26().weights_after_epochs(0, 1)?,
        loss_config,
    )
}

pub(crate) fn supervised_loss_report_with_progressive_weights(
    output: &Output,
    target: &Target,
    progressive_weights: ProgressiveLossWeights,
    loss_config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    match (output, target) {
        (Output::Classify { logits }, Target::Classification { class_ids }) => {
            classification_loss_report(logits, class_ids)
        }
        (Output::Detect(out), Target::Detection(targets)) => {
            detection_loss_report(out, targets, loss_config)
        }
        (
            Output::DetectE2e {
                one_to_many,
                one_to_one,
            },
            Target::Detection(targets),
        ) => progressive_detection_loss_report(
            one_to_many,
            one_to_one,
            targets,
            progressive_weights,
            loss_config,
        ),
        (
            Output::Segment {
                detect,
                masks,
                proto,
                semantic,
            },
            Target::Segmentation(targets),
        ) => segmentation_loss_report(
            detect,
            masks,
            proto,
            semantic.as_ref(),
            targets,
            loss_config,
        ),
        (
            Output::SegmentE2e {
                one_to_many_detect,
                one_to_many_masks,
                one_to_one_detect,
                one_to_one_masks,
                proto,
                semantic,
            },
            Target::Segmentation(targets),
        ) => progressive_segmentation_loss_report(
            one_to_many_detect,
            one_to_many_masks,
            one_to_one_detect,
            one_to_one_masks,
            proto,
            semantic.as_ref(),
            targets,
            progressive_weights,
            loss_config,
        ),
        (Output::Pose { detect, keypoints }, Target::Pose(targets)) => {
            pose_loss_report(detect, keypoints, targets, loss_config)
        }
        (
            Output::PoseE2e {
                one_to_many_detect,
                one_to_many_keypoints,
                one_to_one_detect,
                one_to_one_keypoints,
            },
            Target::Pose(targets),
        ) => progressive_pose_loss_report(
            one_to_many_detect,
            one_to_many_keypoints,
            one_to_one_detect,
            one_to_one_keypoints,
            targets,
            progressive_weights,
            loss_config,
        ),
        (Output::Obb { detect, angles }, Target::Obb(targets)) => {
            obb_loss_report(detect, angles, targets, loss_config)
        }
        (
            Output::ObbE2e {
                one_to_many_detect,
                one_to_many_angles,
                one_to_one_detect,
                one_to_one_angles,
            },
            Target::Obb(targets),
        ) => progressive_obb_loss_report(
            one_to_many_detect,
            one_to_many_angles,
            one_to_one_detect,
            one_to_one_angles,
            targets,
            progressive_weights,
            loss_config,
        ),
        (Output::Segment { detect, .. }, Target::Detection(targets))
        | (Output::Pose { detect, .. }, Target::Detection(targets))
        | (Output::Obb { detect, .. }, Target::Detection(targets)) => {
            detection_loss_report(detect, targets, loss_config)
        }
        (
            Output::SegmentE2e {
                one_to_many_detect,
                one_to_one_detect,
                ..
            }
            | Output::PoseE2e {
                one_to_many_detect,
                one_to_one_detect,
                ..
            }
            | Output::ObbE2e {
                one_to_many_detect,
                one_to_one_detect,
                ..
            },
            Target::Detection(targets),
        ) => progressive_detection_loss_report(
            one_to_many_detect,
            one_to_one_detect,
            targets,
            progressive_weights,
            loss_config,
        ),
        (Output::Semantic { logits }, Target::Semantic { class_map }) => {
            semantic_loss_report(logits, class_map)
        }
        _ => Err(crate::Error::Unsupported(
            "task-specific supervised losses are not implemented yet".to_string(),
        )),
    }
}

fn classification_loss_report(
    logits: &Tensor,
    class_ids: &Tensor,
) -> crate::Result<LossTensorReport> {
    let loss = candle_nn::loss::cross_entropy(logits, class_ids)?;
    Ok(LossTensorReport {
        loss: loss.clone(),
        components: LossTensorComponents {
            classification_loss: Some(loss),
            ..Default::default()
        },
    })
}

fn semantic_loss_report(logits: &Tensor, class_map: &Tensor) -> crate::Result<LossTensorReport> {
    let loss = semantic_loss(logits, class_map)?;
    Ok(LossTensorReport {
        loss: loss.clone(),
        components: LossTensorComponents {
            semantic_loss: Some(loss),
            ..Default::default()
        },
    })
}
