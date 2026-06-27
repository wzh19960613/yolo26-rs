use candle_core::Tensor;

use super::output::Output as YoloeOutput;
use crate::yoloe::segment::model::train_config::PromptMode;
use crate::yoloe::usage::EmbeddingTable;

use super::{
    DenseDetectionOutput, DetectionLossConfig, DetectionMapAccumulator, EvalLoopConfig, EvalReport,
    LossComponents, MaskMapAccumulator, Output, SegmentationTargets, Session, Target,
    detection_eval_metrics, scalar_loss_value, update_segmentation_mask_map,
};

#[derive(Clone, Copy)]
pub(crate) struct BatchEvalConfig {
    pub(crate) loss_config: DetectionLossConfig,
    max_detections: usize,
    confidence_threshold: f32,
    iou_threshold: f32,
    single_class: bool,
}

impl BatchEvalConfig {
    pub(crate) fn from_loss_config(loss_config: DetectionLossConfig) -> Self {
        Self {
            loss_config,
            max_detections: 300,
            confidence_threshold: 0.001,
            iou_threshold: 0.7,
            single_class: false,
        }
    }

    pub(crate) fn from_eval_loop(config: &EvalLoopConfig) -> Self {
        Self {
            loss_config: config.loss_config,
            max_detections: config.max_detections,
            confidence_threshold: config.confidence_threshold,
            iou_threshold: config.iou_threshold,
            single_class: config
                .class_filter
                .as_ref()
                .map(|filter| filter.single_class)
                .unwrap_or(false),
        }
    }
}

pub(crate) fn ensure_yoloe_eval_mode(
    session: &Session,
    expected: PromptMode,
    label: &str,
) -> crate::Result<()> {
    let actual = session.model().config().mode;
    if actual == expected {
        return Ok(());
    }
    Err(crate::Error::InvalidConfig(format!(
        "YOLOE {label} evaluation requires mode {expected:?}, got {actual:?}"
    )))
}

pub(crate) fn yoloe_eval_with_prompt_table(
    session: &Session,
    input: &Tensor,
    target: &SegmentationTargets,
    prompts: &EmbeddingTable,
    config: BatchEvalConfig,
    map: Option<&mut DetectionMapAccumulator>,
    mask_map: Option<&mut MaskMapAccumulator>,
) -> crate::Result<EvalReport> {
    let output = session.model().forward_dense(input, prompts)?;
    let report = super::segmentation_loss(&output, target, &config.loss_config)?;
    yoloe_eval_report_from_loss(input, &report.loss, &output, target, config, map, mask_map)
}

pub(crate) fn yoloe_prompt_free_eval_report(
    session: &Session,
    input: &Tensor,
    target: &SegmentationTargets,
    config: BatchEvalConfig,
    map: Option<&mut DetectionMapAccumulator>,
    mask_map: Option<&mut MaskMapAccumulator>,
) -> crate::Result<EvalReport> {
    let output = session.model().forward_prompt_free_dense(input)?;
    let report = super::segmentation_loss(&output, target, &config.loss_config)?;
    yoloe_eval_report_from_loss(input, &report.loss, &output, target, config, map, mask_map)
}

fn yoloe_eval_report_from_loss(
    input: &Tensor,
    loss: &Tensor,
    output: &YoloeOutput,
    target: &SegmentationTargets,
    config: BatchEvalConfig,
    map: Option<&mut DetectionMapAccumulator>,
    mask_map: Option<&mut MaskMapAccumulator>,
) -> crate::Result<EvalReport> {
    let total = scalar_loss_value(loss)?;
    let output = Output::Segment {
        detect: DenseDetectionOutput {
            boxes: output.boxes.clone(),
            scores: output.scores.clone(),
            anchors: output.anchors.clone(),
            stride_tensor: output.stride_tensor.clone(),
        },
        masks: output.masks.clone(),
        proto: output.proto.clone(),
        semantic: None,
    };
    let target = Target::Segmentation(target.clone());
    update_segmentation_mask_map(
        &output,
        None,
        &target,
        (input.dim(2)?, input.dim(3)?),
        config.confidence_threshold,
        config.iou_threshold,
        config.max_detections,
        config.single_class,
        mask_map.map(|acc| &mut acc.inner),
    )?;
    let detection = detection_eval_metrics(
        &output,
        None,
        &target,
        config.confidence_threshold,
        config.iou_threshold,
        config.max_detections,
        config.single_class,
        map.map(|acc| &mut acc.inner),
    )?;
    Ok(EvalReport {
        loss: total,
        components: LossComponents {
            total,
            ..Default::default()
        },
        samples: input.dim(0)?,
        classification: None,
        detection,
    })
}
