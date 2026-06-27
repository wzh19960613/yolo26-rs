use candle_core::DType;

use super::{
    DenseDetectionOutput, EvalPostprocessOutput, MapAccumulator, Output, SegmentationTargets,
    Target, batch_image_predictions_from_tensors, batch_image_predictions_from_topk_tensor,
    decode_detection_boxes, mask_tp_matrix, prediction_masks_from_tensors, target_masks_for_image,
};

#[expect(
    clippy::too_many_arguments,
    reason = "mask mAP update consumes decoded outputs, targets, thresholds, and accumulator state"
)]
pub(crate) fn update_segmentation_mask_map(
    output: &Output,
    postprocess: Option<&EvalPostprocessOutput>,
    target: &Target,
    input_hw: (usize, usize),
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    single_class: bool,
    map: Option<&mut MapAccumulator>,
) -> crate::Result<bool> {
    let Some((detect, mask_coefficients, proto)) = segment_output(output) else {
        return Ok(false);
    };
    let Some(targets) = target_segmentation(target) else {
        return Ok(false);
    };
    let Some(acc) = map else {
        return Ok(true);
    };
    let (mut batch_preds, mask_coefficients, proto) = match postprocess {
        Some(EvalPostprocessOutput::Segment { predictions, proto }) => {
            let (_, proto_channels, _, _) = proto.dims4()?;
            let cols = predictions.dim(2)?;
            if cols < 6 + proto_channels {
                return Err(crate::Error::InvalidTensor(format!(
                    "segmentation top-k predictions must have at least {} columns, got {:?}",
                    6 + proto_channels,
                    predictions.dims()
                )));
            }
            (
                batch_image_predictions_from_topk_tensor(
                    predictions,
                    confidence_threshold,
                    iou_threshold,
                    max_detections,
                )?,
                predictions.narrow(2, 6, proto_channels)?.contiguous()?,
                proto,
            )
        }
        _ => {
            let pred_xyxy = decode_detection_boxes(detect)?.to_dtype(DType::F32)?;
            let pred_scores = candle_nn::ops::sigmoid(&detect.scores)?.to_dtype(DType::F32)?;
            (
                batch_image_predictions_from_tensors(
                    &pred_xyxy,
                    &pred_scores,
                    confidence_threshold,
                    iou_threshold,
                    max_detections,
                )?,
                mask_coefficients.clone(),
                proto,
            )
        }
    };
    let (_, proto_channels, proto_h, proto_w) = proto.dims4()?;
    let target_boxes = targets
        .detection
        .boxes_xyxy
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let target_classes = targets
        .detection
        .class_ids
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let target_valid = targets
        .detection
        .valid
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let target_masks = tensor_to_vec4(&targets.masks.to_dtype(DType::F32)?)?;
    for b in 0..batch_preds.len() {
        let preds = &mut batch_preds[b];
        if single_class {
            for pred in preds.iter_mut() {
                pred.class_id = 0;
            }
        }
        let pred_masks =
            prediction_masks_from_tensors(preds, &mask_coefficients, proto, b, input_hw)?;
        let gt_masks = target_masks_for_image(
            &target_masks[b],
            targets.mask_encoding,
            target_boxes[b].len(),
            (proto_channels, proto_h, proto_w),
        )?;
        let tp = mask_tp_matrix(
            preds,
            &pred_masks,
            &target_boxes[b],
            &target_classes[b],
            &target_valid[b],
            &gt_masks,
        );
        acc.add_image_with_tp(
            preds,
            &target_boxes[b],
            &target_classes[b],
            &target_valid[b],
            &tp,
        );
    }
    Ok(true)
}

fn segment_output(
    output: &Output,
) -> Option<(
    &DenseDetectionOutput,
    &candle_core::Tensor,
    &candle_core::Tensor,
)> {
    match output {
        Output::Segment {
            detect,
            masks,
            proto,
            ..
        } => Some((detect, masks, proto)),
        Output::SegmentE2e {
            one_to_one_detect,
            one_to_one_masks,
            proto,
            ..
        } => Some((one_to_one_detect, one_to_one_masks, proto)),
        _ => None,
    }
}

fn target_segmentation(target: &Target) -> Option<&SegmentationTargets> {
    match target {
        Target::Segmentation(targets) => Some(targets),
        _ => None,
    }
}

fn tensor_to_vec4(tensor: &candle_core::Tensor) -> crate::Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(crate::Error::InvalidTensor(format!(
            "expected 4D tensor for segmentation mask eval, got {dims:?}"
        )));
    }
    let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(batch);
    for _ in 0..batch {
        let mut batch_out = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut channel_out = Vec::with_capacity(height);
            for _ in 0..height {
                channel_out.push(data[offset..offset + width].to_vec());
                offset += width;
            }
            batch_out.push(channel_out);
        }
        out.push(batch_out);
    }
    Ok(out)
}
