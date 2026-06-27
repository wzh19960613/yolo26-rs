use super::*;

/// Computes semantic segmentation cross entropy over `[B, C, H, W]` logits.
pub fn semantic_loss(logits: &Tensor, class_map: &Tensor) -> crate::Result<Tensor> {
    let (batch, classes, height, width) = logits.dims4()?;
    if class_map.dims() != [batch, height, width] {
        return Err(crate::Error::InvalidTensor(format!(
            "semantic class_map must have shape [{batch}, {height}, {width}], got {:?}",
            class_map.dims()
        )));
    }
    let flattened_logits = logits
        .permute((0, 2, 3, 1))?
        .reshape((batch * height * width, classes))?;
    let flattened_targets = class_map
        .to_dtype(DType::U32)?
        .reshape((batch * height * width,))?;
    semantic_loss_with_ignore(&flattened_logits, &flattened_targets, classes)
}

/// Computes the current supervised detection loss.
///
/// This is a trainable baseline loss: targets are assigned to the nearest anchor
/// center, class logits use BCE-with-logits, and boxes regress l/t/r/b
/// distances. It is intentionally isolated so it can be replaced with the
/// Ultralytics task-aligned assigner, CIoU/DFL, and task-specific heads.
pub fn detection_loss(
    output: &DenseDetectionOutput,
    targets: &DetectionTargets,
    config: DetectionLossConfig,
) -> crate::Result<Tensor> {
    Ok(detection_loss_report(output, targets, config)?.loss)
}

pub(crate) fn detection_loss_report(
    output: &DenseDetectionOutput,
    targets: &DetectionTargets,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    config.validate()?;
    let built = build_detection_targets(output, targets, config)?;
    let mut report = detection_loss_from_built_report(output, &built, config)?;
    report.loss = scale_loss_by_batch(report.loss, output.boxes.dim(0)?)?;
    Ok(report)
}

fn detection_loss_from_built_report(
    output: &DenseDetectionOutput,
    built: &BuiltDetectionTargets,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let bce_sum_t = bce_with_logits_sum(&output.scores, &built.target_scores)?;
    let class_loss = bce_sum_t.broadcast_div(&Tensor::new(
        built.target_scores_sum as f32,
        output.scores.device(),
    )?)?;

    // Official `BboxLoss.forward` (reg_max == 1 → use_dfl = False) computes
    // both `loss_iou` and `loss_dfl` per foreground anchor, weighted by the
    // target-score sum and normalized by `target_scores_sum`. We replicate that
    // exactly: box loss is CIoU-based, dfl is the imgsz-normalized L1 distance
    // between predicted and target l/t/r/b in stride units.
    let pred_xyxy = decode_detection_boxes(output)?;
    let weight = built.target_scores.sum(1)?.unsqueeze(1)?;
    let weighted_fg = built.foreground_mask.broadcast_mul(&weight)?;

    let iou_loss = crate::train::loss::ciou_dfl::ciou_loss(
        &pred_xyxy,
        &built.target_xyxy,
        &weighted_fg,
        built.target_scores_sum,
    )?;
    let distance_loss = crate::train::loss::ciou_dfl::normalized_l1_dfl_loss(
        &output.boxes,
        &built.target_ltrb,
        &output.stride_tensor,
        built.image_width,
        built.image_height,
        &weighted_fg,
        built.target_scores_sum,
    )?;

    let weighted_box = (iou_loss * config.box_weight)?;
    let weighted_class = (class_loss * config.class_weight)?;
    let weighted_distance = (distance_loss * config.distance_weight)?;
    let loss = ((weighted_box.clone() + weighted_class.clone())? + weighted_distance.clone())?;
    Ok(LossTensorReport {
        loss,
        components: LossTensorComponents {
            box_loss: Some(weighted_box),
            cls_loss: Some(weighted_class),
            dfl_loss: Some(weighted_distance),
            ..Default::default()
        },
    })
}

/// Computes instance segmentation loss as detection loss plus foreground mask BCE.
pub fn segmentation_loss(
    detect: &DenseDetectionOutput,
    mask_coefficients: &Tensor,
    proto: &Tensor,
    semantic_logits: Option<&Tensor>,
    targets: &SegmentationTargets,
    config: DetectionLossConfig,
) -> crate::Result<Tensor> {
    Ok(segmentation_loss_report(
        detect,
        mask_coefficients,
        proto,
        semantic_logits,
        targets,
        config,
    )?
    .loss)
}

pub(crate) fn segmentation_loss_report(
    detect: &DenseDetectionOutput,
    mask_coefficients: &Tensor,
    proto: &Tensor,
    semantic_logits: Option<&Tensor>,
    targets: &SegmentationTargets,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let built = build_detection_targets(detect, &targets.detection, config)?;
    let det_report = detection_loss_from_built_report(detect, &built, config)?;
    let mask_loss =
        (instance_mask_loss(mask_coefficients, proto, targets, &built)? * config.box_weight)?;
    let semantic_loss = semantic_logits
        .map(|logits| segmentation_semantic_loss(logits, targets))
        .transpose()?
        .map(|loss| loss * config.box_weight)
        .transpose()?;
    let loss = match &semantic_loss {
        Some(semantic_loss) => ((det_report.loss + mask_loss.clone())? + semantic_loss.clone())?,
        None => (det_report.loss + mask_loss.clone())?,
    };
    let loss = scale_loss_by_batch(loss, detect.boxes.dim(0)?)?;
    let mut components = det_report.components;
    components.mask_loss = Some(mask_loss);
    components.semantic_loss = semantic_loss;
    Ok(LossTensorReport { loss, components })
}

fn segmentation_semantic_loss(
    logits: &Tensor,
    targets: &SegmentationTargets,
) -> crate::Result<Tensor> {
    let (batch, classes, height, width) = logits.dims4()?;
    let semantic_dims = targets.semantic_masks.dims();
    if semantic_dims.len() != 3 || semantic_dims[0] != batch {
        return Err(crate::Error::InvalidTensor(format!(
            "segmentation semantic masks must have shape [{batch}, H, W], got {semantic_dims:?}"
        )));
    }
    let source_h = semantic_dims[1];
    let source_w = semantic_dims[2];
    let mask_channels = targets.masks.dim(1)?;
    let class_map = targets
        .semantic_masks
        .to_dtype(DType::U32)?
        .to_vec3::<u32>()?;
    let mask_data = targets
        .masks
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mask_layout = SemanticMaskLayout {
        channels: mask_channels,
        height: source_h,
        width: source_w,
    };
    let mut target = vec![0f32; batch * classes * height * width];

    for b in 0..batch {
        for y in 0..height {
            let sy = nearest_source_index(y, height, source_h);
            for x in 0..width {
                let sx = nearest_source_index(x, width, source_w);
                if !semantic_foreground(targets.mask_encoding, &mask_data, b, sy, sx, mask_layout) {
                    continue;
                }
                let class_id = class_map[b][sy][sx] as usize;
                if class_id >= classes {
                    return Err(crate::Error::InvalidTensor(format!(
                        "segmentation semantic class id {class_id} is outside logits class count {classes}"
                    )));
                }
                target[((b * classes + class_id) * height + y) * width + x] = 1.0;
            }
        }
    }

    let target = Tensor::from_vec(target, (batch, classes, height, width), logits.device())?
        .to_dtype(logits.dtype())?;
    let bce = bce_with_logits_elementwise(logits, &target)?.mean_all()?;
    let pred = candle_nn::ops::sigmoid(logits)?;
    let pixels = height * width;
    let pred_flat = pred.reshape((batch, classes, pixels))?;
    let target_flat = target.reshape((batch, classes, pixels))?;
    let intersection = pred_flat.broadcast_mul(&target_flat)?.sum(2)?;
    let pred_sum = pred_flat.sum(2)?;
    let target_sum = target_flat.sum(2)?;
    let smooth = Tensor::new(1f32, logits.device())?.to_dtype(logits.dtype())?;
    let two = Tensor::new(2f32, logits.device())?.to_dtype(logits.dtype())?;
    let numerator = intersection.broadcast_mul(&two)?.broadcast_add(&smooth)?;
    let denominator = pred_sum
        .broadcast_add(&target_sum)?
        .broadcast_add(&smooth)?;
    let dice = numerator.broadcast_div(&denominator)?;
    let dice_loss = dice.ones_like()?.broadcast_sub(&dice)?.mean_all()?;
    ((bce * 0.5)? + (dice_loss * 0.5)?).map_err(Into::into)
}

fn nearest_source_index(index: usize, size: usize, source_size: usize) -> usize {
    if source_size <= 1 || size <= 1 {
        return 0;
    }
    // PyTorch `F.interpolate(..., mode="nearest")` uses asymmetric nearest
    // mapping for integer downsampling: output index `i` samples
    // `floor(i * in / out)`, not center-aligned `(i + 0.5)`.
    (index * source_size / size).min(source_size - 1)
}

fn semantic_foreground(
    encoding: SegmentationMaskEncoding,
    masks: &[f32],
    batch_idx: usize,
    y: usize,
    x: usize,
    layout: SemanticMaskLayout,
) -> bool {
    match encoding {
        SegmentationMaskEncoding::Overlap => mask_value(masks, batch_idx, 0, y, x, layout) > 0.0,
        SegmentationMaskEncoding::PerInstance => (0..layout.channels)
            .any(|channel| mask_value(masks, batch_idx, channel, y, x, layout) > 0.0),
    }
}

#[derive(Clone, Copy)]
struct SemanticMaskLayout {
    channels: usize,
    height: usize,
    width: usize,
}

fn mask_value(
    data: &[f32],
    batch_idx: usize,
    channel_idx: usize,
    y: usize,
    x: usize,
    layout: SemanticMaskLayout,
) -> f32 {
    data[((batch_idx * layout.channels + channel_idx) * layout.height + y) * layout.width + x]
}

/// Computes pose loss as detection loss plus foreground keypoint losses.
pub fn pose_loss(
    detect: &DenseDetectionOutput,
    keypoints: &Tensor,
    targets: &PoseTargets,
    config: DetectionLossConfig,
) -> crate::Result<Tensor> {
    Ok(pose_loss_report(detect, keypoints, targets, config)?.loss)
}

pub(crate) fn pose_loss_report(
    detect: &DenseDetectionOutput,
    keypoints: &Tensor,
    targets: &PoseTargets,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let built = build_detection_targets(detect, &targets.detection, config)?;
    let det_report = detection_loss_from_built_report(detect, &built, config)?;
    let kpt_report = keypoint_loss_report(keypoints, detect, targets, &built, config)?;
    let loss = scale_loss_by_batch((det_report.loss + kpt_report.loss)?, detect.boxes.dim(0)?)?;
    let mut components = det_report.components;
    components.pose_loss = Some(kpt_report.pose_loss);
    components.kobj_loss = kpt_report.kobj_loss;
    Ok(LossTensorReport { loss, components })
}

/// Computes OBB loss as detection loss plus foreground periodic angle loss.
pub fn obb_loss(
    detect: &DenseDetectionOutput,
    angles: &Tensor,
    targets: &ObbTargets,
    config: DetectionLossConfig,
) -> crate::Result<Tensor> {
    Ok(obb_loss_report(detect, angles, targets, config)?.loss)
}

pub(crate) fn obb_loss_report(
    detect: &DenseDetectionOutput,
    angles: &Tensor,
    targets: &ObbTargets,
    config: DetectionLossConfig,
) -> crate::Result<LossTensorReport> {
    let built = build_detection_targets(detect, &targets.detection, config)?;
    let det_report = detection_loss_from_built_report(detect, &built, config)?;
    let angle_loss = (obb_angle_loss(angles, targets, &built)? * config.angle_weight)?;
    let loss = scale_loss_by_batch(
        (det_report.loss + angle_loss.clone())?,
        detect.boxes.dim(0)?,
    )?;
    let mut components = det_report.components;
    components.angle_loss = Some(angle_loss);
    Ok(LossTensorReport { loss, components })
}

fn scale_loss_by_batch(loss: Tensor, batch: usize) -> crate::Result<Tensor> {
    (loss * batch.max(1) as f64).map_err(Into::into)
}

pub(crate) struct BuiltDetectionTargets {
    pub(crate) target_ltrb: Tensor,
    pub(crate) target_xyxy: Tensor,
    pub(crate) target_scores: Tensor,
    pub(crate) foreground_mask: Tensor,
    pub(crate) foreground_count: f64,
    pub(crate) target_scores_sum: f64,
    pub(crate) target_gt_idx: Vec<usize>,
    pub(crate) image_width: f32,
    pub(crate) image_height: f32,
}
