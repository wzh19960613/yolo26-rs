use super::*;

pub(crate) fn build_detection_targets(
    output: &DenseDetectionOutput,
    targets: &DetectionTargets,
    config: DetectionLossConfig,
) -> crate::Result<BuiltDetectionTargets> {
    let (batch, _, anchors_len) = output.boxes.dims3()?;
    let (_, classes, score_anchors) = output.scores.dims3()?;
    if score_anchors != anchors_len {
        return Err(crate::Error::InvalidTensor(
            "detection score anchors do not match box anchors".to_string(),
        ));
    }
    if targets.boxes_xyxy.dim(0)? != batch {
        return Err(crate::Error::InvalidTensor(
            "detection target batch does not match model output".to_string(),
        ));
    }

    let max_objects = targets.boxes_xyxy.dim(1)?;
    let anchors = output
        .anchors
        .squeeze(0)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let strides = output
        .stride_tensor
        .squeeze(0)?
        .squeeze(0)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let boxes = targets
        .boxes_xyxy
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let class_ids = targets
        .class_ids
        .contiguous()?
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let valid = targets
        .valid
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let pred_scores = candle_nn::ops::sigmoid(&output.scores)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let pred_xyxy = decode_detection_boxes(output)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let (image_width, image_height) = infer_image_size_from_anchors(&anchors, &strides);

    let mut pending = Vec::new();

    for b in 0..batch {
        for obj in 0..max_objects {
            if valid[b][obj] <= 0.0 {
                continue;
            }
            let xyxy = &boxes[b][obj];
            let class_id = class_ids[b][obj] as usize;
            if class_id >= classes || xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1] {
                continue;
            }
            let candidates = task_aligned_candidates(
                b,
                class_id,
                xyxy,
                &anchors,
                &strides,
                &pred_scores,
                &pred_xyxy,
                config,
            );
            let max_metric = candidates
                .iter()
                .map(|candidate| candidate.metric)
                .fold(0.0f32, f32::max);
            let max_overlap = candidates
                .iter()
                .map(|candidate| candidate.overlap)
                .fold(0.0f32, f32::max);
            for candidate in candidates {
                pending.push(PendingDetectionAssignment {
                    batch_idx: b,
                    object_idx: obj,
                    class_id,
                    anchor_idx: candidate.anchor_idx,
                    metric: candidate.metric,
                    overlap: candidate.overlap,
                    max_metric,
                    max_overlap,
                });
            }
        }
    }

    // Build the geometry context for the official `select_candidates_in_gts`
    // containment check: anchor centers in pixel space, and GT boxes in
    // letterboxed-pixel space.
    let mut gt_boxes = Vec::with_capacity(batch * max_objects);
    for image_boxes in boxes.iter().take(batch) {
        for xyxy in image_boxes.iter().take(max_objects) {
            gt_boxes.push([xyxy[0], xyxy[1], xyxy[2], xyxy[3]]);
        }
    }
    let anchor_centers_pixel: Vec<(f32, f32)> = (0..anchors_len)
        .map(|a| {
            let stride = strides[a].max(f32::EPSILON);
            (anchors[a][0] * stride, anchors[a][1] * stride)
        })
        .collect();
    let geometry = crate::train::eval::detection_assignment::Geometry {
        gt_boxes,
        anchor_centers_pixel,
        max_objects,
    };
    let assign_result = crate::train::eval::detection_assignment::resolve_detection_assignments(
        &pending,
        batch,
        anchors_len,
        max_objects,
        config,
        &geometry,
    );
    let assignments = assign_result.assignments;
    let pos_stats = assign_result.stats;

    crate::train::loss::detection_buffers::build_target_buffers(
        output,
        &assignments,
        &pos_stats,
        &boxes,
        &anchors,
        &strides,
        batch,
        classes,
        anchors_len,
        max_objects,
        image_width,
        image_height,
    )
}

fn infer_image_size_from_anchors(anchors: &[Vec<f32>], strides: &[f32]) -> (f32, f32) {
    let mut width = 1.0f32;
    let mut height = 1.0f32;
    for (anchor, stride) in anchors.iter().zip(strides) {
        if anchor.len() < 2 || !stride.is_finite() || *stride <= 0.0 {
            continue;
        }
        width = width.max((anchor[0] + 0.5) * *stride);
        height = height.max((anchor[1] + 0.5) * *stride);
    }
    (width, height)
}
