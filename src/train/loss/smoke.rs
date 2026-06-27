use super::*;

/// Computes a differentiable smoke loss over raw outputs.
///
/// This is useful for validating graph connectivity and optimizer state. It is
/// not a replacement for task-supervised YOLO losses.
pub fn smoke_loss(output: &Output) -> crate::Result<Tensor> {
    match output {
        Output::Detect(out) => mean_sqr_sum(&[&out.boxes, &out.scores]),
        Output::DetectE2e {
            one_to_many,
            one_to_one,
        } => mean_sqr_sum(&[
            &one_to_many.boxes,
            &one_to_many.scores,
            &one_to_one.boxes,
            &one_to_one.scores,
        ]),
        Output::Classify { logits } => logits.sqr()?.mean_all().map_err(Into::into),
        Output::Segment {
            detect,
            masks,
            proto,
            semantic,
        } => {
            let mut tensors = vec![&detect.boxes, &detect.scores, masks, proto];
            if let Some(semantic) = semantic {
                tensors.push(semantic);
            }
            mean_sqr_sum(&tensors)
        }
        Output::SegmentE2e {
            one_to_many_detect,
            one_to_many_masks,
            one_to_one_detect,
            one_to_one_masks,
            proto,
            semantic,
        } => {
            let mut tensors = vec![
                &one_to_many_detect.boxes,
                &one_to_many_detect.scores,
                one_to_many_masks,
                &one_to_one_detect.boxes,
                &one_to_one_detect.scores,
                one_to_one_masks,
                proto,
            ];
            if let Some(semantic) = semantic {
                tensors.push(semantic);
            }
            mean_sqr_sum(&tensors)
        }
        Output::Pose { detect, keypoints } => {
            mean_sqr_sum(&[&detect.boxes, &detect.scores, keypoints])
        }
        Output::PoseE2e {
            one_to_many_detect,
            one_to_many_keypoints,
            one_to_one_detect,
            one_to_one_keypoints,
        } => mean_sqr_sum(&[
            &one_to_many_detect.boxes,
            &one_to_many_detect.scores,
            one_to_many_keypoints,
            &one_to_one_detect.boxes,
            &one_to_one_detect.scores,
            one_to_one_keypoints,
        ]),
        Output::Semantic { logits } => logits.sqr()?.mean_all().map_err(Into::into),
        Output::Obb { detect, angles } => mean_sqr_sum(&[&detect.boxes, &detect.scores, angles]),
        Output::ObbE2e {
            one_to_many_detect,
            one_to_many_angles,
            one_to_one_detect,
            one_to_one_angles,
        } => mean_sqr_sum(&[
            &one_to_many_detect.boxes,
            &one_to_many_detect.scores,
            one_to_many_angles,
            &one_to_one_detect.boxes,
            &one_to_one_detect.scores,
            one_to_one_angles,
        ]),
    }
}

fn mean_sqr_sum(tensors: &[&Tensor]) -> crate::Result<Tensor> {
    let mut iter = tensors.iter();
    let first = iter
        .next()
        .ok_or_else(|| crate::Error::InvalidTensor("empty smoke-loss tensor list".to_string()))?;
    let mut loss = first.sqr()?.mean_all()?;
    for tensor in iter {
        loss = (loss + tensor.sqr()?.mean_all()?)?;
    }
    Ok(loss)
}

pub(crate) fn classification_eval_metrics(
    output: &Output,
    target: &Target,
) -> crate::Result<Option<ClassificationEvalMetrics>> {
    let (Output::Classify { logits }, Target::Classification { class_ids }) = (output, target)
    else {
        return Ok(None);
    };
    if logits.rank() != 2 {
        return Err(crate::Error::InvalidTensor(format!(
            "classification logits must have shape [batch, classes], got {:?}",
            logits.dims()
        )));
    }
    let predictions = logits.argmax(1)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let targets = class_ids.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    if predictions.len() != targets.len() {
        return Err(crate::Error::InvalidTensor(format!(
            "classification predictions and targets must have matching batch length, got {} and {}",
            predictions.len(),
            targets.len()
        )));
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, target)| pred == target)
        .count();
    let logits_f32 = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let top5_correct = logits_f32
        .iter()
        .zip(targets.iter())
        .filter(|(row, target)| top5_contains(row.as_slice(), **target))
        .count();
    Ok(Some(ClassificationEvalMetrics {
        correct,
        top5_correct,
        total: targets.len(),
    }))
}

fn top5_contains(row: &[f32], target: u32) -> bool {
    let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(5).any(|(cls, _)| *cls as u32 == target)
}
