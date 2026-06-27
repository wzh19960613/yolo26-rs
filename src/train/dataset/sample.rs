use super::*;

/// One sample returned by a dataset.
pub struct Sample {
    /// Preprocessed input tensor.
    pub input: Tensor,
    /// Training target for the input.
    pub target: Target,
}

/// Collates single-image classification samples into one batch.
pub fn collate_classification_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "classification sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut class_refs = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all classification sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Classification { class_ids } = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_classification_samples only accepts classification targets".to_string(),
            ));
        };
        if class_ids.dims() != [1] {
            return Err(crate::Error::InvalidTensor(format!(
                "classification class ids must have shape [1], got {:?}",
                class_ids.dims()
            )));
        }
        input_refs.push(&sample.input);
        class_refs.push(class_ids);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let class_ids = Tensor::cat(&class_refs, 0)?;
    Ok(Sample {
        input,
        target: Target::Classification { class_ids },
    })
}

/// Collates single-image semantic segmentation samples into one batch.
pub fn collate_semantic_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "semantic sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut class_map_refs = Vec::with_capacity(samples.len());
    let mut class_map_shape = None;
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all semantic sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Semantic { class_map } = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_semantic_samples only accepts semantic targets".to_string(),
            ));
        };
        let shape = class_map.dims().to_vec();
        if shape.len() != 3 || shape[0] != 1 {
            return Err(crate::Error::InvalidTensor(format!(
                "semantic class maps must have shape [1, H, W], got {shape:?}"
            )));
        }
        if let Some(expected) = &class_map_shape {
            if expected != &shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all semantic class maps must share shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            class_map_shape = Some(shape);
        }
        input_refs.push(&sample.input);
        class_map_refs.push(class_map);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let class_map = Tensor::cat(&class_map_refs, 0)?;
    Ok(Sample {
        input,
        target: Target::Semantic { class_map },
    })
}

/// Collates single-image detection samples into one batch.
pub fn collate_detection_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "detection sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut box_refs = Vec::with_capacity(samples.len());
    let mut class_refs = Vec::with_capacity(samples.len());
    let mut valid_refs = Vec::with_capacity(samples.len());
    let mut target_shape = None;
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all detection sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Detection(targets) = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_detection_samples only accepts detection targets".to_string(),
            ));
        };
        let shape = targets.boxes_xyxy.dims().to_vec();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
            return Err(crate::Error::InvalidTensor(format!(
                "detection boxes must have shape [1, objects, 4], got {shape:?}"
            )));
        }
        if let Some(expected) = &target_shape {
            if expected != &shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all detection targets must share shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            target_shape = Some(shape);
        }
        input_refs.push(&sample.input);
        box_refs.push(&targets.boxes_xyxy);
        class_refs.push(&targets.class_ids);
        valid_refs.push(&targets.valid);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let boxes_xyxy = Tensor::cat(&box_refs, 0)?;
    let class_ids = Tensor::cat(&class_refs, 0)?;
    let valid = Tensor::cat(&valid_refs, 0)?;
    Ok(Sample {
        input,
        target: Target::Detection(DetectionTargets::new(boxes_xyxy, class_ids, valid)?),
    })
}
