use candle_core::{DType, Tensor};

/// Class id used to mark semantic pixels that should not contribute loss.
pub(crate) const SEMANTIC_IGNORE_CLASS_ID: u32 = u32::MAX;

pub(crate) fn semantic_loss_with_ignore(
    flattened_logits: &Tensor,
    flattened_targets: &Tensor,
    classes: usize,
) -> crate::Result<Tensor> {
    let targets = flattened_targets
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let mut indices = Vec::with_capacity(targets.len());
    let mut kept_targets = Vec::with_capacity(targets.len());
    for (index, class_id) in targets.into_iter().enumerate() {
        if class_id == SEMANTIC_IGNORE_CLASS_ID {
            continue;
        }
        if class_id as usize >= classes {
            return Err(crate::Error::InvalidTensor(format!(
                "semantic class id {class_id} is outside logits class count {classes}"
            )));
        }
        indices.push(index as u32);
        kept_targets.push(class_id);
    }
    if indices.is_empty() {
        return (flattened_logits.sum_all()? * 0.0).map_err(crate::Error::from);
    }
    let index_tensor = Tensor::new(indices, flattened_logits.device())?;
    let target_tensor = Tensor::new(kept_targets, flattened_logits.device())?;
    let logits = flattened_logits.index_select(&index_tensor, 0)?;
    candle_nn::loss::cross_entropy(&logits, &target_tensor).map_err(crate::Error::from)
}
