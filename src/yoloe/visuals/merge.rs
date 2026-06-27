//! Same-class merging of visual prompt masks, matching official
//! `LoadVisualPrompt.get_visuals()` semantics.

use candle_core::Tensor;

use super::Visuals;
use crate::yoloe::prompt::visual::{Visual, visual_prompt_class_ids};

/// Merges per-prompt masks into per-class masks.
///
/// `prompt_masks` is `[batch, prompts, H, W]` where the prompt axis either
/// equals `prompts.len()` (one row per [`Visual`]) or equals the
/// deduplicated class count (already merged). Same-class rows are combined with
/// element-wise `maximum`, and classes are emitted in ascending class-id order.
/// The result is returned wrapped in [`Visuals`]; the caller is expected to
/// pass a single-image batch.
pub(crate) fn merge_visual_prompt_masks(
    prompts: &[Visual],
    prompt_masks: &Tensor,
) -> crate::Result<Visuals> {
    let class_ids = visual_prompt_class_ids(prompts);
    let mask_prompt_count = match prompt_masks.dims() {
        [_, prompt_count, _, _] => *prompt_count,
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE visual prompt masks must have shape [batch, prompts, height, width], got {dims:?}"
            )));
        }
    };
    if mask_prompt_count != prompts.len() {
        if mask_prompt_count == class_ids.len() {
            return Ok(Visuals {
                tensor: prompt_masks.clone(),
            });
        }
        return Err(crate::Error::InvalidTensor(format!(
            "YOLOE visual prompt mask count must match exemplar count {} or merged class count {}, got {mask_prompt_count}",
            prompts.len(),
            class_ids.len()
        )));
    }

    let mut class_masks = Vec::with_capacity(class_ids.len());
    for class_id in class_ids {
        let mut merged: Option<Tensor> = None;
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            if prompt.class_id != class_id {
                continue;
            }
            let mask = prompt_masks.narrow(1, prompt_index, 1)?;
            merged = Some(match merged {
                Some(current) => current.maximum(&mask)?,
                None => mask,
            });
        }
        let mask = merged.ok_or_else(|| {
            crate::Error::InvalidConfig(format!("YOLOE visual class {class_id} has no masks"))
        })?;
        class_masks.push(mask);
    }
    let refs = class_masks.iter().collect::<Vec<_>>();
    let tensor = Tensor::cat(&refs, 1).map_err(crate::Error::from)?;
    Ok(Visuals { tensor })
}
