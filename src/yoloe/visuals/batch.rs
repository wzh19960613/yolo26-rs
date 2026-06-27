//! Batched [`BatchVisuals`] for the official `len(img) > 1` bbox-only branch,
//! and the per-image [`VisualBatchItem`] input type.

use candle_core::{Device, Tensor};

use super::single::Visuals;
use crate::yoloe::prompt::visual::{Visual, VisualKind};

/// One image worth of YOLOE visual bbox prompts for batched inference.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualBatchItem {
    /// Original source image size before model letterboxing.
    pub source_size: crate::ImageSize,
    /// Box prompts in original source-image xyxy coordinates.
    pub prompts: Vec<Visual>,
}

/// Batched visual prompt masks with per-image class-id mapping.
///
/// `tensor` is padded to shape `[batch, max_prompts, h, w]`; `class_ids[b][p]`
/// is the original class id for image `b`, prompt slot `p` (before padding),
/// letting callers map predictions back to user-supplied class ids.
#[derive(Debug, Clone)]
pub struct BatchVisuals {
    /// Padded visual masks with shape `[batch, max_prompts, h, w]`.
    pub tensor: Tensor,
    /// Sorted class ids for each image before prompt-dimension padding.
    pub class_ids: Vec<Vec<u32>>,
}

impl BatchVisuals {
    /// Wraps an already-built batched visual-prompt tensor together with its
    /// per-image class mapping.
    ///
    /// Use this when the batched tensor was constructed outside the YOLOE
    /// helpers (e.g. by a training dataset loader that produces
    /// `[batch, prompts, h, w]` directly); [`BatchVisuals::from_boxes`] is the
    /// letterbox-based construction path for inference.
    pub fn from_tensor(tensor: Tensor, class_ids: Vec<Vec<u32>>) -> Self {
        Self { tensor, class_ids }
    }

    /// Builds official-style batched YOLOE visual masks from bbox prompts.
    ///
    /// Each image's source-coordinate boxes are letterboxed into `target_size`,
    /// converted to class-merged per-image visual masks, then padded along the
    /// prompt dimension so the result has shape `[batch, max_prompts, h, w]`.
    /// Official batch prediction currently supports bbox prompts only; mask
    /// prompts should use [`Visuals::from_masks`] for single-image encoding.
    pub fn from_boxes(
        items: &[VisualBatchItem],
        target_size: crate::ImageSize,
        scale_factor: f32,
        device: &Device,
    ) -> crate::Result<Self> {
        validate_image_size(target_size, "target")?;
        if items.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE batched visual prompts require at least one image".to_string(),
            ));
        }

        let mut max_prompts = 0usize;
        let mut masks = Vec::with_capacity(items.len());
        let mut class_ids = Vec::with_capacity(items.len());
        for item in items {
            let prompts = item.letterboxed_prompts(target_size)?;
            let image_masks = Visuals::from_boxes(&prompts, target_size, scale_factor, device)?;
            max_prompts = max_prompts.max(image_masks.tensor.dim(1)?);
            masks.push(image_masks.tensor);
            class_ids.push(sorted_class_ids(&prompts));
        }

        let mut padded = Vec::with_capacity(masks.len());
        for image_masks in masks {
            padded.push(pad_prompt_dimension(&image_masks, max_prompts)?);
        }
        let refs = padded.iter().collect::<Vec<_>>();
        let tensor = Tensor::cat(&refs, 0)?;
        Ok(Self { tensor, class_ids })
    }
}

impl VisualBatchItem {
    /// Creates a batch item from one image size and its source-coordinate box prompts.
    pub fn new(source_size: crate::ImageSize, prompts: Vec<Visual>) -> crate::Result<Self> {
        validate_image_size(source_size, "source")?;
        if prompts.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE batched visual prompts require at least one prompt per image".to_string(),
            ));
        }
        if prompts.iter().any(|prompt| prompt.kind != VisualKind::Box) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE batched visual prompt helper supports box prompts only".to_string(),
            ));
        }
        Ok(Self {
            source_size,
            prompts,
        })
    }

    fn letterboxed_prompts(&self, target_size: crate::ImageSize) -> crate::Result<Vec<Visual>> {
        let geometry = VisualPromptLetterbox::new(self.source_size, target_size)?;
        self.prompts
            .iter()
            .map(|prompt| Visual::from_box(prompt.class_id, geometry.map_xyxy(prompt.xyxy)))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct VisualPromptLetterbox {
    gain: f32,
    pad_x: f32,
    pad_y: f32,
}

impl VisualPromptLetterbox {
    fn new(source_size: crate::ImageSize, target_size: crate::ImageSize) -> crate::Result<Self> {
        validate_image_size(source_size, "source")?;
        validate_image_size(target_size, "target")?;
        let gain = f32::min(
            target_size.height as f32 / source_size.height as f32,
            target_size.width as f32 / source_size.width as f32,
        );
        let resized_w = (source_size.width as f32 * gain).round();
        let resized_h = (source_size.height as f32 * gain).round();
        let pad_x = ((target_size.width as f32 - resized_w) / 2.0 - 0.1).round();
        let pad_y = ((target_size.height as f32 - resized_h) / 2.0 - 0.1).round();
        Ok(Self { gain, pad_x, pad_y })
    }

    fn map_xyxy(self, xyxy: [f32; 4]) -> [f32; 4] {
        [
            xyxy[0] * self.gain + self.pad_x,
            xyxy[1] * self.gain + self.pad_y,
            xyxy[2] * self.gain + self.pad_x,
            xyxy[3] * self.gain + self.pad_y,
        ]
    }
}

fn pad_prompt_dimension(masks: &Tensor, target_prompts: usize) -> crate::Result<Tensor> {
    let (_, prompts, height, width) = masks.dims4()?;
    if prompts == target_prompts {
        return Ok(masks.clone());
    }
    let padding = Tensor::zeros(
        (1, target_prompts - prompts, height, width),
        masks.dtype(),
        masks.device(),
    )?;
    Tensor::cat(&[masks, &padding], 1).map_err(crate::Error::from)
}

fn validate_image_size(size: crate::ImageSize, role: &str) -> crate::Result<()> {
    if size.width == 0 || size.height == 0 {
        return Err(crate::Error::InvalidConfig(format!(
            "YOLOE visual prompt {role} image size must be non-zero"
        )));
    }
    Ok(())
}

fn sorted_class_ids(prompts: &[Visual]) -> Vec<u32> {
    use std::collections::BTreeSet;
    prompts
        .iter()
        .map(|prompt| prompt.class_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
