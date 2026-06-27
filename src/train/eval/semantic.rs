//! Semantic segmentation evaluation: mIoU / pixel accuracy / per-class IoU.
//!
//! Mirrors the official `SemanticMetrics`: a per-class confusion matrix is
//! accumulated across the validation set, then mean IoU, pixel accuracy and
//! per-class IoU are derived from it. Fitness is mIoU (EVAL-05).

use candle_core::DType;

use super::{Output, Target};

/// Class id marking pixels excluded from semantic metrics (mirrors the loss).
const IGNORE_CLASS: u32 = u32::MAX;

/// Final semantic-segmentation evaluation summary.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticReport {
    /// Mean IoU averaged over classes present in GT or predictions (fitness).
    pub miou: f32,
    /// Pixel accuracy over non-ignored pixels.
    pub pixel_acc: f32,
    /// Per-class IoU; classes absent from GT and predictions are `0.0`.
    pub per_class_iou: Vec<f32>,
}

impl SemanticReport {
    /// Official semantic fitness is mIoU.
    pub fn fitness(&self) -> f32 {
        self.miou
    }
}

/// Accumulates a per-class confusion matrix (`matrix[target * classes + pred]`).
#[derive(Default)]
pub(crate) struct SemanticMapAccumulator {
    matrix: Vec<u64>,
    classes: usize,
    valid_pixels: u64,
}

impl SemanticMapAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds one batch's flattened prediction/target class maps.
    pub(crate) fn add_batch(&mut self, pred: &[u32], target: &[u32], classes: usize) {
        if self.classes == 0 && classes > 0 {
            self.classes = classes;
            self.matrix = vec![0u64; classes * classes];
        }
        for (pred_c, target_c) in pred.iter().zip(target.iter()) {
            if *target_c == IGNORE_CLASS || *target_c as usize >= classes {
                continue;
            }
            let pred_idx = *pred_c as usize;
            if pred_idx >= classes {
                continue;
            }
            self.matrix[*target_c as usize * classes + pred_idx] += 1;
            self.valid_pixels += 1;
        }
    }

    /// Derives the final mIoU / pixel accuracy / per-class IoU.
    pub(crate) fn finalize(&self) -> Option<SemanticReport> {
        let classes = self.classes;
        if classes == 0 {
            return None;
        }
        let mut per_class_iou = vec![0.0f32; classes];
        let mut tp_total = 0u64;
        let mut present = 0usize;
        let mut iou_sum = 0.0f64;
        for (c, per_class) in per_class_iou.iter_mut().enumerate().take(classes) {
            let tp = self.matrix[c * classes + c];
            let mut target_c = 0u64;
            let mut pred_c = 0u64;
            for k in 0..classes {
                target_c += self.matrix[c * classes + k];
                pred_c += self.matrix[k * classes + c];
            }
            tp_total += tp;
            let union = target_c + pred_c - tp;
            if union == 0 {
                continue;
            }
            let iou = tp as f64 / union as f64;
            *per_class = iou as f32;
            iou_sum += iou;
            present += 1;
        }
        if present == 0 {
            return None;
        }
        let miou = (iou_sum / present as f64) as f32;
        let pixel_acc = if self.valid_pixels > 0 {
            tp_total as f32 / self.valid_pixels as f32
        } else {
            0.0
        };
        Some(SemanticReport {
            miou,
            pixel_acc,
            per_class_iou,
        })
    }
}

/// Updates `acc` from a semantic output/target pair; returns true if semantic.
pub(crate) fn update_semantic_acc(
    output: &Output,
    target: &Target,
    acc: &mut SemanticMapAccumulator,
) -> crate::Result<bool> {
    let (Output::Semantic { logits }, Target::Semantic { class_map }) = (output, target) else {
        return Ok(false);
    };
    let (_batch, classes, _h, _w) = logits.dims4()?;
    let pred = logits
        .argmax(1)?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let target_flat = class_map
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    acc.add_batch(&pred, &target_flat, classes);
    Ok(true)
}
