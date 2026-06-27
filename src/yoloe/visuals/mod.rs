//! Named types for the SAVPE-input tensor, plus visual-prompt source dispatch.
//!
//! These types replace the raw `[1, classes, h, w]` `Tensor` that used to flow
//! between visual-prompt builders and the SAVPE encoder. Wrapping it in
//! [`Visuals`] (single image) / [`BatchVisuals`] (batched) makes the "merged
//! per-class prompt mask" invariant part of the type, and removes the double
//! "mask" naming (`visual_prompt_masks_from_masks`) by adopting the official
//! term `visuals` for the SAVPE input tensor.

mod batch;
pub(crate) mod merge;
mod single;

use candle_core::Tensor;

pub use batch::{BatchVisuals, VisualBatchItem};
pub use single::Visuals;

/// Discriminates the source annotation form behind one `predict_visual_prompts`
/// call, so callers don't have to pick between two near-identical methods.
///
/// [`Visual`](super::Visual) already carries a
/// [`VisualKind`](super::VisualKind) tag, but box prompts need no
/// extra data while mask prompts must also carry the source-image mask tensor.
/// This enum is the single dispatch point between the box- and mask-prompt
/// rasterization paths.
#[derive(Debug, Clone)]
pub enum VisualSource<'a> {
    /// Prompt masks are rasterized from the source-image xyxy boxes already
    /// stored on each [`Visual`](super::Visual).
    Boxes,
    /// Prompt masks come from a source-image mask tensor, shaped
    /// `[prompts, H, W]` or `[1, prompts, H, W]` in original-image coordinates.
    Masks(&'a Tensor),
}
