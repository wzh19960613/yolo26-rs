//! Raw YOLOE segmentation training outputs (pre-sigmoid, pre-decode).

use candle_core::Tensor;

/// Raw dense tensors produced by a trainable YOLOE segmentation forward pass.
///
/// All tensors are the **trainable** form (BatchNorm in train mode, pre-sigmoid
/// scores, pre-`dist2bbox` box distances) so losses differentiate through every
/// head. Tensors are concatenated across the three detection scales over the
/// anchor axis.
#[derive(Debug, Clone)]
pub struct Output {
    /// Raw box distances `[batch, 4, anchors]` (anchor-offset form).
    pub boxes: Tensor,
    /// Prompt class logits `[batch, classes, anchors]` (pre-sigmoid).
    pub scores: Tensor,
    /// Region embeddings `[batch, embed_dim, anchors]` (post-cv3 conv).
    pub embeddings: Tensor,
    /// Mask coefficients `[batch, nm, anchors]`.
    pub masks: Tensor,
    /// Mask prototypes `[batch, nm, mask_h, mask_w]`.
    pub proto: Tensor,
    /// Anchor centers `[anchors, 2]`.
    pub anchors: Tensor,
    /// Per-anchor stride tensor broadcastable to decoded boxes.
    pub stride_tensor: Tensor,
    /// Per-scale class feature maps `[B, embed_dim, H, W]` consumed by LRPC.
    pub cls_feature_maps: Vec<Tensor>,
    /// Per-scale box feature maps `[B, 4, H, W]` consumed by LRPC `loc`.
    pub loc_feature_maps: Vec<Tensor>,
}

impl Output {
    /// Number of anchors across all scales.
    pub fn anchor_count(&self) -> candle_core::Result<usize> {
        self.anchors.dim(0)
    }
}
