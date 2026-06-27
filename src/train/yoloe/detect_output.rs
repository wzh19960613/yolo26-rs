//! Open-vocabulary head training output.

use candle_core::Tensor;

/// Raw YOLOE open-vocabulary head outputs for training, including per-scale
/// feature maps reused by the LRPC prompt-free head.
#[derive(Debug, Clone)]
pub struct HeadOutput {
    /// Raw box distances `[batch, 4, anchors]`.
    pub boxes: Tensor,
    /// Prompt class logits `[batch, classes, anchors]` (pre-sigmoid, BN train).
    pub scores: Tensor,
    /// Region embeddings `[batch, embed_dim, anchors]`.
    pub embeddings: Tensor,
    /// Anchor centers `[anchors, 2]`.
    pub anchors: Tensor,
    /// Per-anchor stride tensor.
    pub stride_tensor: Tensor,
    /// Per-scale class feature maps `[B, embed_dim, H, W]`.
    pub cls_feature_maps: Vec<Tensor>,
    /// Per-scale box feature maps `[B, 4*reg_max, H, W]`.
    pub loc_feature_maps: Vec<Tensor>,
}
