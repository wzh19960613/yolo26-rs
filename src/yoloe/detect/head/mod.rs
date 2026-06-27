pub(crate) mod branch_set;
pub(crate) mod forward;
pub(crate) mod load;
pub(crate) mod lrpc;
pub(crate) mod prompt_free;

use candle_core::Tensor;

use crate::network::head::{box_branch::BoxBranch, cls_branch::ClsBranch};

use crate::yoloe::head::contrastive::{BnContrastive, Contrastive};

/// Raw YOLOE open-vocabulary detection head outputs before top-k postprocessing.
#[derive(Debug, Clone)]
pub struct HeadParts {
    /// Decoded or raw box-distance channels with shape `[batch, 4, anchors]`.
    pub boxes: Tensor,
    /// Prompt class logits with shape `[batch, classes, anchors]`.
    pub scores: Tensor,
    /// Region embeddings with shape `[batch, embed_dim, anchors]`.
    pub embeddings: Tensor,
    /// Anchor centers with shape `[anchors, 2]`.
    pub anchors: Tensor,
    /// Per-anchor stride tensor broadcastable to decoded boxes.
    pub stride_tensor: Tensor,
}

/// YOLOE detection head adapter with object embeddings and prompt scoring.
pub struct Head {
    pub(crate) box_branches: Vec<BoxBranch>,
    pub(crate) embedding_branches: Vec<ClsBranch>,
    pub(crate) bn_contrastive_heads: Option<Vec<BnContrastive>>,
    pub(crate) strides: [f32; 3],
    pub(crate) embed_dim: usize,
    pub(crate) max_det: usize,
    pub(crate) contrastive: Contrastive,
}
