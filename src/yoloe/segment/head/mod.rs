pub(crate) mod build;
pub(crate) mod forward;
pub(crate) mod load;
pub(crate) mod lrpc;

use candle_core::Tensor;

use crate::network::head::{dense_branch::DenseBranch, proto::Proto26};

use crate::yoloe::detect::head::branch_set::BranchSet;

/// Raw YOLOE open-vocabulary segmentation head outputs before top-k postprocessing.
#[derive(Debug, Clone)]
pub struct HeadParts {
    /// Detection head outputs from the detect-task head.
    pub detect: crate::yoloe::detect::head::HeadParts,
    /// Mask coefficients with shape `[batch, masks, anchors]`.
    pub masks: Tensor,
    /// Prototype masks with shape `[batch, masks, height, width]`.
    pub proto: Tensor,
}

/// One-to-many branches (`cv2`/`cv3`/`cv4` box+embedding+contrastive plus `cv5`
/// mask coefficients). Present on trainable `-seg` models so the saved
/// checkpoint matches the official symmetric layout; not consumed by inference
/// or the one-to-one training loss.
#[allow(dead_code)]
pub(crate) struct OneToManyBranches {
    /// `cv2`/`cv3`/`cv4` detection branch set.
    pub(crate) detect: BranchSet,
    /// `cv5` mask-coefficient branches.
    pub(crate) mask_branches: Vec<DenseBranch>,
}

/// YOLOE segmentation head adapter with open-vocabulary scoring.
pub struct Head {
    pub(crate) detect: crate::yoloe::detect::head::Head,
    pub(crate) mask_branches: Vec<DenseBranch>,
    pub(crate) proto: Proto26,
    pub(crate) nm: usize,
    /// Official one-to-many branches when building a full `-seg` checkpoint.
    /// Held so its weights stay in the shared `VarMap`; not read at inference.
    #[allow(dead_code)]
    pub(crate) one_to_many: Option<OneToManyBranches>,
}
