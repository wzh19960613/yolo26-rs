/// Concrete YOLOE head key plan derived from a checkpoint layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyPlan {
    /// Head prefix, usually `model.23`.
    pub head_prefix: String,
    /// Box branch prefix.
    pub box_branch: String,
    /// Region embedding branch prefix.
    pub embedding_branch: String,
    /// Official BN contrastive branch prefix, when available.
    pub contrastive_branch: Option<String>,
    /// Model mask branch prefix, when a segmentation head is available.
    pub segment_mask_branch: Option<String>,
    /// Prototype mask prefix, when a segmentation head is available.
    pub proto: Option<String>,
    /// Whether the plan uses official YOLOE `BNContrastiveHead` parameters.
    pub uses_official_bn_contrastive: bool,
    /// Whether the plan uses official YOLOE `one2one_cv5` masks.
    pub uses_official_segment_mask_branch: bool,
}
