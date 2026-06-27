use candle_core::Tensor;

/// Output from a prompt-free LRPC forward pass.
#[derive(Debug, Clone)]
pub struct LrpcOutput {
    /// Selected localization features with shape `[batch, box_channels, selected]`.
    pub boxes: Tensor,
    /// Vocabulary scores with shape `[batch, classes, selected]`.
    pub scores: Tensor,
    /// Valid selected positions with shape `[batch, selected]`.
    pub valid: Tensor,
    /// Original flattened spatial indices selected for each batch item.
    pub selected_indices: Vec<Vec<usize>>,
}

/// Output from an official-style YOLOE LRPC head.
#[derive(Debug, Clone)]
pub struct OfficialOutput {
    /// Localized box-distance features with shape `[batch, box_channels, height, width]`.
    pub boxes: Tensor,
    /// Vocabulary scores with shape `[batch, classes, selected]`.
    pub scores: Tensor,
    /// Per-batch flattened spatial indices selected by the proposal filter.
    pub selected_indices: Vec<Vec<usize>>,
}

/// Output from an official-style three-scale YOLOE LRPC forward pass.
#[derive(Debug, Clone)]
pub struct OfficialPyramidOutput {
    /// Selected box-distance features with shape `[batch, box_channels, selected]`.
    pub boxes: Tensor,
    /// Vocabulary logits with shape `[batch, classes, selected]`.
    pub scores: Tensor,
    /// Valid selected positions with shape `[batch, selected]`.
    pub valid: Tensor,
    /// Selected anchor centers with shape `[batch, 2, selected]`.
    pub anchors: Tensor,
    /// Selected stride tensor with shape `[batch, 1, selected]`.
    pub stride_tensor: Tensor,
    /// Per-batch flattened global spatial indices selected by the proposal filter.
    pub selected_indices: Vec<Vec<usize>>,
}

/// Output from an official-style YOLOE segmentation LRPC forward pass.
#[derive(Debug, Clone)]

pub struct OfficialSegmentParts {
    /// Prompt-free LRPC detection outputs.
    pub detect: OfficialPyramidOutput,
    /// Selected mask coefficients with shape `[batch, masks, selected]`.
    pub masks: Tensor,
    /// Prototype masks with shape `[batch, masks, height, width]`.
    pub proto: Tensor,
}
