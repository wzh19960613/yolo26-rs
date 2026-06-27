use crate::yoloe::head::lrpc::official::Official;

/// Official YOLOE prompt-free LRPC adapter across the three prediction scales.
pub struct Pyramid {
    pub(crate) heads: Vec<Official>,
    pub(crate) strides: [f32; 3],
    pub(crate) classes: usize,
    pub(crate) feature_dim: usize,
    pub(crate) proposal_channels: usize,
    pub(crate) loc_feature_dim: usize,
    pub(crate) box_channels: usize,
}
