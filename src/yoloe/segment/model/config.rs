use candle_core::Device;

use crate::Scale;
use crate::model::DtypeRequest;

use crate::yoloe::head::contrastive::Contrastive;

/// Mask coefficient branch selected for a YOLOE segmentation head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskBranch {
    /// Official YOLOE `one2one_cv5` branch.
    OfficialOne2OneCv5,
    /// Compatibility fallback for converted closed-set segment checkpoints.
    CompatibleOne2OneCv4,
}

impl MaskBranch {
    /// Returns the branch key segment used under the head prefix.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OfficialOne2OneCv5 => "one2one_cv5",
            Self::CompatibleOne2OneCv4 => "one2one_cv4",
        }
    }
}

/// Configuration for loading a YOLOE open-vocabulary segmentation model.
#[derive(Debug, Clone)]
pub struct Config {
    /// Model scale to instantiate.
    pub scale: Scale,
    /// Compute device used by Candle.
    pub device: Device,
    /// Compute dtype request for weights and inference. Defaults to
    /// [`DtypeRequest::Auto`], which infers the dtype from the checkpoint.
    pub dtype: DtypeRequest,
    /// Maximum predictions retained by top-k postprocessing.
    pub max_predictions: usize,
    /// Region/prompt embedding dimension.
    pub embed_dim: usize,
    /// Mask coefficient channels.
    pub mask_channels: usize,
    /// Prototype hidden channels.
    pub proto_channels: usize,
    /// Mask coefficient branch to load.
    pub mask_branch: MaskBranch,
    /// Whether official prompt-free LRPC heads are available and should be loaded.
    pub official_lrpc: bool,
    /// Whether official SAVPE visual-prompt encoder weights are available.
    pub official_savpe: bool,
    /// Whether the checkpoint includes the regular text/visual prompt head
    /// final projections (`one2one_cv2.*.2` and `one2one_cv3.*.2`).
    pub prompt_head: bool,
    /// Intermediate (hidden) width of the classification/embedding branch `cv3`,
    /// inferred from the checkpoint so it matches the official layout instead of
    /// being recomputed from scale formulas (which differ from official sizes).
    pub cls_hidden: usize,
    /// Intermediate (hidden) width of the box-regression branch `cv2`, inferred
    /// from the checkpoint.
    pub box_hidden: usize,
    /// Intermediate (hidden) width of the mask-coefficient branch `cv5`, inferred
    /// from the checkpoint.
    pub mask_hidden: usize,
    /// Intermediate (hidden) width of the official SAVPE encoder.
    pub savpe_hidden: usize,
    /// Contrastive prompt scorer settings.
    pub contrastive: Contrastive,
}

impl Config {
    /// Creates a conservative default for official YOLOE segmentation checkpoints.
    pub fn new(scale: Scale, device: Device, dtype: impl Into<DtypeRequest>) -> Self {
        Self {
            scale,
            device,
            dtype: dtype.into(),
            max_predictions: 300,
            embed_dim: 512,
            mask_channels: 32,
            proto_channels: scale.channel(256),
            mask_branch: MaskBranch::OfficialOne2OneCv5,
            official_lrpc: false,
            official_savpe: false,
            prompt_head: true,
            cls_hidden: 0,
            box_hidden: 0,
            mask_hidden: 0,
            savpe_hidden: 0,
            contrastive: Contrastive::default(),
        }
    }

    /// Validates this model configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_predictions == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segment max_predictions must not be 0".to_string(),
            ));
        }
        if self.embed_dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segment embed_dim must not be 0".to_string(),
            ));
        }
        if self.mask_channels == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segment mask_channels must not be 0".to_string(),
            ));
        }
        if self.proto_channels == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segment proto_channels must not be 0".to_string(),
            ));
        }
        Ok(())
    }
}
