use candle_core::Device;

use crate::Scale;
use crate::model::DtypeRequest;

use crate::yoloe::head::contrastive::Contrastive;

/// Configuration for loading a YOLOE open-vocabulary detection-only model.
///
/// This mirrors [`Config`](crate::yoloe::segment::model::config::Config)
/// but drops the mask-coefficient and prototype fields, since the detect-only
/// path emits boxes and class scores without segmentation prototypes. The
/// `infer_from_*` loaders live in
/// [`detect_model_config_infer`](crate::yoloe::detect::model::config_infer).
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
    /// Whether official prompt-free LRPC heads are available and should be loaded.
    pub official_lrpc: bool,
    /// Whether official SAVPE visual-prompt encoder weights are available.
    pub official_savpe: bool,
    /// Intermediate (hidden) width of the official SAVPE encoder.
    pub savpe_hidden: usize,
    /// Whether the checkpoint includes the regular text/visual prompt head
    /// final projections (`one2one_cv2.*.2` and `one2one_cv3.*.2`).
    pub prompt_head: bool,
    /// Intermediate width of the classification/embedding branch `cv3`,
    /// inferred from the checkpoint.
    pub cls_hidden: usize,
    /// Intermediate width of the box-regression branch `cv2`, inferred from the
    /// checkpoint.
    pub box_hidden: usize,
    /// Contrastive prompt scorer settings.
    pub contrastive: Contrastive,
}

impl Config {
    /// Creates a conservative default for official YOLOE detection checkpoints.
    pub fn new(scale: Scale, device: Device, dtype: impl Into<DtypeRequest>) -> Self {
        Self {
            scale,
            device,
            dtype: dtype.into(),
            max_predictions: 300,
            embed_dim: 512,
            official_lrpc: false,
            official_savpe: false,
            savpe_hidden: 0,
            prompt_head: true,
            cls_hidden: 0,
            box_hidden: 0,
            contrastive: Contrastive::default(),
        }
    }

    /// Validates this model configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_predictions == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE detect max_predictions must not be 0".to_string(),
            ));
        }
        if self.embed_dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE detect embed_dim must not be 0".to_string(),
            ));
        }
        Ok(())
    }
}
