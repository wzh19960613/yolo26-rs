//! Instance segmentation inference.

pub(crate) mod head;
mod mask;
mod model;
pub(crate) mod network;
mod prediction;

#[cfg(any(feature = "yoloe-text", feature = "yoloe-visual", feature = "yoloe-pf"))]
pub(crate) use model::postprocess_segmentation;

pub use mask::Mask;
pub use model::Model;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 instance segmentation model.
pub type Config = crate::model::config::Base;

/// Returns a builder for segmentation model config with segment-specific defaults.
pub fn config_builder() -> crate::model::config::base::Builder {
    crate::model::config::Base::segment_builder()
}

/// Default square YOLO26 segmentation input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
