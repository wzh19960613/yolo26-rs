//! Semantic segmentation inference.

pub(crate) mod head;
mod model;
pub(crate) mod network;
mod prediction;

pub use model::Model;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 semantic segmentation model.
pub type Config = crate::model::config::Base;

/// Returns a builder for semantic segmentation model config with semantic-specific defaults.
pub fn config_builder() -> crate::model::config::base::Builder {
    crate::model::config::Base::semantic_builder()
}

/// Default square YOLO26 semantic segmentation input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
