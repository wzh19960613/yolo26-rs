//! Image classification inference.

pub(crate) mod head;
mod model;
pub(crate) mod network;
mod prediction;

pub use model::Model;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 classification model.
pub type Config = crate::model::config::Base;

/// Returns a builder for classification model config with classify-specific defaults.
pub fn config_builder() -> crate::model::config::base::Builder {
    crate::model::config::Base::classify_builder()
}

/// Default square YOLO26 classification input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
