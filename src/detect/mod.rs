//! Object detection inference.

pub(crate) mod head;
mod model;
pub(crate) mod network;
mod prediction;
pub mod sahi;

pub use model::Model;
#[allow(unused_imports)]
pub(crate) use model::postprocess;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 detection model.
pub type Config = crate::model::config::Base;

/// Detection prediction options.
pub type PredictOptions = crate::options::FilterOption;

/// Returns a builder for detection model config with detect-specific defaults.
pub fn config_builder() -> crate::model::config::base::Builder {
    crate::model::config::Base::detect_builder()
}

/// Default square YOLO26 detection input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
