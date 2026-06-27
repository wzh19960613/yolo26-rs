//! Oriented bounding-box inference.

mod bbox;
pub(crate) mod head;
mod model;
pub(crate) mod network;
mod prediction;

pub use bbox::BBox;
pub use model::Model;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 OBB model.
pub type Config = crate::model::config::Base;

/// Returns a builder for OBB model config with obb-specific defaults.
pub fn config_builder() -> crate::model::config::base::Builder {
    crate::model::config::Base::obb_builder()
}

/// OBB prediction options.
pub type PredictOptions = crate::options::FilterOption;

/// Default square YOLO26 OBB input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
