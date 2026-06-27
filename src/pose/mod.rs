//! Pose and keypoint inference.

pub(crate) mod head;
mod keypoint;
mod model;
pub(crate) mod network;
mod prediction;

pub use keypoint::Keypoint;
pub use model::Model;
pub use prediction::Prediction;

/// Configuration used when loading a YOLO26 pose model.
pub type Config = crate::model::config::ForPose;

/// Returns a builder for pose model config with pose-specific defaults.
pub fn config_builder() -> crate::model::config::for_pose::Builder {
    crate::model::config::ForPose::builder()
}

/// Pose prediction options.
pub type PredictOptions = crate::options::FilterOption;

/// Default square YOLO26 pose input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;
