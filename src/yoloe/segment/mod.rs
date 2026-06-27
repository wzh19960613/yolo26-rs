//! YOLOE open-vocabulary segmentation: task-specific models and heads.

/// Segmentation-specific head (detect branches + mask/proto branches).
pub mod head;
/// Segmentation-specific model wrapping the network.
pub mod model;

pub use model::Model;
