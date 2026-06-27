//! YOLOE open-vocabulary detection: task-specific models and heads.

/// Detection-specific head (objectness/box/embedding branches).
pub mod head;
/// Detection-specific model wrapping the network.
pub mod model;

pub use model::Model;
