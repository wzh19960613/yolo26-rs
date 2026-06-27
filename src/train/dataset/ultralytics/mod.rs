//! Ultralytics-style dataset loaders.
//!
//! Each sub-module owns one task's dataset type and its sample/collate logic.
//! YAML parsing and the per-task `*_dataset_from_file` builders live in
//! [`yaml`]; the [`Yaml`] struct itself is shared with the OBB collation path.

pub(crate) use crate::train::exports::*;

/// Classification dataset loader.
pub mod classify;
/// Detection dataset loader.
pub mod detect;
/// Pose/keypoint dataset loader.
pub mod pose;
/// Instance-segmentation dataset loader.
pub mod seg;
/// Semantic-segmentation and oriented-bounding-box dataset loaders.
pub mod semantic_obb;
/// `Yaml` parsing and per-task dataset builders (`*_dataset_from_file`).
pub mod yaml;

pub use classify::*;
pub use detect::*;
pub use pose::*;
pub use seg::*;
pub use semantic_obb::*;
pub use yaml::*;
