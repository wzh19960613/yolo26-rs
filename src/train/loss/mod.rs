//! Supervised loss computation and per-task loss targets.

pub(crate) mod accumulator;
pub(crate) mod build_angle_targets;
pub(crate) mod build_detection_targets;
pub(crate) mod ciou_dfl;
pub(crate) mod detection_buffers;
pub(crate) mod detection_config;
pub(crate) mod instance_mask;
pub(crate) mod keypoint;
pub(crate) mod progressive;
pub(crate) mod semantic;
pub(crate) mod smoke;
pub(crate) mod supervised_dispatch;
pub(crate) mod supervised_targets;
pub(crate) mod target_detection;

use crate::model::ImageSize;
pub(crate) use crate::train::exports::*;

pub(crate) use accumulator::*;
pub(crate) use build_angle_targets::*;
pub(crate) use build_detection_targets::*;
pub(crate) use ciou_dfl::*;
pub(crate) use detection_buffers::*;
pub(crate) use detection_config::*;
pub(crate) use instance_mask::*;
pub(crate) use keypoint::*;
pub use progressive::*;
pub use semantic::*;
pub use smoke::*;
pub use supervised_dispatch::*;
pub use supervised_targets::*;
pub use target_detection::LossComponents;
pub(crate) use target_detection::*;
