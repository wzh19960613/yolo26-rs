// Prelude types re-exported so files using `use super::*` (resolved through
// each submodule's `pub(crate) use crate::train::exports::*`) get the same
// common imports the flat module used to provide at the train root.
pub(crate) use std::collections::HashMap;
pub(crate) use std::path::{Component, Path, PathBuf};

pub(crate) use candle_core::{DType, Device, Tensor, Var};
pub(crate) use candle_nn::{VarBuilder, VarMap};

pub(crate) use crate::model::ImageSize;

pub(crate) use super::best_metric::*;
pub use super::class_filter::*;
pub use super::early_stopping::*;
pub use super::freeze::*;
pub use super::load_report::*;
pub use super::lr_schedule::*;
pub use super::optimizer::*;
pub use super::warmup_schedule::*;
pub use crate::train::checkpoint::report::*;
pub use crate::train::checkpoint::resume_state::*;
pub use crate::train::dataset::collate::obb::*;
pub use crate::train::dataset::collate::seg::*;
pub(crate) use crate::train::dataset::detection_label::*;
pub(crate) use crate::train::dataset::obb_label::*;
pub(crate) use crate::train::dataset::parse_names::*;
pub use crate::train::dataset::sample::*;
pub use crate::train::dataset::sample_order::*;
pub(crate) use crate::train::dataset::segmentation_label::*;
pub use crate::train::dataset::ultralytics::*;
pub(crate) use crate::train::eval::class_names_in_split::*;
pub use crate::train::eval::classification_metrics::*;
pub(crate) use crate::train::eval::classify_preprocess::*;
pub(crate) use crate::train::eval::decode_keypoint_xy::*;
pub(crate) use crate::train::eval::detection::*;
pub(crate) use crate::train::eval::detection_assignment::*;
pub use crate::train::eval::map_accumulator::MapReport;
pub(crate) use crate::train::eval::map_accumulator::*;
pub(crate) use crate::train::eval::map_match::*;
pub(crate) use crate::train::eval::map_math::*;
pub use crate::train::eval::map_public::{DetectionMapAccumulator, MaskMapAccumulator};
pub(crate) use crate::train::eval::obb::*;
pub(crate) use crate::train::eval::pose::*;
pub(crate) use crate::train::eval::predictions::*;
pub(crate) use crate::train::eval::segment_mask_decode::*;
pub(crate) use crate::train::eval::segment_mask_match::*;
pub(crate) use crate::train::eval::segmentation::*;
pub(crate) use crate::train::eval::semantic::SemanticReport;
pub(crate) use crate::train::eval::semantic::{SemanticMapAccumulator, update_semantic_acc};
pub(crate) use crate::train::eval::semantic_ignore::*;
pub(crate) use crate::train::eval::semantic_mask_decode::*;
pub(crate) use crate::train::loss::accumulator::Report;
pub(crate) use crate::train::loss::accumulator::{
    LossComponentsAccumulator, blend_tensor_components, scalar_loss_value, scalar_optional_loss,
};
pub(crate) use crate::train::loss::build_angle_targets::*;
pub(crate) use crate::train::loss::build_detection_targets::*;
pub(crate) use crate::train::loss::instance_mask::*;
pub(crate) use crate::train::loss::keypoint::*;
pub use crate::train::loss::progressive::*;
pub(crate) use crate::train::loss::semantic::*;
pub(crate) use crate::train::loss::smoke::*;
pub(crate) use crate::train::loss::supervised_dispatch::*;
pub use crate::train::loss::supervised_targets::*;
pub(crate) use crate::train::loss::target_detection::LossComponents;
pub(crate) use crate::train::loss::target_detection::*;
pub use crate::train::model::*;
pub(crate) use crate::train::optimizer::adamw::*;
pub use crate::train::optimizer::musgd::*;
pub(crate) use crate::train::optimizer::state::*;
pub(crate) use crate::train::runner::report::*;
pub use crate::train::runner::*;
pub(crate) use crate::train::session::epoch::*;
pub(crate) use crate::train::session::loop_steps::*;
pub use crate::train::session::*;
pub(crate) use crate::train::yoloe::eval_common::*;
