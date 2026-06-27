//! Dataset abstraction and Ultralytics-format loaders for each task.

/// Per-task sample collation helpers.
pub mod collate;
/// Ultralytics-format dataset loaders split by task.
pub mod ultralytics;

pub(crate) mod detection_label;
pub(crate) mod obb_label;
pub(crate) mod parse_names;
pub(crate) mod sample;
pub(crate) mod sample_order;
pub(crate) mod segmentation_label;

use crate::model::ImageSize;
pub(crate) use crate::train::exports::*;

pub use collate::{Dataset, Split};
pub use sample::{
    Sample, collate_classification_samples, collate_detection_samples, collate_semantic_samples,
};
pub use sample_order::SampleOrder;
