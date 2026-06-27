//! Per-task sample collation helpers.

use crate::model::ImageSize;
use crate::train::dataset::Sample;
pub(crate) use crate::train::exports::*;

pub(crate) mod obb;
pub(crate) mod seg;

pub use obb::collate_obb_samples;
pub use obb::{Dataset, Split, Yaml};
pub use seg::{collate_pose_samples, collate_segmentation_samples};
