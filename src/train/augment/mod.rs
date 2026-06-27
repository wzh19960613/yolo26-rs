//! Native data-augmentation pipeline aligned with Ultralytics defaults.
//!
//! The pipeline mirrors the official YOLO training augmentations enabled by
//! default: HSV jitter, random scale/translation, four-image mosaic, two-image
//! mixup and horizontal/vertical flip. Pose keypoint flips additionally swap
//! left/right pairs via the official `flip_indices` (see [`flip_indices`]).
//! `candle` 0.10 has no affine `grid_sample` op, so rotation/shear/perspective
//! (defaulted to zero upstream) are not modeled.
//!
//! Building blocks are split across [`seeded_rng`] (reproducible noise),
//! [`geometry`] (pure box transforms), [`hsv`] (color jitter), [`affine`],
//! [`mosaic`], [`mixup`] and [`apply`] (single-image orchestration), exposed to
//! callers through [`AugmentingDataset`].

mod affine;
mod affine_mask;
mod affine_target;
mod apply;
mod config;
mod dataset;
mod flip_indices;
mod flip_target;
mod geometry;
mod hsv;
mod mask_resample;
mod mixup;
mod mosaic;
mod mosaic_target;
pub(crate) mod numpy_mt19937;
mod seeded_rng;

use super::{
    Dataset, DetectionTargets, ObbTargets, PoseTargets, Sample, SegmentationTargets, Target,
};

pub use config::AugmentConfig;
pub use dataset::AugmentingDataset;

pub(crate) use seeded_rng::SeededRng;
