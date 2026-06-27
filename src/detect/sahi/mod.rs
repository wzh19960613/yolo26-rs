//! SAHI-style sliced inference and detection merging.
//!
//! The implementation is split across [`options`] (configuration types and the
//! slice window descriptor), [`slicing`] (slice generation and per-slice
//! inference orchestration) and [`merge`] (cross-slice detection merging).

pub mod merge;
pub mod options;
pub mod slicing;

pub use merge::merge_detections;
pub use options::{MatchMetric, MergeStrategy, Options, SliceWindow};
pub use slicing::{generate_slices, sliced_predict};
