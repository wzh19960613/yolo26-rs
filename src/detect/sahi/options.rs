//! SAHI option types and slice window description.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Overlap metric used when merging detections across slices.
#[cfg_attr(feature = "wasm", wasm_bindgen(js_name = MatchMetric))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMetric {
    /// Intersection over union.
    Iou,
    /// Intersection over smaller box area.
    Ios,
}

/// Strategy used to merge overlapping slice detections.
#[cfg_attr(feature = "wasm", wasm_bindgen(js_name = MergeStrategy))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Non-maximum suppression keeps the highest-confidence detection.
    Nms,
    /// Greedy non-maximum merging combines overlapping detections.
    GreedyNmm,
}

/// Options controlling sliced inference and merge behavior.
#[cfg_attr(feature = "wasm", wasm_bindgen(js_name = SahiOptions))]
#[derive(Debug, Clone)]
pub struct Options {
    /// Requested slice width in pixels.
    pub slice_width: u32,
    /// Requested slice height in pixels.
    pub slice_height: u32,
    /// Horizontal overlap ratio between adjacent slices.
    pub overlap_width_ratio: f32,
    /// Vertical overlap ratio between adjacent slices.
    pub overlap_height_ratio: f32,
    /// Whether to also run detection on the full image.
    pub include_full_image: bool,
    /// Merge strategy applied to detections from all slices.
    pub merge_strategy: MergeStrategy,
    /// Metric used to decide whether two detections match.
    pub match_metric: MatchMetric,
    /// Minimum match metric value required for merging.
    pub match_threshold: f32,
    /// Whether detections of different classes can be merged.
    pub class_agnostic: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            slice_width: 640,
            slice_height: 640,
            overlap_width_ratio: 0.2,
            overlap_height_ratio: 0.2,
            include_full_image: false,
            merge_strategy: MergeStrategy::GreedyNmm,
            match_metric: MatchMetric::Ios,
            match_threshold: 0.5,
            class_agnostic: false,
        }
    }
}

/// Pixel window covered by one SAHI slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceWindow {
    /// Left coordinate of the slice.
    pub x: u32,
    /// Top coordinate of the slice.
    pub y: u32,
    /// Width of the slice.
    pub width: u32,
    /// Height of the slice.
    pub height: u32,
}
