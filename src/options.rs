#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Confidence and class filtering options.
#[cfg_attr(
    feature = "wasm",
    wasm_bindgen(js_name = FilterOption, getter_with_clone)
)]
#[derive(Debug, Clone)]
pub struct FilterOption {
    /// Minimum confidence score kept in decoded detection-like results. Defaults to 0.25.
    pub confidence_threshold: f32,
    /// Class ids to keep. Empty means all classes are allowed.
    pub class_filter: Vec<u32>,
}

impl Default for FilterOption {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            class_filter: Vec::new(),
        }
    }
}

impl FilterOption {
    pub(crate) fn allows_class(&self, class_id: u32) -> bool {
        self.class_filter.is_empty() || self.class_filter.contains(&class_id)
    }

    /// Returns `(confidence, class_id)` if the detection passes confidence and class filters.
    pub fn check(&self, confidence: f32, class_id: u32) -> Option<(f32, u32)> {
        if confidence < self.confidence_threshold {
            return None;
        }
        if !self.allows_class(class_id) {
            return None;
        }
        Some((confidence, class_id))
    }
}

/// Mask resolution options for segmentation tasks.
#[cfg_attr(
    feature = "wasm",
    wasm_bindgen(js_name = MaskOption, getter_with_clone)
)]
#[derive(Debug, Clone, Default)]
pub struct MaskOption {
    /// Whether masks are returned at source-image resolution.
    pub high_resolution: bool,
}
