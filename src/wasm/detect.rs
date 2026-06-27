//! Detection entry points and SAHI/option bindings for the WASM API.

use wasm_bindgen::prelude::*;

use crate::detect::sahi::{MatchMetric, MergeStrategy, Options as SahiOptions};
use crate::detect::{self, Prediction};
use crate::{FilterOption, Image};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque detection model.
#[wasm_bindgen(js_name = DetectModel)]
pub struct WasmDetectModel {
    inner: detect::Model,
}

#[wasm_bindgen(js_class = DetectModel)]
impl WasmDetectModel {
    /// Creates a detection model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_detect_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner = detect::Model::from_bytes_with(bytes, native)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs detection on an RGB pixel buffer.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        inference: &FilterOption,
    ) -> Result<WasmDetections, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let detections = self
            .inner
            .predict(&image, inference)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmDetections::new(detections))
    }

    /// Runs detection on an RGBA pixel buffer (alpha discarded).
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        inference: &FilterOption,
    ) -> Result<WasmDetections, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height, inference)
    }

    /// Runs SAHI sliced detection on an RGB pixel buffer.
    pub fn predict_sahi_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        inference: &FilterOption,
        sahi: &SahiOptions,
    ) -> Result<WasmDetections, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let detections = self
            .inner
            .predict_sahi(&image, inference, sahi)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmDetections::new(detections))
    }

    /// Runs SAHI sliced detection on an RGBA pixel buffer (alpha discarded).
    pub fn predict_sahi_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        inference: &FilterOption,
        sahi: &SahiOptions,
    ) -> Result<WasmDetections, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_sahi_rgb(&rgb, width, height, inference, sahi)
    }
}

/// One detection returned by the WASM API.
#[wasm_bindgen(js_name = Detection)]
#[derive(Debug, Clone)]
pub struct WasmDetection {
    detection: Prediction,
}

/// Indexed detection collection returned by the WASM API.
#[wasm_bindgen(js_name = Detections)]
#[derive(Debug, Clone)]
pub struct WasmDetections {
    detections: Vec<Prediction>,
}

impl WasmDetections {
    pub(super) fn new(detections: Vec<Prediction>) -> Self {
        Self { detections }
    }
}

#[wasm_bindgen]
impl WasmDetection {
    /// Returns the left coordinate.
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.detection.bbox.x_min
    }

    /// Returns the top coordinate.
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.detection.bbox.y_min
    }

    /// Returns the box width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f32 {
        self.detection.bbox.width()
    }

    /// Returns the box height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f32 {
        self.detection.bbox.height()
    }

    /// Returns the detection confidence.
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.detection.confidence
    }

    /// Returns the numeric class id.
    #[wasm_bindgen(getter, js_name = classId)]
    pub fn class_id(&self) -> u32 {
        self.detection.class_id
    }
}

#[wasm_bindgen]
impl WasmDetections {
    /// Returns the number of detections.
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    /// Returns whether the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }

    /// Returns one detection by index.
    pub fn get(&self, index: usize) -> Result<WasmDetection, JsValue> {
        let detection = self
            .detections
            .get(index)
            .cloned()
            .ok_or_else(|| js_error(format!("detection index {index} is out of bounds")))?;
        Ok(WasmDetection { detection })
    }

    /// Returns a flat `[x, y, width, height, confidence, class_id, ...]` vector.
    #[wasm_bindgen(js_name = toFlatArray)]
    pub fn to_flat_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.detections.len() * 6);
        for detection in &self.detections {
            out.extend_from_slice(&[
                detection.bbox.x_min,
                detection.bbox.y_min,
                detection.bbox.width(),
                detection.bbox.height(),
                detection.confidence,
                detection.class_id as f32,
            ]);
        }
        out
    }
}

#[wasm_bindgen]
impl SahiOptions {
    /// Creates a SAHI sliced-inference config.
    #[wasm_bindgen(constructor)]
    #[expect(
        clippy::too_many_arguments,
        reason = "wasm constructor preserves the flat JavaScript API for SAHI options"
    )]
    pub fn new(
        slice_width: u32,
        slice_height: u32,
        overlap_width_ratio: f32,
        overlap_height_ratio: f32,
        include_full_image: bool,
        merge_strategy: MergeStrategy,
        match_metric: MatchMetric,
        match_threshold: f32,
        class_agnostic: bool,
    ) -> Self {
        Self {
            slice_width,
            slice_height,
            overlap_width_ratio,
            overlap_height_ratio,
            include_full_image,
            merge_strategy,
            match_metric,
            match_threshold,
            class_agnostic,
        }
    }

    /// Returns the default SAHI config.
    #[wasm_bindgen(js_name = defaultConfig)]
    pub fn default_config() -> Self {
        Self::default()
    }
}
