//! Image-classification entry points for the WASM API.

use wasm_bindgen::prelude::*;

use crate::Image;
use crate::classify::{self, Prediction};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque classification model.
#[wasm_bindgen(js_name = ClassifyModel)]
pub struct WasmClassifyModel {
    inner: classify::Model,
}

#[wasm_bindgen(js_class = ClassifyModel)]
impl WasmClassifyModel {
    /// Creates a classification model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_classify_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner = classify::Model::from_bytes_with(bytes, native)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs image classification on an RGB pixel buffer, returning every class
    /// score sorted by descending confidence.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<WasmClassifications, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let scores = self
            .inner
            .predict(&image)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmClassifications::new(scores))
    }

    /// Runs image classification on an RGBA pixel buffer (alpha discarded).
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<WasmClassifications, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height)
    }
}

/// One classification score returned by the WASM API.
#[wasm_bindgen(js_name = Classification)]
#[derive(Debug, Clone, Copy)]
pub struct WasmClassification {
    score: Prediction,
}

/// Indexed classification collection returned by the WASM API.
#[wasm_bindgen(js_name = Classifications)]
#[derive(Debug, Clone)]
pub struct WasmClassifications {
    scores: Vec<Prediction>,
}

impl WasmClassifications {
    pub(super) fn new(scores: Vec<Prediction>) -> Self {
        Self { scores }
    }
}

#[wasm_bindgen]
impl WasmClassification {
    /// Returns the numeric class id.
    #[wasm_bindgen(getter, js_name = classId)]
    pub fn class_id(&self) -> u32 {
        self.score.class_id
    }

    /// Returns the class probability in `[0, 1]`.
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.score.confidence
    }
}

#[wasm_bindgen]
impl WasmClassifications {
    /// Returns the number of class scores.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns whether the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns one class score by index.
    pub fn get(&self, index: usize) -> Result<WasmClassification, JsValue> {
        let score =
            self.scores.get(index).copied().ok_or_else(|| {
                js_error(format!("classification index {index} is out of bounds"))
            })?;
        Ok(WasmClassification { score })
    }

    /// Returns a flat `[class_id, confidence, ...]` vector, sorted by
    /// descending confidence.
    #[wasm_bindgen(js_name = toFlatArray)]
    pub fn to_flat_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.scores.len() * 2);
        for score in &self.scores {
            out.extend_from_slice(&[score.class_id as f32, score.confidence]);
        }
        out
    }
}
