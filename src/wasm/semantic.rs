//! Semantic-segmentation entry points for the WASM API.

use wasm_bindgen::prelude::*;

use crate::semantic::{self, Prediction};
use crate::{Image, MaskOption};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque semantic model.
#[wasm_bindgen(js_name = SemanticModel)]
pub struct WasmSemanticModel {
    inner: semantic::Model,
}

#[wasm_bindgen(js_class = SemanticModel)]
impl WasmSemanticModel {
    /// Creates a semantic model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_semantic_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner = semantic::Model::from_bytes_with(bytes, native)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs semantic segmentation on an RGB pixel buffer.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        mask: &MaskOption,
    ) -> Result<WasmSemanticMap, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let map = self
            .inner
            .predict(&image, mask)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmSemanticMap { map })
    }

    /// Runs semantic segmentation on an RGBA pixel buffer (alpha discarded).
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        mask: &MaskOption,
    ) -> Result<WasmSemanticMap, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height, mask)
    }
}

/// Semantic-segmentation argmax map returned by the WASM API.
#[wasm_bindgen(js_name = SemanticMap)]
#[derive(Debug, Clone)]
pub struct WasmSemanticMap {
    map: Prediction,
}

#[wasm_bindgen]
impl WasmSemanticMap {
    /// Returns the map width in pixels.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.map.width as u32
    }

    /// Returns the map height in pixels.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.map.height as u32
    }

    /// Returns the number of classes the model discriminates.
    #[wasm_bindgen(getter, js_name = classCount)]
    pub fn class_count(&self) -> usize {
        self.map.classes
    }

    /// Returns the argmax class id at a single pixel.
    pub fn class_at(&self, x: usize, y: usize) -> u32 {
        self.map.class_id(x, y)
    }

    /// Returns the full argmax class map as `Uint32Array`, row-major
    /// `[height * width]` class ids.
    #[wasm_bindgen(js_name = toClassArray)]
    pub fn to_class_array(&self) -> Vec<u32> {
        self.map.class_ids()
    }
}
