//! Oriented bounding-box entry points for the WASM API.

use wasm_bindgen::prelude::*;

use crate::obb::{self, Prediction};
use crate::{FilterOption, Image};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque OBB model.
#[wasm_bindgen(js_name = ObbModel)]
pub struct WasmObbModel {
    inner: obb::Model,
}

#[wasm_bindgen(js_class = ObbModel)]
impl WasmObbModel {
    /// Creates an OBB model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_obb_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner =
            obb::Model::from_bytes_with(bytes, native).map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs oriented bounding-box detection on an RGB pixel buffer.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
    ) -> Result<WasmObbs, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let obbs = self
            .inner
            .predict(&image, filter)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmObbs::new(obbs))
    }

    /// Runs oriented bounding-box detection on an RGBA pixel buffer.
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
    ) -> Result<WasmObbs, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height, filter)
    }
}

/// One oriented bounding-box prediction returned by the WASM API.
#[wasm_bindgen(js_name = Obb)]
#[derive(Debug, Clone, Copy)]
pub struct WasmObb {
    obb: Prediction,
}

/// Indexed OBB collection returned by the WASM API.
#[wasm_bindgen(js_name = Obbs)]
#[derive(Debug, Clone)]
pub struct WasmObbs {
    obbs: Vec<Prediction>,
}

impl WasmObbs {
    pub(super) fn new(obbs: Vec<Prediction>) -> Self {
        Self { obbs }
    }
}

#[wasm_bindgen]
impl WasmObb {
    /// Returns the center x coordinate.
    #[wasm_bindgen(getter, js_name = centerX)]
    pub fn center_x(&self) -> f32 {
        self.obb.bbox.center_x
    }

    /// Returns the center y coordinate.
    #[wasm_bindgen(getter, js_name = centerY)]
    pub fn center_y(&self) -> f32 {
        self.obb.bbox.center_y
    }

    /// Returns the box width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f32 {
        self.obb.bbox.width
    }

    /// Returns the box height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f32 {
        self.obb.bbox.height
    }

    /// Returns the rotation angle in radians.
    #[wasm_bindgen(getter)]
    pub fn angle(&self) -> f32 {
        self.obb.bbox.angle
    }

    /// Returns the detection confidence.
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.obb.confidence
    }

    /// Returns the numeric class id.
    #[wasm_bindgen(getter, js_name = classId)]
    pub fn class_id(&self) -> u32 {
        self.obb.class_id
    }
}

#[wasm_bindgen]
impl WasmObbs {
    /// Returns the number of oriented boxes.
    pub fn len(&self) -> usize {
        self.obbs.len()
    }

    /// Returns whether the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.obbs.is_empty()
    }

    /// Returns one oriented box by index.
    pub fn get(&self, index: usize) -> Result<WasmObb, JsValue> {
        let obb = self
            .obbs
            .get(index)
            .copied()
            .ok_or_else(|| js_error(format!("obb index {index} is out of bounds")))?;
        Ok(WasmObb { obb })
    }

    /// Returns a flat `[centerX, centerY, width, height, angle, confidence,
    /// classId, ...]` vector.
    #[wasm_bindgen(js_name = toFlatArray)]
    pub fn to_flat_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.obbs.len() * 7);
        for obb in &self.obbs {
            out.extend_from_slice(&[
                obb.bbox.center_x,
                obb.bbox.center_y,
                obb.bbox.width,
                obb.bbox.height,
                obb.bbox.angle,
                obb.confidence,
                obb.class_id as f32,
            ]);
        }
        out
    }
}
