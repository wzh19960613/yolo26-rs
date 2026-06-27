//! Instance-segmentation entry points for the WASM API.

use wasm_bindgen::prelude::*;

use crate::segment::{self, Prediction};
use crate::{FilterOption, Image, MaskOption};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque segmentation model.
#[wasm_bindgen(js_name = SegmentModel)]
pub struct WasmSegmentModel {
    inner: segment::Model,
}

#[wasm_bindgen(js_class = SegmentModel)]
impl WasmSegmentModel {
    /// Creates a segmentation model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_segment_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner = segment::Model::from_bytes_with(bytes, native)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs instance segmentation on an RGB pixel buffer.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let segs = self
            .inner
            .predict(&image, filter, mask)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmSegments::new(segs))
    }

    /// Runs instance segmentation on an RGBA pixel buffer (alpha discarded).
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height, filter, mask)
    }
}

/// One instance-segmentation prediction returned by the WASM API.
#[wasm_bindgen(js_name = Segment)]
#[derive(Debug, Clone)]
pub struct WasmSegment {
    seg: Prediction,
}

/// Indexed segmentation collection returned by the WASM API.
#[wasm_bindgen(js_name = Segments)]
#[derive(Debug, Clone)]
pub struct WasmSegments {
    segs: Vec<Prediction>,
}

impl WasmSegments {
    pub(super) fn new(segs: Vec<Prediction>) -> Self {
        Self { segs }
    }
}

#[wasm_bindgen]
impl WasmSegment {
    /// Returns the box left coordinate.
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.seg.detection.bbox.x_min
    }

    /// Returns the box top coordinate.
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.seg.detection.bbox.y_min
    }

    /// Returns the box width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f32 {
        self.seg.detection.bbox.width()
    }

    /// Returns the box height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f32 {
        self.seg.detection.bbox.height()
    }

    /// Returns the instance confidence.
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.seg.detection.confidence
    }

    /// Returns the numeric class id.
    #[wasm_bindgen(getter, js_name = classId)]
    pub fn class_id(&self) -> u32 {
        self.seg.detection.class_id
    }

    /// Returns the mask width in pixels.
    #[wasm_bindgen(getter, js_name = maskWidth)]
    pub fn mask_width(&self) -> u32 {
        self.seg.mask.width as u32
    }

    /// Returns the mask height in pixels.
    #[wasm_bindgen(getter, js_name = maskHeight)]
    pub fn mask_height(&self) -> u32 {
        self.seg.mask.height as u32
    }

    /// Returns the binary mask as a `Uint8Array` (1 where the pixel belongs to
    /// the instance, 0 otherwise), row-major `[height * width]`.
    #[wasm_bindgen(js_name = toMaskArray)]
    pub fn to_mask_array(&self) -> Vec<u8> {
        self.seg.mask.data()
    }
}

#[wasm_bindgen]
impl WasmSegments {
    /// Returns the number of segments.
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// Returns whether the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// Returns one segment by index.
    pub fn get(&self, index: usize) -> Result<WasmSegment, JsValue> {
        let seg = self
            .segs
            .get(index)
            .cloned()
            .ok_or_else(|| js_error(format!("segment index {index} is out of bounds")))?;
        Ok(WasmSegment { seg })
    }
}
