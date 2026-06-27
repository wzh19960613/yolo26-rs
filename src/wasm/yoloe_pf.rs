//! YOLOE prompt-free LRPC entry points for the WASM API.
//!
//! Uses the built-in 4585-entry `LRPC_VOCAB` to construct the prompt-free
//! session automatically, so the JS caller only supplies weights + pixels.

use wasm_bindgen::prelude::*;

use crate::yoloe::prompt::session::Session;
use crate::yoloe::segment::Model as YoloeModel;
use crate::{FilterOption, Image, MaskOption};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;
use super::segment::WasmSegments;

/// Loads SafeTensors/`.pt` bytes into an opaque YOLOE prompt-free model.
#[wasm_bindgen(js_name = YoloePromptFreeModel)]
pub struct WasmYoloePromptFreeModel {
    inner: YoloeModel,
}

#[wasm_bindgen(js_class = YoloePromptFreeModel)]
impl WasmYoloePromptFreeModel {
    /// Creates a YOLOE prompt-free model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_yoloe_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner =
            YoloeModel::from_bytes_with(bytes, native).map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the number of classes in the built-in LRPC vocabulary.
    #[wasm_bindgen(js_name = vocabSize)]
    pub fn vocab_size() -> usize {
        crate::default_labels::LRPC_VOCAB.len()
    }

    /// Returns the display name of a prompt-free class id, or `null` if out of
    /// range. Class ids index the built-in 4585-entry `LRPC_VOCAB`.
    #[wasm_bindgen(js_name = vocabName)]
    pub fn vocab_name(class_id: u32) -> Option<String> {
        crate::default_labels::LRPC_VOCAB
            .get(class_id as usize)
            .map(|s| s.to_string())
    }

    /// Runs prompt-free LRPC segmentation on an RGB pixel buffer, using the
    /// built-in 4585-entry vocabulary as the class set.
    pub fn predict_prompt_free_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let session = Session::prompt_free_default().map_err(|err| js_error(err.to_string()))?;
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let segs = self
            .inner
            .predict_prompt_free(&image, &session, filter, mask)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmSegments::new(segs))
    }

    /// Runs prompt-free LRPC segmentation on an RGBA pixel buffer.
    pub fn predict_prompt_free_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_prompt_free_rgb(&rgb, width, height, filter, mask)
    }
}
