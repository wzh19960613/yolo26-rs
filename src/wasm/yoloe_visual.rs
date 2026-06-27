//! YOLOE visual-prompt entry points for the WASM API.
//!
//! Two workflows:
//! - **Intra-image** (`predict_visual_boxes_*` / `predict_visual_masks_*`):
//!   prompt and recognition on the same image in one pass.
//! - **Cross-image** (`encode_ref_image_*` → `predict_cross_image_*`): encode
//!   prompts on a reference image into a reusable embedding, then recognize on
//!   any target image. The encoded session lives inside the model handle.

use wasm_bindgen::prelude::*;

use crate::yoloe::prompt::session::Session;
use crate::yoloe::segment::Model as YoloeModel;
use crate::yoloe::visuals::VisualSource;
use crate::{FilterOption, Image, MaskOption};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;
use super::segment::WasmSegments;
use super::yoloe_visual_helpers::{build_box_prompts, build_mask_prompts};

/// Loads SafeTensors/`.pt` bytes into an opaque YOLOE visual-prompt model.
#[wasm_bindgen(js_name = YoloeVisualModel)]
pub struct WasmYoloeVisualModel {
    inner: YoloeModel,
    /// Encoded session from a reference image (cross-image mode). `None` until
    /// `encode_ref_image_*` is called; consumed by `predict_cross_image_*`.
    encoded_session: Option<Session>,
}

#[wasm_bindgen(js_class = YoloeVisualModel)]
impl WasmYoloeVisualModel {
    /// Creates a YOLOE visual model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_yoloe_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner =
            YoloeModel::from_bytes_with(bytes, native).map_err(|err| js_error(err.to_string()))?;
        Ok(Self {
            inner,
            encoded_session: None,
        })
    }

    // --- intra-image ---

    /// Intra-image box-prompt segmentation (RGB).
    pub fn predict_visual_boxes_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        box_prompts: &[f32],
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let prompts = build_box_prompts(box_prompts)?;
        let session = Session::visual(prompts.clone()).map_err(|e| js_error(e.to_string()))?;
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|e| js_error(e.to_string()))?;
        let segs = self
            .inner
            .predict_visual_prompts(
                &image,
                &prompts,
                VisualSource::Boxes,
                &session,
                filter,
                mask,
            )
            .map_err(|e| js_error(e.to_string()))?;
        Ok(WasmSegments::new(segs))
    }

    /// Intra-image box-prompt segmentation (RGBA, alpha discarded).
    pub fn predict_visual_boxes_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        box_prompts: &[f32],
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_visual_boxes_rgb(&rgb, width, height, box_prompts, filter, mask)
    }

    /// Intra-image mask-prompt segmentation (RGB).
    pub fn predict_visual_masks_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        mask_data: &[u8],
        mask_w: u32,
        mask_h: u32,
        n_prompts: usize,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let (prompts, mask_tensor) = build_mask_prompts(mask_data, mask_w, mask_h, n_prompts)?;
        let session = Session::visual(prompts.clone()).map_err(|e| js_error(e.to_string()))?;
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|e| js_error(e.to_string()))?;
        let segs = self
            .inner
            .predict_visual_prompts(
                &image,
                &prompts,
                VisualSource::Masks(&mask_tensor),
                &session,
                filter,
                mask,
            )
            .map_err(|e| js_error(e.to_string()))?;
        Ok(WasmSegments::new(segs))
    }

    /// Intra-image mask-prompt segmentation (RGBA, alpha discarded).
    pub fn predict_visual_masks_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        mask_data: &[u8],
        mask_w: u32,
        mask_h: u32,
        n_prompts: usize,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_visual_masks_rgb(
            &rgb, width, height, mask_data, mask_w, mask_h, n_prompts, filter, mask,
        )
    }

    // --- cross-image ---

    /// Encodes box prompts on a reference image into a reusable session (RGB).
    /// After this, call `predict_cross_image_*` on any target image.
    pub fn encode_ref_image_boxes_rgb(
        &mut self,
        ref_pixels: &[u8],
        ref_w: u32,
        ref_h: u32,
        box_prompts: &[f32],
    ) -> Result<usize, JsValue> {
        let prompts = build_box_prompts(box_prompts)?;
        let image =
            Image::new(ref_w, ref_h, ref_pixels.to_vec()).map_err(|e| js_error(e.to_string()))?;
        let table = self
            .inner
            .encode_visual_prompts(&image, &prompts, VisualSource::Boxes)
            .map_err(|e| js_error(e.to_string()))?;
        let n = table.class_count();
        let session = Session::text_with_embeddings(table).map_err(|e| js_error(e.to_string()))?;
        self.encoded_session = Some(session);
        Ok(n)
    }

    /// Encodes box prompts on a reference image (RGBA, alpha discarded).
    pub fn encode_ref_image_boxes_rgba(
        &mut self,
        ref_pixels: &[u8],
        ref_w: u32,
        ref_h: u32,
        box_prompts: &[f32],
    ) -> Result<usize, JsValue> {
        let rgb = strip_alpha(ref_pixels, ref_w, ref_h);
        self.encode_ref_image_boxes_rgb(&rgb, ref_w, ref_h, box_prompts)
    }

    /// Encodes mask prompts on a reference image into a reusable session (RGB).
    pub fn encode_ref_image_masks_rgb(
        &mut self,
        ref_pixels: &[u8],
        ref_w: u32,
        ref_h: u32,
        mask_data: &[u8],
        mask_w: u32,
        mask_h: u32,
        n_prompts: usize,
    ) -> Result<usize, JsValue> {
        let (prompts, mask_tensor) = build_mask_prompts(mask_data, mask_w, mask_h, n_prompts)?;
        let image =
            Image::new(ref_w, ref_h, ref_pixels.to_vec()).map_err(|e| js_error(e.to_string()))?;
        let table = self
            .inner
            .encode_visual_prompts(&image, &prompts, VisualSource::Masks(&mask_tensor))
            .map_err(|e| js_error(e.to_string()))?;
        let n = table.class_count();
        let session = Session::text_with_embeddings(table).map_err(|e| js_error(e.to_string()))?;
        self.encoded_session = Some(session);
        Ok(n)
    }

    /// Encodes mask prompts on a reference image (RGBA, alpha discarded).
    pub fn encode_ref_image_masks_rgba(
        &mut self,
        ref_pixels: &[u8],
        ref_w: u32,
        ref_h: u32,
        mask_data: &[u8],
        mask_w: u32,
        mask_h: u32,
        n_prompts: usize,
    ) -> Result<usize, JsValue> {
        let rgb = strip_alpha(ref_pixels, ref_w, ref_h);
        self.encode_ref_image_masks_rgb(&rgb, ref_w, ref_h, mask_data, mask_w, mask_h, n_prompts)
    }

    /// Cross-image prediction on a target image using the encoded session
    /// from a prior `encode_ref_image_*` call (RGB).
    pub fn predict_cross_image_rgb(
        &self,
        target_pixels: &[u8],
        target_w: u32,
        target_h: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let session = self
            .encoded_session
            .as_ref()
            .ok_or_else(|| js_error("call encode_ref_image_* on a reference image first"))?;
        let image = Image::new(target_w, target_h, target_pixels.to_vec())
            .map_err(|e| js_error(e.to_string()))?;
        let segs = self
            .inner
            .predict(&image, session, filter, mask)
            .map_err(|e| js_error(e.to_string()))?;
        Ok(WasmSegments::new(segs))
    }

    /// Cross-image prediction (RGBA, alpha discarded).
    pub fn predict_cross_image_rgba(
        &self,
        target_pixels: &[u8],
        target_w: u32,
        target_h: u32,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<WasmSegments, JsValue> {
        let rgb = strip_alpha(target_pixels, target_w, target_h);
        self.predict_cross_image_rgb(&rgb, target_w, target_h, filter, mask)
    }
}
