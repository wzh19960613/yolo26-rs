//! Browser-facing WASM bindings for the YOLO26 runtime.
//!
//! Each task root is its own submodule gated by the matching feature, exposing
//! an opaque `*Model` JS class plus typed result wrappers:
//!
//! - [`detect`] (always compiled with `wasm`) — `DetectModel`, SAHI options.
//! - [`segment`] (feature `segment`) — `SegmentModel`.
//! - [`semantic`] (feature `semantic`) — `SemanticModel`.
//! - [`classify`] (feature `classify`) — `ClassifyModel`.
//! - [`pose`] (feature `pose`) — `PoseModel`.
//! - [`obb`] (feature `obb`) — `ObbModel`.
//! - [`yoloe_visual`] (feature `yoloe-visual`) — `YoloeVisualModel`.
//! - [`yoloe_pf`] (feature `yoloe-pf`) — `YoloePromptFreeModel`.
//!
//! Shared infrastructure lives in [`config`] (the `Config` carrier, plus
//! `FilterOption`/`MaskOption` constructors) and [`pixel`] (alpha stripping).

use wasm_bindgen::prelude::*;

mod builders;
mod config;
pub mod detect;
mod labels;
mod options;
mod pixel;
pub use labels::class_names;

#[cfg(feature = "classify")]
pub mod classify;
#[cfg(feature = "obb")]
pub mod obb;
#[cfg(feature = "pose")]
pub mod pose;
#[cfg(feature = "segment")]
pub mod segment;
#[cfg(feature = "semantic")]
pub mod semantic;
#[cfg(feature = "yoloe-pf")]
pub mod yoloe_pf;
#[cfg(feature = "yoloe-visual")]
pub mod yoloe_visual;
#[cfg(feature = "yoloe-visual")]
mod yoloe_visual_helpers;

pub use config::WasmConfig;

/// Installs panic reporting hooks for browser console diagnostics.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

pub(super) fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
