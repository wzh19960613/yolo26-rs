//! wasm-bindgen constructors for the shared prediction/mask options.

use wasm_bindgen::prelude::*;

use crate::{FilterOption, MaskOption};

#[wasm_bindgen]
impl FilterOption {
    /// Creates prediction options.
    #[wasm_bindgen(constructor)]
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            confidence_threshold,
            ..Self::default()
        }
    }

    /// Returns the default prediction options.
    #[wasm_bindgen(js_name = defaultConfig)]
    pub fn default_config() -> Self {
        Self::default()
    }

    /// Returns a copy restricted to the given class ids (empty keeps all).
    #[wasm_bindgen(js_name = withClasses)]
    pub fn with_classes(&self, classes: Vec<u32>) -> Self {
        let mut next = self.clone();
        next.class_filter = classes;
        next
    }
}

#[wasm_bindgen]
impl MaskOption {
    /// Creates a mask-resolution option (`high_resolution=true` returns masks at
    /// source-image resolution).
    #[wasm_bindgen(constructor)]
    pub fn new(high_resolution: bool) -> Self {
        Self { high_resolution }
    }

    /// Returns the default mask option.
    #[wasm_bindgen(js_name = defaultConfig)]
    pub fn default_config() -> Self {
        Self::default()
    }
}
