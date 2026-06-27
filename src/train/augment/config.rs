//! Augmentation configuration aligned with Ultralytics defaults.

/// Configuration for the native training-time augmentation pipeline.
///
/// Probabilities/gains mirror `ultralytics/cfg/default.yaml`. Rotate, shear and
/// perspective default to zero (as upstream) and are intentionally not modeled
/// because `candle` 0.10 has no affine `grid_sample` op; see `PROCESSING.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentConfig {
    /// HSV hue jitter fraction (`hsv_h`).
    pub hsv_h: f32,
    /// HSV saturation jitter fraction (`hsv_s`).
    pub hsv_s: f32,
    /// HSV value (brightness) jitter fraction (`hsv_v`).
    pub hsv_v: f32,
    /// Translation fraction of the canvas (`translate`).
    pub translate: f32,
    /// Scale gain fraction (`scale`).
    pub scale: f32,
    /// Vertical flip probability (`flipud`).
    pub flipud: f32,
    /// Horizontal flip probability (`fliplr`).
    pub fliplr: f32,
    /// Four-image mosaic probability (`mosaic`).
    pub mosaic: f32,
    /// Two-image mixup probability (`mixup`).
    pub mixup: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            hsv_h: 0.015,
            hsv_s: 0.7,
            hsv_v: 0.4,
            translate: 0.1,
            scale: 0.5,
            flipud: 0.0,
            fliplr: 0.5,
            mosaic: 1.0,
            mixup: 0.0,
        }
    }
}

impl AugmentConfig {
    /// Returns a config that disables every augmentation (identity pipeline).
    pub fn disabled() -> Self {
        Self {
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.0,
            translate: 0.0,
            scale: 0.0,
            flipud: 0.0,
            fliplr: 0.0,
            mosaic: 0.0,
            mixup: 0.0,
        }
    }

    /// Returns `true` when no augmentation can fire.
    pub fn is_identity(&self) -> bool {
        self.hsv_h == 0.0
            && self.hsv_s == 0.0
            && self.hsv_v == 0.0
            && self.translate == 0.0
            && self.scale == 0.0
            && self.flipud == 0.0
            && self.fliplr == 0.0
            && self.mosaic == 0.0
            && self.mixup == 0.0
    }
}
