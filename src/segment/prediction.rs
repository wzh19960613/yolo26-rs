use crate::detect;

use super::Mask;

/// One instance-segmentation prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Object detection associated with this mask.
    pub detection: detect::Prediction,
    /// Binary mask in source image coordinates.
    pub mask: Mask,
}

impl Prediction {
    /// Returns this segmentation translated by `dx` and `dy`.
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        self.detection = self.detection.translated(dx, dy);
        self
    }
}
