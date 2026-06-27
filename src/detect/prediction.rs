use crate::bbox::BBox;

/// One object detection prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Detection bounding box in source image coordinates.
    pub bbox: BBox,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
    /// Numeric class id.
    pub class_id: u32,
}

impl Prediction {
    /// Returns this detection translated by `dx` and `dy`.
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        self.bbox = self.bbox.translate(dx, dy);
        self
    }
}
