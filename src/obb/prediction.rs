use super::BBox;

/// One oriented bounding-box prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Oriented bounding box in source image coordinates.
    pub bbox: BBox,
    /// Detection confidence score in `[0, 1]`.
    pub confidence: f32,
    /// Numeric class id.
    pub class_id: u32,
}

impl Prediction {
    /// Returns this oriented detection translated by `dx` and `dy`.
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        self.bbox = self.bbox.translate(dx, dy);
        self
    }
}
