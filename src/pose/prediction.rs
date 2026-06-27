use crate::bbox::BBox;

use super::Keypoint;

/// One pose/keypoint prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Detection bounding box in source image coordinates.
    pub bbox: BBox,
    /// Detection confidence score in `[0, 1]`.
    pub confidence: f32,
    /// Numeric class id.
    pub class_id: u32,
    /// Decoded keypoints in source image coordinates.
    pub keypoints: Vec<Keypoint>,
}

impl Prediction {
    /// Returns this pose detection translated by `dx` and `dy`.
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        self.bbox = self.bbox.translate(dx, dy);
        for keypoint in &mut self.keypoints {
            keypoint.x += dx;
            keypoint.y += dy;
        }
        self
    }
}
