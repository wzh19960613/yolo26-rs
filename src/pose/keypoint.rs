/// One pose keypoint in source image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    /// Keypoint x coordinate.
    pub x: f32,
    /// Keypoint y coordinate.
    pub y: f32,
    /// Optional keypoint visibility/confidence score.
    pub visibility: Option<f32>,
}
