/// One image-classification score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Numeric class id.
    pub class_id: u32,
    /// Class probability in `[0, 1]`.
    pub confidence: f32,
}
