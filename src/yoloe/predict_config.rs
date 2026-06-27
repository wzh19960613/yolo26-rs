/// YOLOE prediction options that differ from normal YOLO usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictConfig {
    /// Prompted YOLOE uses class-agnostic NMS by default.
    pub agnostic_nms: bool,
    /// Internal LRPC proposal threshold for prompt-free inference.
    ///
    /// Ultralytics uses `0.001` by default inside the prompt-free head; final
    /// user-visible confidence filtering is handled by the normal prediction
    /// options, not this internal candidate gate.
    pub lrpc_confidence_threshold: f32,
}

impl Default for PredictConfig {
    fn default() -> Self {
        Self {
            agnostic_nms: true,
            lrpc_confidence_threshold: 0.001,
        }
    }
}
