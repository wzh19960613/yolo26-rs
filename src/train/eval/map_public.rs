use super::{MapAccumulator, MapReport};

/// Accumulates detection-style validation predictions and computes mAP.
///
/// This public wrapper lets task-specific evaluators such as YOLOE reuse the
/// crate's native mAP implementation without exposing the internal per-image
/// prediction representation.
#[derive(Default)]
pub struct DetectionMapAccumulator {
    pub(crate) inner: MapAccumulator,
}

/// Accumulates instance-mask validation predictions and computes mask mAP.
///
/// This uses the same AP integration as [`DetectionMapAccumulator`], but the
/// true-positive matrix is produced from mask IoU to match Ultralytics
/// `SegmentMetrics.seg`.
#[derive(Default)]
pub struct MaskMapAccumulator {
    pub(crate) inner: MapAccumulator,
}

impl DetectionMapAccumulator {
    /// Creates an empty detection mAP accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the final mAP report for all accumulated validation batches.
    pub fn finalize(&self) -> MapReport {
        self.inner.finalize()
    }
}

impl MaskMapAccumulator {
    /// Creates an empty mask mAP accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the final mask mAP report for all accumulated validation batches.
    pub fn finalize(&self) -> MapReport {
        self.inner.finalize()
    }
}
