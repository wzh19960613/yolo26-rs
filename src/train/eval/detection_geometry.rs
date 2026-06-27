//! Geometry context for the task-aligned assigner's in-GT containment check.
//!
//! Holds the letterboxed-pixel ground-truth boxes and pixel-space anchor
//! centers, and implements the official `select_candidates_in_gts` containment
//! test used by [`crate::train::eval::detection_assignment::resolve_detection_assignments`].

/// Geometry context for the in-GT containment check.
#[derive(Default)]
pub(crate) struct Geometry {
    /// `gt_boxes[(b, obj)]` letterboxed-pixel GT xyxy, if populated.
    pub(crate) gt_boxes: Vec<[f32; 4]>,
    /// `anchor_centers_pixel[a]` pixel-space anchor center (x, y).
    pub(crate) anchor_centers_pixel: Vec<(f32, f32)>,
    pub(crate) max_objects: usize,
}

impl Geometry {
    /// Official `select_candidates_in_gts`: returns whether anchor `anchor_idx`
    /// falls inside GT `(batch_idx, object_idx)`.
    pub(crate) fn anchor_in_gt(
        &self,
        batch_idx: usize,
        object_idx: usize,
        anchor_idx: usize,
    ) -> bool {
        if self.gt_boxes.is_empty() || self.anchor_centers_pixel.is_empty() {
            return true;
        }
        let Some(gt) = self
            .gt_boxes
            .get(batch_idx * self.max_objects + object_idx)
            .copied()
        else {
            return true;
        };
        let Some((cx, cy)) = self.anchor_centers_pixel.get(anchor_idx).copied() else {
            return true;
        };
        // Official `select_candidates_in_gts`: convert to xywh, and any GT whose
        // width or height is below the smallest stride (8px) gets expanded to
        // `stride_val` (16px) so that very thin/small GTs still cover at least
        // one anchor cell. Then test anchor-center containment.
        let (mut bw, mut bh) = (gt[2] - gt[0], gt[3] - gt[1]);
        let (cxg, cyg) = ((gt[0] + gt[2]) * 0.5, (gt[1] + gt[3]) * 0.5);
        const STRIDE_0: f32 = 8.0; // smallest stride
        const STRIDE_VAL: f32 = 16.0; // stride[1]
        if bw < STRIDE_0 {
            bw = STRIDE_VAL;
        }
        if bh < STRIDE_0 {
            bh = STRIDE_VAL;
        }
        let (x1, y1, x2, y2) = (
            cxg - bw * 0.5,
            cyg - bh * 0.5,
            cxg + bw * 0.5,
            cyg + bh * 0.5,
        );
        const EPS: f32 = 1e-9;
        cx - x1 > EPS && cy - y1 > EPS && x2 - cx > EPS && y2 - cy > EPS
    }
}
