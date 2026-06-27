/// Zero-copy view over one row of YOLO26 end-to-end head output.
///
/// All detection-like tasks share a common row prefix:
/// `[..., ..., ..., ..., confidence, class_id]` where confidence is at index 4
/// and class_id at index 5.  The preceding fields (bbox coordinates) and any
/// trailing fields (mask coefficients, keypoints, angle) are task-specific and
/// accessed via [`Index`](std::ops::Index) or [`as_slice`](OutputViewer::as_slice).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OutputViewer<'a>(&'a [f32]);

impl<'a> OutputViewer<'a> {
    const DETECT_COLS: usize = 6;
    const OBB_COLS: usize = 7;

    /// Creates a viewer for a detect/segment/pose row (6+ columns).
    pub(crate) fn for_detect(data: &'a [f32], row: usize) -> Option<Self> {
        Self::new(data, row, Self::DETECT_COLS)
    }

    /// Creates a viewer for an OBB row (7 columns).
    pub(crate) fn for_obb(data: &'a [f32], row: usize) -> Option<Self> {
        Self::new(data, row, Self::OBB_COLS)
    }

    /// Creates a viewer for a row with arbitrary column count.
    pub(crate) fn for_dynamic(data: &'a [f32], row: usize, cols: usize) -> Option<Self> {
        Self::new(data, row, cols)
    }

    fn new(data: &'a [f32], row: usize, cols: usize) -> Option<Self> {
        let start = row.checked_mul(cols)?;
        Some(Self(data.get(start..start.checked_add(cols)?)?))
    }
}

impl OutputViewer<'_> {
    /// Left bounding-box coordinate (xyxy tasks, index 0).
    pub(crate) const fn x1(&self) -> f32 {
        self.0[0]
    }

    /// Top bounding-box coordinate (xyxy tasks, index 1).
    pub(crate) const fn y1(&self) -> f32 {
        self.0[1]
    }

    /// Right bounding-box coordinate (xyxy tasks, index 2).
    pub(crate) const fn x2(&self) -> f32 {
        self.0[2]
    }

    /// Bottom bounding-box coordinate (xyxy tasks, index 3).
    pub(crate) const fn y2(&self) -> f32 {
        self.0[3]
    }

    /// Box center x (OBB tasks, index 0).
    pub(crate) const fn cx(&self) -> f32 {
        self.0[0]
    }

    /// Box center y (OBB tasks, index 1).
    pub(crate) const fn cy(&self) -> f32 {
        self.0[1]
    }

    /// Box width (OBB tasks, index 2).
    pub(crate) const fn w(&self) -> f32 {
        self.0[2]
    }

    /// Box height (OBB tasks, index 3).
    pub(crate) const fn h(&self) -> f32 {
        self.0[3]
    }

    /// Rotation angle in radians (OBB tasks, index 6).
    ///
    /// # Safety
    /// The underlying row must contain at least 7 elements.
    pub(crate) unsafe fn angle(&self) -> f32 {
        unsafe { *self.0.get_unchecked(6) }
    }

    /// Detection confidence score (index 4).
    pub(crate) const fn confidence(&self) -> f32 {
        self.0[4]
    }

    /// Predicted class id (index 5).
    pub(crate) fn class_id(&self) -> u32 {
        self.0[5].round().max(0.0) as u32
    }

    /// Returns `(confidence, class_id)` if the row passes the filter.
    pub(crate) fn check(&self, filter: &crate::FilterOption) -> Option<(f32, u32)> {
        filter.check(self.confidence(), self.class_id())
    }

    /// The full row as a raw slice.
    pub(crate) fn as_slice(&self) -> &[f32] {
        self.0
    }
}

impl std::ops::Index<usize> for OutputViewer<'_> {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

use candle_core::{DType, Tensor};

/// Flattens a `[1, N, cols]` or `[N, cols]` tensor into `(row_count, Vec<f32>)`.
pub(crate) fn flattened_rows(output: &Tensor, cols: usize) -> crate::Result<(usize, Vec<f32>)> {
    match output.dims() {
        [1, rows, got_cols] if *got_cols == cols => Ok((
            *rows,
            output
                .squeeze(0)?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
        )),
        [rows, got_cols] if *got_cols == cols => Ok((
            *rows,
            output
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
        )),
        dims => Err(crate::Error::InvalidTensor(format!(
            "expected [1, N, {cols}] or [N, {cols}], got {dims:?}"
        ))),
    }
}
