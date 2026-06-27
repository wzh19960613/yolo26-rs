use crate::bbox::BBox as AxisAlignedBBox;

/// Oriented bounding box in source image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Center x coordinate.
    pub center_x: f32,
    /// Center y coordinate.
    pub center_y: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
    /// Rotation angle in radians.
    pub angle: f32,
}

impl BBox {
    /// Returns the enclosing axis-aligned bounding box.
    pub fn axis_aligned_bbox(self) -> AxisAlignedBBox {
        let (sin, cos) = self.angle.sin_cos();
        let half_w = self.width * 0.5;
        let half_h = self.height * 0.5;
        let corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ];

        let mut x_min = f32::INFINITY;
        let mut y_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        for (x, y) in corners {
            let px = self.center_x + x * cos - y * sin;
            let py = self.center_y + x * sin + y * cos;
            x_min = x_min.min(px);
            y_min = y_min.min(py);
            x_max = x_max.max(px);
            y_max = y_max.max(py);
        }

        AxisAlignedBBox::from_xyxy(x_min, y_min, x_max, y_max)
    }

    /// Returns this oriented box translated by `dx` and `dy`.
    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self {
            center_x: self.center_x + dx,
            center_y: self.center_y + dy,
            ..self
        }
    }
}
