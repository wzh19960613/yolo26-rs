/// Axis-aligned bounding box in image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Left coordinate.
    pub x_min: f32,
    /// Top coordinate.
    pub y_min: f32,
    /// Right coordinate.
    pub x_max: f32,
    /// Bottom coordinate.
    pub y_max: f32,
}

impl BBox {
    /// Creates a bounding box from `(x_min, y_min, x_max, y_max)`.
    pub fn from_xyxy(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Returns the non-negative box width.
    pub fn width(self) -> f32 {
        (self.x_max - self.x_min).max(0.0)
    }

    /// Returns the non-negative box height.
    pub fn height(self) -> f32 {
        (self.y_max - self.y_min).max(0.0)
    }

    /// Returns the box area.
    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    /// Returns this box translated by `dx` and `dy`.
    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self {
            x_min: self.x_min + dx,
            y_min: self.y_min + dy,
            x_max: self.x_max + dx,
            y_max: self.y_max + dy,
        }
    }

    /// Clamps this box to an image with the given dimensions.
    pub fn clamp(self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            x_min: self.x_min.clamp(0.0, w),
            y_min: self.y_min.clamp(0.0, h),
            x_max: self.x_max.clamp(0.0, w),
            y_max: self.y_max.clamp(0.0, h),
        }
    }

    /// Returns the intersection area with another box.
    pub fn intersection(self, other: Self) -> f32 {
        let x_min = self.x_min.max(other.x_min);
        let y_min = self.y_min.max(other.y_min);
        let x_max = self.x_max.min(other.x_max);
        let y_max = self.y_max.min(other.y_max);
        Self::from_xyxy(x_min, y_min, x_max, y_max).area()
    }

    /// Returns intersection-over-union with another box.
    pub fn iou(self, other: Self) -> f32 {
        let inter = self.intersection(other);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    /// Returns intersection-over-smaller-area with another box.
    pub fn ios(self, other: Self) -> f32 {
        let inter = self.intersection(other);
        let smaller = self.area().min(other.area());
        if smaller <= 0.0 { 0.0 } else { inter / smaller }
    }
}
