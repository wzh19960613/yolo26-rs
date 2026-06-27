/// Width and height of a model input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    /// Input tensor width in pixels.
    pub width: usize,
    /// Input tensor height in pixels.
    pub height: usize,
}

impl ImageSize {
    /// Creates a square input size.
    pub const fn square(size: usize) -> Self {
        Self {
            width: size,
            height: size,
        }
    }

    /// Creates an input size from width and height.
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub(crate) const fn square_from_max(self) -> Self {
        let max_side = match self.width > self.height {
            true => self.width,
            false => self.height,
        };
        Self::square(max_side)
    }

    /// Returns dimensions snapped up to the nearest multiple of 32 (minimum 32).
    pub const fn snapped(self) -> Self {
        Self {
            width: snap_multiple_of_32(self.width),
            height: snap_multiple_of_32(self.height),
        }
    }
}

const fn snap_multiple_of_32(n: usize) -> usize {
    match n {
        0 => 32,
        n if n.is_multiple_of(32) => n,
        n => n + 32 - n % 32,
    }
}
