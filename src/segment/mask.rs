/// Instance segmentation mask with soft logits.
///
/// Stores per-pixel logits at the model's native resolution.
/// Positive logits (> 0) indicate the pixel belongs to the instance.
/// Use [`Mask::data`] to get the binary mask, or [`Mask::resize_checked`] to resample.
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    /// Mask width in pixels.
    pub width: u16,
    /// Mask height in pixels.
    pub height: u16,
    /// Row-major logits, one float per pixel.
    pub logits: Vec<f32>,
}

impl Mask {
    /// Creates a mask from logits.
    pub fn new(width: u16, height: u16, logits: Vec<f32>) -> crate::Result<Self> {
        let expected = width as usize * height as usize;
        if logits.len() != expected {
            return Err(crate::Error::InvalidTensor(format!(
                "expected {expected} logits for {width}x{height}, got {}",
                logits.len()
            )));
        }
        Ok(Self {
            width,
            height,
            logits,
        })
    }

    /// Returns the binary mask data: 1 where logit > 0, else 0.
    pub fn data(&self) -> Vec<u8> {
        self.logits.iter().map(|&v| u8::from(v > 0.0)).collect()
    }

    /// Returns the binary value at a single pixel.
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.logits[y * self.width as usize + x] > 0.0
    }

    /// Resizes the mask using bilinear interpolation on logits, then thresholds.
    ///
    /// **Deprecated:** this method panics on tensor failure. Use
    /// [`Mask::resize_checked`] instead, which returns a [`crate::Result`].
    /// The panic variant is retained only to keep the historical `-> Self`
    /// signature stable for existing callers.
    #[deprecated(
        since = "0.2.0",
        note = "use `resize_checked` which returns `Result`; this variant panics on failure"
    )]
    pub fn resize(&self, target_width: u16, target_height: u16) -> Self {
        self.resize_checked(target_width, target_height)
            .expect("mask resize: invariant established by Mask::new makes failure unreachable")
    }

    /// Resizes the mask using bilinear interpolation on logits, returning a
    /// fallible result.
    ///
    /// Given the invariant established by [`Mask::new`] (logits length matches
    /// `width * height`), the underlying tensor calls cannot fail; this method
    /// still propagates any candle error via [`crate::Result`] rather than
    /// panicking, so callers never need to reason about a panic.
    pub fn resize_checked(&self, target_width: u16, target_height: u16) -> crate::Result<Self> {
        if target_width == self.width && target_height == self.height {
            return Ok(self.clone());
        }
        let device = candle_core::Device::Cpu;
        let src = candle_core::Tensor::from_vec(
            self.logits.clone(),
            (1, 1, self.height as usize, self.width as usize),
            &device,
        )
        .map_err(crate::Error::from)?;
        let dst = src
            .upsample_bilinear2d(target_height as usize, target_width as usize, true)
            .map_err(crate::Error::from)?;
        let logits = dst
            .flatten_all()
            .map_err(crate::Error::from)?
            .to_vec1::<f32>()
            .map_err(crate::Error::from)?;
        Ok(Self {
            width: target_width,
            height: target_height,
            logits,
        })
    }
}
