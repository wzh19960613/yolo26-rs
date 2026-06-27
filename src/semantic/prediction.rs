/// Dense semantic-segmentation logits map.
///
/// Stores per-class logits at the model's native resolution.
/// Use [`Prediction::class_ids`] to get the argmax class map,
/// or [`Prediction::resize_checked`] to resample at a different resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Mask width in pixels.
    pub width: u16,
    /// Mask height in pixels.
    pub height: u16,
    /// Number of classes.
    pub classes: usize,
    /// Row-major per-class logits: `[classes][height * width]`.
    pub logits: Vec<f32>,
}

impl Prediction {
    /// Creates a semantic mask from per-class logits.
    pub fn new(width: u16, height: u16, classes: usize, logits: Vec<f32>) -> crate::Result<Self> {
        let expected = classes * width as usize * height as usize;
        if logits.len() != expected {
            return Err(crate::Error::InvalidTensor(format!(
                "expected {expected} logits for {classes}x{height}x{width}, got {}",
                logits.len()
            )));
        }
        Ok(Self {
            width,
            height,
            classes,
            logits,
        })
    }

    /// Returns the argmax class id at a single pixel.
    pub fn class_id(&self, x: usize, y: usize) -> u32 {
        let hw = self.width as usize * self.height as usize;
        let pixel = y * self.width as usize + x;
        let mut best = 0u32;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.classes {
            let v = self.logits[c * hw + pixel];
            if v > best_score {
                best_score = v;
                best = c as u32;
            }
        }
        best
    }

    /// Returns the full argmax class map as `[height * width]` class ids.
    pub fn class_ids(&self) -> Vec<u32> {
        let hw = self.width as usize * self.height as usize;
        let mut out = vec![0u32; hw];
        for (pixel, out_pixel) in out.iter_mut().enumerate().take(hw) {
            let mut best = 0u32;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..self.classes {
                let v = self.logits[c * hw + pixel];
                if v > best_score {
                    best_score = v;
                    best = c as u32;
                }
            }
            *out_pixel = best;
        }
        out
    }

    /// Resizes the mask using bilinear interpolation on logits, then argmax.
    ///
    /// **Deprecated:** this method panics on tensor failure. Use
    /// [`Prediction::resize_checked`] instead, which returns a [`crate::Result`].
    /// The panic variant is retained only to keep the historical `-> Self`
    /// signature stable for existing callers.
    #[deprecated(
        since = "0.2.0",
        note = "use `resize_checked` which returns `Result`; this variant panics on failure"
    )]
    pub fn resize(&self, target_width: u16, target_height: u16) -> Self {
        self.resize_checked(target_width, target_height)
            .expect("semantic mask resize: invariant established by Prediction::new makes failure unreachable")
    }

    /// Resizes the mask using bilinear interpolation on logits, returning a
    /// fallible result.
    ///
    /// Given the invariant established by [`Prediction::new`] (logits length
    /// matches `classes * width * height`), the underlying tensor calls cannot
    /// fail; this method still propagates any candle error via
    /// [`crate::Result`] rather than panicking.
    pub fn resize_checked(&self, target_width: u16, target_height: u16) -> crate::Result<Self> {
        if target_width == self.width && target_height == self.height {
            return Ok(self.clone());
        }
        let device = candle_core::Device::Cpu;
        let src = candle_core::Tensor::from_vec(
            self.logits.clone(),
            (1, self.classes, self.height as usize, self.width as usize),
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
            classes: self.classes,
            logits,
        })
    }
}
