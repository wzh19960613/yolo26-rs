/// Learning-rate schedule used by a training loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRateSchedule {
    /// Initial learning rate, matching Ultralytics `lr0`.
    pub initial_lr: f64,
    /// Final learning-rate fraction, matching Ultralytics `lrf`.
    pub final_lr_fraction: f64,
    /// Whether to use the Ultralytics cosine one-cycle scheduler.
    pub cosine: bool,
}

impl LearningRateSchedule {
    /// Creates the default Ultralytics linear scheduler.
    pub const fn linear(initial_lr: f64, final_lr_fraction: f64) -> Self {
        Self {
            initial_lr,
            final_lr_fraction,
            cosine: false,
        }
    }

    /// Creates the Ultralytics cosine one-cycle scheduler.
    pub const fn cosine(initial_lr: f64, final_lr_fraction: f64) -> Self {
        Self {
            initial_lr,
            final_lr_fraction,
            cosine: true,
        }
    }

    /// Returns the learning rate for a zero-based epoch index.
    pub fn learning_rate(self, epoch: usize, epochs: usize) -> crate::Result<f64> {
        self.validate()?;
        if epochs == 0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate schedule requires epochs > 0".to_string(),
            ));
        }
        let epoch = epoch as f64;
        let epochs = epochs as f64;
        let factor = if self.cosine {
            let ramp = (1.0 - (epoch * std::f64::consts::PI / epochs).cos()) / 2.0;
            ramp.max(0.0) * (self.final_lr_fraction - 1.0) + 1.0
        } else {
            (1.0 - epoch / epochs).max(0.0) * (1.0 - self.final_lr_fraction)
                + self.final_lr_fraction
        };
        Ok(self.initial_lr * factor)
    }

    /// Validates schedule parameters.
    pub fn validate(self) -> crate::Result<()> {
        if !self.initial_lr.is_finite() || self.initial_lr <= 0.0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate schedule lr0 must be finite and positive".to_string(),
            ));
        }
        if !self.final_lr_fraction.is_finite() || self.final_lr_fraction < 0.0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate schedule lrf must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}
