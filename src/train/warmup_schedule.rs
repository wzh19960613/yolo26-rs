/// Learning-rate warmup used before the main per-epoch scheduler settles in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRateWarmup {
    /// Number of warmup epochs, matching Ultralytics `warmup_epochs`.
    pub epochs: f64,
    /// Initial global learning rate for the first training micro-batch.
    pub initial_lr: f64,
}

impl LearningRateWarmup {
    /// Creates a linear global learning-rate warmup.
    pub const fn linear(epochs: f64, initial_lr: f64) -> Self {
        Self { epochs, initial_lr }
    }

    /// Creates the global analogue of Ultralytics warmup.
    ///
    /// Ultralytics warms non-bias parameter groups from 0.0 and bias groups
    /// from `warmup_bias_lr`; this crate has one optimizer learning rate, so
    /// the global warmup follows the non-bias path.
    pub const fn ultralytics_global(epochs: f64) -> Self {
        Self::linear(epochs, 0.0)
    }

    /// Returns the effective number of warmup training micro-batches.
    pub fn steps(self, micro_batches_per_epoch: usize) -> crate::Result<usize> {
        self.validate()?;
        if self.epochs == 0.0 {
            return Ok(0);
        }
        let steps = (self.epochs * micro_batches_per_epoch as f64).round();
        if !steps.is_finite() {
            return Err(crate::Error::InvalidConfig(
                "learning-rate warmup steps must be finite".to_string(),
            ));
        }
        Ok((steps as usize).max(100))
    }

    /// Returns the learning rate for a zero-based training micro-batch.
    pub fn learning_rate(
        self,
        step: usize,
        warmup_steps: usize,
        target_lr: f64,
    ) -> crate::Result<f64> {
        self.validate()?;
        if !target_lr.is_finite() || target_lr < 0.0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate warmup target_lr must be finite and non-negative".to_string(),
            ));
        }
        if warmup_steps == 0 || step >= warmup_steps {
            return Ok(target_lr);
        }
        let fraction = step as f64 / warmup_steps as f64;
        Ok(self.initial_lr + (target_lr - self.initial_lr) * fraction)
    }

    /// Validates warmup parameters.
    pub fn validate(self) -> crate::Result<()> {
        if !self.epochs.is_finite() || self.epochs < 0.0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate warmup epochs must be finite and non-negative".to_string(),
            ));
        }
        if !self.initial_lr.is_finite() || self.initial_lr < 0.0 {
            return Err(crate::Error::InvalidConfig(
                "learning-rate warmup initial_lr must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Momentum warmup used alongside Ultralytics learning-rate warmup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumWarmup {
    /// Initial momentum or Adam beta1 value.
    pub initial_momentum: f64,
    /// Target momentum or Adam beta1 value after warmup.
    pub target_momentum: f64,
}

impl MomentumWarmup {
    /// Creates a linear momentum warmup.
    pub const fn linear(initial_momentum: f64, target_momentum: f64) -> Self {
        Self {
            initial_momentum,
            target_momentum,
        }
    }

    /// Returns momentum for a zero-based training micro-batch.
    pub fn momentum(self, step: usize, warmup_steps: usize) -> crate::Result<f64> {
        self.validate()?;
        if warmup_steps == 0 || step >= warmup_steps {
            return Ok(self.target_momentum);
        }
        let fraction = step as f64 / warmup_steps as f64;
        Ok(self.initial_momentum + (self.target_momentum - self.initial_momentum) * fraction)
    }

    /// Validates momentum warmup parameters.
    pub fn validate(self) -> crate::Result<()> {
        for (name, value) in [
            ("initial_momentum", self.initial_momentum),
            ("target_momentum", self.target_momentum),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(crate::Error::InvalidConfig(format!(
                    "momentum warmup {name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}
