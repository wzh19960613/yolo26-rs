/// Early stopping configuration for dataset training loops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStoppingConfig {
    /// Epochs to wait after the monitored metric stops improving.
    pub patience: usize,
    /// Minimum change required to count as an improvement.
    pub min_delta: f32,
}

impl EarlyStoppingConfig {
    /// Creates an early stopping config that monitors epoch mean training loss.
    pub const fn new(patience: usize) -> Self {
        Self {
            patience,
            min_delta: 0.0,
        }
    }

    pub(crate) fn validate(self) -> crate::Result<()> {
        if !self.min_delta.is_finite() || self.min_delta < 0.0 {
            return Err(crate::Error::InvalidConfig(
                "early stopping min_delta must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Final early stopping state reported by a training loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStoppingReport {
    /// Best one-based epoch index observed by the monitor.
    pub best_epoch: usize,
    /// Best monitored metric (epoch mean training loss, or validation fitness
    /// when the loop supplies one).
    pub best_loss: f32,
    /// Consecutive epochs without improvement at loop end.
    pub stale_epochs: usize,
    /// Whether early stopping ended the loop.
    pub stopped: bool,
}

/// Which direction counts as improvement for the monitored metric.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Direction {
    /// Not yet observed; fixed by the first update.
    Unset,
    /// Lower is better (training loss).
    Lower,
    /// Higher is better (validation fitness).
    Higher,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EarlyStoppingState {
    config: EarlyStoppingConfig,
    best_epoch: usize,
    best_score: f32,
    direction: Direction,
    stale_epochs: usize,
    stopped: bool,
}

impl EarlyStoppingState {
    pub(crate) fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            best_epoch: 0,
            best_score: f32::INFINITY,
            direction: Direction::Unset,
            stale_epochs: 0,
            stopped: false,
        }
    }

    /// Updates the monitor with one epoch's `score`. `higher_is_better` selects
    /// the improvement direction (fixed by the first observation); the monitor
    /// then compares all subsequent epochs in that same direction.
    pub(crate) fn update(
        &mut self,
        epoch: usize,
        score: f32,
        higher_is_better: bool,
    ) -> EarlyStoppingReport {
        if matches!(self.direction, Direction::Unset) {
            self.direction = if higher_is_better {
                Direction::Higher
            } else {
                Direction::Lower
            };
        }
        let first = self.best_epoch == 0;
        if first || self.improved(score) {
            self.best_epoch = epoch;
            self.best_score = score;
            self.stale_epochs = 0;
        } else {
            self.stale_epochs += 1;
        }
        self.stopped = self.config.patience > 0 && self.stale_epochs >= self.config.patience;
        self.report()
    }

    pub(crate) const fn report(self) -> EarlyStoppingReport {
        EarlyStoppingReport {
            best_epoch: self.best_epoch,
            best_loss: self.best_score,
            stale_epochs: self.stale_epochs,
            stopped: self.stopped,
        }
    }

    fn improved(self, score: f32) -> bool {
        if !score.is_finite() {
            return false;
        }
        match self.direction {
            Direction::Higher => score > self.best_score + self.config.min_delta,
            Direction::Lower => score < self.best_score - self.config.min_delta,
            Direction::Unset => true,
        }
    }
}

/// Steps the optional early-stopping monitor for one epoch.
///
/// When `validation_fitness` is `Some`, the monitor tracks validation fitness
/// (higher is better), matching the official trainer; otherwise it falls back to
/// epoch mean training loss (lower is better).
pub(crate) fn update_early_stopping(
    state: &mut Option<EarlyStoppingState>,
    epoch: usize,
    mean_loss: f32,
    validation_fitness: Option<f32>,
) -> bool {
    state
        .as_mut()
        .map(|state| match validation_fitness {
            Some(fitness) => state.update(epoch, fitness, true).stopped,
            None => state.update(epoch, mean_loss, false).stopped,
        })
        .unwrap_or(false)
}
