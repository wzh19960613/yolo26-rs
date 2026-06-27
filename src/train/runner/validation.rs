use super::RunnerConfig;
use std::time::Instant;

impl RunnerConfig {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        self.validate_sizes()?;
        self.validate_filters_and_checkpoints()?;
        self.validate_schedule()?;
        Ok(())
    }

    fn validate_sizes(&self) -> crate::Result<()> {
        if self.epochs == 0 {
            return invalid("training epochs must be greater than zero");
        }
        if self.batch_size == 0 {
            return invalid("training batch_size must be greater than zero");
        }
        if matches!(self.steps_per_epoch, Some(0)) {
            return invalid("training steps_per_epoch must be greater than zero");
        }
        if self.accumulate_steps == 0 {
            return invalid("training accumulate_steps must be greater than zero");
        }
        if !self.sample_fraction.is_finite() || self.sample_fraction <= 0.0 {
            return invalid("training sample_fraction must be finite and greater than zero");
        }
        if self
            .time_limit_hours
            .is_some_and(|hours| !hours.is_finite() || hours <= 0.0)
        {
            return invalid("training time_limit_hours must be finite and greater than zero");
        }
        Ok(())
    }

    fn validate_filters_and_checkpoints(&self) -> crate::Result<()> {
        if matches!(self.class_filter.as_ref(), Some(filter) if !filter.is_enabled()) {
            return invalid("training class_filter must be enabled when set");
        }
        if self.log_every_steps == 0 {
            return invalid("training log_every_steps must be greater than zero");
        }
        if matches!(self.checkpoint_every_steps, Some(0)) {
            return invalid("training checkpoint_every_steps must be greater than zero");
        }
        if matches!(self.checkpoint_every_epochs, Some(0)) {
            return invalid("training checkpoint_every_epochs must be greater than zero");
        }
        Ok(())
    }

    fn validate_schedule(&self) -> crate::Result<()> {
        self.loss_config.validate()?;
        if let Some(schedule) = self.learning_rate_schedule {
            schedule.validate()?;
        }
        if let Some(early_stopping) = self.early_stopping {
            early_stopping.validate()?;
        }
        if let Some(state) = self.resume_state {
            self.validate_resume_state(state)?;
        }
        if let Some(warmup) = self.learning_rate_warmup {
            warmup.validate()?;
        }
        if let Some(warmup) = self.bias_learning_rate_warmup {
            warmup.validate()?;
        }
        if let Some(warmup) = self.momentum_warmup {
            warmup.validate()?;
        }
        Ok(())
    }

    fn validate_resume_state(&self, state: super::ResumeState) -> crate::Result<()> {
        if state.completed_epochs >= self.epochs {
            return invalid("resume completed_epochs must be less than training epochs");
        }
        if state.best_loss.is_some_and(|loss| !loss.is_finite()) {
            return invalid("resume best_loss must be finite");
        }
        Ok(())
    }

    pub(crate) fn time_limit_reached(&self, started_at: Instant) -> bool {
        self.time_limit_hours
            .is_some_and(|hours| started_at.elapsed().as_secs_f64() >= hours * 3600.0)
    }
}

fn invalid<T>(message: &str) -> crate::Result<T> {
    Err(crate::Error::InvalidConfig(message.to_string()))
}
