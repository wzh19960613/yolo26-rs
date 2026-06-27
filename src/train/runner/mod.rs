use crate::model::ImageSize;
pub(crate) use crate::train::exports::*;

pub(crate) mod report;
pub(crate) mod samples;
pub(crate) mod validation;

pub use report::*;
pub(crate) use samples::*;
pub(crate) use validation::*;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Configuration for a dataset training loop.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Number of epochs to run.
    pub epochs: usize,
    /// Batch size in samples.
    pub batch_size: usize,
    /// Optional number of optimizer steps per epoch.
    pub steps_per_epoch: Option<usize>,
    /// Target number of micro-batches accumulated before one optimizer step.
    pub accumulate_steps: usize,
    /// Fraction of the dataset used for training, matching Ultralytics `fraction`.
    pub sample_fraction: f64,
    /// Dataset sample order derived from Ultralytics `seed` and `deterministic`.
    pub sample_order: super::SampleOrder,
    /// Detection-style loss gains and task-aligned assignment settings.
    pub loss_config: super::DetectionLossConfig,
    /// Optional wall-clock training limit in hours, matching Ultralytics `time`.
    pub time_limit_hours: Option<f64>,
    /// Optional class filtering/remapping applied before collation.
    pub class_filter: Option<super::ClassFilter>,
    /// Print/report cadence in optimizer steps.
    pub log_every_steps: usize,
    /// Optional checkpoint directory.
    pub checkpoint_dir: Option<PathBuf>,
    /// Optional checkpoint cadence in optimizer steps.
    pub checkpoint_every_steps: Option<usize>,
    /// Optional checkpoint cadence in epochs, matching Ultralytics `save_period`.
    pub checkpoint_every_epochs: Option<usize>,
    /// Optional early stopping over epoch mean training loss.
    pub early_stopping: Option<super::EarlyStoppingConfig>,
    /// Optional training-loop state restored from a checkpoint sidecar.
    pub resume_state: Option<super::ResumeState>,
    /// Optional per-epoch learning-rate schedule.
    pub learning_rate_schedule: Option<super::LearningRateSchedule>,
    /// Optional per-step learning-rate warmup.
    pub learning_rate_warmup: Option<super::LearningRateWarmup>,
    /// Optional per-step bias learning-rate warmup.
    pub bias_learning_rate_warmup: Option<super::LearningRateWarmup>,
    /// Optional per-step momentum warmup.
    pub momentum_warmup: Option<super::MomentumWarmup>,
    /// Optional shared epoch counter advanced by the loop each epoch, used by an
    /// epoch-aware [`AugmentingDataset`](super::augment::AugmentingDataset) to
    /// disable mosaic/mixup in the last `close_mosaic` epochs.
    pub current_epoch: Option<Arc<AtomicUsize>>,
    /// Optional shared cancellation flag checked between train loop batches.
    pub cancel_requested: Option<Arc<AtomicBool>>,
    /// Optional EMA decay (e.g. 0.9999). When set, the loop maintains a model
    /// weight EMA, updates it after each optimizer step, and saves the EMA
    /// weights as the best/last checkpoint, matching the official trainer.
    pub ema_decay: Option<f32>,
    /// Optional global gradient norm clipping before optimizer step.
    pub gradient_clip_norm: Option<f32>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 1,
            steps_per_epoch: None,
            accumulate_steps: 1,
            sample_fraction: 1.0,
            sample_order: super::SampleOrder::default(),
            loss_config: super::DetectionLossConfig::default(),
            time_limit_hours: None,
            class_filter: None,
            log_every_steps: 1,
            checkpoint_dir: None,
            checkpoint_every_steps: None,
            checkpoint_every_epochs: None,
            early_stopping: None,
            resume_state: None,
            learning_rate_schedule: None,
            learning_rate_warmup: None,
            bias_learning_rate_warmup: None,
            momentum_warmup: None,
            current_epoch: None,
            cancel_requested: None,
            ema_decay: None,
            gradient_clip_norm: None,
        }
    }
}

impl RunnerConfig {
    pub(crate) fn effective_len(&self, dataset_len: usize) -> usize {
        ((dataset_len as f64 * self.sample_fraction).ceil() as usize).clamp(1, dataset_len)
    }

    pub(crate) fn check_cancelled(&self) -> crate::Result<()> {
        if self
            .cancel_requested
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            Err(crate::Error::InvalidConfig(
                "training cancelled".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn effective_micro_batches_per_epoch(&self, effective_len: usize) -> usize {
        let micro_batches = effective_len.div_ceil(self.batch_size);
        self.steps_per_epoch
            .map_or(micro_batches, |steps| steps * self.accumulate_steps)
    }

    pub(crate) fn warmup_steps(&self, micro_batches_per_epoch: usize) -> crate::Result<usize> {
        let mut steps = 0usize;
        for warmup in [self.learning_rate_warmup, self.bias_learning_rate_warmup]
            .into_iter()
            .flatten()
        {
            steps = steps.max(warmup.steps(micro_batches_per_epoch)?);
        }
        Ok(steps)
    }

    pub(crate) fn step_accumulate(&self, step: usize, warmup_steps: usize) -> usize {
        if warmup_steps == 0 || step >= warmup_steps {
            return self.accumulate_steps;
        }
        let target = self.accumulate_steps as f64;
        let fraction = step as f64 / warmup_steps as f64;
        round_half_to_even(1.0 + (target - 1.0) * fraction).max(1.0) as usize
    }

    pub(crate) fn step_bias_learning_rate(
        &self,
        step: usize,
        warmup_steps: usize,
        target_lr: f64,
        current: Option<f64>,
    ) -> crate::Result<Option<f64>> {
        if current.is_none() {
            return Ok(None);
        }
        if let Some(warmup) = self.bias_learning_rate_warmup {
            Ok(Some(warmup.learning_rate(step, warmup_steps, target_lr)?))
        } else {
            Ok(current)
        }
    }

    pub(crate) fn step_learning_rate(
        &self,
        step: usize,
        warmup_steps: usize,
        target_lr: f64,
    ) -> crate::Result<f64> {
        if let Some(warmup) = self.learning_rate_warmup {
            warmup.learning_rate(step, warmup_steps, target_lr)
        } else {
            Ok(target_lr)
        }
    }

    pub(crate) fn step_momentum(
        &self,
        step: usize,
        warmup_steps: usize,
        current: Option<f64>,
    ) -> crate::Result<Option<f64>> {
        if current.is_none() {
            return Ok(None);
        }
        if let Some(warmup) = self.momentum_warmup {
            Ok(Some(warmup.momentum(step, warmup_steps)?))
        } else {
            Ok(current)
        }
    }
}

fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() <= f64::EPSILON * value.abs().max(1.0) {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_accumulate_uses_numpy_half_even_rounding() {
        let config = RunnerConfig {
            accumulate_steps: 4,
            ..RunnerConfig::default()
        };
        assert_eq!(config.step_accumulate(50, 100), 2);
        assert_eq!(config.step_accumulate(84, 100), 4);
    }
}
