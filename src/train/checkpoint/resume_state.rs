use std::path::{Path, PathBuf};

use super::BestMetric;

/// Lightweight training-loop state saved next to model checkpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResumeState {
    /// Completed one-based epochs before the next training loop starts.
    pub completed_epochs: usize,
    /// Completed optimizer steps before the next training loop starts.
    pub completed_steps: usize,
    /// Best epoch known when this state was written.
    pub best_epoch: Option<usize>,
    /// Training loss observed at the best epoch when this state was written.
    pub best_loss: Option<f32>,
    /// Metric used to select the best checkpoint.
    pub best_metric: Option<BestMetric>,
}

impl ResumeState {
    /// Creates a new resume state after validating the numeric fields.
    pub fn new(
        completed_epochs: usize,
        completed_steps: usize,
        best_epoch: Option<usize>,
        best_loss: Option<f32>,
    ) -> crate::Result<Self> {
        Self::new_with_best_metric(
            completed_epochs,
            completed_steps,
            best_epoch,
            best_loss,
            best_loss.map(BestMetric::training_loss).transpose()?,
        )
    }

    /// Creates a new resume state with an explicit best-checkpoint metric.
    pub fn new_with_best_metric(
        completed_epochs: usize,
        completed_steps: usize,
        best_epoch: Option<usize>,
        best_loss: Option<f32>,
        best_metric: Option<BestMetric>,
    ) -> crate::Result<Self> {
        if best_loss.is_some_and(|loss| !loss.is_finite()) {
            return Err(crate::Error::InvalidConfig(
                "resume best_loss must be finite".to_string(),
            ));
        }
        Ok(Self {
            completed_epochs,
            completed_steps,
            best_epoch,
            best_loss,
            best_metric,
        })
    }

    /// Returns the sidecar path used for a checkpoint path.
    pub fn sidecar_path_for_checkpoint(path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("checkpoint");
        path.with_file_name(format!("{stem}.train-state.json"))
    }

    /// Returns the optimizer-state sidecar path used for a checkpoint path.
    pub fn optimizer_sidecar_path_for_checkpoint(path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("checkpoint");
        path.with_file_name(format!("{stem}.optimizer.safetensors"))
    }
}
