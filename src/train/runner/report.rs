/// Per-epoch training summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochReport {
    /// One-based epoch index.
    pub epoch: usize,
    /// Number of optimizer steps run in this epoch.
    pub steps: usize,
    /// Learning rate used for this epoch.
    pub learning_rate: f64,
    /// Learning rate used for the first optimizer step in this epoch.
    pub first_step_learning_rate: f64,
    /// Learning rate used for the last optimizer step in this epoch.
    pub last_step_learning_rate: f64,
    /// Momentum or Adam beta1 used for the first optimizer step in this epoch.
    pub first_step_momentum: Option<f64>,
    /// Momentum or Adam beta1 used for the last optimizer step in this epoch.
    pub last_step_momentum: Option<f64>,
    /// Mean loss over this epoch.
    pub mean_loss: f32,
    /// Last loss observed in this epoch.
    pub last_loss: f32,
    /// Mean component-level losses over this epoch.
    pub mean_components: super::LossComponents,
    /// Last component-level losses observed in this epoch.
    pub last_components: super::LossComponents,
    /// Native validation fitness for this epoch, when an epoch validator ran.
    pub validation_fitness: Option<f32>,
}

/// Dataset training loop summary.
#[derive(Debug, Clone, PartialEq)]
pub struct RunnerReport {
    /// Total optimizer steps run.
    pub total_steps: usize,
    /// Wall-clock seconds spent inside this training loop.
    pub elapsed_seconds: f64,
    /// Whether the optional Ultralytics-style time limit stopped training.
    pub time_limit_reached: bool,
    /// Last loss observed.
    pub last_loss: f32,
    /// Mean loss over all optimizer steps.
    pub mean_loss: f32,
    /// Mean component-level losses over all optimizer steps.
    pub mean_components: super::LossComponents,
    /// Last component-level losses observed.
    pub last_components: super::LossComponents,
    /// Final early stopping state, when configured.
    pub early_stopping: Option<super::EarlyStoppingReport>,
    /// Checkpoints written during this training loop.
    pub checkpoints: super::CheckpointReport,
    /// Epoch summaries.
    pub epochs: Vec<EpochReport>,
}
