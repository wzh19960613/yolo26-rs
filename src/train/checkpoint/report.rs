use std::path::{Path, PathBuf};

use super::BestMetric;
use super::Session;
use crate::train::checkpoint::resume_state::ResumeState;
use crate::train::runner::report::EpochReport;

/// Checkpoints written by a dataset training loop.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CheckpointReport {
    /// Last checkpoint written in official-style `last.pt` form.
    pub last: Option<PathBuf>,
    /// JSON state sidecar written next to `last`.
    pub last_state: Option<PathBuf>,
    /// Optimizer-state sidecar written next to `last`.
    pub last_optimizer_state: Option<PathBuf>,
    /// Best checkpoint written in official-style `best.pt` form.
    pub best: Option<PathBuf>,
    /// JSON state sidecar written next to `best`.
    pub best_state: Option<PathBuf>,
    /// Optimizer-state sidecar written next to `best`.
    pub best_optimizer_state: Option<PathBuf>,
    /// One-based epoch index that produced `best`.
    pub best_epoch: Option<usize>,
    /// Training-loss value observed at the best epoch.
    pub best_loss: Option<f32>,
    /// Metric used to select `best`.
    pub best_metric: Option<BestMetric>,
    /// Step or epoch checkpoint paths written by periodic checkpoint rules.
    pub periodic: Vec<PathBuf>,
    /// JSON state sidecars written next to periodic checkpoints.
    pub periodic_states: Vec<PathBuf>,
    /// Optimizer-state sidecars written next to periodic checkpoints.
    pub periodic_optimizer_states: Vec<PathBuf>,
    /// Resume state used at the start of this training loop.
    pub resumed_from: Option<ResumeState>,
}

impl CheckpointReport {
    fn should_update_best(&self, metric: BestMetric) -> bool {
        metric.is_better_than(self.best_metric)
    }

    fn record_periodic(&mut self, path: PathBuf) {
        if !self.periodic.contains(&path) {
            self.periodic.push(path);
        }
    }

    fn record_periodic_state(&mut self, path: PathBuf) {
        if !self.periodic_states.contains(&path) {
            self.periodic_states.push(path);
        }
    }

    fn record_periodic_optimizer_state(&mut self, path: PathBuf) {
        if !self.periodic_optimizer_states.contains(&path) {
            self.periodic_optimizer_states.push(path);
        }
    }

    pub(crate) fn with_resume_state(state: Option<ResumeState>) -> Self {
        Self {
            resumed_from: state,
            best_epoch: state.and_then(|state| state.best_epoch),
            best_loss: state.and_then(|state| state.best_loss),
            best_metric: state.and_then(|state| state.best_metric),
            ..Default::default()
        }
    }
}

pub(crate) fn save_step_checkpoint(
    session: &Session,
    report: &mut CheckpointReport,
    dir: Option<&Path>,
    every: Option<usize>,
    epoch: usize,
    step: usize,
) -> crate::Result<()> {
    if let (Some(dir), Some(every)) = (dir, every)
        && step.is_multiple_of(every)
    {
        save_numbered_checkpoint(session, report, dir, epoch, step)?;
    }
    Ok(())
}

pub(crate) fn save_epoch_checkpoints(
    session: &Session,
    report: &mut CheckpointReport,
    dir: Option<&Path>,
    every: Option<usize>,
    epoch: &EpochReport,
    best_metric: BestMetric,
    step: usize,
) -> crate::Result<()> {
    if let Some(dir) = dir {
        save_best_checkpoint(session, report, dir, epoch, best_metric, step)?;
        save_last_checkpoint(session, report, dir, epoch, step)?;
    }
    if let (Some(dir), Some(every)) = (dir, every)
        && epoch.epoch.is_multiple_of(every)
    {
        save_numbered_checkpoint(session, report, dir, epoch.epoch, step)?;
    }
    Ok(())
}

fn save_last_checkpoint(
    session: &Session,
    report: &mut CheckpointReport,
    dir: &Path,
    epoch: &EpochReport,
    step: usize,
) -> crate::Result<()> {
    let path = dir.join("last.pt");
    session.save_checkpoint_weights(&path)?;
    let state_path = write_state_sidecar(&path, report, epoch.epoch, step)?;
    let optimizer_path = write_optimizer_sidecar(session, &path)?;
    report.last = Some(path);
    report.last_state = Some(state_path);
    report.last_optimizer_state = optimizer_path;
    Ok(())
}

fn save_best_checkpoint(
    session: &Session,
    report: &mut CheckpointReport,
    dir: &Path,
    epoch: &EpochReport,
    best_metric: BestMetric,
    step: usize,
) -> crate::Result<()> {
    if report.should_update_best(best_metric) {
        let path = dir.join("best.pt");
        session.save_checkpoint_weights(&path)?;
        report.best_epoch = Some(epoch.epoch);
        report.best_loss = Some(epoch.mean_loss);
        report.best_metric = Some(best_metric);
        let state_path = write_state_sidecar(&path, report, epoch.epoch, step)?;
        let optimizer_path = write_optimizer_sidecar(session, &path)?;
        report.best = Some(path);
        report.best_state = Some(state_path);
        report.best_optimizer_state = optimizer_path;
    }
    Ok(())
}

fn save_numbered_checkpoint(
    session: &Session,
    report: &mut CheckpointReport,
    dir: &Path,
    epoch: usize,
    step: usize,
) -> crate::Result<()> {
    let path = dir.join(format!("epoch-{epoch:04}-step-{step:06}.pt"));
    session.save_checkpoint_weights(&path)?;
    let state_path = write_state_sidecar(&path, report, epoch, step)?;
    let optimizer_path = write_optimizer_sidecar(session, &path)?;
    report.record_periodic(path);
    report.record_periodic_state(state_path);
    if let Some(path) = optimizer_path {
        report.record_periodic_optimizer_state(path);
    }
    Ok(())
}

fn write_state_sidecar(
    checkpoint_path: &Path,
    report: &CheckpointReport,
    epoch: usize,
    step: usize,
) -> crate::Result<PathBuf> {
    let state = ResumeState::new_with_best_metric(
        epoch,
        step,
        report.best_epoch,
        report.best_loss,
        report.best_metric,
    )?;
    let path = ResumeState::sidecar_path_for_checkpoint(checkpoint_path);
    state.write_json(&path)?;
    Ok(path)
}

fn write_optimizer_sidecar(
    session: &Session,
    checkpoint_path: &Path,
) -> crate::Result<Option<PathBuf>> {
    let path = ResumeState::optimizer_sidecar_path_for_checkpoint(checkpoint_path);
    if session.save_optimizer_state_safetensors(&path)? {
        Ok(Some(path))
    } else {
        Ok(None)
    }
}
