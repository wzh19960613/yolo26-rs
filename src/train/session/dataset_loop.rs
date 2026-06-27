use super::*;

impl Session {
    /// Runs an epoch/step training loop over a dataset.
    pub fn train_dataset<D, F>(
        &mut self,
        dataset: &D,
        config: RunnerConfig,
        collate: F,
    ) -> crate::Result<RunnerReport>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        self.train_dataset_with_epoch_fitness(dataset, config, collate, |_, _| Ok(None))
    }

    /// Runs training and calls `epoch_fitness` after each epoch.
    ///
    /// When the callback returns `Some(fitness)`, `best.pt` is selected
    /// by maximizing that validation fitness. Returning `None` falls back to the
    /// lower-is-better epoch training loss used by `train_dataset`.
    pub fn train_dataset_with_epoch_fitness<D, F, V>(
        &mut self,
        dataset: &D,
        config: RunnerConfig,
        collate: F,
        mut epoch_fitness: V,
    ) -> crate::Result<RunnerReport>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
        V: FnMut(&Session, &EpochReport) -> crate::Result<Option<f32>>,
    {
        config.validate()?;
        if dataset.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "cannot train on an empty dataset".to_string(),
            ));
        }
        if let Some(dir) = config.checkpoint_dir.as_ref() {
            std::fs::create_dir_all(dir)?;
        }
        let started_at = std::time::Instant::now();
        let effective_len = config.effective_len(dataset.len());
        let micro_batches_per_epoch = config.effective_micro_batches_per_epoch(effective_len);
        let warmup_steps = config.warmup_steps(micro_batches_per_epoch)?;
        let resume_state = config.resume_state;
        let start_epoch = resume_state.map_or(0, |state| state.completed_epochs);
        let mut total_steps = resume_state.map_or(0, |state| state.completed_steps);
        let mut total_report = TrainLoopRunningReport::default();
        let mut epoch_reports = Vec::with_capacity(config.epochs);
        let mut early_stopping = config.early_stopping.map(EarlyStoppingState::new);
        let mut checkpoint_report = CheckpointReport::with_resume_state(resume_state);
        let mut last_opt_micro_index = -1isize;
        let mut pending = PendingTrainStep::default();
        let mut time_limit_reached = false;
        if let Some(decay) = config.ema_decay {
            self.ema = Some(crate::train::session::ema::ModelEma::new(
                decay,
                &self.model.named_variables()?,
            )?);
        }
        for epoch in start_epoch..config.epochs {
            config.check_cancelled()?;
            if let Some(handle) = &config.current_epoch {
                handle.store(epoch, std::sync::atomic::Ordering::Relaxed);
            }
            let epoch_state = self.run_train_epoch(
                dataset,
                &config,
                &collate,
                TrainEpochRunContext {
                    epoch,
                    effective_len,
                    micro_batches_per_epoch,
                    warmup_steps,
                    started_at,
                    checkpoint_report: &mut checkpoint_report,
                    total_report: &mut total_report,
                    total_steps: &mut total_steps,
                    last_opt_micro_index: &mut last_opt_micro_index,
                    pending: &mut pending,
                },
            )?;
            time_limit_reached = epoch_state.time_limit_reached;
            let mut epoch_report = epoch_state.report;
            let validation_fitness = epoch_fitness(self, &epoch_report)?;
            config.check_cancelled()?;
            let best_metric = match validation_fitness {
                Some(fitness) => BestMetric::validation_fitness(fitness)?,
                None => BestMetric::training_loss(epoch_report.mean_loss)?,
            };
            epoch_report.validation_fitness = validation_fitness;
            let stop = update_early_stopping(
                &mut early_stopping,
                epoch_report.epoch,
                epoch_report.mean_loss,
                validation_fitness,
            );
            save_epoch_checkpoints(
                self,
                &mut checkpoint_report,
                config.checkpoint_dir.as_deref(),
                config.checkpoint_every_epochs,
                &epoch_report,
                best_metric,
                total_steps,
            )?;
            epoch_reports.push(epoch_report);
            if stop || time_limit_reached {
                break;
            }
        }
        Ok(RunnerReport {
            total_steps,
            elapsed_seconds: started_at.elapsed().as_secs_f64(),
            time_limit_reached,
            last_loss: total_report.last_loss,
            mean_loss: total_report.mean_loss(),
            mean_components: total_report.components.mean(),
            last_components: total_report.last_components,
            early_stopping: early_stopping.map(EarlyStoppingState::report),
            checkpoints: checkpoint_report,
            epochs: epoch_reports,
        })
    }
}
