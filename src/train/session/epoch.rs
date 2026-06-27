use super::*;

pub(crate) struct TrainEpochRunContext<'a> {
    pub(crate) epoch: usize,
    pub(crate) effective_len: usize,
    pub(crate) micro_batches_per_epoch: usize,
    pub(crate) warmup_steps: usize,
    pub(crate) started_at: std::time::Instant,
    pub(crate) checkpoint_report: &'a mut CheckpointReport,
    pub(crate) total_report: &'a mut TrainLoopRunningReport,
    pub(crate) total_steps: &'a mut usize,
    pub(crate) last_opt_micro_index: &'a mut isize,
    pub(crate) pending: &'a mut PendingTrainStep,
}

pub(crate) struct TrainEpochRunReport {
    pub(crate) report: EpochReport,
    pub(crate) time_limit_reached: bool,
}

impl Session {
    pub(crate) fn run_train_epoch<D, F>(
        &mut self,
        dataset: &D,
        config: &RunnerConfig,
        collate: &F,
        context: TrainEpochRunContext<'_>,
    ) -> crate::Result<TrainEpochRunReport>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        let learning_rate = self.epoch_learning_rate(config, context.epoch)?;
        let progressive_loss_weights =
            ProgressiveLossSchedule::yolo26().weights_after_epochs(context.epoch, config.epochs)?;
        let mut first_step_learning_rate = learning_rate;
        let mut last_step_learning_rate = learning_rate;
        let mut first_step_momentum = self.optimizer.momentum();
        let mut last_step_momentum = first_step_momentum;
        let mut epoch_report_state = TrainLoopRunningReport::default();
        let mut epoch_optimizer_steps = 0usize;
        let mut time_limit_reached = false;
        let mut micro_step = 0usize;
        while micro_step < context.micro_batches_per_epoch {
            config.check_cancelled()?;
            let micro_index = context.epoch * context.micro_batches_per_epoch + micro_step;
            let (loss_report, settings) = self.train_loop_micro_batch(
                dataset,
                config,
                collate,
                TrainLoopMicroBatchContext {
                    effective_len: context.effective_len,
                    micro_index,
                    warmup_steps: context.warmup_steps,
                    learning_rate,
                    progressive_loss_weights,
                    loss_config: config.loss_config,
                },
            )?;
            let micro_report = context.pending.add(loss_report, settings)?;
            context.total_report.add(micro_report);
            epoch_report_state.add(micro_report);
            let is_last_micro_batch = micro_step + 1 == context.micro_batches_per_epoch;
            if should_optimizer_step(config, &context, micro_index, is_last_micro_batch) {
                let step_update = self.finish_train_loop_step(context.pending, config)?;
                *context.last_opt_micro_index = micro_index as isize;
                if epoch_optimizer_steps == 0 {
                    first_step_learning_rate = step_update.first_settings.learning_rate;
                    first_step_momentum = step_update.first_settings.momentum;
                }
                last_step_learning_rate = step_update.last_settings.learning_rate;
                last_step_momentum = step_update.last_settings.momentum;
                *context.total_steps += 1;
                epoch_optimizer_steps += 1;
                if let Some(ema) = self.ema.as_mut() {
                    ema.update(&self.model.named_variables()?)?;
                }
                save_step_checkpoint(
                    self,
                    context.checkpoint_report,
                    config.checkpoint_dir.as_deref(),
                    config.checkpoint_every_steps,
                    context.epoch + 1,
                    *context.total_steps,
                )?;
                time_limit_reached = config.time_limit_reached(context.started_at);
                if time_limit_reached {
                    break;
                }
            }
            micro_step += 1;
            if config
                .steps_per_epoch
                .is_some_and(|steps| epoch_optimizer_steps >= steps)
            {
                break;
            }
        }
        config.check_cancelled()?;
        Ok(TrainEpochRunReport {
            report: EpochReport {
                epoch: context.epoch + 1,
                steps: epoch_optimizer_steps,
                learning_rate,
                first_step_learning_rate,
                last_step_learning_rate,
                first_step_momentum,
                last_step_momentum,
                mean_loss: epoch_report_state.mean_loss(),
                last_loss: epoch_report_state.last_loss,
                mean_components: epoch_report_state.components.mean(),
                last_components: epoch_report_state.last_components,
                validation_fitness: None,
            },
            time_limit_reached,
        })
    }
}

fn should_optimizer_step(
    config: &RunnerConfig,
    context: &TrainEpochRunContext<'_>,
    micro_index: usize,
    _is_last_micro_batch: bool,
) -> bool {
    let accumulate = config.step_accumulate(micro_index, context.warmup_steps);
    micro_index as isize - *context.last_opt_micro_index >= accumulate as isize
}
