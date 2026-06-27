use super::*;
use candle_core::backprop::GradStore;
use std::fmt::Write;

#[derive(Default)]
pub(crate) struct PendingTrainStep {
    grads: Option<GradStore>,
    components: LossComponentsAccumulator,
    micro_batch_count: usize,
    first_settings: Option<super::OptimizerStepSettings>,
    last_settings: Option<super::OptimizerStepSettings>,
}

pub(crate) struct TrainLoopMicroBatchContext {
    pub(crate) effective_len: usize,
    pub(crate) micro_index: usize,
    pub(crate) warmup_steps: usize,
    pub(crate) learning_rate: f64,
    pub(crate) progressive_loss_weights: ProgressiveLossWeights,
    pub(crate) loss_config: DetectionLossConfig,
}

pub(crate) struct TrainLoopStepUpdate {
    pub(crate) report: Report,
    pub(crate) first_settings: super::OptimizerStepSettings,
    pub(crate) last_settings: super::OptimizerStepSettings,
}

#[derive(Default)]
pub(crate) struct TrainLoopRunningReport {
    pub(crate) steps: usize,
    pub(crate) loss_sum: f64,
    pub(crate) last_loss: f32,
    pub(crate) components: LossComponentsAccumulator,
    pub(crate) last_components: LossComponents,
}

impl Session {
    /// Runs a short training loop and returns JSON diagnostics for loss,
    /// gradients, learning rates, and tracked parameter deltas.
    ///
    /// This is intended for parity checks against Ultralytics training traces.
    /// It mutates the session exactly like a normal training loop for the
    /// optimizer steps it executes.
    pub fn debug_train_micro_batches_json<D, F>(
        &mut self,
        dataset: &D,
        config: RunnerConfig,
        collate: F,
        max_optimizer_steps: usize,
        tracked_names: &[&str],
    ) -> crate::Result<String>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        config.validate()?;
        if dataset.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "cannot debug-train on an empty dataset".to_string(),
            ));
        }
        if max_optimizer_steps == 0 {
            return Ok("{\"records\":[],\"deltas\":{}}".to_string());
        }

        let effective_len = config.effective_len(dataset.len());
        let micro_batches_per_epoch = config.effective_micro_batches_per_epoch(effective_len);
        let warmup_steps = config.warmup_steps(micro_batches_per_epoch)?;
        let named = self.model.named_variables()?;
        let before = tracked_tensor_snapshots(&named, tracked_names)?;

        let mut pending = PendingTrainStep::default();
        let mut last_opt_micro_index = -1isize;
        let mut micro_index = 0usize;
        let mut optimizer_steps = 0usize;
        let mut loss_item_sums = [0.0f64; 5];
        let mut loss_item_count = 0usize;
        let mut records = Vec::new();

        while optimizer_steps < max_optimizer_steps {
            let epoch = micro_index / micro_batches_per_epoch;
            if epoch >= config.epochs {
                break;
            }
            if micro_index.is_multiple_of(micro_batches_per_epoch)
                && let Some(handle) = &config.current_epoch
            {
                handle.store(epoch, std::sync::atomic::Ordering::Relaxed);
            }
            let micro_step = micro_index % micro_batches_per_epoch;
            let learning_rate = self.epoch_learning_rate(&config, epoch)?;
            let progressive_loss_weights =
                ProgressiveLossSchedule::yolo26().weights_after_epochs(epoch, config.epochs)?;
            let (loss_report, settings) = self.train_loop_micro_batch(
                dataset,
                &config,
                &collate,
                TrainLoopMicroBatchContext {
                    effective_len,
                    micro_index,
                    warmup_steps,
                    learning_rate,
                    progressive_loss_weights,
                    loss_config: config.loss_config,
                },
            )?;
            let components = loss_report.scalar_components()?;
            let loss_items = segment_loss_items(components);
            for (sum, value) in loss_item_sums.iter_mut().zip(loss_items) {
                *sum += value as f64;
            }
            loss_item_count += 1;
            let _ = pending.add(loss_report, settings)?;

            if debug_should_optimizer_step(
                &config,
                micro_index,
                warmup_steps,
                &last_opt_micro_index,
            ) {
                let lrs = self.optimizer.group_learning_rates();
                let mut grads = pending.grads.take().ok_or_else(|| {
                    crate::Error::InvalidConfig(
                        "cannot debug-train with zero accumulated batches".to_string(),
                    )
                })?;
                let pre_norm = grad_l2_norm(&named, &grads)?;
                let pre_tracked = tracked_grad_abs_sums(&named, &grads, tracked_names)?;
                let top_l2 = top_grad_summaries(&named, &grads, GradSummaryMetric::L2, 40)?;
                let top_abs = top_grad_summaries(&named, &grads, GradSummaryMetric::AbsSum, 40)?;
                if let Some(max_norm) = config.gradient_clip_norm {
                    clip_grad_store(&self.model, &mut grads, max_norm)?;
                }
                let post_norm = grad_l2_norm(&named, &grads)?;
                let post_tracked = tracked_grad_abs_sums(&named, &grads, tracked_names)?;
                self.optimizer.step_with_grads(&grads)?;
                if let Some(ema) = self.ema.as_mut() {
                    ema.update(&named)?;
                }
                let _ = std::mem::take(&mut pending.components).mean();
                pending.micro_batch_count = 0;
                pending.first_settings.take();
                pending.last_settings.take();
                last_opt_micro_index = micro_index as isize;
                records.push(DebugTrainRecord {
                    step: optimizer_steps,
                    epoch,
                    micro_step,
                    lrs,
                    pre_norm,
                    post_norm,
                    pre_tracked,
                    post_tracked,
                    top_l2,
                    top_abs,
                    loss_items,
                    tloss: mean_loss_items(loss_item_sums, loss_item_count),
                });
                optimizer_steps += 1;
            }
            micro_index += 1;
        }

        let after_named = self.model.named_variables()?;
        let deltas = tracked_tensor_deltas(&before, &after_named)?;
        debug_train_json(&records, &deltas)
    }

    pub(crate) fn step_loss(&mut self, report: LossTensorReport) -> crate::Result<Report> {
        let components = report.scalar_components()?;
        self.optimizer.backward_step(&report.loss)?;
        Ok(Report {
            loss: components.total,
            components,
        })
    }

    pub(crate) fn loss_report(
        &self,
        input: &Tensor,
        target: &Target,
        progressive_loss_weights: ProgressiveLossWeights,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<LossTensorReport> {
        // Augmentation can produce non-contiguous tensors; force contiguity.
        let input = input.contiguous()?;
        let output = self.model.forward_raw(&input)?;
        supervised_loss_report_with_progressive_weights(
            &output,
            target,
            progressive_loss_weights,
            loss_config,
        )
    }

    pub(crate) fn epoch_learning_rate(
        &self,
        config: &RunnerConfig,
        epoch: usize,
    ) -> crate::Result<f64> {
        if let Some(schedule) = config.learning_rate_schedule {
            schedule.learning_rate(epoch, config.epochs)
        } else {
            Ok(self.optimizer.learning_rate())
        }
    }

    pub(crate) fn train_loop_micro_batch<D, F>(
        &mut self,
        dataset: &D,
        config: &RunnerConfig,
        collate: &F,
        context: TrainLoopMicroBatchContext,
    ) -> crate::Result<(LossTensorReport, super::OptimizerStepSettings)>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        let settings = self.optimizer.apply_step_settings(
            config,
            context.micro_index,
            context.warmup_steps,
            context.learning_rate,
        )?;
        let base = context.micro_index * config.batch_size;
        let samples = config.collect_step_samples(dataset, base, context.effective_len)?;
        let batch = collate(&samples)?;
        Ok((
            self.loss_report(
                &batch.input,
                &batch.target,
                context.progressive_loss_weights,
                context.loss_config,
            )?,
            settings,
        ))
    }

    pub(crate) fn finish_train_loop_step(
        &mut self,
        pending: &mut PendingTrainStep,
        config: &RunnerConfig,
    ) -> crate::Result<TrainLoopStepUpdate> {
        let mut grads = pending.grads.take().ok_or_else(|| {
            crate::Error::InvalidConfig("cannot train with zero accumulated batches".to_string())
        })?;
        if let Some(max_norm) = config.gradient_clip_norm {
            clip_grad_store(&self.model, &mut grads, max_norm)?;
        }
        self.optimizer.step_with_grads(&grads)?;
        let components = std::mem::take(&mut pending.components).mean();
        let report = Report {
            loss: components.total,
            components,
        };
        Ok(TrainLoopStepUpdate {
            report,
            first_settings: pending.first_settings.take().ok_or_else(|| {
                crate::Error::InvalidConfig("missing first optimizer settings".to_string())
            })?,
            last_settings: pending.last_settings.take().ok_or_else(|| {
                crate::Error::InvalidConfig("missing last optimizer settings".to_string())
            })?,
        })
    }
}

struct DebugTrainRecord {
    step: usize,
    epoch: usize,
    micro_step: usize,
    lrs: Vec<f64>,
    pre_norm: f64,
    post_norm: f64,
    pre_tracked: Vec<(String, f64)>,
    post_tracked: Vec<(String, f64)>,
    top_l2: Vec<GradSummary>,
    top_abs: Vec<GradSummary>,
    loss_items: [f32; 5],
    tloss: [f32; 5],
}

struct GradSummary {
    name: String,
    l2: f64,
    abs_sum: f64,
}

#[derive(Clone, Copy)]
enum GradSummaryMetric {
    L2,
    AbsSum,
}

struct TensorSnapshot {
    name: String,
    tensor: Tensor,
    before_abs_sum: f64,
}

struct TensorDelta {
    name: String,
    abs_sum_delta: f64,
    before_abs_sum: f64,
    after_abs_sum: f64,
}

impl PendingTrainStep {
    pub(crate) fn add(
        &mut self,
        report: LossTensorReport,
        settings: super::OptimizerStepSettings,
    ) -> crate::Result<Report> {
        self.first_settings.get_or_insert(settings);
        self.last_settings = Some(settings);
        self.micro_batch_count += 1;
        let components = report.scalar_components()?;
        self.components.add(components);
        let grads = report.loss.backward()?;
        if let Some(accumulated) = self.grads.as_mut() {
            accumulate_grad_store(accumulated, &grads)?;
        } else {
            self.grads = Some(grads);
        }
        Ok(Report {
            loss: components.total,
            components,
        })
    }
}

fn debug_should_optimizer_step(
    config: &RunnerConfig,
    micro_index: usize,
    warmup_steps: usize,
    last_opt_micro_index: &isize,
) -> bool {
    let accumulate = config.step_accumulate(micro_index, warmup_steps);
    micro_index as isize - *last_opt_micro_index >= accumulate as isize
}

fn segment_loss_items(components: LossComponents) -> [f32; 5] {
    [
        components.box_loss.unwrap_or(0.0),
        components.mask_loss.unwrap_or(0.0),
        components.cls_loss.unwrap_or(0.0),
        components.dfl_loss.unwrap_or(0.0),
        components.semantic_loss.unwrap_or(0.0),
    ]
}

fn mean_loss_items(sums: [f64; 5], count: usize) -> [f32; 5] {
    if count == 0 {
        return [0.0; 5];
    }
    sums.map(|sum| (sum / count as f64) as f32)
}

fn grad_l2_norm(named: &[(String, Var)], grads: &GradStore) -> crate::Result<f64> {
    let mut sum_sq = 0.0f64;
    for (_, var) in named {
        let Some(grad) = grads.get(var) else {
            continue;
        };
        let sq = grad
            .to_dtype(candle_core::DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?;
        sum_sq += sq as f64;
    }
    Ok(sum_sq.sqrt())
}

fn tracked_grad_abs_sums(
    named: &[(String, Var)],
    grads: &GradStore,
    tracked_names: &[&str],
) -> crate::Result<Vec<(String, f64)>> {
    tracked_names
        .iter()
        .map(|name| {
            let value = named
                .iter()
                .find(|(candidate, _)| candidate == name)
                .and_then(|(_, var)| grads.get(var))
                .map(tensor_abs_sum)
                .transpose()?
                .unwrap_or(0.0);
            Ok(((*name).to_string(), value))
        })
        .collect()
}

fn top_grad_summaries(
    named: &[(String, Var)],
    grads: &GradStore,
    metric: GradSummaryMetric,
    limit: usize,
) -> crate::Result<Vec<GradSummary>> {
    let mut summaries = named
        .iter()
        .filter_map(|(name, var)| grads.get(var).map(|grad| (name, grad)))
        .map(|(name, grad)| {
            let grad = grad.to_dtype(candle_core::DType::F32)?;
            let l2 = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            let abs_sum = grad.abs()?.sum_all()?.to_scalar::<f32>()? as f64;
            Ok(GradSummary {
                name: name.clone(),
                l2: l2.sqrt(),
                abs_sum,
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    summaries.sort_by(|left, right| {
        let left_value = match metric {
            GradSummaryMetric::L2 => left.l2,
            GradSummaryMetric::AbsSum => left.abs_sum,
        };
        let right_value = match metric {
            GradSummaryMetric::L2 => right.l2,
            GradSummaryMetric::AbsSum => right.abs_sum,
        };
        right_value.total_cmp(&left_value)
    });
    summaries.truncate(limit);
    Ok(summaries)
}

fn tracked_tensor_snapshots(
    named: &[(String, Var)],
    tracked_names: &[&str],
) -> crate::Result<Vec<TensorSnapshot>> {
    tracked_names
        .iter()
        .filter_map(|name| {
            named
                .iter()
                .find(|(candidate, _)| candidate == name)
                .map(|(_, var)| ((*name).to_string(), var.clone()))
        })
        .map(|(name, var)| {
            let tensor = (var.as_tensor() * 1.0)?;
            let before_abs_sum = tensor_abs_sum(&tensor)?;
            Ok(TensorSnapshot {
                name,
                tensor,
                before_abs_sum,
            })
        })
        .collect()
}

fn tracked_tensor_deltas(
    before: &[TensorSnapshot],
    after_named: &[(String, Var)],
) -> crate::Result<Vec<TensorDelta>> {
    before
        .iter()
        .filter_map(|snapshot| {
            after_named
                .iter()
                .find(|(name, _)| name == &snapshot.name)
                .map(|(_, var)| (snapshot, var))
        })
        .map(|(snapshot, var)| {
            let after = var.as_tensor();
            let after_abs_sum = tensor_abs_sum(after)?;
            let abs_sum_delta = tensor_abs_sum(&(after - &snapshot.tensor)?)?;
            Ok(TensorDelta {
                name: snapshot.name.clone(),
                abs_sum_delta,
                before_abs_sum: snapshot.before_abs_sum,
                after_abs_sum,
            })
        })
        .collect()
}

fn tensor_abs_sum(tensor: &Tensor) -> crate::Result<f64> {
    Ok(tensor
        .to_dtype(candle_core::DType::F32)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()? as f64)
}

fn debug_train_json(records: &[DebugTrainRecord], deltas: &[TensorDelta]) -> crate::Result<String> {
    let mut out = String::new();
    out.push_str("{\"records\":[");
    for (idx, record) in records.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write!(
            out,
            "{{\"step\":{},\"epoch\":{},\"micro_step\":{},\"lrs\":",
            record.step, record.epoch, record.micro_step
        )
        .map_err(fmt_error)?;
        write_f64_array(&mut out, &record.lrs)?;
        out.push_str(",\"pre_norm\":");
        write_f64(&mut out, record.pre_norm)?;
        out.push_str(",\"post_norm\":");
        write_f64(&mut out, record.post_norm)?;
        out.push_str(",\"pre_tracked\":");
        write_f64_map(&mut out, &record.pre_tracked)?;
        out.push_str(",\"post_tracked\":");
        write_f64_map(&mut out, &record.post_tracked)?;
        out.push_str(",\"top_l2\":");
        write_grad_summaries(&mut out, &record.top_l2)?;
        out.push_str(",\"top_abs\":");
        write_grad_summaries(&mut out, &record.top_abs)?;
        out.push_str(",\"loss_items\":");
        write_f32_array(&mut out, &record.loss_items)?;
        out.push_str(",\"tloss\":");
        write_f32_array(&mut out, &record.tloss)?;
        out.push('}');
    }
    out.push_str("],\"deltas\":{");
    for (idx, delta) in deltas.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_json_string(&mut out, &delta.name)?;
        out.push_str(":{\"abs_sum_delta\":");
        write_f64(&mut out, delta.abs_sum_delta)?;
        out.push_str(",\"before_abs_sum\":");
        write_f64(&mut out, delta.before_abs_sum)?;
        out.push_str(",\"after_abs_sum\":");
        write_f64(&mut out, delta.after_abs_sum)?;
        out.push('}');
    }
    out.push_str("}}");
    Ok(out)
}

fn write_grad_summaries(out: &mut String, values: &[GradSummary]) -> crate::Result<()> {
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":");
        write_json_string(out, &value.name)?;
        out.push_str(",\"l2\":");
        write_f64(out, value.l2)?;
        out.push_str(",\"abs_sum\":");
        write_f64(out, value.abs_sum)?;
        out.push('}');
    }
    out.push(']');
    Ok(())
}

fn write_f64_map(out: &mut String, values: &[(String, f64)]) -> crate::Result<()> {
    out.push('{');
    for (idx, (name, value)) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_json_string(out, name)?;
        out.push(':');
        write_f64(out, *value)?;
    }
    out.push('}');
    Ok(())
}

fn write_f32_array(out: &mut String, values: &[f32]) -> crate::Result<()> {
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_f64(out, *value as f64)?;
    }
    out.push(']');
    Ok(())
}

fn write_f64_array(out: &mut String, values: &[f64]) -> crate::Result<()> {
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_f64(out, *value)?;
    }
    out.push(']');
    Ok(())
}

fn write_f64(out: &mut String, value: f64) -> crate::Result<()> {
    if value.is_finite() {
        write!(out, "{value:.12}").map_err(fmt_error)
    } else {
        out.push_str("null");
        Ok(())
    }
}

fn write_json_string(out: &mut String, value: &str) -> crate::Result<()> {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                write!(out, "\\u{:04x}", ch as u32).map_err(fmt_error)?;
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    Ok(())
}

fn fmt_error(err: std::fmt::Error) -> crate::Error {
    crate::Error::InvalidConfig(format!("failed to format training diagnostics: {err}"))
}

fn accumulate_grad_store(accumulated: &mut GradStore, incoming: &GradStore) -> crate::Result<()> {
    let ids = incoming.get_ids().copied().collect::<Vec<_>>();
    for id in ids {
        let Some(incoming_grad) = incoming.get_id(id) else {
            continue;
        };
        let merged = if let Some(existing_grad) = accumulated.get_id(id) {
            existing_grad.add(incoming_grad)?
        } else {
            incoming_grad.clone()
        };
        accumulated.insert_id(id, merged);
    }
    Ok(())
}

fn clip_grad_store(model: &Model, grads: &mut GradStore, max_norm: f32) -> crate::Result<()> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Ok(());
    }
    let vars = model
        .named_variables()?
        .into_iter()
        .map(|(_, var)| var)
        .collect::<Vec<_>>();
    let mut sum_sq = 0.0f64;
    for var in &vars {
        if let Some(grad) = grads.get(var) {
            let sq = grad
                .to_dtype(candle_core::DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?;
            sum_sq += sq as f64;
        }
    }
    let norm = sum_sq.sqrt();
    if !norm.is_finite() || norm <= max_norm as f64 {
        return Ok(());
    }
    let scale = max_norm as f64 / (norm + 1e-6);
    for var in vars {
        if let Some(grad) = grads.get(&var) {
            grads.insert(&var, (grad * scale)?);
        }
    }
    Ok(())
}

impl TrainLoopRunningReport {
    pub(crate) fn add(&mut self, report: Report) {
        self.steps += 1;
        self.loss_sum += report.loss as f64;
        self.last_loss = report.loss;
        self.components.add(report.components);
        self.last_components = report.components;
    }

    pub(crate) fn mean_loss(&self) -> f32 {
        (self.loss_sum / self.steps.max(1) as f64) as f32
    }
}
