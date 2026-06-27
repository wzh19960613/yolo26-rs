use super::{Dataset, EvalLoopConfig, RunnerConfig, Sample};

impl RunnerConfig {
    pub(crate) fn collect_step_samples<D: Dataset>(
        &self,
        dataset: &D,
        base: usize,
        effective_len: usize,
    ) -> crate::Result<Vec<Sample>> {
        collect_filtered_samples(
            dataset,
            base,
            effective_len,
            self.batch_size,
            self.class_filter.as_ref(),
            self.sample_order,
            Some(
                self.current_epoch
                    .as_ref()
                    .map(|handle| handle.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0),
            ),
        )
    }
}

impl EvalLoopConfig {
    pub(crate) fn collect_step_samples<D: Dataset>(
        &self,
        dataset: &D,
        base: usize,
        effective_len: usize,
    ) -> crate::Result<Vec<Sample>> {
        let end = base.saturating_add(self.batch_size).min(effective_len);
        let mut samples = Vec::with_capacity(end.saturating_sub(base));
        for logical_index in base..end {
            let dataset_index = self
                .sample_order
                .dataset_index(logical_index, effective_len);
            let sample = dataset.sample(dataset_index)?;
            if let Some(sample) = filter_sample(sample, self.class_filter.as_ref())? {
                samples.push(sample);
            }
        }
        if samples.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "class filter removed every sample in the evaluation batch".to_string(),
            ));
        }
        Ok(samples)
    }
}

fn collect_filtered_samples<D: Dataset>(
    dataset: &D,
    base: usize,
    effective_len: usize,
    batch_size: usize,
    class_filter: Option<&super::ClassFilter>,
    sample_order: super::SampleOrder,
    epoch: Option<usize>,
) -> crate::Result<Vec<Sample>> {
    let mut samples = Vec::with_capacity(batch_size);
    let max_attempts = effective_len.saturating_mul(batch_size.max(1));
    let epoch_indices = epoch.map(|epoch| sample_order.epoch_indices(effective_len, epoch));
    for offset in 0..max_attempts {
        if samples.len() == batch_size {
            break;
        }
        let logical_index = base + offset;
        let dataset_index = epoch_indices.as_ref().map_or_else(
            || sample_order.dataset_index(logical_index, effective_len),
            |indices| indices[logical_index % effective_len],
        );
        let sample = dataset.sample(dataset_index)?;
        if let Some(sample) = filter_sample(sample, class_filter)? {
            samples.push(sample);
        }
    }
    if samples.is_empty() {
        return Err(crate::Error::InvalidConfig(
            "class filter removed every sample in the effective dataset".to_string(),
        ));
    }
    if samples.len() != batch_size {
        return Err(crate::Error::InvalidConfig(format!(
            "class filter kept only {} samples for batch_size {}",
            samples.len(),
            batch_size
        )));
    }
    Ok(samples)
}

fn filter_sample(
    sample: Sample,
    class_filter: Option<&super::ClassFilter>,
) -> crate::Result<Option<Sample>> {
    if let Some(filter) = class_filter {
        filter.filter_sample(sample)
    } else {
        Ok(Some(sample))
    }
}
