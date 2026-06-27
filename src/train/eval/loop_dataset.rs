use super::super::{
    ClassificationEvalMetrics, Dataset, DetectionEvalMetrics, EvalLoopConfig, EvalLoopReport,
    LossComponents, LossComponentsAccumulator, MapAccumulator, Sample, SemanticMapAccumulator,
    Session,
};

impl Session {
    /// Evaluates a dataset without updating model variables.
    pub fn evaluate_dataset<D, F>(
        &self,
        dataset: &D,
        config: EvalLoopConfig,
        collate: F,
    ) -> crate::Result<EvalLoopReport>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        self.with_ema_weights(|session| session.evaluate_dataset_live(dataset, config, collate))
    }

    fn evaluate_dataset_live<D, F>(
        &self,
        dataset: &D,
        config: EvalLoopConfig,
        collate: F,
    ) -> crate::Result<EvalLoopReport>
    where
        D: Dataset,
        F: Fn(&[Sample]) -> crate::Result<Sample>,
    {
        config.validate()?;
        if dataset.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "cannot evaluate an empty dataset".to_string(),
            ));
        }

        let steps = config
            .steps
            .unwrap_or_else(|| dataset.len().div_ceil(config.batch_size));
        let mut total_loss = 0f64;
        let mut last_loss = 0f32;
        let mut total_components = LossComponentsAccumulator::default();
        let mut last_components = LossComponents::default();
        let mut total_samples = 0usize;
        let mut classification = None;
        let mut detection = None;
        let mut map_acc = MapAccumulator::new();
        let mut mask_map_acc = MapAccumulator::new();
        let mut pose_map_acc = MapAccumulator::new();
        let mut semantic_acc = SemanticMapAccumulator::new();
        let mut has_detection_output = false;
        let mut has_mask_output = false;
        let mut has_pose_output = false;

        for step in 0..steps {
            let base = step * config.batch_size;
            let samples = config.collect_step_samples(dataset, base, dataset.len())?;
            let batch = collate(&samples)?;
            let is_segmentation = matches!(batch.target, super::super::Target::Segmentation(_));
            let is_pose = matches!(batch.target, super::super::Target::Pose(_));
            let single_class = config
                .class_filter
                .as_ref()
                .map(|filter| filter.single_class)
                .unwrap_or(false);
            let report = self.eval_batch_with_config(
                &batch.input,
                &batch.target,
                config.max_detections,
                config.confidence_threshold,
                config.iou_threshold,
                config.loss_config,
                single_class,
                Some(&mut map_acc),
                Some(&mut mask_map_acc),
                Some(&mut pose_map_acc),
                Some(&mut semantic_acc),
            )?;
            total_loss += report.loss as f64;
            last_loss = report.loss;
            total_components.add(report.components);
            last_components = report.components;
            total_samples += report.samples;
            if let Some(metrics) = report.classification {
                classification
                    .get_or_insert(ClassificationEvalMetrics {
                        correct: 0,
                        top5_correct: 0,
                        total: 0,
                    })
                    .add(metrics);
            }
            if let Some(metrics) = report.detection {
                has_detection_output = true;
                has_mask_output |= is_segmentation;
                has_pose_output |= is_pose;
                detection
                    .get_or_insert(DetectionEvalMetrics {
                        matched_targets: 0,
                        total_targets: 0,
                        predictions: 0,
                    })
                    .add(metrics);
            }
        }

        let map = has_detection_output.then(|| map_acc.finalize());
        let mask_map = has_mask_output.then(|| mask_map_acc.finalize());
        let pose_map = has_pose_output.then(|| pose_map_acc.finalize());
        let semantic = semantic_acc.finalize();
        Ok(EvalLoopReport {
            total_steps: steps,
            total_samples,
            mean_loss: (total_loss / steps.max(1) as f64) as f32,
            last_loss,
            mean_components: total_components.mean(),
            last_components,
            classification,
            detection,
            map,
            mask_map,
            pose_map,
            semantic,
        })
    }
}
