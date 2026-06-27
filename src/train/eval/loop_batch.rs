use candle_core::Tensor;

use super::super::{
    DetectionLossConfig, EvalReport, MapAccumulator, SemanticMapAccumulator, Session, Target,
    classification_eval_metrics, detection_eval_metrics, obb_eval_metrics,
    supervised_loss_report_with_config, update_pose_map, update_segmentation_mask_map,
    update_semantic_acc,
};

impl Session {
    /// Evaluates one supervised batch without updating model variables.
    pub fn eval_batch(&self, input: &Tensor, target: &Target) -> crate::Result<EvalReport> {
        self.eval_batch_with_max_detections(input, target, 300)
    }

    pub(crate) fn eval_batch_with_max_detections(
        &self,
        input: &Tensor,
        target: &Target,
        max_detections: usize,
    ) -> crate::Result<EvalReport> {
        self.eval_batch_with_config(
            input,
            target,
            max_detections,
            0.001,
            0.7,
            DetectionLossConfig::default(),
            false,
            None,
            None,
            None,
            None,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "central eval entry point threads validation knobs and task-specific accumulators"
    )]
    pub(crate) fn eval_batch_with_config(
        &self,
        input: &Tensor,
        target: &Target,
        max_detections: usize,
        confidence_threshold: f32,
        iou_threshold: f32,
        loss_config: DetectionLossConfig,
        single_class: bool,
        mut map: Option<&mut MapAccumulator>,
        mask_map: Option<&mut MapAccumulator>,
        pose_map: Option<&mut MapAccumulator>,
        semantic_acc: Option<&mut SemanticMapAccumulator>,
    ) -> crate::Result<EvalReport> {
        let output = self.model.forward_raw_eval(input)?;
        let postprocess = self.model.forward_eval_postprocess(input)?;
        if let Some(acc) = semantic_acc {
            update_semantic_acc(&output, target, acc)?;
        }
        update_segmentation_mask_map(
            &output,
            postprocess.as_ref(),
            target,
            (input.dim(2)?, input.dim(3)?),
            confidence_threshold,
            iou_threshold,
            max_detections,
            single_class,
            mask_map,
        )?;
        update_pose_map(
            &output,
            target,
            confidence_threshold,
            max_detections,
            single_class,
            pose_map,
        )?;
        let loss_report = supervised_loss_report_with_config(&output, target, loss_config)?;
        let components = loss_report.scalar_components()?;
        let obb_detection = obb_eval_metrics(
            &output,
            target,
            confidence_threshold,
            iou_threshold,
            max_detections,
            single_class,
            map.as_deref_mut(),
        )?;
        let detection = if obb_detection.is_some() {
            obb_detection
        } else {
            detection_eval_metrics(
                &output,
                postprocess.as_ref(),
                target,
                confidence_threshold,
                iou_threshold,
                max_detections,
                single_class,
                map,
            )?
        };
        Ok(EvalReport {
            loss: components.total,
            components,
            samples: input.dim(0)?,
            classification: classification_eval_metrics(&output, target)?,
            detection,
        })
    }
}
