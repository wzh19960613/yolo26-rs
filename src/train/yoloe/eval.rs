use candle_core::Tensor;

use crate::yoloe::segment::model::train_config::PromptMode;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::BatchVisuals;

use super::{BatchEvalConfig, ensure_yoloe_eval_mode};
use super::{
    DetectionLossConfig, EvalReport, SegmentationTargets, Session, yoloe_eval_with_prompt_table,
    yoloe_prompt_free_eval_report,
};

impl Session {
    /// Evaluates one text-prompt YOLOE segmentation batch without updating weights.
    pub fn eval_text_batch(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
    ) -> crate::Result<EvalReport> {
        self.eval_text_batch_with_loss_config(
            input,
            target,
            prompts,
            DetectionLossConfig::default(),
        )
    }

    /// Evaluates one text-prompt YOLOE batch with custom loss gains.
    pub fn eval_text_batch_with_loss_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<EvalReport> {
        ensure_yoloe_eval_mode(self, PromptMode::TextPrompt, "text-prompt")?;
        let aligned = self.model().align_text_prompts(prompts)?;
        yoloe_eval_with_prompt_table(
            self,
            input,
            target,
            &aligned,
            BatchEvalConfig::from_loss_config(loss_config),
            None,
            None,
        )
    }

    /// Evaluates one visual-prompt YOLOE segmentation batch without updating weights.
    pub fn eval_visual_batch(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        visuals: &BatchVisuals,
        classes: Vec<String>,
    ) -> crate::Result<EvalReport> {
        self.eval_visual_batch_with_loss_config(
            input,
            target,
            visuals,
            classes,
            DetectionLossConfig::default(),
        )
    }

    /// Evaluates one visual-prompt YOLOE batch with custom loss gains.
    pub fn eval_visual_batch_with_loss_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        visuals: &BatchVisuals,
        classes: Vec<String>,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<EvalReport> {
        ensure_yoloe_eval_mode(self, PromptMode::Visual, "visual-prompt")?;
        let prompts = self
            .model()
            .encode_visual_prompts(input, visuals, classes)?;
        yoloe_eval_with_prompt_table(
            self,
            input,
            target,
            &prompts,
            BatchEvalConfig::from_loss_config(loss_config),
            None,
            None,
        )
    }

    /// Evaluates one prompt-free YOLOE segmentation batch without updating weights.
    pub fn eval_prompt_free_batch(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
    ) -> crate::Result<EvalReport> {
        self.eval_prompt_free_batch_with_loss_config(input, target, DetectionLossConfig::default())
    }

    /// Evaluates one prompt-free YOLOE batch with custom loss gains.
    pub fn eval_prompt_free_batch_with_loss_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<EvalReport> {
        ensure_yoloe_eval_mode(self, PromptMode::PromptFree, "prompt-free")?;
        yoloe_prompt_free_eval_report(
            self,
            input,
            target,
            BatchEvalConfig::from_loss_config(loss_config),
            None,
            None,
        )
    }
}
