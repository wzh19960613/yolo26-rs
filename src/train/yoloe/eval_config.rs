use candle_core::Tensor;

use crate::yoloe::segment::model::train_config::PromptMode;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::BatchVisuals;

use super::{
    BatchEvalConfig, DetectionMapAccumulator, EvalLoopConfig, EvalReport, MaskMapAccumulator,
    SegmentationTargets, Session, ensure_yoloe_eval_mode, yoloe_eval_with_prompt_table,
    yoloe_prompt_free_eval_report,
};

impl Session {
    /// Evaluates one text-prompt YOLOE batch with validation thresholds and mAP accumulation.
    pub fn eval_text_batch_with_eval_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
        config: &EvalLoopConfig,
        map: Option<&mut DetectionMapAccumulator>,
        mask_map: Option<&mut MaskMapAccumulator>,
    ) -> crate::Result<EvalReport> {
        ensure_yoloe_eval_mode(self, PromptMode::TextPrompt, "text-prompt")?;
        let aligned = self.model().align_text_prompts(prompts)?;
        yoloe_eval_with_prompt_table(
            self,
            input,
            target,
            &aligned,
            BatchEvalConfig::from_eval_loop(config),
            map,
            mask_map,
        )
    }

    /// Evaluates one visual-prompt YOLOE batch with validation thresholds and mAP accumulation.
    #[expect(
        clippy::too_many_arguments,
        reason = "visual prompt eval needs prompt inputs plus validation knobs and accumulators"
    )]
    pub fn eval_visual_batch_with_eval_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        visuals: &BatchVisuals,
        classes: Vec<String>,
        config: &EvalLoopConfig,
        map: Option<&mut DetectionMapAccumulator>,
        mask_map: Option<&mut MaskMapAccumulator>,
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
            BatchEvalConfig::from_eval_loop(config),
            map,
            mask_map,
        )
    }

    /// Evaluates one prompt-free YOLOE batch with validation thresholds and mAP accumulation.
    pub fn eval_prompt_free_batch_with_eval_config(
        &self,
        input: &Tensor,
        target: &SegmentationTargets,
        config: &EvalLoopConfig,
        map: Option<&mut DetectionMapAccumulator>,
        mask_map: Option<&mut MaskMapAccumulator>,
    ) -> crate::Result<EvalReport> {
        ensure_yoloe_eval_mode(self, PromptMode::PromptFree, "prompt-free")?;
        yoloe_prompt_free_eval_report(
            self,
            input,
            target,
            BatchEvalConfig::from_eval_loop(config),
            map,
            mask_map,
        )
    }
}
