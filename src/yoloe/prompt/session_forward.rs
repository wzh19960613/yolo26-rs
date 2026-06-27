use candle_core::Tensor;

use crate::yoloe::detect::head::{Head as DetectHead, HeadParts as DetectHeadParts};
use crate::yoloe::head::lrpc::head::LrpcHead;
use crate::yoloe::head::lrpc::output::LrpcOutput;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::prompt::state::State;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::segment::head::{Head, HeadParts};
use crate::yoloe::usage::Usage;
use crate::yoloe::visuals::Visuals;
use crate::yoloe::visuals::merge::merge_visual_prompt_masks;

impl Session {
    /// Encodes the active visual prompt masks with an official SAVPE module.
    ///
    /// `visuals` is the `[1, classes, h, w]` SAVPE-input tensor; it is
    /// re-merged against the session's active prompts so already-merged
    /// `visuals` pass through unchanged.
    pub fn encode_visuals_with_savpe(
        &mut self,
        encoder: &Encoder,
        features: &[&Tensor],
        visuals: &Visuals,
    ) -> crate::Result<()> {
        if self.prompts.active_usage() != Some(Usage::Visual) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt mask encoding requires active visual prompts".to_string(),
            ));
        }
        let classes = self.prompts.active_classes().ok_or_else(|| {
            crate::Error::InvalidConfig("YOLOE visual prompts are empty".to_string())
        })?;
        let merged = self.merged_visual_prompt_masks(&visuals.tensor)?;
        let table = encoder.encode_single_image_table(features, &merged, classes)?;
        self.prompt_table = Some(table);
        Ok(())
    }

    /// Runs prompt-free LRPC scoring with the active prompt-free vocabulary table.
    pub fn forward_lrpc(
        &self,
        cls_features: &Tensor,
        loc_features: &Tensor,
        proposal_logits: &Tensor,
    ) -> crate::Result<LrpcOutput> {
        if self.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE LRPC forward requires active prompt-free vocabulary".to_string(),
            ));
        }
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE LRPC forward requires prompt-free vocabulary embeddings".to_string(),
            )
        })?;
        LrpcHead {
            scorer: self.scorer,
            ..LrpcHead::default()
        }
        .forward(cls_features, loc_features, proposal_logits, table)
    }

    /// Runs an official prompt-free YOLOE LRPC head and returns top-k predictions.
    pub fn forward_official_lrpc_head(
        &self,
        head: &DetectHead,
        lrpc: &Pyramid,
        features: &[&Tensor],
    ) -> crate::Result<Tensor> {
        if self.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "official YOLOE LRPC forward requires active prompt-free vocabulary".to_string(),
            ));
        }
        self.ensure_prompt_free_class_count(lrpc.classes())?;
        head.forward_official_lrpc(
            features,
            lrpc,
            self.predict.lrpc_confidence_threshold,
            self.predict.agnostic_nms,
        )
    }

    /// Runs an official prompt-free YOLOE segmentation head and returns predictions plus proto masks.
    pub fn forward_official_lrpc_segment_head(
        &self,
        head: &Head,
        lrpc: &Pyramid,
        features: &[&Tensor],
    ) -> crate::Result<(Tensor, Tensor)> {
        if self.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "official YOLOE segmentation LRPC forward requires active prompt-free vocabulary"
                    .to_string(),
            ));
        }
        self.ensure_prompt_free_class_count(lrpc.classes())?;
        head.forward_official_lrpc(
            features,
            lrpc,
            self.predict.lrpc_confidence_threshold,
            self.predict.agnostic_nms,
        )
    }

    /// Runs an open-vocabulary YOLOE head and returns raw head parts.
    pub fn forward_open_vocabulary_parts(
        &self,
        head: &DetectHead,
        features: &[&Tensor],
    ) -> crate::Result<DetectHeadParts> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE open-vocabulary head forward requires prompt embeddings".to_string(),
            )
        })?;
        head.forward_parts(features, table)
    }

    /// Runs an open-vocabulary YOLOE head and returns top-k predictions.
    pub fn forward_open_vocabulary_head(
        &self,
        head: &DetectHead,
        features: &[&Tensor],
    ) -> crate::Result<Tensor> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE open-vocabulary head forward requires prompt embeddings".to_string(),
            )
        })?;
        head.forward(features, table, self.predict.agnostic_nms)
    }

    /// Runs an open-vocabulary YOLOE segmentation head and returns raw parts.
    pub fn forward_open_vocabulary_segment_parts(
        &self,
        head: &Head,
        features: &[&Tensor],
    ) -> crate::Result<HeadParts> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE open-vocabulary segment forward requires prompt embeddings".to_string(),
            )
        })?;
        head.forward_parts(features, table)
    }

    /// Runs an open-vocabulary YOLOE segmentation head and returns predictions plus proto masks.
    pub fn forward_open_vocabulary_segment_head(
        &self,
        head: &Head,
        features: &[&Tensor],
    ) -> crate::Result<(Tensor, Tensor)> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE open-vocabulary segment forward requires prompt embeddings".to_string(),
            )
        })?;
        head.forward(features, table, self.predict.agnostic_nms)
    }

    pub(crate) fn merged_visual_prompt_masks(
        &self,
        prompt_masks: &Tensor,
    ) -> crate::Result<Visuals> {
        match &self.prompts.state {
            State::Visual(prompts) => merge_visual_prompt_masks(prompts, prompt_masks),
            _ => Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt mask encoding requires active visual prompts".to_string(),
            )),
        }
    }
}
