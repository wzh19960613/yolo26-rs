use candle_core::Tensor;

use crate::network::head::anchors::dist2bbox_xyxy;
use crate::network::head::topk::postprocess_topk_with_mode;

use crate::yoloe::detect::head::Head;
use crate::yoloe::head::lrpc::output::OfficialPyramidOutput;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::usage::EmbeddingTable;

impl Head {
    /// Runs official prompt-free LRPC over this head's box and embedding branches.
    pub fn forward_official_lrpc_parts(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
    ) -> crate::Result<OfficialPyramidOutput> {
        if features.len() != self.box_branches.len() {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE open-vocabulary LRPC head expected {} feature maps, got {}",
                self.box_branches.len(),
                features.len()
            )));
        }
        if lrpc.feature_dim() != self.embed_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC feature dim {} does not match head embed_dim {}",
                lrpc.feature_dim(),
                self.embed_dim
            )));
        }
        let mut cls_maps = Vec::with_capacity(features.len());
        let mut loc_maps = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            cls_maps.push(self.embedding_branches[i].forward_map(feature)?);
            loc_maps.push(self.box_branches[i].forward_map(feature)?);
        }
        let cls_refs = cls_maps.iter().collect::<Vec<_>>();
        let loc_refs = loc_maps.iter().collect::<Vec<_>>();
        lrpc.forward(&cls_refs, &loc_refs, confidence_threshold)
    }

    /// Runs official LRPC through decode and sigmoid before top-k filtering.
    pub fn forward_official_lrpc_pre_topk(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
    ) -> crate::Result<Tensor> {
        let parts = self.forward_official_lrpc_parts(features, lrpc, confidence_threshold)?;
        let dbox =
            dist2bbox_xyxy(&parts.boxes, &parts.anchors)?.broadcast_mul(&parts.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.scores)?;
        Ok(Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?)
    }

    /// Runs official prompt-free LRPC and returns top-k predictions.
    pub fn forward_official_lrpc(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
        agnostic_nms: bool,
    ) -> crate::Result<Tensor> {
        let preds = self.forward_official_lrpc_pre_topk(features, lrpc, confidence_threshold)?;
        Ok(postprocess_topk_with_mode(
            &preds,
            lrpc.classes(),
            0,
            self.max_det,
            agnostic_nms,
        )?)
    }

    /// Runs the head through decode and sigmoid, before top-k filtering.
    pub fn forward_pre_topk(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<Tensor> {
        let parts = self.forward_parts(features, prompts)?;
        let dbox =
            dist2bbox_xyxy(&parts.boxes, &parts.anchors)?.broadcast_mul(&parts.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.scores)?;
        Ok(Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?)
    }

    /// Runs official-style top-k postprocessing and returns `[batch, det, 6]`.
    pub fn forward(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
        agnostic_nms: bool,
    ) -> crate::Result<Tensor> {
        let decoded = self.forward_pre_topk(features, prompts)?;
        Ok(postprocess_topk_with_mode(
            &decoded,
            prompts.class_count(),
            0,
            self.max_det,
            agnostic_nms,
        )?)
    }
}
