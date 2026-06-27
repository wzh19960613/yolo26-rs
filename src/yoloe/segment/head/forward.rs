use candle_core::Tensor;

use crate::network::head::anchors::dist2bbox_xyxy;
use crate::network::head::{dense_branch::DenseBranch, topk::postprocess_topk_with_mode};

use crate::yoloe::head::lrpc::output::OfficialSegmentParts;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::segment::head::{Head, HeadParts};
use crate::yoloe::select_lrpc_indices::pad_last_dim;
use crate::yoloe::usage::EmbeddingTable;

impl Head {
    /// Runs the head and returns raw detection, mask coefficients, and proto tensors.
    pub fn forward_parts(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<HeadParts> {
        let detect = self.detect.forward_parts(features, prompts)?;
        let masks = DenseBranch::forward_branches(&self.mask_branches, features)?;
        let proto = self.proto.forward(features)?;
        Ok(HeadParts {
            detect,
            masks,
            proto,
        })
    }

    /// Training-time forward returning BN-train-mode dense tensors.
    #[cfg(feature = "train")]
    pub fn forward_dense(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<crate::train::yoloe::output::Output> {
        let detect = self.detect.forward_dense_parts(features, prompts)?;
        let masks = DenseBranch::forward_branches(&self.mask_branches, features)?;
        let proto = self.proto.forward(features)?;
        Ok(crate::train::yoloe::output::Output {
            boxes: detect.boxes,
            scores: detect.scores,
            embeddings: detect.embeddings,
            masks,
            proto,
            anchors: detect.anchors,
            stride_tensor: detect.stride_tensor,
            cls_feature_maps: detect.cls_feature_maps,
            loc_feature_maps: detect.loc_feature_maps,
        })
    }

    /// Runs official prompt-free LRPC and selects mask coefficients with the same proposal index.
    pub fn forward_official_lrpc_parts(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
    ) -> crate::Result<OfficialSegmentParts> {
        let detect =
            self.detect
                .forward_official_lrpc_parts(features, lrpc, confidence_threshold)?;
        let all_masks = DenseBranch::forward_branches(&self.mask_branches, features)?;
        let batch = all_masks.dim(0)?;
        let mut selected_masks = Vec::with_capacity(batch);
        for b in 0..batch {
            let indices = &detect.selected_indices[b];
            let index_tensor = Tensor::new(
                indices.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
                all_masks.device(),
            )?;
            let masks = all_masks
                .narrow(0, b, 1)?
                .squeeze(0)?
                .contiguous()?
                .index_select(&index_tensor, 1)?;
            selected_masks.push(
                pad_last_dim(&masks, detect.scores.dim(2)?)?
                    .unsqueeze(0)?
                    .contiguous()?,
            );
        }
        let proto = self.proto.forward(features)?;
        Ok(OfficialSegmentParts {
            detect,
            masks: Tensor::cat(&selected_masks.iter().collect::<Vec<_>>(), 0)?,
            proto,
        })
    }

    /// Runs official LRPC decode and concatenates selected mask coefficients before top-k.
    pub fn forward_official_lrpc_pre_topk(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
    ) -> crate::Result<(Tensor, Tensor)> {
        let parts = self.forward_official_lrpc_parts(features, lrpc, confidence_threshold)?;
        let dbox = dist2bbox_xyxy(&parts.detect.boxes, &parts.detect.anchors)?
            .broadcast_mul(&parts.detect.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.detect.scores)?;
        let masks = parts.masks.transpose(1, 2)?;
        let preds = Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?;
        Ok((Tensor::cat(&[&preds, &masks], 2)?, parts.proto))
    }

    /// Runs official prompt-free LRPC and returns top-k segmentation predictions.
    pub fn forward_official_lrpc(
        &self,
        features: &[&Tensor],
        lrpc: &Pyramid,
        confidence_threshold: f32,
        agnostic_nms: bool,
    ) -> crate::Result<(Tensor, Tensor)> {
        let (preds, proto) =
            self.forward_official_lrpc_pre_topk(features, lrpc, confidence_threshold)?;
        let preds = postprocess_topk_with_mode(
            &preds,
            lrpc.classes(),
            self.nm,
            self.detect.max_det,
            agnostic_nms,
        )?;
        Ok((preds, proto))
    }

    /// Runs decode and sigmoid, concatenating mask coefficients before top-k filtering.
    pub fn forward_pre_topk(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<(Tensor, Tensor)> {
        let parts = self.forward_parts(features, prompts)?;
        let dbox = dist2bbox_xyxy(&parts.detect.boxes, &parts.detect.anchors)?
            .broadcast_mul(&parts.detect.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.detect.scores)?;
        let masks = parts.masks.transpose(1, 2)?;
        let preds = Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?;
        Ok((Tensor::cat(&[&preds, &masks], 2)?, parts.proto))
    }

    /// Runs top-k postprocessing and returns predictions plus mask prototypes.
    pub fn forward(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
        agnostic_nms: bool,
    ) -> crate::Result<(Tensor, Tensor)> {
        let (preds, proto) = self.forward_pre_topk(features, prompts)?;
        let preds = postprocess_topk_with_mode(
            &preds,
            prompts.class_count(),
            self.nm,
            self.detect.max_det,
            agnostic_nms,
        )?;
        Ok((preds, proto))
    }
}
