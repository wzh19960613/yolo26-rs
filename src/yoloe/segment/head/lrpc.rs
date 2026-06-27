use candle_core::Tensor;
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::anchors::dist2bbox_xyxy;
use crate::network::head::{
    dense_branch::DenseBranch, proto::Proto26, topk::postprocess_topk_with_mode,
};

use crate::yoloe::head::lrpc::feature_branch::OfficialFeaturePyramid;
use crate::yoloe::head::lrpc::output::OfficialSegmentParts;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::segment::model::config::Config;
use crate::yoloe::select_lrpc_indices::pad_last_dim;

pub(crate) struct OfficialSegment {
    feature_branches: OfficialFeaturePyramid,
    mask_branches: Vec<DenseBranch>,
    proto: Proto26,
    lrpc: Pyramid,
    nm: usize,
    max_det: usize,
}

impl OfficialSegment {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> crate::Result<Self> {
        let cls_hidden = nonzero(config.cls_hidden, "cls_hidden")?;
        let loc_hidden = nonzero(config.box_hidden, "box_hidden")?;
        let mask_hidden = nonzero(config.mask_hidden, "mask_hidden")?;
        let input_channels = config.scale.head_input_channels();
        let feature_branches =
            OfficialFeaturePyramid::load(vb.clone(), &input_channels, cls_hidden, loc_hidden)?;
        let lrpc = Pyramid::load_inferred_from_weights(vb.pp("lrpc"), true)?;
        if lrpc.feature_dim() != cls_hidden || lrpc.loc_feature_dim() != loc_hidden {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE prompt-free LRPC dims do not match head stems: cls {} vs {}, loc {} vs {}",
                lrpc.feature_dim(),
                cls_hidden,
                lrpc.loc_feature_dim(),
                loc_hidden
            )));
        }

        let cfg = Conv2dConfig::default();
        let mut mask_branches = Vec::with_capacity(input_channels.len());
        for (i, &channels) in input_channels.iter().enumerate() {
            mask_branches.push(DenseBranch::load(
                vb.pp(config.mask_branch.as_str()).pp(i.to_string()),
                channels,
                mask_hidden,
                config.mask_channels,
                cfg,
            )?);
        }
        let proto = Proto26::load(
            vb.pp("proto"),
            &input_channels,
            config.proto_channels,
            config.mask_channels,
        )?;
        Ok(Self {
            feature_branches,
            mask_branches,
            proto,
            lrpc,
            nm: config.mask_channels,
            max_det: config.max_predictions,
        })
    }

    pub(crate) fn classes(&self) -> usize {
        self.lrpc.classes()
    }

    pub(crate) fn forward_parts(
        &self,
        features: &[&Tensor],
        confidence_threshold: f32,
    ) -> crate::Result<OfficialSegmentParts> {
        let (cls_maps, loc_maps) = self.feature_branches.forward(features)?;
        let cls_refs = cls_maps.iter().collect::<Vec<_>>();
        let loc_refs = loc_maps.iter().collect::<Vec<_>>();
        let detect = self
            .lrpc
            .forward(&cls_refs, &loc_refs, confidence_threshold)?;
        let all_masks = DenseBranch::forward_branches(&self.mask_branches, features)?;
        let batch = all_masks.dim(0)?;
        let mut selected_masks = Vec::with_capacity(batch);
        for b in 0..batch {
            let index_tensor = Tensor::new(
                detect.selected_indices[b]
                    .iter()
                    .map(|idx| *idx as u32)
                    .collect::<Vec<_>>(),
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

    pub(crate) fn forward(
        &self,
        features: &[&Tensor],
        confidence_threshold: f32,
        agnostic_nms: bool,
    ) -> crate::Result<(Tensor, Tensor)> {
        let parts = self.forward_parts(features, confidence_threshold)?;
        let dbox = dist2bbox_xyxy(&parts.detect.boxes, &parts.detect.anchors)?
            .broadcast_mul(&parts.detect.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.detect.scores)?;
        let masks = parts.masks.transpose(1, 2)?;
        let preds = Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?;
        let preds = Tensor::cat(&[&preds, &masks], 2)?;
        Ok((
            postprocess_topk_with_mode(
                &preds,
                self.classes(),
                self.nm,
                self.max_det,
                agnostic_nms,
            )?,
            parts.proto,
        ))
    }
}

fn nonzero(value: usize, name: &str) -> crate::Result<usize> {
    if value == 0 {
        return Err(crate::Error::InvalidConfig(format!(
            "YOLOE prompt-free segment {name} must be inferred from checkpoint"
        )));
    }
    Ok(value)
}
