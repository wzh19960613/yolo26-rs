use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::network::head::anchors::dist2bbox_xyxy;
use crate::network::head::topk::postprocess_topk_with_mode;

use crate::yoloe::detect::model::config::Config;
use crate::yoloe::head::lrpc::feature_branch::OfficialFeaturePyramid;
use crate::yoloe::head::lrpc::output::OfficialPyramidOutput;
use crate::yoloe::head::lrpc::pyramid::Pyramid;

pub(crate) struct OfficialDetect {
    feature_branches: OfficialFeaturePyramid,
    lrpc: Pyramid,
    max_det: usize,
}

impl OfficialDetect {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> crate::Result<Self> {
        let cls_hidden = nonzero(config.cls_hidden, "cls_hidden")?;
        let loc_hidden = nonzero(config.box_hidden, "box_hidden")?;
        let input_channels = config.scale.head_input_channels();
        let feature_branches =
            OfficialFeaturePyramid::load(vb.clone(), &input_channels, cls_hidden, loc_hidden)?;
        let lrpc = Pyramid::load_inferred_from_weights(vb.pp("lrpc"), true)?;
        if lrpc.feature_dim() != cls_hidden || lrpc.loc_feature_dim() != loc_hidden {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE prompt-free LRPC dims do not match detect stems: cls {} vs {}, loc {} vs {}",
                lrpc.feature_dim(),
                cls_hidden,
                lrpc.loc_feature_dim(),
                loc_hidden
            )));
        }
        Ok(Self {
            feature_branches,
            lrpc,
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
    ) -> crate::Result<OfficialPyramidOutput> {
        let (cls_maps, loc_maps) = self.feature_branches.forward(features)?;
        let cls_refs = cls_maps.iter().collect::<Vec<_>>();
        let loc_refs = loc_maps.iter().collect::<Vec<_>>();
        self.lrpc
            .forward(&cls_refs, &loc_refs, confidence_threshold)
    }

    pub(crate) fn forward(
        &self,
        features: &[&Tensor],
        confidence_threshold: f32,
        agnostic_nms: bool,
    ) -> crate::Result<Tensor> {
        let parts = self.forward_parts(features, confidence_threshold)?;
        let dbox =
            dist2bbox_xyxy(&parts.boxes, &parts.anchors)?.broadcast_mul(&parts.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.scores)?;
        let preds = Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?;
        Ok(postprocess_topk_with_mode(
            &preds,
            self.classes(),
            0,
            self.max_det,
            agnostic_nms,
        )?)
    }
}

fn nonzero(value: usize, name: &str) -> crate::Result<usize> {
    if value == 0 {
        return Err(crate::Error::InvalidConfig(format!(
            "YOLOE prompt-free detect {name} must be inferred from checkpoint"
        )));
    }
    Ok(value)
}
