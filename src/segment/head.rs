use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::detect::head::Head as DetectHead;
use crate::network::head::{dense_branch::DenseBranch, proto::Proto26, topk::postprocess_topk};

pub(crate) struct Head {
    detect: DetectHead,
    #[cfg_attr(not(feature = "train"), allow(dead_code))]
    one_to_many_masks: Vec<DenseBranch>,
    one_to_one_masks: Vec<DenseBranch>,
    proto: Proto26,
    nm: usize,
}

#[cfg(feature = "train")]
pub(crate) struct Output {
    pub detect: crate::detect::head::HeadParts,
    pub masks: Tensor,
    pub proto: Tensor,
    pub semantic: Option<Tensor>,
}

#[cfg(feature = "train")]
pub(crate) struct E2eTrainingOutput {
    pub one_to_many_detect: crate::detect::head::HeadParts,
    pub one_to_many_masks: Tensor,
    pub one_to_one_detect: crate::detect::head::HeadParts,
    pub one_to_one_masks: Tensor,
    pub proto: Tensor,
    pub semantic: Option<Tensor>,
}

impl Head {
    pub(crate) fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        nc: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
    ) -> Result<Self> {
        let detect = DetectHead::load(vb.clone(), input_channels, nc, max_det)?;
        let c4 = (input_channels[0] / 4).max(nm);
        let cfg = Conv2dConfig::default();
        let one_to_many_masks = load_mask_branches(vb.clone(), "cv4", input_channels, c4, nm, cfg)?;
        let one_to_one_masks =
            load_mask_branches(vb.clone(), "one2one_cv4", input_channels, c4, nm, cfg)?;
        let proto = Proto26::load_with_semantic(vb.pp("proto"), input_channels, npr, nm, nc)?;
        Ok(Self {
            detect,
            one_to_many_masks,
            one_to_one_masks,
            proto,
            nm,
        })
    }

    pub(crate) fn forward(&self, features: &[&Tensor]) -> Result<(Tensor, Tensor)> {
        let base = self.detect.forward_pre_topk(features)?;
        let masks = DenseBranch::forward_branches(&self.one_to_one_masks, features)?;
        let masks = masks.transpose(1, 2)?;
        let preds = Tensor::cat(&[&base, &masks], 2)?;
        let preds = postprocess_topk(&preds, self.detect.nc(), self.nm, self.detect.max_det())?;
        let proto = self.proto.forward(features)?;
        Ok((preds, proto))
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_training(&self, features: &[&Tensor]) -> Result<Output> {
        let detect = self.detect.forward_training(features)?;
        let masks = DenseBranch::forward_branches(&self.one_to_one_masks, features)?;
        let proto = self.proto.forward(features)?;
        Ok(Output {
            detect,
            masks,
            proto,
            semantic: None,
        })
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_e2e_training(&self, features: &[&Tensor]) -> Result<E2eTrainingOutput> {
        let detect = self.detect.forward_e2e_training(features)?;
        let one_to_many_masks = DenseBranch::forward_branches(&self.one_to_many_masks, features)?;
        let detached_features: Vec<Tensor> =
            features.iter().map(|feature| feature.detach()).collect();
        let detached_refs: Vec<&Tensor> = detached_features.iter().collect();
        let one_to_one_masks =
            DenseBranch::forward_branches(&self.one_to_one_masks, &detached_refs)?;
        let proto = self.proto.forward_training(features)?;
        Ok(E2eTrainingOutput {
            one_to_many_detect: detect.one_to_many,
            one_to_many_masks,
            one_to_one_detect: detect.one_to_one,
            one_to_one_masks,
            proto: proto.proto,
            semantic: proto.semantic,
        })
    }
}

fn load_mask_branches(
    vb: VarBuilder,
    name: &str,
    input_channels: &[usize],
    c4: usize,
    nm: usize,
    cfg: Conv2dConfig,
) -> Result<Vec<DenseBranch>> {
    let mut branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        branches.push(DenseBranch::load(
            vb.pp(name).pp(i.to_string()),
            channels,
            c4,
            nm,
            cfg,
        )?);
    }
    Ok(branches)
}
