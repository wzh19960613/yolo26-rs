use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::detect::head::Head as DetectHead;
use crate::network::head::dense_branch::DenseBranch;
use crate::network::head::{anchors::dist2rbox_xywh, topk::postprocess_topk};

pub(crate) struct Head {
    detect: DetectHead,
    #[cfg_attr(not(feature = "train"), allow(dead_code))]
    one_to_many_angles: Vec<DenseBranch>,
    one_to_one_angles: Vec<DenseBranch>,
    ne: usize,
}

#[cfg(feature = "train")]
pub(crate) struct Output {
    pub detect: crate::detect::head::HeadParts,
    pub angles: Tensor,
}

#[cfg(feature = "train")]
pub(crate) struct E2eTrainingOutput {
    pub one_to_many_detect: crate::detect::head::HeadParts,
    pub one_to_many_angles: Tensor,
    pub one_to_one_detect: crate::detect::head::HeadParts,
    pub one_to_one_angles: Tensor,
}

impl Head {
    pub(crate) fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        nc: usize,
        max_det: usize,
        ne: usize,
    ) -> Result<Self> {
        let detect = DetectHead::load(vb.clone(), input_channels, nc, max_det)?;
        let c4 = (input_channels[0] / 4).max(ne);
        let cfg = Conv2dConfig::default();
        let one_to_many_angles =
            load_angle_branches(vb.clone(), "cv4", input_channels, c4, ne, cfg)?;
        let one_to_one_angles =
            load_angle_branches(vb, "one2one_cv4", input_channels, c4, ne, cfg)?;
        Ok(Self {
            detect,
            one_to_many_angles,
            one_to_one_angles,
            ne,
        })
    }

    pub(crate) fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        let parts = self.detect.forward_parts(features)?;
        let angles = DenseBranch::forward_branches(&self.one_to_one_angles, features)?;
        let boxes = dist2rbox_xywh(&parts.boxes, &angles, &parts.anchors)?
            .broadcast_mul(&parts.stride_tensor)?;
        let scores = candle_nn::ops::sigmoid(&parts.scores)?;
        let preds = Tensor::cat(&[&boxes, &scores, &angles], 1)?.transpose(1, 2)?;
        postprocess_topk(&preds, self.detect.nc(), self.ne, self.detect.max_det())
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_training(&self, features: &[&Tensor]) -> Result<Output> {
        let detect = self.detect.forward_training(features)?;
        let angles = DenseBranch::forward_branches(&self.one_to_one_angles, features)?;
        Ok(Output { detect, angles })
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_e2e_training(&self, features: &[&Tensor]) -> Result<E2eTrainingOutput> {
        let detect = self.detect.forward_e2e_training(features)?;
        let one_to_many_angles = DenseBranch::forward_branches(&self.one_to_many_angles, features)?;
        let one_to_one_angles = DenseBranch::forward_branches(&self.one_to_one_angles, features)?;
        Ok(E2eTrainingOutput {
            one_to_many_detect: detect.one_to_many,
            one_to_many_angles,
            one_to_one_detect: detect.one_to_one,
            one_to_one_angles,
        })
    }
}

fn load_angle_branches(
    vb: VarBuilder,
    name: &str,
    input_channels: &[usize],
    c4: usize,
    ne: usize,
    cfg: Conv2dConfig,
) -> Result<Vec<DenseBranch>> {
    let mut branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        branches.push(DenseBranch::load(
            vb.pp(name).pp(i.to_string()),
            channels,
            c4,
            ne,
            cfg,
        )?);
    }
    Ok(branches)
}
