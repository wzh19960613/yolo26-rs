use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::detect::head::Head as DetectHead;
use crate::network::head::{pose_branch::PoseBranch, topk::postprocess_topk};

pub(crate) struct Head {
    detect: DetectHead,
    #[cfg_attr(not(feature = "train"), allow(dead_code))]
    one_to_many_pose_branches: Vec<PoseBranch>,
    one_to_one_pose_branches: Vec<PoseBranch>,
    nk: usize,
    keypoint_dims: usize,
}

#[cfg(feature = "train")]
pub(crate) struct Output {
    pub detect: crate::detect::head::HeadParts,
    pub keypoints: Tensor,
}

#[cfg(feature = "train")]
pub(crate) struct E2eTrainingOutput {
    pub one_to_many_detect: crate::detect::head::HeadParts,
    pub one_to_many_keypoints: Tensor,
    pub one_to_one_detect: crate::detect::head::HeadParts,
    pub one_to_one_keypoints: Tensor,
}

impl Head {
    pub(crate) fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        nc: usize,
        max_det: usize,
        keypoints_count: usize,
        keypoint_dims: usize,
    ) -> Result<Self> {
        let detect = DetectHead::load(vb.clone(), input_channels, nc, max_det)?;
        let nk = keypoints_count * keypoint_dims;
        let c4 = (input_channels[0] / 4).max(keypoints_count * (keypoint_dims + 2));
        let cfg = Conv2dConfig::default();
        let one_to_many_pose_branches =
            load_pose_branches(vb.clone(), "cv4", "cv4_kpts", input_channels, c4, nk, cfg)?;
        let one_to_one_pose_branches = load_pose_branches(
            vb,
            "one2one_cv4",
            "one2one_cv4_kpts",
            input_channels,
            c4,
            nk,
            cfg,
        )?;
        Ok(Self {
            detect,
            one_to_many_pose_branches,
            one_to_one_pose_branches,
            nk,
            keypoint_dims,
        })
    }

    pub(crate) fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        let base = self.detect.forward_pre_topk(features)?;
        let kpts = self.forward_keypoints(features)?;
        let preds = Tensor::cat(&[&base, &kpts], 2)?;
        postprocess_topk(&preds, self.detect.nc(), self.nk, self.detect.max_det())
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_training(&self, features: &[&Tensor]) -> Result<Output> {
        let detect = self.detect.forward_training(features)?;
        let keypoints =
            self.forward_raw_keypoints_with(features, &self.one_to_one_pose_branches)?;
        Ok(Output { detect, keypoints })
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_e2e_training(&self, features: &[&Tensor]) -> Result<E2eTrainingOutput> {
        let detect = self.detect.forward_e2e_training(features)?;
        let one_to_many_keypoints =
            self.forward_raw_keypoints_with(features, &self.one_to_many_pose_branches)?;
        let one_to_one_keypoints =
            self.forward_raw_keypoints_with(features, &self.one_to_one_pose_branches)?;
        Ok(E2eTrainingOutput {
            one_to_many_detect: detect.one_to_many,
            one_to_many_keypoints,
            one_to_one_detect: detect.one_to_one,
            one_to_one_keypoints,
        })
    }

    fn forward_keypoints(&self, features: &[&Tensor]) -> Result<Tensor> {
        let parts = self.detect.forward_parts(features)?;
        let kpts = self.forward_raw_keypoints_with(features, &self.one_to_one_pose_branches)?;
        decode_keypoints(
            &kpts,
            &parts.anchors,
            &parts.stride_tensor,
            self.keypoint_dims,
        )?
        .transpose(1, 2)
    }

    fn forward_raw_keypoints_with(
        &self,
        features: &[&Tensor],
        branches: &[PoseBranch],
    ) -> Result<Tensor> {
        let batch = features[0].dim(0)?;
        let mut kpts = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            kpts.push(branches[i].forward(feature, batch, h * w)?);
        }
        let refs: Vec<&Tensor> = kpts.iter().collect();
        Tensor::cat(&refs, 2)
    }
}

fn load_pose_branches(
    vb: VarBuilder,
    name: &str,
    keypoint_name: &str,
    input_channels: &[usize],
    c4: usize,
    nk: usize,
    cfg: Conv2dConfig,
) -> Result<Vec<PoseBranch>> {
    let mut branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        branches.push(PoseBranch::load(
            vb.pp(name).pp(i.to_string()),
            vb.pp(keypoint_name).pp(i.to_string()),
            channels,
            c4,
            nk,
            cfg,
        )?);
    }
    Ok(branches)
}

fn decode_keypoints(
    kpts: &Tensor,
    anchors: &Tensor,
    strides: &Tensor,
    dims: usize,
) -> Result<Tensor> {
    let nk = kpts.dim(1)?;
    let mut decoded = Vec::with_capacity(nk);
    for idx in 0..nk {
        let channel = kpts.narrow(1, idx, 1)?;
        let out = match idx % dims {
            0 => channel
                .broadcast_add(&anchors.narrow(1, 0, 1)?)?
                .broadcast_mul(strides)?,
            1 => channel
                .broadcast_add(&anchors.narrow(1, 1, 1)?)?
                .broadcast_mul(strides)?,
            2 => candle_nn::ops::sigmoid(&channel)?,
            _ => channel,
        };
        decoded.push(out);
    }
    let refs: Vec<&Tensor> = decoded.iter().collect();
    Tensor::cat(&refs, 1)
}
