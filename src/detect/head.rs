use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::anchors::{dist2bbox_xyxy, make_anchors};
use crate::network::head::{box_branch::BoxBranch, cls_branch::ClsBranch, topk::postprocess_topk};

pub(crate) struct HeadParts {
    pub(crate) boxes: Tensor,
    pub(crate) scores: Tensor,
    pub(crate) anchors: Tensor,
    pub(crate) stride_tensor: Tensor,
}

#[cfg(feature = "train")]
pub(crate) struct E2eTrainingParts {
    pub(crate) one_to_many: HeadParts,
    pub(crate) one_to_one: HeadParts,
}

struct BranchSet {
    box_branches: Vec<BoxBranch>,
    cls_branches: Vec<ClsBranch>,
}

pub(crate) struct Head {
    #[cfg_attr(not(feature = "train"), allow(dead_code))]
    one_to_many: BranchSet,
    one_to_one: BranchSet,
    strides: [f32; 3],
    nc: usize,
    max_det: usize,
}

impl Head {
    pub(crate) fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        nc: usize,
        max_det: usize,
    ) -> Result<Self> {
        let reg_max = 1;
        let c2 = 16_usize.max(input_channels[0] / 4).max(reg_max * 4);
        let c3 = input_channels[0].max(nc.min(100));
        let cfg = Conv2dConfig::default();
        let one_to_many = load_branch_set(
            vb.clone(),
            input_channels,
            BranchLoadSpec::new("cv2", "cv3", c2, c3, nc, cfg),
        )?;
        let one_to_one = load_branch_set(
            vb,
            input_channels,
            BranchLoadSpec::new("one2one_cv2", "one2one_cv3", c2, c3, nc, cfg),
        )?;

        Ok(Self {
            one_to_many,
            one_to_one,
            strides: [8.0, 16.0, 32.0],
            nc,
            max_det,
        })
    }

    pub(crate) fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        let decoded = self.forward_pre_topk(features)?;
        postprocess_topk(&decoded, self.nc, 0, self.max_det)
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_training(&self, features: &[&Tensor]) -> Result<HeadParts> {
        self.forward_parts(features)
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_e2e_training(&self, features: &[&Tensor]) -> Result<E2eTrainingParts> {
        let detached_features: Vec<Tensor> =
            features.iter().map(|feature| feature.detach()).collect();
        let detached_refs: Vec<&Tensor> = detached_features.iter().collect();
        Ok(E2eTrainingParts {
            one_to_many: self.forward_parts_with(features, &self.one_to_many)?,
            one_to_one: self.forward_parts_with(&detached_refs, &self.one_to_one)?,
        })
    }

    #[allow(dead_code)]
    pub(crate) const fn nc(&self) -> usize {
        self.nc
    }

    #[allow(dead_code)]
    pub(crate) const fn max_det(&self) -> usize {
        self.max_det
    }

    pub(crate) fn forward_parts(&self, features: &[&Tensor]) -> Result<HeadParts> {
        self.forward_parts_with(features, &self.one_to_one)
    }

    fn forward_parts_with(&self, features: &[&Tensor], branches: &BranchSet) -> Result<HeadParts> {
        let device = features[0].device();
        let batch = features[0].dim(0)?;
        let mut all_boxes = Vec::with_capacity(features.len());
        let mut all_scores = Vec::with_capacity(features.len());
        let mut feat_sizes = Vec::with_capacity(features.len());

        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            let spatial = h * w;
            feat_sizes.push((h, w));
            all_boxes.push(branches.box_branches[i].forward(feature, batch, spatial)?);
            all_scores.push(branches.cls_branches[i].forward(feature, batch, self.nc, spatial)?);
        }

        let box_refs: Vec<&Tensor> = all_boxes.iter().collect();
        let score_refs: Vec<&Tensor> = all_scores.iter().collect();
        let boxes = Tensor::cat(&box_refs, 2)?;
        let scores = Tensor::cat(&score_refs, 2)?;
        let (anchors, stride_tensor) =
            make_anchors(&feat_sizes, &self.strides, features[0].dtype(), device)?;

        Ok(HeadParts {
            boxes,
            scores,
            anchors,
            stride_tensor,
        })
    }

    pub(crate) fn forward_pre_topk(&self, features: &[&Tensor]) -> Result<Tensor> {
        let parts = self.forward_parts(features)?;
        let dbox =
            dist2bbox_xyxy(&parts.boxes, &parts.anchors)?.broadcast_mul(&parts.stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&parts.scores)?;
        Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)
    }
}

#[derive(Clone, Copy)]
struct BranchLoadSpec<'a> {
    box_name: &'a str,
    cls_name: &'a str,
    c2: usize,
    c3: usize,
    nc: usize,
    cfg: Conv2dConfig,
}

impl<'a> BranchLoadSpec<'a> {
    const fn new(
        box_name: &'a str,
        cls_name: &'a str,
        c2: usize,
        c3: usize,
        nc: usize,
        cfg: Conv2dConfig,
    ) -> Self {
        Self {
            box_name,
            cls_name,
            c2,
            c3,
            nc,
            cfg,
        }
    }
}

fn load_branch_set(
    vb: VarBuilder,
    input_channels: &[usize],
    spec: BranchLoadSpec<'_>,
) -> Result<BranchSet> {
    let reg_max = 1;
    let mut box_branches = Vec::with_capacity(input_channels.len());
    let mut cls_branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        let stride = match i {
            0 => 8.0,
            1 => 16.0,
            _ => 32.0,
        };
        box_branches.push(BoxBranch::load(
            vb.pp(spec.box_name).pp(i.to_string()),
            channels,
            spec.c2,
            reg_max,
            spec.cfg,
        )?);
        cls_branches.push(ClsBranch::load_with_class_bias(
            vb.pp(spec.cls_name).pp(i.to_string()),
            channels,
            spec.c3,
            spec.nc,
            spec.cfg,
            stride,
        )?);
    }
    Ok(BranchSet {
        box_branches,
        cls_branches,
    })
}
