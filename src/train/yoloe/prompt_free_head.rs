//! Trainable prompt-free (`-seg-pf`) YOLOE head, mirroring the official
//! `yoloe-26{scale}-seg-pf.pt` layout.
//!
//! Unlike the prompted `-seg` head ([`super::Head`]), the
//! prompt-free head uses 2-layer box stems (`one2one_cv2`, no final 1x1) and
//! embedding stems (`one2one_cv3`, no final `.2`), feeding 80-dim class
//! features and 16-dim loc features straight into a 4585-class LRPC
//! vocabulary head. It reuses the inference [`OfficialFeaturePyramid`] for
//! the stems so the saved checkpoint matches the official `-seg-pf` layout for
//! `.pt` template writing.

use candle_core::Tensor;
use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::anchors::make_anchors;
use crate::network::head::{dense_branch::DenseBranch, proto::Proto26};

use super::model_config::ModelConfig;
use super::output::Output;
use crate::yoloe::head::lrpc::feature_branch::OfficialFeaturePyramid;
use crate::yoloe::head::lrpc::pyramid::Pyramid;

/// Trainable prompt-free head. Fields are read indirectly via forward; building
/// them registers the official `-seg-pf` weights into the shared `VarMap`.
#[allow(dead_code)]
pub(crate) struct TrainablePromptFreeHead {
    feature_branches: OfficialFeaturePyramid,
    one2one_mask_branches: Vec<DenseBranch>,
    one_to_many_mask_branches: Vec<DenseBranch>,
    proto: Proto26,
    lrpc: Pyramid,
}

impl TrainablePromptFreeHead {
    /// Builds the head from a config with official `-seg-pf` dimensions.
    pub(crate) fn load(vb: VarBuilder, config: &ModelConfig) -> crate::Result<Self> {
        let input_channels = config.scale.head_input_channels();
        let feature_branches = OfficialFeaturePyramid::load(
            vb.clone(),
            &input_channels,
            config.cls_hidden,
            config.box_hidden,
        )?;
        let lrpc = Pyramid::load_with_dims(
            vb.pp("lrpc"),
            config.cls_hidden,
            config.prompt_free_vocab,
            config.prompt_free_proposal_channels,
            config.box_hidden,
            4,
            true,
        )?;
        let cfg = Conv2dConfig::default();
        let one2one_mask_branches =
            load_mask_branches(vb.clone(), "one2one_cv5", &input_channels, config, cfg)?;
        let one_to_many_mask_branches =
            load_mask_branches(vb.clone(), "cv5", &input_channels, config, cfg)?;
        let proto = Proto26::load(
            vb.pp("proto"),
            &input_channels,
            config.proto_channels,
            config.mask_channels,
        )?;
        Ok(Self {
            feature_branches,
            one2one_mask_branches,
            one_to_many_mask_branches,
            proto,
            lrpc,
        })
    }

    /// Returns the prompt-free LRPC pyramid (for class count / loss plumbing).
    pub(crate) fn lrpc(&self) -> &Pyramid {
        &self.lrpc
    }

    /// Dense training forward producing boxes, vocabulary scores, masks, proto.
    pub(crate) fn forward_train(&self, features: &[&Tensor]) -> crate::Result<Output> {
        let (cls_maps, loc_maps) = self.feature_branches.forward(features)?;
        let batch = features[0].dim(0)?;
        let dtype = features[0].dtype();
        let device = features[0].device();
        let mut feat_sizes = Vec::with_capacity(features.len());
        let mut box_parts = Vec::with_capacity(features.len());
        let mut score_parts = Vec::with_capacity(features.len());
        let mut embedding_parts = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            let spatial = h * w;
            feat_sizes.push((h, w));
            let (boxes, scores) =
                self.lrpc
                    .heads_forward_dense_train(i, &cls_maps[i], &loc_maps[i])?;
            box_parts.push(boxes.reshape((batch, 4, spatial))?);
            score_parts.push(scores.reshape((batch, self.lrpc.classes(), spatial))?);
            embedding_parts.push(cls_maps[i].reshape((batch, self.lrpc.feature_dim(), spatial))?);
        }
        let masks = DenseBranch::forward_branches(&self.one2one_mask_branches, features)?;
        let proto = self.proto.forward(features)?;
        let (anchors, stride_tensor) =
            make_anchors(&feat_sizes, &self.lrpc.strides(), dtype, device)?;
        Ok(Output {
            boxes: Tensor::cat(&box_parts.iter().collect::<Vec<_>>(), 2)?,
            scores: Tensor::cat(&score_parts.iter().collect::<Vec<_>>(), 2)?,
            embeddings: Tensor::cat(&embedding_parts.iter().collect::<Vec<_>>(), 2)?,
            masks,
            proto,
            anchors,
            stride_tensor,
            cls_feature_maps: cls_maps,
            loc_feature_maps: loc_maps,
        })
    }
}

/// Loads the 3-scale DenseBranch mask-coefficient set under `prefix` (`cv5` or
/// `one2one_cv5`). Shared hidden width matches the official `-seg-pf` layout.
fn load_mask_branches(
    vb: VarBuilder,
    prefix: &str,
    input_channels: &[usize],
    config: &ModelConfig,
    cfg: Conv2dConfig,
) -> crate::Result<Vec<DenseBranch>> {
    let c5 = (input_channels[0] / 4).max(config.mask_channels);
    let mut branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        branches.push(DenseBranch::load(
            vb.pp(prefix).pp(i.to_string()),
            channels,
            c5,
            config.mask_channels,
            cfg,
        )?);
    }
    Ok(branches)
}
