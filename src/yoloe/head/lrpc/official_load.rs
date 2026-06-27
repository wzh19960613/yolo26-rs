use candle_core::Tensor;
use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::yoloe::head::lrpc::official::Official;
use crate::yoloe::infer_lrpc_vocab_classes::{infer_lrpc_vocab_classes, load_lrpc_vocab_linear};

impl Official {
    /// Loads an official-style LRPC head with `vocab`, `pf`, and `loc` weights.
    pub fn load(
        vb: VarBuilder,
        feature_dim: usize,
        classes: usize,
        box_channels: usize,
        enabled: bool,
    ) -> crate::Result<Self> {
        Self::load_with_dims(
            vb,
            feature_dim,
            classes,
            1,
            box_channels,
            box_channels,
            enabled,
        )
    }

    /// Loads an official-style LRPC head with explicit feature dimensions.
    pub fn load_with_dims(
        vb: VarBuilder,
        feature_dim: usize,
        classes: usize,
        proposal_channels: usize,
        loc_feature_dim: usize,
        box_channels: usize,
        enabled: bool,
    ) -> crate::Result<Self> {
        validate_dims(
            feature_dim,
            classes,
            proposal_channels,
            loc_feature_dim,
            box_channels,
        )?;
        let vocab = load_lrpc_vocab_linear(vb.pp("vocab"), feature_dim, classes)?;
        let cfg = Conv2dConfig::default();
        Ok(Self {
            vocab,
            pf: conv2d(feature_dim, proposal_channels, 1, cfg, vb.pp("pf"))?,
            loc: conv2d(loc_feature_dim, box_channels, 1, cfg, vb.pp("loc"))?,
            enabled,
            classes,
            feature_dim,
            proposal_channels,
            loc_feature_dim,
            box_channels,
        })
    }

    /// Loads an official-style LRPC head and infers the vocabulary class count from weights.
    pub fn load_inferred(
        vb: VarBuilder,
        feature_dim: usize,
        box_channels: usize,
        enabled: bool,
    ) -> crate::Result<Self> {
        let raw_weight = vb.pp("vocab").get_unchecked("weight")?;
        let classes = infer_lrpc_vocab_classes(&raw_weight, feature_dim)?;
        Self::load(vb, feature_dim, classes, box_channels, enabled)
    }

    /// Loads an official-style LRPC head and infers all dimensions from weights.
    pub fn load_inferred_from_weights(vb: VarBuilder, enabled: bool) -> crate::Result<Self> {
        let raw_vocab = vb.pp("vocab").get_unchecked("weight")?;
        let (classes, feature_dim) = infer_linear_dims(&raw_vocab, "vocab")?;
        let raw_pf = vb.pp("pf").get_unchecked("weight")?;
        let (proposal_channels, pf_feature_dim) = infer_conv1x1_dims(&raw_pf, "pf")?;
        if pf_feature_dim != feature_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC pf input dim {pf_feature_dim} does not match vocab input dim {feature_dim}"
            )));
        }
        let raw_loc = vb.pp("loc").get_unchecked("weight")?;
        let (box_channels, loc_feature_dim) = infer_conv1x1_dims(&raw_loc, "loc")?;
        Self::load_with_dims(
            vb,
            feature_dim,
            classes,
            proposal_channels,
            loc_feature_dim,
            box_channels,
            enabled,
        )
    }
}

fn validate_dims(
    feature_dim: usize,
    classes: usize,
    proposal_channels: usize,
    loc_feature_dim: usize,
    box_channels: usize,
) -> crate::Result<()> {
    if feature_dim == 0
        || classes == 0
        || proposal_channels == 0
        || loc_feature_dim == 0
        || box_channels == 0
    {
        return Err(crate::Error::InvalidConfig(
            "YOLOE LRPC feature_dim, classes, proposal_channels, loc_feature_dim, and box_channels must be greater than zero"
                .to_string(),
        ));
    }
    Ok(())
}

fn infer_linear_dims(raw_weight: &Tensor, role: &str) -> crate::Result<(usize, usize)> {
    match raw_weight.dims() {
        [out, input, 1, 1] => Ok((*out, *input)),
        [out, input] => Ok((*out, *input)),
        dims => Err(crate::Error::InvalidTensor(format!(
            "YOLOE LRPC {role} weight must have shape [out, input, 1, 1] or [out, input], got {dims:?}"
        ))),
    }
}

fn infer_conv1x1_dims(raw_weight: &Tensor, role: &str) -> crate::Result<(usize, usize)> {
    match raw_weight.dims() {
        [out, input, 1, 1] => Ok((*out, *input)),
        dims => Err(crate::Error::InvalidTensor(format!(
            "YOLOE LRPC {role} weight must have shape [out, input, 1, 1], got {dims:?}"
        ))),
    }
}
