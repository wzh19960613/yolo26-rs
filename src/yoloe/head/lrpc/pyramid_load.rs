use candle_nn::VarBuilder;

use crate::yoloe::head::lrpc::official::Official;
use crate::yoloe::head::lrpc::pyramid::Pyramid;

impl Pyramid {
    /// Loads three official LRPC heads with an explicit class count.
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

    /// Loads three official LRPC heads with explicit proposal and localization
    /// feature dims, matching the prompt-free `yoloe-26{scale}-seg-pf.pt` layout
    /// (`proposal_channels=512`, `loc_feature_dim=16`). The plain [`load`] keeps
    /// the historical `proposal_channels=1`, `loc_feature_dim=box_channels`.
    pub fn load_with_dims(
        vb: VarBuilder,
        feature_dim: usize,
        classes: usize,
        proposal_channels: usize,
        loc_feature_dim: usize,
        box_channels: usize,
        enabled: bool,
    ) -> crate::Result<Self> {
        let mut heads = Vec::with_capacity(3);
        for i in 0..3 {
            heads.push(Official::load_with_dims(
                vb.pp(i.to_string()),
                feature_dim,
                classes,
                proposal_channels,
                loc_feature_dim,
                box_channels,
                enabled,
            )?);
        }
        Ok(Self {
            heads,
            strides: [8.0, 16.0, 32.0],
            classes,
            feature_dim,
            proposal_channels,
            loc_feature_dim,
            box_channels,
        })
    }

    /// Loads three official LRPC heads and infers the class count from `lrpc.0.vocab.weight`.
    pub fn load_inferred(
        vb: VarBuilder,
        feature_dim: usize,
        box_channels: usize,
        enabled: bool,
    ) -> crate::Result<Self> {
        let mut heads = Vec::with_capacity(3);
        let mut classes = None;
        for i in 0..3 {
            let head =
                Official::load_inferred(vb.pp(i.to_string()), feature_dim, box_channels, enabled)?;
            if let Some(expected) = classes {
                if head.classes() != expected {
                    return Err(crate::Error::InvalidTensor(format!(
                        "YOLOE LRPC class count mismatch: head {i} has {}, expected {expected}",
                        head.classes()
                    )));
                }
            } else {
                classes = Some(head.classes());
            }
            heads.push(head);
        }
        Ok(Self {
            heads,
            strides: [8.0, 16.0, 32.0],
            classes: classes.unwrap_or(0),
            feature_dim,
            proposal_channels: 1,
            loc_feature_dim: box_channels,
            box_channels,
        })
    }

    /// Loads three official LRPC heads and infers all dimensions from their weights.
    pub fn load_inferred_from_weights(vb: VarBuilder, enabled: bool) -> crate::Result<Self> {
        let mut heads = Vec::with_capacity(3);
        let mut classes = None;
        let mut feature_dim = None;
        let mut proposal_channels = None;
        let mut loc_feature_dim = None;
        let mut box_channels = None;
        for i in 0..3 {
            let head = Official::load_inferred_from_weights(vb.pp(i.to_string()), enabled)?;
            check_dim(&mut classes, head.classes(), "class count", i)?;
            check_dim(&mut feature_dim, head.feature_dim(), "feature dim", i)?;
            check_dim(
                &mut proposal_channels,
                head.proposal_channels(),
                "proposal channels",
                i,
            )?;
            check_dim(
                &mut loc_feature_dim,
                head.loc_feature_dim(),
                "localization feature dim",
                i,
            )?;
            check_dim(&mut box_channels, head.box_channels(), "box channels", i)?;
            heads.push(head);
        }
        Ok(Self {
            heads,
            strides: [8.0, 16.0, 32.0],
            classes: classes.unwrap_or(0),
            feature_dim: feature_dim.unwrap_or(0),
            proposal_channels: proposal_channels.unwrap_or(0),
            loc_feature_dim: loc_feature_dim.unwrap_or(0),
            box_channels: box_channels.unwrap_or(0),
        })
    }

    /// Number of prompt-free classes projected by the pyramid.
    pub const fn classes(&self) -> usize {
        self.classes
    }

    /// Feature dimension accepted by each scale.
    pub const fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Number of channels emitted by each proposal filter.
    pub const fn proposal_channels(&self) -> usize {
        self.proposal_channels
    }

    /// Number of localization feature channels accepted by each scale.
    pub const fn loc_feature_dim(&self) -> usize {
        self.loc_feature_dim
    }

    /// Number of localization channels emitted by each scale.
    pub const fn box_channels(&self) -> usize {
        self.box_channels
    }

    /// Per-scale stride values (`[8.0, 16.0, 32.0]`).
    pub fn strides(&self) -> [f32; 3] {
        self.strides
    }

    /// Runs the dense training forward for scale `i`: box distances and vocab
    /// logits over every anchor. Thin wrapper over the per-scale head.
    pub fn heads_forward_dense_train(
        &self,
        i: usize,
        cls_feat: &candle_core::Tensor,
        loc_feat: &candle_core::Tensor,
    ) -> crate::Result<(candle_core::Tensor, candle_core::Tensor)> {
        self.heads[i].forward_dense_train(cls_feat, loc_feat)
    }
}

fn check_dim(
    expected: &mut Option<usize>,
    got: usize,
    role: &str,
    head_index: usize,
) -> crate::Result<()> {
    if let Some(expected) = expected {
        if got != *expected {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC {role} mismatch: head {head_index} has {got}, expected {expected}"
            )));
        }
    } else {
        *expected = Some(got);
    }
    Ok(())
}
